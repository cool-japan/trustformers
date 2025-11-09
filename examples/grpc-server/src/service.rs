use std::pin::Pin;
#![allow(unused_variables)]
use std::sync::Arc;
use std::time::Instant;

use prost_types::{value::Kind, Struct, Value};
use tokio::sync::Mutex;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status};
use tracing::{debug, info, instrument};
use uuid::Uuid;

use crate::error::{ServiceError, ServiceResult};
use crate::model_manager::ModelManager;

// Import generated proto types
pub mod inference {
    tonic::include_proto!("trustformers.inference");
}

use inference::{
    inference_service_server::InferenceService, BatchPredictRequest, BatchPredictResponse,
    GetModelInfoRequest, ListModelsResponse, LoadModelRequest, LoadModelResponse, ModelConfig,
    ModelInfo, ModelStatus, PredictMetrics, PredictRequest, PredictResponse, StreamPredictRequest,
    StreamPredictResponse, UnloadModelRequest,
};

pub struct InferenceServiceImpl {
    model_manager: Arc<ModelManager>,
    stream_sessions: Arc<Mutex<dashmap::DashMap<String, StreamSession>>>,
}

struct StreamSession {
    model_id: String,
    created_at: Instant,
    last_activity: Instant,
}

impl InferenceServiceImpl {
    pub fn new(model_manager: Arc<ModelManager>) -> Self {
        Self {
            model_manager,
            stream_sessions: Arc::new(Mutex::new(dashmap::DashMap::new())),
        }
    }

    async fn create_predict_metrics(
        &self,
        start: Instant,
        tokens: usize,
        device: &str,
    ) -> PredictMetrics {
        let latency = start.elapsed();
        PredictMetrics {
            latency_ms: latency.as_secs_f32() * 1000.0,
            tokens_per_second: (tokens as f32 / latency.as_secs_f32()) as i32,
            memory_used_bytes: 0, // Would get from actual measurement
            device_used: device.to_string(),
        }
    }
}

#[tonic::async_trait]
impl InferenceService for InferenceServiceImpl {
    #[instrument(skip(self, request))]
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        debug!("Predict request for model: {}", req.model_id);

        // Get the model
        let model = self.model_manager
            .get_model(&req.model_id)
            .map_err(|e| Status::from(e))?;

        // Increment request count
        self.model_manager
            .increment_request_count(&req.model_id)
            .await
            .map_err(|e| Status::from(e))?;

        // Extract input text
        let input_text = match req.input {
            Some(inference::predict_request::Input::Text(text)) => text,
            Some(inference::predict_request::Input::TextInput(text_input)) => text_input.text,
            _ => return Err(Status::invalid_argument("Only text input is supported")),
        };

        // Perform real inference using the loaded model
        let generation_config = trustformers_models::common_patterns::GenerationConfig {
            max_new_tokens: req.max_tokens.unwrap_or(50) as usize,
            temperature: req.temperature.unwrap_or(0.7),
            top_p: req.top_p.unwrap_or(0.9),
            top_k: req.top_k.map(|k| k as usize),
            do_sample: req.do_sample.unwrap_or(true),
            repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
            pad_token_id: Some(0),
            eos_token_id: Some(2),
            use_cache: true,
            stream: false,
            ..Default::default()
        };

        // Generate text using the model's generate method
        let output_text = match model.model.generate(&input_text, &generation_config) {
            Ok(generated) => {
                info!("Successfully generated text for model {}", req.model_id);
                generated
            },
            Err(e) => {
                warn!("Generation failed for model {}: {}", req.model_id, e);
                return Err(Status::internal(format!("Generation failed: {}", e)));
            }
        };

        // Count tokens using the tokenizer for accurate metrics
        let tokens = match model.tokenizer.encode(&output_text) {
            Ok(tokenized) => tokenized.ids.len(),
            Err(_) => output_text.split_whitespace().count(), // fallback to word count
        };

        let metrics = self.create_predict_metrics(start, tokens, &model.device).await;

        let response = PredictResponse {
            output: Some(inference::predict_response::Output::TextOutput(
                inference::TextOutput {
                    text: output_text,
                    texts: vec![],
                    scores: vec![],
                    token_ids: vec![],
                },
            )),
            metrics: Some(metrics),
        };

        Ok(Response::new(response))
    }

    type StreamPredictStream = Pin<
        Box<dyn Stream<Item = Result<StreamPredictResponse, Status>> + Send + 'static>,
    >;

    #[instrument(skip(self, request))]
    async fn stream_predict(
        &self,
        request: Request<tonic::Streaming<StreamPredictRequest>>,
    ) -> Result<Response<Self::StreamPredictStream>, Status> {
        let mut stream = request.into_inner();
        let model_manager = self.model_manager.clone();
        let sessions = self.stream_sessions.clone();

        let output_stream = async_stream::stream! {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(req) => {
                        match req.request {
                            Some(inference::stream_predict_request::Request::Start(start)) => {
                                // Initialize streaming session
                                let session_id = start.session_id.unwrap_or_else(|| Uuid::new_v4().to_string());

                                // Get model
                                let model = match model_manager.get_model(&start.model_id) {
                                    Ok(m) => m,
                                    Err(e) => {
                                        yield Err(Status::from(e));
                                        return;
                                    }
                                };

                                // Create session
                                let session = StreamSession {
                                    model_id: start.model_id.clone(),
                                    created_at: Instant::now(),
                                    last_activity: Instant::now(),
                                };

                                sessions.lock().await.insert(session_id.clone(), session);

                                // Send initial response
                                yield Ok(StreamPredictResponse {
                                    session_id: session_id.clone(),
                                    text: format!("Stream started for: {}", start.initial_text),
                                    token_ids: vec![],
                                    is_final: false,
                                    metrics: None,
                                });
                            }
                            Some(inference::stream_predict_request::Request::Continue(cont)) => {
                                // Continue streaming
                                let sessions_guard = sessions.lock().await;

                                if !sessions_guard.contains_key(&cont.session_id) {
                                    yield Err(Status::not_found("Session not found"));
                                    return;
                                }

                                // Generate response
                                yield Ok(StreamPredictResponse {
                                    session_id: cont.session_id.clone(),
                                    text: format!("Continuing: {}", cont.text),
                                    token_ids: vec![],
                                    is_final: cont.end_stream,
                                    metrics: None,
                                });

                                if cont.end_stream {
                                    sessions_guard.remove(&cont.session_id);
                                }
                            }
                            None => {
                                yield Err(Status::invalid_argument("Invalid request"));
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(output_stream)))
    }

    #[instrument(skip(self, request))]
    async fn batch_predict(
        &self,
        request: Request<BatchPredictRequest>,
    ) -> Result<Response<BatchPredictResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        debug!("Batch predict request for model: {} with {} inputs",
               req.model_id, req.texts.len());

        // Get the model
        let model = self.model_manager
            .get_model(&req.model_id)
            .map_err(|e| Status::from(e))?;

        // Process batch using real model inference
        let mut predictions = Vec::new();
        let mut total_tokens = 0;

        // Create generation config for batch processing
        let generation_config = trustformers_models::common_patterns::GenerationConfig {
            max_new_tokens: req.max_tokens.unwrap_or(50) as usize,
            temperature: req.temperature.unwrap_or(0.7),
            top_p: req.top_p.unwrap_or(0.9),
            top_k: req.top_k.map(|k| k as usize),
            do_sample: req.do_sample.unwrap_or(true),
            repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
            pad_token_id: Some(0),
            eos_token_id: Some(2),
            use_cache: true,
            stream: false,
            ..Default::default()
        };

        // Try batch processing first, fall back to individual processing
        let batch_results = if req.texts.len() > 1 {
            // Use batch generation for efficiency
            let text_refs: Vec<&str> = req.texts.iter().map(|s| s.as_str()).collect();
            match model.model.generate_batch(&text_refs, &generation_config) {
                Ok(batch_outputs) => {
                    info!("Successfully generated batch of {} texts", batch_outputs.len());
                    batch_outputs
                },
                Err(e) => {
                    warn!("Batch generation failed, falling back to individual processing: {}", e);
                    // Fall back to individual processing
                    let mut individual_results = Vec::new();
                    for text in &req.texts {
                        match model.model.generate(text, &generation_config) {
                            Ok(output) => individual_results.push(output),
                            Err(e) => {
                                warn!("Individual generation failed for text: {}", e);
                                individual_results.push(format!("Error: {}", e));
                            }
                        }
                    }
                    individual_results
                }
            }
        } else if req.texts.len() == 1 {
            // Single text processing
            match model.model.generate(&req.texts[0], &generation_config) {
                Ok(output) => vec![output],
                Err(e) => {
                    warn!("Single generation failed: {}", e);
                    vec![format!("Error: {}", e)]
                }
            }
        } else {
            vec![]
        };

        // Process results and create responses
        for (i, output_text) in batch_results.iter().enumerate() {
            // Count tokens using the tokenizer for accurate metrics
            let tokens = match model.tokenizer.encode(output_text) {
                Ok(tokenized) => tokenized.ids.len(),
                Err(_) => output_text.split_whitespace().count(), // fallback to word count
            };
            total_tokens += tokens;

            // Create individual metrics for each prediction
            let individual_metrics = self.create_predict_metrics(start, tokens, &model.device).await;

            predictions.push(PredictResponse {
                output: Some(inference::predict_response::Output::TextOutput(
                    inference::TextOutput {
                        text: output_text.clone(),
                        texts: vec![],
                        scores: vec![],
                        token_ids: vec![],
                    },
                )),
                metrics: Some(individual_metrics),
            });
        }

        let elapsed = start.elapsed();
        let batch_metrics = inference::BatchMetrics {
            total_latency_ms: elapsed.as_secs_f32() * 1000.0,
            avg_latency_ms: (elapsed.as_secs_f32() * 1000.0) / req.texts.len() as f32,
            total_tokens: total_tokens as i32,
            tokens_per_second: (total_tokens as f32 / elapsed.as_secs_f32()) as f32,
            batch_size: req.texts.len() as i32,
        };

        Ok(Response::new(BatchPredictResponse {
            predictions,
            metrics: Some(batch_metrics),
        }))
    }

    #[instrument(skip(self, _request))]
    async fn list_models(
        &self,
        _request: Request<()>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        let models = self.model_manager.list_models();

        let model_infos = models
            .into_iter()
            .map(|(id, info)| ModelInfo {
                model_id: id,
                model_type: "bert".to_string(), // Placeholder
                architecture: "BertForSequenceClassification".to_string(),
                num_parameters: 110_000_000,
                supported_tasks: vec![
                    "text-classification".to_string(),
                    "sentiment-analysis".to_string(),
                ],
                config: Some(ModelConfig {
                    hidden_size: 768,
                    num_layers: 12,
                    num_heads: 12,
                    vocab_size: 30522,
                    max_position_embeddings: 512,
                    model_type: "bert".to_string(),
                    extra_config: None,
                }),
                status: Some(ModelStatus {
                    is_loaded: true,
                    device: info.device,
                    memory_used_bytes: info.memory_used as i64,
                    load_time: info.loaded_at.elapsed().as_secs().to_string(),
                    request_count: info.request_count as i32,
                }),
                metadata: std::collections::HashMap::new(),
            })
            .collect();

        Ok(Response::new(ListModelsResponse {
            models: model_infos,
        }))
    }

    #[instrument(skip(self, request))]
    async fn get_model_info(
        &self,
        request: Request<GetModelInfoRequest>,
    ) -> Result<Response<ModelInfo>, Status> {
        let model_id = request.into_inner().model_id;
        let models = self.model_manager.list_models();

        let (_, info) = models
            .into_iter()
            .find(|(id, _)| id == &model_id)
            .ok_or_else(|| Status::not_found(format!("Model {} not found", model_id)))?;

        Ok(Response::new(ModelInfo {
            model_id,
            model_type: "bert".to_string(),
            architecture: "BertForSequenceClassification".to_string(),
            num_parameters: 110_000_000,
            supported_tasks: vec![
                "text-classification".to_string(),
                "sentiment-analysis".to_string(),
            ],
            config: Some(ModelConfig {
                hidden_size: 768,
                num_layers: 12,
                num_heads: 12,
                vocab_size: 30522,
                max_position_embeddings: 512,
                model_type: "bert".to_string(),
                extra_config: None,
            }),
            status: Some(ModelStatus {
                is_loaded: true,
                device: info.device,
                memory_used_bytes: info.memory_used as i64,
                load_time: info.loaded_at.elapsed().as_secs().to_string(),
                request_count: info.request_count as i32,
            }),
            metadata: std::collections::HashMap::new(),
        }))
    }

    #[instrument(skip(self, request))]
    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        let req = request.into_inner();

        info!("Loading model: {}", req.model_id);

        let load_time = self.model_manager
            .load_model(
                &req.model_id,
                Some(&req.model_path),
                Some(&req.device),
                req.use_fp16,
                req.compile,
            )
            .await
            .map_err(|e| Status::from(e))?;

        Ok(Response::new(LoadModelResponse {
            model_id: req.model_id,
            success: true,
            message: "Model loaded successfully".to_string(),
            load_time_ms: load_time.as_secs_f32() * 1000.0,
        }))
    }

    #[instrument(skip(self, request))]
    async fn unload_model(
        &self,
        request: Request<UnloadModelRequest>,
    ) -> Result<Response<()>, Status> {
        let model_id = request.into_inner().model_id;

        info!("Unloading model: {}", model_id);

        self.model_manager
            .unload_model(&model_id)
            .await
            .map_err(|e| Status::from(e))?;

        Ok(Response::new(()))
    }
}