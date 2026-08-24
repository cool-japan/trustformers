//! `trustformers.inference.InferenceService` implementation.
//!
//! Every response this module produces comes from a real forward pass or from
//! the loaded checkpoint's own configuration. Where the proto has a field this
//! server genuinely cannot measure (`PredictMetrics::memory_used_bytes`), the
//! field is left at its proto default rather than filled with a plausible
//! looking guess.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status};
use tracing::{debug, info, instrument};
use uuid::Uuid;

use trustformers::core::traits::Tokenizer;
use trustformers_models::common_patterns::GenerationConfig;

use crate::error::{ServiceError, ServiceResult};
use crate::model_manager::{LoadedModel, ModelManager, ModelSnapshot};

// Import generated proto types.
//
// `include_proto!` splices in machine-generated code: prost/tonic emit deeply
// nested `match` arms in the service dispatcher and a `*Output` suffix on every
// variant of the `PredictResponse` oneof. Both trip workspace clippy settings
// (`excessive-nesting-threshold` / `enum-variant-name-threshold`) in code this
// crate does not author and cannot reformat.
#[allow(clippy::excessive_nesting, clippy::enum_variant_names)]
pub mod inference {
    tonic::include_proto!("trustformers.inference");
}

use inference::{
    inference_service_server::InferenceService, BatchMetrics, BatchPredictRequest,
    BatchPredictResponse, GetModelInfoRequest, ListModelsResponse, LoadModelRequest,
    LoadModelResponse, ModelConfig, ModelInfo, ModelStatus, PredictMetrics, PredictOptions,
    PredictRequest, PredictResponse, StreamPredictRequest, StreamPredictResponse, TextOutput,
    UnloadModelRequest,
};

/// How many decoding steps may sit in the channel between the blocking
/// generation task and the async response stream before the producer waits.
const STREAM_CHANNEL_CAPACITY: usize = 32;

/// One decoding step handed from the blocking generation task to the async
/// response stream: the token id, the text that became visible because of it,
/// and whether it was the model's end-of-sequence marker.
type StreamedStep = (u32, String, bool);

pub struct InferenceServiceImpl {
    model_manager: Arc<ModelManager>,
}

/// State of one `StreamPredict` conversation.
///
/// Sessions are scoped to a single bidirectional call: the map lives inside the
/// response stream, so a client that disconnects cannot leave a session behind.
struct StreamSession {
    model: Arc<LoadedModel>,
    config: GenerationConfig,
    /// Everything the model has been given or has produced in this session.
    /// A `Continue` conditions on this, so the conversation really does build
    /// on what came before instead of restarting from the new text alone.
    transcript: String,
}

impl InferenceServiceImpl {
    pub fn new(model_manager: Arc<ModelManager>) -> Self {
        Self { model_manager }
    }
}

/// Saturating `u64` -> `i32` for proto fields that are narrower than the value
/// they carry. Only reached by counters far beyond any real deployment.
fn as_i32(value: u64) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

/// Saturating `u64` -> `i64` for the proto's parameter count.
fn as_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// Converts generated token ids into the proto's `int32` `token_ids` field.
///
/// Fails loudly instead of truncating: a vocabulary large enough to overflow
/// `i32` would silently corrupt every id, which is exactly the kind of quietly
/// wrong output this server must not produce.
fn token_ids_to_proto(ids: &[u32]) -> Result<Vec<i32>, Status> {
    ids.iter()
        .map(|&id| {
            i32::try_from(id).map_err(|_| {
                Status::internal(format!(
                    "token id {id} does not fit the proto `int32 token_ids` field"
                ))
            })
        })
        .collect()
}

/// Tokens per second, or `0` when the run was too short to time.
fn tokens_per_second(tokens: usize, elapsed: Duration) -> i32 {
    let seconds = elapsed.as_secs_f64();
    if seconds <= 0.0 {
        return 0;
    }
    let rate = tokens as f64 / seconds;
    if rate.is_finite() {
        // `as` saturates at the integer bounds for out-of-range floats.
        rate.round() as i32
    } else {
        0
    }
}

/// Metrics for one completed generation.
///
/// `memory_used_bytes` is deliberately left at the proto default: this server
/// measures no process or device memory, and emitting a parameter-count-derived
/// estimate would be fabricated telemetry.
fn predict_metrics(elapsed: Duration, tokens: usize, device: &str) -> PredictMetrics {
    PredictMetrics {
        latency_ms: elapsed.as_secs_f32() * 1000.0,
        tokens_per_second: tokens_per_second(tokens, elapsed),
        memory_used_bytes: 0,
        device_used: device.to_string(),
    }
}

/// Translates `PredictOptions` into the library's `GenerationConfig`.
///
/// Proto3 scalars have no "unset" state, so the rule is explicit: when the
/// client sends no `options` message at all the library defaults apply
/// unchanged; when it does send one, every field in it is honoured literally
/// (including `do_sample` / `use_cache`, which are plain `bool`s and therefore
/// default to `false` on the wire).
///
/// Options this server cannot honour are rejected rather than ignored.
fn generation_config(options: Option<&PredictOptions>, default_device: &str) -> ServiceResult<GenerationConfig> {
    let mut config = GenerationConfig::default();
    let Some(options) = options else {
        return Ok(config);
    };

    if !options.stop_sequences.is_empty() {
        return Err(ServiceError::Unsupported(
            "`stop_sequences` is not supported: the generation engine stops on the model's \
             end-of-sequence token or the token budget, and has no string stop criterion"
                .to_string(),
        ));
    }
    if options.seed != 0 {
        return Err(ServiceError::Unsupported(
            "`seed` is not supported: this server cannot pin the sampler's RNG, so it will \
             not promise reproducible output it cannot deliver"
                .to_string(),
        ));
    }
    if options.use_fp16 {
        return Err(ServiceError::Unsupported(
            "`use_fp16` is not supported: inference runs in whatever precision the \
             checkpoint was loaded with"
                .to_string(),
        ));
    }
    if options.extra_params.is_some() {
        return Err(ServiceError::Unsupported(
            "`extra_params` is not supported: this server would have to ignore it".to_string(),
        ));
    }
    let device = options.device.trim();
    if !device.is_empty() && !device.eq_ignore_ascii_case(default_device) {
        return Err(ServiceError::Unsupported(format!(
            "device `{device}` is not available: models are served on `{default_device}`"
        )));
    }

    if options.max_new_tokens < 0 {
        return Err(ServiceError::InvalidInput(
            "`max_new_tokens` must not be negative".to_string(),
        ));
    }
    if options.top_k < 0 {
        return Err(ServiceError::InvalidInput(
            "`top_k` must not be negative".to_string(),
        ));
    }
    if options.num_beams < 0 {
        return Err(ServiceError::InvalidInput(
            "`num_beams` must not be negative".to_string(),
        ));
    }
    if options.temperature < 0.0 || !options.temperature.is_finite() {
        return Err(ServiceError::InvalidInput(
            "`temperature` must be a finite, non-negative number".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&options.top_p) {
        return Err(ServiceError::InvalidInput(
            "`top_p` must lie in [0, 1]".to_string(),
        ));
    }
    if options.repetition_penalty < 0.0 || !options.repetition_penalty.is_finite() {
        return Err(ServiceError::InvalidInput(
            "`repetition_penalty` must be a finite, non-negative number".to_string(),
        ));
    }

    // A zero on the wire means "not set" for the numeric knobs, so the library
    // default survives; anything positive is applied as sent.
    if options.max_new_tokens > 0 {
        config.max_new_tokens = options.max_new_tokens as usize;
    }
    if options.top_k > 0 {
        config.top_k = Some(options.top_k as usize);
    }
    if options.top_p > 0.0 {
        config.top_p = options.top_p;
    }
    if options.temperature > 0.0 {
        config.temperature = options.temperature;
    }
    if options.repetition_penalty > 0.0 {
        config.repetition_penalty = options.repetition_penalty;
    }
    if options.num_beams > 0 {
        // Beam search is implemented for encoder-decoder checkpoints (T5);
        // decoder-only architectures decode token by token and ignore it.
        config.num_beams = Some(options.num_beams as usize);
    }
    config.do_sample = options.do_sample;
    config.use_cache = options.use_cache;

    Ok(config)
}

/// Pulls the prompt out of the request's `oneof input`.
fn extract_prompt(input: Option<inference::predict_request::Input>) -> ServiceResult<String> {
    match input {
        Some(inference::predict_request::Input::Text(text)) => Ok(text),
        Some(inference::predict_request::Input::TextInput(text_input)) => Ok(text_input.text),
        Some(inference::predict_request::Input::ImageInput(_)) => Err(ServiceError::Unsupported(
            "image input is not supported: this server only serves text checkpoints".to_string(),
        )),
        None => Err(ServiceError::InvalidInput(
            "no input was provided".to_string(),
        )),
    }
}

/// Runs one complete generation on the blocking pool and returns the decoded
/// completion together with the token ids that produced it.
async fn generate_once(
    loaded: &Arc<LoadedModel>,
    prompt: String,
    config: GenerationConfig,
) -> Result<(String, Vec<u32>), Status> {
    let model = Arc::clone(&loaded.model);
    // The error is flattened to its message inside the closure: the library's
    // error enum is large, and carrying it across the task boundary would make
    // every `Result` in this path pay for it.
    let generated = tokio::task::spawn_blocking(move || {
        model
            .generate_token_ids(&prompt, &config)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| Status::internal(format!("generation task failed to run: {e}")))?
    .map_err(|message| Status::from(ServiceError::Generation(message)))?;

    let completion_ids = generated.completion_ids().to_vec();
    let text = loaded
        .tokenizer
        .decode(&completion_ids)
        .map_err(|e| Status::internal(format!("failed to decode generated tokens: {e}")))?;

    Ok((text, completion_ids))
}

/// Starts a real incremental decode on the blocking pool.
///
/// Each decoding step is forwarded over the returned channel as it happens, so
/// the client sees the first token after one forward pass rather than after the
/// whole run.
fn spawn_incremental_generation(
    loaded: &Arc<LoadedModel>,
    prompt: String,
    config: GenerationConfig,
) -> mpsc::Receiver<Result<StreamedStep, String>> {
    let (tx, rx) = mpsc::channel::<Result<StreamedStep, String>>(STREAM_CHANNEL_CAPACITY);
    let model = Arc::clone(&loaded.model);

    tokio::task::spawn_blocking(move || {
        let stream = match model.token_stream(&prompt, &config) {
            Ok(stream) => stream,
            Err(e) => {
                let _ = tx.blocking_send(Err(e.to_string()));
                return;
            },
        };

        for step in stream {
            let payload = match step {
                Ok(step) => Ok((step.token_id, step.text_delta, step.is_eos)),
                Err(e) => Err(e.to_string()),
            };
            let was_error = payload.is_err();
            // A send error means the client hung up; stop decoding.
            if tx.blocking_send(payload).is_err() || was_error {
                return;
            }
        }
    });

    rx
}

/// Builds the proto view of a loaded model out of its real configuration.
fn model_info(snapshot: &ModelSnapshot) -> ModelInfo {
    let mut metadata = HashMap::new();
    metadata.insert(
        "uptime_seconds".to_string(),
        format!("{:.3}", snapshot.uptime.as_secs_f64()),
    );
    metadata.insert("generative".to_string(), snapshot.is_generative.to_string());

    ModelInfo {
        model_id: snapshot.model_id.clone(),
        // `AutoConfig` exposes a single architecture identifier (`gpt2`,
        // `bert`, ...); there is no separate HuggingFace class name to report,
        // so both proto fields carry that one real value rather than one real
        // value and one invented one.
        model_type: snapshot.architecture.to_string(),
        architecture: snapshot.architecture.to_string(),
        num_parameters: as_i64(snapshot.num_parameters),
        supported_tasks: snapshot.supported_tasks(),
        config: Some(ModelConfig {
            hidden_size: as_i32(u64::from(snapshot.hidden_size)),
            num_layers: as_i32(u64::from(snapshot.num_layers)),
            num_heads: as_i32(u64::from(snapshot.num_heads)),
            vocab_size: as_i32(u64::from(snapshot.vocab_size)),
            max_position_embeddings: as_i32(u64::from(snapshot.max_position_embeddings)),
            model_type: snapshot.architecture.to_string(),
            extra_config: None,
        }),
        status: Some(ModelStatus {
            is_loaded: true,
            device: snapshot.device.clone(),
            // Not measured; see `predict_metrics`.
            memory_used_bytes: 0,
            load_time: format!("{:.3}s", snapshot.load_duration.as_secs_f64()),
            request_count: as_i32(snapshot.request_count),
        }),
        metadata,
    }
}

#[tonic::async_trait]
impl InferenceService for InferenceServiceImpl {
    #[instrument(skip(self, request))]
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();
        debug!(model_id = %req.model_id, "predict");

        let loaded = self.model_manager.get_model(&req.model_id)?;
        let config = generation_config(req.options.as_ref(), &loaded.device)?;
        let prompt = extract_prompt(req.input)?;
        loaded.record_request();

        let start = Instant::now();
        let (text, completion_ids) = generate_once(&loaded, prompt, config).await?;
        let elapsed = start.elapsed();

        let token_ids = token_ids_to_proto(&completion_ids)?;
        let metrics = predict_metrics(elapsed, completion_ids.len(), &loaded.device);

        Ok(Response::new(PredictResponse {
            output: Some(inference::predict_response::Output::TextOutput(TextOutput {
                text,
                texts: vec![],
                scores: vec![],
                token_ids,
            })),
            metrics: Some(metrics),
        }))
    }

    type StreamPredictStream =
        Pin<Box<dyn Stream<Item = Result<StreamPredictResponse, Status>> + Send + 'static>>;

    /// Bidirectional streaming generation.
    ///
    /// A `Start` opens a session and generates from `initial_text`; each later
    /// `Continue` generates from the session transcript plus the new text, so
    /// the conversation really does build on what came before. Every turn emits
    /// one response per decoded token (`is_final = false`) followed by exactly
    /// one summary response with `is_final = true` carrying that turn's
    /// metrics. A `Continue` with `end_stream` set also closes the session; a
    /// session that is still open simply waits for the next request on the same
    /// call, and is dropped when the call ends.
    #[instrument(skip(self, request))]
    async fn stream_predict(
        &self,
        request: Request<tonic::Streaming<StreamPredictRequest>>,
    ) -> Result<Response<Self::StreamPredictStream>, Status> {
        let mut requests = request.into_inner();
        let model_manager = Arc::clone(&self.model_manager);

        let output = async_stream::stream! {
            // Sessions live for exactly this call, so a disconnect cannot leak
            // one.
            let mut sessions: HashMap<String, StreamSession> = HashMap::new();

            while let Some(message) = requests.next().await {
                let message = match message {
                    Ok(message) => message,
                    Err(status) => {
                        yield Err(status);
                        return;
                    },
                };

                let (session_id, prompt, close_after) = match message.request {
                    Some(inference::stream_predict_request::Request::Start(start)) => {
                        let loaded = match model_manager.get_model(&start.model_id) {
                            Ok(loaded) => loaded,
                            Err(e) => {
                                yield Err(Status::from(e));
                                return;
                            },
                        };
                        let config = match generation_config(
                            start.options.as_ref(),
                            &loaded.device,
                        ) {
                            Ok(config) => config,
                            Err(e) => {
                                yield Err(Status::from(e));
                                return;
                            },
                        };

                        let session_id = if start.session_id.trim().is_empty() {
                            Uuid::new_v4().to_string()
                        } else {
                            start.session_id.clone()
                        };
                        if sessions.contains_key(&session_id) {
                            yield Err(Status::already_exists(format!(
                                "session `{session_id}` is already open on this stream"
                            )));
                            return;
                        }

                        let prompt = start.initial_text.clone();
                        sessions.insert(
                            session_id.clone(),
                            StreamSession {
                                model: loaded,
                                config,
                                transcript: prompt.clone(),
                            },
                        );
                        info!(session_id, model_id = %start.model_id, "stream session opened");
                        (session_id, prompt, false)
                    },
                    Some(inference::stream_predict_request::Request::Continue(cont)) => {
                        let Some(session) = sessions.get_mut(&cont.session_id) else {
                            yield Err(Status::not_found(format!(
                                "no open session `{}` on this stream",
                                cont.session_id
                            )));
                            return;
                        };
                        session.transcript.push_str(&cont.text);
                        (
                            cont.session_id.clone(),
                            session.transcript.clone(),
                            cont.end_stream,
                        )
                    },
                    None => {
                        yield Err(Status::invalid_argument(
                            "StreamPredictRequest carried neither `start` nor `continue`",
                        ));
                        return;
                    },
                };

                let Some(session) = sessions.get(&session_id) else {
                    yield Err(Status::internal(format!(
                        "session `{session_id}` vanished while it was being served"
                    )));
                    return;
                };
                let loaded = Arc::clone(&session.model);
                let config = session.config.clone();
                loaded.record_request();

                let start_time = Instant::now();
                let mut receiver =
                    spawn_incremental_generation(&loaded, prompt, config);
                let mut produced = String::new();
                let mut token_count = 0usize;

                while let Some(step) = receiver.recv().await {
                    let (token_id, text_delta, _is_eos) = match step {
                        Ok(step) => step,
                        Err(message) => {
                            yield Err(Status::from(ServiceError::Generation(message)));
                            return;
                        },
                    };

                    let token_ids = match token_ids_to_proto(&[token_id]) {
                        Ok(ids) => ids,
                        Err(status) => {
                            yield Err(status);
                            return;
                        },
                    };

                    produced.push_str(&text_delta);
                    token_count += 1;

                    yield Ok(StreamPredictResponse {
                        session_id: session_id.clone(),
                        text: text_delta,
                        token_ids,
                        is_final: false,
                        metrics: None,
                    });
                }

                let elapsed = start_time.elapsed();
                if let Some(session) = sessions.get_mut(&session_id) {
                    session.transcript.push_str(&produced);
                }
                if close_after {
                    sessions.remove(&session_id);
                    info!(session_id, "stream session closed");
                }

                // `is_final` marks the end of THIS turn's token stream, which
                // is what a client's `for response in responses: ... if
                // response.is_final: break` loop waits for.
                yield Ok(StreamPredictResponse {
                    session_id: session_id.clone(),
                    text: String::new(),
                    token_ids: vec![],
                    is_final: true,
                    metrics: Some(predict_metrics(elapsed, token_count, &loaded.device)),
                });
            }
        };

        Ok(Response::new(Box::pin(output)))
    }

    /// Batch generation.
    ///
    /// Each input is generated for real; there is no batched kernel behind this
    /// yet, so the inputs are processed one after another. A failure fails the
    /// whole call rather than being written into the response as if it were a
    /// prediction.
    #[instrument(skip(self, request))]
    async fn batch_predict(
        &self,
        request: Request<BatchPredictRequest>,
    ) -> Result<Response<BatchPredictResponse>, Status> {
        let req = request.into_inner();
        debug!(model_id = %req.model_id, inputs = req.texts.len(), "batch predict");

        if req.texts.is_empty() {
            return Err(Status::from(ServiceError::InvalidInput(
                "`texts` was empty; nothing to predict".to_string(),
            )));
        }

        let loaded = self.model_manager.get_model(&req.model_id)?;
        let config = generation_config(req.options.as_ref(), &loaded.device)?;

        let batch_start = Instant::now();
        let mut predictions = Vec::with_capacity(req.texts.len());
        let mut total_tokens = 0usize;

        for text in &req.texts {
            loaded.record_request();
            let start = Instant::now();
            let (output_text, completion_ids) =
                generate_once(&loaded, text.clone(), config.clone()).await?;
            let elapsed = start.elapsed();

            total_tokens += completion_ids.len();
            let token_ids = token_ids_to_proto(&completion_ids)?;

            predictions.push(PredictResponse {
                output: Some(inference::predict_response::Output::TextOutput(TextOutput {
                    text: output_text,
                    texts: vec![],
                    scores: vec![],
                    token_ids,
                })),
                metrics: Some(predict_metrics(
                    elapsed,
                    completion_ids.len(),
                    &loaded.device,
                )),
            });
        }

        let elapsed = batch_start.elapsed();
        let total_ms = elapsed.as_secs_f32() * 1000.0;
        let batch_size = req.texts.len();

        Ok(Response::new(BatchPredictResponse {
            predictions,
            metrics: Some(BatchMetrics {
                total_latency_ms: total_ms,
                avg_latency_ms: total_ms / batch_size as f32,
                total_tokens: as_i32(total_tokens as u64),
                tokens_per_second: {
                    let seconds = elapsed.as_secs_f32();
                    if seconds > 0.0 {
                        total_tokens as f32 / seconds
                    } else {
                        0.0
                    }
                },
                batch_size: as_i32(batch_size as u64),
            }),
        }))
    }

    #[instrument(skip(self, _request))]
    async fn list_models(
        &self,
        _request: Request<()>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        let models = self
            .model_manager
            .list_models()
            .iter()
            .map(model_info)
            .collect();

        Ok(Response::new(ListModelsResponse { models }))
    }

    #[instrument(skip(self, request))]
    async fn get_model_info(
        &self,
        request: Request<GetModelInfoRequest>,
    ) -> Result<Response<ModelInfo>, Status> {
        let model_id = request.into_inner().model_id;
        let loaded = self
            .model_manager
            .get_model(&model_id)
            .map_err(|_| ServiceError::ModelNotFound(model_id.clone()))?;

        Ok(Response::new(model_info(&loaded.snapshot(&model_id))))
    }

    #[instrument(skip(self, request))]
    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        let req = request.into_inner();
        info!(model_id = %req.model_id, "load model");

        let model_manager = Arc::clone(&self.model_manager);
        let model_id = req.model_id.clone();
        let model_path = req.model_path.clone();
        let device = req.device.clone();

        // Loading reads and parses a whole checkpoint, so it runs on the
        // blocking pool instead of stalling the async reactor.
        let load_duration: ServiceResult<Duration> = tokio::task::spawn_blocking(move || {
            model_manager.load_model(
                &model_id,
                Some(model_path.as_str()),
                Some(device.as_str()),
                req.use_fp16,
                req.compile,
            )
        })
        .await
        .map_err(|e| Status::internal(format!("model load task failed to run: {e}")))?;
        let load_duration = load_duration?;

        Ok(Response::new(LoadModelResponse {
            model_id: req.model_id,
            success: true,
            message: format!(
                "loaded in {:.3}s",
                load_duration.as_secs_f64()
            ),
            load_time_ms: load_duration.as_secs_f32() * 1000.0,
        }))
    }

    #[instrument(skip(self, request))]
    async fn unload_model(
        &self,
        request: Request<UnloadModelRequest>,
    ) -> Result<Response<()>, Status> {
        let model_id = request.into_inner().model_id;
        info!(model_id, "unload model");

        self.model_manager.unload_model(&model_id)?;

        Ok(Response::new(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> PredictOptions {
        PredictOptions::default()
    }

    #[test]
    fn absent_options_leave_the_library_defaults_alone() {
        let defaults = GenerationConfig::default();
        let config = generation_config(None, "cpu").expect("no options is valid");
        assert_eq!(config.max_new_tokens, defaults.max_new_tokens);
        assert_eq!(config.do_sample, defaults.do_sample);
        assert_eq!(config.use_cache, defaults.use_cache);
    }

    #[test]
    fn positive_options_are_applied_as_sent() {
        let config = generation_config(
            Some(&PredictOptions {
                temperature: 0.25,
                top_k: 7,
                top_p: 0.5,
                max_new_tokens: 13,
                do_sample: true,
                num_beams: 3,
                repetition_penalty: 1.5,
                use_cache: true,
                ..options()
            }),
            "cpu",
        )
        .expect("valid options");

        assert_eq!(config.max_new_tokens, 13);
        assert_eq!(config.top_k, Some(7));
        assert_eq!(config.num_beams, Some(3));
        assert!((config.temperature - 0.25).abs() < f32::EPSILON);
        assert!((config.top_p - 0.5).abs() < f32::EPSILON);
        assert!((config.repetition_penalty - 1.5).abs() < f32::EPSILON);
        assert!(config.do_sample);
        assert!(config.use_cache);
    }

    #[test]
    fn options_the_server_cannot_honour_are_refused_not_ignored() {
        let cases = [
            PredictOptions {
                stop_sequences: vec!["</s>".to_string()],
                ..options()
            },
            PredictOptions {
                seed: 42,
                ..options()
            },
            PredictOptions {
                use_fp16: true,
                ..options()
            },
            PredictOptions {
                device: "cuda".to_string(),
                ..options()
            },
        ];

        for case in cases {
            let err = generation_config(Some(&case), "cpu")
                .expect_err("an unhonourable option must not be silently dropped");
            assert!(matches!(err, ServiceError::Unsupported(_)), "got {err:?}");
        }
    }

    #[test]
    fn out_of_range_sampling_parameters_are_rejected() {
        let cases = [
            PredictOptions {
                top_p: 1.5,
                ..options()
            },
            PredictOptions {
                temperature: -1.0,
                ..options()
            },
            PredictOptions {
                max_new_tokens: -1,
                ..options()
            },
            PredictOptions {
                top_k: -1,
                ..options()
            },
            PredictOptions {
                num_beams: -1,
                ..options()
            },
            PredictOptions {
                repetition_penalty: f32::NAN,
                ..options()
            },
        ];

        for case in cases {
            let err = generation_config(Some(&case), "cpu")
                .expect_err("an out-of-range parameter must be rejected");
            assert!(matches!(err, ServiceError::InvalidInput(_)), "got {err:?}");
        }
    }

    #[test]
    fn image_input_is_refused_and_missing_input_is_invalid() {
        let err = extract_prompt(Some(inference::predict_request::Input::ImageInput(
            inference::ImageInput::default(),
        )))
        .expect_err("image input must be refused");
        assert!(matches!(err, ServiceError::Unsupported(_)), "got {err:?}");

        let err = extract_prompt(None).expect_err("no input must be invalid");
        assert!(matches!(err, ServiceError::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn throughput_is_zero_for_an_untimed_run_and_never_infinite() {
        assert_eq!(tokens_per_second(10, Duration::ZERO), 0);
        assert_eq!(tokens_per_second(0, Duration::from_secs(1)), 0);
        assert_eq!(tokens_per_second(20, Duration::from_secs(2)), 10);
    }

    #[test]
    fn metrics_do_not_invent_a_memory_measurement() {
        let metrics = predict_metrics(Duration::from_millis(500), 5, "cpu");
        assert_eq!(
            metrics.memory_used_bytes, 0,
            "memory is not measured, so the field must stay at its proto default"
        );
        assert_eq!(metrics.device_used, "cpu");
        assert_eq!(metrics.tokens_per_second, 10);
    }

    #[test]
    fn model_info_reports_the_checkpoint_shape_not_a_hardcoded_bert() {
        let snapshot = ModelSnapshot {
            model_id: "tiny".to_string(),
            architecture: "gpt2",
            num_parameters: 124_439_808,
            is_generative: true,
            hidden_size: 1600,
            num_layers: 48,
            num_heads: 25,
            vocab_size: 50_257,
            max_position_embeddings: 1024,
            device: "cpu".to_string(),
            load_duration: Duration::from_millis(1500),
            uptime: Duration::from_secs(7),
            request_count: 3,
        };

        let info = model_info(&snapshot);
        assert_eq!(info.model_type, "gpt2");
        assert_eq!(info.architecture, "gpt2");
        assert_eq!(info.num_parameters, 124_439_808);
        assert_eq!(
            info.supported_tasks,
            vec![
                "feature-extraction".to_string(),
                "text-generation".to_string()
            ]
        );

        let config = info.config.expect("config is always reported");
        assert_eq!(config.hidden_size, 1600);
        assert_eq!(config.num_layers, 48);
        assert_eq!(config.num_heads, 25);
        assert_eq!(config.vocab_size, 50_257);
        assert_eq!(config.max_position_embeddings, 1024);

        let status = info.status.expect("status is always reported");
        assert_eq!(status.request_count, 3);
        assert_eq!(status.load_time, "1.500s");
        assert_eq!(status.memory_used_bytes, 0);
    }

    #[test]
    fn token_ids_that_do_not_fit_the_wire_field_are_an_error_not_a_truncation() {
        let ok = token_ids_to_proto(&[0, 1, 50_256]).expect("normal ids fit");
        assert_eq!(ok, vec![0, 1, 50_256]);

        let err = token_ids_to_proto(&[u32::MAX]).expect_err("an out-of-range id must fail");
        assert_eq!(err.code(), tonic::Code::Internal);
    }
}
