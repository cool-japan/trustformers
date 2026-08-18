//! Shadow Mode Testing
//!
//! Provides shadow mode testing capabilities for safe model deployment.
//! Shadow mode allows testing new models against production traffic
//! without impacting user experience.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, mpsc, Mutex},
    time::timeout,
};
use uuid::Uuid;

/// Shadow mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowConfig {
    /// Enable shadow mode
    pub enabled: bool,

    /// Percentage of traffic to shadow (0-100)
    pub traffic_percentage: f64,

    /// Maximum shadow request timeout (seconds)
    pub shadow_timeout_seconds: u64,

    /// Maximum number of concurrent shadow requests
    pub max_concurrent_shadow_requests: usize,

    /// Shadow results storage size
    pub max_shadow_results: usize,

    /// Enable detailed logging
    pub enable_detailed_logging: bool,

    /// Shadow model configurations
    pub shadow_models: HashMap<String, ShadowModelConfig>,

    /// Version string reported for the production side of every comparison.
    pub production_model_version: String,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            traffic_percentage: 10.0,
            shadow_timeout_seconds: 30,
            max_concurrent_shadow_requests: 100,
            max_shadow_results: 10000,
            enable_detailed_logging: true,
            shadow_models: HashMap::new(),
            production_model_version: crate::VERSION.to_string(),
        }
    }
}

/// Shadow model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowModelConfig {
    /// Model name or identifier
    pub model_name: String,

    /// Model version
    pub version: String,

    /// Model endpoint or configuration
    pub endpoint: Option<String>,

    /// Model-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,

    /// Enable for this model
    pub enabled: bool,

    /// Specific traffic percentage for this model
    pub traffic_percentage: Option<f64>,
}

/// Shadow request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowRequest {
    /// Request ID
    pub request_id: String,

    /// Original request payload
    pub payload: serde_json::Value,

    /// Request timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Client information
    pub client_info: Option<ClientInfo>,

    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// Client information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Client ID
    pub client_id: Option<String>,

    /// IP address
    pub ip_address: Option<String>,

    /// User agent
    pub user_agent: Option<String>,

    /// Additional headers
    pub headers: HashMap<String, String>,
}

/// Shadow response
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ShadowResponse {
    /// Response ID
    pub response_id: String,

    /// Request ID this response belongs to
    pub request_id: String,

    /// Model that generated this response
    pub model_name: String,

    /// Model version
    pub model_version: String,

    /// Response payload
    pub payload: serde_json::Value,

    /// Processing time in milliseconds
    pub processing_time_ms: f64,

    /// Response timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Response status (success/error)
    pub status: ShadowResponseStatus,

    /// Error information if applicable
    pub error: Option<String>,

    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Shadow response status
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub enum ShadowResponseStatus {
    Success,
    Error,
    Timeout,
    Cancelled,
}

/// Shadow comparison result
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ShadowComparison {
    /// Comparison ID
    pub comparison_id: String,

    /// Request ID
    pub request_id: String,

    /// Production response
    pub production_response: ShadowResponse,

    /// Shadow responses
    pub shadow_responses: Vec<ShadowResponse>,

    /// Comparison metrics
    pub comparison_metrics: ComparisonMetrics,

    /// Comparison timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Comparison metrics
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ComparisonMetrics {
    /// Response similarity score (0-1)
    pub similarity_score: f64,

    /// Latency difference (ms)
    pub latency_difference_ms: f64,

    /// Response length difference
    pub response_length_difference: i64,

    /// Content differences
    pub content_differences: Vec<String>,

    /// Token-level differences (for text generation)
    pub token_differences: Option<TokenDifferences>,

    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Token-level differences
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct TokenDifferences {
    /// Token overlap percentage
    pub token_overlap: f64,

    /// BLEU score
    pub bleu_score: Option<f64>,

    /// ROUGE score
    pub rouge_score: Option<f64>,

    /// Semantic similarity score
    pub semantic_similarity: Option<f64>,
}

/// Shadow testing statistics
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ShadowStats {
    /// Total requests processed
    pub total_requests: u64,

    /// Total shadow requests sent
    pub total_shadow_requests: u64,

    /// Shadow request success rate
    pub shadow_success_rate: f64,

    /// Average shadow latency
    pub avg_shadow_latency_ms: f64,

    /// Average similarity score
    pub avg_similarity_score: f64,

    /// Model-specific statistics
    pub model_stats: HashMap<String, ModelShadowStats>,

    /// Error statistics
    pub error_stats: HashMap<String, u64>,
}

/// Model-specific shadow statistics
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ModelShadowStats {
    /// Model name
    pub model_name: String,

    /// Total requests
    pub total_requests: u64,

    /// Success requests
    pub success_requests: u64,

    /// Error requests
    pub error_requests: u64,

    /// Timeout requests
    pub timeout_requests: u64,

    /// Average latency
    pub avg_latency_ms: f64,

    /// Average similarity
    pub avg_similarity_score: f64,
}

/// Shadow testing service
pub struct ShadowTestingService {
    config: ShadowConfig,

    /// Active shadow requests
    active_requests: Arc<RwLock<HashMap<String, ShadowRequest>>>,

    /// Shadow results storage
    shadow_results: Arc<Mutex<Vec<ShadowComparison>>>,

    /// Statistics
    stats: Arc<RwLock<ShadowStats>>,

    /// Event broadcaster for shadow events
    event_sender: broadcast::Sender<ShadowEvent>,

    /// Request sender for shadow processing. Carries the real production
    /// response alongside the request so the comparator never has to invent one.
    request_sender: mpsc::UnboundedSender<(ShadowRequest, ShadowResponse)>,

    /// Invoker used to run the configured shadow models. When absent, shadow
    /// responses are recorded as explicit errors rather than being synthesized.
    invoker: Option<Arc<dyn ShadowModelInvoker>>,

    /// Background task handles
    task_handles: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

/// Runs a shadow model for a mirrored request.
///
/// Implementors must perform real inference. There is deliberately no default
/// implementation: a service without an invoker reports an error instead of a
/// plausible-looking payload.
#[async_trait::async_trait]
pub trait ShadowModelInvoker: Send + Sync {
    /// Invoke `model_name` for `request` and return its real response payload.
    async fn invoke(
        &self,
        model_name: &str,
        model_config: &ShadowModelConfig,
        request: &ShadowRequest,
    ) -> Result<serde_json::Value>;
}

/// A [`ShadowModelInvoker`] backed by a real batching service.
///
/// The mirrored request's `text` field is submitted to the batching stack and
/// the model's real output is returned.
pub struct BatchingShadowInvoker {
    batching_service: Arc<crate::batching::DynamicBatchingService>,
}

impl BatchingShadowInvoker {
    pub fn new(batching_service: Arc<crate::batching::DynamicBatchingService>) -> Self {
        Self { batching_service }
    }
}

#[async_trait::async_trait]
impl ShadowModelInvoker for BatchingShadowInvoker {
    async fn invoke(
        &self,
        model_name: &str,
        _model_config: &ShadowModelConfig,
        request: &ShadowRequest,
    ) -> Result<serde_json::Value> {
        let text = request
            .payload
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("mirrored payload has no 'text' field to shadow"))?
            .to_string();
        let max_length =
            request.payload.get("max_length").and_then(|v| v.as_u64()).map(|v| v as usize);

        let internal = crate::batching::Request {
            id: crate::batching::RequestId::new(),
            input: crate::batching::aggregator::RequestInput::Text { text, max_length },
            priority: crate::batching::config::Priority::Low,
            submitted_at: std::time::Instant::now(),
            deadline: None,
            metadata: HashMap::new(),
        };

        let result = self.batching_service.submit_request(internal).await?;
        match result.output {
            crate::batching::aggregator::ProcessingOutput::Text(text) => Ok(serde_json::json!({
                "text": text,
                "tokens": text.split_whitespace().collect::<Vec<_>>(),
                "model": model_name,
            })),
            crate::batching::aggregator::ProcessingOutput::Tokens(tokens) => {
                Ok(serde_json::json!({
                    "tokens": tokens,
                    "model": model_name,
                }))
            },
            crate::batching::aggregator::ProcessingOutput::Error(error) => {
                Err(anyhow::anyhow!(error))
            },
            other => Err(anyhow::anyhow!(
                "unsupported shadow output type: {:?}",
                other
            )),
        }
    }
}

/// Shadow testing events
#[derive(Debug, Clone, Serialize)]
pub enum ShadowEvent {
    /// Shadow request started
    RequestStarted {
        request_id: String,
        model_name: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Shadow request completed
    RequestCompleted {
        request_id: String,
        model_name: String,
        status: ShadowResponseStatus,
        processing_time_ms: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Shadow comparison completed
    ComparisonCompleted {
        comparison_id: String,
        request_id: String,
        similarity_score: f64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Shadow error occurred
    ErrorOccurred {
        request_id: String,
        error: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
}

impl ShadowTestingService {
    /// Create a new shadow testing service with no model invoker.
    ///
    /// Mirrored requests are still compared, but every shadow response is
    /// recorded as an explicit error stating that no invoker is configured.
    pub fn new(config: ShadowConfig) -> Self {
        Self::build(config, None)
    }

    /// Create a shadow testing service that invokes real shadow models.
    pub fn with_invoker(config: ShadowConfig, invoker: Arc<dyn ShadowModelInvoker>) -> Self {
        Self::build(config, Some(invoker))
    }

    fn build(config: ShadowConfig, invoker: Option<Arc<dyn ShadowModelInvoker>>) -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        let (request_sender, request_receiver) = mpsc::unbounded_channel();

        let service = Self {
            config,
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            shadow_results: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(RwLock::new(ShadowStats::default())),
            event_sender,
            request_sender,
            invoker,
            task_handles: Arc::new(Mutex::new(Vec::new())),
        };

        // Start background processing
        service.start_background_processing(request_receiver);

        service
    }

    /// Whether a real shadow model invoker is installed.
    pub fn has_invoker(&self) -> bool {
        self.invoker.is_some()
    }

    /// Start shadow testing service
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            tracing::info!("Shadow testing is disabled");
            return Ok(());
        }

        tracing::info!("Starting shadow testing service");
        Ok(())
    }

    /// Stop shadow testing service
    pub async fn stop(&self) -> Result<()> {
        let mut handles = self.task_handles.lock().await;
        for handle in handles.drain(..) {
            handle.abort();
        }

        tracing::info!("Shadow testing service stopped");
        Ok(())
    }

    /// Mirror a request to the configured shadow models.
    ///
    /// `production_payload` and `production_time_ms` describe the response the
    /// production path actually returned; they become the baseline the shadow
    /// responses are compared against, so no side of the comparison is invented.
    ///
    /// Returns the request id that was mirrored, or `None` when shadow testing
    /// is disabled or the request was not sampled.
    pub async fn process_request(
        &self,
        payload: serde_json::Value,
        production_payload: serde_json::Value,
        production_time_ms: f64,
        client_info: Option<ClientInfo>,
        metadata: HashMap<String, String>,
    ) -> Result<Option<String>> {
        if !self.config.enabled {
            return Ok(None);
        }

        // Check if this request should be shadowed
        if !self.should_shadow_request() {
            return Ok(None);
        }

        let request_id = Uuid::new_v4().to_string();
        let shadow_request = ShadowRequest {
            request_id: request_id.clone(),
            payload,
            timestamp: chrono::Utc::now(),
            client_info,
            metadata,
        };

        let production_response = ShadowResponse {
            response_id: Uuid::new_v4().to_string(),
            request_id: request_id.clone(),
            model_name: "production".to_string(),
            model_version: self.config.production_model_version.clone(),
            payload: production_payload,
            processing_time_ms: production_time_ms,
            timestamp: chrono::Utc::now(),
            status: ShadowResponseStatus::Success,
            error: None,
            metrics: HashMap::new(),
        };

        // Store active request
        {
            let mut active_requests =
                self.active_requests.write().unwrap_or_else(|p| p.into_inner());
            active_requests.insert(request_id.clone(), shadow_request.clone());
        }

        // Send for shadow processing
        if let Err(e) = self.request_sender.send((shadow_request, production_response)) {
            tracing::error!("Failed to send shadow request: {}", e);
            return Err(anyhow::anyhow!(
                "shadow pipeline is not accepting requests: {}",
                e
            ));
        }

        Ok(Some(request_id))
    }

    /// Mirror a request and wait until its comparison is recorded.
    ///
    /// Returns `None` when the request was not sampled, and an error when the
    /// comparison does not appear within `timeout_duration`.
    pub async fn process_request_blocking(
        &self,
        payload: serde_json::Value,
        production_payload: serde_json::Value,
        production_time_ms: f64,
        timeout_duration: Duration,
    ) -> Result<Option<ShadowComparison>> {
        let Some(request_id) = self
            .process_request(
                payload,
                production_payload,
                production_time_ms,
                None,
                HashMap::new(),
            )
            .await?
        else {
            return Ok(None);
        };

        let deadline = Instant::now() + timeout_duration;
        loop {
            if let Some(comparison) = self
                .shadow_results
                .lock()
                .await
                .iter()
                .rev()
                .find(|c| c.request_id == request_id)
                .cloned()
            {
                return Ok(Some(comparison));
            }
            if Instant::now() >= deadline {
                return Err(anyhow::anyhow!(
                    "shadow comparison for request {} did not complete within {:?}",
                    request_id,
                    timeout_duration
                ));
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Get shadow testing statistics
    pub async fn get_stats(&self) -> ShadowStats {
        let stats = self.stats.read().unwrap_or_else(|p| p.into_inner());
        stats.clone()
    }

    /// Get shadow results
    pub async fn get_shadow_results(&self, limit: Option<usize>) -> Vec<ShadowComparison> {
        let results = self.shadow_results.lock().await;
        let limit = limit.unwrap_or(results.len());
        results.iter().rev().take(limit).cloned().collect()
    }

    /// Get comparison by ID
    pub async fn get_comparison(&self, comparison_id: &str) -> Option<ShadowComparison> {
        let results = self.shadow_results.lock().await;
        results.iter().find(|c| c.comparison_id == comparison_id).cloned()
    }

    /// Subscribe to shadow events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<ShadowEvent> {
        self.event_sender.subscribe()
    }

    /// Check if a request should be shadowed
    fn should_shadow_request(&self) -> bool {
        use scirs2_core::random::*;
        let mut rng = thread_rng();
        let random_value: f64 = rng.gen_range(0.0..100.0);
        random_value < self.config.traffic_percentage
    }

    /// Start background processing
    fn start_background_processing(
        &self,
        mut request_receiver: mpsc::UnboundedReceiver<(ShadowRequest, ShadowResponse)>,
    ) {
        let service = self.clone();
        let service_for_handles = self.clone();

        let handle = tokio::spawn(async move {
            while let Some((request, production_response)) = request_receiver.recv().await {
                service.process_shadow_request(request, production_response).await;
            }
        });

        tokio::spawn(async move {
            let mut handles = service_for_handles.task_handles.lock().await;
            handles.push(handle);
        });
    }

    /// Process a shadow request against the real production response.
    async fn process_shadow_request(
        &self,
        request: ShadowRequest,
        production_response: ShadowResponse,
    ) {
        let request_id = request.request_id.clone();

        // Send start event
        let _ = self.event_sender.send(ShadowEvent::RequestStarted {
            request_id: request_id.clone(),
            model_name: "shadow".to_string(),
            timestamp: chrono::Utc::now(),
        });

        // Process shadow requests for all configured models
        let mut shadow_responses = Vec::new();

        for (model_name, model_config) in &self.config.shadow_models {
            if !model_config.enabled {
                continue;
            }

            let start_time = Instant::now();

            // Run the real shadow model.
            let response = match timeout(
                Duration::from_secs(self.config.shadow_timeout_seconds),
                self.invoke_shadow_model(model_name, model_config, &request),
            )
            .await
            {
                Ok(Ok(response)) => response,
                Ok(Err(e)) => {
                    tracing::error!("Shadow model {} failed: {}", model_name, e);
                    ShadowResponse {
                        response_id: Uuid::new_v4().to_string(),
                        request_id: request_id.clone(),
                        model_name: model_name.clone(),
                        model_version: model_config.version.clone(),
                        payload: serde_json::Value::Null,
                        processing_time_ms: start_time.elapsed().as_millis() as f64,
                        timestamp: chrono::Utc::now(),
                        status: ShadowResponseStatus::Error,
                        error: Some(e.to_string()),
                        metrics: HashMap::new(),
                    }
                },
                Err(_) => {
                    tracing::warn!("Shadow model {} timed out", model_name);
                    ShadowResponse {
                        response_id: Uuid::new_v4().to_string(),
                        request_id: request_id.clone(),
                        model_name: model_name.clone(),
                        model_version: model_config.version.clone(),
                        payload: serde_json::Value::Null,
                        processing_time_ms: start_time.elapsed().as_millis() as f64,
                        timestamp: chrono::Utc::now(),
                        status: ShadowResponseStatus::Timeout,
                        error: Some("Request timed out".to_string()),
                        metrics: HashMap::new(),
                    }
                },
            };

            // Send completion event
            let _ = self.event_sender.send(ShadowEvent::RequestCompleted {
                request_id: request_id.clone(),
                model_name: model_name.clone(),
                status: response.status.clone(),
                processing_time_ms: response.processing_time_ms,
                timestamp: chrono::Utc::now(),
            });

            shadow_responses.push(response);
        }

        // The production response is the real one recorded by the caller.
        // Compare responses
        let comparison = self.compare_responses(&production_response, &shadow_responses);

        // Store results
        {
            let mut results = self.shadow_results.lock().await;
            results.push(comparison.clone());

            // Keep only the latest results
            if results.len() > self.config.max_shadow_results {
                results.remove(0);
            }
        }

        // Update statistics
        self.update_statistics(&comparison).await;

        // Send comparison event
        let _ = self.event_sender.send(ShadowEvent::ComparisonCompleted {
            comparison_id: comparison.comparison_id.clone(),
            request_id: request_id.clone(),
            similarity_score: comparison.comparison_metrics.similarity_score,
            timestamp: chrono::Utc::now(),
        });

        // Remove from active requests
        {
            let mut active_requests =
                self.active_requests.write().unwrap_or_else(|p| p.into_inner());
            active_requests.remove(&request_id);
        }
    }

    /// Invoke a configured shadow model for a mirrored request.
    ///
    /// # Errors
    ///
    /// Returns an error when no [`ShadowModelInvoker`] is installed, or when the
    /// invoker itself fails. No payload is ever synthesized.
    async fn invoke_shadow_model(
        &self,
        model_name: &str,
        model_config: &ShadowModelConfig,
        request: &ShadowRequest,
    ) -> Result<ShadowResponse> {
        let invoker = self.invoker.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "no shadow model invoker is configured; build the service with \
                 ShadowTestingService::with_invoker to mirror traffic to a real model"
            )
        })?;

        let started = Instant::now();
        let payload = invoker.invoke(model_name, model_config, request).await?;

        Ok(ShadowResponse {
            response_id: Uuid::new_v4().to_string(),
            request_id: request.request_id.clone(),
            model_name: model_name.to_string(),
            model_version: model_config.version.clone(),
            payload,
            processing_time_ms: started.elapsed().as_secs_f64() * 1000.0,
            timestamp: chrono::Utc::now(),
            status: ShadowResponseStatus::Success,
            error: None,
            metrics: HashMap::new(),
        })
    }

    /// Compare responses
    fn compare_responses(
        &self,
        production_response: &ShadowResponse,
        shadow_responses: &[ShadowResponse],
    ) -> ShadowComparison {
        let comparison_id = Uuid::new_v4().to_string();

        // Calculate comparison metrics
        let mut similarity_scores = Vec::new();
        let mut latency_differences = Vec::new();

        for shadow_response in shadow_responses {
            // Simple similarity calculation (placeholder)
            let similarity =
                self.calculate_similarity(&production_response.payload, &shadow_response.payload);
            similarity_scores.push(similarity);

            let latency_diff =
                shadow_response.processing_time_ms - production_response.processing_time_ms;
            latency_differences.push(latency_diff);
        }

        let avg_similarity = if similarity_scores.is_empty() {
            0.0
        } else {
            similarity_scores.iter().sum::<f64>() / similarity_scores.len() as f64
        };

        let avg_latency_diff = if latency_differences.is_empty() {
            0.0
        } else {
            latency_differences.iter().sum::<f64>() / latency_differences.len() as f64
        };

        // Length and content differences are measured against the first shadow
        // response, which is the one a single-shadow deployment has; reporting
        // a hardcoded zero here previously made every comparison look
        // byte-identical in length regardless of what came back.
        let (response_length_difference, content_differences) = match shadow_responses.first() {
            Some(first) => {
                let production_len = production_response.payload.to_string().len() as i64;
                let shadow_len = first.payload.to_string().len() as i64;
                (
                    shadow_len - production_len,
                    describe_differences(&production_response.payload, &first.payload),
                )
            },
            None => (0, Vec::new()),
        };

        let comparison_metrics = ComparisonMetrics {
            similarity_score: avg_similarity,
            latency_difference_ms: avg_latency_diff,
            response_length_difference,
            content_differences,
            token_differences: None,
            custom_metrics: HashMap::new(),
        };

        ShadowComparison {
            comparison_id,
            request_id: production_response.request_id.clone(),
            production_response: production_response.clone(),
            shadow_responses: shadow_responses.to_vec(),
            comparison_metrics,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Similarity between a production and a shadow response payload, in `[0, 1]`.
    ///
    /// This is a real structural comparison of the two JSON documents, not a
    /// flag with a constant attached. Previously any pair of differing payloads
    /// scored a fixed `0.8`, so a shadow model returning complete nonsense was
    /// indistinguishable from one that differed by a single token — and that
    /// figure was then averaged into [`ShadowStats::avg_similarity_score`] and
    /// reported over HTTP as a measurement.
    ///
    /// The score is computed recursively over the JSON structure:
    ///
    /// * two values of different kinds (object vs array vs string …) score `0`;
    /// * strings score by token overlap, see [`Self::string_similarity`];
    /// * numbers score by relative difference, so `100.0` vs `101.0` is close
    ///   to `1.0` while `1.0` vs `1000.0` is near `0`;
    /// * booleans and nulls score `1` when equal and `0` otherwise;
    /// * arrays score as the mean similarity of the positions they share,
    ///   scaled by the fraction of positions that both actually have;
    /// * objects score as the mean similarity over the union of their keys, so
    ///   a key present in only one side counts as a `0` rather than being
    ///   quietly skipped.
    fn calculate_similarity(
        &self,
        response1: &serde_json::Value,
        response2: &serde_json::Value,
    ) -> f64 {
        json_similarity(response1, response2)
    }

    /// Update statistics
    async fn update_statistics(&self, comparison: &ShadowComparison) {
        let mut stats = self.stats.write().unwrap_or_else(|p| p.into_inner());
        stats.total_requests += 1;
        stats.total_shadow_requests += comparison.shadow_responses.len() as u64;

        // Update average similarity
        let total_comparisons = stats.total_requests as f64;
        stats.avg_similarity_score = (stats.avg_similarity_score * (total_comparisons - 1.0)
            + comparison.comparison_metrics.similarity_score)
            / total_comparisons;

        // Update model-specific stats
        for response in &comparison.shadow_responses {
            let model_stats =
                stats.model_stats.entry(response.model_name.clone()).or_insert_with(|| {
                    ModelShadowStats {
                        model_name: response.model_name.clone(),
                        total_requests: 0,
                        success_requests: 0,
                        error_requests: 0,
                        timeout_requests: 0,
                        avg_latency_ms: 0.0,
                        avg_similarity_score: 0.0,
                    }
                });

            model_stats.total_requests += 1;

            match response.status {
                ShadowResponseStatus::Success => model_stats.success_requests += 1,
                ShadowResponseStatus::Error => model_stats.error_requests += 1,
                ShadowResponseStatus::Timeout => model_stats.timeout_requests += 1,
                ShadowResponseStatus::Cancelled => {},
            }

            // Update average latency
            let total_requests = model_stats.total_requests as f64;
            model_stats.avg_latency_ms = (model_stats.avg_latency_ms * (total_requests - 1.0)
                + response.processing_time_ms)
                / total_requests;
        }
    }
}

/// Name the places where two JSON payloads actually differ.
///
/// Returns JSON-pointer-style paths (`/choices/0/text`) paired with a short
/// description of the disagreement, so an operator reading a shadow comparison
/// can see *what* diverged rather than only that something did. The list is
/// capped at [`MAX_REPORTED_DIFFERENCES`] entries — a wholly different response
/// would otherwise produce a path per leaf — and the cap is stated in the
/// output rather than silently truncating.
pub fn describe_differences(
    production: &serde_json::Value,
    shadow: &serde_json::Value,
) -> Vec<String> {
    let mut out = Vec::new();
    collect_differences("", production, shadow, &mut out);
    if out.len() > MAX_REPORTED_DIFFERENCES {
        let hidden = out.len() - MAX_REPORTED_DIFFERENCES;
        out.truncate(MAX_REPORTED_DIFFERENCES);
        out.push(format!("… and {hidden} further difference(s) not listed"));
    }
    out
}

/// Upper bound on the number of individual differences reported by
/// [`describe_differences`] before the rest are summarised as a count.
pub const MAX_REPORTED_DIFFERENCES: usize = 32;

/// Recursive worker behind [`describe_differences`].
fn collect_differences(
    path: &str,
    production: &serde_json::Value,
    shadow: &serde_json::Value,
    out: &mut Vec<String>,
) {
    use serde_json::Value;

    // Stop descending once the cap is exceeded; the caller summarises the rest.
    if out.len() > MAX_REPORTED_DIFFERENCES {
        return;
    }
    let here = if path.is_empty() { "/" } else { path };

    match (production, shadow) {
        (Value::Object(p), Value::Object(s)) => {
            let mut keys: Vec<&String> = p.keys().collect();
            keys.extend(s.keys());
            keys.sort_unstable();
            keys.dedup();
            for key in keys {
                let child = format!("{path}/{key}");
                match (p.get(key), s.get(key)) {
                    (Some(pv), Some(sv)) => collect_differences(&child, pv, sv, out),
                    (Some(_), None) => out.push(format!("{child}: missing from shadow response")),
                    (None, Some(_)) => {
                        out.push(format!("{child}: present only in shadow response"))
                    },
                    (None, None) => {},
                }
            }
        },
        (Value::Array(p), Value::Array(s)) => {
            if p.len() != s.len() {
                out.push(format!(
                    "{here}: length {} in production, {} in shadow",
                    p.len(),
                    s.len()
                ));
            }
            for (index, (pv, sv)) in p.iter().zip(s.iter()).enumerate() {
                collect_differences(&format!("{path}/{index}"), pv, sv, out);
            }
        },
        (p, s) if p == s => {},
        (p, s) => {
            out.push(format!(
                "{here}: {} vs {}",
                truncate_for_report(&p.to_string()),
                truncate_for_report(&s.to_string())
            ));
        },
    }
}

/// Shorten a rendered JSON value so one long string cannot dominate a report.
fn truncate_for_report(rendered: &str) -> String {
    const LIMIT: usize = 60;
    if rendered.chars().count() <= LIMIT {
        return rendered.to_string();
    }
    let head: String = rendered.chars().take(LIMIT).collect();
    format!("{head}…")
}

/// Structural similarity of two JSON values, in `[0, 1]`.
///
/// See [`ShadowTestingService::calculate_similarity`] for the rules. Kept as a
/// free function so it can be exercised directly, without standing up a shadow
/// testing service.
pub fn json_similarity(a: &serde_json::Value, b: &serde_json::Value) -> f64 {
    use serde_json::Value;

    match (a, b) {
        (Value::Null, Value::Null) => 1.0,
        (Value::Bool(x), Value::Bool(y)) => {
            if x == y {
                1.0
            } else {
                0.0
            }
        },
        (Value::Number(x), Value::Number(y)) => number_similarity(x, y),
        (Value::String(x), Value::String(y)) => string_similarity(x, y),
        (Value::Array(x), Value::Array(y)) => {
            if x.is_empty() && y.is_empty() {
                return 1.0;
            }
            let longest = x.len().max(y.len());
            if longest == 0 {
                return 1.0;
            }
            // Positions present in only one array contribute zero: a truncated
            // response is genuinely less similar, not merely shorter.
            let total: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| json_similarity(xi, yi)).sum();
            total / longest as f64
        },
        (Value::Object(x), Value::Object(y)) => {
            let mut keys: Vec<&String> = x.keys().collect();
            keys.extend(y.keys());
            keys.sort_unstable();
            keys.dedup();
            if keys.is_empty() {
                return 1.0;
            }
            let total: f64 = keys
                .iter()
                .map(|key| match (x.get(*key), y.get(*key)) {
                    (Some(xv), Some(yv)) => json_similarity(xv, yv),
                    // A key on one side only is a real difference.
                    _ => 0.0,
                })
                .sum();
            total / keys.len() as f64
        },
        // Different JSON kinds are not comparable on any scale that would mean
        // anything; they are simply different.
        _ => 0.0,
    }
}

/// Similarity of two JSON numbers by relative difference.
///
/// Equal values score `1.0`. Otherwise the score falls off with the difference
/// relative to the larger magnitude, so nearby values stay close to `1.0` and
/// values differing by orders of magnitude approach `0.0`.
fn number_similarity(x: &serde_json::Number, y: &serde_json::Number) -> f64 {
    let (Some(xf), Some(yf)) = (x.as_f64(), y.as_f64()) else {
        // Numbers outside f64 (e.g. u128-scale integers) can still be compared
        // exactly for equality, which is the only honest answer available.
        return if x == y { 1.0 } else { 0.0 };
    };
    if !xf.is_finite() || !yf.is_finite() {
        return if xf == yf { 1.0 } else { 0.0 };
    }
    if (xf - yf).abs() < f64::EPSILON {
        return 1.0;
    }
    let scale = xf.abs().max(yf.abs());
    if scale < f64::EPSILON {
        return 1.0;
    }
    (1.0 - (xf - yf).abs() / scale).clamp(0.0, 1.0)
}

/// Similarity of two strings by whitespace-token overlap (Jaccard index).
///
/// Token overlap is the right granularity for the generated text these shadow
/// comparisons carry: it is insensitive to token order, which two samplings of
/// the same model legitimately differ in, while still separating "almost the
/// same answer" from "a different answer".
fn string_similarity(x: &str, y: &str) -> f64 {
    if x == y {
        return 1.0;
    }
    let left: std::collections::BTreeSet<&str> = x.split_whitespace().collect();
    let right: std::collections::BTreeSet<&str> = y.split_whitespace().collect();
    if left.is_empty() && right.is_empty() {
        // Two different strings of pure whitespace: not equal, but they carry
        // no tokens to disagree about.
        return 1.0;
    }
    let intersection = left.intersection(&right).count() as f64;
    let union = left.union(&right).count() as f64;
    if union < f64::EPSILON {
        return 0.0;
    }
    intersection / union
}

impl Clone for ShadowTestingService {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            active_requests: Arc::clone(&self.active_requests),
            shadow_results: Arc::clone(&self.shadow_results),
            stats: Arc::clone(&self.stats),
            event_sender: self.event_sender.clone(),
            request_sender: self.request_sender.clone(),
            invoker: self.invoker.clone(),
            task_handles: Arc::clone(&self.task_handles),
        }
    }
}

impl Default for ShadowStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_shadow_requests: 0,
            shadow_success_rate: 0.0,
            avg_shadow_latency_ms: 0.0,
            avg_similarity_score: 0.0,
            model_stats: HashMap::new(),
            error_stats: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_config_default() {
        let config = ShadowConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.traffic_percentage, 10.0);
        assert_eq!(config.shadow_timeout_seconds, 30);
    }

    #[tokio::test]
    async fn test_shadow_service_creation() {
        let config = ShadowConfig::default();
        let service = ShadowTestingService::new(config);

        let stats = service.get_stats().await;
        assert_eq!(stats.total_requests, 0);
    }

    fn shadow_config_with_one_model() -> ShadowConfig {
        let mut config = ShadowConfig::default();
        config.enabled = true;
        config.traffic_percentage = 100.0; // Always shadow
        config.shadow_models.insert(
            "candidate".to_string(),
            ShadowModelConfig {
                model_name: "candidate".to_string(),
                version: "2.0.0".to_string(),
                endpoint: None,
                parameters: HashMap::new(),
                enabled: true,
                traffic_percentage: None,
            },
        );
        config
    }

    /// An invoker that performs a real, deterministic transformation of the
    /// mirrored payload, so the comparison is between two genuinely different
    /// computed values.
    #[derive(Debug)]
    struct UppercasingInvoker;

    #[async_trait::async_trait]
    impl ShadowModelInvoker for UppercasingInvoker {
        async fn invoke(
            &self,
            model_name: &str,
            _model_config: &ShadowModelConfig,
            request: &ShadowRequest,
        ) -> Result<serde_json::Value> {
            let text = request
                .payload
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("no text to shadow"))?;
            Ok(serde_json::json!({ "text": text.to_uppercase(), "model": model_name }))
        }
    }

    #[tokio::test]
    async fn test_shadow_request_processing() {
        let service = ShadowTestingService::with_invoker(
            shadow_config_with_one_model(),
            Arc::new(UppercasingInvoker),
        );

        let payload = serde_json::json!({"text": "test input"});
        let client_info = Some(ClientInfo {
            client_id: Some("test-client".to_string()),
            ip_address: Some("127.0.0.1".to_string()),
            user_agent: Some("test-agent".to_string()),
            headers: HashMap::new(),
        });

        service
            .process_request(
                payload,
                serde_json::json!({"text": "test output"}),
                12.5,
                client_info,
                HashMap::new(),
            )
            .await
            .expect("async operation should succeed in test");

        // Wait a bit for processing
        tokio::time::sleep(Duration::from_millis(200)).await;

        let stats = service.get_stats().await;
        assert_eq!(stats.total_requests, 1);
    }

    /// Regression: both sides of a comparison must be real. The production side
    /// used to be the literal `"Production response"` and the shadow side
    /// `"Shadow response from {model}"`.
    #[tokio::test]
    async fn comparison_uses_real_responses_on_both_sides() {
        let service = ShadowTestingService::with_invoker(
            shadow_config_with_one_model(),
            Arc::new(UppercasingInvoker),
        );

        let comparison = service
            .process_request_blocking(
                serde_json::json!({"text": "hello shadow"}),
                serde_json::json!({"text": "hello production"}),
                42.0,
                Duration::from_secs(5),
            )
            .await
            .expect("mirroring must succeed")
            .expect("the request must be sampled at 100%");

        // Production side is exactly what the caller recorded.
        assert_eq!(
            comparison.production_response.payload["text"],
            serde_json::json!("hello production")
        );
        assert!((comparison.production_response.processing_time_ms - 42.0).abs() < 1e-9);
        assert_ne!(
            comparison.production_response.payload["text"],
            serde_json::json!("Production response")
        );

        // Shadow side is what the invoker computed.
        let shadow = comparison.shadow_responses.first().expect("one shadow response");
        assert_eq!(shadow.payload["text"], serde_json::json!("HELLO SHADOW"));
        assert!(!shadow.payload["text"]
            .as_str()
            .unwrap_or_default()
            .starts_with("Shadow response from"));
        assert_eq!(shadow.model_version, "2.0.0");
    }

    /// Regression: with no invoker the shadow side must record an error rather
    /// than a synthesized payload.
    #[tokio::test]
    async fn missing_invoker_records_an_error_not_a_payload() {
        let service = ShadowTestingService::new(shadow_config_with_one_model());
        assert!(!service.has_invoker());

        let comparison = service
            .process_request_blocking(
                serde_json::json!({"text": "hello"}),
                serde_json::json!({"text": "produced"}),
                10.0,
                Duration::from_secs(5),
            )
            .await
            .expect("mirroring must succeed")
            .expect("the request must be sampled at 100%");

        let shadow = comparison.shadow_responses.first().expect("one shadow response");
        assert!(matches!(shadow.status, ShadowResponseStatus::Error));
        assert!(shadow.error.as_deref().unwrap_or_default().contains("no shadow model invoker"));
        assert_eq!(shadow.payload, serde_json::Value::Null);
    }

    /// Regression: a disabled service must report "not sampled", never a
    /// fabricated comparison.
    #[tokio::test]
    async fn disabled_service_does_not_compare() {
        let service = ShadowTestingService::new(ShadowConfig::default());
        let outcome = service
            .process_request_blocking(
                serde_json::json!({"text": "hello"}),
                serde_json::json!({"text": "produced"}),
                10.0,
                Duration::from_millis(200),
            )
            .await
            .expect("call must succeed");
        assert!(outcome.is_none());
    }

    // ── Regression tests: similarity used to be a constant ──

    /// Regression: `calculate_similarity` returned a flat `0.8` for *every*
    /// pair of differing payloads, and that number was averaged into
    /// `ShadowStats::avg_similarity_score` and served over
    /// `/shadow/stats`. A shadow model returning something unrelated must now
    /// score far below one returning nearly the same answer.
    #[test]
    fn similarity_discriminates_between_near_and_unrelated_responses() {
        let production = serde_json::json!({"text": "the quick brown fox jumps"});
        let near = serde_json::json!({"text": "the quick brown fox leaps"});
        let unrelated = serde_json::json!({"text": "unrelated content entirely here"});

        let near_score = json_similarity(&production, &near);
        let unrelated_score = json_similarity(&production, &unrelated);

        assert!(
            near_score > unrelated_score,
            "a near-identical response must score higher: {near_score} vs {unrelated_score}"
        );
        // The old code gave both of these exactly 0.8.
        assert_ne!(near_score, 0.8);
        assert_ne!(unrelated_score, 0.8);
        assert_eq!(unrelated_score, 0.0, "no shared tokens means no similarity");
        assert!((0.0..=1.0).contains(&near_score));
    }

    #[test]
    fn identical_payloads_score_one_and_different_kinds_score_zero() {
        let value = serde_json::json!({"a": [1, 2, {"b": "c"}], "d": null});
        assert_eq!(json_similarity(&value, &value.clone()), 1.0);

        assert_eq!(
            json_similarity(&serde_json::json!("1"), &serde_json::json!(1)),
            0.0,
            "a string and a number are not comparable"
        );
        assert_eq!(
            json_similarity(&serde_json::json!([]), &serde_json::json!({})),
            0.0
        );
    }

    #[test]
    fn numbers_score_by_relative_difference() {
        let close = json_similarity(&serde_json::json!(100.0), &serde_json::json!(101.0));
        let far = json_similarity(&serde_json::json!(1.0), &serde_json::json!(1000.0));
        assert!(
            close > 0.98,
            "100 vs 101 should be nearly identical: {close}"
        );
        assert!(far < 0.01, "1 vs 1000 should be nearly unrelated: {far}");
    }

    #[test]
    fn a_missing_object_key_lowers_the_score() {
        let full = serde_json::json!({"a": 1, "b": 2});
        let partial = serde_json::json!({"a": 1});
        let score = json_similarity(&full, &partial);
        assert!(
            score > 0.0 && score < 1.0,
            "a half-present object is neither identical nor unrelated: {score}"
        );
        assert!((score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_truncated_array_lowers_the_score() {
        let full = serde_json::json!([1, 2, 3, 4]);
        let truncated = serde_json::json!([1, 2]);
        let score = json_similarity(&full, &truncated);
        assert!(
            (score - 0.5).abs() < 1e-9,
            "half the positions match: {score}"
        );
    }

    /// Regression: `content_differences` was hardcoded to an empty vector and
    /// `response_length_difference` to `0`, so a comparison never said what
    /// diverged. Both must now be measured.
    #[test]
    fn differences_name_the_paths_that_actually_diverged() {
        let production = serde_json::json!({"text": "hello", "tokens": 5, "model": "a"});
        let shadow = serde_json::json!({"text": "goodbye", "tokens": 5, "extra": true});

        let differences = describe_differences(&production, &shadow);
        let joined = differences.join("\n");

        assert!(
            !differences.is_empty(),
            "the old code always returned an empty list"
        );
        assert!(
            joined.contains("/text"),
            "the diverging field must be named: {joined}"
        );
        assert!(
            joined.contains("/model") && joined.contains("missing from shadow"),
            "a dropped field must be reported: {joined}"
        );
        assert!(
            joined.contains("/extra") && joined.contains("only in shadow"),
            "an added field must be reported: {joined}"
        );
        assert!(
            !joined.contains("/tokens"),
            "an identical field must not be reported as a difference: {joined}"
        );
    }

    #[test]
    fn identical_payloads_have_no_differences() {
        let value = serde_json::json!({"text": "same", "n": [1, 2]});
        assert!(describe_differences(&value, &value.clone()).is_empty());
    }

    #[test]
    fn difference_reports_are_capped_and_say_so() {
        let production: serde_json::Value = (0..100)
            .map(|i| (format!("k{i}"), serde_json::json!(i)))
            .collect::<serde_json::Map<_, _>>()
            .into();
        let shadow: serde_json::Value = (0..100)
            .map(|i| (format!("k{i}"), serde_json::json!(i + 1)))
            .collect::<serde_json::Map<_, _>>()
            .into();

        let differences = describe_differences(&production, &shadow);
        assert!(differences.len() <= MAX_REPORTED_DIFFERENCES + 1);
        let last = differences.last().expect("at least one difference");
        assert!(
            last.contains("further difference(s) not listed"),
            "truncation must be stated, not silent: {last}"
        );
    }
}
