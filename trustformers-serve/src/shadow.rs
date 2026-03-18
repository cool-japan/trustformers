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

    /// Request sender for shadow processing
    request_sender: mpsc::UnboundedSender<ShadowRequest>,

    /// Background task handles
    task_handles: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
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
    /// Create a new shadow testing service
    pub fn new(config: ShadowConfig) -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        let (request_sender, request_receiver) = mpsc::unbounded_channel();

        let service = Self {
            config,
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            shadow_results: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(RwLock::new(ShadowStats::default())),
            event_sender,
            request_sender,
            task_handles: Arc::new(Mutex::new(Vec::new())),
        };

        // Start background processing
        service.start_background_processing(request_receiver);

        service
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

    /// Process a request with shadow testing
    pub async fn process_request(
        &self,
        payload: serde_json::Value,
        client_info: Option<ClientInfo>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check if this request should be shadowed
        if !self.should_shadow_request() {
            return Ok(());
        }

        let request_id = Uuid::new_v4().to_string();
        let shadow_request = ShadowRequest {
            request_id: request_id.clone(),
            payload,
            timestamp: chrono::Utc::now(),
            client_info,
            metadata,
        };

        // Store active request
        {
            let mut active_requests =
                self.active_requests.write().expect("lock should not be poisoned");
            active_requests.insert(request_id.clone(), shadow_request.clone());
        }

        // Send for shadow processing
        if let Err(e) = self.request_sender.send(shadow_request) {
            tracing::error!("Failed to send shadow request: {}", e);
        }

        Ok(())
    }

    /// Get shadow testing statistics
    pub async fn get_stats(&self) -> ShadowStats {
        let stats = self.stats.read().expect("lock should not be poisoned");
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
        mut request_receiver: mpsc::UnboundedReceiver<ShadowRequest>,
    ) {
        let service = self.clone();
        let service_for_handles = self.clone();

        let handle = tokio::spawn(async move {
            while let Some(request) = request_receiver.recv().await {
                service.process_shadow_request(request).await;
            }
        });

        tokio::spawn(async move {
            let mut handles = service_for_handles.task_handles.lock().await;
            handles.push(handle);
        });
    }

    /// Process a shadow request
    async fn process_shadow_request(&self, request: ShadowRequest) {
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

            // Simulate shadow model processing
            let response = match timeout(
                Duration::from_secs(self.config.shadow_timeout_seconds),
                self.simulate_model_processing(model_name, model_config, &request),
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

        // Create production response (simulated)
        let production_response = ShadowResponse {
            response_id: Uuid::new_v4().to_string(),
            request_id: request_id.clone(),
            model_name: "production".to_string(),
            model_version: "1.0.0".to_string(),
            payload: serde_json::json!({"text": "Production response", "tokens": ["production", "response"]}),
            processing_time_ms: 50.0,
            timestamp: chrono::Utc::now(),
            status: ShadowResponseStatus::Success,
            error: None,
            metrics: HashMap::new(),
        };

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
                self.active_requests.write().expect("lock should not be poisoned");
            active_requests.remove(&request_id);
        }
    }

    /// Simulate model processing (placeholder)
    async fn simulate_model_processing(
        &self,
        model_name: &str,
        _model_config: &ShadowModelConfig,
        request: &ShadowRequest,
    ) -> Result<ShadowResponse> {
        // Simulate processing delay
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Generate mock response
        let response = ShadowResponse {
            response_id: Uuid::new_v4().to_string(),
            request_id: request.request_id.clone(),
            model_name: model_name.to_string(),
            model_version: "1.0.0".to_string(),
            payload: serde_json::json!({
                "text": format!("Shadow response from {}", model_name),
                "tokens": ["shadow", "response", "from", model_name]
            }),
            processing_time_ms: 100.0,
            timestamp: chrono::Utc::now(),
            status: ShadowResponseStatus::Success,
            error: None,
            metrics: HashMap::new(),
        };

        Ok(response)
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

        let comparison_metrics = ComparisonMetrics {
            similarity_score: avg_similarity,
            latency_difference_ms: avg_latency_diff,
            response_length_difference: 0,
            content_differences: vec![],
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

    /// Calculate similarity between two responses
    fn calculate_similarity(
        &self,
        response1: &serde_json::Value,
        response2: &serde_json::Value,
    ) -> f64 {
        // Simple similarity calculation (placeholder)
        if response1 == response2 {
            1.0
        } else {
            0.8 // Mock similarity score
        }
    }

    /// Update statistics
    async fn update_statistics(&self, comparison: &ShadowComparison) {
        let mut stats = self.stats.write().expect("lock should not be poisoned");
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

impl Clone for ShadowTestingService {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            active_requests: Arc::clone(&self.active_requests),
            shadow_results: Arc::clone(&self.shadow_results),
            stats: Arc::clone(&self.stats),
            event_sender: self.event_sender.clone(),
            request_sender: self.request_sender.clone(),
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

    #[tokio::test]
    async fn test_shadow_request_processing() {
        let mut config = ShadowConfig::default();
        config.enabled = true;
        config.traffic_percentage = 100.0; // Always shadow

        let service = ShadowTestingService::new(config);

        let payload = serde_json::json!({"text": "test input"});
        let client_info = Some(ClientInfo {
            client_id: Some("test-client".to_string()),
            ip_address: Some("127.0.0.1".to_string()),
            user_agent: Some("test-agent".to_string()),
            headers: HashMap::new(),
        });

        service.process_request(payload, client_info, HashMap::new()).await.unwrap();

        // Wait a bit for processing
        tokio::time::sleep(Duration::from_millis(200)).await;

        let stats = service.get_stats().await;
        assert_eq!(stats.total_requests, 1);
    }
}
