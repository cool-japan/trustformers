//! Shared plumbing for the REST-based cloud providers.
//!
//! Holds the initialized [`ProviderConfig`], the shared [`reqwest::Client`] and
//! the counters every provider reports its health and metrics from. Those
//! counters are the *observed* behaviour of this process — no provider claims
//! an availability figure or a latency it did not measure.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use chrono::Utc;
use tokio::sync::RwLock;

use super::errors::CloudProviderError;
use super::{
    CostMetrics, HealthStatus, InputData, PerformanceMetrics, ProviderConfig, ProviderMetrics,
    ResponseMetadata,
};

/// Counters describing what this process observed for one provider.
#[derive(Debug, Default)]
pub struct ObservedStats {
    /// Requests attempted.
    pub requests: AtomicU64,
    /// Requests that returned a usable response.
    pub successes: AtomicU64,
    /// Requests that failed for any reason.
    pub failures: AtomicU64,
    /// Sum of measured round-trip latencies, in milliseconds.
    pub total_latency_ms: AtomicU64,
    /// Deployments this process created and has not deleted.
    pub active_deployments: AtomicU64,
    /// Unix seconds of the first observed request, used for a real rate.
    pub first_request_at: AtomicU64,
    /// Unix seconds of the most recent observed request.
    pub last_request_at: AtomicU64,
}

impl ObservedStats {
    /// Record one completed request.
    pub fn record(&self, success: bool, latency: Duration) {
        let now = Utc::now().timestamp().max(0) as u64;
        self.requests.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successes.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failures.fetch_add(1, Ordering::Relaxed);
        }
        self.total_latency_ms.fetch_add(latency.as_millis() as u64, Ordering::Relaxed);
        let _ =
            self.first_request_at
                .compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
        self.last_request_at.store(now, Ordering::Relaxed);
    }

    /// Mean latency over the observed requests, or 0 when nothing was observed.
    pub fn average_latency_ms(&self) -> u64 {
        let requests = self.requests.load(Ordering::Relaxed);
        self.total_latency_ms.load(Ordering::Relaxed).checked_div(requests).unwrap_or(0)
    }

    /// Observed success ratio in `[0, 1]`, or `1.0` when nothing was observed.
    pub fn success_ratio(&self) -> f64 {
        let requests = self.requests.load(Ordering::Relaxed);
        if requests == 0 {
            return 1.0;
        }
        self.successes.load(Ordering::Relaxed) as f64 / requests as f64
    }

    /// Observed error ratio in `[0, 1]`.
    pub fn error_ratio(&self) -> f64 {
        1.0 - self.success_ratio()
    }

    /// Requests per second measured over the observed window.
    pub fn requests_per_second(&self) -> f32 {
        let first = self.first_request_at.load(Ordering::Relaxed);
        let last = self.last_request_at.load(Ordering::Relaxed);
        let requests = self.requests.load(Ordering::Relaxed);
        if requests == 0 || last <= first {
            return 0.0;
        }
        requests as f32 / (last - first) as f32
    }
}

/// State shared by every REST-based provider.
#[derive(Debug)]
pub struct ProviderState {
    /// Human-readable provider name, used in error messages.
    pub name: &'static str,
    /// Configuration supplied by `initialize`, if it ran.
    pub config: RwLock<Option<ProviderConfig>>,
    /// Shared HTTP client.
    pub http: reqwest::Client,
    /// Observed request counters.
    pub stats: ObservedStats,
}

impl ProviderState {
    /// Build the state for a provider, with a client honouring `timeout`.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            config: RwLock::new(None),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .unwrap_or_default(),
            stats: ObservedStats::default(),
        }
    }

    /// The configuration, or [`CloudProviderError::NotInitialized`].
    pub async fn config(
        &self,
        operation: &'static str,
    ) -> Result<ProviderConfig, CloudProviderError> {
        self.config.read().await.clone().ok_or(CloudProviderError::NotInitialized {
            provider: self.name,
            operation,
        })
    }

    /// The API key, or [`CloudProviderError::MissingCredentials`].
    pub async fn api_key(&self, operation: &'static str) -> Result<String, CloudProviderError> {
        let config = self.config(operation).await?;
        config.credentials.api_key.clone().ok_or_else(|| {
            CloudProviderError::missing_credentials(
                self.name,
                operation,
                "set credentials.api_key in the ProviderConfig",
            )
        })
    }

    /// Base URL for the provider's API.
    ///
    /// `endpoints.inference_endpoint` is used when set; otherwise the caller's
    /// `default_base` applies. Tests point this at a local mock server.
    pub async fn base_url(
        &self,
        operation: &'static str,
        default_base: &str,
    ) -> Result<String, CloudProviderError> {
        let config = self.config(operation).await?;
        let base = if config.endpoints.inference_endpoint.trim().is_empty() {
            default_base.to_string()
        } else {
            config.endpoints.inference_endpoint.clone()
        };
        Ok(base.trim_end_matches('/').to_string())
    }

    /// Extra headers configured for this provider.
    pub async fn custom_headers(
        &self,
        operation: &'static str,
    ) -> Result<std::collections::HashMap<String, String>, CloudProviderError> {
        Ok(self.config(operation).await?.credentials.custom_headers.clone())
    }

    /// Health snapshot derived purely from observed counters.
    pub fn health(&self) -> HealthStatus {
        let requests = self.stats.requests.load(Ordering::Relaxed);
        let status = if requests == 0 {
            // Nothing has been observed, so nothing can be claimed.
            "unknown"
        } else if self.stats.error_ratio() == 0.0 {
            "healthy"
        } else if self.stats.success_ratio() > 0.0 {
            "degraded"
        } else {
            "unhealthy"
        };

        HealthStatus {
            provider: self.name.to_string(),
            status: status.to_string(),
            availability: self.stats.success_ratio(),
            last_check: Utc::now(),
            response_time_ms: self.stats.average_latency_ms(),
            error_rate: self.stats.error_ratio(),
            active_deployments: self.stats.active_deployments.load(Ordering::Relaxed) as u32,
            region_status: std::collections::HashMap::new(),
        }
    }

    /// Metrics snapshot derived purely from observed counters.
    pub fn metrics(&self) -> ProviderMetrics {
        ProviderMetrics {
            provider: self.name.to_string(),
            requests_per_second: self.stats.requests_per_second(),
            average_latency_ms: self.stats.average_latency_ms(),
            error_rate: self.stats.error_ratio(),
            // Cost per hour is a billing figure this process cannot observe.
            cost_per_hour: None,
            active_connections: 0,
            queue_depth: 0,
            throughput_tokens_per_second: 0.0,
            // Cloud inference APIs expose no host utilisation to their clients.
            resource_utilization: None,
        }
    }
}

/// Extract the text of a request, rejecting input kinds a text API cannot take.
pub fn require_text(
    provider: &'static str,
    operation: &'static str,
    input: &InputData,
) -> Result<String, CloudProviderError> {
    match input {
        InputData::Text(text) => Ok(text.clone()),
        InputData::Structured(value) => Ok(value.to_string()),
        InputData::Image(_) => Err(CloudProviderError::UnsupportedInput {
            provider,
            operation,
            input_kind: "image",
        }),
        InputData::Audio(_) => Err(CloudProviderError::UnsupportedInput {
            provider,
            operation,
            input_kind: "audio",
        }),
        InputData::Video(_) => Err(CloudProviderError::UnsupportedInput {
            provider,
            operation,
            input_kind: "video",
        }),
        InputData::Document(_) => Err(CloudProviderError::UnsupportedInput {
            provider,
            operation,
            input_kind: "document",
        }),
        InputData::Batch(_) => Err(CloudProviderError::UnsupportedInput {
            provider,
            operation,
            input_kind: "batch",
        }),
    }
}

/// Cost metrics for a response whose provider reported no billing data.
///
/// Every figure is zero and `reported_by_provider` is `false`, so a caller can
/// tell "this cost nothing" apart from "no cost information was returned".
pub fn unreported_cost() -> CostMetrics {
    CostMetrics {
        cost_usd: 0.0,
        input_cost_usd: 0.0,
        output_cost_usd: 0.0,
        compute_cost_usd: 0.0,
        storage_cost_usd: 0.0,
        network_cost_usd: 0.0,
        currency: "USD".to_string(),
        billing_period: "per_request".to_string(),
        reported_by_provider: false,
    }
}

/// Build response metadata from a real measurement.
pub fn metadata(
    latency: Duration,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    finish_reason: Option<String>,
    provider_metadata: std::collections::HashMap<String, serde_json::Value>,
) -> ResponseMetadata {
    ResponseMetadata {
        processing_time_ms: latency.as_millis() as u64,
        // The client cannot separate queue time from processing time, and no
        // provider reports it, so it is left unmeasured rather than invented.
        queue_time_ms: None,
        model_load_time_ms: None,
        input_tokens,
        output_tokens,
        finish_reason,
        // Text-generation APIs do not return a confidence score.
        confidence_score: None,
        provider_metadata,
    }
}

/// Build performance metrics from a real measurement.
pub fn performance(latency: Duration, output_tokens: Option<u32>) -> PerformanceMetrics {
    let seconds = latency.as_secs_f32();
    let throughput = match output_tokens {
        Some(tokens) if seconds > 0.0 => Some(tokens as f32 / seconds),
        _ => None,
    };
    PerformanceMetrics {
        latency_ms: latency.as_millis() as u64,
        throughput_tokens_per_second: throughput,
        memory_usage_mb: None,
        cpu_usage_percent: None,
        gpu_usage_percent: None,
        provider_metrics: std::collections::HashMap::new(),
    }
}

/// Turn a non-success HTTP response into an [`CloudProviderError::ApiError`].
pub async fn api_error(
    provider: &'static str,
    operation: &'static str,
    response: reqwest::Response,
) -> CloudProviderError {
    let status = response.status().as_u16();
    let body = response.text().await.unwrap_or_default();
    CloudProviderError::ApiError {
        provider,
        operation,
        status,
        body: body.chars().take(2_000).collect(),
    }
}
