//! Request Tracing for Observability
//!
//! Provides distributed tracing capabilities for tracking requests across
//! services and components, enabling comprehensive observability.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, RwLock};
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum TracingError {
    #[error("Failed to create trace: {0}")]
    TraceCreationError(String),

    #[error("Failed to serialize trace data: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Span not found: {0}")]
    SpanNotFound(String),

    #[error("Trace export error: {0}")]
    ExportError(String),

    #[error("Invalid trace context: {0}")]
    InvalidTraceContext(String),
}

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Enable distributed tracing
    pub enabled: bool,

    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,

    /// Enable span collection
    pub enable_span_collection: bool,

    /// Maximum number of spans to keep in memory
    pub max_spans_in_memory: usize,

    /// Export interval in seconds
    pub export_interval_seconds: u64,

    /// Trace retention period in hours
    pub trace_retention_hours: u64,

    /// Enable metrics collection
    pub enable_metrics: bool,

    /// Enable log correlation
    pub enable_log_correlation: bool,

    /// Service name
    pub service_name: String,

    /// Service version
    pub service_version: String,

    /// Environment
    pub environment: String,

    /// Export endpoints
    pub export_endpoints: Vec<String>,

    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sampling_rate: 1.0,
            enable_span_collection: true,
            max_spans_in_memory: 10000,
            export_interval_seconds: 30,
            trace_retention_hours: 24,
            enable_metrics: true,
            enable_log_correlation: true,
            service_name: "trustformers-serve".to_string(),
            service_version: "1.0.0".to_string(),
            environment: "development".to_string(),
            export_endpoints: Vec::new(),
            enable_performance_monitoring: true,
        }
    }
}

/// Trace context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Trace ID
    pub trace_id: String,

    /// Span ID
    pub span_id: String,

    /// Parent span ID
    pub parent_span_id: Option<String>,

    /// Trace flags
    pub trace_flags: u8,

    /// Trace state
    pub trace_state: Option<String>,

    /// Baggage
    pub baggage: HashMap<String, String>,
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceContext {
    pub fn new() -> Self {
        Self {
            trace_id: Uuid::new_v4().to_string(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: None,
            trace_flags: 1, // Sampled
            trace_state: None,
            baggage: HashMap::new(),
        }
    }

    pub fn with_parent(parent_span_id: String) -> Self {
        let mut context = Self::new();
        context.parent_span_id = Some(parent_span_id);
        context
    }

    pub fn child_span(&self) -> TraceContext {
        TraceContext {
            trace_id: self.trace_id.clone(),
            span_id: Uuid::new_v4().to_string(),
            parent_span_id: Some(self.span_id.clone()),
            trace_flags: self.trace_flags,
            trace_state: self.trace_state.clone(),
            baggage: self.baggage.clone(),
        }
    }

    pub fn to_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert("X-Trace-Id".to_string(), self.trace_id.clone());
        headers.insert("X-Span-Id".to_string(), self.span_id.clone());

        if let Some(parent_span_id) = &self.parent_span_id {
            headers.insert("X-Parent-Span-Id".to_string(), parent_span_id.clone());
        }

        headers.insert("X-Trace-Flags".to_string(), self.trace_flags.to_string());

        if let Some(trace_state) = &self.trace_state {
            headers.insert("X-Trace-State".to_string(), trace_state.clone());
        }

        headers
    }

    pub fn from_headers(headers: &HashMap<String, String>) -> Result<Self, TracingError> {
        let trace_id = headers
            .get("X-Trace-Id")
            .ok_or_else(|| TracingError::InvalidTraceContext("Missing trace ID".to_string()))?
            .clone();

        let span_id = headers
            .get("X-Span-Id")
            .ok_or_else(|| TracingError::InvalidTraceContext("Missing span ID".to_string()))?
            .clone();

        let parent_span_id = headers.get("X-Parent-Span-Id").cloned();

        let trace_flags = headers.get("X-Trace-Flags").and_then(|f| f.parse().ok()).unwrap_or(1);

        let trace_state = headers.get("X-Trace-State").cloned();

        Ok(TraceContext {
            trace_id,
            span_id,
            parent_span_id,
            trace_flags,
            trace_state,
            baggage: HashMap::new(),
        })
    }
}

/// Span data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Span ID
    pub span_id: String,

    /// Trace ID
    pub trace_id: String,

    /// Parent span ID
    pub parent_span_id: Option<String>,

    /// Operation name
    pub operation_name: String,

    /// Service name
    pub service_name: String,

    /// Start time
    pub start_time: chrono::DateTime<chrono::Utc>,

    /// End time
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,

    /// Duration in microseconds
    pub duration_us: Option<u64>,

    /// Span status
    pub status: SpanStatus,

    /// Tags
    pub tags: HashMap<String, String>,

    /// Logs
    pub logs: Vec<SpanLog>,

    /// References
    pub references: Vec<SpanReference>,

    /// Process information
    pub process: ProcessInfo,
}

impl Span {
    pub fn new(trace_context: &TraceContext, operation_name: String, service_name: String) -> Self {
        Self {
            span_id: trace_context.span_id.clone(),
            trace_id: trace_context.trace_id.clone(),
            parent_span_id: trace_context.parent_span_id.clone(),
            operation_name,
            service_name,
            start_time: chrono::Utc::now(),
            end_time: None,
            duration_us: None,
            status: SpanStatus::Ok,
            tags: HashMap::new(),
            logs: Vec::new(),
            references: Vec::new(),
            process: ProcessInfo::default(),
        }
    }

    pub fn set_tag(&mut self, key: String, value: String) {
        self.tags.insert(key, value);
    }

    pub fn set_status(&mut self, status: SpanStatus) {
        self.status = status;
    }

    pub fn add_log(&mut self, log: SpanLog) {
        self.logs.push(log);
    }

    pub fn finish(&mut self) {
        self.end_time = Some(chrono::Utc::now());
        if let Some(end_time) = self.end_time {
            let duration = end_time - self.start_time;
            self.duration_us = Some(duration.num_microseconds().unwrap_or(0) as u64);
        }
    }

    pub fn is_finished(&self) -> bool {
        self.end_time.is_some()
    }
}

/// Span status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanStatus {
    Ok,
    Cancelled,
    Unknown,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
}

/// Span log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLog {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Fields
    pub fields: HashMap<String, String>,
}

impl SpanLog {
    pub fn new(fields: HashMap<String, String>) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            fields,
        }
    }
}

/// Span reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanReference {
    /// Reference type
    pub ref_type: SpanReferenceType,

    /// Trace ID
    pub trace_id: String,

    /// Span ID
    pub span_id: String,
}

/// Span reference type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanReferenceType {
    ChildOf,
    FollowsFrom,
}

/// Process information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    /// Service name
    pub service_name: String,

    /// Tags
    pub tags: HashMap<String, String>,
}

impl Default for ProcessInfo {
    fn default() -> Self {
        Self {
            service_name: "trustformers-serve".to_string(),
            tags: HashMap::new(),
        }
    }
}

/// Zipkin span format for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZipkinSpan {
    pub id: String,
    pub trace_id: String,
    pub parent_id: Option<String>,
    pub name: String,
    pub timestamp: u64,
    pub duration: Option<u64>,
    pub tags: HashMap<String, String>,
    pub annotations: Vec<serde_json::Value>,
}

impl ZipkinSpan {
    pub fn from_span(span: &Span) -> Self {
        Self {
            id: span.span_id.clone(),
            trace_id: span.trace_id.clone(),
            parent_id: span.parent_span_id.clone(),
            name: span.operation_name.clone(),
            timestamp: span.start_time.timestamp_micros() as u64,
            duration: span.duration_us,
            tags: span.tags.clone(),
            annotations: Vec::new(),
        }
    }
}

/// OpenTelemetry export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenTelemetryExport {
    pub spans: Vec<serde_json::Value>,
    pub metadata: HashMap<String, String>,
}

impl OpenTelemetryExport {
    pub fn from_spans(spans: &[Arc<Span>]) -> Self {
        let span_data: Vec<serde_json::Value> = spans
            .iter()
            .map(|span| serde_json::to_value(&**span).unwrap_or_default())
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "opentelemetry".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());

        Self {
            spans: span_data,
            metadata,
        }
    }
}

/// Trace metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetrics {
    /// Total spans
    pub total_spans: u64,

    /// Active spans
    pub active_spans: u64,

    /// Completed spans
    pub completed_spans: u64,

    /// Error spans
    pub error_spans: u64,

    /// Average span duration
    pub avg_span_duration_us: f64,

    /// P95 span duration
    pub p95_span_duration_us: f64,

    /// P99 span duration
    pub p99_span_duration_us: f64,

    /// Spans per second
    pub spans_per_second: f64,

    /// Traces per second
    pub traces_per_second: f64,

    /// Memory usage
    pub memory_usage_bytes: u64,
}

/// Distributed tracer
pub struct DistributedTracer {
    config: TracingConfig,

    /// Active spans
    active_spans: Arc<RwLock<HashMap<String, Span>>>,

    /// Completed spans
    completed_spans: Arc<RwLock<Vec<Span>>>,

    /// Span sender
    span_sender: mpsc::UnboundedSender<Span>,

    /// Metrics sender
    metrics_sender: broadcast::Sender<TraceMetrics>,

    /// Background task handles
    task_handles: Arc<RwLock<Vec<tokio::task::JoinHandle<()>>>>,
}

impl DistributedTracer {
    pub fn new(config: TracingConfig) -> Self {
        let (span_sender, span_receiver) = mpsc::unbounded_channel();
        let (metrics_sender, _) = broadcast::channel(1000);

        let tracer = Self {
            config,
            active_spans: Arc::new(RwLock::new(HashMap::new())),
            completed_spans: Arc::new(RwLock::new(Vec::new())),
            span_sender,
            metrics_sender,
            task_handles: Arc::new(RwLock::new(Vec::new())),
        };

        // Start background processing
        tracer.start_background_processing(span_receiver);

        tracer
    }

    fn start_background_processing(&self, mut span_receiver: mpsc::UnboundedReceiver<Span>) {
        let config = self.config.clone();
        let completed_spans = Arc::clone(&self.completed_spans);
        let _metrics_sender = self.metrics_sender.clone();

        // Span processing task
        let span_task = tokio::spawn(async move {
            while let Some(span) = span_receiver.recv().await {
                // Add to completed spans
                {
                    let mut spans = completed_spans.write().await;
                    spans.push(span);

                    // Limit memory usage
                    if spans.len() > config.max_spans_in_memory {
                        let current_len = spans.len();
                        spans.drain(0..current_len - config.max_spans_in_memory);
                    }
                }
            }
        });

        // Metrics collection task
        let config_clone = self.config.clone();
        let active_spans = Arc::clone(&self.active_spans);
        let completed_spans_clone = Arc::clone(&self.completed_spans);
        let metrics_sender_clone = self.metrics_sender.clone();

        let metrics_task = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(config_clone.export_interval_seconds));

            loop {
                interval.tick().await;

                // Calculate metrics
                let metrics = Self::calculate_metrics(&active_spans, &completed_spans_clone).await;

                // Send metrics
                let _ = metrics_sender_clone.send(metrics);
            }
        });

        // Store task handles
        let task_handles = Arc::clone(&self.task_handles);
        tokio::spawn(async move {
            let mut handles = task_handles.write().await;
            handles.push(span_task);
            handles.push(metrics_task);
        });
    }

    async fn calculate_metrics(
        active_spans: &Arc<RwLock<HashMap<String, Span>>>,
        completed_spans: &Arc<RwLock<Vec<Span>>>,
    ) -> TraceMetrics {
        let active_spans_count = {
            let active = active_spans.read().await;
            active.len() as u64
        };

        let (completed_spans_count, avg_duration, p95_duration, p99_duration, error_count) = {
            let completed = completed_spans.read().await;
            let count = completed.len() as u64;

            let durations: Vec<u64> =
                completed.iter().filter_map(|span| span.duration_us).collect();

            let avg_duration = if !durations.is_empty() {
                durations.iter().sum::<u64>() as f64 / durations.len() as f64
            } else {
                0.0
            };

            let p95_duration = Self::percentile(&durations, 0.95);
            let p99_duration = Self::percentile(&durations, 0.99);

            let error_count =
                completed.iter().filter(|span| !matches!(span.status, SpanStatus::Ok)).count()
                    as u64;

            (count, avg_duration, p95_duration, p99_duration, error_count)
        };

        TraceMetrics {
            total_spans: active_spans_count + completed_spans_count,
            active_spans: active_spans_count,
            completed_spans: completed_spans_count,
            error_spans: error_count,
            avg_span_duration_us: avg_duration,
            p95_span_duration_us: p95_duration,
            p99_span_duration_us: p99_duration,
            spans_per_second: Self::calculate_spans_per_second(completed_spans).await,
            traces_per_second: Self::calculate_traces_per_second(completed_spans).await,
            memory_usage_bytes: Self::calculate_memory_usage(active_spans, completed_spans).await,
        }
    }

    async fn calculate_spans_per_second(completed_spans: &Arc<RwLock<Vec<Span>>>) -> f64 {
        let completed = completed_spans.read().await;

        // Calculate spans completed in the last 60 seconds
        let now = chrono::Utc::now();
        let one_minute_ago = now - chrono::Duration::seconds(60);

        let recent_spans = completed
            .iter()
            .filter(|span| span.end_time.map(|end_time| end_time > one_minute_ago).unwrap_or(false))
            .count();

        // Convert to per-second rate
        recent_spans as f64 / 60.0
    }

    async fn calculate_traces_per_second(completed_spans: &Arc<RwLock<Vec<Span>>>) -> f64 {
        let completed = completed_spans.read().await;

        // Calculate unique traces completed in the last 60 seconds
        let now = chrono::Utc::now();
        let one_minute_ago = now - chrono::Duration::seconds(60);

        let recent_trace_ids: std::collections::HashSet<_> = completed
            .iter()
            .filter(|span| span.end_time.map(|end_time| end_time > one_minute_ago).unwrap_or(false))
            .map(|span| &span.trace_id)
            .collect();

        // Convert to per-second rate
        recent_trace_ids.len() as f64 / 60.0
    }

    async fn calculate_memory_usage(
        active_spans: &Arc<RwLock<HashMap<String, Span>>>,
        completed_spans: &Arc<RwLock<Vec<Span>>>,
    ) -> u64 {
        let active = active_spans.read().await;
        let completed = completed_spans.read().await;

        // Estimate memory usage based on the size of span data structures
        // This is an approximation based on typical span sizes
        const APPROX_SPAN_SIZE: u64 = 512; // bytes per span (rough estimate)
        const HASHMAP_OVERHEAD: u64 = 64; // bytes per hashmap entry overhead

        let active_memory = active.len() as u64 * (APPROX_SPAN_SIZE + HASHMAP_OVERHEAD);
        let completed_memory = completed.len() as u64 * APPROX_SPAN_SIZE;

        active_memory + completed_memory
    }

    fn percentile(values: &[u64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_unstable();

        let index = (percentile * (sorted.len() - 1) as f64) as usize;
        sorted[index] as f64
    }

    pub async fn start_span(
        &self,
        operation_name: String,
        trace_context: Option<TraceContext>,
    ) -> Result<TraceContext, TracingError> {
        if !self.config.enabled {
            return Err(TracingError::TraceCreationError(
                "Tracing is disabled".to_string(),
            ));
        }

        // Check sampling
        use scirs2_core::random::*;
        let mut rng = thread_rng();
        if rng.random::<f64>() > self.config.sampling_rate {
            return Err(TracingError::TraceCreationError(
                "Span not sampled".to_string(),
            ));
        }

        let trace_context = trace_context.unwrap_or_default();

        let span = Span::new(
            &trace_context,
            operation_name,
            self.config.service_name.clone(),
        );

        // Add to active spans
        {
            let mut active_spans = self.active_spans.write().await;
            active_spans.insert(span.span_id.clone(), span);
        }

        Ok(trace_context)
    }

    pub async fn finish_span(&self, span_id: &str) -> Result<(), TracingError> {
        let mut span = {
            let mut active_spans = self.active_spans.write().await;
            active_spans
                .remove(span_id)
                .ok_or_else(|| TracingError::SpanNotFound(span_id.to_string()))?
        };

        span.finish();

        // Send to background processor
        self.span_sender
            .send(span)
            .map_err(|_| TracingError::TraceCreationError("Failed to send span".to_string()))?;

        Ok(())
    }

    pub async fn add_span_tag(
        &self,
        span_id: &str,
        key: String,
        value: String,
    ) -> Result<(), TracingError> {
        let mut active_spans = self.active_spans.write().await;
        let span = active_spans
            .get_mut(span_id)
            .ok_or_else(|| TracingError::SpanNotFound(span_id.to_string()))?;

        span.set_tag(key, value);
        Ok(())
    }

    pub async fn add_span_log(
        &self,
        span_id: &str,
        fields: HashMap<String, String>,
    ) -> Result<(), TracingError> {
        let mut active_spans = self.active_spans.write().await;
        let span = active_spans
            .get_mut(span_id)
            .ok_or_else(|| TracingError::SpanNotFound(span_id.to_string()))?;

        span.add_log(SpanLog::new(fields));
        Ok(())
    }

    pub async fn set_span_status(
        &self,
        span_id: &str,
        status: SpanStatus,
    ) -> Result<(), TracingError> {
        let mut active_spans = self.active_spans.write().await;
        let span = active_spans
            .get_mut(span_id)
            .ok_or_else(|| TracingError::SpanNotFound(span_id.to_string()))?;

        span.set_status(status);
        Ok(())
    }

    pub async fn get_trace(&self, trace_id: &str) -> Result<Vec<Span>, TracingError> {
        let completed_spans = self.completed_spans.read().await;
        let trace_spans: Vec<Span> = completed_spans
            .iter()
            .filter(|span| span.trace_id == trace_id)
            .cloned()
            .collect();

        Ok(trace_spans)
    }

    pub async fn get_metrics(&self) -> TraceMetrics {
        Self::calculate_metrics(&self.active_spans, &self.completed_spans).await
    }

    pub fn subscribe_to_metrics(&self) -> broadcast::Receiver<TraceMetrics> {
        self.metrics_sender.subscribe()
    }

    pub async fn export_traces(&self, format: TraceExportFormat) -> Result<Vec<u8>, TracingError> {
        let completed_spans = self.completed_spans.read().await;

        match format {
            TraceExportFormat::Jaeger => {
                let json = serde_json::to_string_pretty(&*completed_spans)?;
                Ok(json.into_bytes())
            },
            TraceExportFormat::Zipkin => {
                // Convert spans to Zipkin format
                let zipkin_spans: Vec<ZipkinSpan> =
                    completed_spans.iter().map(ZipkinSpan::from_span).collect();

                let json = serde_json::to_string_pretty(&zipkin_spans)?;
                Ok(json.into_bytes())
            },
            TraceExportFormat::OpenTelemetry => {
                // Convert spans to OpenTelemetry format
                let arc_spans: Vec<Arc<Span>> =
                    completed_spans.iter().map(|span| Arc::new(span.clone())).collect();
                let otel_export = OpenTelemetryExport::from_spans(&arc_spans);

                let json = serde_json::to_string_pretty(&otel_export)?;
                Ok(json.into_bytes())
            },
        }
    }

    pub async fn cleanup_old_traces(&self) -> Result<u64, TracingError> {
        let cutoff_time =
            chrono::Utc::now() - chrono::Duration::hours(self.config.trace_retention_hours as i64);
        let mut completed_spans = self.completed_spans.write().await;

        let initial_count = completed_spans.len();
        completed_spans.retain(|span| span.start_time > cutoff_time);
        let final_count = completed_spans.len();

        Ok((initial_count - final_count) as u64)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceExportFormat {
    Jaeger,
    Zipkin,
    OpenTelemetry,
}

/// Trace span guard for RAII-style tracing
pub struct SpanGuard {
    tracer: Arc<DistributedTracer>,
    span_id: String,
}

impl SpanGuard {
    pub fn new(tracer: Arc<DistributedTracer>, span_id: String) -> Self {
        Self { tracer, span_id }
    }

    pub async fn add_tag(&self, key: String, value: String) -> Result<(), TracingError> {
        self.tracer.add_span_tag(&self.span_id, key, value).await
    }

    pub async fn add_log(&self, fields: HashMap<String, String>) -> Result<(), TracingError> {
        self.tracer.add_span_log(&self.span_id, fields).await
    }

    pub async fn set_status(&self, status: SpanStatus) -> Result<(), TracingError> {
        self.tracer.set_span_status(&self.span_id, status).await
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        let tracer = Arc::clone(&self.tracer);
        let span_id = self.span_id.clone();

        // Use Handle::try_current() for async-safe spawning
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Err(e) = tracer.finish_span(&span_id).await {
                    // Use tracing::error! instead of error! for async safety
                    tracing::error!("Failed to finish span: {}", e);
                }
            });
        } else {
            // If no runtime available, attempt synchronous cleanup
            // This is a fallback for cases where Drop happens outside runtime
            tracing::warn!("No tokio runtime available during SpanGuard drop, span {} may not be properly finished", span_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_context_creation() {
        let context = TraceContext::new();
        assert!(!context.trace_id.is_empty());
        assert!(!context.span_id.is_empty());
        assert_eq!(context.trace_flags, 1);
    }

    #[test]
    fn test_trace_context_child_span() {
        let parent_context = TraceContext::new();
        let child_context = parent_context.child_span();

        assert_eq!(child_context.trace_id, parent_context.trace_id);
        assert_ne!(child_context.span_id, parent_context.span_id);
        assert_eq!(child_context.parent_span_id, Some(parent_context.span_id));
    }

    #[test]
    fn test_trace_context_headers() {
        let context = TraceContext::new();
        let headers = context.to_headers();

        assert!(headers.contains_key("X-Trace-Id"));
        assert!(headers.contains_key("X-Span-Id"));
        assert!(headers.contains_key("X-Trace-Flags"));

        let reconstructed = TraceContext::from_headers(&headers).unwrap();
        assert_eq!(reconstructed.trace_id, context.trace_id);
        assert_eq!(reconstructed.span_id, context.span_id);
    }

    #[tokio::test]
    async fn test_distributed_tracer_span_lifecycle() {
        let config = TracingConfig::default();
        let tracer = DistributedTracer::new(config);

        let context = tracer.start_span("test_operation".to_string(), None).await.unwrap();

        tracer
            .add_span_tag(&context.span_id, "key".to_string(), "value".to_string())
            .await
            .unwrap();

        tracer.finish_span(&context.span_id).await.unwrap();

        // Give some time for background processing
        tokio::time::sleep(Duration::from_millis(100)).await;

        let metrics = tracer.get_metrics().await;
        assert_eq!(metrics.completed_spans, 1);
    }

    #[tokio::test]
    async fn test_span_guard() {
        let config = TracingConfig::default();
        let tracer = Arc::new(DistributedTracer::new(config));

        let _span_id = {
            let context = tracer.start_span("test_operation".to_string(), None).await.unwrap();
            let guard = SpanGuard::new(Arc::clone(&tracer), context.span_id.clone());

            guard.add_tag("test_key".to_string(), "test_value".to_string()).await.unwrap();

            context.span_id
        }; // SpanGuard is dropped here

        // Give some time for background processing
        tokio::time::sleep(Duration::from_millis(100)).await;

        let metrics = tracer.get_metrics().await;
        assert_eq!(metrics.completed_spans, 1);
    }
}
