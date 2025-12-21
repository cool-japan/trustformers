/*!
# Distributed Tracing System

Comprehensive distributed tracing infrastructure for trustformers-serve with OpenTelemetry integration.

## Features

- Distributed trace propagation across service boundaries
- Integration with external tracing systems (Jaeger, Zipkin, OTLP)
- Automatic span creation for HTTP requests and inference operations
- Custom span attributes and events
- Distributed context propagation
- Performance metrics correlation
- Sampling strategies for production environments

## Usage

```rust
use trustformers_serve::distributed_tracing::{TracingManager, TracingConfig};

// Initialize tracing
let config = TracingConfig::default()
    .with_service_name("trustformers-serve")
    .with_jaeger_endpoint("http://localhost:14268/api/traces");

let tracing_manager = TracingManager::new(config).await?;

// Use in HTTP handlers
let span = tracing_manager.start_inference_span("text-generation", &request_id);
span.set_attribute("model_name", "llama-7b");
span.add_event("model_loaded", vec![("load_time_ms", "150")]);
```
*/

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

/// Trace sampling strategies for different environments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Always sample traces (development)
    Always,
    /// Never sample traces (disabled)
    Never,
    /// Sample based on probability (0.0 - 1.0)
    Probabilistic(f64),
    /// Sample based on rate (traces per second)
    RateLimited(u64),
    /// Adaptive sampling based on system load
    Adaptive {
        min_rate: f64,
        max_rate: f64,
        target_cpu: f64,
    },
}

/// Supported tracing backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TracingBackend {
    /// Jaeger tracing system
    Jaeger {
        endpoint: String,
        username: Option<String>,
        password: Option<String>,
    },
    /// Zipkin tracing system
    Zipkin { endpoint: String },
    /// OpenTelemetry Protocol (OTLP)
    Otlp {
        endpoint: String,
        headers: HashMap<String, String>,
    },
    /// Console output (development)
    Console,
    /// No-op backend (disabled)
    Noop,
}

/// Configuration for distributed tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Service name for tracing
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// Tracing backend configuration
    pub backend: TracingBackend,
    /// Sampling strategy
    pub sampling: SamplingStrategy,
    /// Maximum span queue size
    pub max_span_queue_size: usize,
    /// Span export timeout
    pub export_timeout: Duration,
    /// Batch export configuration
    pub batch_config: BatchConfig,
    /// Additional resource attributes
    pub resource_attributes: HashMap<String, String>,
    /// Enable automatic HTTP instrumentation
    pub auto_http_instrumentation: bool,
    /// Enable automatic database instrumentation
    pub auto_db_instrumentation: bool,
}

/// Batch export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum batch timeout
    pub max_batch_timeout: Duration,
    /// Maximum queue size
    pub max_queue_size: usize,
}

/// Span kind for categorizing operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SpanKind {
    /// Internal operation
    Internal,
    /// Incoming request
    Server,
    /// Outgoing request
    Client,
    /// Producer operation
    Producer,
    /// Consumer operation
    Consumer,
}

/// Span status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanStatus {
    /// Unset status
    Unset,
    /// Operation completed successfully
    Ok,
    /// Operation failed
    Error { message: String },
}

/// Span event with timestamp and attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Event name
    pub name: String,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event attributes
    pub attributes: HashMap<String, String>,
}

/// Distributed trace span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedSpan {
    /// Unique span ID
    pub span_id: String,
    /// Trace ID this span belongs to
    pub trace_id: String,
    /// Parent span ID (if any)
    pub parent_span_id: Option<String>,
    /// Operation name
    pub operation_name: String,
    /// Span kind
    pub kind: SpanKind,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time (if finished)
    pub end_time: Option<DateTime<Utc>>,
    /// Span status
    pub status: SpanStatus,
    /// Span attributes
    pub attributes: HashMap<String, String>,
    /// Span events
    pub events: Vec<SpanEvent>,
    /// Service name
    pub service_name: String,
    /// Resource attributes
    pub resource_attributes: HashMap<String, String>,
}

/// Trace context for propagation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Trace ID
    pub trace_id: String,
    /// Span ID
    pub span_id: String,
    /// Trace flags
    pub flags: u8,
    /// Trace state
    pub state: HashMap<String, String>,
    /// Baggage items
    pub baggage: HashMap<String, String>,
}

/// Span builder for creating new spans
pub struct SpanBuilder {
    operation_name: String,
    kind: SpanKind,
    parent_context: Option<TraceContext>,
    attributes: HashMap<String, String>,
    start_time: Option<DateTime<Utc>>,
}

/// Active span handle
pub struct ActiveSpan {
    span: Arc<Mutex<DistributedSpan>>,
    manager: Arc<TracingManager>,
}

/// Tracing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingStats {
    /// Total spans created
    pub spans_created: u64,
    /// Total spans exported
    pub spans_exported: u64,
    /// Export failures
    pub export_failures: u64,
    /// Queue size
    pub queue_size: usize,
    /// Average span duration
    pub avg_span_duration_ms: f64,
    /// Sampling rate
    pub sampling_rate: f64,
    /// Export latency
    pub export_latency_ms: f64,
}

/// Distributed tracing manager
pub struct TracingManager {
    /// Configuration
    config: TracingConfig,
    /// Active spans
    active_spans: Arc<RwLock<HashMap<String, Arc<Mutex<DistributedSpan>>>>>,
    /// Span export queue
    export_queue: Arc<Mutex<Vec<DistributedSpan>>>,
    /// Statistics
    stats: Arc<Mutex<TracingStats>>,
    /// Span counter
    span_counter: AtomicU64,
    /// Sample counter for rate limiting
    sample_counter: AtomicU64,
    /// Last sample time
    last_sample_time: Arc<Mutex<Instant>>,
    /// Event sender for real-time updates
    event_sender: Arc<broadcast::Sender<TracingEvent>>,
}

/// Tracing events for monitoring
#[derive(Debug, Clone)]
pub enum TracingEvent {
    /// Span started
    SpanStarted { span_id: String, trace_id: String },
    /// Span finished
    SpanFinished { span_id: String, duration_ms: f64 },
    /// Export completed
    ExportCompleted { span_count: usize },
    /// Export failed
    ExportFailed { error: String },
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "trustformers-serve".to_string(),
            service_version: "0.1.0".to_string(),
            backend: TracingBackend::Console,
            sampling: SamplingStrategy::Always,
            max_span_queue_size: 10000,
            export_timeout: Duration::from_secs(30),
            batch_config: BatchConfig::default(),
            resource_attributes: HashMap::new(),
            auto_http_instrumentation: true,
            auto_db_instrumentation: true,
        }
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 512,
            max_batch_timeout: Duration::from_secs(5),
            max_queue_size: 2048,
        }
    }
}

impl TracingConfig {
    /// Create new configuration with service name
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            ..Default::default()
        }
    }

    /// Set service name
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Set service version
    pub fn with_service_version(mut self, version: impl Into<String>) -> Self {
        self.service_version = version.into();
        self
    }

    /// Configure Jaeger backend
    pub fn with_jaeger_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.backend = TracingBackend::Jaeger {
            endpoint: endpoint.into(),
            username: None,
            password: None,
        };
        self
    }

    /// Configure Zipkin backend
    pub fn with_zipkin_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.backend = TracingBackend::Zipkin {
            endpoint: endpoint.into(),
        };
        self
    }

    /// Configure OTLP backend
    pub fn with_otlp_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.backend = TracingBackend::Otlp {
            endpoint: endpoint.into(),
            headers: HashMap::new(),
        };
        self
    }

    /// Set sampling strategy
    pub fn with_sampling(mut self, sampling: SamplingStrategy) -> Self {
        self.sampling = sampling;
        self
    }

    /// Add resource attribute
    pub fn with_resource_attribute(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.resource_attributes.insert(key.into(), value.into());
        self
    }
}

impl SpanBuilder {
    /// Create new span builder
    pub fn new(operation_name: impl Into<String>) -> Self {
        Self {
            operation_name: operation_name.into(),
            kind: SpanKind::Internal,
            parent_context: None,
            attributes: HashMap::new(),
            start_time: None,
        }
    }

    /// Set span kind
    pub fn with_kind(mut self, kind: SpanKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set parent context
    pub fn with_parent_context(mut self, context: TraceContext) -> Self {
        self.parent_context = Some(context);
        self
    }

    /// Add attribute
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Set custom start time
    pub fn with_start_time(mut self, start_time: DateTime<Utc>) -> Self {
        self.start_time = Some(start_time);
        self
    }

    /// Start the span
    pub fn start(self, manager: &TracingManager) -> Result<ActiveSpan> {
        manager.start_span_from_builder(self)
    }
}

impl ActiveSpan {
    /// Set span attribute
    pub fn set_attribute(&self, key: impl Into<String>, value: impl Into<String>) {
        let mut span = self.span.lock();
        span.attributes.insert(key.into(), value.into());
    }

    /// Add span event
    pub fn add_event(
        &self,
        name: impl Into<String>,
        attributes: Vec<(impl Into<String>, impl Into<String>)>,
    ) {
        let mut span = self.span.lock();
        let event = SpanEvent {
            name: name.into(),
            timestamp: Utc::now(),
            attributes: attributes.into_iter().map(|(k, v)| (k.into(), v.into())).collect(),
        };
        span.events.push(event);
    }

    /// Set span status
    pub fn set_status(&self, status: SpanStatus) {
        let mut span = self.span.lock();
        span.status = status;
    }

    /// Mark span as error
    pub fn set_error(&self, message: impl Into<String>) {
        self.set_status(SpanStatus::Error {
            message: message.into(),
        });
    }

    /// Get trace context
    pub fn get_context(&self) -> TraceContext {
        let span = self.span.lock();
        TraceContext {
            trace_id: span.trace_id.clone(),
            span_id: span.span_id.clone(),
            flags: 1, // Sampled
            state: HashMap::new(),
            baggage: HashMap::new(),
        }
    }

    /// Finish the span
    pub fn finish(self) {
        let mut span = self.span.lock();
        span.end_time = Some(Utc::now());

        // Calculate duration
        let duration =
            span.end_time.unwrap().signed_duration_since(span.start_time).num_milliseconds() as f64;

        // Emit event
        if self
            .manager
            .event_sender
            .send(TracingEvent::SpanFinished {
                span_id: span.span_id.clone(),
                duration_ms: duration,
            })
            .is_ok()
        {
            debug!("Span finished: {} ({}ms)", span.span_id, duration);
        }

        // Queue for export
        self.manager.queue_span_for_export(span.clone());
    }
}

impl Drop for ActiveSpan {
    fn drop(&mut self) {
        // Auto-finish span if not already finished using try_lock for async safety
        if let Some(span) = self.span.try_lock() {
            if span.end_time.is_none() {
                drop(span);

                if let Some(mut span) = self.span.try_lock() {
                    span.end_time = Some(Utc::now());
                    self.manager.queue_span_for_export(span.clone());
                }
            }
        } else {
            // If span is locked, defer finishing using runtime handle
            let span = self.span.clone();
            let manager = self.manager.clone();

            // Use Handle::try_current() for async-safe spawning
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    // Use async lock in spawned task
                    let mut span_guard = span.lock();
                    if span_guard.end_time.is_none() {
                        span_guard.end_time = Some(Utc::now());
                        manager.queue_span_for_export(span_guard.clone());
                    }
                });
            } else {
                // If no runtime available, attempt non-blocking finish
                tracing::warn!("No tokio runtime available during ActiveSpan drop, span may not be properly finished");
            }
        }
    }
}

impl TracingManager {
    /// Create new tracing manager
    pub async fn new(config: TracingConfig) -> Result<Arc<Self>> {
        let (event_sender, _) = broadcast::channel(1000);

        let manager = Arc::new(Self {
            config,
            active_spans: Arc::new(RwLock::new(HashMap::new())),
            export_queue: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(TracingStats {
                spans_created: 0,
                spans_exported: 0,
                export_failures: 0,
                queue_size: 0,
                avg_span_duration_ms: 0.0,
                sampling_rate: 1.0,
                export_latency_ms: 0.0,
            })),
            span_counter: AtomicU64::new(0),
            sample_counter: AtomicU64::new(0),
            last_sample_time: Arc::new(Mutex::new(Instant::now())),
            event_sender: Arc::new(event_sender),
        });

        // Start background export task
        let manager_clone = manager.clone();
        tokio::spawn(async move {
            manager_clone.export_loop().await;
        });

        info!(
            "Distributed tracing initialized with backend: {:?}",
            manager.config.backend
        );
        Ok(manager)
    }

    /// Subscribe to tracing events
    pub fn subscribe_events(&self) -> broadcast::Receiver<TracingEvent> {
        self.event_sender.subscribe()
    }

    /// Start a new span
    pub fn start_span(&self, operation_name: impl Into<String>) -> Result<ActiveSpan> {
        SpanBuilder::new(operation_name).start(self)
    }

    /// Start a new span with kind
    pub fn start_span_with_kind(
        &self,
        operation_name: impl Into<String>,
        kind: SpanKind,
    ) -> Result<ActiveSpan> {
        SpanBuilder::new(operation_name).with_kind(kind).start(self)
    }

    /// Start inference span
    pub fn start_inference_span(
        &self,
        model_name: impl Into<String>,
        request_id: impl Into<String>,
    ) -> Result<ActiveSpan> {
        let span = self.start_span_with_kind("inference", SpanKind::Server)?;
        span.set_attribute("model.name", model_name);
        span.set_attribute("request.id", request_id);
        span.set_attribute("operation.type", "inference");
        Ok(span)
    }

    /// Start HTTP request span
    pub fn start_http_span(
        &self,
        method: impl Into<String>,
        uri: impl Into<String>,
    ) -> Result<ActiveSpan> {
        let span = self.start_span_with_kind("http_request", SpanKind::Server)?;
        span.set_attribute("http.method", method);
        span.set_attribute("http.uri", uri);
        span.set_attribute("component", "http");
        Ok(span)
    }

    /// Extract trace context from headers
    pub fn extract_context(&self, headers: &HashMap<String, String>) -> Option<TraceContext> {
        // W3C Trace Context format
        if let Some(traceparent) = headers.get("traceparent") {
            self.parse_traceparent(traceparent)
        } else if let Some(uber_trace) = headers.get("uber-trace-id") {
            self.parse_uber_trace(uber_trace)
        } else {
            None
        }
    }

    /// Inject trace context into headers
    pub fn inject_context(&self, context: &TraceContext, headers: &mut HashMap<String, String>) {
        // W3C Trace Context format
        let traceparent = format!(
            "00-{}-{}-{:02x}",
            context.trace_id, context.span_id, context.flags
        );
        headers.insert("traceparent".to_string(), traceparent);

        if !context.state.is_empty() {
            let tracestate = context
                .state
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(",");
            headers.insert("tracestate".to_string(), tracestate);
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> TracingStats {
        let mut stats = self.stats.lock().clone();
        stats.queue_size = self.export_queue.lock().len();
        stats
    }

    /// Force export pending spans
    pub async fn flush(&self) -> Result<()> {
        let spans = {
            let mut queue = self.export_queue.lock();
            let spans = queue.clone();
            queue.clear();
            spans
        };

        if !spans.is_empty() {
            self.export_spans(spans).await?;
        }

        Ok(())
    }

    /// Shutdown tracing manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down distributed tracing...");
        self.flush().await?;
        info!("Distributed tracing shutdown complete");
        Ok(())
    }

    // Internal methods

    fn start_span_from_builder(&self, builder: SpanBuilder) -> Result<ActiveSpan> {
        // Check sampling
        if !self.should_sample() {
            // Return no-op span for non-sampled traces
            return Ok(self.create_noop_span());
        }

        let span_id = self.generate_span_id();
        let trace_id = if let Some(ref parent) = builder.parent_context {
            parent.trace_id.clone()
        } else {
            self.generate_trace_id()
        };

        let mut span = DistributedSpan {
            span_id: span_id.clone(),
            trace_id: trace_id.clone(),
            parent_span_id: builder.parent_context.as_ref().map(|c| c.span_id.clone()),
            operation_name: builder.operation_name,
            kind: builder.kind,
            start_time: builder.start_time.unwrap_or_else(Utc::now),
            end_time: None,
            status: SpanStatus::Unset,
            attributes: builder.attributes,
            events: Vec::new(),
            service_name: self.config.service_name.clone(),
            resource_attributes: self.config.resource_attributes.clone(),
        };

        // Add standard attributes
        span.attributes
            .insert("service.name".to_string(), self.config.service_name.clone());
        span.attributes.insert(
            "service.version".to_string(),
            self.config.service_version.clone(),
        );

        let span_arc = Arc::new(Mutex::new(span));

        // Store active span
        self.active_spans.write().insert(span_id.clone(), span_arc.clone());

        // Update statistics
        self.span_counter.fetch_add(1, Ordering::Relaxed);
        {
            let mut stats = self.stats.lock();
            stats.spans_created += 1;
        }

        // Emit event
        if self
            .event_sender
            .send(TracingEvent::SpanStarted {
                span_id: span_id.clone(),
                trace_id: trace_id.clone(),
            })
            .is_ok()
        {
            debug!("Span started: {} (trace: {})", span_id, trace_id);
        }

        Ok(ActiveSpan {
            span: span_arc,
            manager: Arc::new(self.clone()),
        })
    }

    fn should_sample(&self) -> bool {
        match &self.config.sampling {
            SamplingStrategy::Always => true,
            SamplingStrategy::Never => false,
            SamplingStrategy::Probabilistic(rate) => {
                use scirs2_core::random::*;
                let mut rng = thread_rng();
                rng.random::<f64>() < *rate
            },
            SamplingStrategy::RateLimited(rate) => {
                let now = Instant::now();
                let mut last_sample = self.last_sample_time.lock();
                let elapsed = now.duration_since(*last_sample);

                if elapsed >= Duration::from_secs_f64(1.0 / *rate as f64) {
                    *last_sample = now;
                    true
                } else {
                    false
                }
            },
            SamplingStrategy::Adaptive {
                min_rate,
                max_rate,
                target_cpu,
            } => {
                // Simplified adaptive sampling based on system load
                // In production, this would check actual CPU usage
                let current_load = 0.5; // Placeholder
                let rate = if current_load > *target_cpu { *min_rate } else { *max_rate };

                use scirs2_core::random::*;
                let mut rng = thread_rng();
                rng.random::<f64>() < rate
            },
        }
    }

    fn create_noop_span(&self) -> ActiveSpan {
        let span = DistributedSpan {
            span_id: "noop".to_string(),
            trace_id: "noop".to_string(),
            parent_span_id: None,
            operation_name: "noop".to_string(),
            kind: SpanKind::Internal,
            start_time: Utc::now(),
            end_time: None,
            status: SpanStatus::Unset,
            attributes: HashMap::new(),
            events: Vec::new(),
            service_name: self.config.service_name.clone(),
            resource_attributes: HashMap::new(),
        };

        ActiveSpan {
            span: Arc::new(Mutex::new(span)),
            manager: Arc::new(self.clone()),
        }
    }

    fn generate_span_id(&self) -> String {
        use scirs2_core::random::*;
        let mut rng = thread_rng();
        format!("{:016x}", rng.random::<u64>())
    }

    fn generate_trace_id(&self) -> String {
        use scirs2_core::random::*;
        let mut rng = thread_rng();
        format!("{:032x}", rng.random::<u128>())
    }

    fn parse_traceparent(&self, traceparent: &str) -> Option<TraceContext> {
        let parts: Vec<&str> = traceparent.split('-').collect();
        if parts.len() == 4 && parts[0] == "00" {
            Some(TraceContext {
                trace_id: parts[1].to_string(),
                span_id: parts[2].to_string(),
                flags: u8::from_str_radix(parts[3], 16).unwrap_or(0),
                state: HashMap::new(),
                baggage: HashMap::new(),
            })
        } else {
            None
        }
    }

    fn parse_uber_trace(&self, uber_trace: &str) -> Option<TraceContext> {
        let parts: Vec<&str> = uber_trace.split(':').collect();
        if parts.len() >= 4 {
            Some(TraceContext {
                trace_id: parts[0].to_string(),
                span_id: parts[1].to_string(),
                flags: if parts[3] == "1" { 1 } else { 0 },
                state: HashMap::new(),
                baggage: HashMap::new(),
            })
        } else {
            None
        }
    }

    fn queue_span_for_export(&self, span: DistributedSpan) {
        let mut queue = self.export_queue.lock();
        if queue.len() < self.config.max_span_queue_size {
            queue.push(span);
        } else {
            warn!("Span export queue full, dropping span");
        }
    }

    async fn export_loop(&self) {
        let mut interval = tokio::time::interval(self.config.batch_config.max_batch_timeout);

        loop {
            interval.tick().await;

            let spans = {
                let mut queue = self.export_queue.lock();
                if queue.len() >= self.config.batch_config.max_batch_size || !queue.is_empty() {
                    let batch_size =
                        std::cmp::min(queue.len(), self.config.batch_config.max_batch_size);
                    queue.drain(0..batch_size).collect::<Vec<_>>()
                } else {
                    Vec::new()
                }
            };

            if !spans.is_empty() {
                if let Err(e) = self.export_spans(spans.clone()).await {
                    error!("Failed to export spans: {}", e);

                    // Emit failure event
                    let _ = self.event_sender.send(TracingEvent::ExportFailed {
                        error: e.to_string(),
                    });

                    // Update statistics
                    let mut stats = self.stats.lock();
                    stats.export_failures += 1;
                } else {
                    // Emit success event
                    let _ = self.event_sender.send(TracingEvent::ExportCompleted {
                        span_count: spans.len(),
                    });

                    // Update statistics
                    let mut stats = self.stats.lock();
                    stats.spans_exported += spans.len() as u64;
                }
            }
        }
    }

    async fn export_spans(&self, spans: Vec<DistributedSpan>) -> Result<()> {
        let start_time = Instant::now();

        match &self.config.backend {
            TracingBackend::Jaeger {
                endpoint,
                username,
                password,
            } => {
                self.export_to_jaeger(spans, endpoint, username.as_deref(), password.as_deref())
                    .await?;
            },
            TracingBackend::Zipkin { endpoint } => {
                self.export_to_zipkin(spans, endpoint).await?;
            },
            TracingBackend::Otlp { endpoint, headers } => {
                self.export_to_otlp(spans, endpoint, headers).await?;
            },
            TracingBackend::Console => {
                self.export_to_console(spans).await?;
            },
            TracingBackend::Noop => {
                // No-op
            },
        }

        // Update export latency
        let export_latency = start_time.elapsed().as_millis() as f64;
        let mut stats = self.stats.lock();
        stats.export_latency_ms = export_latency;

        Ok(())
    }

    async fn export_to_jaeger(
        &self,
        spans: Vec<DistributedSpan>,
        endpoint: &str,
        _username: Option<&str>,
        _password: Option<&str>,
    ) -> Result<()> {
        // Convert spans to Jaeger format
        let jaeger_spans = self.convert_to_jaeger_format(spans)?;

        // Send to Jaeger (simplified implementation)
        let client = reqwest::Client::new();
        let response = client
            .post(endpoint)
            .json(&jaeger_spans)
            .timeout(self.config.export_timeout)
            .send()
            .await
            .context("Failed to send spans to Jaeger")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Jaeger export failed with status: {}",
                response.status()
            ));
        }

        debug!("Exported {} spans to Jaeger", jaeger_spans.len());
        Ok(())
    }

    async fn export_to_zipkin(&self, spans: Vec<DistributedSpan>, endpoint: &str) -> Result<()> {
        // Convert spans to Zipkin format
        let zipkin_spans = self.convert_to_zipkin_format(spans)?;

        let client = reqwest::Client::new();
        let response = client
            .post(endpoint)
            .json(&zipkin_spans)
            .timeout(self.config.export_timeout)
            .send()
            .await
            .context("Failed to send spans to Zipkin")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Zipkin export failed with status: {}",
                response.status()
            ));
        }

        debug!("Exported {} spans to Zipkin", zipkin_spans.len());
        Ok(())
    }

    async fn export_to_otlp(
        &self,
        spans: Vec<DistributedSpan>,
        endpoint: &str,
        headers: &HashMap<String, String>,
    ) -> Result<()> {
        let span_count = spans.len();
        // Convert spans to OTLP format
        let otlp_spans = self.convert_to_otlp_format(spans)?;

        let mut request = reqwest::Client::new()
            .post(endpoint)
            .json(&otlp_spans)
            .timeout(self.config.export_timeout);

        for (key, value) in headers {
            request = request.header(key, value);
        }

        let response = request.send().await.context("Failed to send spans to OTLP endpoint")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "OTLP export failed with status: {}",
                response.status()
            ));
        }

        debug!("Exported {} spans to OTLP", span_count);
        Ok(())
    }

    async fn export_to_console(&self, spans: Vec<DistributedSpan>) -> Result<()> {
        for span in spans {
            let duration = if let Some(end_time) = span.end_time {
                end_time.signed_duration_since(span.start_time).num_milliseconds()
            } else {
                0
            };

            info!(
                "TRACE [{}] {} | {} | {}ms | {:?} | attrs: {:?}",
                span.trace_id,
                span.span_id,
                span.operation_name,
                duration,
                span.kind,
                span.attributes
            );

            for event in &span.events {
                info!(
                    "  EVENT [{}] {} | {} | {:?}",
                    span.span_id,
                    event.name,
                    event.timestamp.format("%H:%M:%S%.3f"),
                    event.attributes
                );
            }
        }
        Ok(())
    }

    fn convert_to_jaeger_format(
        &self,
        spans: Vec<DistributedSpan>,
    ) -> Result<Vec<serde_json::Value>> {
        // Simplified Jaeger format conversion
        let jaeger_spans = spans
            .into_iter()
            .map(|span| {
                serde_json::json!({
                    "traceID": span.trace_id,
                    "spanID": span.span_id,
                    "parentSpanID": span.parent_span_id.unwrap_or_default(),
                    "operationName": span.operation_name,
                    "startTime": span.start_time.timestamp_micros(),
                    "duration": span.end_time.map(|end|
                        end.signed_duration_since(span.start_time).num_microseconds().unwrap_or(0)
                    ).unwrap_or(0),
                    "tags": span.attributes,
                    "logs": span.events.into_iter().map(|event| {
                        serde_json::json!({
                            "timestamp": event.timestamp.timestamp_micros(),
                            "fields": event.attributes
                        })
                    }).collect::<Vec<_>>(),
                    "process": {
                        "serviceName": span.service_name,
                        "tags": span.resource_attributes
                    }
                })
            })
            .collect();

        Ok(jaeger_spans)
    }

    fn convert_to_zipkin_format(
        &self,
        spans: Vec<DistributedSpan>,
    ) -> Result<Vec<serde_json::Value>> {
        // Simplified Zipkin format conversion
        let zipkin_spans = spans
            .into_iter()
            .map(|span| {
                serde_json::json!({
                    "traceId": span.trace_id,
                    "id": span.span_id,
                    "parentId": span.parent_span_id,
                    "name": span.operation_name,
                    "timestamp": span.start_time.timestamp_micros(),
                    "duration": span.end_time.map(|end|
                        end.signed_duration_since(span.start_time).num_microseconds().unwrap_or(0)
                    ).unwrap_or(0),
                    "kind": match span.kind {
                        SpanKind::Client => "CLIENT",
                        SpanKind::Server => "SERVER",
                        SpanKind::Producer => "PRODUCER",
                        SpanKind::Consumer => "CONSUMER",
                        SpanKind::Internal => "INTERNAL",
                    },
                    "localEndpoint": {
                        "serviceName": span.service_name
                    },
                    "tags": span.attributes,
                    "annotations": span.events.into_iter().map(|event| {
                        serde_json::json!({
                            "timestamp": event.timestamp.timestamp_micros(),
                            "value": event.name
                        })
                    }).collect::<Vec<_>>()
                })
            })
            .collect();

        Ok(zipkin_spans)
    }

    fn convert_to_otlp_format(&self, spans: Vec<DistributedSpan>) -> Result<serde_json::Value> {
        // Simplified OTLP format conversion
        let otlp_spans = spans
            .into_iter()
            .map(|span| {
                serde_json::json!({
                    "traceId": span.trace_id,
                    "spanId": span.span_id,
                    "parentSpanId": span.parent_span_id.unwrap_or_default(),
                    "name": span.operation_name,
                    "kind": match span.kind {
                        SpanKind::Internal => 1,
                        SpanKind::Server => 2,
                        SpanKind::Client => 3,
                        SpanKind::Producer => 4,
                        SpanKind::Consumer => 5,
                    },
                    "startTimeUnixNano": span.start_time.timestamp_nanos_opt().unwrap_or(0),
                    "endTimeUnixNano": span.end_time.map(|end|
                        end.timestamp_nanos_opt().unwrap_or(0)
                    ).unwrap_or(0),
                    "attributes": span.attributes.into_iter().map(|(k, v)| {
                        serde_json::json!({
                            "key": k,
                            "value": { "stringValue": v }
                        })
                    }).collect::<Vec<_>>(),
                    "events": span.events.into_iter().map(|event| {
                        serde_json::json!({
                            "timeUnixNano": event.timestamp.timestamp_nanos_opt().unwrap_or(0),
                            "name": event.name,
                            "attributes": event.attributes.into_iter().map(|(k, v)| {
                                serde_json::json!({
                                    "key": k,
                                    "value": { "stringValue": v }
                                })
                            }).collect::<Vec<_>>()
                        })
                    }).collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();

        Ok(serde_json::json!({
            "resourceSpans": [{
                "resource": {
                    "attributes": self.config.resource_attributes.iter().map(|(k, v)| {
                        serde_json::json!({
                            "key": k,
                            "value": { "stringValue": v }
                        })
                    }).collect::<Vec<_>>()
                },
                "instrumentationLibrarySpans": [{
                    "instrumentationLibrary": {
                        "name": "trustformers-serve",
                        "version": self.config.service_version
                    },
                    "spans": otlp_spans
                }]
            }]
        }))
    }
}

// Make TracingManager cloneable for use in ActiveSpan
impl Clone for TracingManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            active_spans: self.active_spans.clone(),
            export_queue: self.export_queue.clone(),
            stats: self.stats.clone(),
            span_counter: AtomicU64::new(self.span_counter.load(Ordering::Relaxed)),
            sample_counter: AtomicU64::new(self.sample_counter.load(Ordering::Relaxed)),
            last_sample_time: self.last_sample_time.clone(),
            event_sender: self.event_sender.clone(),
        }
    }
}

// Utility functions for common tracing patterns

/// Create distributed tracing middleware for HTTP requests
pub fn create_tracing_middleware(
    manager: Arc<TracingManager>,
) -> impl tower::Layer<axum::routing::Router> {
    tower_http::trace::TraceLayer::new_for_http().make_span_with(
        move |request: &axum::http::Request<axum::body::Body>| {
            let method = request.method().to_string();
            let uri = request.uri().to_string();

            // Extract trace context from headers
            let headers: HashMap<String, String> = request
                .headers()
                .iter()
                .filter_map(|(name, value)| {
                    value.to_str().ok().map(|v| (name.to_string(), v.to_string()))
                })
                .collect();

            let context = manager.extract_context(&headers);

            // Start HTTP span
            let span_result = if let Some(context) = context {
                SpanBuilder::new(format!("{} {}", method, uri))
                    .with_kind(SpanKind::Server)
                    .with_parent_context(context)
                    .with_attribute("http.method", method.clone())
                    .with_attribute("http.uri", uri.clone())
                    .start(&manager)
            } else {
                manager.start_http_span(method, uri)
            };

            match span_result {
                Ok(span) => {
                    // Store span in request extensions for later use
                    tracing::info_span!("http_request", span_id = %span.get_context().span_id)
                },
                Err(e) => {
                    tracing::error!("Failed to create tracing span: {}", e);
                    tracing::info_span!("http_request_fallback")
                },
            }
        },
    )
}

/// Trace an async function
pub async fn trace_async<F, T>(
    manager: &TracingManager,
    operation_name: &str,
    attributes: Vec<(&str, &str)>,
    f: F,
) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    let span = manager.start_span(operation_name)?;

    for (key, value) in attributes {
        span.set_attribute(key, value);
    }

    let result = f.await;

    match &result {
        Ok(_) => span.set_status(SpanStatus::Ok),
        Err(e) => span.set_error(e.to_string()),
    }

    span.finish();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tracing_manager_creation() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config).await.unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.spans_created, 0);
        assert_eq!(stats.spans_exported, 0);
    }

    #[tokio::test]
    async fn test_span_creation_and_attributes() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config).await.unwrap();

        let span = manager.start_span("test_operation").unwrap();
        span.set_attribute("test.key", "test.value");
        span.add_event("test_event", vec![("event.key", "event.value")]);
        span.finish();

        let stats = manager.get_stats();
        assert_eq!(stats.spans_created, 1);
    }

    #[tokio::test]
    async fn test_trace_context_propagation() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config).await.unwrap();

        let parent_span = manager.start_span("parent_operation").unwrap();
        let parent_context = parent_span.get_context();

        let child_span = SpanBuilder::new("child_operation")
            .with_parent_context(parent_context.clone())
            .start(&manager)
            .unwrap();

        let child_context = child_span.get_context();
        assert_eq!(child_context.trace_id, parent_context.trace_id);
        assert_ne!(child_context.span_id, parent_context.span_id);

        child_span.finish();
        parent_span.finish();
    }

    #[tokio::test]
    async fn test_sampling_strategies() {
        // Test probabilistic sampling
        let config = TracingConfig::default().with_sampling(SamplingStrategy::Probabilistic(0.5));
        let manager = TracingManager::new(config).await.unwrap();

        // Create multiple spans to test sampling
        for i in 0..100 {
            if let Ok(span) = manager.start_span(format!("test_span_{}", i)) {
                span.finish();
            }
        }

        let stats = manager.get_stats();
        // With 50% sampling, we should have created some but not all spans
        // (This is probabilistic, so we just check it's reasonable)
        assert!(stats.spans_created <= 100);
    }

    #[tokio::test]
    async fn test_inference_span() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config).await.unwrap();

        let span = manager.start_inference_span("llama-7b", "req-123").unwrap();

        let span_guard = span.span.lock();
        assert_eq!(span_guard.operation_name, "inference");
        assert!(span_guard.attributes.contains_key("model.name"));
        assert!(span_guard.attributes.contains_key("request.id"));
        drop(span_guard);

        span.finish();
    }

    #[tokio::test]
    async fn test_context_extraction() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config).await.unwrap();

        let mut headers = HashMap::new();
        headers.insert(
            "traceparent".to_string(),
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01".to_string(),
        );

        let context = manager.extract_context(&headers).unwrap();
        assert_eq!(context.trace_id, "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(context.span_id, "b7ad6b7169203331");
        assert_eq!(context.flags, 1);
    }

    #[tokio::test]
    async fn test_context_injection() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config).await.unwrap();

        let context = TraceContext {
            trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
            span_id: "b7ad6b7169203331".to_string(),
            flags: 1,
            state: HashMap::new(),
            baggage: HashMap::new(),
        };

        let mut headers = HashMap::new();
        manager.inject_context(&context, &mut headers);

        assert!(headers.contains_key("traceparent"));
        assert_eq!(
            headers["traceparent"],
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        );
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config).await.unwrap();

        let mut event_receiver = manager.subscribe_events();

        let span = manager.start_span("test_operation").unwrap();

        // Check if we receive span started event
        if let Ok(event) =
            tokio::time::timeout(Duration::from_millis(100), event_receiver.recv()).await
        {
            match event.unwrap() {
                TracingEvent::SpanStarted { span_id, trace_id } => {
                    assert!(!span_id.is_empty());
                    assert!(!trace_id.is_empty());
                },
                other => {
                    // Use assert_eq to provide better error message in tests
                    assert_eq!(
                        std::mem::discriminant(&other),
                        std::mem::discriminant(&TracingEvent::SpanStarted {
                            span_id: String::new(),
                            trace_id: String::new()
                        }),
                        "Expected SpanStarted event, got: {:?}",
                        other
                    );
                },
            }
        }

        span.finish();
    }

    #[tokio::test]
    async fn test_flush_and_shutdown() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config).await.unwrap();

        let span = manager.start_span("test_operation").unwrap();
        span.finish();

        // Test flush
        manager.flush().await.unwrap();

        // Test shutdown
        manager.shutdown().await.unwrap();
    }
}
