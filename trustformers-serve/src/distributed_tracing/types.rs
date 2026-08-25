//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
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

// Reused for its already-audited Jaeger tag shape (`{key, type, value}`,
// matching Jaeger's real `model.KeyValue`) rather than re-deriving it here.
use crate::tracing::legacy::export::JaegerTag;

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
/// Distributed tracing manager
pub struct TracingManager {
    /// Configuration
    pub(super) config: TracingConfig,
    /// Active spans
    pub(super) active_spans: Arc<RwLock<HashMap<String, Arc<Mutex<DistributedSpan>>>>>,
    /// Span export queue
    pub(super) export_queue: Arc<Mutex<Vec<DistributedSpan>>>,
    /// Statistics
    pub(super) stats: Arc<Mutex<TracingStats>>,
    /// Span counter
    pub(super) span_counter: AtomicU64,
    /// Sample counter for rate limiting
    pub(super) sample_counter: AtomicU64,
    /// Last sample time
    pub(super) last_sample_time: Arc<Mutex<Instant>>,
    /// Event sender for real-time updates
    pub(super) event_sender: Arc<broadcast::Sender<TracingEvent>>,
    /// Cached system CPU reading backing `SamplingStrategy::Adaptive`. See
    /// [`CpuLoadMonitor`].
    pub(super) cpu_load_monitor: Arc<Mutex<CpuLoadMonitor>>,
}
/// Minimum interval between `sysinfo` CPU refreshes for adaptive sampling.
///
/// `should_sample` runs synchronously on every span start; refreshing
/// `sysinfo::System`'s CPU usage that often would both cost far more than it
/// is worth on a hot path and, per `sysinfo`'s own documentation, make
/// consecutive readings less accurate (CPU usage is a delta over the time
/// since the last refresh).
const CPU_LOAD_REFRESH_INTERVAL: Duration = Duration::from_millis(500);
/// Backing store for [`SamplingStrategy::Adaptive`]'s system-load reading.
///
/// 0.2.1: `should_sample`'s `Adaptive` branch used to read `let current_load
/// = 0.5;` -- a constant, so the branch was in fact always exactly `min_rate`
/// or always exactly `max_rate` for any given config, never actually
/// adapting to load the way the variant's name and doc promise. `sysinfo` is
/// already a workspace dependency (see
/// `resource_management::manager::ResourceManagementSystem::get_performance_snapshot`
/// for the same CPU-usage pattern); this caches one `System` instance so
/// paying its refresh cost is bounded by [`CPU_LOAD_REFRESH_INTERVAL`]
/// rather than incurred on every call.
pub(super) struct CpuLoadMonitor {
    system: sysinfo::System,
    last_refresh: Instant,
}
impl TracingManager {
    /// Create new tracing manager
    pub async fn new(config: TracingConfig) -> Result<Arc<Self>> {
        let (event_sender, _) = broadcast::channel(1000);
        let mut cpu_system = sysinfo::System::new();
        cpu_system.refresh_cpu_usage();
        let manager = Arc::new(Self {
            config,
            active_spans: Arc::new(RwLock::new(HashMap::new())),
            export_queue: Arc::new(Mutex::new(Vec::new())),
            cpu_load_monitor: Arc::new(Mutex::new(CpuLoadMonitor {
                system: cpu_system,
                last_refresh: Instant::now(),
            })),
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
    /// Force export pending spans.
    ///
    /// 0.2.1: this used to clear the queue and *then* export, so a failed
    /// export lost the spans for good -- the same fire-and-forget loss
    /// `export_loop` was fixed for (see its comment above). A failed flush
    /// now puts the spans back at the front of the queue (bounded by
    /// `max_span_queue_size`, like any other queued span) before propagating
    /// the error, so a caller that retries, or the background export loop's
    /// next tick, can still recover them. `shutdown()` calls this, so this
    /// also means a failure right at shutdown no longer silently discards
    /// whatever was still queued. A successful flush now also counts toward
    /// `TracingStats::spans_exported`, which previously only `export_loop`
    /// updated -- a caller relying on `flush()` (as `shutdown()` does) saw an
    /// undercount of what was actually delivered.
    pub async fn flush(&self) -> Result<()> {
        let spans = {
            let mut queue = self.export_queue.lock();
            let spans = queue.clone();
            queue.clear();
            spans
        };
        if spans.is_empty() {
            return Ok(());
        }
        if let Err(error) = self.export_spans(spans.clone()).await {
            let mut queue = self.export_queue.lock();
            let room = self.config.max_span_queue_size.saturating_sub(queue.len());
            let requeued = spans.len().min(room);
            if requeued > 0 {
                let mut retry_batch = spans;
                retry_batch.truncate(requeued);
                retry_batch.extend(std::mem::take(&mut *queue));
                *queue = retry_batch;
            }
            return Err(error);
        }
        let mut stats = self.stats.lock();
        stats.spans_exported += spans.len() as u64;
        Ok(())
    }
    /// Shutdown tracing manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down distributed tracing...");
        self.flush().await?;
        info!("Distributed tracing shutdown complete");
        Ok(())
    }
    fn start_span_from_builder(&self, builder: SpanBuilder) -> Result<ActiveSpan> {
        if !self.should_sample() {
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
        span.attributes
            .insert("service.name".to_string(), self.config.service_name.clone());
        span.attributes.insert(
            "service.version".to_string(),
            self.config.service_version.clone(),
        );
        let span_arc = Arc::new(Mutex::new(span));
        self.active_spans.write().insert(span_id.clone(), span_arc.clone());
        self.span_counter.fetch_add(1, Ordering::Relaxed);
        {
            let mut stats = self.stats.lock();
            stats.spans_created += 1;
        }
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
                let current_load = self.current_cpu_load_fraction();
                let rate = if current_load > *target_cpu { *min_rate } else { *max_rate };
                use scirs2_core::random::*;
                let mut rng = thread_rng();
                rng.random::<f64>() < rate
            },
        }
    }
    /// Current system-wide CPU load as a `0.0..=1.0`-ish fraction (matching
    /// `SamplingStrategy::Adaptive::target_cpu`'s scale), refreshed at most
    /// once per [`CPU_LOAD_REFRESH_INTERVAL`]. See [`CpuLoadMonitor`].
    pub(crate) fn current_cpu_load_fraction(&self) -> f64 {
        let mut monitor = self.cpu_load_monitor.lock();
        if monitor.last_refresh.elapsed() >= CPU_LOAD_REFRESH_INTERVAL {
            monitor.system.refresh_cpu_usage();
            monitor.last_refresh = Instant::now();
        }
        monitor.system.global_cpu_usage() as f64 / 100.0
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
    pub(crate) fn queue_span_for_export(&self, span: DistributedSpan) {
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
                    let _ = self.event_sender.send(TracingEvent::ExportFailed {
                        error: e.to_string(),
                    });
                    {
                        let mut stats = self.stats.lock();
                        stats.export_failures += 1;
                    }
                    // 0.2.1: a failed batch used to be dropped outright here --
                    // logged and counted, but the spans themselves were gone
                    // for good. Put them back at the front of the queue so the
                    // next tick retries them (a bounded, best-effort retry,
                    // not an indefinite one: a batch that keeps failing stays
                    // capped at `max_span_queue_size` like any other queued
                    // span, and is not separately backed off).
                    let mut queue = self.export_queue.lock();
                    let room = self.config.max_span_queue_size.saturating_sub(queue.len());
                    let requeued = spans.len().min(room);
                    if requeued < spans.len() {
                        warn!(
                            "Export queue full: retrying {} of {} failed span(s), dropping {}",
                            requeued,
                            spans.len(),
                            spans.len() - requeued
                        );
                    }
                    if requeued > 0 {
                        let mut retry_batch = spans;
                        retry_batch.truncate(requeued);
                        retry_batch.extend(std::mem::take(&mut *queue));
                        *queue = retry_batch;
                    }
                } else {
                    let _ = self.event_sender.send(TracingEvent::ExportCompleted {
                        span_count: spans.len(),
                    });
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
            TracingBackend::Noop => {},
        }
        let export_latency = start_time.elapsed().as_millis() as f64;
        let mut stats = self.stats.lock();
        stats.export_latency_ms = export_latency;
        Ok(())
    }
    async fn export_to_jaeger(
        &self,
        spans: Vec<DistributedSpan>,
        endpoint: &str,
        username: Option<&str>,
        password: Option<&str>,
    ) -> Result<()> {
        let jaeger_spans = self.convert_to_jaeger_format(spans)?;
        let client = reqwest::Client::new();
        let mut request =
            client.post(endpoint).json(&jaeger_spans).timeout(self.config.export_timeout);
        // 0.2.1: `username`/`password` used to be accepted onto
        // `TracingBackend::Jaeger` and threaded all the way down to here,
        // then silently discarded (`_username`, `_password`): a caller who
        // configured credentials for an authenticated collector got
        // unauthenticated requests, which any collector actually enforcing
        // auth would reject with 401 -- reported to the caller as a generic
        // Jaeger export failure with no indication the credentials were
        // never sent.
        if let Some(username) = username {
            request = request.basic_auth(username, password);
        }
        let response = request.send().await.context("Failed to send spans to Jaeger")?;
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
    /// Converts spans to the `jaeger-query` `/api/traces` JSON document shape
    /// (the same target `tracing::legacy::export`'s `JaegerExport` documents,
    /// for the same "Load JSON File" UI use case) -- not the collector's own
    /// wire protocol, which is Thrift, not JSON.
    ///
    /// 0.2.1: `tags` and `process.tags` used to serialize `span.attributes`
    /// (a `HashMap<String, String>`) directly, producing a flat JSON object.
    /// Jaeger's real `model.KeyValue` tag is `{key, type, value}` -- an
    /// object keyed by attribute name is not a list of those, and a reader
    /// expecting the documented shape would fail to parse it as tags at all.
    /// This now builds the same [`JaegerTag`] type `tracing::legacy::export`
    /// already uses, rather than a second, differently-wrong shape.
    /// `parentSpanID` used to default to `""` for a root span via
    /// `unwrap_or_default()`; it is now `null` when absent, like every other
    /// converter in this file already represents "no parent".
    pub(crate) fn convert_to_jaeger_format(
        &self,
        spans: Vec<DistributedSpan>,
    ) -> Result<Vec<serde_json::Value>> {
        fn tags_from_attributes(attributes: HashMap<String, String>) -> Vec<JaegerTag> {
            let mut pairs: Vec<(String, String)> = attributes.into_iter().collect();
            pairs.sort_by(|(a, _), (b, _)| a.cmp(b));
            pairs
                .into_iter()
                .map(|(key, value)| JaegerTag {
                    key,
                    value_type: "string".to_string(),
                    value: serde_json::Value::String(value),
                })
                .collect()
        }
        let jaeger_spans = spans
            .into_iter()
            .map(|span| {
                let tags = tags_from_attributes(span.attributes);
                let process_tags = tags_from_attributes(span.resource_attributes);
                let logs: Vec<serde_json::Value> = span
                    .events
                    .into_iter()
                    .map(|event| {
                        serde_json::json!({
                            "timestamp": event.timestamp.timestamp_micros(),
                            "fields": tags_from_attributes(event.attributes),
                        })
                    })
                    .collect();
                serde_json::json!({
                    "traceID": span.trace_id,
                    "spanID": span.span_id,
                    "parentSpanID": span.parent_span_id,
                    "operationName": span.operation_name,
                    "startTime": span.start_time.timestamp_micros(),
                    "duration": span
                        .end_time
                        .map(|end| end.signed_duration_since(span.start_time).num_microseconds().unwrap_or(0))
                        .unwrap_or(0),
                    "tags": tags,
                    "logs": logs,
                    "process": {
                        "serviceName": span.service_name,
                        "tags": process_tags,
                    },
                })
            })
            .collect();
        Ok(jaeger_spans)
    }
    pub(crate) fn convert_to_zipkin_format(
        &self,
        spans: Vec<DistributedSpan>,
    ) -> Result<Vec<serde_json::Value>> {
        let zipkin_spans = spans
            .into_iter()
            .map(|span| {
                serde_json::json!(
                    { "traceId" : span.trace_id, "id" : span.span_id, "parentId" : span
                    .parent_span_id, "name" : span.operation_name, "timestamp" : span
                    .start_time.timestamp_micros(), "duration" : span.end_time.map(| end
                    | end.signed_duration_since(span.start_time).num_microseconds()
                    .unwrap_or(0)).unwrap_or(0), "kind" : match span.kind {
                    SpanKind::Client => "CLIENT", SpanKind::Server => "SERVER",
                    SpanKind::Producer => "PRODUCER", SpanKind::Consumer => "CONSUMER",
                    SpanKind::Internal => "INTERNAL", }, "localEndpoint" : {
                    "serviceName" : span.service_name }, "tags" : span.attributes,
                    "annotations" : span.events.into_iter().map(| event | {
                    serde_json::json!({ "timestamp" : event.timestamp.timestamp_micros(),
                    "value" : event.name }) }).collect::< Vec < _ >> () }
                )
            })
            .collect();
        Ok(zipkin_spans)
    }
    /// Converts spans to OTLP/JSON (`resourceSpans[].scopeSpans[].spans[]`).
    ///
    /// 0.2.1: this nested `instrumentationLibrarySpans`/`instrumentationLibrary`
    /// -- the pre-1.0 OTLP field names, renamed to `scopeSpans`/`scope` when
    /// OTLP went stable and no longer recognized by a current collector (see
    /// `tracing::legacy::export`'s own `OtlpScopeSpans`/`OtlpScope`, which
    /// already use the current names). `startTimeUnixNano`/`endTimeUnixNano`
    /// are now JSON strings: OTLP/JSON represents protobuf `fixed64` fields
    /// as strings precisely so a 64-bit nanosecond timestamp survives a
    /// JSON-number round-trip through a language whose numbers are
    /// `f64`-precision, which every one of these values otherwise would.
    /// `parentSpanId` used to default to `""` via `unwrap_or_default()`; it
    /// is now `null` when absent, matching every other converter here.
    ///
    /// 0.2.1: `resourceSpans[].resource.attributes` never carried
    /// `service.name`, the one resource attribute OTLP semantic conventions
    /// require to identify which service emitted a trace (see
    /// `tracing::legacy::export::OpenTelemetryExport::from_spans`, which
    /// already sets it correctly) -- every export from this function reported
    /// spans with no identifiable service. It is now always the first
    /// resource attribute; a `TracingConfig::resource_attributes` entry
    /// literally keyed `"service.name"` (a caller could set one, since the
    /// field is a free-form `HashMap`) is skipped rather than emitted a
    /// second time.
    pub(crate) fn convert_to_otlp_format(
        &self,
        spans: Vec<DistributedSpan>,
    ) -> Result<serde_json::Value> {
        let otlp_spans = spans
            .into_iter()
            .map(|span| {
                serde_json::json!(
                    { "traceId" : span.trace_id, "spanId" : span.span_id, "parentSpanId"
                    : span.parent_span_id, "name" : span
                    .operation_name, "kind" : match span.kind { SpanKind::Internal => 1,
                    SpanKind::Server => 2, SpanKind::Client => 3, SpanKind::Producer =>
                    4, SpanKind::Consumer => 5, }, "startTimeUnixNano" : span.start_time
                    .timestamp_nanos_opt().unwrap_or(0).to_string(), "endTimeUnixNano" : span
                    .end_time.map(| end | end.timestamp_nanos_opt().unwrap_or(0))
                    .unwrap_or(0).to_string(), "attributes" : span.attributes.into_iter().map(| (k,
                    v) | { serde_json::json!({ "key" : k, "value" : { "stringValue" : v }
                    }) }).collect::< Vec < _ >> (), "events" : span.events.into_iter()
                    .map(| event | { serde_json::json!({ "timeUnixNano" : event.timestamp
                    .timestamp_nanos_opt().unwrap_or(0).to_string(), "name" : event.name,
                    "attributes" : event.attributes.into_iter().map(| (k, v) | {
                    serde_json::json!({ "key" : k, "value" : { "stringValue" : v } }) })
                    .collect::< Vec < _ >> () }) }).collect::< Vec < _ >> () }
                )
            })
            .collect::<Vec<_>>();
        let mut resource_attributes = vec![serde_json::json!({
            "key": "service.name",
            "value": { "stringValue": self.config.service_name },
        })];
        resource_attributes.extend(
            self.config
                .resource_attributes
                .iter()
                .filter(|(k, _)| k.as_str() != "service.name")
                .map(|(k, v)| serde_json::json!({ "key": k, "value": { "stringValue": v } })),
        );
        Ok(serde_json::json!(
            { "resourceSpans" : [{ "resource" : { "attributes" : resource_attributes },
            "scopeSpans" : [{ "scope" : { "name" :
            "trustformers-serve", "version" : self.config.service_version }, "spans"
            : otlp_spans }] }] }
        ))
    }
}
/// Active span handle
pub struct ActiveSpan {
    pub(super) span: Arc<Mutex<DistributedSpan>>,
    pub(super) manager: Arc<TracingManager>,
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
            flags: 1,
            state: HashMap::new(),
            baggage: HashMap::new(),
        }
    }
    /// Finish the span
    pub fn finish(self) {
        let mut span = self.span.lock();
        let end_time = Utc::now();
        span.end_time = Some(end_time);
        let duration = end_time.signed_duration_since(span.start_time).num_milliseconds() as f64;
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
        self.manager.queue_span_for_export(span.clone());
    }
}
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
