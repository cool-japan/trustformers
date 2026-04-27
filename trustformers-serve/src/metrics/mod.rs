//! Prometheus-compatible metrics for inference serving.
//!
//! This module provides two layers:
//! 1. `serving` — a pure-Rust Prometheus-style metrics registry with counters,
//!    gauges, histograms, and text-format export (no external crate required).
//! 2. Re-exports of the legacy `MetricsCollector` / `MetricsService` built on
//!    the `prometheus` crate (kept for backwards compatibility).

// ── Legacy prometheus-crate implementation ──────────────────────────────────

use once_cell::sync::Lazy;
use prometheus::{
    opts, register_histogram, register_int_counter, register_int_gauge, Encoder, Histogram,
    HistogramOpts, IntCounter, IntGauge, Registry, TextEncoder,
};
use std::sync::Arc;

static REGISTRY: Lazy<Registry> = Lazy::new(Registry::new);

static REQUEST_COUNTER: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!(opts!(
        "inference_requests_total",
        "Total number of inference requests"
    ))
    .expect("failed to register REQUEST_COUNTER metric")
});

static REQUEST_DURATION: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(HistogramOpts::new(
        "inference_request_duration_seconds",
        "Request duration in seconds"
    ))
    .expect("failed to register REQUEST_DURATION metric")
});

static BATCH_SIZE: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(HistogramOpts::new(
        "inference_batch_size",
        "Batch size for inference requests"
    )
    .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]))
    .expect("failed to register BATCH_SIZE metric")
});

static ACTIVE_REQUESTS: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!(opts!(
        "inference_active_requests",
        "Number of active inference requests"
    ))
    .expect("failed to register ACTIVE_REQUESTS metric")
});

static MODEL_LOADS: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!(opts!(
        "inference_model_loads_total",
        "Total number of model loads"
    ))
    .expect("failed to register MODEL_LOADS metric")
});

static ERRORS: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!(opts!(
        "inference_errors_total",
        "Total number of inference errors"
    ))
    .expect("failed to register ERRORS metric")
});

static QUEUE_SIZE: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!(opts!(
        "inference_queue_size",
        "Current size of the inference queue"
    ))
    .expect("failed to register QUEUE_SIZE metric")
});

#[derive(Clone)]
pub struct MetricsCollector {
    registry: Arc<Registry>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(REGISTRY.clone()),
        }
    }

    pub fn increment_requests(&self) {
        REQUEST_COUNTER.inc();
    }

    pub fn observe_request_duration(&self, duration_seconds: f64) {
        REQUEST_DURATION.observe(duration_seconds);
    }

    pub fn observe_batch_size(&self, size: f64) {
        BATCH_SIZE.observe(size);
    }

    pub fn increment_active_requests(&self) {
        ACTIVE_REQUESTS.inc();
    }

    pub fn decrement_active_requests(&self) {
        ACTIVE_REQUESTS.dec();
    }

    pub fn increment_model_loads(&self) {
        MODEL_LOADS.inc();
    }

    pub fn increment_errors(&self) {
        ERRORS.inc();
    }

    pub fn set_queue_size(&self, size: i64) {
        QUEUE_SIZE.set(size);
    }

    pub fn export_metrics(&self) -> Result<String, Box<dyn std::error::Error>> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }

    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct RequestMetrics {
    pub start_time: std::time::Instant,
    pub request_id: String,
}

impl RequestMetrics {
    pub fn new(request_id: String) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            request_id,
        }
    }

    pub fn finish(&self, collector: &MetricsCollector, success: bool) {
        let duration = self.start_time.elapsed().as_secs_f64();
        collector.observe_request_duration(duration);
        collector.decrement_active_requests();

        if !success {
            collector.increment_errors();
        }
    }
}

pub struct MetricsService {
    collector: MetricsCollector,
}

impl MetricsService {
    pub fn new() -> Self {
        Self {
            collector: MetricsCollector::new(),
        }
    }

    pub fn collector(&self) -> &MetricsCollector {
        &self.collector
    }

    pub async fn get_metrics(&self) -> Result<String, Box<dyn std::error::Error>> {
        self.collector.export_metrics()
    }
}

impl Default for MetricsService {
    fn default() -> Self {
        Self::new()
    }
}

// ── Pure-Rust Prometheus-style metrics ───────────────────────────────────────

/// Metric types following Prometheus conventions.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    /// Monotonically increasing value.
    Counter,
    /// Value that can go up or down.
    Gauge,
    /// Bucketed observation distribution.
    Histogram,
    /// Quantile-based distribution.
    Summary,
}

impl std::fmt::Display for MetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricType::Counter => write!(f, "counter"),
            MetricType::Gauge => write!(f, "gauge"),
            MetricType::Histogram => write!(f, "histogram"),
            MetricType::Summary => write!(f, "summary"),
        }
    }
}

// ── Label ────────────────────────────────────────────────────────────────────

/// A single Prometheus label key-value pair.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label {
    pub key: String,
    pub value: String,
}

impl Label {
    pub fn new(key: &str, value: &str) -> Self {
        Self {
            key: key.to_owned(),
            value: value.to_owned(),
        }
    }

    /// Render as `key="value"`.
    fn render(&self) -> String {
        format!(r#"{}="{}""#, self.key, self.value)
    }
}

fn render_labels(labels: &[Label]) -> String {
    if labels.is_empty() {
        String::new()
    } else {
        let parts: Vec<String> = labels.iter().map(Label::render).collect();
        format!("{{{}}}", parts.join(","))
    }
}

// ── Counter ──────────────────────────────────────────────────────────────────

/// A monotonically increasing counter metric.
pub struct Counter {
    pub name: String,
    pub help: String,
    value: f64,
    labels: Vec<Label>,
}

impl Counter {
    pub fn new(name: &str, help: &str) -> Self {
        Self {
            name: name.to_owned(),
            help: help.to_owned(),
            value: 0.0,
            labels: Vec::new(),
        }
    }

    pub fn with_labels(mut self, labels: Vec<Label>) -> Self {
        self.labels = labels;
        self
    }

    pub fn inc(&mut self) {
        self.value += 1.0;
    }

    pub fn inc_by(&mut self, amount: f64) {
        self.value += amount;
    }

    pub fn get(&self) -> f64 {
        self.value
    }

    pub fn reset(&mut self) {
        self.value = 0.0;
    }

    fn export_prometheus(&self) -> String {
        format!(
            "# HELP {} {}\n# TYPE {} counter\n{}{} {}\n",
            self.name,
            self.help,
            self.name,
            self.name,
            render_labels(&self.labels),
            self.value
        )
    }
}

// ── Gauge ────────────────────────────────────────────────────────────────────

/// A gauge metric that can go up or down.
pub struct Gauge {
    pub name: String,
    pub help: String,
    value: f64,
    labels: Vec<Label>,
}

impl Gauge {
    pub fn new(name: &str, help: &str) -> Self {
        Self {
            name: name.to_owned(),
            help: help.to_owned(),
            value: 0.0,
            labels: Vec::new(),
        }
    }

    pub fn with_labels(mut self, labels: Vec<Label>) -> Self {
        self.labels = labels;
        self
    }

    pub fn set(&mut self, v: f64) {
        self.value = v;
    }

    pub fn inc(&mut self) {
        self.value += 1.0;
    }

    pub fn dec(&mut self) {
        self.value -= 1.0;
    }

    pub fn inc_by(&mut self, amount: f64) {
        self.value += amount;
    }

    pub fn get(&self) -> f64 {
        self.value
    }

    fn export_prometheus(&self) -> String {
        format!(
            "# HELP {} {}\n# TYPE {} gauge\n{}{} {}\n",
            self.name,
            self.help,
            self.name,
            self.name,
            render_labels(&self.labels),
            self.value
        )
    }
}

// ── Histogram ────────────────────────────────────────────────────────────────

/// Default histogram bucket upper bounds (in seconds / ms depending on usage).
const DEFAULT_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Histogram with configurable buckets.
pub struct HistogramMetric {
    pub name: String,
    pub help: String,
    /// Upper bounds of each bucket (exclusive of +Inf).
    pub buckets: Vec<f64>,
    /// Cumulative counts per bucket (observations <= upper_bound).
    bucket_counts: Vec<u64>,
    count: u64,
    sum: f64,
    labels: Vec<Label>,
}

impl HistogramMetric {
    /// Create with default buckets.
    pub fn new(name: &str, help: &str) -> Self {
        Self::with_buckets(name, help, DEFAULT_BUCKETS.to_vec())
    }

    /// Create with custom bucket upper bounds.
    pub fn with_buckets(name: &str, help: &str, buckets: Vec<f64>) -> Self {
        let n = buckets.len();
        Self {
            name: name.to_owned(),
            help: help.to_owned(),
            buckets,
            bucket_counts: vec![0u64; n],
            count: 0,
            sum: 0.0,
            labels: Vec::new(),
        }
    }

    pub fn with_labels(mut self, labels: Vec<Label>) -> Self {
        self.labels = labels;
        self
    }

    /// Record an observation.
    pub fn observe(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        for (i, &ub) in self.buckets.iter().enumerate() {
            if value <= ub {
                self.bucket_counts[i] += 1;
            }
        }
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn sum(&self) -> f64 {
        self.sum
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Estimate quantile `q` (0.0–1.0) via linear interpolation over buckets.
    pub fn quantile(&self, q: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let target = (q * self.count as f64) as u64;
        let mut prev_count = 0u64;
        let mut prev_upper = 0.0_f64;
        for (i, &ub) in self.buckets.iter().enumerate() {
            let bucket_count = self.bucket_counts[i];
            if bucket_count >= target {
                if bucket_count == prev_count {
                    return prev_upper;
                }
                let frac = (target - prev_count) as f64 / (bucket_count - prev_count) as f64;
                return prev_upper + frac * (ub - prev_upper);
            }
            prev_count = bucket_count;
            prev_upper = ub;
        }
        *self.buckets.last().unwrap_or(&f64::INFINITY)
    }

    fn export_prometheus(&self) -> String {
        let mut out = format!(
            "# HELP {} {}\n# TYPE {} histogram\n",
            self.name, self.help, self.name
        );
        let label_str = render_labels(&self.labels);
        for (i, &ub) in self.buckets.iter().enumerate() {
            let le_label = if label_str.is_empty() {
                format!(r#"{{le="{}"}}"#, ub)
            } else {
                // Strip closing } and append le="..."}
                let inner = &label_str[1..label_str.len() - 1];
                format!(r#"{{{},"le":"{}"}}"#, inner, ub)
            };
            out.push_str(&format!(
                "{}_bucket{} {}\n",
                self.name, le_label, self.bucket_counts[i]
            ));
        }
        // +Inf bucket
        let inf_label = if label_str.is_empty() {
            r#"{le="+Inf"}"#.to_owned()
        } else {
            let inner = &label_str[1..label_str.len() - 1];
            format!(r#"{{{},"le":"+Inf"}}"#, inner)
        };
        out.push_str(&format!(
            "{}_bucket{} {}\n",
            self.name, inf_label, self.count
        ));
        out.push_str(&format!("{}_sum{} {}\n", self.name, label_str, self.sum));
        out.push_str(&format!(
            "{}_count{} {}\n",
            self.name, label_str, self.count
        ));
        out
    }
}

// ── ServingMetrics ───────────────────────────────────────────────────────────

/// Complete inference-serving metrics registry.
pub struct ServingMetrics {
    // Request counters
    pub requests_total: Counter,
    pub requests_failed: Counter,
    // Latency histograms
    pub request_latency_ms: HistogramMetric,
    pub ttft_ms: HistogramMetric,
    pub tpot_ms: HistogramMetric,
    // Throughput gauges
    pub tokens_per_second: Gauge,
    pub active_requests: Gauge,
    pub queue_depth: Gauge,
    // Cache metrics
    pub kv_cache_hit_rate: Gauge,
    pub kv_cache_utilization: Gauge,
    // Model metrics
    pub model_load_time_ms: Gauge,
    pub gpu_utilization: Gauge,
    pub memory_used_bytes: Gauge,
}

impl ServingMetrics {
    pub fn new() -> Self {
        // Latency buckets in milliseconds.
        let latency_buckets = vec![
            1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
        ];
        Self {
            requests_total: Counter::new(
                "serving_requests_total",
                "Total number of inference requests handled",
            ),
            requests_failed: Counter::new(
                "serving_requests_failed_total",
                "Total number of failed inference requests",
            ),
            request_latency_ms: HistogramMetric::with_buckets(
                "serving_request_latency_ms",
                "End-to-end request latency in milliseconds",
                latency_buckets.clone(),
            ),
            ttft_ms: HistogramMetric::with_buckets(
                "serving_ttft_ms",
                "Time to first token in milliseconds",
                latency_buckets.clone(),
            ),
            tpot_ms: HistogramMetric::with_buckets(
                "serving_tpot_ms",
                "Time per output token in milliseconds",
                vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0],
            ),
            tokens_per_second: Gauge::new(
                "serving_tokens_per_second",
                "Current throughput in tokens per second",
            ),
            active_requests: Gauge::new(
                "serving_active_requests",
                "Number of requests currently being processed",
            ),
            queue_depth: Gauge::new(
                "serving_queue_depth",
                "Number of requests waiting in the queue",
            ),
            kv_cache_hit_rate: Gauge::new(
                "serving_kv_cache_hit_rate",
                "KV cache hit rate (0.0–1.0)",
            ),
            kv_cache_utilization: Gauge::new(
                "serving_kv_cache_utilization",
                "KV cache memory utilization (0.0–1.0)",
            ),
            model_load_time_ms: Gauge::new(
                "serving_model_load_time_ms",
                "Time taken to load the model in milliseconds",
            ),
            gpu_utilization: Gauge::new(
                "serving_gpu_utilization",
                "GPU utilization fraction (0.0–1.0)",
            ),
            memory_used_bytes: Gauge::new(
                "serving_memory_used_bytes",
                "Current memory consumption in bytes",
            ),
        }
    }

    /// Record a completed request.
    pub fn record_request(
        &mut self,
        latency_ms: f64,
        ttft_ms: f64,
        output_tokens: usize,
        success: bool,
    ) {
        self.requests_total.inc();
        if !success {
            self.requests_failed.inc();
        }
        self.request_latency_ms.observe(latency_ms);
        self.ttft_ms.observe(ttft_ms);
        if output_tokens > 0 && latency_ms > 0.0 {
            let tpot = latency_ms / output_tokens as f64;
            self.tpot_ms.observe(tpot);
        }
    }

    /// Export all metrics in Prometheus text format.
    pub fn export_prometheus(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.requests_total.export_prometheus());
        out.push_str(&self.requests_failed.export_prometheus());
        out.push_str(&self.request_latency_ms.export_prometheus());
        out.push_str(&self.ttft_ms.export_prometheus());
        out.push_str(&self.tpot_ms.export_prometheus());
        out.push_str(&self.tokens_per_second.export_prometheus());
        out.push_str(&self.active_requests.export_prometheus());
        out.push_str(&self.queue_depth.export_prometheus());
        out.push_str(&self.kv_cache_hit_rate.export_prometheus());
        out.push_str(&self.kv_cache_utilization.export_prometheus());
        out.push_str(&self.model_load_time_ms.export_prometheus());
        out.push_str(&self.gpu_utilization.export_prometheus());
        out.push_str(&self.memory_used_bytes.export_prometheus());
        out
    }

    /// Human-readable metrics summary.
    pub fn summary(&self) -> MetricsSummary {
        let total = self.requests_total.get() as u64;
        let failed = self.requests_failed.get() as u64;
        let success_rate = if total == 0 { 1.0 } else { (total - failed) as f32 / total as f32 };
        MetricsSummary {
            total_requests: total,
            failed_requests: failed,
            success_rate,
            p50_latency_ms: self.request_latency_ms.quantile(0.50),
            p95_latency_ms: self.request_latency_ms.quantile(0.95),
            p99_latency_ms: self.request_latency_ms.quantile(0.99),
            mean_latency_ms: self.request_latency_ms.mean(),
            mean_ttft_ms: self.ttft_ms.mean(),
            mean_tpot_ms: self.tpot_ms.mean(),
            active_requests: self.active_requests.get(),
        }
    }
}

impl Default for ServingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of key serving metrics.
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub total_requests: u64,
    pub failed_requests: u64,
    pub success_rate: f32,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub mean_latency_ms: f64,
    pub mean_ttft_ms: f64,
    pub mean_tpot_ms: f64,
    pub active_requests: f64,
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Counter ──────────────────────────────────────────────────────────────

    #[test]
    fn test_counter_inc() {
        let mut c = Counter::new("test_counter", "help text");
        assert_eq!(c.get(), 0.0);
        c.inc();
        c.inc();
        assert_eq!(c.get(), 2.0);
    }

    #[test]
    fn test_counter_inc_by() {
        let mut c = Counter::new("test_counter_by", "help");
        c.inc_by(5.0);
        c.inc_by(3.0);
        assert_eq!(c.get(), 8.0);
    }

    #[test]
    fn test_counter_reset() {
        let mut c = Counter::new("test_reset", "help");
        c.inc_by(100.0);
        c.reset();
        assert_eq!(c.get(), 0.0);
    }

    // ── Gauge ────────────────────────────────────────────────────────────────

    #[test]
    fn test_gauge_set_inc_dec() {
        let mut g = Gauge::new("test_gauge", "help");
        g.set(10.0);
        assert_eq!(g.get(), 10.0);
        g.inc();
        assert_eq!(g.get(), 11.0);
        g.dec();
        assert_eq!(g.get(), 10.0);
        g.inc_by(5.5);
        assert!((g.get() - 15.5).abs() < 1e-10);
    }

    // ── Histogram ────────────────────────────────────────────────────────────

    #[test]
    fn test_histogram_default_buckets() {
        let h = HistogramMetric::new("latency", "latency help");
        assert_eq!(h.buckets.len(), DEFAULT_BUCKETS.len());
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn test_histogram_observe_basic() {
        let mut h = HistogramMetric::new("h", "help");
        h.observe(0.005); // falls in 0.005 bucket
        assert_eq!(h.count(), 1);
        assert!((h.sum() - 0.005).abs() < 1e-12);
    }

    #[test]
    fn test_histogram_observe_count_sum() {
        let mut h = HistogramMetric::new("h2", "help");
        h.observe(0.1);
        h.observe(0.2);
        h.observe(0.5);
        assert_eq!(h.count(), 3);
        assert!((h.sum() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_mean() {
        let mut h = HistogramMetric::new("mean_test", "help");
        h.observe(2.0);
        h.observe(4.0);
        assert!((h.mean() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_quantile_empty() {
        let h = HistogramMetric::new("empty_hist", "help");
        assert_eq!(h.quantile(0.5), 0.0);
        assert_eq!(h.quantile(0.99), 0.0);
    }

    #[test]
    fn test_histogram_quantile_p50() {
        // Use explicit buckets so we can reason about the result.
        let mut h = HistogramMetric::with_buckets("p50", "help", vec![1.0, 2.0, 5.0, 10.0]);
        // 10 observations: 4 × 1.0, 3 × 2.0, 2 × 5.0, 1 × 10.0
        for _ in 0..4 {
            h.observe(1.0);
        }
        for _ in 0..3 {
            h.observe(2.0);
        }
        for _ in 0..2 {
            h.observe(5.0);
        }
        h.observe(10.0);
        // p50 = 5th observation → should fall in bucket [1.0, 2.0].
        let p50 = h.quantile(0.50);
        assert!(
            (0.0..=5.0).contains(&p50),
            "p50={p50} out of expected range"
        );
    }

    #[test]
    fn test_histogram_quantile_p99() {
        let mut h = HistogramMetric::with_buckets("p99", "help", vec![10.0, 50.0, 100.0, 500.0]);
        // 100 observations mostly at 10ms, one outlier at 490ms.
        for _ in 0..99 {
            h.observe(10.0);
        }
        h.observe(490.0);
        let p99 = h.quantile(0.99);
        // p99 should be in the 10–50ms bucket (99th of 100 observations).
        assert!(p99 <= 50.0, "p99={p99} expected ≤ 50");
    }

    // ── ServingMetrics ────────────────────────────────────────────────────────

    #[test]
    fn test_serving_metrics_new() {
        let m = ServingMetrics::new();
        assert_eq!(m.requests_total.get(), 0.0);
        assert_eq!(m.active_requests.get(), 0.0);
    }

    #[test]
    fn test_record_request_success() {
        let mut m = ServingMetrics::new();
        m.record_request(50.0, 10.0, 100, true);
        assert_eq!(m.requests_total.get(), 1.0);
        assert_eq!(m.requests_failed.get(), 0.0);
        assert_eq!(m.request_latency_ms.count(), 1);
    }

    #[test]
    fn test_record_request_failure() {
        let mut m = ServingMetrics::new();
        m.record_request(100.0, 20.0, 0, false);
        assert_eq!(m.requests_total.get(), 1.0);
        assert_eq!(m.requests_failed.get(), 1.0);
    }

    #[test]
    fn test_record_multiple_requests() {
        let mut m = ServingMetrics::new();
        for i in 0..10 {
            m.record_request(i as f64 * 10.0, i as f64 * 2.0, 50, i % 3 != 0);
        }
        assert_eq!(m.requests_total.get(), 10.0);
        // i=0,3,6,9 → 4 failures
        assert_eq!(m.requests_failed.get(), 4.0);
        assert_eq!(m.request_latency_ms.count(), 10);
    }

    #[test]
    fn test_export_prometheus_format() {
        let mut m = ServingMetrics::new();
        m.record_request(50.0, 10.0, 50, true);
        let output = m.export_prometheus();
        assert!(
            output.contains("# HELP serving_requests_total"),
            "missing HELP line"
        );
        assert!(
            output.contains("# TYPE serving_requests_total counter"),
            "missing TYPE line"
        );
        assert!(
            output.contains("serving_requests_total"),
            "missing metric value"
        );
        assert!(
            output.contains("serving_request_latency_ms_bucket"),
            "missing histogram buckets"
        );
    }

    #[test]
    fn test_metrics_summary() {
        let mut m = ServingMetrics::new();
        m.active_requests.set(3.0);
        m.record_request(100.0, 20.0, 50, true);
        m.record_request(200.0, 40.0, 50, false);
        let s = m.summary();
        assert_eq!(s.total_requests, 2);
        assert_eq!(s.failed_requests, 1);
        assert!((s.success_rate - 0.5).abs() < 1e-6);
        assert!(s.mean_latency_ms > 0.0);
        assert_eq!(s.active_requests, 3.0);
    }

    #[test]
    fn test_metric_type_display() {
        assert_eq!(MetricType::Counter.to_string(), "counter");
        assert_eq!(MetricType::Gauge.to_string(), "gauge");
        assert_eq!(MetricType::Histogram.to_string(), "histogram");
        assert_eq!(MetricType::Summary.to_string(), "summary");
    }

    #[test]
    fn test_label_new() {
        let l = Label::new("model", "gpt-4");
        assert_eq!(l.key, "model");
        assert_eq!(l.value, "gpt-4");
        assert_eq!(l.render(), r#"model="gpt-4""#);
    }
}
