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
