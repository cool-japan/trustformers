//! Custom Metrics Collection System
//!
//! Advanced metrics collection beyond basic Prometheus metrics, including
//! business metrics, performance analytics, and real-time monitoring.

use anyhow::Result;
use prometheus::{Gauge, Histogram, IntCounter};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, warn};

use crate::performance_optimizer::real_time_metrics::analytics::analyzers::series::pearson_p_value;
use crate::server::system_stats::HostSnapshot;

/// Custom metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetricsConfig {
    /// Enable custom metrics collection
    pub enabled: bool,
    /// Metrics collection interval in seconds
    pub collection_interval_seconds: u64,
    /// Enable real-time analytics
    pub enable_real_time_analytics: bool,
    /// Enable business metrics
    pub enable_business_metrics: bool,
    /// Enable performance profiling
    pub enable_performance_profiling: bool,
    /// Enable adaptive sampling
    pub enable_adaptive_sampling: bool,
    /// Metric retention period in hours
    pub retention_period_hours: u64,
    /// Maximum number of custom metric series
    pub max_metric_series: usize,
    /// Alert thresholds configuration
    pub alert_thresholds: AlertThresholds,
    /// Export configuration
    pub export_config: MetricsExportConfig,
}

impl Default for CustomMetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_seconds: 10,
            enable_real_time_analytics: true,
            enable_business_metrics: true,
            enable_performance_profiling: true,
            enable_adaptive_sampling: true,
            retention_period_hours: 24,
            max_metric_series: 10000,
            alert_thresholds: AlertThresholds::default(),
            export_config: MetricsExportConfig::default(),
        }
    }
}

/// Alert thresholds configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// CPU usage threshold (0.0-1.0)
    pub cpu_usage_threshold: f64,
    /// Memory usage threshold (0.0-1.0)
    pub memory_usage_threshold: f64,
    /// Error rate threshold (0.0-1.0)
    pub error_rate_threshold: f64,
    /// Latency threshold in milliseconds
    pub latency_threshold_ms: f64,
    /// Queue depth threshold
    pub queue_depth_threshold: usize,
    /// GPU utilization threshold (0.0-1.0)
    pub gpu_utilization_threshold: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_threshold: 0.8,
            memory_usage_threshold: 0.9,
            error_rate_threshold: 0.05,
            latency_threshold_ms: 1000.0,
            queue_depth_threshold: 100,
            gpu_utilization_threshold: 0.95,
        }
    }
}

/// Metrics export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExportConfig {
    /// Export to Prometheus
    pub prometheus_enabled: bool,
    /// Export to InfluxDB
    pub influxdb_enabled: bool,
    /// Export to custom endpoints
    pub custom_endpoints: Vec<CustomEndpoint>,
    /// Export format
    pub format: MetricsFormat,
    /// Export interval in seconds
    pub export_interval_seconds: u64,
}

impl Default for MetricsExportConfig {
    fn default() -> Self {
        Self {
            prometheus_enabled: true,
            influxdb_enabled: false,
            custom_endpoints: Vec::new(),
            format: MetricsFormat::Prometheus,
            export_interval_seconds: 60,
        }
    }
}

/// Custom export endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomEndpoint {
    /// Endpoint name
    pub name: String,
    /// URL to export to
    pub url: String,
    /// Authentication header
    pub auth_header: Option<String>,
    /// Custom headers
    pub headers: HashMap<String, String>,
}

/// Metrics export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricsFormat {
    Prometheus,
    InfluxDB,
    OpenTelemetry,
    Json,
    Custom { format_name: String },
}

/// Custom metric types
#[derive(Debug, Clone)]
pub enum CustomMetric {
    /// Business metrics (revenue, usage, etc.)
    Business {
        name: String,
        value: f64,
        labels: HashMap<String, String>,
        timestamp: SystemTime,
    },
    /// Performance metrics (latency percentiles, throughput, etc.)
    Performance {
        name: String,
        value: f64,
        percentile: Option<f64>,
        labels: HashMap<String, String>,
        timestamp: SystemTime,
    },
    /// System metrics (CPU, memory, GPU, etc.)
    System {
        name: String,
        value: f64,
        metric_type: SystemMetricType,
        labels: HashMap<String, String>,
        timestamp: SystemTime,
    },
    /// Application metrics (model accuracy, inference quality, etc.)
    Application {
        name: String,
        value: f64,
        metric_type: ApplicationMetricType,
        labels: HashMap<String, String>,
        timestamp: SystemTime,
    },
    /// Custom user-defined metrics
    Custom {
        name: String,
        value: f64,
        metric_type: String,
        labels: HashMap<String, String>,
        timestamp: SystemTime,
    },
}

/// System metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemMetricType {
    CpuUsage,
    MemoryUsage,
    GpuUsage,
    GpuMemory,
    NetworkIO,
    DiskIO,
    Temperature,
    PowerConsumption,
}

/// Application metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApplicationMetricType {
    ModelAccuracy,
    InferenceQuality,
    CacheHitRate,
    BatchEfficiency,
    TokensPerSecond,
    SequenceLength,
    ModelSize,
    LoadTime,
}

/// Real-time analytics data
#[derive(Debug, Clone)]
pub struct RealTimeAnalytics {
    /// Sliding window of metrics
    pub metrics_window: VecDeque<CustomMetric>,
    /// Window size in seconds
    pub window_size: Duration,
    /// Current averages
    pub current_averages: HashMap<String, f64>,
    /// Trend analysis
    pub trends: HashMap<String, Trend>,
    /// Anomaly detection results
    pub anomalies: Vec<Anomaly>,
}

/// Trend analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trend {
    /// Trend direction (positive, negative, stable)
    pub direction: TrendDirection,
    /// Trend strength (0.0-1.0)
    pub strength: f64,
    /// Trend duration
    pub duration: Duration,
    /// Confidence level (0.0-1.0)
    pub confidence: f64,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Metric name
    pub metric_name: String,
    /// Anomaly value
    pub value: f64,
    /// Expected value
    pub expected_value: f64,
    /// Deviation from expected
    pub deviation: f64,
    /// Anomaly severity
    pub severity: AnomalySeverity,
    /// Detection timestamp
    pub timestamp: SystemTime,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Anomaly types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    Spike,
    Drop,
    Drift,
    Oscillation,
    Flatline,
}

/// Performance profiling data
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Function call traces
    pub call_traces: Vec<CallTrace>,
    /// Memory allocation patterns
    pub memory_patterns: Vec<MemoryAllocation>,
    /// Hot spots (most time-consuming operations)
    pub hot_spots: Vec<HotSpot>,
    /// Bottleneck analysis
    pub bottlenecks: Vec<Bottleneck>,
}

/// Function call trace
#[derive(Debug, Clone)]
pub struct CallTrace {
    /// Function name
    pub function_name: String,
    /// Call duration
    pub duration: Duration,
    /// Call depth
    pub depth: usize,
    /// Thread ID
    pub thread_id: u64,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Memory allocation pattern
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    /// Allocation size
    pub size: usize,
    /// Allocation location
    pub location: String,
    /// Allocation type
    pub allocation_type: AllocationType,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Memory allocation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationType {
    Stack,
    Heap,
    Gpu,
    SharedMemory,
}

/// Performance hot spot
#[derive(Debug, Clone)]
pub struct HotSpot {
    /// Operation name
    pub operation: String,
    /// Time spent (percentage of total)
    pub time_percentage: f64,
    /// Call count
    pub call_count: u64,
    /// Average duration per call
    pub avg_duration: Duration,
}

/// Performance bottleneck
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Bottleneck location
    pub location: String,
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Impact score (0.0-1.0)
    pub impact_score: f64,
    /// Suggested optimization
    pub optimization_suggestion: String,
}

/// Bottleneck types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CpuBound,
    MemoryBound,
    IoBound,
    NetworkBound,
    GpuBound,
    LockContention,
}

/// Main custom metrics collector
#[derive(Clone)]
pub struct CustomMetricsCollector {
    /// Configuration
    config: CustomMetricsConfig,
    /// Metrics storage
    metrics_storage: Arc<RwLock<HashMap<String, VecDeque<CustomMetric>>>>,
    /// Real-time analytics
    analytics: Arc<Mutex<RealTimeAnalytics>>,
    // 0.2.1: a `profiler: Arc<Mutex<PerformanceProfile>>` field lived here. It
    // was constructed with four empty vectors and never written to or read
    // again, so it could only ever have reported "no call traces, no hot spots,
    // no bottlenecks" -- an empty profile presented as a measured one. Nothing
    // in this crate traces calls or allocations, so the field is gone rather
    // than kept as permanently-empty state. `PerformanceProfile` itself stays:
    // it is a public type a real profiler can fill in.
    /// Prometheus metrics
    prometheus_metrics: Arc<PrometheusMetrics>,
    /// Collection statistics
    stats: Arc<CollectionStats>,
    /// Active metric IDs for tracking
    active_metrics: Arc<RwLock<HashSet<String>>>,
}

use std::collections::HashSet;

/// Prometheus metrics for custom system
struct PrometheusMetrics {
    /// Custom counter metrics
    custom_counters: RwLock<HashMap<String, IntCounter>>,
    /// Custom gauge metrics
    custom_gauges: RwLock<HashMap<String, Gauge>>,
    /// Custom histogram metrics
    custom_histograms: RwLock<HashMap<String, Histogram>>,
}

/// Collection statistics
#[derive(Debug, Default)]
pub struct CollectionStats {
    /// Total metrics collected
    pub total_metrics: AtomicU64,
    /// Metrics collection rate
    pub collection_rate: AtomicU64,
    /// Storage size
    pub storage_size_bytes: AtomicU64,
    /// Alert count
    pub alert_count: AtomicU64,
    /// Anomaly count
    pub anomaly_count: AtomicU64,
    /// Export count
    pub export_count: AtomicU64,
}

impl CustomMetricsCollector {
    /// Create a new custom metrics collector
    pub fn new(config: CustomMetricsConfig) -> Result<Self> {
        let analytics = RealTimeAnalytics {
            metrics_window: VecDeque::new(),
            window_size: Duration::from_secs(300), // 5 minutes
            current_averages: HashMap::new(),
            trends: HashMap::new(),
            anomalies: Vec::new(),
        };

        Ok(Self {
            config,
            metrics_storage: Arc::new(RwLock::new(HashMap::new())),
            analytics: Arc::new(Mutex::new(analytics)),
            prometheus_metrics: Arc::new(PrometheusMetrics::new()),
            stats: Arc::new(CollectionStats::default()),
            active_metrics: Arc::new(RwLock::new(HashSet::new())),
        })
    }

    /// Start the metrics collection service
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Start collection task
        self.start_collection_task().await?;

        // Start analytics task
        if self.config.enable_real_time_analytics {
            self.start_analytics_task().await?;
        }

        // Start export task
        self.start_export_task().await?;

        // Start cleanup task
        self.start_cleanup_task().await?;

        Ok(())
    }

    /// Collect a custom metric
    pub async fn collect_metric(&self, metric: CustomMetric) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let metric_name = self.get_metric_name(&metric);

        // Store metric
        {
            let mut storage = self.metrics_storage.write().await;
            let metric_series = storage.entry(metric_name.clone()).or_insert_with(VecDeque::new);
            metric_series.push_back(metric.clone());

            // Limit series size
            while metric_series.len() > 1000 {
                metric_series.pop_front();
            }
        }

        // Update Prometheus metrics
        self.update_prometheus_metrics(&metric).await?;

        // Update real-time analytics
        if self.config.enable_real_time_analytics {
            let mut analytics = self.analytics.lock().await;
            analytics.metrics_window.push_back(metric);

            // Limit window size
            let max_size = (self.config.collection_interval_seconds * 60) as usize; // 1 hour of data
            while analytics.metrics_window.len() > max_size {
                analytics.metrics_window.pop_front();
            }
        }

        // Update statistics
        self.stats.total_metrics.fetch_add(1, Ordering::Relaxed);

        // Track active metrics
        self.active_metrics.write().await.insert(metric_name);

        Ok(())
    }

    /// Collect business metric
    pub async fn collect_business_metric(
        &self,
        name: &str,
        value: f64,
        labels: HashMap<String, String>,
    ) -> Result<()> {
        if !self.config.enable_business_metrics {
            return Ok(());
        }

        let metric = CustomMetric::Business {
            name: name.to_string(),
            value,
            labels,
            timestamp: SystemTime::now(),
        };

        self.collect_metric(metric).await
    }

    /// Collect performance metric
    pub async fn collect_performance_metric(
        &self,
        name: &str,
        value: f64,
        percentile: Option<f64>,
        labels: HashMap<String, String>,
    ) -> Result<()> {
        if !self.config.enable_performance_profiling {
            return Ok(());
        }

        let metric = CustomMetric::Performance {
            name: name.to_string(),
            value,
            percentile,
            labels,
            timestamp: SystemTime::now(),
        };

        self.collect_metric(metric).await
    }

    /// Collect system metric
    pub async fn collect_system_metric(
        &self,
        name: &str,
        value: f64,
        metric_type: SystemMetricType,
        labels: HashMap<String, String>,
    ) -> Result<()> {
        let metric = CustomMetric::System {
            name: name.to_string(),
            value,
            metric_type,
            labels,
            timestamp: SystemTime::now(),
        };

        self.collect_metric(metric).await
    }

    /// Perform real-time analytics
    pub async fn analyze_metrics(&self) -> Result<AnalyticsResult> {
        let mut analytics = self.analytics.lock().await;

        // Calculate current averages
        self.calculate_averages(&mut analytics).await?;

        // Perform trend analysis
        self.analyze_trends(&mut analytics).await?;

        // Detect anomalies
        self.detect_anomalies(&mut analytics).await?;

        // Generate insights
        let insights = self.generate_insights(&analytics).await?;

        Ok(AnalyticsResult {
            averages: analytics.current_averages.clone(),
            trends: analytics.trends.clone(),
            anomalies: analytics.anomalies.clone(),
            insights,
        })
    }

    /// Get metrics summary
    pub async fn get_metrics_summary(&self) -> MetricsSummary {
        let storage = self.metrics_storage.read().await;
        let analytics = self.analytics.lock().await;

        MetricsSummary {
            total_metrics: self.stats.total_metrics.load(Ordering::Relaxed),
            active_metric_series: storage.len(),
            collection_rate: self.stats.collection_rate.load(Ordering::Relaxed),
            storage_size_bytes: self.stats.storage_size_bytes.load(Ordering::Relaxed),
            alert_count: self.stats.alert_count.load(Ordering::Relaxed),
            anomaly_count: analytics.anomalies.len() as u64,
            recent_trends: analytics.trends.len() as u64,
        }
    }

    /// Export metrics to configured endpoints
    pub async fn export_metrics(&self) -> Result<()> {
        if self.config.export_config.prometheus_enabled {
            self.export_to_prometheus().await?;
        }

        if self.config.export_config.influxdb_enabled {
            self.export_to_influxdb().await?;
        }

        for endpoint in &self.config.export_config.custom_endpoints {
            self.export_to_custom_endpoint(endpoint).await?;
        }

        self.stats.export_count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    // Private helper methods

    async fn start_collection_task(&self) -> Result<()> {
        let collector = self.clone();
        let interval = Duration::from_secs(collector.config.collection_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // Collect system metrics
                if let Err(e) = collector.collect_system_metrics().await {
                    error!("Failed to collect system metrics: {}", e);
                }

                // Update collection rate
                collector.stats.collection_rate.store(
                    collector.stats.total_metrics.load(Ordering::Relaxed),
                    Ordering::Relaxed,
                );
            }
        });

        Ok(())
    }

    async fn start_analytics_task(&self) -> Result<()> {
        let collector = self.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                if let Err(e) = collector.analyze_metrics().await {
                    error!("Analytics failed: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn start_export_task(&self) -> Result<()> {
        let collector = self.clone();
        let interval = Duration::from_secs(collector.config.export_config.export_interval_seconds);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                if let Err(e) = collector.export_metrics().await {
                    error!("Metrics export failed: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn start_cleanup_task(&self) -> Result<()> {
        let collector = self.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600)); // 1 hour

            loop {
                interval.tick().await;

                if let Err(e) = collector.cleanup_old_metrics().await {
                    error!("Metrics cleanup failed: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Take one host measurement and publish the readings it actually carries.
    ///
    /// ## Changed in 0.2.1
    ///
    /// `get_cpu_usage`, `get_memory_usage` and `get_gpu_usage` returned the
    /// constants `0.5`, `0.7` and `0.8`. With the default configuration
    /// (`enabled: true`, a ten-second interval) those three numbers were
    /// published as measurements of the running host every ten seconds, and
    /// [`Self::detect_anomalies`] compared them against the alert thresholds as
    /// if they had been sampled. CPU and memory are now read from `sysinfo`
    /// through [`crate::server::system_stats`]; GPU utilization is read from
    /// the NVIDIA driver and *omitted* when no driver answers, because there is
    /// no portable source for it.
    async fn collect_system_metrics(&self) -> Result<()> {
        // One snapshot per tick: `measure_host_async` performs the two CPU
        // samples `sysinfo` requires (separated by
        // `MINIMUM_CPU_UPDATE_INTERVAL`) off the runtime, so both readings
        // below come from the same measurement rather than from two.
        let host = crate::server::system_stats::measure_host_async().await;

        // `measure_host_async` substitutes a zeroed snapshot when its blocking
        // task fails. A host that reports zero bytes of memory in total is that
        // failure, not a machine without RAM -- publishing it would put an
        // "idle CPU, empty memory" sample into the stream.
        let Some(cpu_usage) = Self::cpu_usage_fraction(&host) else {
            warn!("host measurement unavailable this tick; no system metric collected");
            return Ok(());
        };

        self.collect_system_metric(
            "cpu_usage",
            cpu_usage,
            SystemMetricType::CpuUsage,
            HashMap::new(),
        )
        .await?;

        if let Some(memory_usage) = Self::memory_usage_fraction(&host) {
            self.collect_system_metric(
                "memory_usage",
                memory_usage,
                SystemMetricType::MemoryUsage,
                HashMap::new(),
            )
            .await?;
        }

        // Collect GPU metrics only when a driver actually reports one.
        match self.get_gpu_usage().await {
            Ok(gpu_usage) => {
                self.collect_system_metric(
                    "gpu_usage",
                    gpu_usage,
                    SystemMetricType::GpuUsage,
                    HashMap::new(),
                )
                .await?;
            },
            Err(e) => debug!("GPU utilization not collected: {}", e),
        }

        Ok(())
    }

    /// Mean CPU utilization of `host`, as a fraction of one.
    ///
    /// `sysinfo` reports percent (0-100) while [`AlertThresholds`] documents
    /// its CPU threshold as `0.0-1.0` and [`Self::detect_anomalies`] compares
    /// against it directly, so the reading is converted here rather than at the
    /// comparison. `None` when the snapshot is the zeroed
    /// measurement-unavailable fallback.
    fn cpu_usage_fraction(host: &HostSnapshot) -> Option<f64> {
        if host.total_memory_bytes == 0 {
            return None;
        }
        Some((host.cpu_percent / 100.0).clamp(0.0, 1.0))
    }

    /// System memory in use on `host`, as a fraction of one.
    ///
    /// `None` when the platform reports no total memory, which is the
    /// measurement-unavailable fallback rather than a real reading.
    fn memory_usage_fraction(host: &HostSnapshot) -> Option<f64> {
        if host.total_memory_bytes == 0 {
            return None;
        }
        Some((host.memory_percent / 100.0).clamp(0.0, 1.0))
    }

    /// Utilization of GPU 0, as a fraction of one, as reported by the driver.
    ///
    /// There is no portable GPU utilization source: this reads `nvidia-smi`
    /// through [`GpuResourceManager::device_telemetry`], the same path the GPU
    /// manager uses for its own telemetry.
    ///
    /// # Errors
    ///
    /// Returns an error naming the missing source when no NVIDIA driver
    /// answers, or when the driver answers without a utilization reading. The
    /// caller omits the `gpu_usage` metric in that case; it is never defaulted.
    ///
    /// [`GpuResourceManager::device_telemetry`]:
    ///     crate::resource_management::gpu_manager::manager::GpuResourceManager::device_telemetry
    async fn get_gpu_usage(&self) -> Result<f64> {
        let sample =
            crate::resource_management::gpu_manager::manager::GpuResourceManager::device_telemetry(
                0,
            )
            .await
            .map_err(|e| anyhow::anyhow!("GPU telemetry query failed: {}", e))?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no NVIDIA driver reported GPU 0; GPU utilization has no other source on \
                     this host"
                )
            })?;

        let utilization = sample.utilization_percent.ok_or_else(|| {
            anyhow::anyhow!("the driver reported GPU 0 without a utilization reading")
        })?;

        Ok((utilization as f64 / 100.0).clamp(0.0, 1.0))
    }

    fn get_metric_name(&self, metric: &CustomMetric) -> String {
        match metric {
            CustomMetric::Business { name, .. } => format!("business_{}", name),
            CustomMetric::Performance { name, .. } => format!("performance_{}", name),
            CustomMetric::System { name, .. } => format!("system_{}", name),
            CustomMetric::Application { name, .. } => format!("application_{}", name),
            CustomMetric::Custom { name, .. } => format!("custom_{}", name),
        }
    }

    /// Record `metric` into this collector's Prometheus registry.
    ///
    /// 0.2.1: this was `Ok(())` with the argument bound to `_metric`. Every
    /// metric passed through `record_metric` was silently discarded while the
    /// caller treated the call as a successful export, and the three
    /// `PrometheusMetrics` maps stayed permanently empty.
    ///
    /// # Errors
    ///
    /// Returns [`CustomMetricsError::ExportError`] when a Prometheus collector
    /// cannot be created for the metric's name (an invalid metric name, for
    /// instance).
    async fn update_prometheus_metrics(&self, metric: &CustomMetric) -> Result<()> {
        let name = self.get_metric_name(metric);
        let value = self.get_metric_value(metric);
        match metric {
            // Business metrics are monotonic totals, so they accumulate into a
            // counter. A negative delta cannot be represented by a counter and
            // is rejected rather than silently dropped or made positive.
            CustomMetric::Business { .. } => {
                if value < 0.0 {
                    return Err(CustomMetricsError::ExportError {
                        message: format!(
                            "business metric {name} carried a negative value ({value}); a \
                             Prometheus counter cannot decrease"
                        ),
                    }
                    .into());
                }
                let mut counters = self.prometheus_metrics.custom_counters.write().await;
                if !counters.contains_key(&name) {
                    let counter = IntCounter::new(name.clone(), format!("custom metric {name}"))
                        .map_err(|e| CustomMetricsError::ExportError {
                            message: format!("cannot create counter {name}: {e}"),
                        })?;
                    counters.insert(name.clone(), counter);
                }
                if let Some(counter) = counters.get(&name) {
                    counter.inc_by(value as u64);
                }
            },
            // Latency-style samples belong in a histogram so percentiles are
            // computed from the real distribution rather than the last value.
            CustomMetric::Performance { .. } => {
                let mut histograms = self.prometheus_metrics.custom_histograms.write().await;
                if !histograms.contains_key(&name) {
                    let opts = prometheus::HistogramOpts::new(
                        name.clone(),
                        format!("custom metric {name}"),
                    );
                    let histogram = Histogram::with_opts(opts).map_err(|e| {
                        CustomMetricsError::ExportError {
                            message: format!("cannot create histogram {name}: {e}"),
                        }
                    })?;
                    histograms.insert(name.clone(), histogram);
                }
                if let Some(histogram) = histograms.get(&name) {
                    histogram.observe(value);
                }
            },
            // Everything else is a point-in-time reading: a gauge.
            CustomMetric::System { .. }
            | CustomMetric::Application { .. }
            | CustomMetric::Custom { .. } => {
                let mut gauges = self.prometheus_metrics.custom_gauges.write().await;
                if !gauges.contains_key(&name) {
                    let gauge =
                        Gauge::new(name.clone(), format!("custom metric {name}")).map_err(|e| {
                            CustomMetricsError::ExportError {
                                message: format!("cannot create gauge {name}: {e}"),
                            }
                        })?;
                    gauges.insert(name.clone(), gauge);
                }
                if let Some(gauge) = gauges.get(&name) {
                    gauge.set(value);
                }
            },
        }
        Ok(())
    }

    /// Current value of a recorded gauge metric, if one exists under `name`.
    pub async fn prometheus_gauge_value(&self, name: &str) -> Option<f64> {
        let gauges = self.prometheus_metrics.custom_gauges.read().await;
        gauges.get(name).map(|gauge| gauge.get())
    }

    /// Number of observations recorded into a histogram metric, if any.
    pub async fn prometheus_histogram_count(&self, name: &str) -> Option<u64> {
        let histograms = self.prometheus_metrics.custom_histograms.read().await;
        histograms.get(name).map(|histogram| histogram.get_sample_count())
    }

    /// Current value of a recorded counter metric, if one exists under `name`.
    pub async fn prometheus_counter_value(&self, name: &str) -> Option<u64> {
        let counters = self.prometheus_metrics.custom_counters.read().await;
        counters.get(name).map(|counter| counter.get())
    }

    async fn calculate_averages(&self, analytics: &mut RealTimeAnalytics) -> Result<()> {
        let mut metric_sums: HashMap<String, f64> = HashMap::new();
        let mut metric_counts: HashMap<String, usize> = HashMap::new();

        for metric in &analytics.metrics_window {
            let name = self.get_metric_name(metric);
            let value = self.get_metric_value(metric);

            *metric_sums.entry(name.clone()).or_insert(0.0) += value;
            *metric_counts.entry(name).or_insert(0) += 1;
        }

        analytics.current_averages.clear();
        for (name, sum) in metric_sums {
            if let Some(&count) = metric_counts.get(&name) {
                analytics.current_averages.insert(name, sum / count as f64);
            }
        }

        Ok(())
    }

    /// Fit a trend to every metric series held in the analytics window.
    ///
    /// ## Changed in 0.2.1
    ///
    /// Every metric in `current_averages` was handed the same
    /// `TrendDirection::Stable`, `strength: 0.5`, `confidence: 0.8` and a
    /// five-minute `duration`, whatever its samples did -- a series that had
    /// doubled and one that had never moved produced byte-identical trends,
    /// and `AnalyticsResult::trends` is published to callers. Each series is
    /// now fitted by least squares over its own observed timestamps.
    async fn analyze_trends(&self, analytics: &mut RealTimeAnalytics) -> Result<()> {
        // Group the window into per-metric series of (seconds since that
        // series' first sample, value), in arrival order.
        let mut series: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
        let mut origins: HashMap<String, SystemTime> = HashMap::new();

        for metric in &analytics.metrics_window {
            let name = self.get_metric_name(metric);
            let timestamp = Self::metric_timestamp(metric);
            let origin = *origins.entry(name.clone()).or_insert(timestamp);
            let seconds = timestamp.duration_since(origin).map(|d| d.as_secs_f64()).unwrap_or(0.0);
            series.entry(name).or_default().push((seconds, self.get_metric_value(metric)));
        }

        // A metric whose window no longer supports a fit loses its trend rather
        // than keeping the last one computed for it.
        analytics.trends.clear();
        for (name, points) in series {
            if let Some(trend) = Self::fit_trend(&points) {
                analytics.trends.insert(name, trend);
            }
        }

        Ok(())
    }

    /// Timestamp carried by `metric`, whichever variant it is.
    fn metric_timestamp(metric: &CustomMetric) -> SystemTime {
        match metric {
            CustomMetric::Business { timestamp, .. }
            | CustomMetric::Performance { timestamp, .. }
            | CustomMetric::System { timestamp, .. }
            | CustomMetric::Application { timestamp, .. }
            | CustomMetric::Custom { timestamp, .. } => *timestamp,
        }
    }

    /// Coefficient of variation above which a series with no significant
    /// monotone component is reported as [`TrendDirection::Volatile`] rather
    /// than [`TrendDirection::Stable`].
    ///
    /// This is a classification boundary, not a measurement: it says how much
    /// relative movement counts as "volatile", and both branches describe the
    /// same measured spread.
    const VOLATILITY_CV_THRESHOLD: f64 = 0.1;

    /// Least-squares trend of one metric series, or `None` when the samples
    /// cannot support one.
    ///
    /// `strength` is the magnitude of the Pearson correlation between time and
    /// value (so 1.0 is a perfectly straight line and 0.0 is no linear
    /// relationship at all), `confidence` is `1 - p` for the two-sided t-test
    /// on that correlation, and `duration` is the span the samples actually
    /// cover. A direction is only reported when the correlation is significant
    /// at the conventional 5% level.
    ///
    /// Returns `None` for fewer than three samples or a zero-length observation
    /// span -- both are absences of evidence, not flat trends.
    fn fit_trend(points: &[(f64, f64)]) -> Option<Trend> {
        if points.len() < 3 {
            return None;
        }
        let first = points.first()?;
        let last = points.last()?;
        let span = last.0 - first.0;
        if span.is_nan() || span <= 0.0 {
            return None;
        }

        let n = points.len() as f64;
        let mean_x = points.iter().map(|(x, _)| *x).sum::<f64>() / n;
        let mean_y = points.iter().map(|(_, y)| *y).sum::<f64>() / n;

        let mut sxx = 0.0;
        let mut syy = 0.0;
        let mut sxy = 0.0;
        for (x, y) in points {
            let dx = x - mean_x;
            let dy = y - mean_y;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }

        let duration = Duration::from_secs_f64(span);

        // A series that never moved is a measured result: it is exactly stable,
        // and nothing about that statement is uncertain given these samples.
        if syy <= 0.0 || sxx <= 0.0 {
            return Some(Trend {
                direction: TrendDirection::Stable,
                strength: 0.0,
                duration,
                confidence: 1.0,
            });
        }

        let correlation = sxy / (sxx * syy).sqrt();
        let slope = sxy / sxx;
        let p_value = pearson_p_value(correlation, points.len());
        let confidence = (1.0 - p_value).clamp(0.0, 1.0);

        let direction = if p_value > 0.05 {
            // No monotone component the samples can distinguish from noise.
            let std_dev = (syy / n).sqrt();
            let coefficient_of_variation = std_dev / mean_y.abs().max(f64::EPSILON);
            if coefficient_of_variation > Self::VOLATILITY_CV_THRESHOLD {
                TrendDirection::Volatile
            } else {
                TrendDirection::Stable
            }
        } else if slope > 0.0 {
            TrendDirection::Increasing
        } else {
            TrendDirection::Decreasing
        };

        Some(Trend {
            direction,
            strength: correlation.abs().clamp(0.0, 1.0),
            duration,
            confidence,
        })
    }

    async fn detect_anomalies(&self, analytics: &mut RealTimeAnalytics) -> Result<()> {
        analytics.anomalies.clear();

        // Simple anomaly detection based on thresholds
        for (metric_name, &value) in &analytics.current_averages {
            if metric_name.contains("cpu")
                && value > self.config.alert_thresholds.cpu_usage_threshold
            {
                analytics.anomalies.push(Anomaly {
                    metric_name: metric_name.clone(),
                    value,
                    expected_value: self.config.alert_thresholds.cpu_usage_threshold,
                    deviation: value - self.config.alert_thresholds.cpu_usage_threshold,
                    severity: AnomalySeverity::High,
                    timestamp: SystemTime::now(),
                    anomaly_type: AnomalyType::Spike,
                });
            }
        }

        self.stats
            .anomaly_count
            .store(analytics.anomalies.len() as u64, Ordering::Relaxed);

        Ok(())
    }

    /// Statements read off the analytics that were just computed.
    ///
    /// ## Changed in 0.2.1
    ///
    /// This returned the same three sentences ("CPU usage is trending upward",
    /// "Memory pressure detected", "Batch efficiency can be improved") on every
    /// call, on an empty window as readily as on a loaded one, and
    /// `AnalyticsResult::insights` publishes them. Every line now names a
    /// metric and quotes the numbers it was derived from; a window with nothing
    /// to report yields an empty list.
    async fn generate_insights(&self, analytics: &RealTimeAnalytics) -> Result<Vec<String>> {
        let mut insights = Vec::new();

        let mut anomalies: Vec<&Anomaly> = analytics.anomalies.iter().collect();
        anomalies.sort_by(|a, b| a.metric_name.cmp(&b.metric_name));
        for anomaly in anomalies {
            insights.push(format!(
                "{} averaged {:.4} over the window, {:.4} above its configured threshold of {:.4}",
                anomaly.metric_name, anomaly.value, anomaly.deviation, anomaly.expected_value
            ));
        }

        let mut trends: Vec<(&String, &Trend)> = analytics
            .trends
            .iter()
            .filter(|(_, trend)| {
                matches!(
                    trend.direction,
                    TrendDirection::Increasing | TrendDirection::Decreasing
                )
            })
            .collect();
        trends.sort_by(|a, b| a.0.cmp(b.0));
        for (name, trend) in trends {
            let direction = match trend.direction {
                TrendDirection::Increasing => "rising",
                _ => "falling",
            };
            insights.push(format!(
                "{} is {} over the {:.0}s covered by the window (|r| = {:.2}, confidence {:.2})",
                name,
                direction,
                trend.duration.as_secs_f64(),
                trend.strength,
                trend.confidence
            ));
        }

        Ok(insights)
    }

    fn get_metric_value(&self, metric: &CustomMetric) -> f64 {
        match metric {
            CustomMetric::Business { value, .. } => *value,
            CustomMetric::Performance { value, .. } => *value,
            CustomMetric::System { value, .. } => *value,
            CustomMetric::Application { value, .. } => *value,
            CustomMetric::Custom { value, .. } => *value,
        }
    }

    /// The last recorded value of every metric series currently held.
    async fn current_metric_values(&self) -> HashMap<String, f64> {
        let storage = self.metrics_storage.read().await;
        storage
            .iter()
            .filter_map(|(name, metrics)| {
                metrics.back().map(|metric| (name.clone(), self.get_metric_value(metric)))
            })
            .collect()
    }

    /// Render `values` in `format`, returning the body and its content type.
    ///
    /// # Errors
    ///
    /// Returns [`CustomMetricsError::ExportError`] naming the format when this
    /// crate carries no encoder for it, rather than silently shipping some
    /// other encoding under that name.
    fn encode_metrics(
        values: &HashMap<String, f64>,
        format: &MetricsFormat,
        timestamp: SystemTime,
    ) -> Result<(String, &'static str)> {
        // Sorted so a body is reproducible across calls with the same values.
        let mut entries: Vec<(&String, &f64)> = values.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        match format {
            MetricsFormat::Prometheus => {
                let mut body = String::new();
                for (name, value) in entries {
                    body.push_str(&format!("# TYPE {name} gauge\n{name} {value}\n"));
                }
                Ok((body, "text/plain; version=0.0.4"))
            },
            MetricsFormat::InfluxDB => {
                let nanos = timestamp
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                let mut body = String::new();
                for (name, value) in entries {
                    body.push_str(&format!("{name} value={value} {nanos}\n"));
                }
                Ok((body, "text/plain; charset=utf-8"))
            },
            MetricsFormat::Json => {
                let body =
                    serde_json::to_string(values).map_err(|e| CustomMetricsError::ExportError {
                        message: format!("cannot serialize metrics as JSON: {e}"),
                    })?;
                Ok((body, "application/json"))
            },
            MetricsFormat::OpenTelemetry => Err(CustomMetricsError::ExportError {
                message: "this crate carries no OpenTelemetry encoder; choose Prometheus, \
                          InfluxDB or Json in `MetricsExportConfig::format`"
                    .to_string(),
            }
            .into()),
            MetricsFormat::Custom { format_name } => Err(CustomMetricsError::ExportError {
                message: format!("no encoder is registered for the custom format {format_name:?}"),
            }
            .into()),
        }
    }

    /// Prometheus is scraped, not pushed to.
    ///
    /// ## Changed in 0.2.1
    ///
    /// This was `Ok(())` under the comment "Export to Prometheus endpoint",
    /// which reported a successful export of a push that never happened. There
    /// is genuinely nothing to push: [`Self::update_prometheus_metrics`]
    /// records every collected metric into this collector's Prometheus
    /// collectors as it arrives, and those values are what a scrape reads
    /// (they are also readable directly through
    /// [`Self::prometheus_gauge_value`], [`Self::prometheus_counter_value`] and
    /// [`Self::prometheus_histogram_count`]). `MetricsExportConfig` carries no
    /// push-gateway URL, so no second destination exists either.
    async fn export_to_prometheus(&self) -> Result<()> {
        debug!(
            "Prometheus export is pull-based; {} metric series are recorded and awaiting scrape",
            self.active_metrics.read().await.len()
        );
        Ok(())
    }

    /// Push the current metric values to InfluxDB.
    ///
    /// # Errors
    ///
    /// Always returns [`CustomMetricsError::ExportError`]: `influxdb_enabled`
    /// is a bare flag and [`MetricsExportConfig`] carries no InfluxDB URL,
    /// organisation, bucket or token, so there is no destination to write to.
    /// The error names the configuration that would make the push possible.
    ///
    /// ## Changed in 0.2.1
    ///
    /// This was `Ok(())` under the comment "Export to InfluxDB": with
    /// `influxdb_enabled` set, the export task reported a successful write
    /// every interval while nothing left the process.
    async fn export_to_influxdb(&self) -> Result<()> {
        Err(CustomMetricsError::ExportError {
            message: "influxdb_enabled is set, but MetricsExportConfig carries no InfluxDB URL, \
                      organisation, bucket or token; add the server to `custom_endpoints` with \
                      `format: MetricsFormat::InfluxDB` to push line protocol to it"
                .to_string(),
        }
        .into())
    }

    /// POST the current metric values to `endpoint`.
    ///
    /// ## Changed in 0.2.1
    ///
    /// This was `Ok(())` with the endpoint bound to `_endpoint`, so every
    /// configured endpoint was reported as exported to while no request was
    /// ever made. The body is now encoded in the configured
    /// [`MetricsFormat`] and sent, and a non-success response is an error.
    ///
    /// # Errors
    ///
    /// Returns [`CustomMetricsError::ExportError`] when the body cannot be
    /// encoded, when the request cannot be sent, or when the endpoint answers
    /// with a non-success status.
    async fn export_to_custom_endpoint(&self, endpoint: &CustomEndpoint) -> Result<()> {
        let values = self.current_metric_values().await;
        let (body, content_type) = Self::encode_metrics(
            &values,
            &self.config.export_config.format,
            SystemTime::now(),
        )?;

        let mut request = reqwest::Client::new()
            .post(&endpoint.url)
            .header(reqwest::header::CONTENT_TYPE, content_type)
            .body(body);

        if let Some(auth_header) = &endpoint.auth_header {
            request = request.header(reqwest::header::AUTHORIZATION, auth_header);
        }
        for (key, value) in &endpoint.headers {
            request = request.header(key.as_str(), value.as_str());
        }

        let response = request.send().await.map_err(|e| CustomMetricsError::ExportError {
            message: format!(
                "cannot POST metrics to endpoint {} ({}): {e}",
                endpoint.name, endpoint.url
            ),
        })?;

        if !response.status().is_success() {
            return Err(CustomMetricsError::ExportError {
                message: format!(
                    "endpoint {} ({}) answered {}",
                    endpoint.name,
                    endpoint.url,
                    response.status()
                ),
            }
            .into());
        }

        Ok(())
    }

    async fn cleanup_old_metrics(&self) -> Result<()> {
        let retention_duration = Duration::from_secs(self.config.retention_period_hours * 3600);
        let cutoff_time = SystemTime::now() - retention_duration;

        let mut storage = self.metrics_storage.write().await;
        for (_, metrics) in storage.iter_mut() {
            metrics.retain(|metric| {
                let timestamp = match metric {
                    CustomMetric::Business { timestamp, .. } => *timestamp,
                    CustomMetric::Performance { timestamp, .. } => *timestamp,
                    CustomMetric::System { timestamp, .. } => *timestamp,
                    CustomMetric::Application { timestamp, .. } => *timestamp,
                    CustomMetric::Custom { timestamp, .. } => *timestamp,
                };
                timestamp > cutoff_time
            });
        }

        Ok(())
    }
}

impl PrometheusMetrics {
    fn new() -> Self {
        Self {
            custom_counters: RwLock::new(HashMap::new()),
            custom_gauges: RwLock::new(HashMap::new()),
            custom_histograms: RwLock::new(HashMap::new()),
        }
    }
}

/// Analytics result
#[derive(Debug, Clone)]
pub struct AnalyticsResult {
    pub averages: HashMap<String, f64>,
    pub trends: HashMap<String, Trend>,
    pub anomalies: Vec<Anomaly>,
    pub insights: Vec<String>,
}

/// Metrics summary
#[derive(Debug, Serialize)]
pub struct MetricsSummary {
    pub total_metrics: u64,
    pub active_metric_series: usize,
    pub collection_rate: u64,
    pub storage_size_bytes: u64,
    pub alert_count: u64,
    pub anomaly_count: u64,
    pub recent_trends: u64,
}

/// Custom metrics error types
#[derive(Debug, thiserror::Error)]
pub enum CustomMetricsError {
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Collection error: {message}")]
    CollectionError { message: String },

    #[error("Export error: {message}")]
    ExportError { message: String },

    #[error("Analytics error: {message}")]
    AnalyticsError { message: String },

    #[error("Storage error: {message}")]
    StorageError { message: String },
}

#[cfg(test)]
#[path = "custom_metrics_tests.rs"]
mod tests;
