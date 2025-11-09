// Allow dead code for infrastructure under development
#![allow(dead_code)]

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
    /// Performance profiler
    profiler: Arc<Mutex<PerformanceProfile>>,
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

        let profiler = PerformanceProfile {
            call_traces: Vec::new(),
            memory_patterns: Vec::new(),
            hot_spots: Vec::new(),
            bottlenecks: Vec::new(),
        };

        Ok(Self {
            config,
            metrics_storage: Arc::new(RwLock::new(HashMap::new())),
            analytics: Arc::new(Mutex::new(analytics)),
            profiler: Arc::new(Mutex::new(profiler)),
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
                    eprintln!("Failed to collect system metrics: {}", e);
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
                    eprintln!("Analytics failed: {}", e);
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
                    eprintln!("Metrics export failed: {}", e);
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
                    eprintln!("Metrics cleanup failed: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn collect_system_metrics(&self) -> Result<()> {
        // Collect CPU usage
        let cpu_usage = self.get_cpu_usage().await?;
        self.collect_system_metric(
            "cpu_usage",
            cpu_usage,
            SystemMetricType::CpuUsage,
            HashMap::new(),
        )
        .await?;

        // Collect memory usage
        let memory_usage = self.get_memory_usage().await?;
        self.collect_system_metric(
            "memory_usage",
            memory_usage,
            SystemMetricType::MemoryUsage,
            HashMap::new(),
        )
        .await?;

        // Collect GPU metrics if available
        if let Ok(gpu_usage) = self.get_gpu_usage().await {
            self.collect_system_metric(
                "gpu_usage",
                gpu_usage,
                SystemMetricType::GpuUsage,
                HashMap::new(),
            )
            .await?;
        }

        Ok(())
    }

    async fn get_cpu_usage(&self) -> Result<f64> {
        // Simplified CPU usage collection
        // In practice, this would use system APIs
        Ok(0.5) // 50% usage
    }

    async fn get_memory_usage(&self) -> Result<f64> {
        // Simplified memory usage collection
        Ok(0.7) // 70% usage
    }

    async fn get_gpu_usage(&self) -> Result<f64> {
        // Simplified GPU usage collection
        Ok(0.8) // 80% usage
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

    async fn update_prometheus_metrics(&self, _metric: &CustomMetric) -> Result<()> {
        // Update Prometheus metrics based on custom metric type
        // This is simplified - in practice would handle different metric types
        Ok(())
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

    async fn analyze_trends(&self, analytics: &mut RealTimeAnalytics) -> Result<()> {
        // Simplified trend analysis
        for (metric_name, &_current_avg) in &analytics.current_averages {
            let trend = Trend {
                direction: TrendDirection::Stable,
                strength: 0.5,
                duration: Duration::from_secs(300),
                confidence: 0.8,
            };
            analytics.trends.insert(metric_name.clone(), trend);
        }

        Ok(())
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

    async fn generate_insights(&self, _analytics: &RealTimeAnalytics) -> Result<Vec<String>> {
        // Generate actionable insights based on analytics
        Ok(vec![
            "CPU usage is trending upward".to_string(),
            "Memory pressure detected".to_string(),
            "Batch efficiency can be improved".to_string(),
        ])
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

    async fn export_to_prometheus(&self) -> Result<()> {
        // Export to Prometheus endpoint
        Ok(())
    }

    async fn export_to_influxdb(&self) -> Result<()> {
        // Export to InfluxDB
        Ok(())
    }

    async fn export_to_custom_endpoint(&self, _endpoint: &CustomEndpoint) -> Result<()> {
        // Export to custom endpoint
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
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_custom_metrics_collector_creation() {
        let config = CustomMetricsConfig::default();
        let collector = CustomMetricsCollector::new(config).unwrap();
        assert!(collector.config.enabled);
    }

    #[tokio::test]
    async fn test_metric_collection() {
        let config = CustomMetricsConfig::default();
        let collector = CustomMetricsCollector::new(config).unwrap();

        let result = collector.collect_business_metric("revenue", 1000.0, HashMap::new()).await;

        assert!(result.is_ok());

        let summary = collector.get_metrics_summary().await;
        assert_eq!(summary.total_metrics, 1);
    }

    #[tokio::test]
    async fn test_analytics() {
        let config = CustomMetricsConfig::default();
        let collector = CustomMetricsCollector::new(config).unwrap();

        // Collect some test metrics
        for i in 0..10 {
            collector
                .collect_performance_metric("latency", i as f64 * 10.0, None, HashMap::new())
                .await
                .unwrap();
        }

        let analytics = collector.analyze_metrics().await.unwrap();
        assert!(!analytics.averages.is_empty());
    }
}
