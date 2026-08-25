//! Resource management statistics and performance tracking.
//!
//! This module provides comprehensive statistics collection, performance tracking,
//! analytics, and reporting capabilities for the resource management system.

use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::{debug, info};

use super::types::{
    AlertThreshold, PerformanceBaseline, ResourceStatistics, ResourceUtilizationMetrics,
    SystemPerformanceSnapshot,
};

/// Comprehensive statistics collection and analysis system
pub struct StatisticsCollector {
    /// Performance snapshots
    performance_snapshots: Arc<Mutex<Vec<SystemPerformanceSnapshot>>>,
    /// Resource utilization history
    utilization_history: Arc<Mutex<Vec<ResourceUtilizationSnapshot>>>,
    /// Performance baselines
    performance_baselines: Arc<Mutex<HashMap<String, PerformanceBaseline>>>,
    /// Statistics configuration
    config: Arc<Mutex<StatisticsConfig>>,
    /// Analytics engine
    analytics_engine: Arc<AnalyticsEngine>,
    /// Report generator
    report_generator: Arc<ReportGenerator>,
    /// Metrics aggregator
    metrics_aggregator: Arc<MetricsAggregator>,
}

/// Resource utilization snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationSnapshot {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Port utilization
    pub port_utilization: ResourceUtilizationMetrics,
    /// Directory utilization
    pub directory_utilization: ResourceUtilizationMetrics,
    /// GPU utilization
    pub gpu_utilization: ResourceUtilizationMetrics,
    /// Database utilization
    pub database_utilization: ResourceUtilizationMetrics,
    /// Custom resource utilization
    pub custom_resource_utilization: HashMap<String, ResourceUtilizationMetrics>,
    /// System-wide metrics
    pub system_metrics: SystemMetrics,
}

/// System-wide performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Memory utilization percentage
    pub memory_utilization: f32,
    /// Network throughput (bytes/sec)
    pub network_throughput: u64,
    /// Disk I/O rate (operations/sec)
    pub disk_io_rate: u64,
    /// Active processes count
    pub active_processes: usize,
    /// System load average
    pub load_average: f32,
}

/// Statistics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsConfig {
    /// Collection interval
    pub collection_interval: Duration,
    /// Retention period for snapshots
    pub retention_period: Duration,
    /// Maximum snapshots to retain
    pub max_snapshots: usize,
    /// Performance baseline update interval
    pub baseline_update_interval: Duration,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, AlertThreshold>,
    /// Aggregation settings
    pub aggregation_settings: AggregationSettings,
}

/// Aggregation settings for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationSettings {
    /// Aggregation window size
    pub window_size: Duration,
    /// Aggregation methods to use
    pub methods: Vec<AggregationMethod>,
    /// Percentiles to calculate
    pub percentiles: Vec<f32>,
    /// Rolling window size
    pub rolling_window_size: usize,
}

/// Aggregation methods
///
/// Note: Cannot derive Hash and Eq due to Percentile(f32) variant (f32 doesn't implement Hash/Eq)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AggregationMethod {
    /// Mean/average
    Mean,
    /// Median
    Median,
    /// Minimum
    Minimum,
    /// Maximum
    Maximum,
    /// Standard deviation
    StandardDeviation,
    /// Percentile
    Percentile(f32),
    /// Sum
    Sum,
    /// Count
    Count,
}

/// Analytics engine for performance analysis
pub struct AnalyticsEngine {
    /// Anomaly detector
    anomaly_detector: AnomalyDetector,
    /// Performance predictor
    performance_predictor: PerformancePredictor,
    /// Bottleneck analyzer
    bottleneck_analyzer: BottleneckAnalyzer,
}

/// Report generation system
pub struct ReportGenerator {
    /// Report templates
    report_templates: HashMap<String, ReportTemplate>,
    /// Report history
    report_history: Arc<Mutex<Vec<GeneratedReport>>>,
}

/// Metrics aggregation system
pub struct MetricsAggregator {
    /// Aggregated metrics
    aggregated_metrics: Arc<Mutex<HashMap<String, AggregatedMetric>>>,
    /// Which aggregation methods and percentiles to compute, and how large a
    /// rolling window of history to compute them over. `StatisticsCollector::new`
    /// passes this in from `StatisticsConfig::aggregation_settings`; it used
    /// to be accepted and silently discarded (`fn new(_config: ...)`).
    settings: AggregationSettings,
}

/// Trend analysis system
pub struct TrendAnalyzer {}

/// Anomaly detection system
pub struct AnomalyDetector {}

/// Performance prediction system
pub struct PerformancePredictor {}

/// Bottleneck analysis system
pub struct BottleneckAnalyzer {}

/// Aggregated metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetric {
    /// Metric name
    pub metric_name: String,
    /// Aggregation period
    pub period: Duration,
    /// Values by aggregation method (keyed by method name string due to AggregationMethod containing f32)
    pub values: HashMap<String, f64>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Sample count
    pub sample_count: usize,
}

/// Report template
#[derive(Debug, Clone)]
pub struct ReportTemplate {
    /// Template name
    pub name: String,
    /// Sections to include
    pub sections: Vec<ReportSection>,
    /// Output format
    pub format: ReportFormat,
    /// Generation parameters
    pub parameters: HashMap<String, String>,
}

/// Generated report
#[derive(Debug, Clone)]
pub struct GeneratedReport {
    /// Report ID
    pub report_id: String,
    /// Template used
    pub template_name: String,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Report content
    pub content: String,
    /// Report metadata
    pub metadata: HashMap<String, String>,
}

/// Report section types
#[derive(Debug, Clone)]
pub enum ReportSection {
    /// Executive summary
    ExecutiveSummary,
    /// Resource utilization
    ResourceUtilization,
    /// Performance trends
    PerformanceTrends,
    /// Anomaly detection
    AnomalyDetection,
    /// Bottleneck analysis
    BottleneckAnalysis,
    /// Recommendations
    Recommendations,
    /// Raw data
    RawData,
}

/// Report output formats
#[derive(Debug, Clone)]
pub enum ReportFormat {
    /// Plain text
    Text,
    /// Markdown
    Markdown,
    /// HTML
    Html,
    /// JSON
    Json,
    /// CSV
    Csv,
    /// PDF
    Pdf,
}

/// Anomaly detection model
#[derive(Debug, Clone)]
pub struct AnomalyDetectionModel {
    /// Model type
    pub model_type: AnomalyModelType,
    /// Sensitivity threshold
    pub sensitivity: f32,
    /// Training data window
    pub training_window: Duration,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
}

/// Anomaly model types
#[derive(Debug, Clone)]
pub enum AnomalyModelType {
    /// Statistical outlier detection
    StatisticalOutlier,
    /// Moving average deviation
    MovingAverageDeviation,
    /// Seasonal decomposition
    SeasonalDecomposition,
    /// Isolation forest
    IsolationForest,
    /// One-class SVM
    OneClassSVM,
}

/// Performance anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnomaly {
    /// Anomaly ID
    pub anomaly_id: String,
    /// Resource type
    pub resource_type: String,
    /// Metric name
    pub metric_name: String,
    /// Detected timestamp
    pub detected_at: DateTime<Utc>,
    /// Anomaly score
    pub anomaly_score: f32,
    /// Expected value
    pub expected_value: f64,
    /// Actual value
    pub actual_value: f64,
    /// Anomaly description
    pub description: String,
    /// Severity level
    pub severity: AnomalySeverity,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Prediction model
#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Model type
    pub model_type: PredictionModelType,
    /// Prediction horizon
    pub horizon: Duration,
    /// Model accuracy
    pub accuracy: f32,
    /// Training data window
    pub training_window: Duration,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
}

/// Prediction model types
#[derive(Debug, Clone)]
pub enum PredictionModelType {
    /// Linear regression
    LinearRegression,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// ARIMA model
    Arima,
    /// Neural network
    NeuralNetwork,
    /// Random forest
    RandomForest,
}

/// Performance prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePrediction {
    /// Prediction ID
    pub prediction_id: String,
    /// Resource type
    pub resource_type: String,
    /// Metric name
    pub metric_name: String,
    /// Prediction timestamp
    pub predicted_at: DateTime<Utc>,
    /// Prediction target time
    pub target_time: DateTime<Utc>,
    /// Predicted value
    pub predicted_value: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Prediction confidence
    pub confidence: f32,
}

/// Performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Bottleneck ID
    pub bottleneck_id: String,
    /// Resource type
    pub resource_type: String,
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Detected timestamp
    pub detected_at: DateTime<Utc>,
    /// Impact severity
    pub impact_severity: f32,
    /// Root cause analysis
    pub root_cause: String,
    /// Recommended actions
    pub recommendations: Vec<String>,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
}

/// Bottleneck types
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum BottleneckType {
    /// Resource contention
    ResourceContention,
    /// Memory pressure
    MemoryPressure,
    /// CPU throttling
    CpuThrottling,
    /// Network bandwidth
    NetworkBandwidth,
    /// Disk I/O
    DiskIo,
    /// Database connection pool
    DatabaseConnectionPool,
    /// Custom resource exhaustion
    CustomResourceExhaustion,
}

/// Bottleneck analysis configuration
#[derive(Debug, Clone)]
pub struct BottleneckAnalysisConfig {
    /// Analysis window
    pub analysis_window: Duration,
    /// Detection thresholds
    pub detection_thresholds: HashMap<BottleneckType, f32>,
    /// Minimum impact severity
    pub min_impact_severity: f32,
    /// Analysis frequency
    pub analysis_frequency: Duration,
}

impl StatisticsCollector {
    /// Create new statistics collector
    pub async fn new(config: StatisticsConfig) -> Result<Self> {
        let analytics_engine = Arc::new(AnalyticsEngine::new());
        let report_generator = Arc::new(ReportGenerator::new());
        let metrics_aggregator =
            Arc::new(MetricsAggregator::new(config.aggregation_settings.clone()));

        info!("Initialized statistics collector");

        Ok(Self {
            performance_snapshots: Arc::new(Mutex::new(Vec::new())),
            utilization_history: Arc::new(Mutex::new(Vec::new())),
            performance_baselines: Arc::new(Mutex::new(HashMap::new())),
            config: Arc::new(Mutex::new(config)),
            analytics_engine,
            report_generator,
            metrics_aggregator,
        })
    }

    /// Record performance snapshot
    pub async fn record_snapshot(&self, snapshot: SystemPerformanceSnapshot) -> Result<()> {
        let mut performance_snapshots = self.performance_snapshots.lock();
        performance_snapshots.push(snapshot);

        // Maintain maximum snapshot count
        let config = self.config.lock();
        if performance_snapshots.len() > config.max_snapshots {
            performance_snapshots.remove(0);
        }

        debug!("Recorded performance snapshot");
        Ok(())
    }

    /// Record utilization snapshot
    pub async fn record_utilization(&self, utilization: ResourceUtilizationSnapshot) -> Result<()> {
        let mut utilization_history = self.utilization_history.lock();
        utilization_history.push(utilization);

        // Aggregate metrics
        self.metrics_aggregator
            .aggregate_utilization_metrics(&utilization_history)
            .await?;

        debug!("Recorded utilization snapshot");
        Ok(())
    }

    /// Get performance statistics
    pub async fn get_performance_statistics(&self, period: Duration) -> Result<ResourceStatistics> {
        let performance_snapshots = self.performance_snapshots.lock();
        let cutoff_time = Utc::now() - ChronoDuration::from_std(period)?;

        let recent_snapshots: Vec<_> = performance_snapshots
            .iter()
            .filter(|snapshot| snapshot.timestamp >= cutoff_time)
            .collect();

        if recent_snapshots.is_empty() {
            return Ok(ResourceStatistics::default());
        }

        // Calculate statistics
        let total_snapshots = recent_snapshots.len() as f64;
        let average_cpu = recent_snapshots.iter().map(|s| s.cpu_utilization as f64).sum::<f64>()
            / total_snapshots;

        let average_memory =
            recent_snapshots.iter().map(|s| s.memory_utilization as f64).sum::<f64>()
                / total_snapshots;

        Ok(ResourceStatistics {
            total_allocated: recent_snapshots.len() as u64,
            active_resources: recent_snapshots
                .last()
                .map(|s| {
                    (s.port_stats.currently_allocated
                        + s.directory_stats.currently_allocated
                        + s.gpu_stats.currently_allocated
                        + s.database_stats.currently_active) as u32
                })
                .unwrap_or(0),
            peak_usage: recent_snapshots
                .iter()
                .map(|s| {
                    (s.port_stats.peak_usage
                        + s.directory_stats.peak_usage
                        + s.gpu_stats.peak_usage
                        + s.database_stats.peak_usage) as u64
                })
                .max()
                .unwrap_or(0),
            avg_lifetime: period / recent_snapshots.len().max(1) as u32,
            utilization_rate: average_cpu.max(average_memory) / 100.0,
            cpu_utilization: average_cpu,
            memory_utilization: average_memory,
            allocation_count: recent_snapshots.len() as u64,
            total_duration: period,
            // Mean of each snapshot's own measured `overall_efficiency` (itself
            // a real average of whichever subsystem occupancy signals had
            // recorded activity -- see `ResourceManagementSystem::get_performance_snapshot`),
            // not a constant.
            efficiency_score: recent_snapshots
                .iter()
                .map(|s| s.overall_efficiency as f64)
                .sum::<f64>()
                / total_snapshots,
        })
    }

    /// Generate performance report
    pub async fn generate_report(
        &self,
        template_name: &str,
        parameters: HashMap<String, String>,
    ) -> Result<String> {
        self.report_generator.generate_report(template_name, parameters).await
    }

    /// Detect performance anomalies
    pub async fn detect_anomalies(&self) -> Result<Vec<PerformanceAnomaly>> {
        self.analytics_engine.detect_anomalies(&self.performance_snapshots).await
    }

    /// Predict future performance
    pub async fn predict_performance(
        &self,
        metric_name: &str,
        horizon: Duration,
    ) -> Result<PerformancePrediction> {
        self.analytics_engine
            .predict_performance(metric_name, horizon, &self.performance_snapshots)
            .await
    }

    /// Analyze bottlenecks
    pub async fn analyze_bottlenecks(&self) -> Result<Vec<PerformanceBottleneck>> {
        self.analytics_engine.analyze_bottlenecks(&self.performance_snapshots).await
    }

    /// Update performance baseline
    pub async fn update_baseline(
        &self,
        metric_name: &str,
        baseline: PerformanceBaseline,
    ) -> Result<()> {
        let mut performance_baselines = self.performance_baselines.lock();
        performance_baselines.insert(metric_name.to_string(), baseline);

        info!("Updated performance baseline for metric: {}", metric_name);
        Ok(())
    }

    /// Get aggregated metrics
    pub async fn get_aggregated_metrics(
        &self,
        period: Duration,
    ) -> Result<HashMap<String, AggregatedMetric>> {
        self.metrics_aggregator.get_aggregated_metrics(period).await
    }
}

impl Default for AnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalyticsEngine {
    /// Create new analytics engine
    pub fn new() -> Self {
        Self {
            anomaly_detector: AnomalyDetector::new(),
            performance_predictor: PerformancePredictor::new(),
            bottleneck_analyzer: BottleneckAnalyzer::new(),
        }
    }

    /// Detect performance anomalies
    pub async fn detect_anomalies(
        &self,
        snapshots: &Arc<Mutex<Vec<SystemPerformanceSnapshot>>>,
    ) -> Result<Vec<PerformanceAnomaly>> {
        self.anomaly_detector.detect_anomalies(snapshots).await
    }

    /// Predict future performance
    pub async fn predict_performance(
        &self,
        metric_name: &str,
        horizon: Duration,
        snapshots: &Arc<Mutex<Vec<SystemPerformanceSnapshot>>>,
    ) -> Result<PerformancePrediction> {
        self.performance_predictor.predict(metric_name, horizon, snapshots).await
    }

    /// Analyze performance bottlenecks
    pub async fn analyze_bottlenecks(
        &self,
        snapshots: &Arc<Mutex<Vec<SystemPerformanceSnapshot>>>,
    ) -> Result<Vec<PerformanceBottleneck>> {
        self.bottleneck_analyzer.analyze_bottlenecks(snapshots).await
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ReportGenerator {
    /// Create new report generator
    pub fn new() -> Self {
        let mut report_templates = HashMap::new();

        // Add default templates
        report_templates.insert(
            "system_overview".to_string(),
            ReportTemplate {
                name: "System Overview".to_string(),
                sections: vec![
                    ReportSection::ExecutiveSummary,
                    ReportSection::ResourceUtilization,
                    ReportSection::PerformanceTrends,
                ],
                format: ReportFormat::Markdown,
                parameters: HashMap::new(),
            },
        );

        Self {
            report_templates,
            report_history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Generate report
    pub async fn generate_report(
        &self,
        template_name: &str,
        parameters: HashMap<String, String>,
    ) -> Result<String> {
        if let Some(template) = self.report_templates.get(template_name) {
            let report_content = self.render_template(template, &parameters).await?;

            let report = GeneratedReport {
                report_id: format!("report_{}", Utc::now().timestamp_millis()),
                template_name: template_name.to_string(),
                generated_at: Utc::now(),
                content: report_content.clone(),
                metadata: parameters,
            };

            let mut report_history = self.report_history.lock();
            report_history.push(report);

            Ok(report_content)
        } else {
            Err(anyhow::anyhow!(
                "Report template '{}' not found",
                template_name
            ))
        }
    }

    /// Render template
    async fn render_template(
        &self,
        template: &ReportTemplate,
        _parameters: &HashMap<String, String>,
    ) -> Result<String> {
        let mut content = String::new();

        content.push_str(&format!("# {}\n\n", template.name));
        content.push_str(&format!(
            "Generated at: {}\n\n",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        for section in &template.sections {
            match section {
                ReportSection::ExecutiveSummary => {
                    content.push_str("## Executive Summary\n\n");
                    content.push_str(
                        "Resource management system is operating within normal parameters.\n\n",
                    );
                },
                ReportSection::ResourceUtilization => {
                    content.push_str("## Resource Utilization\n\n");
                    content.push_str("- CPU: 65%\n- Memory: 78%\n- Network: 45%\n\n");
                },
                ReportSection::PerformanceTrends => {
                    content.push_str("## Performance Trends\n\n");
                    content.push_str("Performance has been stable over the last 24 hours.\n\n");
                },
                _ => {
                    content.push_str(&format!("## {:?}\n\n", section));
                    content.push_str("Section content would be generated here.\n\n");
                },
            }
        }

        Ok(content)
    }
}

impl MetricsAggregator {
    /// Create new metrics aggregator
    pub fn new(settings: AggregationSettings) -> Self {
        Self {
            aggregated_metrics: Arc::new(Mutex::new(HashMap::new())),
            settings,
        }
    }

    /// Aggregate utilization metrics.
    ///
    /// Computes every configured [`AggregationMethod`] (plus every configured
    /// percentile) over the most recent `settings.rolling_window_size`
    /// entries of `utilization_history`, for each series
    /// [`resource_utilization_series`] flattens out of it, and stores the
    /// result keyed by series name. Refuses only when there is nothing
    /// recorded at all; `record_utilization` always calls this with the entry
    /// it just pushed already included, so that case does not arise on the
    /// live path.
    pub async fn aggregate_utilization_metrics(
        &self,
        utilization_history: &Vec<ResourceUtilizationSnapshot>,
    ) -> Result<()> {
        if utilization_history.is_empty() {
            return Err(anyhow::anyhow!(
                "no utilization snapshots have been recorded; cannot aggregate"
            ));
        }
        let window = self.settings.rolling_window_size.max(1);
        let recent: Vec<&ResourceUtilizationSnapshot> =
            utilization_history.iter().rev().take(window).collect();
        let series = resource_utilization_series(&recent);
        let series_count = series.len();
        let mut aggregated = self.aggregated_metrics.lock();
        for (name, values) in series {
            let mut computed: HashMap<String, f64> = self
                .settings
                .methods
                .iter()
                .map(|method| {
                    (
                        aggregation_method_key(method),
                        compute_aggregation(&values, method),
                    )
                })
                .collect();
            for configured_percentile in &self.settings.percentiles {
                computed
                    .entry(format!("p{:.0}", configured_percentile))
                    .or_insert_with(|| percentile(&values, *configured_percentile as f64));
            }
            let sample_count = values.len();
            aggregated.insert(
                name.clone(),
                AggregatedMetric {
                    metric_name: name,
                    period: self.settings.window_size,
                    values: computed,
                    timestamp: Utc::now(),
                    sample_count,
                },
            );
        }
        debug!(
            "Aggregated utilization metrics into {} series from {} recent snapshot(s)",
            series_count,
            recent.len()
        );
        Ok(())
    }

    /// Get aggregated metrics computed within the last `period`.
    pub async fn get_aggregated_metrics(
        &self,
        period: Duration,
    ) -> Result<HashMap<String, AggregatedMetric>> {
        let cutoff_time = Utc::now() - ChronoDuration::from_std(period)?;
        let aggregated_metrics = self.aggregated_metrics.lock();
        Ok(aggregated_metrics
            .iter()
            .filter(|(_, metric)| metric.timestamp >= cutoff_time)
            .map(|(name, metric)| (name.clone(), metric.clone()))
            .collect())
    }
}

impl Default for TrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendAnalyzer {
    /// Create new trend analyzer
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyDetector {
    /// Create new anomaly detector
    pub fn new() -> Self {
        Self {}
    }

    /// Detect anomalies in performance data using a three-sigma rule.
    ///
    /// For each metric in [`KNOWN_METRICS`], computes the mean and sample
    /// standard deviation over every finite recorded reading, then flags any
    /// reading more than three standard deviations from that mean. A metric
    /// with fewer than two finite readings, or with zero variance (nothing
    /// has ever moved it, so nothing can be "3 sigma away" from anything),
    /// contributes no anomalies rather than a division by zero. Refuses only
    /// when no snapshots have been recorded at all.
    pub async fn detect_anomalies(
        &self,
        snapshots: &Arc<Mutex<Vec<SystemPerformanceSnapshot>>>,
    ) -> Result<Vec<PerformanceAnomaly>> {
        let guard = snapshots.lock();
        if guard.is_empty() {
            return Err(anyhow::anyhow!(
                "no performance snapshots have been recorded; cannot detect anomalies"
            ));
        }
        let mut anomalies = Vec::new();
        for &metric_name in KNOWN_METRICS {
            let series = extract_metric_series(&guard, metric_name);
            if series.len() < 2 {
                continue;
            }
            let values: Vec<f64> = series.iter().map(|(_, value)| *value).collect();
            let mean_value = mean(&values);
            let std_dev = sample_std_dev(&values, mean_value);
            if std_dev <= f64::EPSILON {
                // A perfectly constant series has no statistical outliers by
                // definition.
                continue;
            }
            for (timestamp, value) in &series {
                let z_score = (value - mean_value) / std_dev;
                if z_score.abs() <= 3.0 {
                    continue;
                }
                let severity = if z_score.abs() >= 6.0 {
                    AnomalySeverity::Critical
                } else if z_score.abs() >= 5.0 {
                    AnomalySeverity::High
                } else if z_score.abs() >= 4.0 {
                    AnomalySeverity::Medium
                } else {
                    AnomalySeverity::Low
                };
                anomalies.push(PerformanceAnomaly {
                    anomaly_id: format!("anomaly_{}_{}", metric_name, timestamp.timestamp_millis()),
                    resource_type: metric_resource_label(metric_name).to_string(),
                    metric_name: metric_name.to_string(),
                    detected_at: *timestamp,
                    anomaly_score: z_score.abs() as f32,
                    expected_value: mean_value,
                    actual_value: *value,
                    description: format!(
                        "{metric_name} deviates {z_score:.2} standard deviations from the \
                         recorded mean ({mean_value:.4}); observed {value:.4}"
                    ),
                    severity,
                });
            }
        }
        Ok(anomalies)
    }
}

impl Default for PerformancePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformancePredictor {
    /// Create new performance predictor
    pub fn new() -> Self {
        Self {}
    }

    /// Predict future performance with a simple linear trend fit.
    ///
    /// Fits ordinary least squares to every finite recorded reading of
    /// `metric_name` (elapsed seconds since the first reading vs. reading
    /// value) and extrapolates to `horizon` from now. The confidence interval
    /// is the fit's residual-based prediction interval (a 95% interval under
    /// a normal approximation of the residuals -- a fixed 1.96 multiplier
    /// rather than a t-distribution critical value, since this module has no
    /// inverse-t implementation), which widens both with residual scatter and
    /// with how far `horizon` extrapolates past the recorded data.
    /// `confidence` is the fit's R^2 (how much of the metric's variance the
    /// linear trend explains), zero when the recorded values have no
    /// variance to explain rather than a fabricated "perfect fit".
    ///
    /// Refuses when `metric_name` is not recognized, when no snapshots have
    /// been recorded, or when fewer than 2 finite readings of the metric
    /// exist (a trend needs at least two points).
    pub async fn predict(
        &self,
        metric_name: &str,
        horizon: Duration,
        snapshots: &Arc<Mutex<Vec<SystemPerformanceSnapshot>>>,
    ) -> Result<PerformancePrediction> {
        let series = {
            let guard = snapshots.lock();
            if guard.is_empty() {
                return Err(anyhow::anyhow!(
                    "no performance snapshots have been recorded; cannot predict '{metric_name}'"
                ));
            }
            collect_metric_series(&guard, metric_name)?
        };
        if series.len() < 2 {
            return Err(anyhow::anyhow!(
                "need at least 2 recorded, finite readings of '{metric_name}' to fit a trend, have {}",
                series.len()
            ));
        }
        let first_timestamp = series[0].0;
        let elapsed_seconds = |timestamp: DateTime<Utc>| -> f64 {
            (timestamp - first_timestamp).num_milliseconds() as f64 / 1000.0
        };
        let n = series.len() as f64;
        let xs: Vec<f64> =
            series.iter().map(|(timestamp, _)| elapsed_seconds(*timestamp)).collect();
        let ys: Vec<f64> = series.iter().map(|(_, value)| *value).collect();
        let mean_x = mean(&xs);
        let mean_y = mean(&ys);
        let sum_xx: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
        let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
        let slope = if sum_xx > f64::EPSILON { sum_xy / sum_xx } else { 0.0 };
        let intercept = mean_y - slope * mean_x;

        let target_time = Utc::now() + ChronoDuration::from_std(horizon)?;
        let x_target = elapsed_seconds(target_time);
        let predicted_value = intercept + slope * x_target;

        let sse: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| {
                let residual = y - (intercept + slope * x);
                residual * residual
            })
            .sum();
        let sst: f64 = ys.iter().map(|y| (y - mean_y).powi(2)).sum();

        // Degrees of freedom for the residual variance: n-2 for this
        // 2-parameter fit, floored at 1 so the minimum viable n=2 sample
        // (which fits exactly, 0 degrees of freedom) still yields a defined,
        // conservative variance estimate instead of a division by zero.
        let degrees_of_freedom = (n - 2.0).max(1.0);
        let residual_std_error = (sse / degrees_of_freedom).sqrt();
        let extrapolation_factor = if sum_xx > f64::EPSILON {
            1.0 + 1.0 / n + (x_target - mean_x).powi(2) / sum_xx
        } else {
            1.0 + 1.0 / n
        };
        let margin = 1.96 * residual_std_error * extrapolation_factor.sqrt();
        let confidence_interval = (predicted_value - margin, predicted_value + margin);

        // R^2 is undefined (not "perfect") when the recorded values never
        // varied at all: report zero confidence rather than fabricate
        // certainty from a flat series.
        let confidence =
            if sst > f64::EPSILON { (1.0 - sse / sst).clamp(0.0, 1.0) } else { 0.0 } as f32;

        Ok(PerformancePrediction {
            prediction_id: format!("pred_{}", Utc::now().timestamp_millis()),
            resource_type: metric_resource_label(metric_name).to_string(),
            metric_name: metric_name.to_string(),
            predicted_at: Utc::now(),
            target_time,
            predicted_value,
            confidence_interval,
            confidence,
        })
    }
}

impl Default for BottleneckAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl BottleneckAnalyzer {
    /// Create new bottleneck analyzer
    pub fn new() -> Self {
        Self {}
    }

    /// Analyze performance bottlenecks, ranked by measured utilization.
    ///
    /// Compares the most recent snapshot's utilization of each resource in
    /// [`KNOWN_METRICS`] (excluding `overall_efficiency`, a derived summary
    /// rather than a resource that can itself be exhausted) against
    /// [`BottleneckAnalysisConfig`]'s configured threshold for the matching
    /// [`BottleneckType`] (a resource with no dedicated threshold entry uses
    /// a conservative 0.85 default). A resource at or above its threshold
    /// becomes a bottleneck whose `impact_severity` is the measured
    /// utilization ratio itself; the result is sorted by that severity,
    /// worst first. Refuses only when no snapshot has ever been recorded --
    /// unlike anomaly detection this needs no history, a single reading is
    /// enough to compare against a static limit.
    pub async fn analyze_bottlenecks(
        &self,
        snapshots: &Arc<Mutex<Vec<SystemPerformanceSnapshot>>>,
    ) -> Result<Vec<PerformanceBottleneck>> {
        let guard = snapshots.lock();
        let Some(latest) = guard.last() else {
            return Err(anyhow::anyhow!(
                "no performance snapshots have been recorded; cannot analyze bottlenecks"
            ));
        };
        let thresholds = BottleneckAnalysisConfig::default().detection_thresholds;
        const DEFAULT_THRESHOLD: f32 = 0.85;
        let mut bottlenecks = Vec::new();
        for &metric_name in KNOWN_METRICS {
            if metric_name == "overall_efficiency" {
                continue;
            }
            let Some(ratio) =
                extract_metric_ratio(latest, metric_name).filter(|value| value.is_finite())
            else {
                continue;
            };
            let bottleneck_type = metric_bottleneck_type(metric_name);
            let threshold = thresholds.get(&bottleneck_type).copied().unwrap_or(DEFAULT_THRESHOLD);
            let impact_severity = ratio as f32;
            if impact_severity < threshold {
                continue;
            }
            let recommendations = bottleneck_recommendations(&bottleneck_type);
            let root_cause = format!(
                "{} utilization at {:.1}% exceeds the {:.1}% {:?} threshold",
                metric_resource_label(metric_name),
                impact_severity * 100.0,
                threshold * 100.0,
                bottleneck_type
            );
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_id: format!(
                    "bottleneck_{}_{}",
                    metric_name,
                    latest.timestamp.timestamp_millis()
                ),
                resource_type: metric_resource_label(metric_name).to_string(),
                bottleneck_type,
                detected_at: latest.timestamp,
                impact_severity,
                root_cause,
                recommendations,
                affected_metrics: vec![metric_name.to_string()],
            });
        }
        bottlenecks.sort_by(|a, b| {
            b.impact_severity
                .partial_cmp(&a.impact_severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(bottlenecks)
    }
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(60),
            retention_period: Duration::from_secs(86400 * 7), // 7 days
            max_snapshots: 10080,                             // 7 days at 1-minute intervals
            baseline_update_interval: Duration::from_secs(3600), // 1 hour
            alert_thresholds: HashMap::new(),
            aggregation_settings: AggregationSettings::default(),
        }
    }
}

impl Default for AggregationSettings {
    fn default() -> Self {
        Self {
            window_size: Duration::from_secs(300), // 5 minutes
            methods: vec![
                AggregationMethod::Mean,
                AggregationMethod::Minimum,
                AggregationMethod::Maximum,
                AggregationMethod::Percentile(95.0),
            ],
            percentiles: vec![50.0, 90.0, 95.0, 99.0],
            rolling_window_size: 100,
        }
    }
}

impl Default for BottleneckAnalysisConfig {
    fn default() -> Self {
        let mut detection_thresholds = HashMap::new();
        detection_thresholds.insert(BottleneckType::ResourceContention, 0.8);
        detection_thresholds.insert(BottleneckType::MemoryPressure, 0.85);
        detection_thresholds.insert(BottleneckType::CpuThrottling, 0.9);

        Self {
            analysis_window: Duration::from_secs(3600), // 1 hour
            detection_thresholds,
            min_impact_severity: 0.5,
            analysis_frequency: Duration::from_secs(300), // 5 minutes
        }
    }
}

// ================================
// Metric extraction and statistics helpers
//
// Shared by `PerformancePredictor`, `AnomalyDetector` and `BottleneckAnalyzer`
// below, all of which analyze the same recorded `SystemPerformanceSnapshot`
// history rather than fabricating their results.
// ================================

/// The metrics this module knows how to analyze from a
/// `SystemPerformanceSnapshot`.
const KNOWN_METRICS: &[&str] = &[
    "cpu_utilization",
    "memory_utilization",
    "gpu_utilization",
    "network_utilization",
    "disk_utilization",
    "overall_efficiency",
];

/// Extracts a named metric from a snapshot as a dimensionless ratio (not a
/// 0-100 percentage), so every metric this module analyzes shares one scale.
/// `SystemPerformanceSnapshot` itself mixes both conventions --
/// `cpu_utilization`/`memory_utilization`/`gpu_utilization` are recorded as
/// 0-100 percentages, `network_utilization`/`disk_utilization`/
/// `overall_efficiency` are already ratios (see
/// `resource_management::manager::ResourceManagementSystem::get_performance_snapshot`)
/// -- this is the single place that normalizes them. Returns `None` when
/// `metric_name` is not one of `KNOWN_METRICS`, or when this particular
/// snapshot has no reading for it (`gpu_utilization` on a snapshot taken
/// while no GPU was allocated).
fn extract_metric_ratio(snapshot: &SystemPerformanceSnapshot, metric_name: &str) -> Option<f64> {
    match metric_name {
        "cpu_utilization" => Some(snapshot.cpu_utilization as f64 / 100.0),
        "memory_utilization" => Some(snapshot.memory_utilization as f64 / 100.0),
        "gpu_utilization" => snapshot.gpu_utilization.map(|value| value as f64 / 100.0),
        "network_utilization" => Some(snapshot.network_utilization as f64),
        "disk_utilization" => Some(snapshot.disk_utilization as f64),
        "overall_efficiency" => Some(snapshot.overall_efficiency as f64),
        _ => None,
    }
}

/// Collects `(timestamp, value)` pairs for `metric_name` across `snapshots`,
/// unconditionally: `metric_name` is assumed already validated (this is used
/// by the loops that iterate `KNOWN_METRICS` themselves). Drops any
/// non-finite reading -- a sensor artifact must not poison a mean, standard
/// deviation or regression into `NaN`.
fn extract_metric_series(
    snapshots: &[SystemPerformanceSnapshot],
    metric_name: &str,
) -> Vec<(DateTime<Utc>, f64)> {
    let mut series: Vec<(DateTime<Utc>, f64)> = snapshots
        .iter()
        .filter_map(|snapshot| {
            extract_metric_ratio(snapshot, metric_name)
                .filter(|value| value.is_finite())
                .map(|value| (snapshot.timestamp, value))
        })
        .collect();
    series.sort_by_key(|(timestamp, _)| *timestamp);
    series
}

/// [`extract_metric_series`], but refuses an unrecognized `metric_name`
/// instead of silently returning an empty series -- for the entry points
/// that take a caller-supplied metric name.
fn collect_metric_series(
    snapshots: &[SystemPerformanceSnapshot],
    metric_name: &str,
) -> Result<Vec<(DateTime<Utc>, f64)>> {
    if !KNOWN_METRICS.contains(&metric_name) {
        return Err(anyhow::anyhow!(
            "unknown metric '{metric_name}'; supported metrics are: {}",
            KNOWN_METRICS.join(", ")
        ));
    }
    Ok(extract_metric_series(snapshots, metric_name))
}

/// A short, human-readable label for the resource a metric describes.
fn metric_resource_label(metric_name: &str) -> &'static str {
    match metric_name {
        "cpu_utilization" => "cpu",
        "memory_utilization" => "memory",
        "gpu_utilization" => "gpu",
        "network_utilization" => "network",
        "disk_utilization" => "disk",
        "overall_efficiency" => "system",
        _ => "unknown",
    }
}

/// The `BottleneckType` a metric's saturation corresponds to. `BottleneckType`
/// has no dedicated GPU variant, so a saturated GPU is reported as general
/// resource contention.
fn metric_bottleneck_type(metric_name: &str) -> BottleneckType {
    match metric_name {
        "cpu_utilization" => BottleneckType::CpuThrottling,
        "memory_utilization" => BottleneckType::MemoryPressure,
        "network_utilization" => BottleneckType::NetworkBandwidth,
        "disk_utilization" => BottleneckType::DiskIo,
        _ => BottleneckType::ResourceContention,
    }
}

/// Stock, honest advice for a saturated resource -- policy text, not a
/// measured value, exactly like `BottleneckAnalysisConfig`'s own threshold
/// constants.
fn bottleneck_recommendations(bottleneck_type: &BottleneckType) -> Vec<String> {
    match bottleneck_type {
        BottleneckType::CpuThrottling => {
            vec!["Scale out CPU-bound workloads or increase available CPU cores.".to_string()]
        },
        BottleneckType::MemoryPressure => {
            vec!["Increase available memory or reduce concurrent allocations.".to_string()]
        },
        BottleneckType::NetworkBandwidth => {
            vec!["Increase network capacity or reduce concurrent port allocations.".to_string()]
        },
        BottleneckType::DiskIo => {
            vec![
                "Increase temporary-directory capacity or reduce concurrent directory allocations."
                    .to_string(),
            ]
        },
        BottleneckType::ResourceContention => {
            vec!["Add GPU capacity or reduce concurrent GPU-bound workloads.".to_string()]
        },
        BottleneckType::DatabaseConnectionPool => {
            vec![
                "Increase the database connection pool size or reduce concurrent database usage."
                    .to_string(),
            ]
        },
        BottleneckType::CustomResourceExhaustion => {
            vec!["Review custom resource limits against current demand.".to_string()]
        },
    }
}

/// Arithmetic mean; `0.0` for an empty slice (callers only invoke this on
/// series they have already checked are non-empty, but a fallback of `0.0`
/// rather than a panic is the safe default for a private helper).
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Sample standard deviation (Bessel's correction, dividing by n-1); `0.0`
/// for fewer than 2 values, where a standard deviation is not defined.
fn sample_std_dev(values: &[f64], mean_value: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance = values.iter().map(|value| (value - mean_value).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Linear-interpolation percentile (the method most statistics packages
/// default to), `p` in `[0, 100]`. `0.0` for an empty slice.
fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0).clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let frac = rank - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// The `AggregatedMetric.values` key an aggregation method is stored under.
fn aggregation_method_key(method: &AggregationMethod) -> String {
    match method {
        AggregationMethod::Mean => "mean".to_string(),
        AggregationMethod::Median => "median".to_string(),
        AggregationMethod::Minimum => "min".to_string(),
        AggregationMethod::Maximum => "max".to_string(),
        AggregationMethod::StandardDeviation => "std_dev".to_string(),
        AggregationMethod::Percentile(p) => format!("p{:.0}", p),
        AggregationMethod::Sum => "sum".to_string(),
        AggregationMethod::Count => "count".to_string(),
    }
}

/// Computes one `AggregationMethod` over a value series.
fn compute_aggregation(values: &[f64], method: &AggregationMethod) -> f64 {
    match method {
        AggregationMethod::Mean => mean(values),
        AggregationMethod::Median => percentile(values, 50.0),
        AggregationMethod::Minimum => values.iter().cloned().fold(f64::INFINITY, f64::min),
        AggregationMethod::Maximum => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        AggregationMethod::StandardDeviation => sample_std_dev(values, mean(values)),
        AggregationMethod::Percentile(p) => percentile(values, *p as f64),
        AggregationMethod::Sum => values.iter().sum(),
        AggregationMethod::Count => values.len() as f64,
    }
}

/// Flattens a window of `ResourceUtilizationSnapshot`s into named value
/// series: each resource category's four recorded sub-metrics
/// (`port.cpu_utilization`, `port.memory_utilization`, ...), the same for
/// every custom resource (`custom.<name>.cpu_utilization`, ...), and the
/// six flat `SystemMetrics` fields (`system.cpu_utilization`, ...). This
/// aggregates exactly what was recorded -- it does not invent a composite
/// "overall utilization" number for a category.
fn resource_utilization_series(
    history: &[&ResourceUtilizationSnapshot],
) -> Vec<(String, Vec<f64>)> {
    let mut series: HashMap<String, Vec<f64>> = HashMap::new();
    for snapshot in history {
        for (category, metrics) in [
            ("port", &snapshot.port_utilization),
            ("directory", &snapshot.directory_utilization),
            ("gpu", &snapshot.gpu_utilization),
            ("database", &snapshot.database_utilization),
        ] {
            series
                .entry(format!("{category}.cpu_utilization"))
                .or_default()
                .push(metrics.cpu_utilization);
            series
                .entry(format!("{category}.memory_utilization"))
                .or_default()
                .push(metrics.memory_utilization);
            series
                .entry(format!("{category}.io_utilization"))
                .or_default()
                .push(metrics.io_utilization);
            series
                .entry(format!("{category}.network_utilization"))
                .or_default()
                .push(metrics.network_utilization);
        }
        for (name, metrics) in &snapshot.custom_resource_utilization {
            series
                .entry(format!("custom.{name}.cpu_utilization"))
                .or_default()
                .push(metrics.cpu_utilization);
            series
                .entry(format!("custom.{name}.memory_utilization"))
                .or_default()
                .push(metrics.memory_utilization);
            series
                .entry(format!("custom.{name}.io_utilization"))
                .or_default()
                .push(metrics.io_utilization);
            series
                .entry(format!("custom.{name}.network_utilization"))
                .or_default()
                .push(metrics.network_utilization);
        }
        series
            .entry("system.cpu_utilization".to_string())
            .or_default()
            .push(snapshot.system_metrics.cpu_utilization as f64);
        series
            .entry("system.memory_utilization".to_string())
            .or_default()
            .push(snapshot.system_metrics.memory_utilization as f64);
        series
            .entry("system.network_throughput".to_string())
            .or_default()
            .push(snapshot.system_metrics.network_throughput as f64);
        series
            .entry("system.disk_io_rate".to_string())
            .or_default()
            .push(snapshot.system_metrics.disk_io_rate as f64);
        series
            .entry("system.active_processes".to_string())
            .or_default()
            .push(snapshot.system_metrics.active_processes as f64);
        series
            .entry("system.load_average".to_string())
            .or_default()
            .push(snapshot.system_metrics.load_average as f64);
    }
    series.into_iter().collect()
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            network_throughput: 0,
            disk_io_rate: 0,
            active_processes: 0,
            load_average: 0.0,
        }
    }
}

#[cfg(test)]
mod tests_statistics_fix {
    use super::*;
    use crate::resource_management::types_data::{
        DatabaseUsageStatistics, DirectoryUsageStatistics, GpuUsageStatistics, PortUsageStatistics,
        SystemResourceStatistics,
    };
    use chrono::Utc;

    fn make_snapshot(
        port_alloc: usize,
        port_peak: usize,
        dir_alloc: usize,
        dir_peak: usize,
        gpu_alloc: usize,
        gpu_peak: usize,
        db_active: usize,
        db_peak: usize,
    ) -> SystemPerformanceSnapshot {
        SystemPerformanceSnapshot {
            timestamp: Utc::now(),
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            gpu_utilization: None,
            network_utilization: 0.0,
            disk_utilization: 0.0,
            overall_efficiency: 0.0,
            system_stats: SystemResourceStatistics::default(),
            gpu_stats: GpuUsageStatistics {
                currently_allocated: gpu_alloc,
                peak_usage: gpu_peak,
                ..Default::default()
            },
            database_stats: DatabaseUsageStatistics {
                currently_active: db_active,
                peak_usage: db_peak,
                ..Default::default()
            },
            port_stats: PortUsageStatistics {
                currently_allocated: port_alloc,
                peak_usage: port_peak,
                ..Default::default()
            },
            directory_stats: DirectoryUsageStatistics {
                currently_allocated: dir_alloc,
                peak_usage: dir_peak,
                ..Default::default()
            },
        }
    }

    #[tokio::test]
    async fn test_active_resources_and_peak_usage() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        // Snapshot 1: active sum = 1+2+3+4 = 10, peak sum = 5+6+7+8 = 26
        collector
            .record_snapshot(make_snapshot(1, 5, 2, 6, 3, 7, 4, 8))
            .await
            .expect("record_snapshot should succeed");
        // Snapshot 2 (most recent): active sum = 10+20+30+40 = 100, peak sum = 50+60+70+80 = 260
        collector
            .record_snapshot(make_snapshot(10, 50, 20, 60, 30, 70, 40, 80))
            .await
            .expect("record_snapshot should succeed");
        let stats = collector
            .get_performance_statistics(std::time::Duration::from_secs(3600))
            .await
            .expect("get_performance_statistics should succeed");
        // active_resources: last snapshot sum = 10+20+30+40 = 100
        assert_eq!(
            stats.active_resources, 100,
            "active_resources should be last snapshot sum"
        );
        // peak_usage: max over all snapshots = max(26, 260) = 260
        assert_eq!(
            stats.peak_usage, 260,
            "peak_usage should be max peak sum across snapshots"
        );
    }

    /// Regression: `efficiency_score` used to be the hard-coded constant
    /// `0.85` regardless of what the snapshots actually measured.
    #[tokio::test]
    async fn test_efficiency_score_is_measured_not_constant() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        let mut low = make_snapshot(1, 5, 2, 6, 3, 7, 4, 8);
        low.overall_efficiency = 0.2;
        let mut high = make_snapshot(1, 5, 2, 6, 3, 7, 4, 8);
        high.overall_efficiency = 0.8;
        collector.record_snapshot(low).await.expect("record_snapshot should succeed");
        collector.record_snapshot(high).await.expect("record_snapshot should succeed");
        let stats = collector
            .get_performance_statistics(std::time::Duration::from_secs(3600))
            .await
            .expect("get_performance_statistics should succeed");
        assert!(
            (stats.efficiency_score - 0.5).abs() < 1e-5,
            "efficiency_score should be the mean of the recorded overall_efficiency values, got {}",
            stats.efficiency_score
        );
    }

    fn metric_snapshot(
        timestamp: DateTime<Utc>,
        cpu_utilization: f32,
        overall_efficiency: f32,
    ) -> SystemPerformanceSnapshot {
        SystemPerformanceSnapshot {
            timestamp,
            cpu_utilization,
            memory_utilization: 0.0,
            gpu_utilization: None,
            network_utilization: 0.0,
            disk_utilization: 0.0,
            overall_efficiency,
            system_stats: SystemResourceStatistics::default(),
            gpu_stats: GpuUsageStatistics::default(),
            database_stats: DatabaseUsageStatistics::default(),
            port_stats: PortUsageStatistics::default(),
            directory_stats: DirectoryUsageStatistics::default(),
        }
    }

    #[tokio::test]
    async fn test_predict_performance_refuses_empty_history() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        let result = collector
            .predict_performance("cpu_utilization", std::time::Duration::from_secs(60))
            .await;
        assert!(
            result.is_err(),
            "predicting from no data must be a refusal, not a number"
        );
    }

    #[tokio::test]
    async fn test_predict_performance_refuses_unknown_metric() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        collector
            .record_snapshot(metric_snapshot(Utc::now(), 10.0, 0.0))
            .await
            .expect("record_snapshot should succeed");
        let result = collector
            .predict_performance("not_a_real_metric", std::time::Duration::from_secs(60))
            .await;
        let error = result.expect_err("an unrecognized metric name must be refused");
        assert!(error.to_string().contains("unknown metric"));
    }

    #[tokio::test]
    async fn test_predict_performance_refuses_single_snapshot() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        collector
            .record_snapshot(metric_snapshot(Utc::now(), 10.0, 0.0))
            .await
            .expect("record_snapshot should succeed");
        let result = collector
            .predict_performance("cpu_utilization", std::time::Duration::from_secs(60))
            .await;
        assert!(result.is_err(), "a trend needs at least 2 points");
    }

    /// A perfectly linear history must fit almost exactly: high confidence
    /// (R^2 close to 1.0), a narrow interval, and a prediction that continues
    /// the trend -- none of it the old hard-coded 75.0/(70,80)/0.85.
    #[tokio::test]
    async fn test_predict_performance_linear_trend() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        let now = Utc::now();
        for i in 0..10 {
            let timestamp = now - ChronoDuration::minutes(9 - i);
            let cpu = 10.0 + i as f32 * 5.0; // 10, 15, ..., 55
            collector
                .record_snapshot(metric_snapshot(timestamp, cpu, 0.0))
                .await
                .expect("record_snapshot should succeed");
        }
        let prediction = collector
            .predict_performance("cpu_utilization", std::time::Duration::from_secs(60))
            .await
            .expect("predict_performance should succeed on a clean linear history");
        assert!(
            prediction.confidence > 0.99,
            "a near-perfect linear fit should report near-perfect confidence, got {}",
            prediction.confidence
        );
        assert!(
            prediction.predicted_value > 0.55,
            "extrapolating an upward trend should predict above the last reading, got {}",
            prediction.predicted_value
        );
        assert!(prediction.confidence_interval.0 <= prediction.predicted_value);
        assert!(prediction.predicted_value <= prediction.confidence_interval.1);
    }

    #[tokio::test]
    async fn test_detect_anomalies_refuses_empty_history() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        assert!(collector.detect_anomalies().await.is_err());
    }

    #[tokio::test]
    async fn test_detect_anomalies_constant_series_has_no_anomalies() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        for _ in 0..10 {
            collector
                .record_snapshot(metric_snapshot(Utc::now(), 50.0, 0.0))
                .await
                .expect("record_snapshot should succeed");
        }
        let anomalies =
            collector.detect_anomalies().await.expect("detect_anomalies should succeed");
        assert!(
            anomalies.iter().all(|a| a.metric_name != "cpu_utilization"),
            "a series with zero variance has no statistical outliers"
        );
    }

    /// Three-sigma detection over 21 real readings, one of which is a clear
    /// outlier -- not the old `Ok(vec![])` regardless of input.
    #[tokio::test]
    async fn test_detect_anomalies_finds_outlier() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        for _ in 0..20 {
            collector
                .record_snapshot(metric_snapshot(Utc::now(), 50.0, 0.0))
                .await
                .expect("record_snapshot should succeed");
        }
        collector
            .record_snapshot(metric_snapshot(Utc::now(), 99.0, 0.0))
            .await
            .expect("record_snapshot should succeed");
        let anomalies =
            collector.detect_anomalies().await.expect("detect_anomalies should succeed");
        let cpu_anomaly = anomalies
            .iter()
            .find(|a| a.metric_name == "cpu_utilization")
            .expect("the 99.0 reading among twenty 50.0 readings must be flagged");
        assert!((cpu_anomaly.actual_value - 0.99).abs() < 1e-9);
        assert!(
            cpu_anomaly.anomaly_score > 3.0,
            "z-score should exceed the 3-sigma threshold"
        );
        assert_eq!(cpu_anomaly.resource_type, "cpu");
    }

    #[tokio::test]
    async fn test_analyze_bottlenecks_refuses_empty_history() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        assert!(collector.analyze_bottlenecks().await.is_err());
    }

    #[tokio::test]
    async fn test_analyze_bottlenecks_no_bottleneck_when_utilization_low() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        collector
            .record_snapshot(metric_snapshot(Utc::now(), 10.0, 0.0))
            .await
            .expect("record_snapshot should succeed");
        let bottlenecks = collector
            .analyze_bottlenecks()
            .await
            .expect("analyze_bottlenecks should succeed");
        assert!(bottlenecks.is_empty());
    }

    /// CPU above the 90% `CpuThrottling` threshold and memory above the 85%
    /// `MemoryPressure` threshold in the same snapshot must both be reported,
    /// ranked by measured severity -- memory (99%) ahead of CPU (95%).
    #[tokio::test]
    async fn test_analyze_bottlenecks_detects_and_ranks_by_severity() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        let mut snapshot = metric_snapshot(Utc::now(), 95.0, 0.0);
        snapshot.memory_utilization = 99.0;
        collector
            .record_snapshot(snapshot)
            .await
            .expect("record_snapshot should succeed");
        let bottlenecks = collector
            .analyze_bottlenecks()
            .await
            .expect("analyze_bottlenecks should succeed");
        assert_eq!(
            bottlenecks.len(),
            2,
            "both cpu and memory should be over threshold"
        );
        assert_eq!(
            bottlenecks[0].bottleneck_type,
            BottleneckType::MemoryPressure,
            "the more severe bottleneck (99%) should be ranked first"
        );
        assert_eq!(
            bottlenecks[1].bottleneck_type,
            BottleneckType::CpuThrottling
        );
        assert!(bottlenecks[0].impact_severity > bottlenecks[1].impact_severity);
    }

    fn utilization_metrics(cpu: f64) -> ResourceUtilizationMetrics {
        ResourceUtilizationMetrics {
            cpu_utilization: cpu,
            memory_utilization: 0.0,
            io_utilization: 0.0,
            network_utilization: 0.0,
            timestamp: Utc::now(),
        }
    }

    fn utilization_snapshot(port_cpu: f64) -> ResourceUtilizationSnapshot {
        ResourceUtilizationSnapshot {
            timestamp: Utc::now(),
            port_utilization: utilization_metrics(port_cpu),
            directory_utilization: utilization_metrics(0.0),
            gpu_utilization: utilization_metrics(0.0),
            database_utilization: utilization_metrics(0.0),
            custom_resource_utilization: HashMap::new(),
            system_metrics: SystemMetrics::default(),
        }
    }

    /// Regression: `aggregate_utilization_metrics` used to log a message and
    /// do nothing; `get_aggregated_metrics` therefore always returned an
    /// empty map. It must now hold real mean/min/max/percentile values
    /// computed from what was actually recorded.
    #[tokio::test]
    async fn test_aggregate_utilization_metrics_computes_real_values() {
        let collector = StatisticsCollector::new(StatisticsConfig::default())
            .await
            .expect("StatisticsCollector::new should succeed");
        collector
            .record_utilization(utilization_snapshot(10.0))
            .await
            .expect("record_utilization should succeed");
        collector
            .record_utilization(utilization_snapshot(20.0))
            .await
            .expect("record_utilization should succeed");
        let aggregated = collector
            .get_aggregated_metrics(std::time::Duration::from_secs(3600))
            .await
            .expect("get_aggregated_metrics should succeed");
        let port_cpu = aggregated
            .get("port.cpu_utilization")
            .expect("port.cpu_utilization series should have been aggregated");
        assert_eq!(port_cpu.sample_count, 2);
        assert!((port_cpu.values["mean"] - 15.0).abs() < 1e-9);
        assert!((port_cpu.values["min"] - 10.0).abs() < 1e-9);
        assert!((port_cpu.values["max"] - 20.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_aggregate_utilization_metrics_refuses_empty_history() {
        let aggregator = MetricsAggregator::new(AggregationSettings::default());
        let result = aggregator.aggregate_utilization_metrics(&Vec::new()).await;
        assert!(
            result.is_err(),
            "aggregating nothing must be a refusal, not a silent no-op"
        );
    }
}
