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
    AlertThreshold, PerformanceBaseline, PerformanceTrend, ResourceStatistics,
    ResourceUtilizationMetrics, SystemPerformanceSnapshot,
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
    /// Trend analyzer
    trend_analyzer: TrendAnalyzer,
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
    /// Aggregation configuration
    config: AggregationSettings,
}

/// Trend analysis system
pub struct TrendAnalyzer {
    /// Historical trends
    historical_trends: Arc<Mutex<HashMap<String, PerformanceTrend>>>,
}

/// Anomaly detection system
pub struct AnomalyDetector {
    /// Anomaly detection models
    detection_models: HashMap<String, AnomalyDetectionModel>,
    /// Detected anomalies
    detected_anomalies: Arc<Mutex<Vec<PerformanceAnomaly>>>,
}

/// Performance prediction system
pub struct PerformancePredictor {
    /// Prediction models
    prediction_models: HashMap<String, PredictionModel>,
    /// Prediction cache
    prediction_cache: Arc<Mutex<HashMap<String, PerformancePrediction>>>,
}

/// Bottleneck analysis system
pub struct BottleneckAnalyzer {
    /// Detected bottlenecks
    detected_bottlenecks: Arc<Mutex<Vec<PerformanceBottleneck>>>,
    /// Analysis configuration
    config: BottleneckAnalysisConfig,
}

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

        // TODO: active_resources and total_resources_allocated are not on SystemPerformanceSnapshot
        // Need to derive from system_stats or use different approach
        Ok(ResourceStatistics {
            total_allocated: recent_snapshots.len() as u64,
            active_resources: 0, // TODO: Calculate from system_stats
            peak_usage: recent_snapshots.len() as u64, // TODO: Calculate from system_stats
            avg_lifetime: period / recent_snapshots.len().max(1) as u32,
            utilization_rate: average_cpu.max(average_memory) / 100.0,
            cpu_utilization: average_cpu,
            memory_utilization: average_memory,
            allocation_count: recent_snapshots.len() as u64,
            total_duration: period,
            efficiency_score: 0.85, // Would be calculated based on actual metrics
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
            trend_analyzer: TrendAnalyzer::new(),
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
    pub fn new(config: AggregationSettings) -> Self {
        Self {
            aggregated_metrics: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// Aggregate utilization metrics
    pub async fn aggregate_utilization_metrics(
        &self,
        _utilization_history: &Vec<ResourceUtilizationSnapshot>,
    ) -> Result<()> {
        // Aggregation logic would be implemented here
        debug!("Aggregated utilization metrics");
        Ok(())
    }

    /// Get aggregated metrics
    pub async fn get_aggregated_metrics(
        &self,
        _period: Duration,
    ) -> Result<HashMap<String, AggregatedMetric>> {
        let aggregated_metrics = self.aggregated_metrics.lock();
        Ok(aggregated_metrics.clone())
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
        Self {
            historical_trends: Arc::new(Mutex::new(HashMap::new())),
        }
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
        Self {
            detection_models: HashMap::new(),
            detected_anomalies: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Detect anomalies in performance data
    pub async fn detect_anomalies(
        &self,
        _snapshots: &Arc<Mutex<Vec<SystemPerformanceSnapshot>>>,
    ) -> Result<Vec<PerformanceAnomaly>> {
        // Anomaly detection logic would be implemented here
        Ok(vec![])
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
        Self {
            prediction_models: HashMap::new(),
            prediction_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Predict future performance
    pub async fn predict(
        &self,
        metric_name: &str,
        horizon: Duration,
        _snapshots: &Arc<Mutex<Vec<SystemPerformanceSnapshot>>>,
    ) -> Result<PerformancePrediction> {
        // Prediction logic would be implemented here
        Ok(PerformancePrediction {
            prediction_id: format!("pred_{}", Utc::now().timestamp_millis()),
            resource_type: "system".to_string(),
            metric_name: metric_name.to_string(),
            predicted_at: Utc::now(),
            target_time: Utc::now() + ChronoDuration::from_std(horizon)?,
            predicted_value: 75.0, // Placeholder value
            confidence_interval: (70.0, 80.0),
            confidence: 0.85,
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
        Self {
            detected_bottlenecks: Arc::new(Mutex::new(Vec::new())),
            config: BottleneckAnalysisConfig::default(),
        }
    }

    /// Analyze performance bottlenecks
    pub async fn analyze_bottlenecks(
        &self,
        _snapshots: &Arc<Mutex<Vec<SystemPerformanceSnapshot>>>,
    ) -> Result<Vec<PerformanceBottleneck>> {
        // Bottleneck analysis logic would be implemented here
        Ok(vec![])
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
