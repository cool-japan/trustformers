//! Performance Analytics and Intelligence for Mobile AI Optimization
//!
//! This module provides advanced performance analytics, pattern recognition,
//! and intelligent optimization recommendations for mobile AI inference.

use crate::{MobileBackend, MobilePlatform};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

/// Performance analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalyticsConfig {
    /// Enable real-time analytics
    pub enable_real_time: bool,
    /// Enable machine learning based predictions
    pub enable_ml_predictions: bool,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Enable performance forecasting
    pub enable_forecasting: bool,
    /// Historical data retention period (hours)
    pub retention_hours: u32,
    /// Sampling frequency (milliseconds)
    pub sampling_frequency_ms: u64,
    /// Minimum data points for analysis
    pub min_data_points: usize,
    /// Enable cross-session learning
    pub enable_cross_session_learning: bool,
    /// Export analytics data
    pub enable_export: bool,
}

impl Default for PerformanceAnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_real_time: true,
            enable_ml_predictions: true,
            enable_anomaly_detection: true,
            enable_forecasting: true,
            retention_hours: 24,
            sampling_frequency_ms: 1000,
            min_data_points: 10,
            enable_cross_session_learning: false,
            enable_export: false,
        }
    }
}

/// Performance metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// Inference latency in milliseconds
    InferenceLatency,
    /// Memory usage in bytes
    MemoryUsage,
    /// CPU utilization percentage
    CpuUtilization,
    /// GPU utilization percentage
    GpuUtilization,
    /// Power consumption in watts
    PowerConsumption,
    /// Thermal temperature in Celsius
    Temperature,
    /// Battery level percentage
    BatteryLevel,
    /// Network usage in bytes
    NetworkUsage,
    /// Cache hit rate percentage
    CacheHitRate,
    /// Throughput in operations per second
    Throughput,
    /// Error rate percentage
    ErrorRate,
}

/// Performance data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp (Unix epoch milliseconds)
    pub timestamp: u64,
    /// Metric value
    pub value: f64,
    /// Optional context information
    pub context: Option<HashMap<String, String>>,
}

/// Performance trend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceTrend {
    /// Performance is improving
    Improving,
    /// Performance is stable
    Stable,
    /// Performance is degrading
    Degrading,
    /// Insufficient data to determine trend
    Unknown,
}

/// Performance anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnomaly {
    /// Anomaly detection timestamp
    pub timestamp: u64,
    /// Metric type affected
    pub metric_type: MetricType,
    /// Anomaly severity (0.0-1.0)
    pub severity: f32,
    /// Anomaly description
    pub description: String,
    /// Suggested remediation
    pub remediation: Vec<String>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// Performance forecast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceForecast {
    /// Metric type
    pub metric_type: MetricType,
    /// Forecast horizon (minutes)
    pub horizon_minutes: u32,
    /// Predicted values with timestamps
    pub predictions: Vec<DataPoint>,
    /// Confidence intervals (lower, upper)
    pub confidence_intervals: Vec<(f64, f64)>,
    /// Forecast accuracy score (0.0-1.0)
    pub accuracy_score: f32,
}

/// Performance insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInsights {
    /// Overall performance score (0.0-1.0)
    pub overall_score: f32,
    /// Performance trends by metric
    pub trends: HashMap<MetricType, PerformanceTrend>,
    /// Detected anomalies
    pub anomalies: Vec<PerformanceAnomaly>,
    /// Performance forecasts
    pub forecasts: HashMap<MetricType, PerformanceForecast>,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Key performance indicators
    pub kpis: HashMap<String, f64>,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation ID
    pub id: String,
    /// Recommendation title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Expected impact (0.0-1.0)
    pub expected_impact: f32,
    /// Implementation difficulty (0.0-1.0)
    pub difficulty: f32,
    /// Priority level (0.0-1.0)
    pub priority: f32,
    /// Implementation steps
    pub steps: Vec<String>,
    /// Related metrics
    pub related_metrics: Vec<MetricType>,
}

/// Time series data storage
#[derive(Debug, Clone)]
struct TimeSeriesData {
    /// Data points sorted by timestamp
    data: VecDeque<DataPoint>,
    /// Maximum retention period
    max_age: Duration,
    /// Last cleanup timestamp
    last_cleanup: Instant,
}

impl TimeSeriesData {
    fn new(retention_hours: u32) -> Self {
        Self {
            data: VecDeque::new(),
            max_age: Duration::from_secs(retention_hours as u64 * 3600),
            last_cleanup: Instant::now(),
        }
    }

    fn add_point(&mut self, point: DataPoint) {
        self.data.push_back(point);
        self.maybe_cleanup();
    }

    fn maybe_cleanup(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_cleanup) > Duration::from_secs(300) {
            // Cleanup every 5 minutes
            let cutoff_timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .saturating_sub(self.max_age)
                .as_millis() as u64;

            while let Some(front) = self.data.front() {
                if front.timestamp < cutoff_timestamp {
                    self.data.pop_front();
                } else {
                    break;
                }
            }

            self.last_cleanup = now;
        }
    }

    fn get_recent_data(&self, duration: Duration) -> Vec<DataPoint> {
        let cutoff_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .saturating_sub(duration)
            .as_millis() as u64;

        self.data
            .iter()
            .filter(|point| point.timestamp >= cutoff_timestamp)
            .cloned()
            .collect()
    }
}

/// Performance analytics engine
pub struct PerformanceAnalyticsEngine {
    config: PerformanceAnalyticsConfig,
    platform: MobilePlatform,
    backend: MobileBackend,

    // Time series data storage
    metrics_data: Arc<Mutex<HashMap<MetricType, TimeSeriesData>>>,

    // Analytics state
    anomaly_models: Arc<Mutex<HashMap<MetricType, AnomalyDetector>>>,
    forecasting_models: Arc<Mutex<HashMap<MetricType, ForecastingModel>>>,

    // Performance tracking
    session_start: Instant,
    total_samples: AtomicUsize,
    last_analysis: Arc<Mutex<Option<PerformanceInsights>>>,

    // Configuration
    is_running: AtomicBool,
}

impl PerformanceAnalyticsEngine {
    /// Create a new performance analytics engine
    pub fn new(
        config: PerformanceAnalyticsConfig,
        platform: MobilePlatform,
        backend: MobileBackend,
    ) -> Self {
        let mut metrics_data = HashMap::new();
        let mut anomaly_models = HashMap::new();
        let mut forecasting_models = HashMap::new();

        // Initialize time series storage for each metric type
        for metric_type in [
            MetricType::InferenceLatency,
            MetricType::MemoryUsage,
            MetricType::CpuUtilization,
            MetricType::GpuUtilization,
            MetricType::PowerConsumption,
            MetricType::Temperature,
            MetricType::BatteryLevel,
            MetricType::NetworkUsage,
            MetricType::CacheHitRate,
            MetricType::Throughput,
            MetricType::ErrorRate,
        ] {
            metrics_data.insert(metric_type, TimeSeriesData::new(config.retention_hours));

            if config.enable_anomaly_detection {
                anomaly_models.insert(metric_type, AnomalyDetector::new(metric_type));
            }

            if config.enable_forecasting {
                forecasting_models.insert(metric_type, ForecastingModel::new(metric_type));
            }
        }

        Self {
            config,
            platform,
            backend,
            metrics_data: Arc::new(Mutex::new(metrics_data)),
            anomaly_models: Arc::new(Mutex::new(anomaly_models)),
            forecasting_models: Arc::new(Mutex::new(forecasting_models)),
            session_start: Instant::now(),
            total_samples: AtomicUsize::new(0),
            last_analysis: Arc::new(Mutex::new(None)),
            is_running: AtomicBool::new(false),
        }
    }

    /// Start the analytics engine
    pub fn start(&self) {
        self.is_running.store(true, Ordering::Relaxed);

        if self.config.enable_real_time {
            self.start_real_time_monitoring();
        }
    }

    /// Stop the analytics engine
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    /// Record a performance metric
    pub fn record_metric(
        &self,
        metric_type: MetricType,
        value: f64,
        context: Option<HashMap<String, String>>,
    ) {
        if !self.is_running.load(Ordering::Relaxed) {
            return;
        }

        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;

        let data_point = DataPoint {
            timestamp,
            value,
            context,
        };

        // Store the data point
        if let Ok(mut metrics_data) = self.metrics_data.lock() {
            if let Some(time_series) = metrics_data.get_mut(&metric_type) {
                time_series.add_point(data_point.clone());
            }
        }

        // Update anomaly detection models
        if self.config.enable_anomaly_detection {
            if let Ok(mut models) = self.anomaly_models.lock() {
                if let Some(detector) = models.get_mut(&metric_type) {
                    detector.update(value);
                }
            }
        }

        // Update forecasting models
        if self.config.enable_forecasting {
            if let Ok(mut models) = self.forecasting_models.lock() {
                if let Some(forecaster) = models.get_mut(&metric_type) {
                    forecaster.update(data_point);
                }
            }
        }

        self.total_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Generate comprehensive performance insights
    pub fn generate_insights(&self) -> Result<PerformanceInsights> {
        let mut trends = HashMap::new();
        let mut anomalies = Vec::new();
        let mut forecasts = HashMap::new();
        let mut kpis = HashMap::new();

        // Analyze trends for each metric
        if let Ok(metrics_data) = self.metrics_data.lock() {
            for (metric_type, time_series) in metrics_data.iter() {
                let recent_data = time_series.get_recent_data(Duration::from_secs(3600)); // Last hour

                if recent_data.len() >= self.config.min_data_points {
                    trends.insert(*metric_type, self.analyze_trend(&recent_data));

                    // Calculate KPIs
                    let metric_name = format!("{:?}", metric_type);
                    if !recent_data.is_empty() {
                        let values: Vec<f64> = recent_data.iter().map(|p| p.value).collect();
                        kpis.insert(format!("{}_avg", metric_name), self.calculate_mean(&values));
                        kpis.insert(
                            format!("{}_p95", metric_name),
                            self.calculate_percentile(&values, 0.95),
                        );
                        kpis.insert(
                            format!("{}_min", metric_name),
                            values.iter().copied().fold(f64::INFINITY, f64::min),
                        );
                        kpis.insert(
                            format!("{}_max", metric_name),
                            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                        );
                    }
                }
            }
        }

        // Detect anomalies
        if self.config.enable_anomaly_detection {
            if let Ok(models) = self.anomaly_models.lock() {
                for (metric_type, detector) in models.iter() {
                    if let Some(anomaly) = detector.detect_anomaly() {
                        anomalies.push(PerformanceAnomaly {
                            timestamp: SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64,
                            metric_type: *metric_type,
                            severity: anomaly.severity,
                            description: anomaly.description,
                            remediation: anomaly.remediation,
                            confidence: anomaly.confidence,
                        });
                    }
                }
            }
        }

        // Generate forecasts
        if self.config.enable_forecasting {
            if let Ok(models) = self.forecasting_models.lock() {
                for (metric_type, forecaster) in models.iter() {
                    if let Ok(forecast) = forecaster.generate_forecast(30) {
                        // 30-minute forecast
                        forecasts.insert(*metric_type, forecast);
                    }
                }
            }
        }

        // Generate optimization recommendations
        let recommendations = self.generate_recommendations(&trends, &anomalies, &kpis);

        // Calculate overall performance score
        let overall_score = self.calculate_overall_score(&trends, &anomalies, &kpis);

        let insights = PerformanceInsights {
            overall_score,
            trends,
            anomalies,
            forecasts,
            recommendations,
            kpis,
        };

        // Cache the analysis
        if let Ok(mut last_analysis) = self.last_analysis.lock() {
            *last_analysis = Some(insights.clone());
        }

        Ok(insights)
    }

    /// Get the latest cached insights
    pub fn get_cached_insights(&self) -> Option<PerformanceInsights> {
        self.last_analysis.lock().ok()?.clone()
    }

    /// Export analytics data
    pub fn export_data(&self, format: ExportFormat) -> Result<String> {
        if !self.config.enable_export {
            return Err(TrustformersError::invalid_input("Export is disabled".to_string()).into());
        }

        let metrics_data = self.metrics_data.lock().unwrap();

        match format {
            ExportFormat::Json => {
                let mut export_data = HashMap::new();
                for (metric_type, time_series) in metrics_data.iter() {
                    let recent_data = time_series.get_recent_data(Duration::from_secs(3600 * 24)); // Last 24 hours
                    export_data.insert(format!("{:?}", metric_type), recent_data);
                }
                serde_json::to_string_pretty(&export_data)
                    .map_err(|e| TrustformersError::serialization_error(e.to_string()).into())
            },
            ExportFormat::Csv => {
                let mut csv_data = String::from("timestamp,metric_type,value\n");
                for (metric_type, time_series) in metrics_data.iter() {
                    let recent_data = time_series.get_recent_data(Duration::from_secs(3600 * 24));
                    for point in recent_data {
                        csv_data.push_str(&format!(
                            "{},{:?},{}\n",
                            point.timestamp, metric_type, point.value
                        ));
                    }
                }
                Ok(csv_data)
            },
        }
    }

    /// Start real-time monitoring
    fn start_real_time_monitoring(&self) {
        // This would start a background thread for real-time monitoring
        // For now, just a placeholder
    }

    /// Analyze performance trend
    fn analyze_trend(&self, data: &[DataPoint]) -> PerformanceTrend {
        if data.len() < 3 {
            return PerformanceTrend::Unknown;
        }

        // Simple linear regression to detect trend
        let n = data.len() as f64;
        let sum_x: f64 = (0..data.len()).map(|i| i as f64).sum();
        let sum_y: f64 = data.iter().map(|p| p.value).sum();
        let sum_xy: f64 = data.iter().enumerate().map(|(i, p)| i as f64 * p.value).sum();
        let sum_x2: f64 = (0..data.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));

        let threshold = 0.01; // Adjust based on metric type
        if slope > threshold {
            PerformanceTrend::Improving
        } else if slope < -threshold {
            PerformanceTrend::Degrading
        } else {
            PerformanceTrend::Stable
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        trends: &HashMap<MetricType, PerformanceTrend>,
        anomalies: &[PerformanceAnomaly],
        kpis: &HashMap<String, f64>,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check for memory issues
        if let Some(&PerformanceTrend::Degrading) = trends.get(&MetricType::MemoryUsage) {
            recommendations.push(OptimizationRecommendation {
                id: "memory_optimization".to_string(),
                title: "Memory Usage Optimization".to_string(),
                description: "Memory usage is trending upward. Consider enabling memory pooling and garbage collection.".to_string(),
                expected_impact: 0.7,
                difficulty: 0.3,
                priority: 0.8,
                steps: vec![
                    "Enable automatic garbage collection".to_string(),
                    "Implement memory pooling".to_string(),
                    "Review memory allocation patterns".to_string(),
                ],
                related_metrics: vec![MetricType::MemoryUsage],
            });
        }

        // Check for thermal issues
        if anomalies
            .iter()
            .any(|a| a.metric_type == MetricType::Temperature && a.severity > 0.7)
        {
            recommendations.push(OptimizationRecommendation {
                id: "thermal_management".to_string(),
                title: "Thermal Management".to_string(),
                description:
                    "High temperature detected. Implement thermal throttling to prevent damage."
                        .to_string(),
                expected_impact: 0.9,
                difficulty: 0.4,
                priority: 0.9,
                steps: vec![
                    "Enable thermal monitoring".to_string(),
                    "Implement CPU/GPU throttling".to_string(),
                    "Reduce inference frequency under high temperature".to_string(),
                ],
                related_metrics: vec![MetricType::Temperature, MetricType::PowerConsumption],
            });
        }

        // Check for performance issues
        if let Some(&PerformanceTrend::Degrading) = trends.get(&MetricType::InferenceLatency) {
            recommendations.push(OptimizationRecommendation {
                id: "performance_optimization".to_string(),
                title: "Inference Performance Optimization".to_string(),
                description: "Inference latency is increasing. Consider optimization techniques."
                    .to_string(),
                expected_impact: 0.6,
                difficulty: 0.5,
                priority: 0.7,
                steps: vec![
                    "Enable quantization".to_string(),
                    "Apply operator fusion".to_string(),
                    "Optimize memory layout".to_string(),
                    "Consider model compression".to_string(),
                ],
                related_metrics: vec![MetricType::InferenceLatency, MetricType::Throughput],
            });
        }

        recommendations
    }

    /// Calculate overall performance score
    fn calculate_overall_score(
        &self,
        trends: &HashMap<MetricType, PerformanceTrend>,
        anomalies: &[PerformanceAnomaly],
        kpis: &HashMap<String, f64>,
    ) -> f32 {
        let mut score = 1.0f32;

        // Penalize degrading trends
        for trend in trends.values() {
            match trend {
                PerformanceTrend::Degrading => score *= 0.8,
                PerformanceTrend::Stable => score *= 0.95,
                PerformanceTrend::Improving => score *= 1.05,
                PerformanceTrend::Unknown => {}, // No change
            }
        }

        // Penalize anomalies
        for anomaly in anomalies {
            score *= 1.0 - (anomaly.severity * 0.5);
        }

        score.clamp(0.0, 1.0)
    }

    /// Calculate mean of values
    fn calculate_mean(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    /// Calculate percentile of values
    fn calculate_percentile(&self, values: &[f64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (percentile * (sorted_values.len() - 1) as f64) as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }
}

/// Export format options
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    Json,
    Csv,
}

/// Simple anomaly detector
struct AnomalyDetector {
    metric_type: MetricType,
    values: VecDeque<f64>,
    mean: f64,
    std_dev: f64,
    threshold_multiplier: f64,
}

impl AnomalyDetector {
    fn new(metric_type: MetricType) -> Self {
        Self {
            metric_type,
            values: VecDeque::with_capacity(100),
            mean: 0.0,
            std_dev: 0.0,
            threshold_multiplier: 2.0, // 2 standard deviations
        }
    }

    fn update(&mut self, value: f64) {
        self.values.push_back(value);
        if self.values.len() > 100 {
            self.values.pop_front();
        }

        // Recalculate statistics
        if self.values.len() >= 10 {
            self.mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
            let variance = self.values.iter().map(|v| (v - self.mean).powi(2)).sum::<f64>()
                / self.values.len() as f64;
            self.std_dev = variance.sqrt();
        }
    }

    fn detect_anomaly(&self) -> Option<DetectedAnomaly> {
        if self.values.len() < 10 || self.std_dev == 0.0 {
            return None;
        }

        if let Some(&latest_value) = self.values.back() {
            let z_score = (latest_value - self.mean).abs() / self.std_dev;

            if z_score > self.threshold_multiplier {
                let severity = (z_score / self.threshold_multiplier - 1.0).min(1.0) as f32;

                return Some(DetectedAnomaly {
                    severity,
                    description: format!(
                        "Unusual {} value detected: {:.2} (z-score: {:.2})",
                        format!("{:?}", self.metric_type),
                        latest_value,
                        z_score
                    ),
                    remediation: self.get_remediation_suggestions(),
                    confidence: 0.8, // Fixed confidence for simple detector
                });
            }
        }

        None
    }

    fn get_remediation_suggestions(&self) -> Vec<String> {
        match self.metric_type {
            MetricType::MemoryUsage => vec![
                "Enable garbage collection".to_string(),
                "Check for memory leaks".to_string(),
                "Reduce batch size".to_string(),
            ],
            MetricType::Temperature => vec![
                "Enable thermal throttling".to_string(),
                "Reduce inference frequency".to_string(),
                "Check device ventilation".to_string(),
            ],
            MetricType::InferenceLatency => vec![
                "Enable quantization".to_string(),
                "Apply operator fusion".to_string(),
                "Reduce model complexity".to_string(),
            ],
            _ => vec!["Monitor system resources".to_string()],
        }
    }
}

/// Detected anomaly information
struct DetectedAnomaly {
    severity: f32,
    description: String,
    remediation: Vec<String>,
    confidence: f32,
}

/// Simple forecasting model
struct ForecastingModel {
    metric_type: MetricType,
    historical_data: VecDeque<DataPoint>,
}

impl ForecastingModel {
    fn new(metric_type: MetricType) -> Self {
        Self {
            metric_type,
            historical_data: VecDeque::with_capacity(1000),
        }
    }

    fn update(&mut self, data_point: DataPoint) {
        self.historical_data.push_back(data_point);
        if self.historical_data.len() > 1000 {
            self.historical_data.pop_front();
        }
    }

    fn generate_forecast(&self, horizon_minutes: u32) -> Result<PerformanceForecast> {
        if self.historical_data.len() < 10 {
            return Err(TrustformersError::invalid_input(
                "Insufficient data for forecasting".to_string(),
            )
            .into());
        }

        // Simple linear extrapolation
        let recent_data: Vec<_> = self.historical_data.iter()
            .rev()
            .take(30) // Use last 30 points
            .collect();

        if recent_data.len() < 2 {
            return Err(
                TrustformersError::invalid_input("Insufficient recent data".to_string()).into(),
            );
        }

        // Calculate simple trend
        let first_point = recent_data.last().unwrap();
        let last_point = recent_data.first().unwrap();
        let time_diff = (last_point.timestamp - first_point.timestamp) as f64;
        let value_diff = last_point.value - first_point.value;
        let slope = if time_diff > 0.0 { value_diff / time_diff } else { 0.0 };

        // Generate predictions
        let mut predictions = Vec::new();
        let mut confidence_intervals = Vec::new();
        let current_time =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;

        for i in 1..=horizon_minutes {
            let future_time = current_time + (i as u64 * 60 * 1000); // Convert minutes to milliseconds
            let time_offset = (future_time - last_point.timestamp) as f64;
            let predicted_value = last_point.value + slope * time_offset;

            predictions.push(DataPoint {
                timestamp: future_time,
                value: predicted_value,
                context: None,
            });

            // Simple confidence interval (±10%)
            let confidence_range = predicted_value.abs() * 0.1;
            confidence_intervals.push((
                predicted_value - confidence_range,
                predicted_value + confidence_range,
            ));
        }

        Ok(PerformanceForecast {
            metric_type: self.metric_type,
            horizon_minutes,
            predictions,
            confidence_intervals,
            accuracy_score: 0.7, // Fixed accuracy for simple model
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_engine_creation() {
        let config = PerformanceAnalyticsConfig::default();
        let engine =
            PerformanceAnalyticsEngine::new(config, MobilePlatform::Generic, MobileBackend::CPU);

        // Test basic functionality
        engine.start();
        engine.record_metric(MetricType::InferenceLatency, 50.0, None);
        engine.stop();
    }

    #[test]
    fn test_trend_analysis() {
        let config = PerformanceAnalyticsConfig::default();
        let engine =
            PerformanceAnalyticsEngine::new(config, MobilePlatform::Generic, MobileBackend::CPU);

        // Create ascending data points
        let data = vec![
            DataPoint {
                timestamp: 1000,
                value: 1.0,
                context: None,
            },
            DataPoint {
                timestamp: 2000,
                value: 2.0,
                context: None,
            },
            DataPoint {
                timestamp: 3000,
                value: 3.0,
                context: None,
            },
            DataPoint {
                timestamp: 4000,
                value: 4.0,
                context: None,
            },
        ];

        let trend = engine.analyze_trend(&data);
        assert_eq!(trend, PerformanceTrend::Improving);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut detector = AnomalyDetector::new(MetricType::InferenceLatency);

        // Add normal values
        for i in 0..20 {
            detector.update(10.0 + (i as f64 * 0.1));
        }

        // Add an anomaly
        detector.update(50.0);

        let anomaly = detector.detect_anomaly();
        assert!(anomaly.is_some());
        assert!(anomaly.unwrap().severity > 0.0);
    }

    #[test]
    fn test_forecasting() {
        let mut forecaster = ForecastingModel::new(MetricType::MemoryUsage);

        // Add historical data
        for i in 0..15 {
            forecaster.update(DataPoint {
                timestamp: (i * 1000) as u64,
                value: (i * 2) as f64,
                context: None,
            });
        }

        let forecast = forecaster.generate_forecast(5);
        assert!(forecast.is_ok());

        let forecast = forecast.unwrap();
        assert_eq!(forecast.predictions.len(), 5);
        assert_eq!(forecast.confidence_intervals.len(), 5);
    }
}
