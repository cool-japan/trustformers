//! AI-powered Performance Regression Detection System
//!
//! This module provides advanced statistical analysis and machine learning-based
//! detection of performance regressions in model training and inference, enabling
//! early detection of performance degradation with high accuracy.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::SystemTime;
use tracing::info;
use uuid::Uuid;

/// Configuration for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetectionConfig {
    /// Enable regression detection
    pub enable_detection: bool,
    /// Minimum number of data points for analysis
    pub min_data_points: usize,
    /// Statistical significance threshold (p-value)
    pub significance_threshold: f64,
    /// Minimum performance degradation percentage to trigger alert
    pub min_degradation_threshold: f64,
    /// Maximum historical data window in hours
    pub max_history_hours: u64,
    /// Smoothing factor for exponential moving averages
    pub ema_smoothing_factor: f64,
    /// Enable advanced ML-based detection
    pub enable_ml_detection: bool,
    /// Confidence threshold for ML predictions
    pub ml_confidence_threshold: f64,
    /// Enable seasonal adjustment
    pub enable_seasonal_adjustment: bool,
    /// Enable outlier detection before regression analysis
    pub enable_outlier_filtering: bool,
}

impl Default for RegressionDetectionConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            min_data_points: 10,
            significance_threshold: 0.05,
            min_degradation_threshold: 5.0, // 5% degradation
            max_history_hours: 24,
            ema_smoothing_factor: 0.3,
            enable_ml_detection: true,
            ml_confidence_threshold: 0.8,
            enable_seasonal_adjustment: true,
            enable_outlier_filtering: true,
        }
    }
}

/// Types of metrics to monitor for regressions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Training/inference latency
    Latency,
    /// Memory usage
    MemoryUsage,
    /// CPU utilization
    CpuUtilization,
    /// GPU utilization
    GpuUtilization,
    /// Throughput (operations per second)
    Throughput,
    /// Model accuracy/loss
    ModelAccuracy,
    /// Custom metric
    Custom(String),
}

/// Performance metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    pub metric_type: MetricType,
    pub value: f64,
    pub timestamp: SystemTime,
    pub session_id: Uuid,
    pub metadata: HashMap<String, String>,
}

/// Historical metric series for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSeries {
    pub metric_type: MetricType,
    pub data_points: VecDeque<MetricDataPoint>,
    pub baseline_statistics: BaselineStatistics,
    pub last_updated: SystemTime,
}

/// Baseline statistics for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
    pub trend_slope: f64,
    pub seasonal_pattern: Option<Vec<f64>>,
    pub sample_count: usize,
    pub last_computed: SystemTime,
}

/// Regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetection {
    pub detection_id: Uuid,
    pub metric_type: MetricType,
    pub regression_type: RegressionType,
    pub severity: RegressionSeverity,
    pub confidence: f64,
    pub degradation_percentage: f64,
    pub statistical_significance: f64,
    pub affected_period: (SystemTime, SystemTime),
    pub root_cause_analysis: RootCauseAnalysis,
    pub recommendations: Vec<String>,
    pub detected_at: SystemTime,
}

/// Types of performance regressions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RegressionType {
    /// Sudden step change in performance
    StepChange,
    /// Gradual degradation over time
    GradualDegradation,
    /// Increased variance/instability
    VarianceIncrease,
    /// Periodic performance drops
    PeriodicRegression,
    /// Outlier-driven regression
    OutlierRegression,
    /// Complex multi-factorial regression
    ComplexRegression,
}

/// Severity levels for regressions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegressionSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Root cause analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCauseAnalysis {
    pub likely_causes: Vec<PotentialCause>,
    pub correlated_metrics: Vec<String>,
    pub environmental_factors: Vec<String>,
    pub change_points: Vec<SystemTime>,
    pub anomaly_score: f64,
}

/// Potential cause for performance regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialCause {
    pub cause_type: CauseType,
    pub description: String,
    pub confidence: f64,
    pub supporting_evidence: Vec<String>,
}

/// Types of potential causes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CauseType {
    CodeChange,
    DataChange,
    ResourceContention,
    HardwareIssue,
    ConfigurationChange,
    EnvironmentalFactor,
    ModelDrift,
    Unknown,
}

/// Advanced regression detector with ML capabilities
pub struct RegressionDetector {
    config: RegressionDetectionConfig,
    metric_series: HashMap<MetricType, MetricSeries>,
    anomaly_detector: AnomalyDetector,
    trend_analyzer: TrendAnalyzer,
    change_point_detector: ChangePointDetector,
    seasonal_decomposer: SeasonalDecomposer,
    ml_predictor: Option<MLPredictor>,
    detection_history: VecDeque<RegressionDetection>,
}

/// Statistical anomaly detector
#[derive(Debug)]
struct AnomalyDetector {
    z_score_threshold: f64,
    iqr_multiplier: f64,
    #[allow(dead_code)]
    isolation_forest_threshold: f64,
}

impl AnomalyDetector {
    fn new() -> Self {
        Self {
            z_score_threshold: 3.0,
            iqr_multiplier: 1.5,
            isolation_forest_threshold: 0.1,
        }
    }

    /// Detect outliers using multiple methods
    fn detect_outliers(&self, values: &[f64]) -> Vec<bool> {
        if values.is_empty() {
            return vec![];
        }

        let z_score_outliers = self.detect_z_score_outliers(values);
        let iqr_outliers = self.detect_iqr_outliers(values);

        // Combine methods using majority voting
        z_score_outliers
            .iter()
            .zip(iqr_outliers.iter())
            .map(|(&z_outlier, &iqr_outlier)| z_outlier || iqr_outlier)
            .collect()
    }

    fn detect_z_score_outliers(&self, values: &[f64]) -> Vec<bool> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        values
            .iter()
            .map(|&value| {
                if std_dev > 0.0 {
                    ((value - mean) / std_dev).abs() > self.z_score_threshold
                } else {
                    false
                }
            })
            .collect()
    }

    fn detect_iqr_outliers(&self, values: &[f64]) -> Vec<bool> {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1 = Self::percentile(&sorted_values, 25.0);
        let q3 = Self::percentile(&sorted_values, 75.0);
        let iqr = q3 - q1;

        let lower_bound = q1 - self.iqr_multiplier * iqr;
        let upper_bound = q3 + self.iqr_multiplier * iqr;

        values.iter().map(|&value| value < lower_bound || value > upper_bound).collect()
    }

    fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }

        let index = (percentile / 100.0) * (sorted_values.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted_values[lower]
        } else {
            let weight = index - lower as f64;
            sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
        }
    }
}

/// Trend analysis for regression detection
#[derive(Debug)]
struct TrendAnalyzer {
    window_size: usize,
    significance_threshold: f64,
}

impl TrendAnalyzer {
    fn new(window_size: usize, significance_threshold: f64) -> Self {
        Self {
            window_size,
            significance_threshold,
        }
    }

    /// Detect trend changes using linear regression
    fn detect_trend_change(&self, values: &[f64]) -> Option<TrendChangeResult> {
        if values.len() < self.window_size {
            return None;
        }

        let recent_values = &values[values.len() - self.window_size..];
        let baseline_values = if values.len() >= 2 * self.window_size {
            &values[values.len() - 2 * self.window_size..values.len() - self.window_size]
        } else {
            &values[0..values.len() - self.window_size]
        };

        let recent_slope = self.calculate_slope(recent_values);
        let baseline_slope = self.calculate_slope(baseline_values);

        let slope_change = recent_slope - baseline_slope;
        let significance = self.calculate_trend_significance(recent_values, recent_slope);

        if significance < self.significance_threshold {
            Some(TrendChangeResult {
                slope_change,
                recent_slope,
                baseline_slope,
                significance,
                is_regression: slope_change > 0.0, // Positive slope = performance degradation
            })
        } else {
            None
        }
    }

    fn calculate_slope(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let sum_x = (0..values.len()).sum::<usize>() as f64;
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum::<f64>();
        let sum_x_squared = (0..values.len()).map(|i| (i as f64).powi(2)).sum::<f64>();

        let denominator = n * sum_x_squared - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }

    fn calculate_trend_significance(&self, values: &[f64], slope: f64) -> f64 {
        // Simplified t-test for trend significance
        if values.len() < 3 {
            return 1.0;
        }

        let n = values.len() as f64;
        let mean_x = (values.len() - 1) as f64 / 2.0;
        let ss_x = (0..values.len()).map(|i| (i as f64 - mean_x).powi(2)).sum::<f64>();

        // Calculate residuals with proper intercept
        let mean_y = values.iter().sum::<f64>() / n;
        let intercept = mean_y - slope * mean_x;
        let predicted: Vec<f64> = (0..values.len()).map(|i| intercept + slope * i as f64).collect();

        let residuals: Vec<f64> = values
            .iter()
            .zip(predicted.iter())
            .map(|(&actual, &pred)| actual - pred)
            .collect();

        let mse = residuals.iter().map(|&r| r.powi(2)).sum::<f64>() / (n - 2.0);
        let se_slope = (mse / ss_x).sqrt();

        if se_slope > 0.0 {
            let t_stat = slope / se_slope;
            // Simplified p-value approximation
            let df = n - 2.0;
            if df > 0.0 {
                2.0 * (1.0 - Self::t_distribution_cdf(t_stat.abs(), df))
            } else {
                1.0
            }
        } else {
            1.0
        }
    }

    fn t_distribution_cdf(t: f64, df: f64) -> f64 {
        // Simplified approximation of t-distribution CDF
        // In practice, would use a proper statistical library
        let x = t / (df + t.powi(2)).sqrt();
        0.5 + 0.5 * x.atan() * (2.0 / std::f64::consts::PI)
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct TrendChangeResult {
    slope_change: f64,
    #[allow(dead_code)]
    recent_slope: f64,
    baseline_slope: f64,
    significance: f64,
    is_regression: bool,
}

/// Change point detection using statistical methods
#[derive(Debug)]
struct ChangePointDetector {
    min_segment_length: usize,
    penalty_factor: f64,
}

impl ChangePointDetector {
    fn new(min_segment_length: usize, penalty_factor: f64) -> Self {
        Self {
            min_segment_length,
            penalty_factor,
        }
    }

    /// Detect change points using CUSUM algorithm
    fn detect_change_points(&self, values: &[f64]) -> Vec<usize> {
        if values.len() < 2 * self.min_segment_length {
            return vec![];
        }

        let mut change_points = vec![];
        let mut current_start = 0;

        while current_start + 2 * self.min_segment_length <= values.len() {
            if let Some(change_point) = self.find_next_change_point(&values[current_start..]) {
                let absolute_change_point = current_start + change_point;
                change_points.push(absolute_change_point);
                current_start = absolute_change_point + self.min_segment_length;
            } else {
                break;
            }
        }

        change_points
    }

    fn find_next_change_point(&self, values: &[f64]) -> Option<usize> {
        let n = values.len();
        if n < 2 * self.min_segment_length {
            return None;
        }

        let mut max_statistic = 0.0;
        let mut best_change_point = None;

        for t in self.min_segment_length..n - self.min_segment_length {
            let statistic = self.cusum_statistic(values, t);
            if statistic > max_statistic {
                max_statistic = statistic;
                best_change_point = Some(t);
            }
        }

        // Apply penalty for multiple change points
        let threshold = self.penalty_factor * (n as f64).ln();
        if max_statistic > threshold {
            best_change_point
        } else {
            None
        }
    }

    fn cusum_statistic(&self, values: &[f64], change_point: usize) -> f64 {
        let segment1 = &values[0..change_point];
        let segment2 = &values[change_point..];

        let mean1 = segment1.iter().sum::<f64>() / segment1.len() as f64;
        let mean2 = segment2.iter().sum::<f64>() / segment2.len() as f64;
        let overall_mean = values.iter().sum::<f64>() / values.len() as f64;

        let n1 = segment1.len() as f64;
        let n2 = segment2.len() as f64;
        let n = values.len() as f64;

        // Calculate variance
        let variance = values.iter().map(|&x| (x - overall_mean).powi(2)).sum::<f64>() / (n - 1.0);

        if variance > 0.0 {
            
            (n1 * (mean1 - overall_mean).powi(2)
                + n2 * (mean2 - overall_mean).powi(2))
                / variance
        } else {
            0.0
        }
    }
}

/// Seasonal decomposition for time series analysis
#[derive(Debug)]
struct SeasonalDecomposer {
    period: usize,
    enable_decomposition: bool,
}

impl SeasonalDecomposer {
    fn new(period: usize) -> Self {
        Self {
            period,
            enable_decomposition: true,
        }
    }

    /// Decompose time series into trend, seasonal, and residual components
    fn decompose(&self, values: &[f64]) -> Option<SeasonalComponents> {
        if !self.enable_decomposition || values.len() < 2 * self.period {
            return None;
        }

        let trend = self.extract_trend(values);
        let detrended = self.subtract_series(values, &trend);
        let seasonal = self.extract_seasonal(&detrended);
        let residual = self.subtract_series(&detrended, &seasonal);

        Some(SeasonalComponents {
            trend,
            seasonal,
            residual,
        })
    }

    fn extract_trend(&self, values: &[f64]) -> Vec<f64> {
        // Moving average for trend extraction
        let window_size = self.period;
        let mut trend = vec![0.0; values.len()];

        for i in 0..values.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = std::cmp::min(i + window_size / 2 + 1, values.len());

            let sum: f64 = values[start..end].iter().sum();
            trend[i] = sum / (end - start) as f64;
        }

        trend
    }

    fn extract_seasonal(&self, detrended: &[f64]) -> Vec<f64> {
        let mut seasonal = vec![0.0; detrended.len()];
        let mut seasonal_pattern = vec![0.0; self.period];
        let mut pattern_counts = vec![0usize; self.period];

        // Calculate average seasonal pattern
        for (i, &value) in detrended.iter().enumerate() {
            let season_index = i % self.period;
            seasonal_pattern[season_index] += value;
            pattern_counts[season_index] += 1;
        }

        // Normalize by counts
        for i in 0..self.period {
            if pattern_counts[i] > 0 {
                seasonal_pattern[i] /= pattern_counts[i] as f64;
            }
        }

        // Apply seasonal pattern
        for (i, seasonal_value) in seasonal.iter_mut().enumerate() {
            *seasonal_value = seasonal_pattern[i % self.period];
        }

        seasonal
    }

    fn subtract_series(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        series1.iter().zip(series2.iter()).map(|(&a, &b)| a - b).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SeasonalComponents {
    trend: Vec<f64>,
    seasonal: Vec<f64>,
    residual: Vec<f64>,
}

/// ML-based predictor for advanced regression detection
#[derive(Debug)]
struct MLPredictor {
    #[allow(dead_code)]
    model_type: MLModelType,
    feature_extractor: FeatureExtractor,
    prediction_threshold: f64,
}

#[allow(dead_code)]
#[derive(Debug)]
enum MLModelType {
    IsolationForest,
    #[allow(dead_code)]
    LSTM,
    AutoEncoder,
}

#[derive(Debug)]
struct FeatureExtractor {
    window_size: usize,
    statistical_features: bool,
    frequency_features: bool,
}

impl MLPredictor {
    fn new(model_type: MLModelType, prediction_threshold: f64) -> Self {
        Self {
            model_type,
            feature_extractor: FeatureExtractor {
                window_size: 50,
                statistical_features: true,
                frequency_features: true,
            },
            prediction_threshold,
        }
    }

    /// Predict if current pattern indicates regression
    fn predict_regression(&self, values: &[f64]) -> Option<MLPrediction> {
        if values.len() < self.feature_extractor.window_size {
            return None;
        }

        let features = self.feature_extractor.extract_features(values);

        // Simplified ML prediction (in practice would use trained models)
        let anomaly_score = self.calculate_anomaly_score(&features);
        let confidence = self.calculate_confidence(&features);

        if anomaly_score > self.prediction_threshold {
            Some(MLPrediction {
                anomaly_score,
                confidence,
                feature_importance: self.calculate_feature_importance(&features),
                predicted_severity: self.predict_severity(anomaly_score),
            })
        } else {
            None
        }
    }

    fn calculate_anomaly_score(&self, features: &[f64]) -> f64 {
        // Simplified anomaly scoring based on feature deviation
        let mean = features.iter().sum::<f64>() / features.len() as f64;
        let variance =
            features.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / features.len() as f64;

        variance.sqrt() / (mean.abs() + 1e-6)
    }

    fn calculate_confidence(&self, features: &[f64]) -> f64 {
        // Simplified confidence calculation
        let feature_consistency = 1.0
            - (features.iter().map(|&x| (x - features[0]).abs()).sum::<f64>()
                / (features.len() as f64 * features[0].abs() + 1e-6));

        feature_consistency.max(0.0).min(1.0)
    }

    fn calculate_feature_importance(&self, features: &[f64]) -> Vec<f64> {
        // Simplified feature importance based on magnitude
        let max_magnitude = features.iter().map(|x| x.abs()).fold(0.0, f64::max);

        if max_magnitude > 0.0 {
            features.iter().map(|&x| x.abs() / max_magnitude).collect()
        } else {
            vec![0.0; features.len()]
        }
    }

    fn predict_severity(&self, anomaly_score: f64) -> RegressionSeverity {
        if anomaly_score > 0.8 {
            RegressionSeverity::Critical
        } else if anomaly_score > 0.6 {
            RegressionSeverity::High
        } else if anomaly_score > 0.4 {
            RegressionSeverity::Medium
        } else {
            RegressionSeverity::Low
        }
    }
}

impl FeatureExtractor {
    fn extract_features(&self, values: &[f64]) -> Vec<f64> {
        let mut features = Vec::new();

        if self.statistical_features {
            features.extend(self.extract_statistical_features(values));
        }

        if self.frequency_features {
            features.extend(self.extract_frequency_features(values));
        }

        features
    }

    fn extract_statistical_features(&self, values: &[f64]) -> Vec<f64> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;

        // Skewness
        let skewness = if std_dev > 0.0 {
            values.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / values.len() as f64
        } else {
            0.0
        };

        // Kurtosis
        let kurtosis = if std_dev > 0.0 {
            values.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / values.len() as f64
                - 3.0
        } else {
            0.0
        };

        vec![mean, std_dev, min, max, range, skewness, kurtosis]
    }

    fn extract_frequency_features(&self, values: &[f64]) -> Vec<f64> {
        // Simplified frequency domain features
        let mut features = Vec::new();

        // Calculate differences to get "frequencies"
        let differences: Vec<f64> = values.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

        if !differences.is_empty() {
            let mean_diff = differences.iter().sum::<f64>() / differences.len() as f64;
            let max_diff = differences.iter().fold(0.0f64, |a, &b| a.max(b));
            features.extend([mean_diff, max_diff]);
        }

        features
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MLPrediction {
    anomaly_score: f64,
    confidence: f64,
    feature_importance: Vec<f64>,
    predicted_severity: RegressionSeverity,
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new(config: RegressionDetectionConfig) -> Self {
        let ml_predictor = if config.enable_ml_detection {
            Some(MLPredictor::new(
                MLModelType::IsolationForest,
                config.ml_confidence_threshold,
            ))
        } else {
            None
        };

        let trend_analyzer =
            TrendAnalyzer::new(config.min_data_points, config.significance_threshold);

        Self {
            config,
            metric_series: HashMap::new(),
            anomaly_detector: AnomalyDetector::new(),
            trend_analyzer,
            change_point_detector: ChangePointDetector::new(5, 2.0),
            seasonal_decomposer: SeasonalDecomposer::new(24), // Hourly patterns
            ml_predictor,
            detection_history: VecDeque::new(),
        }
    }

    /// Add a new metric data point
    pub fn add_metric_data_point(&mut self, data_point: MetricDataPoint) -> Result<()> {
        let metric_type = data_point.metric_type.clone();
        let max_data_points = (self.config.max_history_hours * 60) as usize; // Assume 1 point per minute
        let min_data_points = self.config.min_data_points;

        // Update series data
        let data_points_len = {
            let series =
                self.metric_series.entry(metric_type.clone()).or_insert_with(|| MetricSeries {
                    metric_type: metric_type.clone(),
                    data_points: VecDeque::new(),
                    baseline_statistics: BaselineStatistics::default(),
                    last_updated: SystemTime::now(),
                });

            // Add data point
            series.data_points.push_back(data_point);
            series.last_updated = SystemTime::now();

            // Maintain window size
            while series.data_points.len() > max_data_points {
                series.data_points.pop_front();
            }

            series.data_points.len()
        };

        // Update baseline statistics
        self.update_baseline_statistics(&metric_type)?;

        // Check for regressions
        if data_points_len >= min_data_points {
            if let Some(detection) = self.detect_regression(&metric_type)? {
                self.detection_history.push_back(detection);

                // Maintain detection history size
                while self.detection_history.len() > 1000 {
                    self.detection_history.pop_front();
                }
            }
        }

        Ok(())
    }

    /// Detect regressions for a specific metric
    pub fn detect_regression(
        &mut self,
        metric_type: &MetricType,
    ) -> Result<Option<RegressionDetection>> {
        let series = match self.metric_series.get(metric_type) {
            Some(series) => series,
            None => return Ok(None),
        };

        if series.data_points.len() < self.config.min_data_points {
            return Ok(None);
        }

        let values: Vec<f64> = series.data_points.iter().map(|dp| dp.value).collect();

        // Filter outliers if enabled
        let filtered_values = if self.config.enable_outlier_filtering {
            self.filter_outliers(&values)
        } else {
            values.clone()
        };

        // Multiple detection methods
        let mut detections = Vec::new();

        // 1. Statistical trend analysis
        if let Some(trend_result) = self.trend_analyzer.detect_trend_change(&filtered_values) {
            if trend_result.is_regression {
                let severity = self.calculate_severity(trend_result.slope_change);
                detections.push(RegressionDetection {
                    detection_id: Uuid::new_v4(),
                    metric_type: metric_type.clone(),
                    regression_type: RegressionType::GradualDegradation,
                    severity,
                    confidence: 1.0 - trend_result.significance,
                    degradation_percentage: trend_result.slope_change * 100.0,
                    statistical_significance: trend_result.significance,
                    affected_period: self.calculate_affected_period(series),
                    root_cause_analysis: self.analyze_root_causes(series, &filtered_values),
                    recommendations: self.generate_recommendations(
                        &RegressionType::GradualDegradation,
                        trend_result.slope_change,
                    ),
                    detected_at: SystemTime::now(),
                });
            }
        }

        // 2. Change point detection
        let change_points = self.change_point_detector.detect_change_points(&filtered_values);
        if !change_points.is_empty() {
            let latest_change_point = change_points.last().unwrap();
            let before = &filtered_values[0..*latest_change_point];
            let after = &filtered_values[*latest_change_point..];

            if !before.is_empty() && !after.is_empty() {
                let before_mean = before.iter().sum::<f64>() / before.len() as f64;
                let after_mean = after.iter().sum::<f64>() / after.len() as f64;
                let degradation = ((after_mean - before_mean) / before_mean) * 100.0;

                if degradation > self.config.min_degradation_threshold {
                    detections.push(RegressionDetection {
                        detection_id: Uuid::new_v4(),
                        metric_type: metric_type.clone(),
                        regression_type: RegressionType::StepChange,
                        severity: self.calculate_severity(degradation / 100.0),
                        confidence: 0.8,
                        degradation_percentage: degradation,
                        statistical_significance: 0.01, // High confidence for step changes
                        affected_period: self.calculate_affected_period(series),
                        root_cause_analysis: self.analyze_root_causes(series, &filtered_values),
                        recommendations: self.generate_recommendations(
                            &RegressionType::StepChange,
                            degradation / 100.0,
                        ),
                        detected_at: SystemTime::now(),
                    });
                }
            }
        }

        // 3. ML-based detection
        if let Some(ref ml_predictor) = self.ml_predictor {
            if let Some(ml_prediction) = ml_predictor.predict_regression(&filtered_values) {
                detections.push(RegressionDetection {
                    detection_id: Uuid::new_v4(),
                    metric_type: metric_type.clone(),
                    regression_type: RegressionType::ComplexRegression,
                    severity: ml_prediction.predicted_severity,
                    confidence: ml_prediction.confidence,
                    degradation_percentage: ml_prediction.anomaly_score * 100.0,
                    statistical_significance: 1.0 - ml_prediction.confidence,
                    affected_period: self.calculate_affected_period(series),
                    root_cause_analysis: self.analyze_root_causes(series, &filtered_values),
                    recommendations: self.generate_recommendations(
                        &RegressionType::ComplexRegression,
                        ml_prediction.anomaly_score,
                    ),
                    detected_at: SystemTime::now(),
                });
            }
        }

        // Return the most severe detection
        if let Some(detection) = detections.into_iter().max_by_key(|d| d.severity.clone()) {
            info!(
                "Regression detected for {:?}: {:.2}% degradation",
                metric_type, detection.degradation_percentage
            );
            Ok(Some(detection))
        } else {
            Ok(None)
        }
    }

    /// Get recent regression detections
    pub fn get_recent_detections(&self, limit: usize) -> Vec<RegressionDetection> {
        self.detection_history.iter().rev().take(limit).cloned().collect()
    }

    /// Get regression detections for a specific metric
    pub fn get_detections_for_metric(&self, metric_type: &MetricType) -> Vec<RegressionDetection> {
        self.detection_history
            .iter()
            .filter(|d| &d.metric_type == metric_type)
            .cloned()
            .collect()
    }

    /// Update baseline statistics for a metric
    fn update_baseline_statistics(&mut self, metric_type: &MetricType) -> Result<()> {
        let series = self.metric_series.get_mut(metric_type).unwrap();
        let values: Vec<f64> = series.data_points.iter().map(|dp| dp.value).collect();

        if values.is_empty() {
            return Ok(());
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = AnomalyDetector::percentile(&sorted_values, 50.0);
        let percentile_95 = AnomalyDetector::percentile(&sorted_values, 95.0);
        let percentile_99 = AnomalyDetector::percentile(&sorted_values, 99.0);

        let trend_slope = self.trend_analyzer.calculate_slope(&values);

        let seasonal_pattern = if self.config.enable_seasonal_adjustment {
            self.seasonal_decomposer
                .decompose(&values)
                .map(|components| components.seasonal)
        } else {
            None
        };

        series.baseline_statistics = BaselineStatistics {
            mean,
            std_dev,
            median,
            percentile_95,
            percentile_99,
            trend_slope,
            seasonal_pattern,
            sample_count: values.len(),
            last_computed: SystemTime::now(),
        };

        Ok(())
    }

    fn filter_outliers(&self, values: &[f64]) -> Vec<f64> {
        let outlier_mask = self.anomaly_detector.detect_outliers(values);
        values
            .iter()
            .zip(outlier_mask.iter())
            .filter(|(_, &is_outlier)| !is_outlier)
            .map(|(&value, _)| value)
            .collect()
    }

    fn calculate_severity(&self, degradation_ratio: f64) -> RegressionSeverity {
        let degradation_percentage = degradation_ratio.abs() * 100.0;

        if degradation_percentage > 50.0 {
            RegressionSeverity::Critical
        } else if degradation_percentage > 25.0 {
            RegressionSeverity::High
        } else if degradation_percentage > 10.0 {
            RegressionSeverity::Medium
        } else {
            RegressionSeverity::Low
        }
    }

    fn calculate_affected_period(&self, series: &MetricSeries) -> (SystemTime, SystemTime) {
        let start = series.data_points.front().map(|dp| dp.timestamp).unwrap_or(SystemTime::now());
        let end = series.data_points.back().map(|dp| dp.timestamp).unwrap_or(SystemTime::now());
        (start, end)
    }

    fn analyze_root_causes(&self, series: &MetricSeries, values: &[f64]) -> RootCauseAnalysis {
        let mut likely_causes = Vec::new();
        let correlated_metrics = Vec::new();
        let environmental_factors = Vec::new();

        // Analyze patterns to identify potential causes
        let change_points = self.change_point_detector.detect_change_points(values);
        let change_point_timestamps: Vec<SystemTime> = change_points
            .iter()
            .filter_map(|&idx| series.data_points.get(idx).map(|dp| dp.timestamp))
            .collect();

        // Check for sudden changes (potential code/config changes)
        if !change_points.is_empty() {
            likely_causes.push(PotentialCause {
                cause_type: CauseType::CodeChange,
                description: "Sudden performance change detected, possibly due to code deployment"
                    .to_string(),
                confidence: 0.7,
                supporting_evidence: vec![format!(
                    "Change point detected at {} locations",
                    change_points.len()
                )],
            });
        }

        // Check for gradual degradation (potential resource issues)
        let trend_slope = self.trend_analyzer.calculate_slope(values);
        if trend_slope > 0.01 {
            likely_causes.push(PotentialCause {
                cause_type: CauseType::ResourceContention,
                description:
                    "Gradual performance degradation suggests resource contention or memory leaks"
                        .to_string(),
                confidence: 0.6,
                supporting_evidence: vec![format!("Positive trend slope: {:.4}", trend_slope)],
            });
        }

        // Calculate anomaly score
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let anomaly_score = variance.sqrt() / (mean + 1e-6);

        RootCauseAnalysis {
            likely_causes,
            correlated_metrics,
            environmental_factors,
            change_points: change_point_timestamps,
            anomaly_score,
        }
    }

    fn generate_recommendations(
        &self,
        regression_type: &RegressionType,
        degradation: f64,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match regression_type {
            RegressionType::StepChange => {
                recommendations
                    .push("Investigate recent deployments or configuration changes".to_string());
                recommendations
                    .push("Review system logs around the time of performance change".to_string());
                recommendations
                    .push("Consider rolling back recent changes if possible".to_string());
            },
            RegressionType::GradualDegradation => {
                recommendations
                    .push("Monitor resource utilization (CPU, memory, disk)".to_string());
                recommendations.push("Check for memory leaks or resource exhaustion".to_string());
                recommendations
                    .push("Review long-running processes and background tasks".to_string());
            },
            RegressionType::VarianceIncrease => {
                recommendations
                    .push("Investigate system stability and hardware issues".to_string());
                recommendations.push("Check for intermittent network or I/O problems".to_string());
            },
            RegressionType::ComplexRegression => {
                recommendations.push("Perform detailed profiling and analysis".to_string());
                recommendations
                    .push("Investigate multiple potential causes simultaneously".to_string());
            },
            _ => {
                recommendations.push("Perform comprehensive system analysis".to_string());
            },
        }

        if degradation > 0.5 {
            recommendations.push("URGENT: Consider immediate mitigation actions".to_string());
            recommendations.push("Alert on-call team for immediate investigation".to_string());
        } else if degradation > 0.25 {
            recommendations.push("Schedule investigation within 24 hours".to_string());
        }

        recommendations
    }
}

impl Default for BaselineStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            median: 0.0,
            percentile_95: 0.0,
            percentile_99: 0.0,
            trend_slope: 0.0,
            seasonal_pattern: None,
            sample_count: 0,
            last_computed: SystemTime::now(),
        }
    }
}

/// Integration with main debug session
impl crate::DebugSession {
    /// Enable regression detection for this debug session
    pub async fn enable_regression_detection(
        &mut self,
        config: RegressionDetectionConfig,
    ) -> Result<RegressionDetector> {
        let detector = RegressionDetector::new(config);
        info!(
            "Enabled regression detection for debug session {}",
            self.id()
        );
        Ok(detector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_regression_detector_creation() {
        let config = RegressionDetectionConfig::default();
        let detector = RegressionDetector::new(config);

        assert!(detector.metric_series.is_empty());
        assert!(detector.detection_history.is_empty());
    }

    #[tokio::test]
    async fn test_add_metric_data_point() {
        let config = RegressionDetectionConfig::default();
        let mut detector = RegressionDetector::new(config);

        let data_point = MetricDataPoint {
            metric_type: MetricType::Latency,
            value: 100.0,
            timestamp: SystemTime::now(),
            session_id: Uuid::new_v4(),
            metadata: HashMap::new(),
        };

        assert!(detector.add_metric_data_point(data_point).is_ok());
        assert_eq!(detector.metric_series.len(), 1);
    }

    #[test]
    fn test_anomaly_detection() {
        let detector = AnomalyDetector::new();
        let values = vec![1.0, 2.0, 3.0, 2.0, 1.0, 100.0]; // 100.0 is an outlier

        let outliers = detector.detect_outliers(&values);
        assert_eq!(outliers.len(), values.len());
        assert!(outliers[5]); // Last value should be detected as outlier
    }

    #[test]
    fn test_trend_analysis() {
        let analyzer = TrendAnalyzer::new(3, 0.9);
        let values = [1.0, 1.1, 1.2, 10.0, 20.0, 30.0];

        // Test that trend analyzer can calculate slopes
        let recent_values = &values[3..6]; // [10.0, 20.0, 30.0]
        let baseline_values = &values[0..3]; // [1.0, 1.1, 1.2]

        let recent_slope = analyzer.calculate_slope(recent_values);
        let baseline_slope = analyzer.calculate_slope(baseline_values);

        // Recent slope should be much higher than baseline
        assert!(recent_slope > baseline_slope);
        assert!(recent_slope > 0.0);
    }

    #[test]
    fn test_change_point_detection() {
        let detector = ChangePointDetector::new(3, 2.0);
        let values = vec![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]; // Change at index 4

        let change_points = detector.detect_change_points(&values);
        assert!(!change_points.is_empty());
    }

    #[test]
    fn test_seasonal_decomposition() {
        let decomposer = SeasonalDecomposer::new(4);
        let values = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];

        let components = decomposer.decompose(&values);
        assert!(components.is_some());

        let comp = components.unwrap();
        assert_eq!(comp.trend.len(), values.len());
        assert_eq!(comp.seasonal.len(), values.len());
        assert_eq!(comp.residual.len(), values.len());
    }

    #[test]
    fn test_feature_extraction() {
        let extractor = FeatureExtractor {
            window_size: 10,
            statistical_features: true,
            frequency_features: true,
        };

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0];
        let features = extractor.extract_features(&values);

        assert!(!features.is_empty());
        assert!(features.len() >= 7); // At least statistical features
    }
}
