//! Anomaly Detection System
//!
//! This module provides advanced anomaly detection capabilities for performance anomalies,
//! including statistical methods, machine learning algorithms, and predictive anomaly
//! detection. It detects unusual patterns, performance spikes, degradation, and system
//! instabilities to enable proactive optimization responses.

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, sync::Arc, time::Duration};

use super::types::*;
use crate::performance_optimizer::types::PerformanceDataPoint;

// =============================================================================
// ANOMALY DETECTION SYSTEM
// =============================================================================

/// Advanced anomaly detection system for performance anomalies
///
/// Detects unusual patterns, performance spikes, degradation, and system
/// instabilities using statistical methods and machine learning algorithms.
pub struct AnomalyDetectionSystem {
    /// Anomaly detection algorithms
    anomaly_detectors: Arc<Mutex<Vec<Box<dyn AnomalyDetector + Send + Sync>>>>,
    /// Detected anomalies cache
    anomaly_cache: Arc<RwLock<HashMap<String, DetectedAnomaly>>>,
    /// Anomaly learning models
    learning_models: Arc<Mutex<Vec<Box<dyn AnomalyLearningModel + Send + Sync>>>>,
    /// Detection configuration
    config: Arc<RwLock<AnomalyDetectionConfig>>,
    /// Performance baselines
    baselines: Arc<RwLock<HashMap<String, PerformanceBaseline>>>,
}

impl AnomalyDetectionSystem {
    /// Create new anomaly detection system
    pub fn new() -> Self {
        let mut system = Self {
            anomaly_detectors: Arc::new(Mutex::new(Vec::new())),
            anomaly_cache: Arc::new(RwLock::new(HashMap::new())),
            learning_models: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(AnomalyDetectionConfig::default())),
            baselines: Arc::new(RwLock::new(HashMap::new())),
        };

        // Initialize default detectors and models
        system.initialize_default_detectors();
        system.initialize_default_models();

        system
    }

    /// Create with custom configuration
    pub fn with_config(config: AnomalyDetectionConfig) -> Self {
        let mut system = Self {
            anomaly_detectors: Arc::new(Mutex::new(Vec::new())),
            anomaly_cache: Arc::new(RwLock::new(HashMap::new())),
            learning_models: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(config)),
            baselines: Arc::new(RwLock::new(HashMap::new())),
        };

        system.initialize_default_detectors();
        system.initialize_default_models();

        system
    }

    /// Detect anomalies in performance data
    pub async fn detect_anomalies(
        &self,
        data_points: &[PerformanceDataPoint],
    ) -> Result<Vec<DetectedAnomaly>> {
        let (enable_detection, min_severity_threshold, enable_ml_detection) = {
            let config = self.config.read();
            (
                config.enable_detection,
                config.min_severity_threshold,
                config.enable_ml_detection,
            )
        };

        if !enable_detection {
            return Ok(Vec::new());
        }

        if data_points.len() < 3 {
            return Err(anyhow::anyhow!(
                "Insufficient data points for anomaly detection"
            ));
        }

        // Update baselines
        self.update_baselines(data_points).await;

        let all_anomalies = {
            let detectors = self.anomaly_detectors.lock();
            let mut collected = Vec::new();

            for detector in detectors.iter() {
                if detector.confidence(data_points) >= min_severity_threshold {
                    match detector.detect_anomalies(data_points) {
                        Ok(anomalies) => {
                            for anomaly in anomalies {
                                if anomaly.severity >= min_severity_threshold {
                                    collected.push(anomaly);
                                }
                            }
                        },
                        Err(e) => {
                            tracing::warn!("Anomaly detector {} failed: {}", detector.name(), e);
                        },
                    }
                }
            }

            collected
        };

        // Filter and deduplicate anomalies
        let filtered_anomalies = self.filter_and_deduplicate_anomalies(all_anomalies)?;

        // Cache detected anomalies
        self.cache_anomalies(&filtered_anomalies).await;

        // Update learning models if enabled
        if enable_ml_detection {
            self.update_learning_models(&filtered_anomalies).await?;
        }

        Ok(filtered_anomalies)
    }

    /// Predict anomaly likelihood
    pub async fn predict_anomaly(
        &self,
        context: &AnomalyContext,
    ) -> Result<Vec<AnomalyPrediction>> {
        let models = self.learning_models.lock();
        let mut predictions = Vec::new();

        for model in models.iter() {
            match model.predict_anomaly(context) {
                Ok(prediction) => predictions.push(prediction),
                Err(e) => {
                    tracing::warn!("Anomaly prediction model {} failed: {}", model.name(), e);
                },
            }
        }

        // Filter predictions by confidence
        let config = self.config.read();
        let high_confidence_predictions: Vec<AnomalyPrediction> = predictions
            .into_iter()
            .filter(|p| p.confidence >= config.min_severity_threshold)
            .collect();

        Ok(high_confidence_predictions)
    }

    /// Get anomalies by type
    pub async fn get_anomalies_by_type(&self, anomaly_type: AnomalyType) -> Vec<DetectedAnomaly> {
        let cache = self.anomaly_cache.read();
        cache
            .values()
            .filter(|anomaly| {
                std::mem::discriminant(&anomaly.anomaly_type)
                    == std::mem::discriminant(&anomaly_type)
            })
            .cloned()
            .collect()
    }

    /// Get recent anomalies
    pub async fn get_recent_anomalies(&self, duration: Duration) -> Vec<DetectedAnomaly> {
        let cache = self.anomaly_cache.read();
        let cutoff_time = Utc::now() - chrono::Duration::from_std(duration).unwrap_or_default();

        cache
            .values()
            .filter(|anomaly| anomaly.detected_at >= cutoff_time)
            .cloned()
            .collect()
    }

    /// Add anomaly detector
    pub fn add_detector(&self, detector: Box<dyn AnomalyDetector + Send + Sync>) {
        let mut detectors = self.anomaly_detectors.lock();
        detectors.push(detector);
    }

    /// Add learning model
    pub fn add_learning_model(&self, model: Box<dyn AnomalyLearningModel + Send + Sync>) {
        let mut models = self.learning_models.lock();
        models.push(model);
    }

    /// Update configuration
    pub fn update_config(&self, new_config: AnomalyDetectionConfig) {
        let mut config = self.config.write();
        *config = new_config;
    }

    /// Clear anomaly cache
    pub async fn clear_cache(&self) {
        let mut cache = self.anomaly_cache.write();
        cache.clear();
    }

    /// Get anomaly statistics
    pub fn get_anomaly_statistics(&self) -> AnomalyStatistics {
        let cache = self.anomaly_cache.read();

        let total_anomalies = cache.len();
        let mut type_counts: HashMap<AnomalyType, usize> = HashMap::new();
        let mut severity_distribution: HashMap<String, usize> = HashMap::new();
        let mut total_severity = 0.0f32;

        for anomaly in cache.values() {
            *type_counts.entry(anomaly.anomaly_type.clone()).or_insert(0) += 1;
            total_severity += anomaly.severity;

            let severity_bucket = if anomaly.severity >= 0.8 {
                "High".to_string()
            } else if anomaly.severity >= 0.5 {
                "Medium".to_string()
            } else {
                "Low".to_string()
            };
            *severity_distribution.entry(severity_bucket).or_insert(0) += 1;
        }

        let average_severity =
            if total_anomalies > 0 { total_severity / total_anomalies as f32 } else { 0.0 };

        AnomalyStatistics {
            total_anomalies,
            anomaly_type_distribution: type_counts,
            severity_distribution,
            average_severity,
            cache_memory_usage: cache.len() * std::mem::size_of::<DetectedAnomaly>(),
        }
    }

    /// Initialize default anomaly detectors
    fn initialize_default_detectors(&mut self) {
        let mut detectors = self.anomaly_detectors.lock();

        detectors.push(Box::new(StatisticalAnomalyDetector::new()));
        detectors.push(Box::new(ZScoreAnomalyDetector::new()));
        detectors.push(Box::new(IQRAnomalyDetector::new()));
        detectors.push(Box::new(MovingAverageAnomalyDetector::new()));
        detectors.push(Box::new(ThresholdAnomalyDetector::new()));
    }

    /// Initialize default learning models
    fn initialize_default_models(&mut self) {
        let mut models = self.learning_models.lock();

        models.push(Box::new(SimpleAnomalyLearner::new()));
        models.push(Box::new(HistoricalAnomalyPredictor::new()));
        models.push(Box::new(PatternBasedAnomalyPredictor::new()));
    }

    /// Update performance baselines
    async fn update_baselines(&self, data_points: &[PerformanceDataPoint]) {
        if data_points.len() < 5 {
            return; // Not enough data for reliable baseline
        }

        let mut baselines = self.baselines.write();

        let throughputs: Vec<f64> = data_points.iter().map(|p| p.throughput).collect();
        let latencies: Vec<f64> =
            data_points.iter().map(|p| p.latency.as_millis() as f64).collect();

        let baseline_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let baseline_latency =
            Duration::from_millis((latencies.iter().sum::<f64>() / latencies.len() as f64) as u64);

        let throughput_variance = calculate_variance(&throughputs, baseline_throughput);
        let latency_variance = calculate_variance(
            &latencies,
            latencies.iter().sum::<f64>() / latencies.len() as f64,
        );

        // Calculate confidence interval (95%)
        let throughput_std = throughput_variance.sqrt();
        let confidence_interval = (
            baseline_throughput - 1.96 * throughput_std,
            baseline_throughput + 1.96 * throughput_std,
        );

        let baseline = PerformanceBaseline {
            baseline_throughput,
            baseline_latency,
            throughput_variance,
            latency_variance,
            baseline_timestamp: Utc::now(),
            sample_size: data_points.len(),
            confidence_interval,
        };

        baselines.insert("default".to_string(), baseline);
    }

    /// Filter and deduplicate anomalies
    fn filter_and_deduplicate_anomalies(
        &self,
        anomalies: Vec<DetectedAnomaly>,
    ) -> Result<Vec<DetectedAnomaly>> {
        let mut filtered_anomalies = Vec::new();

        for anomaly in anomalies {
            // Check for duplicates based on type, timestamp, and data point
            let is_duplicate = filtered_anomalies
                .iter()
                .any(|existing: &DetectedAnomaly| self.are_anomalies_similar(&anomaly, existing));

            if !is_duplicate {
                filtered_anomalies.push(anomaly);
            }
        }

        Ok(filtered_anomalies)
    }

    /// Check if two anomalies are similar (for deduplication)
    fn are_anomalies_similar(
        &self,
        anomaly1: &DetectedAnomaly,
        anomaly2: &DetectedAnomaly,
    ) -> bool {
        // Same type and close in time and value
        std::mem::discriminant(&anomaly1.anomaly_type)
            == std::mem::discriminant(&anomaly2.anomaly_type)
            && (anomaly1.data_point.timestamp - anomaly2.data_point.timestamp).abs()
                < chrono::Duration::minutes(5)
            && (anomaly1.data_point.throughput - anomaly2.data_point.throughput).abs()
                < anomaly1.data_point.throughput * 0.1
    }

    /// Cache detected anomalies
    async fn cache_anomalies(&self, anomalies: &[DetectedAnomaly]) {
        let mut cache = self.anomaly_cache.write();

        for anomaly in anomalies {
            cache.insert(anomaly.id.clone(), anomaly.clone());
        }

        // Maintain cache size (keep last 1000 anomalies)
        if cache.len() > 1000 {
            let mut anomalies_vec: Vec<_> = cache.values().cloned().collect();
            anomalies_vec.sort_by_key(|a| a.detected_at);

            let to_remove = cache.len() - 1000;
            for anomaly in anomalies_vec.iter().take(to_remove) {
                cache.remove(&anomaly.id);
            }
        }
    }

    /// Update learning models with detected anomalies
    async fn update_learning_models(&self, anomalies: &[DetectedAnomaly]) -> Result<()> {
        let mut models = self.learning_models.lock();

        for model in models.iter_mut() {
            if let Err(e) = model.learn_from_anomalies(anomalies) {
                tracing::warn!(
                    "Failed to update anomaly learning model {}: {}",
                    model.name(),
                    e
                );
            }
        }

        Ok(())
    }
}

impl Default for AnomalyDetectionSystem {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// ANOMALY DETECTOR IMPLEMENTATIONS
// =============================================================================

/// Statistical anomaly detector using standard deviation
pub struct StatisticalAnomalyDetector {
    threshold_multiplier: f64,
}

impl Default for StatisticalAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalAnomalyDetector {
    pub fn new() -> Self {
        Self {
            threshold_multiplier: 2.5, // 2.5 standard deviations
        }
    }

    pub fn with_threshold(threshold_multiplier: f64) -> Self {
        Self {
            threshold_multiplier,
        }
    }
}

impl AnomalyDetector for StatisticalAnomalyDetector {
    fn detect_anomalies(
        &self,
        data_points: &[PerformanceDataPoint],
    ) -> Result<Vec<DetectedAnomaly>> {
        if data_points.len() < 3 {
            return Ok(Vec::new());
        }

        let throughputs: Vec<f64> = data_points.iter().map(|p| p.throughput).collect();
        let mean = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let variance = calculate_variance(&throughputs, mean);
        let std_dev = variance.sqrt();

        let mut anomalies = Vec::new();

        for (i, point) in data_points.iter().enumerate() {
            let deviation = (point.throughput - mean).abs();
            let z_score = if std_dev > 0.0 { deviation / std_dev } else { 0.0 };

            if z_score > self.threshold_multiplier {
                let severity = (z_score / (self.threshold_multiplier * 2.0)).min(1.0) as f32;
                let confidence = z_score.min(5.0) / 5.0; // Normalize to 0-1

                let anomaly_type =
                    if point.throughput > mean { AnomalyType::Spike } else { AnomalyType::Drop };

                let mut metadata = HashMap::new();
                metadata.insert("z_score".to_string(), z_score.to_string());
                metadata.insert("mean".to_string(), mean.to_string());
                metadata.insert("std_dev".to_string(), std_dev.to_string());

                anomalies.push(DetectedAnomaly {
                    id: format!(
                        "statistical_{}_{}",
                        i,
                        Utc::now().timestamp_nanos_opt().unwrap_or(0)
                    ),
                    anomaly_type,
                    description: format!("Statistical anomaly with z-score {:.2}", z_score),
                    severity,
                    confidence: confidence as f32,
                    data_point: point.clone(),
                    detected_at: Utc::now(),
                    expected_range: (
                        mean - self.threshold_multiplier * std_dev,
                        mean + self.threshold_multiplier * std_dev,
                    ),
                    deviation,
                    detection_method: "Statistical Standard Deviation".to_string(),
                    metadata,
                });
            }
        }

        Ok(anomalies)
    }

    fn name(&self) -> &str {
        "statistical_anomaly_detector"
    }

    fn confidence(&self, data_points: &[PerformanceDataPoint]) -> f32 {
        if data_points.len() >= 10 {
            0.9
        } else if data_points.len() >= 5 {
            0.7
        } else {
            0.3
        }
    }
}

/// Z-score based anomaly detector
pub struct ZScoreAnomalyDetector {
    threshold: f64,
    window_size: usize,
}

impl Default for ZScoreAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ZScoreAnomalyDetector {
    pub fn new() -> Self {
        Self {
            threshold: 2.0,
            window_size: 10,
        }
    }

    pub fn with_params(threshold: f64, window_size: usize) -> Self {
        Self {
            threshold,
            window_size,
        }
    }
}

impl AnomalyDetector for ZScoreAnomalyDetector {
    fn detect_anomalies(
        &self,
        data_points: &[PerformanceDataPoint],
    ) -> Result<Vec<DetectedAnomaly>> {
        if data_points.len() < self.window_size {
            return Ok(Vec::new());
        }

        let mut anomalies = Vec::new();

        for i in self.window_size..data_points.len() {
            let window = &data_points[i - self.window_size..i];
            let throughputs: Vec<f64> = window.iter().map(|p| p.throughput).collect();

            let mean = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
            let variance = calculate_variance(&throughputs, mean);
            let std_dev = variance.sqrt();

            let current_value = data_points[i].throughput;
            let z_score = if std_dev > 0.0 { (current_value - mean).abs() / std_dev } else { 0.0 };

            if z_score > self.threshold {
                let severity = (z_score / (self.threshold * 2.0)).min(1.0) as f32;
                let confidence = (z_score / 5.0).min(1.0) as f32;

                let anomaly_type =
                    if current_value > mean { AnomalyType::Spike } else { AnomalyType::Drop };

                let mut metadata = HashMap::new();
                metadata.insert("z_score".to_string(), z_score.to_string());
                metadata.insert("window_mean".to_string(), mean.to_string());
                metadata.insert("window_std".to_string(), std_dev.to_string());

                anomalies.push(DetectedAnomaly {
                    id: format!(
                        "zscore_{}_{}",
                        i,
                        Utc::now().timestamp_nanos_opt().unwrap_or(0)
                    ),
                    anomaly_type,
                    description: format!("Z-score anomaly with score {:.2}", z_score),
                    severity,
                    confidence,
                    data_point: data_points[i].clone(),
                    detected_at: Utc::now(),
                    expected_range: (
                        mean - self.threshold * std_dev,
                        mean + self.threshold * std_dev,
                    ),
                    deviation: (current_value - mean).abs(),
                    detection_method: "Z-Score".to_string(),
                    metadata,
                });
            }
        }

        Ok(anomalies)
    }

    fn name(&self) -> &str {
        "zscore_anomaly_detector"
    }

    fn confidence(&self, data_points: &[PerformanceDataPoint]) -> f32 {
        if data_points.len() >= self.window_size * 2 {
            0.85
        } else {
            0.5
        }
    }
}

/// Interquartile Range (IQR) based anomaly detector
pub struct IQRAnomalyDetector {
    iqr_multiplier: f64,
}

impl Default for IQRAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl IQRAnomalyDetector {
    pub fn new() -> Self {
        Self {
            iqr_multiplier: 1.5,
        }
    }

    pub fn with_multiplier(iqr_multiplier: f64) -> Self {
        Self { iqr_multiplier }
    }
}

impl AnomalyDetector for IQRAnomalyDetector {
    fn detect_anomalies(
        &self,
        data_points: &[PerformanceDataPoint],
    ) -> Result<Vec<DetectedAnomaly>> {
        if data_points.len() < 4 {
            return Ok(Vec::new());
        }

        let mut throughputs: Vec<f64> = data_points.iter().map(|p| p.throughput).collect();
        throughputs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = throughputs.len() / 4;
        let q3_idx = (3 * throughputs.len()) / 4;
        let q1 = throughputs[q1_idx];
        let q3 = throughputs[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - self.iqr_multiplier * iqr;
        let upper_bound = q3 + self.iqr_multiplier * iqr;

        let mut anomalies = Vec::new();

        for (i, point) in data_points.iter().enumerate() {
            if point.throughput < lower_bound || point.throughput > upper_bound {
                let deviation = if point.throughput < lower_bound {
                    lower_bound - point.throughput
                } else {
                    point.throughput - upper_bound
                };

                let severity = (deviation / iqr).min(1.0) as f32;
                let confidence = severity;

                let anomaly_type = if point.throughput > upper_bound {
                    AnomalyType::Spike
                } else {
                    AnomalyType::Drop
                };

                let mut metadata = HashMap::new();
                metadata.insert("q1".to_string(), q1.to_string());
                metadata.insert("q3".to_string(), q3.to_string());
                metadata.insert("iqr".to_string(), iqr.to_string());
                metadata.insert("lower_bound".to_string(), lower_bound.to_string());
                metadata.insert("upper_bound".to_string(), upper_bound.to_string());

                anomalies.push(DetectedAnomaly {
                    id: format!(
                        "iqr_{}_{}",
                        i,
                        Utc::now().timestamp_nanos_opt().unwrap_or(0)
                    ),
                    anomaly_type,
                    description: format!("IQR outlier with deviation {:.2}", deviation),
                    severity,
                    confidence,
                    data_point: point.clone(),
                    detected_at: Utc::now(),
                    expected_range: (lower_bound, upper_bound),
                    deviation,
                    detection_method: "Interquartile Range".to_string(),
                    metadata,
                });
            }
        }

        Ok(anomalies)
    }

    fn name(&self) -> &str {
        "iqr_anomaly_detector"
    }

    fn confidence(&self, data_points: &[PerformanceDataPoint]) -> f32 {
        if data_points.len() >= 20 {
            0.8
        } else if data_points.len() >= 10 {
            0.6
        } else {
            0.4
        }
    }
}

/// Moving average based anomaly detector
pub struct MovingAverageAnomalyDetector {
    window_size: usize,
    threshold_multiplier: f64,
}

impl Default for MovingAverageAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl MovingAverageAnomalyDetector {
    pub fn new() -> Self {
        Self {
            window_size: 5,
            threshold_multiplier: 2.0,
        }
    }

    pub fn with_params(window_size: usize, threshold_multiplier: f64) -> Self {
        Self {
            window_size,
            threshold_multiplier,
        }
    }
}

impl AnomalyDetector for MovingAverageAnomalyDetector {
    fn detect_anomalies(
        &self,
        data_points: &[PerformanceDataPoint],
    ) -> Result<Vec<DetectedAnomaly>> {
        if data_points.len() < self.window_size + 1 {
            return Ok(Vec::new());
        }

        let mut anomalies = Vec::new();

        for i in self.window_size..data_points.len() {
            let window = &data_points[i - self.window_size..i];
            let window_values: Vec<f64> = window.iter().map(|p| p.throughput).collect();
            let moving_avg = window_values.iter().sum::<f64>() / window_values.len() as f64;

            // Calculate standard deviation of the window
            let variance = calculate_variance(&window_values, moving_avg);
            let std_dev = variance.sqrt();

            let current_value = data_points[i].throughput;
            let deviation = (current_value - moving_avg).abs();

            if std_dev > 0.0 && deviation > self.threshold_multiplier * std_dev {
                let severity =
                    (deviation / (self.threshold_multiplier * std_dev * 2.0)).min(1.0) as f32;
                let confidence = (deviation / (std_dev * 3.0)).min(1.0) as f32;

                let anomaly_type =
                    if current_value > moving_avg { AnomalyType::Spike } else { AnomalyType::Drop };

                let mut metadata = HashMap::new();
                metadata.insert("moving_average".to_string(), moving_avg.to_string());
                metadata.insert("window_std".to_string(), std_dev.to_string());
                metadata.insert("deviation".to_string(), deviation.to_string());

                anomalies.push(DetectedAnomaly {
                    id: format!(
                        "mavg_{}_{}",
                        i,
                        Utc::now().timestamp_nanos_opt().unwrap_or(0)
                    ),
                    anomaly_type,
                    description: format!("Moving average anomaly with deviation {:.2}", deviation),
                    severity,
                    confidence,
                    data_point: data_points[i].clone(),
                    detected_at: Utc::now(),
                    expected_range: (
                        moving_avg - self.threshold_multiplier * std_dev,
                        moving_avg + self.threshold_multiplier * std_dev,
                    ),
                    deviation,
                    detection_method: "Moving Average".to_string(),
                    metadata,
                });
            }
        }

        Ok(anomalies)
    }

    fn name(&self) -> &str {
        "moving_average_anomaly_detector"
    }

    fn confidence(&self, data_points: &[PerformanceDataPoint]) -> f32 {
        if data_points.len() >= self.window_size * 3 {
            0.8
        } else {
            0.5
        }
    }
}

/// Threshold-based anomaly detector
pub struct ThresholdAnomalyDetector {
    thresholds: HashMap<String, (f64, f64)>, // (min, max) thresholds
}

impl Default for ThresholdAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ThresholdAnomalyDetector {
    pub fn new() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("throughput".to_string(), (0.0, 10000.0));
        thresholds.insert("latency_ms".to_string(), (0.0, 5000.0));

        Self { thresholds }
    }

    pub fn with_thresholds(thresholds: HashMap<String, (f64, f64)>) -> Self {
        Self { thresholds }
    }
}

impl AnomalyDetector for ThresholdAnomalyDetector {
    fn detect_anomalies(
        &self,
        data_points: &[PerformanceDataPoint],
    ) -> Result<Vec<DetectedAnomaly>> {
        let mut anomalies = Vec::new();

        if let Some(&(min_throughput, max_throughput)) = self.thresholds.get("throughput") {
            for (i, point) in data_points.iter().enumerate() {
                if point.throughput < min_throughput || point.throughput > max_throughput {
                    let (deviation, anomaly_type) = if point.throughput < min_throughput {
                        (min_throughput - point.throughput, AnomalyType::Drop)
                    } else {
                        (point.throughput - max_throughput, AnomalyType::Spike)
                    };

                    let threshold_range = max_throughput - min_throughput;
                    let severity = (deviation / threshold_range).min(1.0) as f32;

                    let mut metadata = HashMap::new();
                    metadata.insert("min_threshold".to_string(), min_throughput.to_string());
                    metadata.insert("max_threshold".to_string(), max_throughput.to_string());

                    anomalies.push(DetectedAnomaly {
                        id: format!(
                            "threshold_{}_{}",
                            i,
                            Utc::now().timestamp_nanos_opt().unwrap_or(0)
                        ),
                        anomaly_type,
                        description: format!(
                            "Threshold violation with value {:.2}",
                            point.throughput
                        ),
                        severity,
                        confidence: 1.0, // High confidence for threshold violations
                        data_point: point.clone(),
                        detected_at: Utc::now(),
                        expected_range: (min_throughput, max_throughput),
                        deviation,
                        detection_method: "Threshold".to_string(),
                        metadata,
                    });
                }
            }
        }

        Ok(anomalies)
    }

    fn name(&self) -> &str {
        "threshold_anomaly_detector"
    }

    fn confidence(&self, _data_points: &[PerformanceDataPoint]) -> f32 {
        1.0 // Always confident for threshold-based detection
    }
}

// =============================================================================
// ANOMALY LEARNING MODEL IMPLEMENTATIONS
// =============================================================================

/// Simple anomaly learner that tracks anomaly patterns
pub struct SimpleAnomalyLearner {
    anomaly_patterns: HashMap<String, AnomalyPattern>,
    learning_rate: f32,
}

impl Default for SimpleAnomalyLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleAnomalyLearner {
    pub fn new() -> Self {
        Self {
            anomaly_patterns: HashMap::new(),
            learning_rate: 0.1,
        }
    }
}

impl AnomalyLearningModel for SimpleAnomalyLearner {
    fn learn_from_anomalies(&mut self, anomalies: &[DetectedAnomaly]) -> Result<()> {
        for anomaly in anomalies {
            let pattern_key = format!(
                "{:?}_{}",
                anomaly.anomaly_type,
                (anomaly.severity * 10.0) as i32 // Bucket by severity
            );

            let pattern =
                self.anomaly_patterns.entry(pattern_key).or_insert_with(|| AnomalyPattern {
                    anomaly_type: anomaly.anomaly_type.clone(),
                    frequency: 0.0,
                    average_severity: 0.0,
                    last_occurrence: anomaly.detected_at,
                });

            pattern.frequency += self.learning_rate;
            pattern.average_severity = pattern.average_severity * (1.0 - self.learning_rate)
                + anomaly.severity * self.learning_rate;
            pattern.last_occurrence = anomaly.detected_at.max(pattern.last_occurrence);
        }

        Ok(())
    }

    fn predict_anomaly(&self, _context: &AnomalyContext) -> Result<AnomalyPrediction> {
        // Find the most frequent anomaly pattern
        let most_frequent = self.anomaly_patterns.iter().max_by(|a, b| {
            a.1.frequency.partial_cmp(&b.1.frequency).unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some((_, pattern)) = most_frequent {
            let time_since_last = Utc::now().signed_duration_since(pattern.last_occurrence);
            let likelihood = if let Ok(duration) = time_since_last.to_std() {
                (duration.as_secs() as f32 / 3600.0).min(1.0) // Increase likelihood over time
            } else {
                0.5
            };

            Ok(AnomalyPrediction {
                predicted_anomaly: pattern.anomaly_type.clone(),
                likelihood: pattern.frequency * likelihood,
                expected_time: Utc::now() + chrono::Duration::hours(1),
                confidence: pattern.frequency,
                preventive_measures: vec![
                    "Monitor system closely".to_string(),
                    "Check resource utilization".to_string(),
                    "Review recent optimizations".to_string(),
                ],
            })
        } else {
            Err(anyhow::anyhow!("No anomaly patterns learned yet"))
        }
    }

    fn name(&self) -> &str {
        "simple_anomaly_learner"
    }
}

/// Historical anomaly predictor
pub struct HistoricalAnomalyPredictor {
    historical_intervals: HashMap<AnomalyType, Vec<Duration>>,
}

impl Default for HistoricalAnomalyPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl HistoricalAnomalyPredictor {
    pub fn new() -> Self {
        Self {
            historical_intervals: HashMap::new(),
        }
    }
}

impl AnomalyLearningModel for HistoricalAnomalyPredictor {
    fn learn_from_anomalies(&mut self, anomalies: &[DetectedAnomaly]) -> Result<()> {
        // Group anomalies by type and calculate intervals
        let mut type_timestamps: HashMap<AnomalyType, Vec<DateTime<Utc>>> = HashMap::new();

        for anomaly in anomalies {
            type_timestamps
                .entry(anomaly.anomaly_type.clone())
                .or_default()
                .push(anomaly.detected_at);
        }

        for (anomaly_type, mut timestamps) in type_timestamps {
            timestamps.sort();

            let intervals: Vec<Duration> = timestamps
                .windows(2)
                .filter_map(|window| {
                    let interval = window[1].signed_duration_since(window[0]);
                    interval.to_std().ok()
                })
                .collect();

            if !intervals.is_empty() {
                self.historical_intervals.insert(anomaly_type, intervals);
            }
        }

        Ok(())
    }

    fn predict_anomaly(&self, _context: &AnomalyContext) -> Result<AnomalyPrediction> {
        // Find the anomaly type with the most predictable interval
        let mut best_prediction = None;
        let mut best_confidence = 0.0f32;

        for (anomaly_type, intervals) in &self.historical_intervals {
            if !intervals.is_empty() {
                let avg_interval = intervals.iter().sum::<Duration>() / intervals.len() as u32;
                let confidence = 1.0 / (intervals.len() as f32); // Simple confidence metric

                if confidence > best_confidence {
                    best_confidence = confidence;
                    best_prediction = Some(AnomalyPrediction {
                        predicted_anomaly: anomaly_type.clone(),
                        likelihood: confidence,
                        expected_time: Utc::now()
                            + chrono::Duration::from_std(avg_interval).unwrap_or_default(),
                        confidence,
                        preventive_measures: vec![
                            format!("Prepare for {:?} anomaly", anomaly_type),
                            "Increase monitoring frequency".to_string(),
                        ],
                    });
                }
            }
        }

        best_prediction.ok_or_else(|| anyhow::anyhow!("No historical data for prediction"))
    }

    fn name(&self) -> &str {
        "historical_anomaly_predictor"
    }
}

/// Pattern-based anomaly predictor
pub struct PatternBasedAnomalyPredictor {
    context_anomaly_map: HashMap<String, Vec<DetectedAnomaly>>,
}

impl Default for PatternBasedAnomalyPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternBasedAnomalyPredictor {
    pub fn new() -> Self {
        Self {
            context_anomaly_map: HashMap::new(),
        }
    }

    fn generate_context_key(&self, context: &AnomalyContext) -> String {
        format!(
            "load_{:.1}_cores_{}",
            context.system_state.load_average, context.system_state.available_cores
        )
    }
}

impl AnomalyLearningModel for PatternBasedAnomalyPredictor {
    fn learn_from_anomalies(&mut self, anomalies: &[DetectedAnomaly]) -> Result<()> {
        // For now, store all anomalies under a generic context
        // In a real implementation, we would extract context from the anomaly data
        let generic_key = "generic".to_string();
        self.context_anomaly_map
            .entry(generic_key)
            .or_default()
            .extend(anomalies.iter().cloned());
        Ok(())
    }

    fn predict_anomaly(&self, context: &AnomalyContext) -> Result<AnomalyPrediction> {
        let context_key = self.generate_context_key(context);

        // Look for anomalies in similar context
        let relevant_anomalies = self
            .context_anomaly_map
            .get(&context_key)
            .or_else(|| self.context_anomaly_map.get("generic"));

        if let Some(anomalies) = relevant_anomalies {
            if let Some(most_severe) = anomalies.iter().max_by(|a, b| {
                a.severity.partial_cmp(&b.severity).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                return Ok(AnomalyPrediction {
                    predicted_anomaly: most_severe.anomaly_type.clone(),
                    likelihood: most_severe.severity * 0.7, // Reduce likelihood for prediction
                    expected_time: Utc::now() + chrono::Duration::minutes(15),
                    confidence: most_severe.confidence * 0.8,
                    preventive_measures: vec![
                        format!(
                            "Monitor for {:?} based on system context",
                            most_severe.anomaly_type
                        ),
                        "Check system resources".to_string(),
                        "Review performance trends".to_string(),
                    ],
                });
            }
        }

        Err(anyhow::anyhow!("No contextual anomaly data available"))
    }

    fn name(&self) -> &str {
        "pattern_based_anomaly_predictor"
    }
}

// =============================================================================
// UTILITY TYPES AND FUNCTIONS
// =============================================================================

/// Anomaly pattern for learning
#[derive(Debug, Clone)]
struct AnomalyPattern {
    anomaly_type: AnomalyType,
    frequency: f32,
    average_severity: f32,
    last_occurrence: DateTime<Utc>,
}

/// Anomaly detection statistics
#[derive(Debug, Clone)]
pub struct AnomalyStatistics {
    pub total_anomalies: usize,
    pub anomaly_type_distribution: HashMap<AnomalyType, usize>,
    pub severity_distribution: HashMap<String, usize>,
    pub average_severity: f32,
    pub cache_memory_usage: usize,
}

/// Calculate variance
fn calculate_variance(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
}
