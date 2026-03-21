//! Enhanced Trend Analysis Engine
//!
//! This module provides sophisticated trend detection, analysis, and prediction capabilities
//! using multiple algorithms, machine learning models, and statistical methods. It includes
//! caching, confidence scoring, and comprehensive trend analytics for optimization insights.

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, sync::Arc, time::Duration};

use super::types::*;
use crate::performance_optimizer::types::PerformanceDataPoint;
use crate::test_performance_monitoring::TrendDirection;

// =============================================================================
// ENHANCED TREND ANALYSIS ENGINE
// =============================================================================

/// Enhanced trend analysis engine with advanced algorithms
///
/// Provides sophisticated trend detection, analysis, and prediction capabilities
/// using multiple algorithms, machine learning models, and statistical methods.
pub struct EnhancedTrendAnalysisEngine {
    /// Trend detection algorithms
    algorithms: Arc<Mutex<Vec<Box<dyn TrendDetectionAlgorithm + Send + Sync>>>>,
    /// Trend analysis cache
    trend_cache: Arc<RwLock<HashMap<String, CachedTrendAnalysis>>>,
    /// Trend prediction models
    prediction_models: Arc<Mutex<Vec<Box<dyn TrendPredictionModel + Send + Sync>>>>,
    /// Analysis configuration
    config: Arc<RwLock<TrendAnalysisConfig>>,
}

impl EnhancedTrendAnalysisEngine {
    /// Create new enhanced trend analysis engine
    pub fn new() -> Self {
        let mut engine = Self {
            algorithms: Arc::new(Mutex::new(Vec::new())),
            trend_cache: Arc::new(RwLock::new(HashMap::new())),
            prediction_models: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(TrendAnalysisConfig::default())),
        };

        // Initialize default algorithms
        engine.initialize_default_algorithms();
        engine.initialize_default_models();

        engine
    }

    /// Create with custom configuration
    pub fn with_config(config: TrendAnalysisConfig) -> Self {
        let mut engine = Self {
            algorithms: Arc::new(Mutex::new(Vec::new())),
            trend_cache: Arc::new(RwLock::new(HashMap::new())),
            prediction_models: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(config)),
        };

        engine.initialize_default_algorithms();
        engine.initialize_default_models();

        engine
    }

    /// Analyze trend in performance data
    pub async fn analyze_trend(
        &self,
        data_points: &[PerformanceDataPoint],
    ) -> Result<TrendAnalysisResult> {
        let config = self.config.read();

        if data_points.len() < config.min_data_points {
            return Err(anyhow::anyhow!(
                "Insufficient data points: {} < {}",
                data_points.len(),
                config.min_data_points
            ));
        }

        // Check cache first
        let cache_key = self.generate_cache_key(data_points);
        if let Some(cached) = self.get_cached_analysis(&cache_key) {
            return Ok(cached.trend.clone().into());
        }

        // Run analysis with all applicable algorithms
        let algorithms = self.algorithms.lock();
        let mut results = Vec::new();

        for algorithm in algorithms.iter() {
            if algorithm.is_applicable(data_points) {
                match algorithm.detect_trend(data_points) {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        tracing::warn!("Algorithm {} failed: {}", algorithm.name(), e);
                    },
                }
            }
        }

        if results.is_empty() {
            return Err(anyhow::anyhow!("No algorithms could analyze the trend"));
        }

        // Aggregate results
        let aggregated_result = self.aggregate_trend_results(&results)?;

        // Cache the result
        self.cache_trend_analysis(&cache_key, &aggregated_result).await;

        Ok(aggregated_result)
    }

    /// Predict future trend
    pub async fn predict_trend(
        &self,
        current_trend: &PerformanceTrend,
        horizon: Duration,
    ) -> Result<TrendPrediction> {
        let config = self.config.read();

        if !config.enable_prediction {
            return Err(anyhow::anyhow!("Trend prediction is disabled"));
        }

        let models = self.prediction_models.lock();
        let mut predictions = Vec::new();

        for model in models.iter() {
            if model.prediction_confidence(current_trend) > config.confidence_threshold {
                match model.predict_trend(current_trend, horizon) {
                    Ok(prediction) => predictions.push(prediction),
                    Err(e) => {
                        tracing::warn!("Prediction model {} failed: {}", model.name(), e);
                    },
                }
            }
        }

        if predictions.is_empty() {
            return Err(anyhow::anyhow!(
                "No prediction models could generate predictions"
            ));
        }

        // Aggregate predictions
        self.aggregate_predictions(&predictions)
    }

    /// Add trend detection algorithm
    pub fn add_algorithm(&self, algorithm: Box<dyn TrendDetectionAlgorithm + Send + Sync>) {
        let mut algorithms = self.algorithms.lock();
        algorithms.push(algorithm);
    }

    /// Add trend prediction model
    pub fn add_prediction_model(&self, model: Box<dyn TrendPredictionModel + Send + Sync>) {
        let mut models = self.prediction_models.lock();
        models.push(model);
    }

    /// Update configuration
    pub fn update_config(&self, new_config: TrendAnalysisConfig) {
        let mut config = self.config.write();
        *config = new_config;
    }

    /// Clear trend cache
    pub async fn clear_cache(&self) {
        let mut cache = self.trend_cache.write();
        cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStatistics {
        let cache = self.trend_cache.read();
        let current_time = Utc::now();
        let mut expired_count = 0;

        for cached in cache.values() {
            let age = current_time.signed_duration_since(cached.cached_at);
            if age.to_std().unwrap_or_default() > self.config.read().cache_expiry {
                expired_count += 1;
            }
        }

        CacheStatistics {
            total_entries: cache.len(),
            expired_entries: expired_count,
            hit_rate: 0.0, // Would need to track hits/misses
            memory_usage: cache.len() * std::mem::size_of::<CachedTrendAnalysis>(),
        }
    }

    /// Initialize default trend detection algorithms
    fn initialize_default_algorithms(&mut self) {
        let mut algorithms = self.algorithms.lock();

        algorithms.push(Box::new(LinearRegressionTrendDetector::new()));
        algorithms.push(Box::new(MovingAverageTrendDetector::new()));
        algorithms.push(Box::new(ExponentialSmoothingTrendDetector::new()));
        algorithms.push(Box::new(StatisticalTrendDetector::new()));
    }

    /// Initialize default prediction models
    fn initialize_default_models(&mut self) {
        let mut models = self.prediction_models.lock();

        models.push(Box::new(LinearRegressionPredictor::new()));
        models.push(Box::new(MovingAveragePredictor::new()));
        models.push(Box::new(ExponentialSmoothingPredictor::new()));
    }

    /// Generate cache key for data points
    fn generate_cache_key(&self, data_points: &[PerformanceDataPoint]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for point in data_points {
            point.throughput.to_bits().hash(&mut hasher);
            point.latency.as_nanos().hash(&mut hasher);
            point.timestamp.timestamp_nanos_opt().unwrap_or(0).hash(&mut hasher);
        }

        format!("trend_{}_{}", hasher.finish(), data_points.len())
    }

    /// Get cached trend analysis
    fn get_cached_analysis(&self, cache_key: &str) -> Option<CachedTrendAnalysis> {
        let cache = self.trend_cache.read();
        let config = self.config.read();

        if let Some(cached) = cache.get(cache_key) {
            let age = Utc::now().signed_duration_since(cached.cached_at);
            if age.to_std().unwrap_or_default() <= config.cache_expiry {
                return Some(cached.clone());
            }
        }

        None
    }

    /// Cache trend analysis result
    async fn cache_trend_analysis(&self, cache_key: &str, result: &TrendAnalysisResult) {
        let mut cache = self.trend_cache.write();

        let cached = CachedTrendAnalysis {
            trend: result.clone().into(),
            cached_at: Utc::now(),
            analysis_duration: Duration::from_millis(0), // Would track actual duration
            confidence: result.confidence,
        };

        cache.insert(cache_key.to_string(), cached);

        // Cleanup expired entries
        let config = self.config.read();
        let current_time = Utc::now();
        cache.retain(|_, cached_analysis| {
            let age = current_time.signed_duration_since(cached_analysis.cached_at);
            age.to_std().unwrap_or_default() <= config.cache_expiry
        });
    }

    /// Aggregate multiple trend analysis results
    fn aggregate_trend_results(
        &self,
        results: &[TrendAnalysisResult],
    ) -> Result<TrendAnalysisResult> {
        if results.is_empty() {
            return Err(anyhow::anyhow!("No results to aggregate"));
        }

        // Weighted aggregation based on confidence
        let total_confidence: f32 = results.iter().map(|r| r.confidence).sum();

        let mut weighted_strength = 0.0f32;
        let mut weighted_significance = 0.0f32;
        let mut all_data_points = Vec::new();
        let mut all_metrics = HashMap::new();
        let mut methods = Vec::new();

        for result in results {
            let weight = result.confidence / total_confidence;
            weighted_strength += result.strength * weight;
            weighted_significance += result.significance * weight;
            all_data_points.extend(result.data_points.clone());
            methods.push(result.method.clone());

            for (key, value) in &result.metrics {
                *all_metrics.entry(key.clone()).or_insert(0.0) += value * weight as f64;
            }
        }

        // Determine consensus direction
        let direction_votes: HashMap<TrendDirection, usize> =
            results.iter().map(|r| r.direction).fold(HashMap::new(), |mut acc, direction| {
                *acc.entry(direction).or_insert(0) += 1;
                acc
            });

        let consensus_direction = direction_votes
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(direction, _)| direction)
            .unwrap_or(TrendDirection::Stable);

        Ok(TrendAnalysisResult {
            direction: consensus_direction,
            strength: weighted_strength,
            confidence: total_confidence / results.len() as f32,
            duration: results.iter().map(|r| r.duration).max().unwrap_or_default(),
            significance: weighted_significance,
            data_points: all_data_points,
            method: format!("Aggregated({})", methods.join(", ")),
            metrics: all_metrics,
        })
    }

    /// Aggregate multiple trend predictions
    fn aggregate_predictions(&self, predictions: &[TrendPrediction]) -> Result<TrendPrediction> {
        if predictions.is_empty() {
            return Err(anyhow::anyhow!("No predictions to aggregate"));
        }

        // Weighted aggregation based on confidence
        let total_confidence: f32 = predictions.iter().map(|p| p.confidence).sum();

        let mut weighted_uncertainty = 0.0f32;
        let mut all_expected_values = Vec::new();
        let mut models = Vec::new();

        for prediction in predictions {
            let weight = prediction.confidence / total_confidence;
            weighted_uncertainty += prediction.uncertainty * weight;
            all_expected_values.extend(prediction.expected_values.clone());
            models.push(prediction.model.clone());
        }

        // Determine consensus direction
        let direction_votes: HashMap<TrendDirection, usize> = predictions
            .iter()
            .map(|p| p.predicted_direction)
            .fold(HashMap::new(), |mut acc, direction| {
                *acc.entry(direction).or_insert(0) += 1;
                acc
            });

        let consensus_direction = direction_votes
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(direction, _)| direction)
            .unwrap_or(TrendDirection::Stable);

        // Aggregate expected values by timestamp
        let mut value_groups: HashMap<DateTime<Utc>, Vec<PredictedPerformancePoint>> =
            HashMap::new();
        for value in all_expected_values {
            value_groups.entry(value.timestamp).or_default().push(value);
        }

        let aggregated_values: Vec<PredictedPerformancePoint> = value_groups
            .into_iter()
            .map(|(timestamp, points)| {
                let avg_throughput = points.iter().map(|p| p.predicted_throughput).sum::<f64>()
                    / points.len() as f64;
                let avg_latency = Duration::from_nanos(
                    (points.iter().map(|p| p.predicted_latency.as_nanos()).sum::<u128>()
                        / points.len() as u128) as u64,
                );
                let avg_confidence =
                    points.iter().map(|p| p.confidence).sum::<f32>() / points.len() as f32;

                // Calculate confidence interval
                let throughputs: Vec<f64> = points.iter().map(|p| p.predicted_throughput).collect();
                let std_dev = calculate_standard_deviation(&throughputs);
                let margin = 1.96 * std_dev; // 95% confidence interval

                PredictedPerformancePoint {
                    timestamp,
                    predicted_throughput: avg_throughput,
                    predicted_latency: avg_latency,
                    confidence: avg_confidence,
                    confidence_interval: (avg_throughput - margin, avg_throughput + margin),
                }
            })
            .collect();

        Ok(TrendPrediction {
            predicted_direction: consensus_direction,
            confidence: total_confidence / predictions.len() as f32,
            horizon: predictions.iter().map(|p| p.horizon).max().unwrap_or_default(),
            expected_values: aggregated_values,
            model: format!("Aggregated({})", models.join(", ")),
            uncertainty: weighted_uncertainty,
        })
    }
}

impl Default for EnhancedTrendAnalysisEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TREND DETECTION ALGORITHM IMPLEMENTATIONS
// =============================================================================

/// Linear regression trend detector
pub struct LinearRegressionTrendDetector;

impl Default for LinearRegressionTrendDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegressionTrendDetector {
    pub fn new() -> Self {
        Self
    }
}

impl TrendDetectionAlgorithm for LinearRegressionTrendDetector {
    fn detect_trend(&self, data_points: &[PerformanceDataPoint]) -> Result<TrendAnalysisResult> {
        if data_points.len() < 2 {
            return Err(anyhow::anyhow!(
                "Insufficient data points for linear regression"
            ));
        }

        // Convert to time series
        let times: Vec<f64> = data_points.iter().enumerate().map(|(i, _)| i as f64).collect();
        let values: Vec<f64> = data_points.iter().map(|p| p.throughput).collect();

        // Calculate linear regression
        let (slope, intercept, r_squared) = calculate_linear_regression(&times, &values)?;

        // Determine trend direction and strength
        let direction = if slope > 0.1 {
            TrendDirection::Increasing
        } else if slope < -0.1 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        let strength = slope.abs().min(1.0) as f32;
        let confidence = r_squared as f32;

        let mut metrics = HashMap::new();
        metrics.insert("slope".to_string(), slope);
        metrics.insert("intercept".to_string(), intercept);
        metrics.insert("r_squared".to_string(), r_squared);

        Ok(TrendAnalysisResult {
            direction,
            strength,
            confidence,
            duration: Duration::from_secs((data_points.len() * 60) as u64), // Assume 1-minute intervals
            significance: confidence,
            data_points: data_points.to_vec(),
            method: "Linear Regression".to_string(),
            metrics,
        })
    }

    fn name(&self) -> &str {
        "linear_regression"
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

    fn is_applicable(&self, data_points: &[PerformanceDataPoint]) -> bool {
        data_points.len() >= 2
    }
}

/// Moving average trend detector
pub struct MovingAverageTrendDetector {
    window_size: usize,
}

impl Default for MovingAverageTrendDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl MovingAverageTrendDetector {
    pub fn new() -> Self {
        Self { window_size: 5 }
    }

    pub fn with_window_size(window_size: usize) -> Self {
        Self { window_size }
    }
}

impl TrendDetectionAlgorithm for MovingAverageTrendDetector {
    fn detect_trend(&self, data_points: &[PerformanceDataPoint]) -> Result<TrendAnalysisResult> {
        if data_points.len() < self.window_size {
            return Err(anyhow::anyhow!(
                "Insufficient data points for moving average"
            ));
        }

        let values: Vec<f64> = data_points.iter().map(|p| p.throughput).collect();
        let moving_averages = calculate_moving_average(&values, self.window_size);

        // Analyze trend in moving averages
        let first_avg = moving_averages
            .first()
            .ok_or_else(|| anyhow::anyhow!("Moving averages is empty"))?;
        let last_avg = moving_averages
            .last()
            .ok_or_else(|| anyhow::anyhow!("Moving averages is empty"))?;
        let change_ratio = (last_avg - first_avg) / first_avg;

        let direction = if change_ratio > 0.05 {
            TrendDirection::Increasing
        } else if change_ratio < -0.05 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        let strength = change_ratio.abs().min(1.0) as f32;
        let confidence = calculate_trend_confidence(&moving_averages);

        let mut metrics = HashMap::new();
        metrics.insert("change_ratio".to_string(), change_ratio);
        metrics.insert("first_avg".to_string(), *first_avg);
        metrics.insert("last_avg".to_string(), *last_avg);

        Ok(TrendAnalysisResult {
            direction,
            strength,
            confidence,
            duration: Duration::from_secs((data_points.len() * 60) as u64),
            significance: confidence,
            data_points: data_points.to_vec(),
            method: format!("Moving Average (window={})", self.window_size),
            metrics,
        })
    }

    fn name(&self) -> &str {
        "moving_average"
    }

    fn confidence(&self, data_points: &[PerformanceDataPoint]) -> f32 {
        if data_points.len() >= self.window_size * 2 {
            0.8
        } else {
            0.5
        }
    }

    fn is_applicable(&self, data_points: &[PerformanceDataPoint]) -> bool {
        data_points.len() >= self.window_size
    }
}

/// Exponential smoothing trend detector
pub struct ExponentialSmoothingTrendDetector {
    alpha: f64,
}

impl Default for ExponentialSmoothingTrendDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ExponentialSmoothingTrendDetector {
    pub fn new() -> Self {
        Self { alpha: 0.3 }
    }

    pub fn with_alpha(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl TrendDetectionAlgorithm for ExponentialSmoothingTrendDetector {
    fn detect_trend(&self, data_points: &[PerformanceDataPoint]) -> Result<TrendAnalysisResult> {
        if data_points.len() < 3 {
            return Err(anyhow::anyhow!(
                "Insufficient data points for exponential smoothing"
            ));
        }

        let values: Vec<f64> = data_points.iter().map(|p| p.throughput).collect();
        let smoothed_values = calculate_exponential_smoothing(&values, self.alpha);

        // Analyze trend in smoothed values
        let trend_slope = calculate_trend_slope(&smoothed_values);

        let direction = if trend_slope > 0.01 {
            TrendDirection::Increasing
        } else if trend_slope < -0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        let strength = trend_slope.abs().min(1.0) as f32;
        let confidence = calculate_smoothing_confidence(&values, &smoothed_values);

        let mut metrics = HashMap::new();
        metrics.insert("trend_slope".to_string(), trend_slope);
        metrics.insert("alpha".to_string(), self.alpha);

        Ok(TrendAnalysisResult {
            direction,
            strength,
            confidence,
            duration: Duration::from_secs((data_points.len() * 60) as u64),
            significance: confidence,
            data_points: data_points.to_vec(),
            method: format!("Exponential Smoothing (alpha={})", self.alpha),
            metrics,
        })
    }

    fn name(&self) -> &str {
        "exponential_smoothing"
    }

    fn confidence(&self, data_points: &[PerformanceDataPoint]) -> f32 {
        if data_points.len() >= 10 {
            0.85
        } else {
            0.6
        }
    }

    fn is_applicable(&self, data_points: &[PerformanceDataPoint]) -> bool {
        data_points.len() >= 3
    }
}

/// Statistical trend detector using correlation analysis
pub struct StatisticalTrendDetector;

impl Default for StatisticalTrendDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalTrendDetector {
    pub fn new() -> Self {
        Self
    }
}

impl TrendDetectionAlgorithm for StatisticalTrendDetector {
    fn detect_trend(&self, data_points: &[PerformanceDataPoint]) -> Result<TrendAnalysisResult> {
        if data_points.len() < 5 {
            return Err(anyhow::anyhow!(
                "Insufficient data points for statistical analysis"
            ));
        }

        let times: Vec<f64> = (0..data_points.len()).map(|i| i as f64).collect();
        let values: Vec<f64> = data_points.iter().map(|p| p.throughput).collect();

        // Calculate Pearson correlation coefficient
        let correlation = calculate_correlation(&times, &values)?;

        let direction = if correlation > 0.1 {
            TrendDirection::Increasing
        } else if correlation < -0.1 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        let strength = correlation.abs() as f32;
        let confidence = strength;

        // Additional statistical metrics
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = calculate_variance(&values, mean);
        let std_dev = variance.sqrt();

        let mut metrics = HashMap::new();
        metrics.insert("correlation".to_string(), correlation);
        metrics.insert("mean".to_string(), mean);
        metrics.insert("variance".to_string(), variance);
        metrics.insert("std_dev".to_string(), std_dev);

        Ok(TrendAnalysisResult {
            direction,
            strength,
            confidence,
            duration: Duration::from_secs((data_points.len() * 60) as u64),
            significance: confidence,
            data_points: data_points.to_vec(),
            method: "Statistical Correlation".to_string(),
            metrics,
        })
    }

    fn name(&self) -> &str {
        "statistical_correlation"
    }

    fn confidence(&self, data_points: &[PerformanceDataPoint]) -> f32 {
        if data_points.len() >= 20 {
            0.95
        } else if data_points.len() >= 10 {
            0.8
        } else {
            0.6
        }
    }

    fn is_applicable(&self, data_points: &[PerformanceDataPoint]) -> bool {
        data_points.len() >= 5
    }
}

// =============================================================================
// TREND PREDICTION MODEL IMPLEMENTATIONS
// =============================================================================

/// Linear regression predictor
pub struct LinearRegressionPredictor;

impl Default for LinearRegressionPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegressionPredictor {
    pub fn new() -> Self {
        Self
    }
}

impl TrendPredictionModel for LinearRegressionPredictor {
    fn predict_trend(
        &self,
        current_trend: &PerformanceTrend,
        horizon: Duration,
    ) -> Result<TrendPrediction> {
        // Extract slope from trend data
        let times: Vec<f64> = (0..current_trend.data_points.len()).map(|i| i as f64).collect();
        let values: Vec<f64> = current_trend.data_points.iter().map(|p| p.throughput).collect();

        let (slope, intercept, _) = calculate_linear_regression(&times, &values)?;

        // Generate predictions
        let horizon_minutes = horizon.as_secs() / 60;
        let mut expected_values = Vec::new();

        for i in 1..=horizon_minutes {
            let future_time = times.len() as f64 + i as f64;
            let predicted_value = slope * future_time + intercept;

            expected_values.push(PredictedPerformancePoint {
                timestamp: Utc::now() + chrono::Duration::minutes(i as i64),
                predicted_throughput: predicted_value.max(0.0),
                predicted_latency: Duration::from_millis(100), // Simplified
                confidence: current_trend.confidence * 0.9,    // Decrease over time
                confidence_interval: (predicted_value * 0.9, predicted_value * 1.1),
            });
        }

        Ok(TrendPrediction {
            predicted_direction: current_trend.direction,
            confidence: current_trend.confidence * 0.8,
            horizon,
            expected_values,
            model: "Linear Regression".to_string(),
            uncertainty: 0.2,
        })
    }

    fn name(&self) -> &str {
        "linear_regression_predictor"
    }

    fn prediction_confidence(&self, trend: &PerformanceTrend) -> f32 {
        trend.confidence * 0.8
    }
}

/// Moving average predictor
pub struct MovingAveragePredictor {
    window_size: usize,
}

impl Default for MovingAveragePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl MovingAveragePredictor {
    pub fn new() -> Self {
        Self { window_size: 5 }
    }
}

impl TrendPredictionModel for MovingAveragePredictor {
    fn predict_trend(
        &self,
        current_trend: &PerformanceTrend,
        horizon: Duration,
    ) -> Result<TrendPrediction> {
        let values: Vec<f64> = current_trend.data_points.iter().map(|p| p.throughput).collect();
        let recent_avg =
            values.iter().rev().take(self.window_size).sum::<f64>() / self.window_size as f64;

        let horizon_minutes = horizon.as_secs() / 60;
        let mut expected_values = Vec::new();

        for i in 1..=horizon_minutes {
            expected_values.push(PredictedPerformancePoint {
                timestamp: Utc::now() + chrono::Duration::minutes(i as i64),
                predicted_throughput: recent_avg,
                predicted_latency: Duration::from_millis(100),
                confidence: current_trend.confidence * 0.7,
                confidence_interval: (recent_avg * 0.8, recent_avg * 1.2),
            });
        }

        Ok(TrendPrediction {
            predicted_direction: TrendDirection::Stable,
            confidence: current_trend.confidence * 0.7,
            horizon,
            expected_values,
            model: format!("Moving Average (window={})", self.window_size),
            uncertainty: 0.3,
        })
    }

    fn name(&self) -> &str {
        "moving_average_predictor"
    }

    fn prediction_confidence(&self, trend: &PerformanceTrend) -> f32 {
        trend.confidence * 0.6
    }
}

/// Exponential smoothing predictor
pub struct ExponentialSmoothingPredictor {
    alpha: f64,
}

impl Default for ExponentialSmoothingPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl ExponentialSmoothingPredictor {
    pub fn new() -> Self {
        Self { alpha: 0.3 }
    }
}

impl TrendPredictionModel for ExponentialSmoothingPredictor {
    fn predict_trend(
        &self,
        current_trend: &PerformanceTrend,
        horizon: Duration,
    ) -> Result<TrendPrediction> {
        let values: Vec<f64> = current_trend.data_points.iter().map(|p| p.throughput).collect();
        let smoothed = calculate_exponential_smoothing(&values, self.alpha);
        let last_smoothed =
            *smoothed.last().ok_or_else(|| anyhow::anyhow!("Smoothed values is empty"))?;

        // Simple exponential smoothing prediction (constant)
        let horizon_minutes = horizon.as_secs() / 60;
        let mut expected_values = Vec::new();

        for i in 1..=horizon_minutes {
            expected_values.push(PredictedPerformancePoint {
                timestamp: Utc::now() + chrono::Duration::minutes(i as i64),
                predicted_throughput: last_smoothed,
                predicted_latency: Duration::from_millis(100),
                confidence: current_trend.confidence * 0.8,
                confidence_interval: (last_smoothed * 0.85, last_smoothed * 1.15),
            });
        }

        Ok(TrendPrediction {
            predicted_direction: current_trend.direction,
            confidence: current_trend.confidence * 0.75,
            horizon,
            expected_values,
            model: format!("Exponential Smoothing (alpha={})", self.alpha),
            uncertainty: 0.25,
        })
    }

    fn name(&self) -> &str {
        "exponential_smoothing_predictor"
    }

    fn prediction_confidence(&self, trend: &PerformanceTrend) -> f32 {
        trend.confidence * 0.75
    }
}

// =============================================================================
// UTILITY TYPES AND FUNCTIONS
// =============================================================================

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub total_entries: usize,
    pub expired_entries: usize,
    pub hit_rate: f64,
    pub memory_usage: usize,
}

/// Helper function to convert TrendAnalysisResult to PerformanceTrend
impl From<TrendAnalysisResult> for PerformanceTrend {
    fn from(result: TrendAnalysisResult) -> Self {
        PerformanceTrend {
            direction: result.direction,
            strength: result.strength,
            confidence: result.confidence,
            period: result.duration,
            data_points: result.data_points,
        }
    }
}

/// Helper function to convert PerformanceTrend to TrendAnalysisResult
impl From<PerformanceTrend> for TrendAnalysisResult {
    fn from(trend: PerformanceTrend) -> Self {
        TrendAnalysisResult {
            direction: trend.direction,
            strength: trend.strength,
            confidence: trend.confidence,
            duration: trend.period,
            significance: trend.confidence, // Use confidence as significance
            data_points: trend.data_points,
            method: "Cached".to_string(),
            metrics: HashMap::new(), // Empty metrics for cached results
        }
    }
}

// =============================================================================
// STATISTICAL UTILITY FUNCTIONS
// =============================================================================

/// Calculate linear regression coefficients
fn calculate_linear_regression(x: &[f64], y: &[f64]) -> Result<(f64, f64, f64)> {
    if x.len() != y.len() || x.is_empty() {
        return Err(anyhow::anyhow!("Invalid input data for linear regression"));
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();
    let _sum_y2: f64 = y.iter().map(|yi| yi * yi).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    // Calculate R-squared
    let y_mean = sum_y / n;
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
        .sum();
    let r_squared = 1.0 - (ss_res / ss_tot);

    Ok((slope, intercept, r_squared))
}

/// Calculate moving average
fn calculate_moving_average(values: &[f64], window_size: usize) -> Vec<f64> {
    if window_size > values.len() {
        return vec![values.iter().sum::<f64>() / values.len() as f64];
    }

    let mut moving_averages = Vec::new();
    for i in window_size..=values.len() {
        let window_sum: f64 = values[i - window_size..i].iter().sum();
        moving_averages.push(window_sum / window_size as f64);
    }
    moving_averages
}

/// Calculate exponential smoothing
fn calculate_exponential_smoothing(values: &[f64], alpha: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut smoothed = Vec::with_capacity(values.len());
    smoothed.push(values[0]);

    for i in 1..values.len() {
        let smoothed_value = alpha * values[i] + (1.0 - alpha) * smoothed[i - 1];
        smoothed.push(smoothed_value);
    }

    smoothed
}

/// Calculate trend confidence from moving averages
fn calculate_trend_confidence(moving_averages: &[f64]) -> f32 {
    if moving_averages.len() < 2 {
        return 0.5;
    }

    // Safe: we already check len() >= 2 above
    let first = match moving_averages.first() {
        Some(f) => f,
        None => return 0.5,
    };
    let last = match moving_averages.last() {
        Some(l) => l,
        None => return 0.5,
    };
    let change = (last - first).abs() / first;

    (change.min(1.0) * 0.8 + 0.2) as f32
}

/// Calculate trend slope from smoothed values
fn calculate_trend_slope(values: &[f64]) -> f64 {
    // Safe: we check len() >= 2 so first/last are guaranteed
    let (first, last) = match (values.first(), values.last()) {
        (Some(f), Some(l)) => (f, l),
        _ => return 0.0,
    };
    (last - first) / values.len() as f64
}

/// Calculate smoothing confidence
fn calculate_smoothing_confidence(original: &[f64], smoothed: &[f64]) -> f32 {
    if original.len() != smoothed.len() || original.is_empty() {
        return 0.5;
    }

    let mse: f64 = original.iter().zip(smoothed.iter()).map(|(o, s)| (o - s).powi(2)).sum::<f64>()
        / original.len() as f64;

    let variance = calculate_variance(
        original,
        original.iter().sum::<f64>() / original.len() as f64,
    );

    if variance > 0.0 {
        (1.0 - (mse / variance)).clamp(0.0, 1.0) as f32
    } else {
        0.5
    }
}

/// Calculate Pearson correlation coefficient
fn calculate_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() || x.is_empty() {
        return Err(anyhow::anyhow!("Invalid input for correlation calculation"));
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_y2: f64 = y.iter().map(|yi| yi * yi).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
}

/// Calculate variance
fn calculate_variance(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Calculate standard deviation
fn calculate_standard_deviation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = calculate_variance(values, mean);
    variance.sqrt()
}
