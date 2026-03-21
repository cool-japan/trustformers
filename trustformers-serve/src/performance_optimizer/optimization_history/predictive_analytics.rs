//! Predictive Analytics Engine
//!
//! This module provides comprehensive predictive analytics capabilities for optimization
//! history data, including multiple prediction models, performance forecasting, model
//! validation, and ensemble prediction methods. It enables data-driven forecasting and
//! proactive optimization planning.

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, sync::Arc, time::Duration};

use super::types::*;
use crate::performance_optimizer::types::PerformanceDataPoint;

// =============================================================================
// PREDICTIVE ANALYTICS ENGINE
// =============================================================================

/// Predictive analytics engine with multiple prediction models
///
/// Provides sophisticated forecasting capabilities using various prediction models,
/// ensemble methods, model validation, and performance prediction for optimization
/// planning and proactive decision-making.
pub struct PredictiveAnalyticsEngine {
    /// Prediction models
    models: Arc<Mutex<Vec<Box<dyn PredictiveModel + Send + Sync>>>>,
    /// Model performance tracking
    model_performance: Arc<RwLock<HashMap<String, ModelPerformanceMetrics>>>,
    /// Prediction cache
    prediction_cache: Arc<RwLock<HashMap<String, CachedPrediction>>>,
    /// Configuration
    config: Arc<RwLock<PredictiveAnalyticsConfig>>,
    /// Historical data for training
    training_data: Arc<RwLock<Vec<PerformanceDataPoint>>>,
}

impl PredictiveAnalyticsEngine {
    /// Create new predictive analytics engine
    pub fn new() -> Self {
        let mut engine = Self {
            models: Arc::new(Mutex::new(Vec::new())),
            model_performance: Arc::new(RwLock::new(HashMap::new())),
            prediction_cache: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(PredictiveAnalyticsConfig::default())),
            training_data: Arc::new(RwLock::new(Vec::new())),
        };

        // Initialize default models
        engine.initialize_default_models();

        engine
    }

    /// Create with custom configuration
    pub fn with_config(config: PredictiveAnalyticsConfig) -> Self {
        let mut engine = Self {
            models: Arc::new(Mutex::new(Vec::new())),
            model_performance: Arc::new(RwLock::new(HashMap::new())),
            prediction_cache: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(config)),
            training_data: Arc::new(RwLock::new(Vec::new())),
        };

        engine.initialize_default_models();

        engine
    }

    /// Train models with historical data
    pub async fn train_models(&mut self, data: &[PerformanceDataPoint]) -> Result<()> {
        let config = self.config.read();

        if data.len() < config.min_data_points {
            return Err(anyhow::anyhow!(
                "Insufficient training data: {} < {}",
                data.len(),
                config.min_data_points
            ));
        }

        // Store training data
        {
            let mut training_data = self.training_data.write();
            *training_data = data.to_vec();
        }

        // Train all models
        let mut models = self.models.lock();
        let mut model_performance = self.model_performance.write();

        for model in models.iter_mut() {
            match model.train(data) {
                Ok(_) => {
                    let performance = model.performance_metrics();
                    let model_type_name = format!("{:?}", model.model_type());
                    model_performance.insert(model_type_name, performance);
                    tracing::info!("Successfully trained model: {:?}", model.model_type());
                },
                Err(e) => {
                    tracing::warn!("Failed to train model {:?}: {}", model.model_type(), e);
                },
            }
        }

        // Update models periodically if configured
        if config.model_update_frequency > Duration::from_secs(0) {
            // In a real implementation, this would set up a periodic update task
            tracing::info!(
                "Model training completed. Next update in {:?}",
                config.model_update_frequency
            );
        }

        Ok(())
    }

    /// Generate predictions using all models
    pub async fn generate_predictions(
        &self,
        horizon: Duration,
    ) -> Result<Vec<PerformancePrediction>> {
        let config = self.config.read();

        if !config.enable_prediction {
            return Err(anyhow::anyhow!("Prediction is disabled"));
        }

        // Check cache first
        let cache_key = self.generate_cache_key(horizon);
        if let Some(cached) = self.get_cached_prediction(&cache_key) {
            return Ok(vec![cached.prediction]);
        }

        let models = self.models.lock();
        let mut predictions = Vec::new();

        for model in models.iter() {
            match model.predict(horizon) {
                Ok(prediction) => predictions.push(prediction),
                Err(e) => {
                    tracing::warn!("Model {:?} prediction failed: {}", model.model_type(), e);
                },
            }
        }

        if predictions.is_empty() {
            return Err(anyhow::anyhow!("No models could generate predictions"));
        }

        // Cache the predictions
        if predictions.len() == 1 {
            self.cache_prediction(&cache_key, &predictions[0]).await;
        }

        Ok(predictions)
    }

    /// Generate ensemble prediction
    pub async fn generate_ensemble_prediction(
        &self,
        horizon: Duration,
    ) -> Result<PerformancePrediction> {
        let predictions = self.generate_predictions(horizon).await?;

        if predictions.is_empty() {
            return Err(anyhow::anyhow!("No predictions available for ensemble"));
        }

        if predictions.len() == 1 {
            return Ok(predictions[0].clone());
        }

        // Ensemble prediction using weighted averaging based on model performance
        let model_performance = self.model_performance.read();
        let mut weights = Vec::new();
        let mut total_weight = 0.0f64;

        for prediction in &predictions {
            let model_type_str = format!("{:?}", prediction.model);
            let performance = model_performance.get(&model_type_str);

            let weight = if let Some(perf) = performance {
                // Use R-squared as weight (higher is better)
                perf.r_squared.max(0.1) // Minimum weight of 0.1
            } else {
                0.5 // Default weight
            };

            weights.push(weight);
            total_weight += weight;
        }

        // Normalize weights
        for weight in &mut weights {
            *weight /= total_weight;
        }

        // Create ensemble prediction
        let ensemble_prediction =
            self.create_ensemble_prediction(&predictions, &weights, horizon)?;

        // Cache the ensemble prediction
        let cache_key = format!("ensemble_{}", self.generate_cache_key(horizon));
        self.cache_prediction(&cache_key, &ensemble_prediction).await;

        Ok(ensemble_prediction)
    }

    /// Get best performing model
    pub fn get_best_model(&self) -> Option<PredictionModelType> {
        let model_performance = self.model_performance.read();

        model_performance
            .iter()
            .max_by(|a, b| {
                a.1.r_squared.partial_cmp(&b.1.r_squared).unwrap_or(std::cmp::Ordering::Equal)
            })
            .and_then(|(model_name, _)| model_name.parse().ok())
    }

    /// Get model performance summary
    pub fn get_performance_summary(&self) -> HashMap<PredictionModelType, ModelPerformanceMetrics> {
        let model_performance = self.model_performance.read();
        let mut summary = HashMap::new();

        for (model_name, performance) in model_performance.iter() {
            if let Ok(model_type) = model_name.parse::<PredictionModelType>() {
                summary.insert(model_type, performance.clone());
            }
        }

        summary
    }

    /// Add prediction model
    pub fn add_model(&self, model: Box<dyn PredictiveModel + Send + Sync>) {
        let mut models = self.models.lock();
        models.push(model);
    }

    /// Update configuration
    pub fn update_config(&self, new_config: PredictiveAnalyticsConfig) {
        let mut config = self.config.write();
        *config = new_config;
    }

    /// Clear prediction cache
    pub async fn clear_cache(&self) {
        let mut cache = self.prediction_cache.write();
        cache.clear();
    }

    /// Get prediction statistics
    pub fn get_prediction_statistics(&self) -> PredictionStatistics {
        let cache = self.prediction_cache.read();
        let model_performance = self.model_performance.read();

        let total_predictions = cache.len();
        let model_count = model_performance.len();

        let average_confidence = if !cache.is_empty() {
            cache.values().map(|p| p.prediction.confidence).sum::<f32>() / cache.len() as f32
        } else {
            0.0
        };

        let average_uncertainty = if !cache.is_empty() {
            cache.values().map(|p| p.prediction.uncertainty).sum::<f32>() / cache.len() as f32
        } else {
            0.0
        };

        PredictionStatistics {
            total_predictions,
            active_models: model_count,
            average_confidence,
            average_uncertainty,
            cache_memory_usage: cache.len() * std::mem::size_of::<CachedPrediction>(),
        }
    }

    /// Initialize default prediction models
    fn initialize_default_models(&mut self) {
        let mut models = self.models.lock();

        models.push(Box::new(LinearRegressionModel::new()));
        models.push(Box::new(MovingAverageModel::new()));
        models.push(Box::new(ExponentialSmoothingModel::new()));
        models.push(Box::new(ARIMAModel::new()));
    }

    /// Generate cache key for predictions
    fn generate_cache_key(&self, horizon: Duration) -> String {
        format!("prediction_{}s", horizon.as_secs())
    }

    /// Get cached prediction
    fn get_cached_prediction(&self, cache_key: &str) -> Option<CachedPrediction> {
        let cache = self.prediction_cache.read();
        let config = self.config.read();

        if let Some(cached) = cache.get(cache_key) {
            let age = Utc::now().signed_duration_since(cached.cached_at);
            if age.to_std().unwrap_or_default() <= config.model_update_frequency {
                return Some(cached.clone());
            }
        }

        None
    }

    /// Cache prediction result
    async fn cache_prediction(&self, cache_key: &str, prediction: &PerformancePrediction) {
        let mut cache = self.prediction_cache.write();

        let cached = CachedPrediction {
            prediction: prediction.clone(),
            cached_at: Utc::now(),
        };

        cache.insert(cache_key.to_string(), cached);

        // Maintain cache size (keep last 100 predictions)
        if cache.len() > 100 {
            let mut predictions: Vec<_> = cache.values().cloned().collect();
            predictions.sort_by_key(|p| p.cached_at);

            let to_remove = cache.len() - 100;
            for cached_pred in predictions.iter().take(to_remove) {
                let keys_to_remove: Vec<String> = cache
                    .iter()
                    .filter(|(_, v)| v.cached_at == cached_pred.cached_at)
                    .map(|(k, _)| k.clone())
                    .collect();

                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }
    }

    /// Create ensemble prediction from multiple predictions
    fn create_ensemble_prediction(
        &self,
        predictions: &[PerformancePrediction],
        weights: &[f64],
        horizon: Duration,
    ) -> Result<PerformancePrediction> {
        if predictions.is_empty() || weights.is_empty() || predictions.len() != weights.len() {
            return Err(anyhow::anyhow!("Invalid inputs for ensemble prediction"));
        }

        // Aggregate predicted values using weighted averaging
        let mut aggregated_values = HashMap::new();

        for (prediction, &weight) in predictions.iter().zip(weights.iter()) {
            for predicted_point in &prediction.predicted_values {
                let entry = aggregated_values
                    .entry(predicted_point.timestamp)
                    .or_insert(WeightedPredictionAggregator::new());
                entry.add_prediction(predicted_point, weight as f32);
            }
        }

        // Convert aggregated values to final predicted points
        let mut final_predicted_values: Vec<PredictedPerformancePoint> = aggregated_values
            .into_iter()
            .map(|(timestamp, aggregator)| aggregator.finalize(timestamp))
            .collect();

        // Sort by timestamp
        final_predicted_values.sort_by_key(|p| p.timestamp);

        // Calculate ensemble confidence and uncertainty
        let weighted_confidence = predictions
            .iter()
            .zip(weights.iter())
            .map(|(p, w)| p.confidence * *w as f32)
            .sum::<f32>();

        let weighted_uncertainty = predictions
            .iter()
            .zip(weights.iter())
            .map(|(p, w)| p.uncertainty * *w as f32)
            .sum::<f32>();

        Ok(PerformancePrediction {
            id: format!("ensemble_{}", Utc::now().timestamp()),
            predicted_values: final_predicted_values,
            model: PredictionModelType::Custom("Ensemble".to_string()),
            confidence: weighted_confidence,
            horizon,
            uncertainty: weighted_uncertainty,
            predicted_at: Utc::now(),
        })
    }
}

impl Default for PredictiveAnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// PREDICTION MODEL IMPLEMENTATIONS
// =============================================================================

/// Linear regression prediction model
pub struct LinearRegressionModel {
    slope: f64,
    intercept: f64,
    r_squared: f64,
    trained: bool,
}

impl Default for LinearRegressionModel {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegressionModel {
    pub fn new() -> Self {
        Self {
            slope: 0.0,
            intercept: 0.0,
            r_squared: 0.0,
            trained: false,
        }
    }
}

impl PredictiveModel for LinearRegressionModel {
    fn train(&mut self, data: &[PerformanceDataPoint]) -> Result<()> {
        if data.len() < 2 {
            return Err(anyhow::anyhow!(
                "Insufficient data for linear regression training"
            ));
        }

        let x_values: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();
        let y_values: Vec<f64> = data.iter().map(|p| p.throughput).collect();

        let (slope, intercept, r_squared) = calculate_linear_regression(&x_values, &y_values)?;

        self.slope = slope;
        self.intercept = intercept;
        self.r_squared = r_squared;
        self.trained = true;

        Ok(())
    }

    fn predict(&self, horizon: Duration) -> Result<PerformancePrediction> {
        if !self.trained {
            return Err(anyhow::anyhow!("Model must be trained before prediction"));
        }

        let horizon_minutes = horizon.as_secs() / 60;
        let mut predicted_values = Vec::new();

        for i in 1..=horizon_minutes {
            let x = i as f64;
            let predicted_throughput = self.slope * x + self.intercept;

            predicted_values.push(PredictedPerformancePoint {
                timestamp: Utc::now() + chrono::Duration::minutes(i as i64),
                predicted_throughput: predicted_throughput.max(0.0),
                predicted_latency: Duration::from_millis(100), // Simplified
                confidence: self.r_squared as f32,
                confidence_interval: (predicted_throughput * 0.9, predicted_throughput * 1.1),
            });
        }

        Ok(PerformancePrediction {
            id: format!("linear_regression_{}", Utc::now().timestamp()),
            predicted_values,
            model: PredictionModelType::LinearRegression,
            confidence: self.r_squared as f32,
            horizon,
            uncertainty: 1.0 - self.r_squared as f32,
            predicted_at: Utc::now(),
        })
    }

    fn model_type(&self) -> PredictionModelType {
        PredictionModelType::LinearRegression
    }

    fn performance_metrics(&self) -> ModelPerformanceMetrics {
        ModelPerformanceMetrics {
            mae: 0.0, // Would be calculated during validation
            mse: 0.0,
            rmse: 0.0,
            r_squared: self.r_squared,
            mape: 0.0,
        }
    }
}

/// Moving average prediction model
pub struct MovingAverageModel {
    window_size: usize,
    recent_values: Vec<f64>,
    average: f64,
    trained: bool,
}

impl Default for MovingAverageModel {
    fn default() -> Self {
        Self::new()
    }
}

impl MovingAverageModel {
    pub fn new() -> Self {
        Self {
            window_size: 10,
            recent_values: Vec::new(),
            average: 0.0,
            trained: false,
        }
    }

    pub fn with_window_size(window_size: usize) -> Self {
        Self {
            window_size,
            recent_values: Vec::new(),
            average: 0.0,
            trained: false,
        }
    }
}

impl PredictiveModel for MovingAverageModel {
    fn train(&mut self, data: &[PerformanceDataPoint]) -> Result<()> {
        if data.len() < self.window_size {
            return Err(anyhow::anyhow!(
                "Insufficient data for moving average training"
            ));
        }

        let values: Vec<f64> = data.iter().map(|p| p.throughput).collect();
        self.recent_values = values.iter().rev().take(self.window_size).cloned().collect();
        self.recent_values.reverse();

        self.average = self.recent_values.iter().sum::<f64>() / self.recent_values.len() as f64;
        self.trained = true;

        Ok(())
    }

    fn predict(&self, horizon: Duration) -> Result<PerformancePrediction> {
        if !self.trained {
            return Err(anyhow::anyhow!("Model must be trained before prediction"));
        }

        let horizon_minutes = horizon.as_secs() / 60;
        let mut predicted_values = Vec::new();

        for i in 1..=horizon_minutes {
            predicted_values.push(PredictedPerformancePoint {
                timestamp: Utc::now() + chrono::Duration::minutes(i as i64),
                predicted_throughput: self.average,
                predicted_latency: Duration::from_millis(100),
                confidence: 0.6, // Lower confidence for simple average
                confidence_interval: (self.average * 0.8, self.average * 1.2),
            });
        }

        Ok(PerformancePrediction {
            id: format!("moving_average_{}", Utc::now().timestamp()),
            predicted_values,
            model: PredictionModelType::MovingAverage,
            confidence: 0.6,
            horizon,
            uncertainty: 0.4,
            predicted_at: Utc::now(),
        })
    }

    fn model_type(&self) -> PredictionModelType {
        PredictionModelType::MovingAverage
    }

    fn performance_metrics(&self) -> ModelPerformanceMetrics {
        ModelPerformanceMetrics {
            mae: 0.0,
            mse: 0.0,
            rmse: 0.0,
            r_squared: 0.4, // Typical for moving average
            mape: 0.0,
        }
    }
}

/// Exponential smoothing prediction model
pub struct ExponentialSmoothingModel {
    alpha: f64,
    smoothed_value: f64,
    trend: f64,
    trained: bool,
}

impl Default for ExponentialSmoothingModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ExponentialSmoothingModel {
    pub fn new() -> Self {
        Self {
            alpha: 0.3,
            smoothed_value: 0.0,
            trend: 0.0,
            trained: false,
        }
    }

    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha,
            smoothed_value: 0.0,
            trend: 0.0,
            trained: false,
        }
    }
}

impl PredictiveModel for ExponentialSmoothingModel {
    fn train(&mut self, data: &[PerformanceDataPoint]) -> Result<()> {
        if data.len() < 2 {
            return Err(anyhow::anyhow!(
                "Insufficient data for exponential smoothing training"
            ));
        }

        let values: Vec<f64> = data.iter().map(|p| p.throughput).collect();

        // Initialize with first value
        self.smoothed_value = values[0];
        self.trend = 0.0;

        // Apply exponential smoothing
        for i in 1..values.len() {
            let previous_smoothed = self.smoothed_value;
            self.smoothed_value = self.alpha * values[i] + (1.0 - self.alpha) * self.smoothed_value;

            // Calculate trend (Holt's method)
            let beta = 0.2; // Trend smoothing parameter
            let _previous_trend = self.trend;
            self.trend =
                beta * (self.smoothed_value - previous_smoothed) + (1.0 - beta) * self.trend;
        }

        self.trained = true;

        Ok(())
    }

    fn predict(&self, horizon: Duration) -> Result<PerformancePrediction> {
        if !self.trained {
            return Err(anyhow::anyhow!("Model must be trained before prediction"));
        }

        let horizon_minutes = horizon.as_secs() / 60;
        let mut predicted_values = Vec::new();

        for i in 1..=horizon_minutes {
            let forecast = self.smoothed_value + self.trend * i as f64;

            predicted_values.push(PredictedPerformancePoint {
                timestamp: Utc::now() + chrono::Duration::minutes(i as i64),
                predicted_throughput: forecast.max(0.0),
                predicted_latency: Duration::from_millis(100),
                confidence: 0.7,
                confidence_interval: (forecast * 0.85, forecast * 1.15),
            });
        }

        Ok(PerformancePrediction {
            id: format!("exponential_smoothing_{}", Utc::now().timestamp()),
            predicted_values,
            model: PredictionModelType::ExponentialSmoothing,
            confidence: 0.7,
            horizon,
            uncertainty: 0.3,
            predicted_at: Utc::now(),
        })
    }

    fn model_type(&self) -> PredictionModelType {
        PredictionModelType::ExponentialSmoothing
    }

    fn performance_metrics(&self) -> ModelPerformanceMetrics {
        ModelPerformanceMetrics {
            mae: 0.0,
            mse: 0.0,
            rmse: 0.0,
            r_squared: 0.6, // Typical for exponential smoothing
            mape: 0.0,
        }
    }
}

/// ARIMA prediction model (simplified implementation)
pub struct ARIMAModel {
    ar_coefficients: Vec<f64>,
    ma_coefficients: Vec<f64>,
    differenced_data: Vec<f64>,
    trained: bool,
}

impl Default for ARIMAModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ARIMAModel {
    pub fn new() -> Self {
        Self {
            ar_coefficients: vec![0.5], // Simple AR(1)
            ma_coefficients: vec![0.3], // Simple MA(1)
            differenced_data: Vec::new(),
            trained: false,
        }
    }
}

impl PredictiveModel for ARIMAModel {
    fn train(&mut self, data: &[PerformanceDataPoint]) -> Result<()> {
        if data.len() < 5 {
            return Err(anyhow::anyhow!("Insufficient data for ARIMA training"));
        }

        let values: Vec<f64> = data.iter().map(|p| p.throughput).collect();

        // Simple differencing to make series stationary
        self.differenced_data = values.windows(2).map(|w| w[1] - w[0]).collect();

        // Simplified parameter estimation (in practice, would use MLE or similar)
        if self.differenced_data.len() >= 2 {
            let mean =
                self.differenced_data.iter().sum::<f64>() / self.differenced_data.len() as f64;

            // Simple AR(1) coefficient estimation
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for i in 1..self.differenced_data.len() {
                let lag1 = self.differenced_data[i - 1] - mean;
                let current = self.differenced_data[i] - mean;
                numerator += lag1 * current;
                denominator += lag1 * lag1;
            }

            if denominator > 0.0 {
                self.ar_coefficients[0] = (numerator / denominator).clamp(-0.9, 0.9);
            }
        }

        self.trained = true;

        Ok(())
    }

    fn predict(&self, horizon: Duration) -> Result<PerformancePrediction> {
        if !self.trained {
            return Err(anyhow::anyhow!("Model must be trained before prediction"));
        }

        let horizon_minutes = horizon.as_secs() / 60;
        let mut predicted_values = Vec::new();

        // Get last few values for prediction
        let last_value = self.differenced_data.last().unwrap_or(&0.0);
        let mut current_diff = *last_value;

        for i in 1..=horizon_minutes {
            // Simple ARIMA(1,1,0) prediction
            current_diff *= self.ar_coefficients[0];

            // Convert back to level (reverse differencing)
            // This is a simplified approach
            let predicted_throughput = 100.0 + current_diff; // Assume base level of 100

            predicted_values.push(PredictedPerformancePoint {
                timestamp: Utc::now() + chrono::Duration::minutes(i as i64),
                predicted_throughput: predicted_throughput.max(0.0),
                predicted_latency: Duration::from_millis(100),
                confidence: 0.65,
                confidence_interval: (predicted_throughput * 0.8, predicted_throughput * 1.2),
            });
        }

        Ok(PerformancePrediction {
            id: format!("arima_{}", Utc::now().timestamp()),
            predicted_values,
            model: PredictionModelType::ARIMA,
            confidence: 0.65,
            horizon,
            uncertainty: 0.35,
            predicted_at: Utc::now(),
        })
    }

    fn model_type(&self) -> PredictionModelType {
        PredictionModelType::ARIMA
    }

    fn performance_metrics(&self) -> ModelPerformanceMetrics {
        ModelPerformanceMetrics {
            mae: 0.0,
            mse: 0.0,
            rmse: 0.0,
            r_squared: 0.55, // Typical for ARIMA
            mape: 0.0,
        }
    }
}

// =============================================================================
// UTILITY TYPES AND FUNCTIONS
// =============================================================================

/// Cached prediction with timestamp
#[derive(Debug, Clone)]
struct CachedPrediction {
    prediction: PerformancePrediction,
    cached_at: DateTime<Utc>,
}

/// Weighted prediction aggregator for ensemble predictions
#[derive(Debug)]
struct WeightedPredictionAggregator {
    weighted_throughput: f64,
    weighted_latency: f64,
    weighted_confidence: f32,
    total_weight: f32,
    confidence_intervals: Vec<(f64, f64)>,
}

impl WeightedPredictionAggregator {
    fn new() -> Self {
        Self {
            weighted_throughput: 0.0,
            weighted_latency: 0.0,
            weighted_confidence: 0.0,
            total_weight: 0.0,
            confidence_intervals: Vec::new(),
        }
    }

    fn add_prediction(&mut self, point: &PredictedPerformancePoint, weight: f32) {
        self.weighted_throughput += point.predicted_throughput * weight as f64;
        self.weighted_latency += point.predicted_latency.as_millis() as f64 * weight as f64;
        self.weighted_confidence += point.confidence * weight;
        self.total_weight += weight;
        self.confidence_intervals.push(point.confidence_interval);
    }

    fn finalize(self, timestamp: DateTime<Utc>) -> PredictedPerformancePoint {
        let final_throughput = if self.total_weight > 0.0 {
            self.weighted_throughput / self.total_weight as f64
        } else {
            0.0
        };

        let final_latency = if self.total_weight > 0.0 {
            Duration::from_millis((self.weighted_latency / self.total_weight as f64) as u64)
        } else {
            Duration::from_millis(100)
        };

        let final_confidence = if self.total_weight > 0.0 {
            self.weighted_confidence / self.total_weight
        } else {
            0.5
        };

        // Aggregate confidence intervals
        let (min_lower, max_upper) = if !self.confidence_intervals.is_empty() {
            let min_lower =
                self.confidence_intervals.iter().map(|ci| ci.0).fold(f64::INFINITY, f64::min);
            let max_upper = self
                .confidence_intervals
                .iter()
                .map(|ci| ci.1)
                .fold(f64::NEG_INFINITY, f64::max);
            (min_lower, max_upper)
        } else {
            (final_throughput * 0.9, final_throughput * 1.1)
        };

        PredictedPerformancePoint {
            timestamp,
            predicted_throughput: final_throughput,
            predicted_latency: final_latency,
            confidence: final_confidence,
            confidence_interval: (min_lower, max_upper),
        }
    }
}

/// Prediction statistics
#[derive(Debug, Clone)]
pub struct PredictionStatistics {
    pub total_predictions: usize,
    pub active_models: usize,
    pub average_confidence: f32,
    pub average_uncertainty: f32,
    pub cache_memory_usage: usize,
}

/// Parse PredictionModelType from string (for model performance tracking)
impl std::str::FromStr for PredictionModelType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "LinearRegression" => Ok(PredictionModelType::LinearRegression),
            "MovingAverage" => Ok(PredictionModelType::MovingAverage),
            "ExponentialSmoothing" => Ok(PredictionModelType::ExponentialSmoothing),
            "ARIMA" => Ok(PredictionModelType::ARIMA),
            "NeuralNetwork" => Ok(PredictionModelType::NeuralNetwork),
            other => Ok(PredictionModelType::Custom(other.to_string())),
        }
    }
}

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

    let r_squared = if ss_tot > 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };

    Ok((slope, intercept, r_squared.max(0.0)))
}
