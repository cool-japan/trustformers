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
                    let model_type_name = format!("{:?}", model.model_type());
                    // A model that reports no fit is not recorded: the ensemble
                    // weights by `r_squared`, and an absent measurement must not
                    // be silently substituted with a number.
                    match model.performance_metrics() {
                        Some(performance) => {
                            model_performance.insert(model_type_name, performance);
                            tracing::info!("Successfully trained model: {:?}", model.model_type());
                        },
                        None => {
                            model_performance.remove(&model_type_name);
                            tracing::warn!(
                                "Model {:?} trained but reported no measurable fit",
                                model.model_type()
                            );
                        },
                    }
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
            .filter_map(|(timestamp, aggregator)| aggregator.finalize(timestamp))
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

/// In-sample fit of a model's one-step-ahead predictions against what actually
/// happened.
///
/// ## Added in 0.2.1
///
/// Every `PredictiveModel::performance_metrics` in this file used to return
/// `mae: 0.0, mse: 0.0, rmse: 0.0, mape: 0.0` with an `r_squared` that was a
/// hardcoded "typical" figure for the model family (0.4 for moving average,
/// 0.6 for exponential smoothing, 0.55 for ARIMA) rather than anything measured
/// on the data the model was just trained on. Those numbers are load-bearing:
/// `PerformanceHistoryAnalyzer` weights its ensemble by `r_squared` and
/// `get_best_model` ranks by it, so the "best" model was decided by a constant
/// before any data was seen.
///
/// This computes the real thing. `fitted[i]` is what the model predicted for
/// `actual[i]`; `None` when the two do not line up or nothing was fitted.
fn in_sample_metrics(actual: &[f64], fitted: &[f64]) -> Option<ModelPerformanceMetrics> {
    if actual.len() != fitted.len() || actual.is_empty() {
        return None;
    }
    let count = actual.len() as f64;
    let mean = actual.iter().sum::<f64>() / count;

    let mut absolute_error_total = 0.0;
    let mut squared_error_total = 0.0;
    let mut total_sum_of_squares = 0.0;
    // MAPE is undefined where the actual value is zero, so those points are
    // excluded from it (and only from it) rather than silently inflating it.
    let mut percentage_error_total = 0.0;
    let mut percentage_points = 0.0;

    for (observed, predicted) in actual.iter().zip(fitted) {
        let error = observed - predicted;
        absolute_error_total += error.abs();
        squared_error_total += error * error;
        total_sum_of_squares += (observed - mean).powi(2);
        if observed.abs() > f64::EPSILON {
            percentage_error_total += (error / observed).abs();
            percentage_points += 1.0;
        }
    }

    let mean_squared_error = squared_error_total / count;
    Some(ModelPerformanceMetrics {
        mae: absolute_error_total / count,
        mse: mean_squared_error,
        rmse: mean_squared_error.sqrt(),
        // With no variance in the observations R^2 is undefined; 0.0 is the
        // honest floor -- the model explains none of a variance that is zero.
        r_squared: if total_sum_of_squares > 0.0 {
            1.0 - squared_error_total / total_sum_of_squares
        } else {
            0.0
        },
        mape: if percentage_points > 0.0 {
            percentage_error_total / percentage_points
        } else {
            0.0
        },
    })
}

/// Confidence a fitted model can honestly claim: its measured in-sample
/// accuracy, `1 - MAPE`, clamped to a probability.
///
/// ## Changed in 0.2.1
///
/// `predict` published a constant per model family -- 0.6 for the moving
/// average ("Lower confidence for simple average"), 0.7 for exponential
/// smoothing, 0.65 for ARIMA -- and the linear model published its
/// `r_squared`. Those figures went straight into `PerformancePrediction`,
/// which the ensemble weights by, so the weighting was fixed before any data
/// arrived.
///
/// `1 - MAPE` is used rather than `R²` because it is scale-free *and* defined
/// for every model here: a moving average's fitted values are the window mean,
/// whose `R²` is identically zero by construction, which would silently give
/// every moving average zero weight in the ensemble whatever its accuracy.
/// A model with no measured fit gets no confidence at all.
fn confidence_from_fit(fit: Option<&ModelPerformanceMetrics>) -> f32 {
    fit.map(|metrics| (1.0 - metrics.mape).clamp(0.0, 1.0) as f32).unwrap_or(0.0)
}

/// Half-width of a 95% prediction interval: `1.96 * RMSE` of the in-sample
/// residuals.
///
/// This is the interval implied by the residual spread the model actually
/// produced, under the usual assumption that the residuals are approximately
/// normal. `None` when nothing has been fitted.
///
/// ## Changed in 0.2.1
///
/// The intervals were fixed fractions of the point forecast --
/// `(x * 0.8, x * 1.2)` for the moving average, `(x * 0.85, x * 1.15)` for
/// exponential smoothing, `(x * 0.9, x * 1.1)` for linear regression -- so a
/// model that fit its training window perfectly and one that missed by half
/// published intervals of the same relative width.
fn interval_half_width(fit: Option<&ModelPerformanceMetrics>) -> Option<f64> {
    fit.map(|metrics| 1.96 * metrics.rmse)
}

/// Prediction interval around `point`, floored at zero for quantities that
/// cannot be negative. Falls back to a degenerate interval at the point itself
/// when no fit has been measured -- an interval of zero width says "no spread
/// was measured", where a made-up percentage would say "this much spread was".
fn prediction_interval(point: f64, half_width: Option<f64>) -> (f64, f64) {
    let half = half_width.unwrap_or(0.0);
    ((point - half).max(0.0), point + half)
}

/// Holt's linear trend (double exponential smoothing) state after a fit.
struct HoltState {
    /// Level after the last observation.
    smoothed: f64,
    /// Trend per step after the last observation.
    trend: f64,
}

impl HoltState {
    /// Fit `values` with smoothing parameters `alpha` (level) and `beta`
    /// (trend). Returns `None` for fewer than two observations.
    fn fit(values: &[f64], alpha: f64, beta: f64) -> Option<Self> {
        if values.len() < 2 {
            return None;
        }
        let mut state = HoltState {
            smoothed: *values.first()?,
            trend: 0.0,
        };
        for value in values.iter().skip(1) {
            let previous = state.smoothed;
            state.smoothed = alpha * value + (1.0 - alpha) * state.smoothed;
            state.trend = beta * (state.smoothed - previous) + (1.0 - beta) * state.trend;
        }
        Some(state)
    }
}

/// AR(1) fitted to the first differences of a series, with the last observed
/// level retained so the differencing can be reversed.
struct DifferencedAutoregression {
    /// AR(1) coefficient on the differenced series.
    coefficient: f64,
    /// Last observed difference.
    last_difference: f64,
    /// Last observed level.
    last_level: f64,
}

impl DifferencedAutoregression {
    /// Fit the model to `values`. Returns `None` for fewer than five
    /// observations, which is what an AR(1) on differences needs to be
    /// estimated at all.
    fn fit(values: &[f64]) -> Option<Self> {
        if values.len() < 5 {
            return None;
        }
        let differences: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = differences.iter().sum::<f64>() / differences.len() as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for index in 1..differences.len() {
            let lag = differences[index - 1] - mean;
            let current = differences[index] - mean;
            numerator += lag * current;
            denominator += lag * lag;
        }
        let coefficient =
            if denominator > 0.0 { (numerator / denominator).clamp(-0.9, 0.9) } else { 0.5 };

        Some(Self {
            coefficient,
            last_difference: differences.last().copied().unwrap_or(0.0),
            last_level: values.last().copied().unwrap_or(0.0),
        })
    }

    /// Forecast `steps` ahead by accumulating the forecast differences onto the
    /// last observed level.
    fn forecast(&self, steps: u64) -> f64 {
        let mut level = self.last_level;
        let mut difference = self.last_difference;
        for _ in 0..steps {
            difference *= self.coefficient;
            level += difference;
        }
        level
    }
}

/// Latency of a data point in milliseconds, the unit every latency forecast in
/// this module is expressed in.
fn latency_millis(point: &PerformanceDataPoint) -> f64 {
    point.latency.as_secs_f64() * 1000.0
}

/// A forecast latency, as a `Duration`. Negative forecasts are floored at zero:
/// no elapsed time is negative.
fn latency_duration(millis: f64) -> Duration {
    Duration::from_secs_f64((millis.max(0.0) / 1000.0).min(u32::MAX as f64))
}

/// Linear regression prediction model
pub struct LinearRegressionModel {
    slope: f64,
    intercept: f64,
    r_squared: f64,
    /// Regression of the latency series on the same time index.
    ///
    /// 0.2.1: `predict` published `Duration::from_millis(100)` for every
    /// forecast point of every model in this file, marked "Simplified". A
    /// latency forecast read off a throughput model is not a measurement of
    /// anything; `PerformanceDataPoint` carries the latency, so it is fitted.
    latency_slope: f64,
    latency_intercept: f64,
    /// Fit measured on the training window, or `None` before training.
    fit: Option<ModelPerformanceMetrics>,
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
            latency_slope: 0.0,
            latency_intercept: 0.0,
            fit: None,
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

        // The same regression, on the latency series the data points carry.
        let latency_values: Vec<f64> = data.iter().map(latency_millis).collect();
        let (latency_slope, latency_intercept, _) =
            calculate_linear_regression(&x_values, &latency_values)?;
        self.latency_slope = latency_slope;
        self.latency_intercept = latency_intercept;

        let fitted: Vec<f64> = x_values.iter().map(|x| slope * x + intercept).collect();
        self.fit = in_sample_metrics(&y_values, &fitted);
        self.trained = true;

        Ok(())
    }

    fn predict(&self, horizon: Duration) -> Result<PerformancePrediction> {
        if !self.trained {
            return Err(anyhow::anyhow!("Model must be trained before prediction"));
        }

        let horizon_minutes = horizon.as_secs() / 60;
        let mut predicted_values = Vec::new();
        let confidence = confidence_from_fit(self.fit.as_ref());
        let half_width = interval_half_width(self.fit.as_ref());

        for i in 1..=horizon_minutes {
            let x = i as f64;
            let predicted_throughput = (self.slope * x + self.intercept).max(0.0);

            predicted_values.push(PredictedPerformancePoint {
                timestamp: Utc::now() + chrono::Duration::minutes(i as i64),
                predicted_throughput,
                predicted_latency: latency_duration(
                    self.latency_slope * x + self.latency_intercept,
                ),
                confidence,
                confidence_interval: prediction_interval(predicted_throughput, half_width),
            });
        }

        Ok(PerformancePrediction {
            id: format!("linear_regression_{}", Utc::now().timestamp()),
            predicted_values,
            model: PredictionModelType::LinearRegression,
            confidence,
            horizon,
            uncertainty: 1.0 - confidence,
            predicted_at: Utc::now(),
        })
    }

    fn model_type(&self) -> PredictionModelType {
        PredictionModelType::LinearRegression
    }

    fn performance_metrics(&self) -> Option<ModelPerformanceMetrics> {
        self.fit.clone()
    }
}

/// Moving average prediction model
pub struct MovingAverageModel {
    window_size: usize,
    recent_values: Vec<f64>,
    average: f64,
    /// Mean latency over the same window, in milliseconds.
    latency_average: f64,
    /// Fit measured on the training window, or `None` before training.
    fit: Option<ModelPerformanceMetrics>,
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
            latency_average: 0.0,
            fit: None,
            trained: false,
        }
    }

    pub fn with_window_size(window_size: usize) -> Self {
        Self {
            window_size,
            recent_values: Vec::new(),
            average: 0.0,
            latency_average: 0.0,
            fit: None,
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

        // The same window mean, over the latency the data points carry.
        let latencies: Vec<f64> =
            data.iter().rev().take(self.window_size).map(latency_millis).collect();
        self.latency_average = if latencies.is_empty() {
            0.0
        } else {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        };
        // The model forecasts the window mean for every future point, so its
        // one-step-ahead fit over the window is that same constant.
        let fitted = vec![self.average; self.recent_values.len()];
        self.fit = in_sample_metrics(&self.recent_values, &fitted);
        self.trained = true;

        Ok(())
    }

    fn predict(&self, horizon: Duration) -> Result<PerformancePrediction> {
        if !self.trained {
            return Err(anyhow::anyhow!("Model must be trained before prediction"));
        }

        let horizon_minutes = horizon.as_secs() / 60;
        let mut predicted_values = Vec::new();
        let confidence = confidence_from_fit(self.fit.as_ref());
        let half_width = interval_half_width(self.fit.as_ref());

        for i in 1..=horizon_minutes {
            predicted_values.push(PredictedPerformancePoint {
                timestamp: Utc::now() + chrono::Duration::minutes(i as i64),
                predicted_throughput: self.average,
                predicted_latency: latency_duration(self.latency_average),
                confidence,
                confidence_interval: prediction_interval(self.average, half_width),
            });
        }

        Ok(PerformancePrediction {
            id: format!("moving_average_{}", Utc::now().timestamp()),
            predicted_values,
            model: PredictionModelType::MovingAverage,
            confidence,
            horizon,
            uncertainty: 1.0 - confidence,
            predicted_at: Utc::now(),
        })
    }

    fn model_type(&self) -> PredictionModelType {
        PredictionModelType::MovingAverage
    }

    fn performance_metrics(&self) -> Option<ModelPerformanceMetrics> {
        self.fit.clone()
    }
}

/// Exponential smoothing prediction model
pub struct ExponentialSmoothingModel {
    alpha: f64,
    smoothed_value: f64,
    trend: f64,
    /// Holt state of the latency series, in milliseconds.
    latency_smoothed: f64,
    latency_trend: f64,
    /// Fit measured on the training window, or `None` before training.
    fit: Option<ModelPerformanceMetrics>,
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
            latency_smoothed: 0.0,
            latency_trend: 0.0,
            fit: None,
            trained: false,
        }
    }

    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha,
            smoothed_value: 0.0,
            trend: 0.0,
            latency_smoothed: 0.0,
            latency_trend: 0.0,
            fit: None,
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

        // Apply exponential smoothing, recording the one-step-ahead forecast
        // made before each observation was seen so the fit is measured against
        // genuine predictions rather than against the smoothed line itself.
        let mut fitted = Vec::with_capacity(values.len() - 1);
        for value in values.iter().skip(1) {
            fitted.push(self.smoothed_value + self.trend);

            let previous_smoothed = self.smoothed_value;
            self.smoothed_value = self.alpha * value + (1.0 - self.alpha) * self.smoothed_value;

            // Calculate trend (Holt's method)
            let beta = 0.2; // Trend smoothing parameter
            self.trend =
                beta * (self.smoothed_value - previous_smoothed) + (1.0 - beta) * self.trend;
        }

        self.fit = in_sample_metrics(&values[1..], &fitted);

        // The same Holt recursion, over the latency the data points carry.
        let latencies: Vec<f64> = data.iter().map(latency_millis).collect();
        if let Some(latency_state) = HoltState::fit(&latencies, self.alpha, 0.2) {
            self.latency_smoothed = latency_state.smoothed;
            self.latency_trend = latency_state.trend;
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
        let confidence = confidence_from_fit(self.fit.as_ref());
        let half_width = interval_half_width(self.fit.as_ref());

        for i in 1..=horizon_minutes {
            let forecast = (self.smoothed_value + self.trend * i as f64).max(0.0);

            predicted_values.push(PredictedPerformancePoint {
                timestamp: Utc::now() + chrono::Duration::minutes(i as i64),
                predicted_throughput: forecast,
                predicted_latency: latency_duration(
                    self.latency_smoothed + self.latency_trend * i as f64,
                ),
                confidence,
                confidence_interval: prediction_interval(forecast, half_width),
            });
        }

        Ok(PerformancePrediction {
            id: format!("exponential_smoothing_{}", Utc::now().timestamp()),
            predicted_values,
            model: PredictionModelType::ExponentialSmoothing,
            confidence,
            horizon,
            uncertainty: 1.0 - confidence,
            predicted_at: Utc::now(),
        })
    }

    fn model_type(&self) -> PredictionModelType {
        PredictionModelType::ExponentialSmoothing
    }

    fn performance_metrics(&self) -> Option<ModelPerformanceMetrics> {
        self.fit.clone()
    }
}

/// ARIMA(1,1,0): a first-order autoregression on the once-differenced series.
///
/// ## Changed in 0.2.1
///
/// `predict` reconstructed the level as `100.0 + current_diff` under the
/// comment "Assume base level of 100" -- the last observed value was never
/// carried, so the forecast was anchored to a constant that had nothing to do
/// with the data. The last level is now retained and the differences are
/// accumulated onto it, which is what reversing a first difference means.
pub struct ARIMAModel {
    ar_coefficients: Vec<f64>,
    differenced_data: Vec<f64>,
    /// Last observed level, the anchor for reversing the differencing.
    last_level: f64,
    /// The same AR(1)-on-differences fitted to the latency series, or `None`
    /// when the window was too short to estimate one.
    latency_model: Option<DifferencedAutoregression>,
    /// Fit measured on the training window, or `None` before training.
    fit: Option<ModelPerformanceMetrics>,
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
            differenced_data: Vec::new(),
            last_level: 0.0,
            latency_model: None,
            fit: None,
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

        self.last_level = values.last().copied().unwrap_or(0.0);

        // One-step-ahead fit: each differenced value predicted from its
        // predecessor, then added back onto the preceding level.
        let coefficient = self.ar_coefficients[0];
        let mut fitted = Vec::new();
        for index in 1..self.differenced_data.len() {
            fitted.push(values[index] + coefficient * self.differenced_data[index - 1]);
        }
        self.fit = in_sample_metrics(&values[2..], &fitted);

        // The same model, fitted to the latency series the data points carry.
        let latencies: Vec<f64> = data.iter().map(latency_millis).collect();
        self.latency_model = DifferencedAutoregression::fit(&latencies);

        self.trained = true;

        Ok(())
    }

    fn predict(&self, horizon: Duration) -> Result<PerformancePrediction> {
        if !self.trained {
            return Err(anyhow::anyhow!("Model must be trained before prediction"));
        }

        let horizon_minutes = horizon.as_secs() / 60;
        let mut predicted_values = Vec::new();

        let mut current_diff = self.differenced_data.last().copied().unwrap_or(0.0);
        // Reversing a first difference means accumulating the forecast
        // differences onto the last observed level.
        let mut level = self.last_level;
        // Confidence is the model's measured in-sample accuracy; before 0.2.1
        // it was the constant 0.65, and between then and now it was the
        // model's `r_squared` (see `confidence_from_fit` for why every model
        // in this file now reports the same scale-free measure).
        let confidence = confidence_from_fit(self.fit.as_ref());
        let half_width = interval_half_width(self.fit.as_ref());

        for i in 1..=horizon_minutes {
            current_diff *= self.ar_coefficients[0];
            level += current_diff;
            let predicted_throughput = level.max(0.0);

            predicted_values.push(PredictedPerformancePoint {
                timestamp: Utc::now() + chrono::Duration::minutes(i as i64),
                predicted_throughput,
                predicted_latency: self
                    .latency_model
                    .as_ref()
                    .map(|model| latency_duration(model.forecast(i)))
                    .unwrap_or_default(),
                confidence,
                confidence_interval: prediction_interval(predicted_throughput, half_width),
            });
        }

        Ok(PerformancePrediction {
            id: format!("arima_{}", Utc::now().timestamp()),
            predicted_values,
            model: PredictionModelType::ARIMA,
            confidence,
            horizon,
            uncertainty: 1.0 - confidence,
            predicted_at: Utc::now(),
        })
    }

    fn model_type(&self) -> PredictionModelType {
        PredictionModelType::ARIMA
    }

    fn performance_metrics(&self) -> Option<ModelPerformanceMetrics> {
        self.fit.clone()
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

    /// The weighted ensemble point, or `None` when no model contributed any
    /// weight to this timestamp.
    ///
    /// ## Changed in 0.2.1
    ///
    /// The zero-weight branch returned a whole invented point --
    /// `Duration::from_millis(100)`, `confidence: 0.5` and an interval of
    /// `(throughput * 0.9, throughput * 1.1)` -- so an ensemble that no model
    /// had contributed to still published a forecast with a stated confidence.
    /// The caller now drops such timestamps.
    fn finalize(self, timestamp: DateTime<Utc>) -> Option<PredictedPerformancePoint> {
        if self.total_weight.is_nan() || self.total_weight <= 0.0 {
            return None;
        }
        let total_weight = self.total_weight as f64;

        let final_throughput = self.weighted_throughput / total_weight;
        let final_latency =
            Duration::from_secs_f64((self.weighted_latency / total_weight).max(0.0) / 1000.0);
        let final_confidence = self.weighted_confidence / self.total_weight;

        // The ensemble interval is the union of the contributing intervals:
        // no narrower claim is supported by models that disagree.
        let (min_lower, max_upper) = if self.confidence_intervals.is_empty() {
            (final_throughput, final_throughput)
        } else {
            let min_lower =
                self.confidence_intervals.iter().map(|ci| ci.0).fold(f64::INFINITY, f64::min);
            let max_upper = self
                .confidence_intervals
                .iter()
                .map(|ci| ci.1)
                .fold(f64::NEG_INFINITY, f64::max);
            (min_lower, max_upper)
        };

        Some(PredictedPerformancePoint {
            timestamp,
            predicted_throughput: final_throughput,
            predicted_latency: final_latency,
            confidence: final_confidence,
            confidence_interval: (min_lower, max_upper),
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance_optimizer::types::{
        PerformanceDataPoint, SystemState, TestCharacteristics,
    };
    use std::time::Duration;

    fn make_data_point(throughput: f64, latency_ms: u64) -> PerformanceDataPoint {
        PerformanceDataPoint {
            parallelism: 1,
            throughput,
            latency: Duration::from_millis(latency_ms),
            cpu_utilization: 0.5,
            memory_utilization: 0.5,
            resource_efficiency: 0.8,
            timestamp: chrono::Utc::now(),
            test_characteristics: TestCharacteristics::default(),
            system_state: SystemState::default(),
        }
    }

    fn make_training_data(count: usize) -> Vec<PerformanceDataPoint> {
        (0..count).map(|i| make_data_point(100.0 + i as f64 * 5.0, 50)).collect()
    }

    // =========================================================================
    // ENGINE TESTS
    // =========================================================================

    #[test]
    fn test_predictive_engine_new() {
        let engine = PredictiveAnalyticsEngine::new();
        let stats = engine.get_prediction_statistics();
        assert_eq!(stats.total_predictions, 0);
        // Engine initializes with default models
        let _models = stats.active_models;
    }

    #[test]
    fn test_predictive_engine_with_config() {
        let config = PredictiveAnalyticsConfig {
            enable_prediction: true,
            prediction_models: vec![PredictionModelType::LinearRegression],
            prediction_horizon: Duration::from_secs(600),
            model_update_frequency: Duration::from_secs(120),
            min_data_points: 5,
        };
        let engine = PredictiveAnalyticsEngine::with_config(config);
        let stats = engine.get_prediction_statistics();
        let _models = stats.active_models;
    }

    #[test]
    fn test_predictive_engine_default() {
        let engine = PredictiveAnalyticsEngine::default();
        let stats = engine.get_prediction_statistics();
        assert_eq!(stats.total_predictions, 0);
    }

    #[test]
    fn test_engine_update_config() {
        let engine = PredictiveAnalyticsEngine::new();
        let config = PredictiveAnalyticsConfig {
            enable_prediction: false,
            prediction_models: vec![],
            prediction_horizon: Duration::from_secs(300),
            model_update_frequency: Duration::from_secs(60),
            min_data_points: 3,
        };
        engine.update_config(config);
    }

    #[test]
    fn test_engine_add_model() {
        let engine = PredictiveAnalyticsEngine::new();
        engine.add_model(Box::new(LinearRegressionModel::new()));
    }

    #[test]
    fn test_engine_get_best_model_no_training() {
        let engine = PredictiveAnalyticsEngine::new();
        let best = engine.get_best_model();
        // May or may not have a best model before training
        let _ = best;
    }

    #[test]
    fn test_engine_get_performance_summary() {
        let engine = PredictiveAnalyticsEngine::new();
        let summary = engine.get_performance_summary();
        let _ = summary; // Just verify it doesn't panic
    }

    // =========================================================================
    // MODEL TESTS
    // =========================================================================

    #[test]
    fn test_linear_regression_model_new() {
        let model = LinearRegressionModel::new();
        assert_eq!(model.model_type(), PredictionModelType::LinearRegression);
    }

    #[test]
    fn test_linear_regression_model_train() {
        let mut model = LinearRegressionModel::new();
        let data = make_training_data(20);
        let result = model.train(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_linear_regression_model_predict_after_train() {
        let mut model = LinearRegressionModel::new();
        let data = make_training_data(20);
        if model.train(&data).is_ok() {
            let result = model.predict(Duration::from_secs(1800));
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_linear_regression_model_predict_without_train() {
        let model = LinearRegressionModel::new();
        let result = model.predict(Duration::from_secs(1800));
        assert!(result.is_err());
    }

    /// Regression: `performance_metrics` used to answer on an untrained model,
    /// returning `mae/mse/rmse/mape = 0.0` -- a perfect fit -- with a
    /// hardcoded `r_squared`. It now reports that nothing has been measured.
    #[test]
    fn an_untrained_model_reports_no_measured_fit() {
        assert!(LinearRegressionModel::new().performance_metrics().is_none());
        assert!(MovingAverageModel::new().performance_metrics().is_none());
        assert!(ExponentialSmoothingModel::new().performance_metrics().is_none());
        assert!(ARIMAModel::new().performance_metrics().is_none());
    }

    /// Regression: three of the four models returned a constant `r_squared`
    /// ("typical for moving average" = 0.4, etc.), which
    /// `PerformanceHistoryAnalyzer` uses to weight its ensemble and to pick a
    /// best model -- so the ranking was decided before any data arrived.
    #[test]
    fn model_fit_is_measured_on_the_training_data() {
        // A dead-straight line: linear regression fits it exactly, a flat mean
        // does not.
        let rising: Vec<PerformanceDataPoint> = (0..20)
            .map(|i| PerformanceDataPoint {
                throughput: 100.0 + 5.0 * i as f64,
                ..PerformanceDataPoint::default()
            })
            .collect();

        let mut linear = LinearRegressionModel::new();
        linear.train(&rising).expect("training succeeds");
        let linear_fit = linear.performance_metrics().expect("fit is measured");
        assert!(
            linear_fit.r_squared > 0.999,
            "a straight line is fit exactly: {linear_fit:?}"
        );
        assert!(linear_fit.mae < 1e-6, "{linear_fit:?}");
        assert!(linear_fit.rmse < 1e-6, "{linear_fit:?}");

        let mut average = MovingAverageModel::with_window_size(20);
        average.train(&rising).expect("training succeeds");
        let average_fit = average.performance_metrics().expect("fit is measured");
        assert!(
            average_fit.r_squared < linear_fit.r_squared,
            "a flat mean cannot fit a trend as well as a line does: {average_fit:?}"
        );
        assert!(
            average_fit.mae > 0.0,
            "a flat mean over a rising series has real error: {average_fit:?}"
        );
        // The old code answered 0.4 here regardless of the data.
        assert!(
            (average_fit.r_squared - 0.4).abs() > 1e-9,
            "no longer the hardcoded 'typical' value: {average_fit:?}"
        );
    }

    /// Regression: ARIMA reconstructed its forecast level as `100.0 + diff`
    /// ("Assume base level of 100"), so a series sitting near 5000 forecast
    /// values near 100.
    #[test]
    fn arima_anchors_its_forecast_to_the_last_observed_level() {
        let data: Vec<PerformanceDataPoint> = (0..20)
            .map(|i| PerformanceDataPoint {
                throughput: 5000.0 + i as f64,
                ..PerformanceDataPoint::default()
            })
            .collect();
        let mut model = ARIMAModel::new();
        model.train(&data).expect("training succeeds");
        let prediction = model.predict(Duration::from_secs(300)).expect("prediction runs");
        let first = prediction
            .predicted_values
            .first()
            .expect("a five-minute horizon yields points");
        assert!(
            first.predicted_throughput > 4000.0,
            "the forecast must sit near the observed level, not near 100: {}",
            first.predicted_throughput
        );
    }

    #[test]
    fn test_moving_average_model_new() {
        let model = MovingAverageModel::new();
        assert_eq!(model.model_type(), PredictionModelType::MovingAverage);
    }

    #[test]
    fn test_moving_average_model_with_window() {
        let model = MovingAverageModel::with_window_size(10);
        assert_eq!(model.model_type(), PredictionModelType::MovingAverage);
    }

    #[test]
    fn test_moving_average_model_train() {
        let mut model = MovingAverageModel::new();
        let data = make_training_data(20);
        let result = model.train(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_moving_average_model_predict_after_train() {
        let mut model = MovingAverageModel::new();
        let data = make_training_data(20);
        if model.train(&data).is_ok() {
            let result = model.predict(Duration::from_secs(1800));
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_exponential_smoothing_model_new() {
        let model = ExponentialSmoothingModel::new();
        assert_eq!(
            model.model_type(),
            PredictionModelType::ExponentialSmoothing
        );
    }

    #[test]
    fn test_exponential_smoothing_model_with_alpha() {
        let model = ExponentialSmoothingModel::with_alpha(0.5);
        assert_eq!(
            model.model_type(),
            PredictionModelType::ExponentialSmoothing
        );
    }

    #[test]
    fn test_exponential_smoothing_model_train() {
        let mut model = ExponentialSmoothingModel::new();
        let data = make_training_data(20);
        let result = model.train(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_arima_model_new() {
        let model = ARIMAModel::new();
        assert_eq!(model.model_type(), PredictionModelType::ARIMA);
    }

    #[test]
    fn test_arima_model_train() {
        let mut model = ARIMAModel::new();
        let data = make_training_data(20);
        let result = model.train(&data);
        assert!(result.is_ok());
    }

    // =========================================================================
    // PREDICTION MODEL TYPE PARSING TESTS
    // =========================================================================

    #[test]
    fn test_parse_prediction_model_type_linear_regression() {
        let result: Result<PredictionModelType, _> = "LinearRegression".parse();
        assert!(result.is_ok());
        if let Ok(model_type) = result {
            assert_eq!(model_type, PredictionModelType::LinearRegression);
        }
    }

    #[test]
    fn test_parse_prediction_model_type_moving_average() {
        let result: Result<PredictionModelType, _> = "MovingAverage".parse();
        assert!(result.is_ok());
        if let Ok(model_type) = result {
            assert_eq!(model_type, PredictionModelType::MovingAverage);
        }
    }

    #[test]
    fn test_parse_prediction_model_type_exponential_smoothing() {
        let result: Result<PredictionModelType, _> = "ExponentialSmoothing".parse();
        assert!(result.is_ok());
        if let Ok(model_type) = result {
            assert_eq!(model_type, PredictionModelType::ExponentialSmoothing);
        }
    }

    #[test]
    fn test_parse_prediction_model_type_arima() {
        let result: Result<PredictionModelType, _> = "ARIMA".parse();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_prediction_model_type_neural_network() {
        let result: Result<PredictionModelType, _> = "NeuralNetwork".parse();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_prediction_model_type_custom() {
        let result: Result<PredictionModelType, _> = "MyCustomModel".parse();
        assert!(result.is_ok());
        if let Ok(PredictionModelType::Custom(name)) = result {
            assert_eq!(name, "MyCustomModel");
        }
    }

    // =========================================================================
    // PREDICTION STATISTICS TESTS
    // =========================================================================

    #[test]
    fn test_prediction_statistics_construction() {
        let stats = PredictionStatistics {
            total_predictions: 50,
            active_models: 4,
            average_confidence: 0.85,
            average_uncertainty: 0.15,
            cache_memory_usage: 4096,
        };
        assert_eq!(stats.total_predictions, 50);
        assert_eq!(stats.active_models, 4);
        assert!(stats.average_confidence + stats.average_uncertainty <= 1.01);
    }

    // =========================================================================
    // ASYNC TESTS
    // =========================================================================

    #[tokio::test]
    async fn test_train_models_sufficient_data() {
        let mut engine = PredictiveAnalyticsEngine::new();
        let data = make_training_data(20);
        let result = engine.train_models(&data).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_train_models_insufficient_data() {
        let mut engine = PredictiveAnalyticsEngine::new();
        let data = make_training_data(3); // Default min_data_points is 10
        let result = engine.train_models(&data).await;
        assert!(result.is_err());
    }

    /// Regression: every model published `predicted_latency:
    /// Duration::from_millis(100)` for every forecast point, marked
    /// "Simplified". `PerformanceDataPoint` carries the latency, so it is now
    /// fitted by the same algorithm as the throughput.
    #[test]
    fn latency_is_forecast_from_the_latency_series() {
        let slow: Vec<PerformanceDataPoint> =
            (0..20).map(|i| make_data_point(100.0 + i as f64, 900)).collect();
        let fast: Vec<PerformanceDataPoint> =
            (0..20).map(|i| make_data_point(100.0 + i as f64, 7)).collect();

        for (name, slow_prediction, fast_prediction) in [
            (
                "linear regression",
                trained_prediction(LinearRegressionModel::new(), &slow),
                trained_prediction(LinearRegressionModel::new(), &fast),
            ),
            (
                "moving average",
                trained_prediction(MovingAverageModel::with_window_size(20), &slow),
                trained_prediction(MovingAverageModel::with_window_size(20), &fast),
            ),
            (
                "exponential smoothing",
                trained_prediction(ExponentialSmoothingModel::new(), &slow),
                trained_prediction(ExponentialSmoothingModel::new(), &fast),
            ),
            (
                "arima",
                trained_prediction(ARIMAModel::new(), &slow),
                trained_prediction(ARIMAModel::new(), &fast),
            ),
        ] {
            let slow_latency = slow_prediction.predicted_values[0].predicted_latency;
            let fast_latency = fast_prediction.predicted_values[0].predicted_latency;
            assert!(
                slow_latency > fast_latency,
                "{name}: a 900ms series must forecast slower than a 7ms one, got {slow_latency:?} \
                 vs {fast_latency:?}"
            );
            assert_ne!(
                slow_latency,
                Duration::from_millis(100),
                "{name}: no longer the hardcoded 100ms"
            );
            assert_ne!(
                fast_latency,
                Duration::from_millis(100),
                "{name}: no longer the hardcoded 100ms"
            );
        }
    }

    /// Regression: confidence was a per-family constant (0.6 moving average,
    /// 0.7 exponential smoothing, 0.65 ARIMA) and the interval was a fixed
    /// fraction of the point forecast, so a model that fit its window exactly
    /// and one that missed wildly published the same claim.
    #[test]
    fn confidence_and_interval_come_from_the_measured_fit() {
        // A dead-flat series: the moving average fits it exactly.
        let steady: Vec<PerformanceDataPoint> =
            (0..20).map(|_| make_data_point(500.0, 50)).collect();
        // A wildly swinging one: the same model cannot.
        let erratic: Vec<PerformanceDataPoint> = (0..20)
            .map(|i| make_data_point(if i % 2 == 0 { 50.0 } else { 950.0 }, 50))
            .collect();

        let steady_prediction =
            trained_prediction(MovingAverageModel::with_window_size(20), &steady);
        let erratic_prediction =
            trained_prediction(MovingAverageModel::with_window_size(20), &erratic);

        assert!(
            steady_prediction.confidence > erratic_prediction.confidence,
            "a model that fits must claim more than one that does not: {} vs {}",
            steady_prediction.confidence,
            erratic_prediction.confidence
        );
        assert!(
            (steady_prediction.confidence - 0.6).abs() > 1e-6
                && (erratic_prediction.confidence - 0.6).abs() > 1e-6,
            "no longer the hardcoded 0.6"
        );
        assert!(
            (steady_prediction.confidence + steady_prediction.uncertainty - 1.0).abs() < 1e-6,
            "uncertainty is the complement of confidence"
        );

        let steady_point = &steady_prediction.predicted_values[0];
        let erratic_point = &erratic_prediction.predicted_values[0];
        let steady_width = steady_point.confidence_interval.1 - steady_point.confidence_interval.0;
        let erratic_width =
            erratic_point.confidence_interval.1 - erratic_point.confidence_interval.0;
        assert!(
            erratic_width > steady_width,
            "the interval must widen with the residual spread: {erratic_width} vs {steady_width}"
        );

        // The old rule was a fixed +/-20% of the point forecast.
        let fixed_fraction_width = erratic_point.predicted_throughput * 0.4;
        assert!(
            (erratic_width - fixed_fraction_width).abs() > 1e-6,
            "no longer a fixed fraction of the forecast: {erratic_width} vs {fixed_fraction_width}"
        );
    }

    /// Regression: `WeightedPredictionAggregator::finalize` invented a whole
    /// point (100ms, confidence 0.5, +/-10%) when no model had contributed any
    /// weight.
    #[test]
    fn an_unweighted_ensemble_point_is_dropped() {
        let empty = WeightedPredictionAggregator::new();
        assert!(
            empty.finalize(chrono::Utc::now()).is_none(),
            "nothing contributed, so there is nothing to publish"
        );

        let mut aggregator = WeightedPredictionAggregator::new();
        aggregator.add_prediction(
            &PredictedPerformancePoint {
                timestamp: chrono::Utc::now(),
                predicted_throughput: 200.0,
                predicted_latency: Duration::from_millis(40),
                confidence: 0.8,
                confidence_interval: (180.0, 220.0),
            },
            1.0,
        );
        let point = aggregator
            .finalize(chrono::Utc::now())
            .expect("a weighted contribution yields a point");
        assert!((point.predicted_throughput - 200.0).abs() < 1e-9);
        assert_eq!(point.predicted_latency, Duration::from_millis(40));
    }

    fn trained_prediction<M: PredictiveModel>(
        mut model: M,
        data: &[PerformanceDataPoint],
    ) -> PerformancePrediction {
        model.train(data).unwrap_or_else(|e| panic!("training failed: {e}"));
        model
            .predict(Duration::from_secs(600))
            .unwrap_or_else(|e| panic!("prediction failed: {e}"))
    }

    #[tokio::test]
    async fn test_clear_cache() {
        let engine = PredictiveAnalyticsEngine::new();
        engine.clear_cache().await;
        let stats = engine.get_prediction_statistics();
        assert_eq!(stats.total_predictions, 0);
    }
}
