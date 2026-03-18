//! Network condition prediction and pattern analysis for mobile federated learning.
//!
//! This module provides intelligent network condition prediction capabilities including
//! historical pattern analysis, machine learning-based forecasting, and adaptive
//! prediction models for proactive optimization in mobile federated learning.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use trustformers_core::Result;

use super::types::{
    NetworkAdaptationConfig, NetworkConditions, NetworkPredictionConfig, NetworkQuality,
    TrendDirection,
};
use crate::profiler::NetworkConnectionType;

/// Network condition predictor
pub struct NetworkPredictor {
    config: NetworkPredictionConfig,
    historical_data: VecDeque<NetworkConditions>,
    prediction_models: HashMap<String, PredictionModel>,
    pattern_analyzer: NetworkPatternAnalyzer,
}

/// Pattern analysis for network conditions
pub struct NetworkPatternAnalyzer {
    daily_patterns: HashMap<u8, DailyPattern>, // Hour -> Pattern
    weekly_patterns: HashMap<u8, WeeklyPattern>, // Day of week -> Pattern
    seasonal_patterns: HashMap<u8, SeasonalPattern>, // Month -> Pattern
}

/// Machine learning prediction model
pub struct PredictionModel {
    model_type: PredictionModelType,
    weights: Vec<f32>,
    training_data: VecDeque<TrainingDataPoint>,
    accuracy: f32,
    last_updated: Instant,
}

/// Training data point for ML models
#[derive(Debug, Clone)]
pub struct TrainingDataPoint {
    features: Vec<f32>,
    target: f32,
    timestamp: Instant,
    weight: f32,
}

/// Daily network pattern
#[derive(Debug, Clone)]
pub struct DailyPattern {
    hourly_bandwidth: [f32; 24],
    hourly_latency: [f32; 24],
    hourly_quality: [NetworkQuality; 24],
    confidence: f32,
    sample_count: u32,
}

/// Weekly network pattern
#[derive(Debug, Clone)]
pub struct WeeklyPattern {
    daily_trends: [TrendDirection; 7],
    peak_hours: Vec<u8>,
    low_usage_periods: Vec<(u8, u8)>, // (start_hour, end_hour)
    confidence: f32,
}

/// Seasonal network pattern
#[derive(Debug, Clone)]
pub struct SeasonalPattern {
    monthly_averages: MonthlyAverages,
    seasonal_trends: TrendDirection,
    holiday_adjustments: HashMap<String, f32>,
    confidence: f32,
}

/// Monthly network averages
#[derive(Debug, Clone)]
pub struct MonthlyAverages {
    bandwidth_mbps: f32,
    latency_ms: f32,
    packet_loss_percent: f32,
    quality_score: f32,
}

/// Prediction model types
#[derive(Debug, Clone)]
pub enum PredictionModelType {
    LinearRegression,
    ExponentialSmoothing,
    MovingAverage,
    NeuralNetwork,
    EnsembleMethod,
}

/// Prediction result with confidence intervals
#[derive(Debug, Clone)]
pub struct PredictionResult {
    predicted_conditions: Vec<(Instant, NetworkConditions)>,
    confidence_intervals: Vec<(f32, f32)>, // (lower_bound, upper_bound)
    prediction_accuracy: f32,
    model_used: PredictionModelType,
    timestamp: Instant,
}

impl PredictionResult {
    /// Get the predicted conditions with their timestamps
    pub fn get_predicted_conditions(&self) -> &Vec<(Instant, NetworkConditions)> {
        &self.predicted_conditions
    }
}

impl NetworkPredictor {
    /// Create new network predictor
    pub fn new(config: NetworkAdaptationConfig) -> Result<Self> {
        Ok(Self {
            config: config.prediction_config,
            historical_data: VecDeque::new(),
            prediction_models: HashMap::new(),
            pattern_analyzer: NetworkPatternAnalyzer::new(),
        })
    }

    /// Start prediction system
    pub fn start(&mut self) -> Result<()> {
        // Initialize prediction models
        self.initialize_prediction_models()?;

        // Start pattern analysis
        self.pattern_analyzer.start_analysis();

        Ok(())
    }

    /// Stop prediction system
    pub fn stop(&mut self) -> Result<()> {
        // Save models and patterns for persistence
        Ok(())
    }

    /// Add historical data point
    pub fn add_historical_data(&mut self, conditions: NetworkConditions) -> Result<()> {
        // Add to historical data
        self.historical_data.push_back(conditions.clone());

        // Limit historical data size
        if self.historical_data.len() > (self.config.historical_window_hours * 60) as usize {
            // Convert hours to data points (assuming 1 per minute)
            self.historical_data.pop_front();
        }

        // Update pattern analysis
        self.pattern_analyzer.analyze_conditions(&conditions)?;

        // Update prediction models
        self.update_prediction_models(&conditions)?;

        Ok(())
    }

    /// Predict network conditions for specified time window
    pub fn predict_conditions(&self, window_minutes: u32) -> Result<PredictionResult> {
        let now = Instant::now();
        let mut predictions = Vec::new();
        let mut confidence_intervals = Vec::new();

        // Generate predictions for each minute in the window
        for i in 0..window_minutes {
            let target_time = now + Duration::from_secs(i as u64 * 60);

            // Get prediction from best model
            let (predicted_conditions, confidence) = self.predict_single_point(target_time)?;

            predictions.push((target_time, predicted_conditions));
            confidence_intervals.push(confidence);
        }

        // Select best model based on recent accuracy
        let best_model = self.select_best_model()?;

        Ok(PredictionResult {
            predicted_conditions: predictions,
            confidence_intervals,
            prediction_accuracy: self.get_model_accuracy(&best_model),
            model_used: best_model,
            timestamp: now,
        })
    }

    /// Predict conditions for a single time point
    fn predict_single_point(
        &self,
        target_time: Instant,
    ) -> Result<(NetworkConditions, (f32, f32))> {
        // Get base prediction from historical average
        let base_conditions = self.get_baseline_prediction(target_time)?;

        // Apply pattern-based adjustments
        let pattern_adjusted =
            self.pattern_analyzer.apply_pattern_adjustments(base_conditions, target_time)?;

        // Apply ML model predictions
        let ml_adjusted = self.apply_ml_predictions(pattern_adjusted, target_time)?;

        // Calculate confidence interval
        let confidence = self.calculate_confidence_interval(&ml_adjusted)?;

        Ok((ml_adjusted, confidence))
    }

    /// Get baseline prediction from historical average
    fn get_baseline_prediction(&self, _target_time: Instant) -> Result<NetworkConditions> {
        if self.historical_data.is_empty() {
            return Ok(NetworkConditions::default());
        }

        // Calculate averages from recent historical data
        let recent_count = (self.historical_data.len() / 4).max(1); // Use last 25%
        let recent_data: Vec<_> = self.historical_data.iter().rev().take(recent_count).collect();

        let avg_bandwidth =
            recent_data.iter().map(|c| c.bandwidth_mbps).sum::<f32>() / recent_data.len() as f32;

        let avg_latency =
            recent_data.iter().map(|c| c.latency_ms).sum::<f32>() / recent_data.len() as f32;

        let avg_packet_loss = recent_data.iter().map(|c| c.packet_loss_percent).sum::<f32>()
            / recent_data.len() as f32;

        let avg_jitter =
            recent_data.iter().map(|c| c.jitter_ms).sum::<f32>() / recent_data.len() as f32;

        let avg_stability =
            recent_data.iter().map(|c| c.stability_score).sum::<f32>() / recent_data.len() as f32;

        // Use most common connection type
        let connection_type = recent_data
            .iter()
            .map(|c| &c.connection_type)
            .fold(HashMap::new(), |mut acc, ct| {
                *acc.entry(ct).or_insert(0) += 1;
                acc
            })
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(ct, _)| *ct)
            .unwrap_or(NetworkConnectionType::WiFi);

        Ok(NetworkConditions {
            bandwidth_mbps: avg_bandwidth,
            latency_ms: avg_latency,
            packet_loss_percent: avg_packet_loss,
            jitter_ms: avg_jitter,
            stability_score: avg_stability,
            connection_type,
            signal_strength_dbm: Some(-60),           // Default value
            quality_assessment: NetworkQuality::Fair, // Will be updated
            timestamp: Instant::now(),
            available_data_mb: Some(1000), // Default value
        })
    }

    /// Apply ML model predictions
    fn apply_ml_predictions(
        &self,
        mut conditions: NetworkConditions,
        target_time: Instant,
    ) -> Result<NetworkConditions> {
        // Get features for prediction
        let features = self.extract_features(&conditions, target_time)?;

        // Apply each model and ensemble the results
        let mut bandwidth_predictions = Vec::new();
        let mut latency_predictions = Vec::new();

        for (model_name, model) in &self.prediction_models {
            match model_name.as_str() {
                "bandwidth" => {
                    let prediction = model.predict(&features)?;
                    bandwidth_predictions.push(prediction);
                },
                "latency" => {
                    let prediction = model.predict(&features)?;
                    latency_predictions.push(prediction);
                },
                _ => {},
            }
        }

        // Ensemble predictions (simple average)
        if !bandwidth_predictions.is_empty() {
            let avg_bandwidth =
                bandwidth_predictions.iter().sum::<f32>() / bandwidth_predictions.len() as f32;
            conditions.bandwidth_mbps = avg_bandwidth.max(0.1); // Minimum 0.1 Mbps
        }

        if !latency_predictions.is_empty() {
            let avg_latency =
                latency_predictions.iter().sum::<f32>() / latency_predictions.len() as f32;
            conditions.latency_ms = avg_latency.max(1.0); // Minimum 1ms
        }

        Ok(conditions)
    }

    /// Extract features for ML prediction
    fn extract_features(
        &self,
        conditions: &NetworkConditions,
        target_time: Instant,
    ) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        // Time-based features
        let time_since_epoch = target_time.elapsed().as_secs() as f32;
        features.push(time_since_epoch % (24.0 * 3600.0) / 3600.0); // Hour of day
        features.push((time_since_epoch / (24.0 * 3600.0)) % 7.0); // Day of week

        // Current network features
        features.push(conditions.bandwidth_mbps);
        features.push(conditions.latency_ms);
        features.push(conditions.packet_loss_percent);
        features.push(conditions.jitter_ms);
        features.push(conditions.stability_score);

        // Connection type as numeric
        features.push(match conditions.connection_type {
            NetworkConnectionType::WiFi => 1.0,
            NetworkConnectionType::Cellular4G => 2.0,
            NetworkConnectionType::Cellular5G => 3.0,
            NetworkConnectionType::Ethernet => 4.0,
            NetworkConnectionType::Offline => 0.0,
            NetworkConnectionType::Unknown => 0.5,
        });

        // Historical trend features
        if let Some(trend_features) = self.extract_trend_features() {
            features.extend(trend_features);
        }

        Ok(features)
    }

    /// Extract trend features from historical data
    fn extract_trend_features(&self) -> Option<Vec<f32>> {
        if self.historical_data.len() < 5 {
            return None;
        }

        let recent: Vec<_> = self.historical_data.iter().rev().take(5).collect();

        // Calculate trends
        let bandwidth_trend =
            self.calculate_trend(&recent.iter().map(|c| c.bandwidth_mbps).collect::<Vec<_>>());
        let latency_trend =
            self.calculate_trend(&recent.iter().map(|c| c.latency_ms).collect::<Vec<_>>());

        Some(vec![bandwidth_trend, latency_trend])
    }

    /// Calculate trend from values
    fn calculate_trend(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mut slope_sum = 0.0;
        for i in 1..values.len() {
            slope_sum += values[i] - values[i - 1];
        }

        slope_sum / (values.len() - 1) as f32
    }

    /// Calculate confidence interval
    fn calculate_confidence_interval(&self, _conditions: &NetworkConditions) -> Result<(f32, f32)> {
        // Simple confidence interval based on historical variance
        let variance = self.calculate_historical_variance()?;
        let confidence_factor = 1.96; // 95% confidence interval

        let lower_bound = -variance * confidence_factor;
        let upper_bound = variance * confidence_factor;

        Ok((lower_bound, upper_bound))
    }

    /// Calculate historical variance
    fn calculate_historical_variance(&self) -> Result<f32> {
        if self.historical_data.len() < 2 {
            return Ok(0.1); // Default variance
        }

        let bandwidths: Vec<f32> = self.historical_data.iter().map(|c| c.bandwidth_mbps).collect();

        let mean = bandwidths.iter().sum::<f32>() / bandwidths.len() as f32;
        let variance =
            bandwidths.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / bandwidths.len() as f32;

        Ok(variance.sqrt())
    }

    /// Initialize prediction models
    fn initialize_prediction_models(&mut self) -> Result<()> {
        // Initialize bandwidth prediction model
        let bandwidth_model = PredictionModel::new(
            PredictionModelType::LinearRegression,
            vec![0.1, 0.2, -0.05, 0.3], // Simple weights
        );
        self.prediction_models.insert("bandwidth".to_string(), bandwidth_model);

        // Initialize latency prediction model
        let latency_model = PredictionModel::new(
            PredictionModelType::ExponentialSmoothing,
            vec![0.3, 0.7], // Alpha and beta parameters
        );
        self.prediction_models.insert("latency".to_string(), latency_model);

        Ok(())
    }

    /// Update prediction models with new data
    fn update_prediction_models(&mut self, conditions: &NetworkConditions) -> Result<()> {
        let features = self.extract_features(conditions, conditions.timestamp)?;

        // Update bandwidth model
        if let Some(bandwidth_model) = self.prediction_models.get_mut("bandwidth") {
            let training_point = TrainingDataPoint {
                features: features.clone(),
                target: conditions.bandwidth_mbps,
                timestamp: conditions.timestamp,
                weight: 1.0,
            };
            bandwidth_model.add_training_data(training_point);
        }

        // Update latency model
        if let Some(latency_model) = self.prediction_models.get_mut("latency") {
            let training_point = TrainingDataPoint {
                features,
                target: conditions.latency_ms,
                timestamp: conditions.timestamp,
                weight: 1.0,
            };
            latency_model.add_training_data(training_point);
        }

        Ok(())
    }

    /// Select best performing model
    fn select_best_model(&self) -> Result<PredictionModelType> {
        let mut best_model = PredictionModelType::LinearRegression;
        let mut best_accuracy = 0.0;

        for model in self.prediction_models.values() {
            if model.accuracy > best_accuracy {
                best_accuracy = model.accuracy;
                best_model = model.model_type.clone();
            }
        }

        Ok(best_model)
    }

    /// Get model accuracy
    fn get_model_accuracy(&self, model_type: &PredictionModelType) -> f32 {
        self.prediction_models
            .values()
            .find(|m| std::mem::discriminant(&m.model_type) == std::mem::discriminant(model_type))
            .map(|m| m.accuracy)
            .unwrap_or(0.5)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: NetworkAdaptationConfig) -> Result<()> {
        self.config = config.prediction_config;
        Ok(())
    }

    /// Get prediction statistics
    pub fn get_prediction_stats(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();

        // Overall accuracy
        let avg_accuracy = self.prediction_models.values().map(|m| m.accuracy).sum::<f32>()
            / self.prediction_models.len().max(1) as f32;
        stats.insert("average_accuracy".to_string(), avg_accuracy);

        // Historical data size
        stats.insert(
            "historical_data_points".to_string(),
            self.historical_data.len() as f32,
        );

        // Pattern confidence
        stats.insert(
            "pattern_confidence".to_string(),
            self.pattern_analyzer.get_average_confidence(),
        );

        stats
    }
}

impl NetworkPatternAnalyzer {
    /// Create new pattern analyzer
    pub fn new() -> Self {
        Self {
            daily_patterns: HashMap::new(),
            weekly_patterns: HashMap::new(),
            seasonal_patterns: HashMap::new(),
        }
    }

    /// Start pattern analysis
    pub fn start_analysis(&mut self) {
        // Initialize default patterns
        self.initialize_default_patterns();
    }

    /// Initialize default patterns
    fn initialize_default_patterns(&mut self) {
        // Initialize daily patterns for each hour
        for hour in 0..24 {
            let pattern = DailyPattern {
                hourly_bandwidth: [10.0; 24], // Default 10 Mbps
                hourly_latency: [50.0; 24],   // Default 50ms
                hourly_quality: [NetworkQuality::Fair; 24],
                confidence: 0.1, // Low initial confidence
                sample_count: 0,
            };
            self.daily_patterns.insert(hour, pattern);
        }

        // Initialize weekly patterns
        for day in 0..7 {
            let pattern = WeeklyPattern {
                daily_trends: [TrendDirection::Stable; 7],
                peak_hours: vec![9, 12, 18, 21], // Common peak hours
                low_usage_periods: vec![(2, 6), (14, 16)], // Night and early afternoon
                confidence: 0.1,
            };
            self.weekly_patterns.insert(day, pattern);
        }

        // Initialize seasonal patterns
        for month in 1..=12 {
            let pattern = SeasonalPattern {
                monthly_averages: MonthlyAverages {
                    bandwidth_mbps: 10.0,
                    latency_ms: 50.0,
                    packet_loss_percent: 1.0,
                    quality_score: 3.0,
                },
                seasonal_trends: TrendDirection::Stable,
                holiday_adjustments: HashMap::new(),
                confidence: 0.1,
            };
            self.seasonal_patterns.insert(month, pattern);
        }
    }

    /// Analyze network conditions and update patterns
    pub fn analyze_conditions(&mut self, conditions: &NetworkConditions) -> Result<()> {
        let now = Instant::now();
        let time_since_epoch = now.elapsed().as_secs();

        // Extract time components
        let hour = ((time_since_epoch / 3600) % 24) as u8;
        let day_of_week = ((time_since_epoch / (24 * 3600)) % 7) as u8;
        let month = ((time_since_epoch / (30 * 24 * 3600)) % 12 + 1) as u8;

        // Update daily patterns
        self.update_daily_pattern(hour, conditions)?;

        // Update weekly patterns
        self.update_weekly_pattern(day_of_week, hour, conditions)?;

        // Update seasonal patterns
        self.update_seasonal_pattern(month, conditions)?;

        Ok(())
    }

    /// Update daily pattern
    fn update_daily_pattern(&mut self, hour: u8, conditions: &NetworkConditions) -> Result<()> {
        if let Some(pattern) = self.daily_patterns.get_mut(&hour) {
            // Update with exponential moving average
            let alpha = 0.1; // Learning rate

            pattern.hourly_bandwidth[hour as usize] = pattern.hourly_bandwidth[hour as usize]
                * (1.0 - alpha)
                + conditions.bandwidth_mbps * alpha;

            pattern.hourly_latency[hour as usize] = pattern.hourly_latency[hour as usize]
                * (1.0 - alpha)
                + conditions.latency_ms * alpha;

            pattern.hourly_quality[hour as usize] = conditions.quality_assessment;

            // Update confidence and sample count
            pattern.sample_count += 1;
            pattern.confidence =
                (pattern.sample_count as f32 / (pattern.sample_count + 10) as f32).min(1.0);
        }

        Ok(())
    }

    /// Update weekly pattern
    fn update_weekly_pattern(
        &mut self,
        day: u8,
        hour: u8,
        conditions: &NetworkConditions,
    ) -> Result<()> {
        if let Some(pattern) = self.weekly_patterns.get_mut(&day) {
            // Update peak hours if this is a high-usage period
            if conditions.bandwidth_mbps > 15.0 && !pattern.peak_hours.contains(&hour) {
                pattern.peak_hours.push(hour);
                pattern.peak_hours.sort();

                // Keep only top 6 peak hours
                if pattern.peak_hours.len() > 6 {
                    pattern.peak_hours.remove(0);
                }
            }

            // Update confidence
            pattern.confidence = (pattern.confidence * 0.9 + 0.1).min(1.0);
        }

        Ok(())
    }

    /// Update seasonal pattern
    fn update_seasonal_pattern(&mut self, month: u8, conditions: &NetworkConditions) -> Result<()> {
        if let Some(pattern) = self.seasonal_patterns.get_mut(&month) {
            let alpha = 0.05; // Slower learning for seasonal patterns

            // Update monthly averages
            pattern.monthly_averages.bandwidth_mbps = pattern.monthly_averages.bandwidth_mbps
                * (1.0 - alpha)
                + conditions.bandwidth_mbps * alpha;

            pattern.monthly_averages.latency_ms =
                pattern.monthly_averages.latency_ms * (1.0 - alpha) + conditions.latency_ms * alpha;

            pattern.monthly_averages.packet_loss_percent =
                pattern.monthly_averages.packet_loss_percent * (1.0 - alpha)
                    + conditions.packet_loss_percent * alpha;

            // Update confidence
            pattern.confidence = (pattern.confidence * 0.95 + 0.05).min(1.0);
        }

        Ok(())
    }

    /// Apply pattern-based adjustments to predictions
    pub fn apply_pattern_adjustments(
        &self,
        mut conditions: NetworkConditions,
        target_time: Instant,
    ) -> Result<NetworkConditions> {
        let time_since_epoch = target_time.elapsed().as_secs();
        let hour = ((time_since_epoch / 3600) % 24) as u8;
        let day_of_week = ((time_since_epoch / (24 * 3600)) % 7) as u8;
        let month = ((time_since_epoch / (30 * 24 * 3600)) % 12 + 1) as u8;

        // Apply daily pattern adjustments
        if let Some(daily_pattern) = self.daily_patterns.get(&hour) {
            if daily_pattern.confidence > 0.3 {
                let adjustment_factor = daily_pattern.confidence;
                conditions.bandwidth_mbps = conditions.bandwidth_mbps * (1.0 - adjustment_factor)
                    + daily_pattern.hourly_bandwidth[hour as usize] * adjustment_factor;
                conditions.latency_ms = conditions.latency_ms * (1.0 - adjustment_factor)
                    + daily_pattern.hourly_latency[hour as usize] * adjustment_factor;
            }
        }

        // Apply weekly pattern adjustments
        if let Some(weekly_pattern) = self.weekly_patterns.get(&day_of_week) {
            if weekly_pattern.confidence > 0.3 {
                // Adjust for peak hours
                if weekly_pattern.peak_hours.contains(&hour) {
                    conditions.bandwidth_mbps *= 1.2; // 20% boost during peak hours
                    conditions.latency_ms *= 1.1; // 10% increase in latency
                }

                // Adjust for low usage periods
                for &(start, end) in &weekly_pattern.low_usage_periods {
                    if hour >= start && hour <= end {
                        conditions.bandwidth_mbps *= 0.8; // 20% reduction during low usage
                        conditions.latency_ms *= 0.9; // 10% improvement in latency
                    }
                }
            }
        }

        // Apply seasonal pattern adjustments
        if let Some(seasonal_pattern) = self.seasonal_patterns.get(&month) {
            if seasonal_pattern.confidence > 0.2 {
                let seasonal_factor = 0.1; // Small seasonal adjustment
                conditions.bandwidth_mbps = conditions.bandwidth_mbps * (1.0 - seasonal_factor)
                    + seasonal_pattern.monthly_averages.bandwidth_mbps * seasonal_factor;
            }
        }

        Ok(conditions)
    }

    /// Get average confidence across all patterns
    pub fn get_average_confidence(&self) -> f32 {
        let daily_conf: f32 = self.daily_patterns.values().map(|p| p.confidence).sum();
        let weekly_conf: f32 = self.weekly_patterns.values().map(|p| p.confidence).sum();
        let seasonal_conf: f32 = self.seasonal_patterns.values().map(|p| p.confidence).sum();

        let total_patterns =
            self.daily_patterns.len() + self.weekly_patterns.len() + self.seasonal_patterns.len();

        if total_patterns > 0 {
            (daily_conf + weekly_conf + seasonal_conf) / total_patterns as f32
        } else {
            0.0
        }
    }

    /// Get pattern insights for debugging
    pub fn get_pattern_insights(&self) -> HashMap<String, String> {
        let mut insights = HashMap::new();

        // Daily pattern insights
        if let Some(peak_hour) = self
            .daily_patterns
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.hourly_bandwidth[0]
                    .partial_cmp(&b.hourly_bandwidth[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(hour, _)| hour)
        {
            insights.insert("peak_daily_hour".to_string(), format!("{}:00", peak_hour));
        }

        // Weekly pattern insights
        let total_peak_hours: usize =
            self.weekly_patterns.values().map(|p| p.peak_hours.len()).sum();
        insights.insert(
            "total_weekly_peak_hours".to_string(),
            total_peak_hours.to_string(),
        );

        // Seasonal pattern insights
        let avg_seasonal_confidence =
            self.seasonal_patterns.values().map(|p| p.confidence).sum::<f32>()
                / self.seasonal_patterns.len().max(1) as f32;
        insights.insert(
            "seasonal_confidence".to_string(),
            format!("{:.2}", avg_seasonal_confidence),
        );

        insights
    }
}

impl PredictionModel {
    /// Create new prediction model
    pub fn new(model_type: PredictionModelType, initial_weights: Vec<f32>) -> Self {
        Self {
            model_type,
            weights: initial_weights,
            training_data: VecDeque::new(),
            accuracy: 0.5, // Start with 50% accuracy
            last_updated: Instant::now(),
        }
    }

    /// Add training data point
    pub fn add_training_data(&mut self, data_point: TrainingDataPoint) {
        self.training_data.push_back(data_point);

        // Limit training data size
        if self.training_data.len() > 1000 {
            self.training_data.pop_front();
        }

        // Retrain model periodically
        if self.training_data.len() % 10 == 0 {
            let _ = self.retrain();
        }
    }

    /// Make prediction
    pub fn predict(&self, features: &[f32]) -> Result<f32> {
        match self.model_type {
            PredictionModelType::LinearRegression => self.linear_regression_predict(features),
            PredictionModelType::ExponentialSmoothing => {
                self.exponential_smoothing_predict(features)
            },
            PredictionModelType::MovingAverage => self.moving_average_predict(features),
            PredictionModelType::NeuralNetwork => self.neural_network_predict(features),
            PredictionModelType::EnsembleMethod => self.ensemble_predict(features),
        }
    }

    /// Linear regression prediction
    fn linear_regression_predict(&self, features: &[f32]) -> Result<f32> {
        if features.len() != self.weights.len() {
            return Ok(0.0); // Fallback
        }

        let prediction = features.iter().zip(self.weights.iter()).map(|(f, w)| f * w).sum::<f32>();

        Ok(prediction.max(0.0)) // Ensure non-negative
    }

    /// Exponential smoothing prediction
    fn exponential_smoothing_predict(&self, _features: &[f32]) -> Result<f32> {
        if self.training_data.is_empty() {
            return Ok(0.0);
        }

        // Simple exponential smoothing
        let alpha = self.weights.first().unwrap_or(&0.3);
        let recent_values: Vec<f32> =
            self.training_data.iter().rev().take(10).map(|d| d.target).collect();

        if recent_values.is_empty() {
            return Ok(0.0);
        }

        let mut smoothed = recent_values[0];
        for &value in &recent_values[1..] {
            smoothed = alpha * value + (1.0 - alpha) * smoothed;
        }

        Ok(smoothed)
    }

    /// Moving average prediction
    fn moving_average_predict(&self, _features: &[f32]) -> Result<f32> {
        let window_size = *self.weights.first().unwrap_or(&5.0) as usize;

        if self.training_data.len() < window_size {
            return Ok(0.0);
        }

        let recent_avg =
            self.training_data.iter().rev().take(window_size).map(|d| d.target).sum::<f32>()
                / window_size as f32;

        Ok(recent_avg)
    }

    /// Neural network prediction (simplified)
    fn neural_network_predict(&self, features: &[f32]) -> Result<f32> {
        // Simplified single-layer neural network
        if features.len() < self.weights.len() {
            return Ok(0.0);
        }

        let weighted_sum =
            features.iter().zip(self.weights.iter()).map(|(f, w)| f * w).sum::<f32>();

        // Apply sigmoid activation
        let activated = 1.0 / (1.0 + (-weighted_sum).exp());

        Ok(activated * 100.0) // Scale to reasonable range
    }

    /// Ensemble prediction (placeholder)
    fn ensemble_predict(&self, features: &[f32]) -> Result<f32> {
        // Combine multiple prediction methods
        let linear = self.linear_regression_predict(features)?;
        let smoothing = self.exponential_smoothing_predict(features)?;
        let average = self.moving_average_predict(features)?;

        // Simple ensemble average
        Ok((linear + smoothing + average) / 3.0)
    }

    /// Retrain the model with accumulated data
    fn retrain(&mut self) -> Result<()> {
        match self.model_type {
            PredictionModelType::LinearRegression => self.retrain_linear_regression(),
            _ => Ok(()), // Other models use online learning
        }
    }

    /// Retrain linear regression model
    fn retrain_linear_regression(&mut self) -> Result<()> {
        if self.training_data.len() < 2 {
            return Ok(());
        }

        // Simple gradient descent (very simplified)
        let learning_rate = 0.01;
        let mut gradients = vec![0.0; self.weights.len()];

        for data_point in &self.training_data {
            if data_point.features.len() == self.weights.len() {
                let prediction = self.linear_regression_predict(&data_point.features)?;
                let error = data_point.target - prediction;

                for (i, &feature) in data_point.features.iter().enumerate() {
                    gradients[i] += error * feature;
                }
            }
        }

        // Update weights
        for (i, gradient) in gradients.iter().enumerate() {
            self.weights[i] += learning_rate * gradient / self.training_data.len() as f32;
        }

        // Update accuracy (simplified)
        self.update_accuracy()?;
        self.last_updated = Instant::now();

        Ok(())
    }

    /// Update model accuracy
    fn update_accuracy(&mut self) -> Result<()> {
        if self.training_data.len() < 5 {
            return Ok(());
        }

        let recent_data: Vec<_> = self.training_data.iter().rev().take(10).collect();
        let mut total_error = 0.0;

        for data_point in recent_data {
            if let Ok(prediction) = self.predict(&data_point.features) {
                let error = (prediction - data_point.target).abs() / data_point.target.max(1.0);
                total_error += error;
            }
        }

        let mean_error = total_error / 10.0;
        self.accuracy = (1.0 - mean_error).clamp(0.0, 1.0);

        Ok(())
    }
}

// Default implementations for convenience
impl Default for NetworkPatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DailyPattern {
    fn default() -> Self {
        Self {
            hourly_bandwidth: [10.0; 24],
            hourly_latency: [50.0; 24],
            hourly_quality: [NetworkQuality::Fair; 24],
            confidence: 0.1,
            sample_count: 0,
        }
    }
}

impl Default for WeeklyPattern {
    fn default() -> Self {
        Self {
            daily_trends: [TrendDirection::Stable; 7],
            peak_hours: vec![9, 12, 18, 21],
            low_usage_periods: vec![(2, 6), (14, 16)],
            confidence: 0.1,
        }
    }
}

impl Default for SeasonalPattern {
    fn default() -> Self {
        Self {
            monthly_averages: MonthlyAverages {
                bandwidth_mbps: 10.0,
                latency_ms: 50.0,
                packet_loss_percent: 1.0,
                quality_score: 3.0,
            },
            seasonal_trends: TrendDirection::Stable,
            holiday_adjustments: HashMap::new(),
            confidence: 0.1,
        }
    }
}
