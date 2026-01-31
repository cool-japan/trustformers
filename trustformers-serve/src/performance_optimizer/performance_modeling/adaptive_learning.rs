//! Adaptive Learning System for Performance Models
//!
//! This module provides comprehensive adaptive learning capabilities including
//! online learning, concept drift detection, active learning, and continuous
//! model adaptation for performance prediction in dynamic environments.

use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};

use super::types::*;
use crate::performance_optimizer::types::PerformanceDataPoint;

// =============================================================================
// ADAPTIVE LEARNING ENGINE
// =============================================================================

/// Adaptive learning engine for continuous model improvement
#[derive(Debug)]
pub struct AdaptiveLearningEngine {
    /// Learning configuration
    config: Arc<RwLock<AdaptiveLearningConfig>>,
    /// Performance history buffer
    performance_buffer: Arc<Mutex<VecDeque<PerformanceDataPoint>>>,
    /// Concept drift detector
    drift_detector: Arc<Mutex<ConceptDriftDetector>>,
    /// Online learners registry
    online_learners: Arc<RwLock<HashMap<String, Box<dyn OnlineLearner>>>>,
    /// Active learning controller
    active_learning: Arc<Mutex<ActiveLearningController>>,
    /// Learning metrics tracker
    metrics_tracker: Arc<Mutex<LearningMetricsTracker>>,
    /// Adaptation scheduler
    scheduler: Arc<Mutex<AdaptationScheduler>>,
}

impl AdaptiveLearningEngine {
    /// Create new adaptive learning engine
    pub fn new(config: AdaptiveLearningConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config.clone())),
            performance_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(
                config.adaptation_window * 2,
            ))),
            drift_detector: Arc::new(Mutex::new(ConceptDriftDetector::new(
                config.drift_threshold,
            ))),
            online_learners: Arc::new(RwLock::new(HashMap::new())),
            active_learning: Arc::new(Mutex::new(ActiveLearningController::new(
                config.uncertainty_threshold,
            ))),
            metrics_tracker: Arc::new(Mutex::new(LearningMetricsTracker::new())),
            scheduler: Arc::new(Mutex::new(AdaptationScheduler::new(
                config.update_frequency,
            ))),
        }
    }

    /// Register an online learner
    pub fn register_learner(&self, name: String, learner: Box<dyn OnlineLearner>) {
        let mut learners = self.online_learners.write();
        learners.insert(name, learner);
    }

    /// Process new data (alias for process_data_point for compatibility)
    pub async fn process_new_data(
        &self,
        data_points: Vec<PerformanceDataPoint>,
    ) -> Result<Vec<LearningUpdate>> {
        let mut updates = Vec::new();
        for data_point in &data_points {
            if let Some(update) = self.process_data_point(data_point).await? {
                updates.push(update);
            }
        }
        Ok(updates)
    }

    /// Process new performance data point
    pub async fn process_data_point(
        &self,
        data_point: &PerformanceDataPoint,
    ) -> Result<Option<LearningUpdate>> {
        // Add to performance buffer
        {
            let mut buffer = self.performance_buffer.lock();
            buffer.push_back(data_point.clone());

            let config = self.config.read();
            if buffer.len() > config.adaptation_window * 2 {
                buffer.pop_front();
            }
        }

        // Check if adaptation is needed
        let mut scheduler = self.scheduler.lock();
        if !scheduler.should_adapt() {
            return Ok(None);
        }
        drop(scheduler);

        // Detect concept drift
        let drift_detected = {
            let mut drift_detector = self.drift_detector.lock();
            drift_detector.detect_drift(data_point)?
        };

        if drift_detected {
            self.handle_concept_drift().await
        } else {
            self.perform_incremental_update(data_point).await
        }
    }

    /// Handle concept drift
    async fn handle_concept_drift(&self) -> Result<Option<LearningUpdate>> {
        let recent_data = {
            let buffer = self.performance_buffer.lock();
            let config = self.config.read();
            buffer.iter().rev().take(config.adaptation_window).cloned().collect::<Vec<_>>()
        };

        if recent_data.len() < self.config.read().min_adaptation_samples {
            return Ok(None);
        }

        let start_time = std::time::Instant::now();
        let mut successful_adaptations = 0;
        let mut total_performance_impact = 0.0f32;

        // Adapt all registered learners
        {
            let mut learners = self.online_learners.write();
            for (name, learner) in learners.iter_mut() {
                match learner.adapt_to_drift(&recent_data) {
                    Ok(impact) => {
                        successful_adaptations += 1;
                        total_performance_impact += impact;
                        tracing::info!(
                            "Successfully adapted learner '{}' with impact: {}",
                            name,
                            impact
                        );
                    },
                    Err(e) => {
                        tracing::warn!("Failed to adapt learner '{}': {}", name, e);
                    },
                }
            }
        }

        // Update metrics
        let learning_metrics = LearningMetrics {
            learning_rate: self.config.read().learning_rate_decay,
            gradient_norm: 0.0, // Would be calculated from actual gradients
            loss_reduction: total_performance_impact / successful_adaptations.max(1) as f32,
            convergence_score: if successful_adaptations > 0 { 0.8 } else { 0.0 },
            training_time: start_time.elapsed(),
            memory_usage_mb: self.estimate_memory_usage(),
        };

        {
            let mut tracker = self.metrics_tracker.lock();
            tracker.record_adaptation(&learning_metrics);
        }

        Ok(Some(LearningUpdate {
            update_type: LearningUpdateType::ConceptDriftAdaptation,
            performance_impact: total_performance_impact / successful_adaptations.max(1) as f32,
            confidence_delta: 0.1, // Simplified
            learning_metrics,
            updated_at: Utc::now(),
        }))
    }

    /// Perform incremental learning update
    async fn perform_incremental_update(
        &self,
        data_point: &PerformanceDataPoint,
    ) -> Result<Option<LearningUpdate>> {
        let start_time = std::time::Instant::now();
        let mut total_impact = 0.0f32;
        let mut update_count = 0;

        // Check if active learning is needed
        let should_use_active_learning = {
            let mut active_learning = self.active_learning.lock();
            active_learning.should_query_oracle(data_point)?
        };

        if should_use_active_learning {
            return self.perform_active_learning_update(data_point).await;
        }

        // Perform incremental updates
        {
            let mut learners = self.online_learners.write();
            for (name, learner) in learners.iter_mut() {
                match learner.incremental_update(data_point) {
                    Ok(impact) => {
                        total_impact += impact;
                        update_count += 1;
                        tracing::debug!(
                            "Incremental update for '{}' with impact: {}",
                            name,
                            impact
                        );
                    },
                    Err(e) => {
                        tracing::warn!("Failed incremental update for '{}': {}", name, e);
                    },
                }
            }
        }

        if update_count == 0 {
            return Ok(None);
        }

        let learning_metrics = LearningMetrics {
            learning_rate: self.config.read().learning_rate_decay,
            gradient_norm: 0.5, // Simplified
            loss_reduction: total_impact / update_count as f32,
            convergence_score: 0.9, // Incremental updates generally converge well
            training_time: start_time.elapsed(),
            memory_usage_mb: self.estimate_memory_usage(),
        };

        Ok(Some(LearningUpdate {
            update_type: LearningUpdateType::Incremental,
            performance_impact: total_impact / update_count as f32,
            confidence_delta: 0.02, // Small confidence change for incremental updates
            learning_metrics,
            updated_at: Utc::now(),
        }))
    }

    /// Perform active learning update
    async fn perform_active_learning_update(
        &self,
        data_point: &PerformanceDataPoint,
    ) -> Result<Option<LearningUpdate>> {
        let start_time = std::time::Instant::now();

        // Simulate active learning query (in practice, this would involve human expert or oracle)
        let oracle_response = self.simulate_oracle_query(data_point)?;

        let mut total_impact = 0.0f32;
        let mut update_count = 0;

        // Update learners with oracle-labeled data
        {
            let mut learners = self.online_learners.write();
            for (name, learner) in learners.iter_mut() {
                if let Some(ref oracle_data) = oracle_response {
                    match learner.active_learning_update(oracle_data) {
                        Ok(impact) => {
                            total_impact += impact;
                            update_count += 1;
                            tracing::info!(
                                "Active learning update for '{}' with impact: {}",
                                name,
                                impact
                            );
                        },
                        Err(e) => {
                            tracing::warn!("Failed active learning update for '{}': {}", name, e);
                        },
                    }
                }
            }
        }

        // Update active learning controller
        {
            let mut active_learning = self.active_learning.lock();
            active_learning.update_query_strategy(data_point, &oracle_response)?;
        }

        let learning_metrics = LearningMetrics {
            learning_rate: self.config.read().learning_rate_decay * 1.5, // Higher learning rate for active learning
            gradient_norm: 0.8,
            loss_reduction: total_impact / update_count.max(1) as f32,
            convergence_score: 0.85,
            training_time: start_time.elapsed(),
            memory_usage_mb: self.estimate_memory_usage(),
        };

        Ok(Some(LearningUpdate {
            update_type: LearningUpdateType::ActiveLearning,
            performance_impact: total_impact / update_count.max(1) as f32,
            confidence_delta: 0.05, // Active learning can provide more confidence
            learning_metrics,
            updated_at: Utc::now(),
        }))
    }

    /// Simulate oracle query for active learning
    fn simulate_oracle_query(
        &self,
        data_point: &PerformanceDataPoint,
    ) -> Result<Option<OracleResponse>> {
        // In a real implementation, this would query a human expert or external oracle
        // For simulation, we provide a response based on heuristics

        let confidence_score = if data_point.throughput > 100.0 { 0.9 } else { 0.7 };

        Ok(Some(OracleResponse {
            data_point: data_point.clone(),
            expert_label: data_point.throughput, // Use actual throughput as "expert" label
            confidence: confidence_score,
            explanation: format!(
                "Simulated oracle response for throughput {}",
                data_point.throughput
            ),
            queried_at: Utc::now(),
        }))
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> f32 {
        let buffer_size = {
            let buffer = self.performance_buffer.lock();
            buffer.len() * std::mem::size_of::<PerformanceDataPoint>()
        };

        let learners_size = self.online_learners.read().len() * 1024; // Simplified estimate

        ((buffer_size + learners_size) as f32) / 1024.0 / 1024.0 // Convert to MB
    }

    /// Get learning statistics
    pub fn get_learning_statistics(&self) -> LearningStatistics {
        let metrics_tracker = self.metrics_tracker.lock();
        let buffer_size = self.performance_buffer.lock().len();
        let active_learners = self.online_learners.read().len();

        LearningStatistics {
            total_adaptations: metrics_tracker.total_adaptations,
            successful_adaptations: metrics_tracker.successful_adaptations,
            average_adaptation_time: metrics_tracker.average_adaptation_time(),
            current_learning_rate: self.config.read().learning_rate_decay,
            buffer_utilization: buffer_size as f32 / self.config.read().adaptation_window as f32,
            active_learners,
            drift_detections: metrics_tracker.drift_detections,
            active_learning_queries: metrics_tracker.active_learning_queries,
            last_adaptation: metrics_tracker.last_adaptation,
        }
    }
}

// =============================================================================
// CONCEPT DRIFT DETECTOR
// =============================================================================

/// Concept drift detector using statistical methods
#[derive(Debug)]
pub struct ConceptDriftDetector {
    /// Drift detection threshold
    threshold: f32,
    /// Reference window for drift detection
    reference_window: VecDeque<f64>,
    /// Detection window
    detection_window: VecDeque<f64>,
    /// Statistical test results history
    test_history: VecDeque<DriftTestResult>,
    /// Window size for drift detection
    window_size: usize,
}

#[derive(Debug, Clone)]
pub struct DriftTestResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub drift_detected: bool,
    pub tested_at: DateTime<Utc>,
}

impl ConceptDriftDetector {
    /// Create new concept drift detector
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            reference_window: VecDeque::with_capacity(100),
            detection_window: VecDeque::with_capacity(50),
            test_history: VecDeque::with_capacity(20),
            window_size: 50,
        }
    }

    /// Detect concept drift in new data point
    pub fn detect_drift(&mut self, data_point: &PerformanceDataPoint) -> Result<bool> {
        // Add data point to detection window
        self.detection_window.push_back(data_point.throughput);
        if self.detection_window.len() > self.window_size {
            // Move oldest point from detection to reference window
            if let Some(old_point) = self.detection_window.pop_front() {
                self.reference_window.push_back(old_point);
                if self.reference_window.len() > self.window_size * 2 {
                    self.reference_window.pop_front();
                }
            }
        }

        // Need sufficient data in both windows for drift detection
        if self.reference_window.len() < self.window_size / 2
            || self.detection_window.len() < self.window_size / 2
        {
            return Ok(false);
        }

        // Perform statistical test for drift detection
        let drift_detected = self.perform_drift_test()?;

        // Record test result
        let test_result = DriftTestResult {
            test_statistic: self.calculate_test_statistic()?,
            p_value: 0.05, // Simplified
            drift_detected,
            tested_at: Utc::now(),
        };

        self.test_history.push_back(test_result);
        if self.test_history.len() > 20 {
            self.test_history.pop_front();
        }

        Ok(drift_detected)
    }

    /// Perform statistical test for drift detection
    fn perform_drift_test(&self) -> Result<bool> {
        // Use Kolmogorov-Smirnov test to detect distribution changes
        let ks_statistic = self.kolmogorov_smirnov_test()?;
        Ok(ks_statistic > self.threshold as f64)
    }

    /// Kolmogorov-Smirnov test implementation
    fn kolmogorov_smirnov_test(&self) -> Result<f64> {
        let ref_data: Vec<f64> = self.reference_window.iter().cloned().collect();
        let det_data: Vec<f64> = self.detection_window.iter().cloned().collect();

        if ref_data.is_empty() || det_data.is_empty() {
            return Ok(0.0);
        }

        // Sort both datasets
        let mut sorted_ref = ref_data.clone();
        let mut sorted_det = det_data.clone();
        sorted_ref.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_det.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate empirical CDFs and find maximum difference
        let mut max_diff = 0.0_f64;
        let all_values = [sorted_ref.clone(), sorted_det.clone()].concat();
        let mut unique_values = all_values;
        unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_values.dedup();

        for value in unique_values {
            let cdf_ref = self.empirical_cdf(&sorted_ref, value);
            let cdf_det = self.empirical_cdf(&sorted_det, value);
            let diff = (cdf_ref - cdf_det).abs();
            // TODO: Added f64 type annotation to fix E0689 ambiguous numeric type
            max_diff = max_diff.max(diff);
        }

        Ok(max_diff)
    }

    /// Calculate empirical CDF value
    fn empirical_cdf(&self, sorted_data: &[f64], value: f64) -> f64 {
        let count = sorted_data.iter().take_while(|&&x| x <= value).count();
        count as f64 / sorted_data.len() as f64
    }

    /// Calculate test statistic
    fn calculate_test_statistic(&self) -> Result<f64> {
        // Use mean difference as a simple test statistic
        let ref_mean =
            self.reference_window.iter().sum::<f64>() / self.reference_window.len().max(1) as f64;
        let det_mean =
            self.detection_window.iter().sum::<f64>() / self.detection_window.len().max(1) as f64;
        Ok((ref_mean - det_mean).abs())
    }

    /// Get drift detection history
    pub fn get_detection_history(&self) -> Vec<DriftTestResult> {
        self.test_history.iter().cloned().collect()
    }
}

// =============================================================================
// ONLINE LEARNER TRAIT AND IMPLEMENTATIONS
// =============================================================================

/// Trait for online learning algorithms
pub trait OnlineLearner: std::fmt::Debug + Send + Sync {
    /// Perform incremental update with new data point
    fn incremental_update(&mut self, data_point: &PerformanceDataPoint) -> Result<f32>;

    /// Adapt to concept drift
    fn adapt_to_drift(&mut self, adaptation_data: &[PerformanceDataPoint]) -> Result<f32>;

    /// Active learning update with oracle-labeled data
    fn active_learning_update(&mut self, oracle_data: &OracleResponse) -> Result<f32>;

    /// Get learner name
    fn name(&self) -> &str;

    /// Get current learning rate
    fn learning_rate(&self) -> f32;

    /// Set learning rate
    fn set_learning_rate(&mut self, rate: f32);
}

/// Online gradient descent learner
#[derive(Debug)]
pub struct OnlineGradientDescentLearner {
    /// Model weights
    weights: Vec<f64>,
    /// Learning rate
    learning_rate: f32,
    /// Momentum coefficient
    momentum: f32,
    /// Previous gradients for momentum
    previous_gradients: Vec<f64>,
    /// Learner name
    name: String,
    /// Learning statistics
    stats: OnlineLearningStats,
}

#[derive(Debug, Clone)]
struct OnlineLearningStats {
    total_updates: u64,
    cumulative_loss: f64,
    last_gradient_norm: f64,
    convergence_indicator: f64,
}

impl OnlineGradientDescentLearner {
    /// Create new online gradient descent learner
    pub fn new(name: String, feature_count: usize, learning_rate: f32) -> Self {
        Self {
            weights: vec![0.0; feature_count],
            learning_rate,
            momentum: 0.9,
            previous_gradients: vec![0.0; feature_count],
            name,
            stats: OnlineLearningStats {
                total_updates: 0,
                cumulative_loss: 0.0,
                last_gradient_norm: 0.0,
                convergence_indicator: 1.0,
            },
        }
    }

    /// Extract features from data point
    fn extract_features(&self, data_point: &PerformanceDataPoint) -> Vec<f64> {
        vec![
            data_point.parallelism as f64,
            data_point.system_state.available_cores as f64,
            data_point.system_state.load_average as f64,
            data_point.test_characteristics.average_duration.as_secs_f64(),
            data_point.test_characteristics.resource_intensity.cpu_intensity as f64,
        ]
    }

    /// Predict throughput
    fn predict(&self, features: &[f64]) -> f64 {
        let mut prediction = 0.0;
        for (weight, &feature) in self.weights.iter().zip(features.iter()) {
            prediction += weight * feature;
        }
        prediction.max(0.0)
    }

    /// Calculate loss
    fn calculate_loss(&self, predicted: f64, actual: f64) -> f64 {
        (predicted - actual).powi(2) / 2.0
    }

    /// Calculate gradients
    fn calculate_gradients(&self, features: &[f64], predicted: f64, actual: f64) -> Vec<f64> {
        let error = predicted - actual;
        features.iter().map(|&feature| error * feature).collect()
    }

    /// Update weights with momentum
    fn update_weights(&mut self, gradients: &[f64]) {
        for i in 0..self.weights.len() {
            // Momentum update
            self.previous_gradients[i] = (self.momentum as f64) * self.previous_gradients[i]
                + (1.0 - (self.momentum as f64)) * gradients[i];

            // Weight update
            self.weights[i] -= (self.learning_rate as f64) * self.previous_gradients[i];
        }

        // Update statistics
        self.stats.last_gradient_norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
    }
}

impl OnlineLearner for OnlineGradientDescentLearner {
    fn incremental_update(&mut self, data_point: &PerformanceDataPoint) -> Result<f32> {
        let features = self.extract_features(data_point);
        let predicted = self.predict(&features);
        let actual = data_point.throughput;
        let loss = self.calculate_loss(predicted, actual);

        // Calculate gradients and update weights
        let gradients = self.calculate_gradients(&features, predicted, actual);
        self.update_weights(&gradients);

        // Update statistics
        self.stats.total_updates += 1;
        self.stats.cumulative_loss += loss;

        // Calculate performance impact (reduction in loss)
        let previous_loss = self.stats.cumulative_loss / self.stats.total_updates.max(1) as f64;
        let current_loss = loss;
        let impact = ((previous_loss - current_loss) / previous_loss.max(0.001)) as f32;

        Ok(impact.clamp(-1.0, 1.0))
    }

    fn adapt_to_drift(&mut self, adaptation_data: &[PerformanceDataPoint]) -> Result<f32> {
        if adaptation_data.is_empty() {
            return Ok(0.0);
        }

        // Increase learning rate for faster adaptation
        let original_rate = self.learning_rate;
        self.learning_rate *= 2.0;

        let mut total_impact = 0.0;
        for data_point in adaptation_data {
            let impact = self.incremental_update(data_point)?;
            total_impact += impact;
        }

        // Restore learning rate
        self.learning_rate = original_rate;

        Ok(total_impact / adaptation_data.len() as f32)
    }

    fn active_learning_update(&mut self, oracle_data: &OracleResponse) -> Result<f32> {
        // Use oracle label as target
        let mut modified_data_point = oracle_data.data_point.clone();
        modified_data_point.throughput = oracle_data.expert_label;

        // Weight the update by oracle confidence
        let original_rate = self.learning_rate;
        self.learning_rate *= oracle_data.confidence;

        let impact = self.incremental_update(&modified_data_point)?;

        // Restore learning rate
        self.learning_rate = original_rate;

        Ok(impact * oracle_data.confidence)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate;
    }
}

// =============================================================================
// ACTIVE LEARNING CONTROLLER
// =============================================================================

/// Active learning controller for intelligent sample selection
#[derive(Debug)]
pub struct ActiveLearningController {
    /// Uncertainty threshold for querying oracle
    uncertainty_threshold: f32,
    /// Query history
    query_history: VecDeque<ActiveLearningQuery>,
    /// Oracle response cache
    oracle_cache: HashMap<String, OracleResponse>,
    /// Query strategy
    strategy: ActiveLearningStrategy,
    /// Budget constraints
    budget: QueryBudget,
}

#[derive(Debug, Clone)]
struct ActiveLearningQuery {
    data_point: PerformanceDataPoint,
    uncertainty_score: f32,
    queried_at: DateTime<Utc>,
    response_received: bool,
}

#[derive(Debug, Clone)]
pub enum ActiveLearningStrategy {
    /// Query points with highest uncertainty
    UncertaintySampling,
    /// Query points that maximize information gain
    InformationGain,
    /// Query points that represent diverse regions
    DiversitySampling,
    /// Hybrid approach combining multiple strategies
    Hybrid,
}

#[derive(Debug, Clone)]
struct QueryBudget {
    /// Maximum queries per time period
    max_queries_per_hour: usize,
    /// Queries used in current hour
    queries_this_hour: usize,
    /// Last query time
    last_query_time: DateTime<Utc>,
}

impl ActiveLearningController {
    /// Create new active learning controller
    pub fn new(uncertainty_threshold: f32) -> Self {
        Self {
            uncertainty_threshold,
            query_history: VecDeque::with_capacity(1000),
            oracle_cache: HashMap::new(),
            strategy: ActiveLearningStrategy::UncertaintySampling,
            budget: QueryBudget {
                max_queries_per_hour: 10,
                queries_this_hour: 0,
                last_query_time: Utc::now() - ChronoDuration::hours(1),
            },
        }
    }

    /// Determine if oracle should be queried for this data point
    pub fn should_query_oracle(&mut self, data_point: &PerformanceDataPoint) -> Result<bool> {
        // Check budget constraints
        if !self.check_budget() {
            return Ok(false);
        }

        // Calculate uncertainty score
        let uncertainty = self.calculate_uncertainty(data_point)?;

        // Apply strategy-specific logic
        let should_query = match self.strategy {
            ActiveLearningStrategy::UncertaintySampling => uncertainty > self.uncertainty_threshold,
            ActiveLearningStrategy::InformationGain => {
                self.calculate_information_gain(data_point)? > 0.1
            },
            ActiveLearningStrategy::DiversitySampling => self.is_diverse_sample(data_point)?,
            ActiveLearningStrategy::Hybrid => {
                uncertainty > self.uncertainty_threshold && self.is_diverse_sample(data_point)?
            },
        };

        if should_query {
            // Record the query
            self.query_history.push_back(ActiveLearningQuery {
                data_point: data_point.clone(),
                uncertainty_score: uncertainty,
                queried_at: Utc::now(),
                response_received: false,
            });

            // Update budget
            self.budget.queries_this_hour += 1;
            self.budget.last_query_time = Utc::now();
        }

        Ok(should_query)
    }

    /// Update query strategy based on oracle response
    pub fn update_query_strategy(
        &mut self,
        data_point: &PerformanceDataPoint,
        oracle_response: &Option<OracleResponse>,
    ) -> Result<()> {
        if let Some(response) = oracle_response {
            // Cache the oracle response
            let cache_key = self.generate_cache_key(data_point);
            self.oracle_cache.insert(cache_key, response.clone());

            // Mark query as completed
            if let Some(query) = self.query_history.back_mut() {
                if query.data_point.timestamp == data_point.timestamp {
                    query.response_received = true;
                }
            }

            // Adapt strategy based on oracle feedback quality
            self.adapt_strategy_from_feedback(response)?;
        }

        Ok(())
    }

    /// Check if budget allows for more queries
    fn check_budget(&mut self) -> bool {
        let now = Utc::now();

        // Reset budget if hour has passed
        if now - self.budget.last_query_time > ChronoDuration::hours(1) {
            self.budget.queries_this_hour = 0;
        }

        self.budget.queries_this_hour < self.budget.max_queries_per_hour
    }

    /// Calculate uncertainty score for data point
    fn calculate_uncertainty(&self, data_point: &PerformanceDataPoint) -> Result<f32> {
        // Simplified uncertainty calculation based on system variability
        let system_variability = data_point.system_state.load_average;
        let resource_pressure = data_point.system_state.io_wait_percent;

        let uncertainty = (system_variability + resource_pressure) / 2.0;
        Ok(uncertainty.min(1.0))
    }

    /// Calculate information gain for data point
    fn calculate_information_gain(&self, _data_point: &PerformanceDataPoint) -> Result<f32> {
        // Simplified information gain calculation
        // In practice, this would use entropy-based measures
        Ok(0.15) // Placeholder
    }

    /// Check if data point represents a diverse sample
    fn is_diverse_sample(&self, data_point: &PerformanceDataPoint) -> Result<bool> {
        // Check diversity against recent queries
        let recent_queries: Vec<_> = self.query_history.iter().rev().take(10).collect();

        if recent_queries.is_empty() {
            return Ok(true);
        }

        // Calculate diversity score based on feature differences
        let mut min_distance = f64::INFINITY;
        for query in recent_queries {
            let distance = self.calculate_feature_distance(data_point, &query.data_point);
            min_distance = min_distance.min(distance);
        }

        // Consider diverse if minimum distance is above threshold
        Ok(min_distance > 0.2)
    }

    /// Calculate feature distance between two data points
    fn calculate_feature_distance(
        &self,
        point1: &PerformanceDataPoint,
        point2: &PerformanceDataPoint,
    ) -> f64 {
        let features1 = [
            point1.parallelism as f64,
            point1.system_state.load_average as f64,
            point1.test_characteristics.resource_intensity.cpu_intensity as f64,
        ];

        let features2 = [
            point2.parallelism as f64,
            point2.system_state.load_average as f64,
            point2.test_characteristics.resource_intensity.cpu_intensity as f64,
        ];

        // Euclidean distance
        features1
            .iter()
            .zip(features2.iter())
            .map(|(f1, f2)| (f1 - f2).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Generate cache key for data point
    fn generate_cache_key(&self, data_point: &PerformanceDataPoint) -> String {
        format!(
            "{}_{}_{}",
            data_point.parallelism,
            data_point.system_state.load_average as u32,
            data_point.timestamp.timestamp()
        )
    }

    /// Adapt strategy based on oracle feedback
    fn adapt_strategy_from_feedback(&mut self, response: &OracleResponse) -> Result<()> {
        // If oracle confidence is low, consider changing strategy
        if response.confidence < 0.5 {
            self.strategy = match self.strategy {
                ActiveLearningStrategy::UncertaintySampling => {
                    ActiveLearningStrategy::DiversitySampling
                },
                ActiveLearningStrategy::DiversitySampling => {
                    ActiveLearningStrategy::InformationGain
                },
                ActiveLearningStrategy::InformationGain => ActiveLearningStrategy::Hybrid,
                ActiveLearningStrategy::Hybrid => ActiveLearningStrategy::UncertaintySampling,
            };
        }

        Ok(())
    }

    /// Get active learning statistics
    pub fn get_statistics(&self) -> ActiveLearningStatistics {
        let total_queries = self.query_history.len();
        let completed_queries = self.query_history.iter().filter(|q| q.response_received).count();

        let average_uncertainty = if !self.query_history.is_empty() {
            self.query_history.iter().map(|q| q.uncertainty_score).sum::<f32>()
                / self.query_history.len() as f32
        } else {
            0.0
        };

        ActiveLearningStatistics {
            total_queries,
            completed_queries,
            current_strategy: self.strategy.clone(),
            average_uncertainty,
            budget_utilization: self.budget.queries_this_hour as f32
                / self.budget.max_queries_per_hour as f32,
            cache_hit_rate: self.calculate_cache_hit_rate(),
        }
    }

    /// Calculate cache hit rate
    fn calculate_cache_hit_rate(&self) -> f32 {
        if self.query_history.is_empty() {
            return 0.0;
        }

        let cache_hits = self.oracle_cache.len();
        cache_hits as f32 / self.query_history.len() as f32
    }
}

// =============================================================================
// SUPPORTING TYPES
// =============================================================================

/// Oracle response for active learning
#[derive(Debug, Clone)]
pub struct OracleResponse {
    /// Original data point
    pub data_point: PerformanceDataPoint,
    /// Expert-provided label
    pub expert_label: f64,
    /// Confidence in the label
    pub confidence: f32,
    /// Explanation or reasoning
    pub explanation: String,
    /// Response timestamp
    pub queried_at: DateTime<Utc>,
}

/// Learning metrics tracker
#[derive(Debug)]
pub struct LearningMetricsTracker {
    /// Total adaptation attempts
    pub total_adaptations: u64,
    /// Successful adaptations
    pub successful_adaptations: u64,
    /// Adaptation times
    adaptation_times: Vec<Duration>,
    /// Drift detections
    pub drift_detections: u64,
    /// Active learning queries
    pub active_learning_queries: u64,
    /// Last adaptation timestamp
    pub last_adaptation: Option<DateTime<Utc>>,
}

impl Default for LearningMetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningMetricsTracker {
    /// Create new metrics tracker
    pub fn new() -> Self {
        Self {
            total_adaptations: 0,
            successful_adaptations: 0,
            adaptation_times: Vec::new(),
            drift_detections: 0,
            active_learning_queries: 0,
            last_adaptation: None,
        }
    }

    /// Record an adaptation
    pub fn record_adaptation(&mut self, metrics: &LearningMetrics) {
        self.total_adaptations += 1;
        if metrics.loss_reduction > 0.0 {
            self.successful_adaptations += 1;
        }
        self.adaptation_times.push(metrics.training_time);
        self.last_adaptation = Some(Utc::now());

        // Keep only recent adaptation times
        if self.adaptation_times.len() > 100 {
            self.adaptation_times.remove(0);
        }
    }

    /// Calculate average adaptation time
    pub fn average_adaptation_time(&self) -> Duration {
        if self.adaptation_times.is_empty() {
            return Duration::from_secs(0);
        }

        let total_nanos: u128 = self.adaptation_times.iter().map(|d| d.as_nanos()).sum();

        Duration::from_nanos((total_nanos / self.adaptation_times.len() as u128) as u64)
    }
}

/// Adaptation scheduler
#[derive(Debug)]
pub struct AdaptationScheduler {
    /// Update frequency
    update_frequency: Duration,
    /// Last adaptation time
    last_adaptation: DateTime<Utc>,
    /// Adaptation counter
    adaptation_count: u64,
}

impl AdaptationScheduler {
    /// Create new adaptation scheduler
    pub fn new(update_frequency: Duration) -> Self {
        let chrono_duration = ChronoDuration::from_std(update_frequency)
            .unwrap_or_else(|_| ChronoDuration::seconds(60));
        Self {
            update_frequency,
            last_adaptation: Utc::now() - chrono_duration,
            adaptation_count: 0,
        }
    }

    /// Check if adaptation should be performed
    pub fn should_adapt(&mut self) -> bool {
        let now = Utc::now();
        let time_since_last = now - self.last_adaptation;

        let threshold = ChronoDuration::from_std(self.update_frequency)
            .unwrap_or_else(|_| ChronoDuration::seconds(60));
        if time_since_last > threshold {
            self.last_adaptation = now;
            self.adaptation_count += 1;
            true
        } else {
            false
        }
    }
}

/// Learning statistics
#[derive(Debug, Clone)]
pub struct LearningStatistics {
    /// Total adaptations performed
    pub total_adaptations: u64,
    /// Successful adaptations
    pub successful_adaptations: u64,
    /// Average adaptation time
    pub average_adaptation_time: Duration,
    /// Current learning rate
    pub current_learning_rate: f32,
    /// Buffer utilization (0.0 to 1.0)
    pub buffer_utilization: f32,
    /// Number of active learners
    pub active_learners: usize,
    /// Number of drift detections
    pub drift_detections: u64,
    /// Number of active learning queries
    pub active_learning_queries: u64,
    /// Last adaptation timestamp
    pub last_adaptation: Option<DateTime<Utc>>,
}

/// Active learning statistics
#[derive(Debug, Clone)]
pub struct ActiveLearningStatistics {
    /// Total oracle queries
    pub total_queries: usize,
    /// Completed queries with responses
    pub completed_queries: usize,
    /// Current query strategy
    pub current_strategy: ActiveLearningStrategy,
    /// Average uncertainty of queried samples
    pub average_uncertainty: f32,
    /// Budget utilization
    pub budget_utilization: f32,
    /// Cache hit rate for oracle responses
    pub cache_hit_rate: f32,
}

// =============================================================================
// TYPE ALIAS FOR COMPATIBILITY
// =============================================================================

/// Orchestrator alias for AdaptiveLearningEngine
/// Provides compatibility with naming convention used in other modules
pub type AdaptiveLearningOrchestrator = AdaptiveLearningEngine;
