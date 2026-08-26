//! Adaptive Learning System for Performance Models
//!
//! This module provides comprehensive adaptive learning capabilities including
//! online learning, concept drift detection, active learning, and continuous
//! model adaptation for performance prediction in dynamic environments.

use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};

use super::types::*;
use crate::performance_optimizer::real_time_metrics::analytics::analyzers::series::ks_p_value;
use crate::performance_optimizer::types::PerformanceDataPoint;

// =============================================================================
// MEASUREMENT HELPERS
// =============================================================================

/// Number of features the active-learning controller reasons over.
const ACTIVE_LEARNING_FEATURES: usize = 3;

/// The feature vector the active-learning controller compares points by.
///
/// Parallelism, host load average and the test's CPU intensity: the three
/// numbers that were already being differenced in
/// [`ActiveLearningController::calculate_feature_distance`], extracted once so
/// the distance and the information gain describe the same space.
fn active_learning_features(data_point: &PerformanceDataPoint) -> [f64; ACTIVE_LEARNING_FEATURES] {
    [
        data_point.parallelism as f64,
        data_point.system_state.load_average as f64,
        data_point.test_characteristics.resource_intensity.cpu_intensity as f64,
    ]
}

/// Solve `matrix · x = rhs` for a small positive-definite matrix.
///
/// Gaussian elimination with partial pivoting. `None` when the matrix is
/// singular to working precision, which a positive-definite design cannot be
/// unless it has overflowed.
fn solve_positive_definite(
    matrix: &[[f64; ACTIVE_LEARNING_FEATURES]; ACTIVE_LEARNING_FEATURES],
    rhs: &[f64; ACTIVE_LEARNING_FEATURES],
) -> Option<[f64; ACTIVE_LEARNING_FEATURES]> {
    const N: usize = ACTIVE_LEARNING_FEATURES;
    let mut work = [[0.0f64; N + 1]; N];
    for (i, row) in work.iter_mut().enumerate() {
        row[..N].copy_from_slice(&matrix[i]);
        row[N] = rhs[i];
    }

    for column in 0..N {
        let mut pivot_row = column;
        for row in column + 1..N {
            if work[row][column].abs() > work[pivot_row][column].abs() {
                pivot_row = row;
            }
        }
        if work[pivot_row][column].abs() < 1e-12 {
            return None;
        }
        work.swap(column, pivot_row);

        let pivot = work[column][column];
        for value in work[column].iter_mut() {
            *value /= pivot;
        }
        for row in 0..N {
            if row == column {
                continue;
            }
            let factor = work[row][column];
            if factor == 0.0 {
                continue;
            }
            for index in 0..=N {
                work[row][index] -= factor * work[column][index];
            }
        }
    }

    let mut solution = [0.0f64; N];
    for (index, value) in solution.iter_mut().enumerate() {
        *value = work[index][N];
        if !value.is_finite() {
            return None;
        }
    }
    Some(solution)
}

/// Norm of the joint gradient of one adaptation round.
///
/// Stacking the learners' gradients into one vector gives it the norm
/// `sqrt(Σ‖gᵢ‖²)`, so that is what a round of updates reports. `None` when no
/// learner reported a gradient -- the metric then says "not measured" rather
/// than `0.0`, which is the norm of a gradient that is genuinely zero.
fn joint_gradient_norm(norms: &[f64]) -> Option<f32> {
    if norms.is_empty() {
        return None;
    }
    let sum_of_squares: f64 = norms.iter().map(|norm| norm * norm).sum();
    Some(sum_of_squares.sqrt() as f32)
}

/// Share of the learners updated in a round whose loss impact was positive.
///
/// `None` when no learner was updated, because zero of zero is not a share.
fn improving_fraction(improving: usize, total: usize) -> Option<f32> {
    (total > 0).then(|| improving as f32 / total as f32)
}

// =============================================================================
// ORACLE
// =============================================================================

/// A source of ground-truth labels for the active-learning path.
///
/// Implementors answer with the throughput an expert (or a slower, more
/// trustworthy measurement) attributes to a performance point. There is no
/// in-tree implementation: an operator supplies one through
/// [`AdaptiveLearningEngine::set_oracle`], and until they do the active-learning
/// path is never entered.
pub trait PerformanceOracle: std::fmt::Debug + Send + Sync {
    /// Label `data_point`, or answer `None` to decline this query.
    ///
    /// # Errors
    ///
    /// Implementations return an error when the oracle could not be reached or
    /// answered unusably; the engine propagates it rather than continuing with
    /// an assumed label.
    fn label(&self, data_point: &PerformanceDataPoint) -> Result<Option<OracleResponse>>;

    /// Name used in logs and in the errors raised around this oracle.
    fn name(&self) -> &str;
}

// =============================================================================
// ADAPTIVE LEARNING ENGINE
// =============================================================================

/// Adaptive learning engine for continuous model improvement
#[derive(Debug)]
pub struct AdaptiveLearningEngine {
    /// Oracle consulted by the active-learning path, when one is installed.
    ///
    /// ## Changed in 0.2.1
    ///
    /// There was no oracle: `simulate_oracle_query` manufactured an
    /// [`OracleResponse`] whose "expert label" was the throughput already
    /// measured in the data point and whose confidence was `0.9` above 100
    /// units of throughput and `0.7` below it. Every learner was then updated
    /// against that manufactured label, the query was counted in the
    /// active-learning statistics, and the resulting [`LearningUpdate`] was
    /// reported as `LearningUpdateType::ActiveLearning`. The engine now holds
    /// a real oracle or none; with none installed the active-learning path is
    /// not entered at all.
    oracle: Arc<RwLock<Option<Arc<dyn PerformanceOracle>>>>,
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
    ///
    /// No oracle is installed, so the active-learning path stays dormant until
    /// [`Self::set_oracle`] supplies one.
    pub fn new(config: AdaptiveLearningConfig) -> Self {
        Self {
            oracle: Arc::new(RwLock::new(None)),
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

    /// Install the oracle the active-learning path queries.
    ///
    /// Until one is installed, [`Self::process_data_point`] never enters that
    /// path: it neither records a query nor spends the query budget, because
    /// there is nobody to answer.
    pub fn set_oracle(&self, oracle: Arc<dyn PerformanceOracle>) {
        *self.oracle.write() = Some(oracle);
    }

    /// The installed oracle, if any.
    pub fn oracle(&self) -> Option<Arc<dyn PerformanceOracle>> {
        self.oracle.read().clone()
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
            self.metrics_tracker.lock().record_drift_detection();
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

        // Adapt all registered learners, collecting the gradient norms they
        // report so the published metrics describe this round's adaptation
        // rather than a constant.
        let mut gradient_norms = Vec::new();
        let mut improving_learners = 0usize;
        {
            let mut learners = self.online_learners.write();
            for (name, learner) in learners.iter_mut() {
                match learner.adapt_to_drift(&recent_data) {
                    Ok(impact) => {
                        successful_adaptations += 1;
                        total_performance_impact += impact;
                        if impact > 0.0 {
                            improving_learners += 1;
                        }
                        if let Some(norm) = learner.last_gradient_norm() {
                            gradient_norms.push(norm);
                        }
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
            gradient_norm: joint_gradient_norm(&gradient_norms),
            loss_reduction: total_performance_impact / successful_adaptations.max(1) as f32,
            improving_learner_fraction: improving_fraction(
                improving_learners,
                successful_adaptations as usize,
            ),
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

        // Check if active learning is needed.
        //
        // The oracle is checked *before* the strategy is consulted:
        // `should_query_oracle` records the query and spends the hourly budget
        // as a side effect, so asking it with no oracle installed would leave a
        // log of queries that were never made.
        if let Some(oracle) = self.oracle() {
            let should_query = {
                let mut active_learning = self.active_learning.lock();
                active_learning.should_query_oracle(data_point)?
            };
            if should_query {
                self.metrics_tracker.lock().record_active_learning_query();
                return self.perform_active_learning_update(data_point, oracle).await;
            }
        }

        // Perform incremental updates
        let mut gradient_norms = Vec::new();
        let mut improving_learners = 0usize;
        {
            let mut learners = self.online_learners.write();
            for (name, learner) in learners.iter_mut() {
                match learner.incremental_update(data_point) {
                    Ok(impact) => {
                        total_impact += impact;
                        update_count += 1;
                        if impact > 0.0 {
                            improving_learners += 1;
                        }
                        if let Some(norm) = learner.last_gradient_norm() {
                            gradient_norms.push(norm);
                        }
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
            gradient_norm: joint_gradient_norm(&gradient_norms),
            loss_reduction: total_impact / update_count as f32,
            improving_learner_fraction: improving_fraction(improving_learners, update_count),
            training_time: start_time.elapsed(),
            memory_usage_mb: self.estimate_memory_usage(),
        };

        Ok(Some(LearningUpdate {
            update_type: LearningUpdateType::Incremental,
            performance_impact: total_impact / update_count as f32,
            learning_metrics,
            updated_at: Utc::now(),
        }))
    }

    /// Perform an active learning update against a real oracle.
    ///
    /// The label comes from `oracle`; a caller reaches this method only after
    /// [`Self::oracle`] returned one. An oracle that declines to label the
    /// point returns `Ok(None)` from [`PerformanceOracle::label`], and this
    /// method then reports no update rather than inventing one -- as does an
    /// oracle that fails, whose error is propagated with the point that caused
    /// it.
    async fn perform_active_learning_update(
        &self,
        data_point: &PerformanceDataPoint,
        oracle: Arc<dyn PerformanceOracle>,
    ) -> Result<Option<LearningUpdate>> {
        let start_time = std::time::Instant::now();

        let oracle_response = oracle.label(data_point).with_context(|| {
            format!(
                "oracle '{}' failed to label the performance point recorded at {}",
                oracle.name(),
                data_point.timestamp
            )
        })?;

        // Record what the oracle said (or that it declined) before deciding
        // whether an update happened, so the controller's view of its queries
        // stays accurate either way.
        {
            let mut active_learning = self.active_learning.lock();
            active_learning.update_query_strategy(data_point, &oracle_response)?;
        }

        let Some(oracle_data) = oracle_response else {
            tracing::info!(
                "oracle '{}' declined to label a queried point; no active-learning update",
                oracle.name()
            );
            return Ok(None);
        };

        let mut total_impact = 0.0f32;
        let mut update_count = 0usize;
        let mut gradient_norms = Vec::new();
        let mut improving_learners = 0usize;

        // Update learners with oracle-labeled data
        {
            let mut learners = self.online_learners.write();
            for (name, learner) in learners.iter_mut() {
                match learner.active_learning_update(&oracle_data) {
                    Ok(impact) => {
                        total_impact += impact;
                        update_count += 1;
                        if impact > 0.0 {
                            improving_learners += 1;
                        }
                        if let Some(norm) = learner.last_gradient_norm() {
                            gradient_norms.push(norm);
                        }
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

        if update_count == 0 {
            return Ok(None);
        }

        let learning_metrics = LearningMetrics {
            learning_rate: self.config.read().learning_rate_decay * 1.5, // Higher learning rate for active learning
            gradient_norm: joint_gradient_norm(&gradient_norms),
            loss_reduction: total_impact / update_count as f32,
            improving_learner_fraction: improving_fraction(improving_learners, update_count),
            training_time: start_time.elapsed(),
            memory_usage_mb: self.estimate_memory_usage(),
        };

        Ok(Some(LearningUpdate {
            update_type: LearningUpdateType::ActiveLearning,
            performance_impact: total_impact / update_count as f32,
            learning_metrics,
            updated_at: Utc::now(),
        }))
    }

    /// Storage held by the buffered performance points and the registered
    /// learners, in MiB.
    ///
    /// ## Changed in 0.2.1
    ///
    /// Every learner was charged a flat `1024` bytes ("Simplified estimate")
    /// regardless of how many weights it held. Learners report their own
    /// storage now; a learner that cannot makes the whole figure `None`,
    /// because a total missing one learner's storage would understate the real
    /// one without saying so. The buffer term counts the buffered points' own
    /// bytes, not heap allocations reachable from them.
    fn estimate_memory_usage(&self) -> Option<f32> {
        let buffer_bytes = {
            let buffer = self.performance_buffer.lock();
            buffer.len() * std::mem::size_of::<PerformanceDataPoint>()
        };

        let mut learner_bytes = 0usize;
        for learner in self.online_learners.read().values() {
            learner_bytes = learner_bytes.checked_add(learner.footprint_bytes()?)?;
        }

        Some((buffer_bytes.checked_add(learner_bytes)? as f32) / 1024.0 / 1024.0)
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

/// One drift test: the two-sample Kolmogorov-Smirnov statistic between the
/// reference and detection windows, its p-value, and the decision it produced.
///
/// ## Changed in 0.2.1
///
/// `p_value` was the constant `0.05` ("Simplified") and `test_statistic` was
/// the absolute difference of the two window means -- a *different* statistic
/// from the Kolmogorov-Smirnov one that actually drove `drift_detected`, so the
/// record described a test that was never run. Both fields now come from the
/// test whose result is reported.
#[derive(Debug, Clone)]
pub struct DriftTestResult {
    /// Kolmogorov-Smirnov statistic `D` between the two windows
    pub test_statistic: f64,
    /// Asymptotic two-sample p-value of `test_statistic`
    pub p_value: f64,
    /// Whether `test_statistic` exceeded the configured drift threshold
    pub drift_detected: bool,
    /// When the test ran
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
        let statistic = self.kolmogorov_smirnov_test()?;
        let drift_detected = statistic > self.threshold as f64;

        // Record the statistic that made the decision, together with its own
        // p-value.
        let test_result = DriftTestResult {
            test_statistic: statistic,
            p_value: self.two_sample_ks_p_value(statistic),
            drift_detected,
            tested_at: Utc::now(),
        };

        self.test_history.push_back(test_result);
        if self.test_history.len() > 20 {
            self.test_history.pop_front();
        }

        Ok(drift_detected)
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

    /// Asymptotic p-value of a two-sample Kolmogorov-Smirnov statistic.
    ///
    /// The two-sample test uses the same Kolmogorov limiting distribution as
    /// the one-sample test with the effective sample size
    /// `n_e = n1·n2 / (n1 + n2)`, so this defers to the shared
    /// [`ks_p_value`] implementation. `n_e` is rounded to the nearest whole
    /// sample, which moves the p-value by far less than the asymptotic
    /// approximation itself at the window sizes this detector uses (25 and up).
    ///
    /// Returns `1.0` -- no evidence of drift -- when either window is empty.
    fn two_sample_ks_p_value(&self, statistic: f64) -> f64 {
        let reference = self.reference_window.len() as f64;
        let detection = self.detection_window.len() as f64;
        if reference <= 0.0 || detection <= 0.0 {
            return 1.0;
        }
        let effective = reference * detection / (reference + detection);
        ks_p_value(statistic, effective.round().max(1.0) as usize)
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

    /// Euclidean norm of the gradient of this learner's most recent update.
    ///
    /// `None` -- the default -- for learners that do not compute a gradient, so
    /// that [`LearningMetrics::gradient_norm`] can say "not measured" instead
    /// of publishing a number nobody produced.
    ///
    /// [`LearningMetrics::gradient_norm`]:
    ///     crate::performance_optimizer::performance_modeling::types::LearningMetrics::gradient_norm
    fn last_gradient_norm(&self) -> Option<f64> {
        None
    }

    /// Bytes of storage this learner occupies, including its own struct.
    ///
    /// `None` -- the default -- for learners that cannot account for their
    /// storage; one such learner makes the engine's whole memory figure `None`
    /// rather than a total that silently omits it.
    fn footprint_bytes(&self) -> Option<usize> {
        None
    }

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
    /// Norm of the gradient this learner last applied.
    ///
    /// `Self::update_weights` records it on every update; before 0.2.1 it was
    /// recorded and never read, while the engine published `0.0`, `0.5` or
    /// `0.8` depending on which of its three code paths built the metrics.
    /// `None` before the first update, when there is no gradient yet.
    fn last_gradient_norm(&self) -> Option<f64> {
        (self.stats.total_updates > 0).then_some(self.stats.last_gradient_norm)
    }

    /// Own struct plus both weight vectors plus the learner's name.
    fn footprint_bytes(&self) -> Option<usize> {
        let elements = self.weights.len() + self.previous_gradients.len();
        Some(
            std::mem::size_of::<Self>()
                + elements * std::mem::size_of::<f64>()
                + self.name.capacity(),
        )
    }

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

    /// Calculate uncertainty score for data point.
    ///
    /// Uses load per available core: a machine running at or beyond one
    /// runnable task per core is contended, and a measurement taken under
    /// contention says less about the parallelism level than the same
    /// measurement taken on an idle machine.
    ///
    /// This used to average `load_average` with `io_wait_percent`. That second
    /// term was removed with the field — see
    /// [`crate::performance_optimizer::types::SystemState`] — because nothing
    /// ever measured it.
    fn calculate_uncertainty(&self, data_point: &PerformanceDataPoint) -> Result<f32> {
        let system = &data_point.system_state;
        let load_per_core = system.load_average / (system.available_cores as f32).max(1.0);
        Ok(load_per_core.clamp(0.0, 1.0))
    }

    /// Expected information gain, in nats, from having `data_point` labelled.
    ///
    /// The model is the Bayesian linear model over the three features
    /// [`active_learning_features`] extracts, with a unit-precision Gaussian
    /// prior on the weights and unit observation noise. Adding an observation
    /// `x` to a design of precision `G` multiplies the posterior precision
    /// determinant by `1 + xᵀG⁻¹x`, so the expected gain is
    /// `0.5·ln(1 + xᵀG⁻¹x)`; `G` is the identity plus the outer products of the
    /// points already queried, which is positive definite for any history,
    /// including an empty one.
    ///
    /// The figure therefore falls as queried points fill the feature space and
    /// rises for a point unlike anything queried so far -- which is what
    /// [`Self::should_query_oracle`] reads it for.
    ///
    /// ## Changed in 0.2.1
    ///
    /// This returned `Ok(0.15)` ("Placeholder"), a constant above the `0.1`
    /// threshold it is compared against, so the `InformationGain` strategy
    /// queried the oracle for every point it was ever given.
    ///
    /// # Errors
    ///
    /// Returns an error when the accumulated design is singular to working
    /// precision, rather than substituting a value that would decide the
    /// comparison one way or the other.
    fn calculate_information_gain(&self, data_point: &PerformanceDataPoint) -> Result<f32> {
        let candidate = active_learning_features(data_point);

        // G = I + Σ xᵢxᵢᵀ over the points already queried.
        let mut precision = [[0.0f64; ACTIVE_LEARNING_FEATURES]; ACTIVE_LEARNING_FEATURES];
        for (index, row) in precision.iter_mut().enumerate() {
            row[index] = 1.0;
        }
        for query in &self.query_history {
            let features = active_learning_features(&query.data_point);
            for (i, row) in precision.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell += features[i] * features[j];
                }
            }
        }

        let solved = solve_positive_definite(&precision, &candidate).ok_or_else(|| {
            anyhow::anyhow!(
                "the active-learning design over {} queried points is singular to working                  precision; the information gain of this point is undefined",
                self.query_history.len()
            )
        })?;
        let quadratic: f64 = candidate.iter().zip(solved.iter()).map(|(x, y)| x * y).sum();
        if !quadratic.is_finite() || quadratic < 0.0 {
            return Err(anyhow::anyhow!(
                "the information gain of this point evaluated to {quadratic}, which no                  positive-definite design can produce"
            ));
        }

        Ok((0.5 * (1.0 + quadratic).ln()) as f32)
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
        let features1 = active_learning_features(point1);
        let features2 = active_learning_features(point2);

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

    /// Record that the drift detector reported drift.
    ///
    /// 0.2.1: `drift_detections` was initialised to zero and never touched, so
    /// [`LearningStatistics::drift_detections`] reported no drift however many
    /// times the detector fired.
    pub fn record_drift_detection(&mut self) {
        self.drift_detections += 1;
    }

    /// Record that the oracle was queried.
    ///
    /// 0.2.1: `active_learning_queries` was likewise never incremented.
    pub fn record_active_learning_query(&mut self) {
        self.active_learning_queries += 1;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance_optimizer::types::{
        ResourceIntensity, SystemState, TestCharacteristics,
    };

    /// A performance point whose three active-learning features are set
    /// explicitly; everything else takes its default.
    fn data_point(
        parallelism: usize,
        load: f32,
        cpu_intensity: f32,
        throughput: f64,
    ) -> PerformanceDataPoint {
        PerformanceDataPoint {
            parallelism,
            throughput,
            latency: Duration::from_millis(10),
            cpu_utilization: 0.5,
            memory_utilization: 0.5,
            resource_efficiency: 0.5,
            timestamp: Utc::now(),
            test_characteristics: TestCharacteristics {
                resource_intensity: ResourceIntensity {
                    cpu_intensity,
                    ..ResourceIntensity::default()
                },
                ..TestCharacteristics::default()
            },
            system_state: SystemState {
                load_average: load,
                available_cores: 4,
                ..SystemState::default()
            },
        }
    }

    /// An oracle that labels every point with a fixed throughput and counts its
    /// calls, so a test can prove the engine reached it.
    #[derive(Debug)]
    struct CountingOracle {
        label: f64,
        calls: Arc<Mutex<usize>>,
    }

    impl PerformanceOracle for CountingOracle {
        fn label(&self, data_point: &PerformanceDataPoint) -> Result<Option<OracleResponse>> {
            *self.calls.lock() += 1;
            Ok(Some(OracleResponse {
                data_point: data_point.clone(),
                expert_label: self.label,
                confidence: 1.0,
                explanation: "test oracle".to_string(),
                queried_at: Utc::now(),
            }))
        }

        fn name(&self) -> &str {
            "counting-test-oracle"
        }
    }

    fn engine_with_learner() -> AdaptiveLearningEngine {
        let config = AdaptiveLearningConfig {
            // Adapt on every point, so a test does not have to wait out a
            // schedule.
            update_frequency: Duration::from_nanos(1),
            min_adaptation_samples: 1,
            ..AdaptiveLearningConfig::default()
        };
        let engine = AdaptiveLearningEngine::new(config);
        engine.register_learner(
            "ogd".to_string(),
            Box::new(OnlineGradientDescentLearner::new(
                "ogd".to_string(),
                5,
                0.01,
            )),
        );
        engine
    }

    /// Regression: the learner recorded its gradient norm on every update and
    /// nothing read it, while the engine published `0.0` on the drift path,
    /// `0.5` on the incremental path and `0.8` on the active-learning path.
    #[test]
    fn gradient_norm_comes_from_the_learner() {
        let mut learner = OnlineGradientDescentLearner::new("ogd".to_string(), 5, 0.01);
        assert_eq!(
            learner.last_gradient_norm(),
            None,
            "there is no gradient before the first update"
        );

        learner
            .incremental_update(&data_point(4, 1.0, 0.5, 100.0))
            .expect("update succeeds");
        let small = learner.last_gradient_norm().expect("a gradient was applied");

        let mut louder = OnlineGradientDescentLearner::new("ogd".to_string(), 5, 0.01);
        louder
            .incremental_update(&data_point(64, 1.0, 0.5, 10_000.0))
            .expect("update succeeds");
        let large = louder.last_gradient_norm().expect("a gradient was applied");

        assert!(
            large > small,
            "a larger error over larger features must give a larger gradient: \
             {large} vs {small}"
        );
    }

    /// The published norm is the norm of the learners' gradients stacked, and
    /// it is absent rather than zero when nothing reported one.
    #[test]
    fn joint_gradient_norm_stacks_the_reported_gradients() {
        assert_eq!(joint_gradient_norm(&[]), None);
        let stacked = joint_gradient_norm(&[3.0, 4.0]).expect("two learners reported");
        assert!(
            (stacked - 5.0).abs() < 1e-6,
            "sqrt(3² + 4²) = 5, got {stacked}"
        );
    }

    /// Regression: `convergence_score` was `0.8`/`0.9`/`0.85` per code path.
    #[test]
    fn improving_fraction_counts_the_learners_that_improved() {
        assert_eq!(
            improving_fraction(0, 0),
            None,
            "zero of zero is not a share"
        );
        assert_eq!(improving_fraction(1, 4), Some(0.25));
        assert_eq!(improving_fraction(3, 3), Some(1.0));
    }

    /// Regression: every learner was charged a flat kilobyte.
    #[test]
    fn footprint_scales_with_the_weight_vector() {
        let small = OnlineGradientDescentLearner::new("ogd".to_string(), 4, 0.01);
        let large = OnlineGradientDescentLearner::new("ogd".to_string(), 400, 0.01);
        let small_bytes = small.footprint_bytes().expect("reported");
        let large_bytes = large.footprint_bytes().expect("reported");
        assert!(
            large_bytes > small_bytes + 6000,
            "396 extra weights and gradients are 6336 bytes: {large_bytes} vs {small_bytes}"
        );
    }

    /// A published learning update carries measured metrics: a real gradient
    /// norm, a real improving-learner share and a real memory figure.
    #[tokio::test]
    async fn published_metrics_are_measured() {
        let engine = engine_with_learner();
        let update = engine
            .process_data_point(&data_point(4, 1.0, 0.5, 120.0))
            .await
            .expect("processing succeeds")
            .expect("the first point adapts");

        let norm = update
            .learning_metrics
            .gradient_norm
            .expect("the gradient-descent learner reports a norm");
        assert!(
            norm > 0.0 && norm.is_finite(),
            "the norm must be a real measurement, got {norm}"
        );
        assert!(
            update.learning_metrics.improving_learner_fraction.is_some(),
            "one learner was updated, so the share is defined"
        );
        let memory = update
            .learning_metrics
            .memory_usage_mb
            .expect("the only learner reports its footprint");
        assert!(memory > 0.0, "the buffer and learner occupy something");
    }

    /// Regression: with no oracle in existence the engine manufactured an
    /// "expert label" from the measurement it already had and reported an
    /// `ActiveLearning` update. With no oracle installed the path is not
    /// entered at all, and no query is recorded.
    #[tokio::test]
    async fn without_an_oracle_no_query_is_made() {
        let engine = engine_with_learner();
        // A heavily loaded host: uncertainty is above the sampling threshold,
        // so the strategy would query if it could.
        let update = engine
            .process_data_point(&data_point(4, 8.0, 0.5, 120.0))
            .await
            .expect("processing succeeds")
            .expect("an update is produced");

        assert!(
            matches!(update.update_type, LearningUpdateType::Incremental),
            "without an oracle the update is an ordinary incremental one, got {:?}",
            update.update_type
        );
        assert_eq!(
            engine.get_learning_statistics().active_learning_queries,
            0,
            "no query may be recorded when no oracle exists"
        );
    }

    /// With an oracle installed the query really reaches it, and the update is
    /// reported as active learning.
    #[tokio::test]
    async fn an_installed_oracle_is_queried() {
        let engine = engine_with_learner();
        let calls = Arc::new(Mutex::new(0usize));
        engine.set_oracle(Arc::new(CountingOracle {
            label: 150.0,
            calls: calls.clone(),
        }));

        let update = engine
            .process_data_point(&data_point(4, 8.0, 0.5, 120.0))
            .await
            .expect("processing succeeds")
            .expect("an update is produced");

        assert!(
            matches!(update.update_type, LearningUpdateType::ActiveLearning),
            "the oracle answered, so this is an active-learning update, got {:?}",
            update.update_type
        );
        assert_eq!(*calls.lock(), 1, "the oracle was consulted exactly once");
        assert_eq!(
            engine.get_learning_statistics().active_learning_queries,
            1,
            "the query is counted"
        );
    }

    /// Regression: `DriftTestResult` reported `p_value: 0.05` for every test and
    /// a `test_statistic` computed by a different test than the one that
    /// decided `drift_detected`.
    #[test]
    fn drift_test_reports_its_own_statistic_and_p_value() {
        let mut detector = ConceptDriftDetector::new(0.5);
        // Fill both windows from one distribution: no drift.
        for index in 0..120 {
            let value = 100.0 + (index % 7) as f64;
            detector
                .detect_drift(&data_point(4, 1.0, 0.5, value))
                .expect("detection succeeds");
        }
        let quiet = detector.get_detection_history().last().cloned().expect("a test was recorded");
        assert!(
            quiet.p_value > 0.05,
            "a stationary series must not look like drift, got p = {}",
            quiet.p_value
        );

        // Now shift the distribution far away.
        let mut shifted = ConceptDriftDetector::new(0.5);
        for index in 0..60 {
            shifted
                .detect_drift(&data_point(4, 1.0, 0.5, 100.0 + (index % 7) as f64))
                .expect("detection succeeds");
        }
        for index in 0..60 {
            shifted
                .detect_drift(&data_point(4, 1.0, 0.5, 900.0 + (index % 7) as f64))
                .expect("detection succeeds");
        }
        let drifted = shifted.get_detection_history().last().cloned().expect("a test was recorded");
        assert!(
            drifted.p_value < quiet.p_value,
            "a shifted series must be less likely under the null: {} vs {}",
            drifted.p_value,
            quiet.p_value
        );
        assert!(
            drifted.test_statistic > quiet.test_statistic,
            "the recorded statistic must be the one that saw the shift: {} vs {}",
            drifted.test_statistic,
            quiet.test_statistic
        );
        assert!(
            (0.0..=1.0).contains(&drifted.test_statistic),
            "a Kolmogorov-Smirnov statistic is a probability difference, got {}",
            drifted.test_statistic
        );
    }

    /// Regression: `calculate_information_gain` returned `Ok(0.15)`, a constant
    /// above the `0.1` threshold it is compared against, so the
    /// `InformationGain` strategy queried for every point.
    #[test]
    fn information_gain_falls_as_the_design_fills() {
        let mut controller = ActiveLearningController::new(0.3);
        let candidate = data_point(8, 2.0, 0.5, 100.0);

        let fresh = controller
            .calculate_information_gain(&candidate)
            .expect("an empty design is still invertible");

        for _ in 0..20 {
            controller.query_history.push_back(ActiveLearningQuery {
                data_point: candidate.clone(),
                uncertainty_score: 0.5,
                response_received: true,
            });
        }
        let saturated = controller.calculate_information_gain(&candidate).expect("gain is defined");
        assert!(
            saturated < fresh,
            "a point already queried twenty times must inform less: {saturated} vs {fresh}"
        );

        let novel = controller
            .calculate_information_gain(&data_point(64, 9.0, 0.9, 100.0))
            .expect("gain is defined");
        assert!(
            novel > saturated,
            "a point unlike the history must inform more: {novel} vs {saturated}"
        );
    }

    /// Regression: `drift_detections` and `active_learning_queries` were
    /// initialised to zero and never incremented, so
    /// [`LearningStatistics`] reported that no drift had ever been detected and
    /// no oracle query ever made, whatever the engine had done.
    #[test]
    fn the_tracker_counts_drift_detections() {
        let mut tracker = LearningMetricsTracker::new();
        assert_eq!(tracker.drift_detections, 0);
        tracker.record_drift_detection();
        tracker.record_drift_detection();
        assert_eq!(
            tracker.drift_detections, 2,
            "each detection must be counted"
        );
        assert_eq!(
            tracker.active_learning_queries, 0,
            "a drift detection is not an oracle query"
        );
        tracker.record_active_learning_query();
        assert_eq!(tracker.active_learning_queries, 1);
    }

    /// A drift-detecting run reaches the published statistics.
    #[tokio::test]
    async fn detected_drift_is_reported_in_the_statistics() {
        let engine = engine_with_learner();
        // Fill the detector's windows from one distribution, then shift far
        // away: the second half must be recognised as drift.
        for index in 0..60 {
            engine
                .process_data_point(&data_point(4, 1.0, 0.5, 100.0 + (index % 5) as f64))
                .await
                .expect("processing succeeds");
        }
        let before = engine.get_learning_statistics().drift_detections;
        for index in 0..60 {
            engine
                .process_data_point(&data_point(4, 1.0, 0.5, 5_000.0 + (index % 5) as f64))
                .await
                .expect("processing succeeds");
        }
        let after = engine.get_learning_statistics().drift_detections;
        assert!(
            after > before,
            "a distribution shift must show up in the drift count: {after} vs {before}"
        );
    }
}
