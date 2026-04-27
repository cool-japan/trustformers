//! Canary deployment controller for TrustformeRS Serve.
//!
//! Provides gradual traffic shifting between a baseline model and a canary
//! model, with automated rollback on error rate or latency degradation and
//! optional automatic promotion when promotion criteria are met.

mod canary_extra_tests;

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from the canary deployment controller.
#[derive(Debug, Clone, Error)]
pub enum CanaryError {
    #[error("Canary is not currently active (phase: {phase})")]
    NotActive { phase: String },

    #[error("Cannot promote: traffic would exceed max ({max:.1}%)")]
    ExceedsMaxTraffic { max: f32 },

    #[error("Cannot promote: promotion criteria not met")]
    CriteriaNotMet,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Rollback threshold
// ─────────────────────────────────────────────────────────────────────────────

/// Thresholds that trigger an automatic rollback when exceeded.
#[derive(Debug, Clone)]
pub struct RollbackThreshold {
    /// Maximum canary error rate before rollback (default 0.05 = 5%).
    pub max_error_rate: f32,
    /// Maximum ratio of canary p99 latency to baseline latency (default 1.5).
    pub max_latency_p99_ratio: f32,
    /// Minimum required success rate (default 0.95 = 95%).
    pub min_success_rate: f32,
}

impl Default for RollbackThreshold {
    fn default() -> Self {
        Self {
            max_error_rate: 0.05,
            max_latency_p99_ratio: 1.5,
            min_success_rate: 0.95,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Promotion criteria
// ─────────────────────────────────────────────────────────────────────────────

/// Criteria that must all be satisfied before the canary can be promoted.
#[derive(Debug, Clone)]
pub struct PromotionCriteria {
    /// Minimum number of requests through the canary before promotion is allowed.
    pub min_requests: u64,
    /// Maximum acceptable canary error rate for promotion (default 0.01 = 1%).
    pub max_error_rate: f32,
    /// Maximum ratio of canary mean latency to baseline mean latency for
    /// promotion (default 1.1 = within 10% of baseline).
    pub max_latency_ratio: f32,
    /// Minimum wall-clock seconds the canary must have been running.
    pub min_evaluation_seconds: u64,
}

impl Default for PromotionCriteria {
    fn default() -> Self {
        Self {
            min_requests: 1000,
            max_error_rate: 0.01,
            max_latency_ratio: 1.1,
            min_evaluation_seconds: 300,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Canary phase
// ─────────────────────────────────────────────────────────────────────────────

/// Current lifecycle phase of a canary deployment.
#[derive(Debug, Clone)]
pub enum CanaryPhase {
    /// No canary is active.
    Idle,
    /// Canary is routing a percentage of traffic.
    Running {
        traffic_percent: f32,
        started_at: std::time::Instant,
    },
    /// Traffic is being stepped up from `from_percent` to `to_percent`.
    Promoting { from_percent: f32, to_percent: f32 },
    /// Canary was rolled back due to the given reason.
    RolledBack { reason: String },
    /// Canary has been fully promoted to 100%.
    FullyPromoted,
}

impl CanaryPhase {
    fn display_name(&self) -> &'static str {
        match self {
            CanaryPhase::Idle => "Idle",
            CanaryPhase::Running { .. } => "Running",
            CanaryPhase::Promoting { .. } => "Promoting",
            CanaryPhase::RolledBack { .. } => "RolledBack",
            CanaryPhase::FullyPromoted => "FullyPromoted",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Canary metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulated request metrics for both the canary and baseline models.
#[derive(Debug, Clone, Default)]
pub struct CanaryMetrics {
    pub canary_requests: u64,
    pub baseline_requests: u64,
    pub canary_errors: u64,
    pub baseline_errors: u64,
    pub canary_total_latency_ms: f64,
    pub baseline_total_latency_ms: f64,
}

impl CanaryMetrics {
    /// Fraction of canary requests that resulted in an error (0.0 if no requests).
    pub fn canary_error_rate(&self) -> f32 {
        if self.canary_requests == 0 {
            return 0.0;
        }
        self.canary_errors as f32 / self.canary_requests as f32
    }

    /// Fraction of baseline requests that resulted in an error (0.0 if no requests).
    pub fn baseline_error_rate(&self) -> f32 {
        if self.baseline_requests == 0 {
            return 0.0;
        }
        self.baseline_errors as f32 / self.baseline_requests as f32
    }

    /// Mean canary request latency in milliseconds (0.0 if no requests).
    pub fn canary_mean_latency_ms(&self) -> f64 {
        if self.canary_requests == 0 {
            return 0.0;
        }
        self.canary_total_latency_ms / self.canary_requests as f64
    }

    /// Mean baseline request latency in milliseconds (0.0 if no requests).
    pub fn baseline_mean_latency_ms(&self) -> f64 {
        if self.baseline_requests == 0 {
            return 0.0;
        }
        self.baseline_total_latency_ms / self.baseline_requests as f64
    }

    /// Ratio of canary mean latency to baseline mean latency.
    ///
    /// Returns 1.0 when either side has no data (neutral / unknown).
    pub fn latency_ratio(&self) -> f32 {
        let baseline = self.baseline_mean_latency_ms();
        if baseline < f64::EPSILON {
            return 1.0;
        }
        (self.canary_mean_latency_ms() / baseline) as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Canary configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a canary deployment.
#[derive(Debug, Clone)]
pub struct CanaryConfig {
    /// Model ID of the canary (new) model.
    pub canary_model_id: String,
    /// Model ID of the baseline (stable) model.
    pub baseline_model_id: String,
    /// Initial percentage of traffic routed to the canary (default 5.0).
    pub initial_traffic_percent: f32,
    /// Maximum percentage of traffic that can be routed to the canary (default 50.0).
    pub max_traffic_percent: f32,
    /// How much traffic (in percentage points) to add at each promotion step (default 5.0).
    pub step_size: f32,
    /// Window (in milliseconds) over which metrics are evaluated (default 60 000 ms).
    pub evaluation_window_ms: u64,
    /// Conditions that trigger an automatic rollback.
    pub rollback_threshold: RollbackThreshold,
    /// Conditions that must be met before promotion.
    pub promotion_criteria: PromotionCriteria,
    /// If `true`, the controller automatically promotes when criteria are met.
    pub auto_promote: bool,
    /// If `true`, the controller automatically rolls back when thresholds are exceeded.
    pub auto_rollback: bool,
}

impl CanaryConfig {
    /// Create a minimal canary config with default values.
    pub fn new(canary_model_id: impl Into<String>, baseline_model_id: impl Into<String>) -> Self {
        Self {
            canary_model_id: canary_model_id.into(),
            baseline_model_id: baseline_model_id.into(),
            initial_traffic_percent: 5.0,
            max_traffic_percent: 50.0,
            step_size: 5.0,
            evaluation_window_ms: 60_000,
            rollback_threshold: RollbackThreshold::default(),
            promotion_criteria: PromotionCriteria::default(),
            auto_promote: false,
            auto_rollback: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CanaryMetricsHistory
// ─────────────────────────────────────────────────────────────────────────────

/// A single snapshot recorded into the canary metrics history.
#[derive(Debug, Clone)]
pub struct CanaryMetricsSnapshot {
    /// Monotonically increasing step counter.
    pub step: usize,
    /// Fraction of canary requests that resulted in an error.
    pub error_rate: f32,
    /// 99th-percentile latency observed for the canary (milliseconds).
    pub p99_latency_ms: f64,
}

/// Rolling window of canary metric snapshots used for stability analysis and
/// trend detection.
#[derive(Debug, Clone, Default)]
pub struct CanaryMetricsHistory {
    pub snapshots: Vec<CanaryMetricsSnapshot>,
}

impl CanaryMetricsHistory {
    /// Append a new snapshot to the history.
    pub fn push_canary_metrics(&mut self, step: usize, error_rate: f32, p99_latency_ms: f64) {
        self.snapshots.push(CanaryMetricsSnapshot {
            step,
            error_rate,
            p99_latency_ms,
        });
    }

    /// Return the most recent `window` snapshots (or all if fewer exist).
    fn last_n(&self, window: usize) -> &[CanaryMetricsSnapshot] {
        let len = self.snapshots.len();
        if len <= window {
            &self.snapshots
        } else {
            &self.snapshots[len - window..]
        }
    }

    /// Compute the sample standard deviation of `values`.
    fn std_dev(values: &[f64]) -> f64 {
        let n = values.len();
        if n < 2 {
            return 0.0;
        }
        let mean: f64 = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        variance.sqrt()
    }

    /// Return `true` if both error rate and p99 latency are stable
    /// (low coefficient of variation) within the most recent `window` snapshots.
    ///
    /// "Stable" means the standard deviation of each metric is below 10 % of
    /// its mean (if mean > 0) or simply below 0.01 when the mean is near zero.
    pub fn is_stable(&self, window: usize) -> bool {
        let snaps = self.last_n(window);
        if snaps.len() < 2 {
            return false; // not enough data to declare stability
        }

        let error_rates: Vec<f64> = snaps.iter().map(|s| s.error_rate as f64).collect();
        let latencies: Vec<f64> = snaps.iter().map(|s| s.p99_latency_ms).collect();

        let err_std = Self::std_dev(&error_rates);
        let lat_std = Self::std_dev(&latencies);

        let err_mean: f64 = error_rates.iter().sum::<f64>() / error_rates.len() as f64;
        let lat_mean: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;

        let err_stable = if err_mean > 0.01 { err_std / err_mean < 0.1 } else { err_std < 0.005 };

        let lat_stable = if lat_mean > 1.0 { lat_std / lat_mean < 0.1 } else { lat_std < 0.5 };

        err_stable && lat_stable
    }

    /// Compute the linear regression slope of error_rate over the most recent
    /// `window` snapshots.
    ///
    /// A positive slope means error rate is trending upward.
    /// Returns 0.0 when fewer than 2 snapshots are available.
    pub fn trend_error_rate(&self, window: usize) -> f32 {
        let snaps = self.last_n(window);
        let n = snaps.len();
        if n < 2 {
            return 0.0;
        }

        // Use the sequential 0..n indices as x values (not raw step numbers)
        // so the computation is always well-conditioned.
        let x_mean: f64 = (n - 1) as f64 / 2.0;
        let y_values: Vec<f64> = snaps.iter().map(|s| s.error_rate as f64).collect();
        let y_mean: f64 = y_values.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;
        for (i, &y) in y_values.iter().enumerate() {
            let xi = i as f64 - x_mean;
            numerator += xi * (y - y_mean);
            denominator += xi * xi;
        }

        if denominator < f64::EPSILON {
            return 0.0;
        }
        (numerator / denominator) as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CanaryDecision
// ─────────────────────────────────────────────────────────────────────────────

/// High-level decision produced by the automated canary governor.
#[derive(Debug, Clone, PartialEq)]
pub enum CanaryDecision {
    /// Everything looks healthy — continue as-is.
    Continue,
    /// Canary is consistently healthy — promote to next traffic step.
    Promote,
    /// Canary shows degradation — roll back with the given reason.
    Rollback(String),
    /// Ambiguous / insufficient data — pause evaluation.
    Pause,
}

/// Derive a `CanaryDecision` from a `CanaryMetricsHistory` and a `CanaryConfig`.
///
/// Decision logic (checked in order):
/// 1. If the history has fewer than 3 snapshots → `Pause` (not enough data).
/// 2. If the error-rate trend slope > 0.01/step → `Rollback` (rising errors).
/// 3. If the latest error_rate > `rollback_threshold.max_error_rate` → `Rollback`.
/// 4. If the latest p99_latency is > 2× the earliest p99 in the window → `Rollback`.
/// 5. If `is_stable(window)` for the most recent 5 snapshots → `Promote`.
/// 6. Otherwise → `Continue`.
pub fn canary_decide(
    metrics_history: &CanaryMetricsHistory,
    config: &CanaryConfig,
) -> CanaryDecision {
    const MIN_SNAPSHOTS: usize = 3;
    let window = MIN_SNAPSHOTS.max(5);

    if metrics_history.snapshots.len() < MIN_SNAPSHOTS {
        return CanaryDecision::Pause;
    }

    // Check rising error trend.
    let slope = metrics_history.trend_error_rate(window);
    if slope > 0.01 {
        return CanaryDecision::Rollback(format!(
            "error rate trend is rising (slope = {slope:.4}/step)"
        ));
    }

    // Check latest absolute error rate.
    if let Some(latest) = metrics_history.snapshots.last() {
        if latest.error_rate > config.rollback_threshold.max_error_rate {
            return CanaryDecision::Rollback(format!(
                "current error rate {:.2}% exceeds threshold {:.2}%",
                latest.error_rate * 100.0,
                config.rollback_threshold.max_error_rate * 100.0,
            ));
        }

        // Check latency spike vs earliest snapshot in the window.
        let first = &metrics_history.snapshots[0];
        if first.p99_latency_ms > 0.0 && latest.p99_latency_ms > first.p99_latency_ms * 2.0 {
            return CanaryDecision::Rollback(format!(
                "p99 latency {:.1}ms is more than 2× baseline {:.1}ms",
                latest.p99_latency_ms, first.p99_latency_ms
            ));
        }
    }

    // Check stability for promotion.
    if metrics_history.is_stable(window) {
        return CanaryDecision::Promote;
    }

    CanaryDecision::Continue
}

// ─────────────────────────────────────────────────────────────────────────────
// CanaryPhaseTransition
// ─────────────────────────────────────────────────────────────────────────────

/// Records a lifecycle transition between two canary phases.
#[derive(Debug)]
pub struct CanaryPhaseTransition {
    pub from: String,
    pub to: String,
    pub reason: String,
    pub timestamp: std::time::Instant,
}

/// Create a `CanaryPhaseTransition` logging a phase change.
pub fn log_phase_transition(
    from: &CanaryPhase,
    to: &CanaryPhase,
    reason: &str,
) -> CanaryPhaseTransition {
    CanaryPhaseTransition {
        from: from.display_name().to_string(),
        to: to.display_name().to_string(),
        reason: reason.to_string(),
        timestamp: std::time::Instant::now(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Canary controller
// ─────────────────────────────────────────────────────────────────────────────

/// Controls the lifecycle of a canary deployment.
pub struct CanaryController {
    pub config: CanaryConfig,
    pub phase: CanaryPhase,
    pub metrics: CanaryMetrics,
}

impl CanaryController {
    /// Create a new controller in the `Running` phase with the configured
    /// initial traffic percentage.
    pub fn new(config: CanaryConfig) -> Self {
        let initial = config.initial_traffic_percent;
        Self {
            config,
            phase: CanaryPhase::Running {
                traffic_percent: initial,
                started_at: std::time::Instant::now(),
            },
            metrics: CanaryMetrics::default(),
        }
    }

    /// Decide whether `request_hash` should be routed to the canary.
    ///
    /// Uses the current traffic percentage: a request is routed to the canary
    /// when `(request_hash % 10_000) < (traffic_percent * 100.0)`.
    ///
    /// Always returns `false` when the canary is not active.
    pub fn route_request(&self, request_hash: u64) -> bool {
        let traffic_percent = match &self.phase {
            CanaryPhase::Running {
                traffic_percent, ..
            } => *traffic_percent,
            CanaryPhase::Promoting { to_percent, .. } => *to_percent,
            CanaryPhase::FullyPromoted => 100.0,
            _ => return false,
        };
        let threshold = (traffic_percent * 100.0) as u64;
        (request_hash % 10_000) < threshold
    }

    /// Record a request handled by the canary model.
    pub fn record_canary_request(&mut self, latency_ms: f64, is_error: bool) {
        self.metrics.canary_requests += 1;
        self.metrics.canary_total_latency_ms += latency_ms;
        if is_error {
            self.metrics.canary_errors += 1;
        }
    }

    /// Record a request handled by the baseline model.
    pub fn record_baseline_request(&mut self, latency_ms: f64, is_error: bool) {
        self.metrics.baseline_requests += 1;
        self.metrics.baseline_total_latency_ms += latency_ms;
        if is_error {
            self.metrics.baseline_errors += 1;
        }
    }

    /// Check whether the canary should be rolled back.
    ///
    /// Returns `Some(reason)` if any threshold is exceeded, `None` otherwise.
    pub fn check_rollback(&self) -> Option<String> {
        let threshold = &self.config.rollback_threshold;
        let m = &self.metrics;

        if m.canary_requests == 0 {
            return None;
        }

        let error_rate = m.canary_error_rate();
        if error_rate > threshold.max_error_rate {
            return Some(format!(
                "Canary error rate {:.2}% exceeds threshold {:.2}%",
                error_rate * 100.0,
                threshold.max_error_rate * 100.0,
            ));
        }

        let success_rate = 1.0 - error_rate;
        if success_rate < threshold.min_success_rate {
            return Some(format!(
                "Canary success rate {:.2}% is below minimum {:.2}%",
                success_rate * 100.0,
                threshold.min_success_rate * 100.0,
            ));
        }

        let latency_ratio = m.latency_ratio();
        if latency_ratio > threshold.max_latency_p99_ratio {
            return Some(format!(
                "Canary latency ratio {:.2} exceeds threshold {:.2}",
                latency_ratio, threshold.max_latency_p99_ratio,
            ));
        }

        None
    }

    /// Check whether the canary meets all promotion criteria.
    pub fn check_promote(&self) -> bool {
        let criteria = &self.config.promotion_criteria;
        let m = &self.metrics;

        if m.canary_requests < criteria.min_requests {
            return false;
        }

        if m.canary_error_rate() > criteria.max_error_rate {
            return false;
        }

        if m.latency_ratio() > criteria.max_latency_ratio {
            return false;
        }

        // Check elapsed time if the canary is in the Running phase.
        if let CanaryPhase::Running { started_at, .. } = &self.phase {
            let elapsed_secs = started_at.elapsed().as_secs();
            if elapsed_secs < criteria.min_evaluation_seconds {
                return false;
            }
        }

        true
    }

    /// Promote the canary by one `step_size` percentage points.
    ///
    /// Returns the new traffic percentage, or an error if the canary is not
    /// active or would exceed `max_traffic_percent`.
    pub fn promote(&mut self) -> Result<f32, CanaryError> {
        let current = match &self.phase {
            CanaryPhase::Running {
                traffic_percent, ..
            } => *traffic_percent,
            CanaryPhase::Promoting { to_percent, .. } => *to_percent,
            other => {
                return Err(CanaryError::NotActive {
                    phase: other.display_name().to_string(),
                });
            },
        };

        let new_percent = current + self.config.step_size;

        if new_percent > self.config.max_traffic_percent + f32::EPSILON {
            return Err(CanaryError::ExceedsMaxTraffic {
                max: self.config.max_traffic_percent,
            });
        }

        let clamped = new_percent.min(self.config.max_traffic_percent);

        if (clamped - 100.0).abs() < f32::EPSILON || clamped >= 100.0 {
            self.phase = CanaryPhase::FullyPromoted;
        } else {
            self.phase = CanaryPhase::Promoting {
                from_percent: current,
                to_percent: clamped,
            };
        }

        Ok(clamped)
    }

    /// Roll back the canary with the given human-readable reason.
    pub fn rollback(&mut self, reason: String) {
        self.phase = CanaryPhase::RolledBack { reason };
    }

    /// Return the current traffic percentage directed at the canary.
    pub fn current_traffic_percent(&self) -> f32 {
        match &self.phase {
            CanaryPhase::Running {
                traffic_percent, ..
            } => *traffic_percent,
            CanaryPhase::Promoting { to_percent, .. } => *to_percent,
            CanaryPhase::FullyPromoted => 100.0,
            _ => 0.0,
        }
    }

    /// Return `true` if the canary is actively routing traffic.
    pub fn is_active(&self) -> bool {
        matches!(
            &self.phase,
            CanaryPhase::Running { .. }
                | CanaryPhase::Promoting { .. }
                | CanaryPhase::FullyPromoted
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_controller() -> CanaryController {
        CanaryController::new(CanaryConfig::new("canary-v2", "baseline-v1"))
    }

    // ── Initial state ─────────────────────────────────────────────────────────

    #[test]
    fn test_initial_traffic_percent() {
        let ctrl = default_controller();
        assert!(
            (ctrl.current_traffic_percent() - 5.0).abs() < f32::EPSILON,
            "Default initial traffic should be 5%"
        );
    }

    #[test]
    fn test_is_active_initially() {
        let ctrl = default_controller();
        assert!(
            ctrl.is_active(),
            "Controller should be active after creation"
        );
    }

    // ── route_request determinism ─────────────────────────────────────────────

    #[test]
    fn test_route_request_determinism() {
        let ctrl = default_controller();
        let hash = 42_u64;
        let r1 = ctrl.route_request(hash);
        let r2 = ctrl.route_request(hash);
        assert_eq!(r1, r2, "Same hash must always route to same model");
    }

    #[test]
    fn test_route_request_at_5_percent() {
        let ctrl = default_controller(); // 5% canary
        let mut canary_count = 0u64;
        let total = 10_000u64;
        for i in 0..total {
            if ctrl.route_request(i) {
                canary_count += 1;
            }
        }
        // With hash % 10_000, threshold = 5 * 100 = 500, so exactly 5%
        assert_eq!(
            canary_count, 500,
            "Exactly 5% of sequential hashes should go to canary"
        );
    }

    // ── CanaryMetrics updates ─────────────────────────────────────────────────

    #[test]
    fn test_canary_metrics_update() {
        let mut ctrl = default_controller();
        ctrl.record_canary_request(100.0, false);
        ctrl.record_canary_request(200.0, true);
        ctrl.record_baseline_request(80.0, false);

        assert_eq!(ctrl.metrics.canary_requests, 2);
        assert_eq!(ctrl.metrics.canary_errors, 1);
        assert_eq!(ctrl.metrics.baseline_requests, 1);
        assert!((ctrl.metrics.canary_mean_latency_ms() - 150.0).abs() < 1e-9);
        assert!((ctrl.metrics.baseline_mean_latency_ms() - 80.0).abs() < 1e-9);
    }

    // ── Latency ratio ─────────────────────────────────────────────────────────

    #[test]
    fn test_latency_ratio_computation() {
        let mut ctrl = default_controller();
        ctrl.record_canary_request(200.0, false);
        ctrl.record_baseline_request(100.0, false);

        let ratio = ctrl.metrics.latency_ratio();
        assert!(
            (ratio - 2.0_f32).abs() < 1e-4,
            "ratio should be 2.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_latency_ratio_no_baseline() {
        let mut ctrl = default_controller();
        ctrl.record_canary_request(100.0, false);
        // No baseline data → neutral ratio
        assert!((ctrl.metrics.latency_ratio() - 1.0_f32).abs() < f32::EPSILON);
    }

    // ── Rollback trigger ──────────────────────────────────────────────────────

    #[test]
    fn test_rollback_triggered_by_error_rate() {
        let mut ctrl = default_controller();
        // 10 errors out of 10 requests → 100% error rate, well above 5% threshold
        for _ in 0..10 {
            ctrl.record_canary_request(50.0, true);
        }
        let reason = ctrl.check_rollback();
        assert!(reason.is_some(), "High error rate should trigger rollback");
        assert!(reason.unwrap().contains("error rate"));
    }

    #[test]
    fn test_no_rollback_below_threshold() {
        let mut ctrl = default_controller();
        // 1 error out of 100 requests → 1% error rate, below 5% threshold
        ctrl.record_canary_request(50.0, true);
        for _ in 0..99 {
            ctrl.record_canary_request(50.0, false);
        }
        assert!(
            ctrl.check_rollback().is_none(),
            "Low error rate should not trigger rollback"
        );
    }

    #[test]
    fn test_rollback_triggered_by_latency() {
        let mut ctrl = default_controller();
        // Canary 300 ms, baseline 100 ms → ratio 3.0 > threshold 1.5
        for _ in 0..50 {
            ctrl.record_canary_request(300.0, false);
            ctrl.record_baseline_request(100.0, false);
        }
        let reason = ctrl.check_rollback();
        assert!(
            reason.is_some(),
            "High latency ratio should trigger rollback"
        );
    }

    // ── Rollback phase transition ─────────────────────────────────────────────

    #[test]
    fn test_rollback_sets_phase() {
        let mut ctrl = default_controller();
        ctrl.rollback("test reason".to_string());

        assert!(
            !ctrl.is_active(),
            "Controller should not be active after rollback"
        );
        assert_eq!(ctrl.current_traffic_percent(), 0.0);
        match &ctrl.phase {
            CanaryPhase::RolledBack { reason } => assert_eq!(reason, "test reason"),
            other => panic!("Expected RolledBack phase, got {:?}", other.display_name()),
        }
    }

    // ── Promotion check ───────────────────────────────────────────────────────

    #[test]
    fn test_promotion_check_all_criteria_met() {
        let mut config = CanaryConfig::new("c-v2", "b-v1");
        config.promotion_criteria = PromotionCriteria {
            min_requests: 10,
            max_error_rate: 0.05,
            max_latency_ratio: 1.5,
            min_evaluation_seconds: 0, // disable time gate for unit test
        };
        let mut ctrl = CanaryController::new(config);

        // Simulate healthy canary traffic
        for _ in 0..10 {
            ctrl.record_canary_request(80.0, false);
            ctrl.record_baseline_request(80.0, false);
        }

        assert!(
            ctrl.check_promote(),
            "All criteria met → should be promotable"
        );
    }

    #[test]
    fn test_promotion_check_insufficient_requests() {
        let mut ctrl = default_controller();
        // Default min_requests = 1000, only 5 recorded
        for _ in 0..5 {
            ctrl.record_canary_request(50.0, false);
        }
        assert!(
            !ctrl.check_promote(),
            "Too few requests → should not be promotable"
        );
    }

    // ── promote increments traffic ────────────────────────────────────────────

    #[test]
    fn test_promote_increments_traffic() {
        let mut ctrl = default_controller(); // 5%, step=5%
        let new_pct = ctrl.promote().unwrap();
        assert!(
            (new_pct - 10.0_f32).abs() < f32::EPSILON,
            "Traffic should increase from 5% to 10%, got {}%",
            new_pct
        );
        assert!((ctrl.current_traffic_percent() - 10.0_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_promote_respects_max_traffic() {
        let mut config = CanaryConfig::new("c", "b");
        config.initial_traffic_percent = 50.0;
        config.max_traffic_percent = 50.0;
        config.step_size = 5.0;
        let mut ctrl = CanaryController::new(config);

        let result = ctrl.promote();
        assert!(
            matches!(result, Err(CanaryError::ExceedsMaxTraffic { .. })),
            "Promoting past max should return ExceedsMaxTraffic error"
        );
    }

    // ── Controller lifecycle ──────────────────────────────────────────────────

    #[test]
    fn test_controller_lifecycle() {
        let mut ctrl = default_controller();

        // Step 1: verify initial state
        assert!(ctrl.is_active());
        assert!((ctrl.current_traffic_percent() - 5.0).abs() < f32::EPSILON);

        // Step 2: promote
        ctrl.promote().unwrap();
        assert!(ctrl.is_active());
        assert!((ctrl.current_traffic_percent() - 10.0).abs() < f32::EPSILON);

        // Step 3: rollback
        ctrl.rollback("integration test rollback".to_string());
        assert!(!ctrl.is_active());
        assert_eq!(ctrl.current_traffic_percent(), 0.0);
    }

    // ── CanaryMetricsHistory ──────────────────────────────────────────────────

    // ── Test 17: push_canary_metrics appends correctly ──
    #[test]
    fn test_push_canary_metrics() {
        let mut history = CanaryMetricsHistory::default();
        history.push_canary_metrics(0, 0.01, 50.0);
        history.push_canary_metrics(1, 0.02, 52.0);
        assert_eq!(history.snapshots.len(), 2);
        assert!((history.snapshots[1].error_rate - 0.02).abs() < 1e-6);
    }

    // ── Test 18: is_stable returns false with < 2 snapshots ──
    #[test]
    fn test_is_stable_insufficient_data() {
        let mut history = CanaryMetricsHistory::default();
        assert!(!history.is_stable(5));
        history.push_canary_metrics(0, 0.01, 50.0);
        assert!(!history.is_stable(5), "single snapshot is not enough");
    }

    // ── Test 19: is_stable returns true for constant metrics ──
    #[test]
    fn test_is_stable_constant_metrics() {
        let mut history = CanaryMetricsHistory::default();
        for i in 0..10 {
            history.push_canary_metrics(i, 0.01, 100.0); // completely flat
        }
        assert!(history.is_stable(10), "flat metrics should be stable");
    }

    // ── Test 20: is_stable returns false when error rate is volatile ──
    #[test]
    fn test_is_stable_volatile_error_rate() {
        let mut history = CanaryMetricsHistory::default();
        // Alternating 0.0 and 0.5 → very high std dev relative to mean
        for i in 0..10 {
            let er = if i % 2 == 0 { 0.0_f32 } else { 0.5_f32 };
            history.push_canary_metrics(i, er, 100.0);
        }
        assert!(
            !history.is_stable(10),
            "volatile error rate should not be stable"
        );
    }

    // ── Test 21: trend_error_rate — flat history gives ~0 slope ──
    #[test]
    fn test_trend_error_rate_flat() {
        let mut history = CanaryMetricsHistory::default();
        for i in 0..10 {
            history.push_canary_metrics(i, 0.03, 100.0);
        }
        let slope = history.trend_error_rate(10);
        assert!(
            slope.abs() < 1e-6,
            "flat error rate should have ~0 slope, got {slope}"
        );
    }

    // ── Test 22: trend_error_rate — rising history gives positive slope ──
    #[test]
    fn test_trend_error_rate_rising() {
        let mut history = CanaryMetricsHistory::default();
        // Strictly increasing error rate
        for i in 0..10u32 {
            history.push_canary_metrics(i as usize, i as f32 * 0.01, 100.0);
        }
        let slope = history.trend_error_rate(10);
        assert!(
            slope > 0.0,
            "rising error rate should have positive slope, got {slope}"
        );
    }

    // ── Test 23: trend_error_rate — falling history gives negative slope ──
    #[test]
    fn test_trend_error_rate_falling() {
        let mut history = CanaryMetricsHistory::default();
        for i in 0..10u32 {
            let er = 0.1 - i as f32 * 0.009;
            history.push_canary_metrics(i as usize, er, 100.0);
        }
        let slope = history.trend_error_rate(10);
        assert!(
            slope < 0.0,
            "falling error rate should have negative slope, got {slope}"
        );
    }

    // ── Test 24: canary_decide Pause when too few snapshots ──
    #[test]
    fn test_canary_decide_pause_insufficient_data() {
        let history = CanaryMetricsHistory::default();
        let config = CanaryConfig::new("c-v2", "b-v1");
        assert_eq!(canary_decide(&history, &config), CanaryDecision::Pause);
    }

    // ── Test 25: canary_decide Rollback when error rate is too high ──
    #[test]
    fn test_canary_decide_rollback_high_error_rate() {
        let mut history = CanaryMetricsHistory::default();
        let config = CanaryConfig::new("c-v2", "b-v1"); // max_error_rate = 5%
                                                        // Push 5 snapshots with error rate > 5%
        for i in 0..5 {
            history.push_canary_metrics(i, 0.20, 100.0); // 20% error rate
        }
        assert!(
            matches!(
                canary_decide(&history, &config),
                CanaryDecision::Rollback(_)
            ),
            "high error rate should trigger rollback"
        );
    }

    // ── Test 26: canary_decide Rollback when latency 2× baseline ──
    #[test]
    fn test_canary_decide_rollback_latency_spike() {
        let mut history = CanaryMetricsHistory::default();
        let config = CanaryConfig::new("c-v2", "b-v1");
        // First 4 snapshots: healthy latency 100 ms, then a 5× spike
        for i in 0..4 {
            history.push_canary_metrics(i, 0.00, 100.0);
        }
        history.push_canary_metrics(4, 0.00, 600.0); // latency 6× baseline
        assert!(
            matches!(
                canary_decide(&history, &config),
                CanaryDecision::Rollback(_)
            ),
            "latency spike should trigger rollback"
        );
    }

    // ── Test 27: canary_decide Promote when stable ──
    #[test]
    fn test_canary_decide_promote_when_stable() {
        let mut history = CanaryMetricsHistory::default();
        let config = CanaryConfig::new("c-v2", "b-v1");
        // Push 10 perfectly flat snapshots with healthy metrics
        for i in 0..10 {
            history.push_canary_metrics(i, 0.001, 80.0); // perfectly flat
        }
        let decision = canary_decide(&history, &config);
        assert_eq!(
            decision,
            CanaryDecision::Promote,
            "stable metrics should trigger Promote"
        );
    }

    // ── Test 28: canary_decide Continue when healthy but not yet stable ──
    #[test]
    fn test_canary_decide_continue_when_healthy_not_stable() {
        let mut history = CanaryMetricsHistory::default();
        let config = CanaryConfig::new("c-v2", "b-v1");
        // 3 healthy but varied snapshots → not stable, not rollback → Continue
        history.push_canary_metrics(0, 0.01, 90.0);
        history.push_canary_metrics(1, 0.02, 110.0);
        history.push_canary_metrics(2, 0.015, 100.0);
        let decision = canary_decide(&history, &config);
        // Could be Continue or Promote; not Rollback or Pause
        assert!(
            decision == CanaryDecision::Continue || decision == CanaryDecision::Promote,
            "healthy varied metrics should give Continue or Promote, got {decision:?}"
        );
    }

    // ── Test 29: canary_decide Rollback when error trend is rising sharply ──
    #[test]
    fn test_canary_decide_rollback_rising_trend() {
        let mut history = CanaryMetricsHistory::default();
        let config = CanaryConfig::new("c-v2", "b-v1");
        // Sharply rising error rate from 0% to 8%
        for i in 0..8 {
            history.push_canary_metrics(i, i as f32 * 0.01, 100.0);
        }
        // The latest value (0.07) may or may not trigger rollback alone,
        // but the trend (slope 0.01/step) should.
        // Add a high point to ensure trend is clearly positive.
        history.push_canary_metrics(8, 0.10, 100.0);
        assert!(
            matches!(
                canary_decide(&history, &config),
                CanaryDecision::Rollback(_)
            ),
            "rising error trend should trigger rollback"
        );
    }

    // ── Test 30: log_phase_transition records correct phase names ──
    #[test]
    fn test_log_phase_transition_phase_names() {
        let from = CanaryPhase::Running {
            traffic_percent: 5.0,
            started_at: std::time::Instant::now(),
        };
        let to = CanaryPhase::RolledBack {
            reason: "test".to_string(),
        };
        let transition = log_phase_transition(&from, &to, "high error rate");
        assert_eq!(transition.from, "Running");
        assert_eq!(transition.to, "RolledBack");
        assert_eq!(transition.reason, "high error rate");
    }

    // ── Test 31: log_phase_transition timestamp is recent ──
    #[test]
    fn test_log_phase_transition_timestamp_recent() {
        let from = CanaryPhase::Idle;
        let to = CanaryPhase::Running {
            traffic_percent: 5.0,
            started_at: std::time::Instant::now(),
        };
        let before = std::time::Instant::now();
        let transition = log_phase_transition(&from, &to, "starting");
        let after = std::time::Instant::now();
        assert!(transition.timestamp >= before);
        assert!(transition.timestamp <= after);
    }
}
