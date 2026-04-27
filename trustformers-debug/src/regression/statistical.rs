//! Statistical regression detection algorithms.
//!
//! Provides:
//! - [`StatRegressionDetector`] — z-score + relative-change detector with
//!   Welford online mean/variance estimation.
//! - [`CusumDetector`] — CUSUM sequential change-point detection algorithm.
//!
//! These supplement the performance-baseline detector in `detector.rs` with
//! algorithms operating directly on a streaming sequence of scalar metric
//! values rather than on full profiling reports.

// ─────────────────────────────────────────────────────────────────────────────
// BaselineStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistical summary of a reference sample set.
///
/// Built via [`StatRegressionDetector::build_baseline`] using Welford's
/// online algorithm for numerically stable mean and variance.
///
/// # Example
///
/// ```
/// use trustformers_debug::regression::statistical::StatRegressionDetector;
///
/// let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let b = StatRegressionDetector::build_baseline(&samples);
/// assert!((b.mean - 3.0).abs() < 1e-9);
/// assert_eq!(b.sample_count, 5);
/// ```
#[derive(Debug, Clone)]
pub struct StatBaselineStats {
    /// Arithmetic mean of the baseline samples.
    pub mean: f64,
    /// Sample standard deviation (Bessel-corrected).
    pub std: f64,
    /// Minimum value in the baseline sample set.
    pub min: f64,
    /// Maximum value in the baseline sample set.
    pub max: f64,
    /// Number of samples used to compute these statistics.
    pub sample_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// RegressionDirection / RegressionSeverity / RegressionEvent
// ─────────────────────────────────────────────────────────────────────────────

/// Indicates the direction that is considered "bad" for a metric.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatRegressionDirection {
    /// Metric should increase over time (e.g. accuracy, F1).
    Higher,
    /// Metric should decrease over time (e.g. loss, latency).
    Lower,
    /// Flag any significant deviation in either direction.
    Either,
}

/// Severity class for a detected regression event.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum StatRegressionSeverity {
    /// |z| or |rel_change| is slightly above threshold.
    Mild,
    /// Clearly detectable regression (|rel_change| > 10 %).
    Moderate,
    /// Strong regression (|rel_change| > 25 %).
    Severe,
    /// Extreme regression (|rel_change| > 50 %).
    Critical,
}

impl StatRegressionSeverity {
    /// Determines severity from the absolute relative change (expressed as a
    /// fraction, not a percentage).
    pub fn from_relative_change(rel: f64) -> Self {
        let abs = rel.abs();
        if abs > 0.50 {
            Self::Critical
        } else if abs > 0.25 {
            Self::Severe
        } else if abs > 0.10 {
            Self::Moderate
        } else {
            Self::Mild
        }
    }
}

/// A single detected regression event.
#[derive(Debug, Clone)]
pub struct StatRegressionEvent {
    /// Training step at which the anomaly was observed.
    pub step: u64,
    /// The observed metric value.
    pub value: f64,
    /// Signed z-score: `(value − mean) / std`.
    pub z_score: f64,
    /// Signed relative change: `(value − mean) / mean`.
    pub relative_change: f64,
    /// Severity class.
    pub severity: StatRegressionSeverity,
}

// ─────────────────────────────────────────────────────────────────────────────
// StatRegressionConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`StatRegressionDetector`].
#[derive(Debug, Clone)]
pub struct StatRegressionConfig {
    /// Minimum |z-score| to flag a regression (default: 3.0).
    pub z_score_threshold: f64,
    /// Minimum |relative change| to flag a regression (default: 0.10 = 10 %).
    pub relative_threshold: f64,
    /// Minimum number of baseline samples before detection is enabled.
    pub min_samples_for_detection: usize,
    /// Which deviation direction triggers an alert.
    pub direction: StatRegressionDirection,
}

impl Default for StatRegressionConfig {
    fn default() -> Self {
        Self {
            z_score_threshold: 3.0,
            relative_threshold: 0.10,
            min_samples_for_detection: 5,
            direction: StatRegressionDirection::Either,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StatRegressionDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Statistical regression detector for a single scalar training metric.
///
/// Uses a fixed [`StatBaselineStats`] as the reference distribution and emits
/// [`StatRegressionEvent`]s whenever a new observation deviates significantly
/// from that baseline.
///
/// # Example
///
/// ```
/// use trustformers_debug::regression::statistical::{
///     StatBaselineStats, StatRegressionConfig, StatRegressionDetector, StatRegressionDirection,
/// };
///
/// let samples = [1.0f64, 1.1, 0.9, 1.05, 0.95];
/// let baseline = StatRegressionDetector::build_baseline(&samples);
///
/// let config = StatRegressionConfig {
///     z_score_threshold: 2.0,
///     relative_threshold: 0.15,
///     min_samples_for_detection: 2,
///     direction: StatRegressionDirection::Either,
/// };
///
/// let mut detector = StatRegressionDetector::new("loss", baseline, config);
/// // Far-out value should trigger a regression
/// let event = detector.check_point(10, 5.0);
/// assert!(event.is_some());
/// ```
pub struct StatRegressionDetector {
    pub metric_name: String,
    pub baseline: StatBaselineStats,
    pub config: StatRegressionConfig,
    pub detection_history: Vec<StatRegressionEvent>,
}

impl StatRegressionDetector {
    /// Creates a new detector for the named metric.
    pub fn new(
        metric_name: &str,
        baseline: StatBaselineStats,
        config: StatRegressionConfig,
    ) -> Self {
        Self {
            metric_name: metric_name.to_string(),
            baseline,
            config,
            detection_history: Vec::new(),
        }
    }

    /// Computes the z-score of a new observation relative to the baseline.
    pub fn z_score(&self, value: f64) -> f64 {
        if self.baseline.std.abs() < f64::EPSILON {
            return 0.0;
        }
        (value - self.baseline.mean) / self.baseline.std
    }

    /// Computes the relative change `(value − mean) / mean`.
    pub fn relative_change(&self, value: f64) -> f64 {
        if self.baseline.mean.abs() < f64::EPSILON {
            return 0.0;
        }
        (value - self.baseline.mean) / self.baseline.mean
    }

    /// Evaluates a new data point and returns a [`StatRegressionEvent`] if a
    /// regression is detected, or `None` otherwise.
    ///
    /// A regression is detected when:
    /// 1. `|z_score| > config.z_score_threshold` **and**
    /// 2. `|relative_change| > config.relative_threshold` **and**
    /// 3. The direction constraint is satisfied.
    ///
    /// Detection only fires once `baseline.sample_count >= min_samples_for_detection`.
    pub fn check_point(&mut self, step: u64, value: f64) -> Option<StatRegressionEvent> {
        if self.baseline.sample_count < self.config.min_samples_for_detection {
            return None;
        }

        let z = self.z_score(value);
        let rel = self.relative_change(value);

        // Check direction constraint.
        let direction_ok = match self.config.direction {
            StatRegressionDirection::Higher => rel < -self.config.relative_threshold,
            StatRegressionDirection::Lower => rel > self.config.relative_threshold,
            StatRegressionDirection::Either => rel.abs() > self.config.relative_threshold,
        };

        if z.abs() < self.config.z_score_threshold || !direction_ok {
            return None;
        }

        let severity = StatRegressionSeverity::from_relative_change(rel);
        let event = StatRegressionEvent { step, value, z_score: z, relative_change: rel, severity };
        self.detection_history.push(event.clone());
        Some(event)
    }

    /// Returns the last `n` events in detection history (oldest first).
    pub fn recent_events(&self, n: usize) -> &[StatRegressionEvent] {
        let len = self.detection_history.len();
        let start = len.saturating_sub(n);
        &self.detection_history[start..]
    }

    /// Builds a [`StatBaselineStats`] from a slice of samples using Welford's
    /// one-pass algorithm for numerically stable variance computation.
    ///
    /// Returns a baseline with `sample_count = 0` and all zeros when `samples`
    /// is empty.
    pub fn build_baseline(samples: &[f64]) -> StatBaselineStats {
        if samples.is_empty() {
            return StatBaselineStats {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                sample_count: 0,
            };
        }

        // Welford's online algorithm
        let mut count = 0usize;
        let mut mean = 0.0f64;
        let mut m2 = 0.0f64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for &x in samples {
            count += 1;
            let delta = x - mean;
            mean += delta / count as f64;
            let delta2 = x - mean;
            m2 += delta * delta2;
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
        }

        let variance = if count > 1 { m2 / (count - 1) as f64 } else { 0.0 };
        let std = variance.sqrt();

        StatBaselineStats { mean, std, min, max, sample_count: count }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CUSUM
// ─────────────────────────────────────────────────────────────────────────────

/// The direction of a CUSUM change-point alert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeDirection {
    /// The process mean appears to have shifted upward.
    Up,
    /// The process mean appears to have shifted downward.
    Down,
}

/// Alert emitted by [`CusumDetector`] when a change point is detected.
#[derive(Debug, Clone)]
pub struct CusumAlert {
    /// Direction of the detected mean shift.
    pub direction: ChangeDirection,
    /// The cumulative-sum value that crossed the decision threshold.
    pub s_value: f64,
}

/// CUSUM (Cumulative Sum) sequential change-point detector.
///
/// Detects a sustained shift in the mean of a metric stream.  The algorithm
/// maintains two accumulators:
///
/// - `S_hi` detects **upward** mean shifts (metric increasing).
/// - `S_lo` detects **downward** mean shifts (metric decreasing).
///
/// # Parameters
///
/// - `k` — allowable slack, typically `0.5 * σ * shift_size_in_sigmas`.
/// - `h` — decision threshold (number of standard deviations of the
///   cumulative sum before an alarm is raised).  A common choice is `4.0 – 5.0`.
///
/// # Example
///
/// ```
/// use trustformers_debug::regression::statistical::{CusumDetector, ChangeDirection};
///
/// let mut cusum = CusumDetector::new(0.0, 1.0, 0.5, 4.0);
/// // feed values well above the target mean — should eventually alert
/// let mut alerted = false;
/// for _ in 0..20 {
///     if let Some(a) = cusum.update(3.0) {
///         assert_eq!(a.direction, ChangeDirection::Up);
///         alerted = true;
///         break;
///     }
/// }
/// assert!(alerted, "CUSUM should have detected upward shift");
/// ```
pub struct CusumDetector {
    /// Allowable slack (reference value).
    pub k: f64,
    /// Decision threshold — alarm when `S_hi > h` or `S_lo > h`.
    pub h: f64,
    /// Upper cumulative sum.
    pub s_hi: f64,
    /// Lower cumulative sum.
    pub s_lo: f64,
    /// Target (in-control) process mean.
    pub target_mean: f64,
    /// Target (in-control) process standard deviation.
    pub target_std: f64,
}

impl CusumDetector {
    /// Creates a new CUSUM detector.
    ///
    /// # Arguments
    ///
    /// - `target_mean` — in-control mean.
    /// - `target_std` — in-control standard deviation (used to normalise inputs).
    /// - `k` — slack parameter (0.5 is a common default for detecting 1σ shifts).
    /// - `h` — threshold (4.0–5.0 gives low false-alarm rates for Gaussian inputs).
    pub fn new(target_mean: f64, target_std: f64, k: f64, h: f64) -> Self {
        Self {
            k,
            h,
            s_hi: 0.0,
            s_lo: 0.0,
            target_mean,
            target_std,
        }
    }

    /// Incorporates a new observation and returns an alert if a change point is
    /// detected.
    ///
    /// The observation is first standardised as `z = (value − target_mean) / target_std`
    /// (unless `target_std == 0`, in which case the raw deviation is used).
    ///
    /// After firing, the triggering accumulator is reset to zero so detection
    /// can resume.  Callers that wish to track sustained changes should call
    /// [`reset`](Self::reset) manually instead.
    pub fn update(&mut self, value: f64) -> Option<CusumAlert> {
        let z = if self.target_std.abs() > f64::EPSILON {
            (value - self.target_mean) / self.target_std
        } else {
            value - self.target_mean
        };

        self.s_hi = (self.s_hi + z - self.k).max(0.0);
        self.s_lo = (self.s_lo - z - self.k).max(0.0);

        if self.s_hi > self.h {
            let s_value = self.s_hi;
            self.s_hi = 0.0; // reset after alarm
            return Some(CusumAlert { direction: ChangeDirection::Up, s_value });
        }
        if self.s_lo > self.h {
            let s_value = self.s_lo;
            self.s_lo = 0.0;
            return Some(CusumAlert { direction: ChangeDirection::Down, s_value });
        }
        None
    }

    /// Resets both cumulative sums to zero without changing parameters.
    pub fn reset(&mut self) {
        self.s_hi = 0.0;
        self.s_lo = 0.0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── StatRegressionDetector::build_baseline ────────────────────────────────

    #[test]
    fn test_build_baseline_mean() {
        let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = StatRegressionDetector::build_baseline(&samples);
        assert!((b.mean - 3.0).abs() < 1e-9);
        assert_eq!(b.sample_count, 5);
        assert_eq!(b.min, 1.0);
        assert_eq!(b.max, 5.0);
    }

    #[test]
    fn test_build_baseline_std() {
        // For [1, 2, 3, 4, 5] mean=3, sample std = sqrt(2.5) ≈ 1.5811
        let samples = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let b = StatRegressionDetector::build_baseline(&samples);
        assert!((b.mean - 3.0).abs() < 1e-9);
        // Bessel-corrected std for [1,2,3,4,5]: variance = (4+1+0+1+4)/4 = 2.5, std = sqrt(2.5)
        let expected_std = 2.5f64.sqrt();
        assert!((b.std - expected_std).abs() < 1e-9, "std={}", b.std);
        assert!(b.std > 0.0);
    }

    #[test]
    fn test_build_baseline_empty() {
        let b = StatRegressionDetector::build_baseline(&[]);
        assert_eq!(b.sample_count, 0);
        assert_eq!(b.mean, 0.0);
    }

    #[test]
    fn test_build_baseline_single() {
        let b = StatRegressionDetector::build_baseline(&[42.0]);
        assert_eq!(b.mean, 42.0);
        assert_eq!(b.std, 0.0);
        assert_eq!(b.sample_count, 1);
    }

    // ── z_score and relative_change ──────────────────────────────────────────

    #[test]
    fn test_z_score_positive() {
        let samples: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let baseline = StatRegressionDetector::build_baseline(&samples);
        let detector = StatRegressionDetector::new("metric", baseline, Default::default());
        let z = detector.z_score(20.0); // far above mean (4.5)
        assert!(z > 3.0, "z should be > 3.0, got {z}");
    }

    #[test]
    fn test_z_score_zero_std() {
        let baseline = StatBaselineStats { mean: 5.0, std: 0.0, min: 5.0, max: 5.0, sample_count: 5 };
        let detector = StatRegressionDetector::new("m", baseline, Default::default());
        assert_eq!(detector.z_score(5.0), 0.0);
        assert_eq!(detector.z_score(10.0), 0.0);
    }

    #[test]
    fn test_relative_change_positive() {
        let baseline = StatBaselineStats { mean: 2.0, std: 0.1, min: 1.9, max: 2.1, sample_count: 10 };
        let detector = StatRegressionDetector::new("x", baseline, Default::default());
        let rel = detector.relative_change(3.0); // 50% increase
        assert!((rel - 0.5).abs() < 1e-9, "rel={rel}");
    }

    // ── check_point regression detection ─────────────────────────────────────

    #[test]
    fn test_check_point_detects_regression() {
        let samples: Vec<f64> = (0..20).map(|_| 1.0).collect();
        let baseline = StatRegressionDetector::build_baseline(&samples);
        let config = StatRegressionConfig {
            z_score_threshold: 2.0,
            relative_threshold: 0.10,
            min_samples_for_detection: 5,
            direction: StatRegressionDirection::Either,
        };
        // std is 0 for constant samples, so z_score will be 0 — test with realistic data
        let samples2: Vec<f64> = (0..20).map(|i| 1.0 + (i as f64) * 0.01).collect();
        let baseline2 = StatRegressionDetector::build_baseline(&samples2);
        let mut detector = StatRegressionDetector::new("loss", baseline2, config);
        // Inject a value 5 std-devs away from the mean
        let mean = detector.baseline.mean;
        let std = detector.baseline.std;
        let far_value = mean + 6.0 * std;
        let event = detector.check_point(100, far_value);
        assert!(event.is_some(), "should detect regression for extreme value");
    }

    #[test]
    fn test_check_point_no_detection_below_threshold() {
        let samples: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.1).collect();
        let baseline = StatRegressionDetector::build_baseline(&samples);
        let config = StatRegressionConfig {
            z_score_threshold: 3.0,
            relative_threshold: 0.50,
            min_samples_for_detection: 5,
            direction: StatRegressionDirection::Either,
        };
        let mut detector = StatRegressionDetector::new("acc", baseline, config);
        // A value only 1% away from mean should not trigger
        let close_val = detector.baseline.mean * 1.01;
        let event = detector.check_point(1, close_val);
        assert!(event.is_none(), "should not detect for small deviation");
    }

    #[test]
    fn test_check_point_direction_lower_only() {
        let samples: Vec<f64> = (0..20).map(|i| 10.0 + (i as f64) * 0.05).collect();
        let baseline = StatRegressionDetector::build_baseline(&samples);
        let config = StatRegressionConfig {
            z_score_threshold: 1.5,
            relative_threshold: 0.05,
            min_samples_for_detection: 5,
            direction: StatRegressionDirection::Lower, // higher is worse
        };
        let mut detector = StatRegressionDetector::new("loss", baseline, config);
        let mean = detector.baseline.mean;
        let std = detector.baseline.std.max(0.05);
        // Value far BELOW mean (improvement) — should NOT trigger for Lower direction
        let below = mean - 5.0 * std;
        assert!(detector.check_point(1, below).is_none());
        // Value far ABOVE mean (regression for loss) — should trigger
        let above = mean + 5.0 * std;
        assert!(detector.check_point(2, above).is_some());
    }

    #[test]
    fn test_check_point_insufficient_samples() {
        let baseline = StatBaselineStats { mean: 5.0, std: 1.0, min: 4.0, max: 6.0, sample_count: 2 };
        let config = StatRegressionConfig {
            min_samples_for_detection: 10,
            ..Default::default()
        };
        let mut detector = StatRegressionDetector::new("m", baseline, config);
        assert!(detector.check_point(0, 100.0).is_none());
    }

    #[test]
    fn test_recent_events() {
        let samples: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
        let baseline = StatRegressionDetector::build_baseline(&samples);
        let config = StatRegressionConfig {
            z_score_threshold: 1.0,
            relative_threshold: 0.05,
            min_samples_for_detection: 5,
            direction: StatRegressionDirection::Either,
        };
        let mut detector = StatRegressionDetector::new("m", baseline, config);
        let mean = detector.baseline.mean;
        let std = detector.baseline.std.max(0.01);
        for step in 0..5_u64 {
            detector.check_point(step, mean + 10.0 * std);
        }
        let recent = detector.recent_events(3);
        assert!(recent.len() <= 3);
    }

    // ── StatRegressionSeverity ────────────────────────────────────────────────

    #[test]
    fn test_severity_thresholds() {
        assert_eq!(StatRegressionSeverity::from_relative_change(0.05), StatRegressionSeverity::Mild);
        assert_eq!(StatRegressionSeverity::from_relative_change(0.15), StatRegressionSeverity::Moderate);
        assert_eq!(StatRegressionSeverity::from_relative_change(0.30), StatRegressionSeverity::Severe);
        assert_eq!(StatRegressionSeverity::from_relative_change(0.60), StatRegressionSeverity::Critical);
        // negative (improvement) uses abs
        assert_eq!(StatRegressionSeverity::from_relative_change(-0.60), StatRegressionSeverity::Critical);
    }

    // ── CusumDetector ─────────────────────────────────────────────────────────

    #[test]
    fn test_cusum_no_alert_for_in_control() {
        let mut cusum = CusumDetector::new(0.0, 1.0, 0.5, 4.0);
        // Feed values close to target mean — should not alert
        for i in 0..50 {
            let v = if i % 2 == 0 { 0.1 } else { -0.1 };
            assert!(cusum.update(v).is_none(), "should not alert for in-control data");
        }
    }

    #[test]
    fn test_cusum_detects_upward_shift() {
        let mut cusum = CusumDetector::new(0.0, 1.0, 0.5, 4.0);
        let mut alerted = false;
        for _ in 0..50 {
            if let Some(a) = cusum.update(2.0) {
                assert_eq!(a.direction, ChangeDirection::Up);
                assert!(a.s_value > 4.0);
                alerted = true;
                break;
            }
        }
        assert!(alerted, "CUSUM must detect upward shift");
    }

    #[test]
    fn test_cusum_detects_downward_shift() {
        let mut cusum = CusumDetector::new(0.0, 1.0, 0.5, 4.0);
        let mut alerted = false;
        for _ in 0..50 {
            if let Some(a) = cusum.update(-2.0) {
                assert_eq!(a.direction, ChangeDirection::Down);
                alerted = true;
                break;
            }
        }
        assert!(alerted, "CUSUM must detect downward shift");
    }

    #[test]
    fn test_cusum_reset() {
        let mut cusum = CusumDetector::new(0.0, 1.0, 0.5, 4.0);
        cusum.s_hi = 3.9;
        cusum.s_lo = 3.9;
        cusum.reset();
        assert_eq!(cusum.s_hi, 0.0);
        assert_eq!(cusum.s_lo, 0.0);
    }

    #[test]
    fn test_cusum_alert_resets_accumulator() {
        let mut cusum = CusumDetector::new(0.0, 1.0, 0.5, 4.0);
        // Force s_hi to be just below threshold, then push it over.
        cusum.s_hi = 4.4;
        let alert = cusum.update(0.2);
        assert!(alert.is_some());
        // After the alert, s_hi should have been reset.
        assert_eq!(cusum.s_hi, 0.0);
    }

    #[test]
    fn test_cusum_zero_std_uses_raw_deviation() {
        let mut cusum = CusumDetector::new(5.0, 0.0, 0.5, 4.0);
        let mut alerted = false;
        for _ in 0..20 {
            if cusum.update(7.0).is_some() {
                alerted = true;
                break;
            }
        }
        assert!(alerted, "CUSUM with zero std should still detect large deviation");
    }
}
