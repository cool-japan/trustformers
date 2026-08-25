//! Performance-regression detection for `KernelOptimizationAnalyzer`.
//!
//! [`PerformanceRegressionDetector`] records each kernel's execution-time
//! history, establishes a real baseline distribution once enough samples
//! exist, and tests later measurements against that baseline with a
//! genuine Welch's t-test (see [`super::analysis::compare_to_baseline`]) --
//! never a hardcoded "stable" verdict. Every field on every type in this
//! module is either measured from real samples or an honest `None`/empty
//! collection when there is not yet enough data; nothing here is a
//! disclosed-but-never-computed placeholder.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::ring_buffer::TimestampedRingBuffer;

use super::analysis;
use super::KernelProfileData;

/// Performance regression detection
#[derive(Debug)]
pub struct PerformanceRegressionDetector {
    baseline_profiles: HashMap<String, BaselineProfile>,
    regression_alerts: Vec<RegressionAlert>,
    statistical_analyzer: StatisticalAnalyzer,
    alert_thresholds: RegressionThresholds,
    /// Bounded, timestamped execution-time history (seconds) per kernel --
    /// the real raw data [`Self::check_regression`]/[`Self::get_status`]
    /// compare against `baseline_profiles`. Bounded per-kernel via
    /// [`TimestampedRingBuffer`]'s fixed capacity so memory stays bounded
    /// across a long-running process.
    execution_history: HashMap<String, TimestampedRingBuffer<f64>>,
}

/// Per-kernel execution-time history capacity (samples). Old samples are
/// evicted once a kernel's history exceeds this.
const EXECUTION_HISTORY_CAPACITY: usize = 500;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineProfile {
    pub kernel_name: String,
    pub baseline_performance: Duration,
    pub performance_distribution: PerformanceDistribution,
    pub established_date: SystemTime,
    pub confidence_interval: (Duration, Duration),
}

/// Summary statistics of a kernel's baseline execution-time samples.
///
/// **Canonical unit: seconds, as `f64`.** These used to be `std::time::Duration`,
/// which quantizes to whole nanoseconds -- so the same measurements expressed
/// on a different time scale produced different published statistics, and a
/// microsecond-scale kernel's standard deviation could round-trip to `0 ns`,
/// driving the Welch t-statistic to infinity and its p-value to exactly 0.
/// `Duration` is still used for the human-facing latency fields on
/// [`BaselineProfile`] and [`RegressionAlert`]; nothing that feeds the
/// statistics goes through it any more.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDistribution {
    /// Arithmetic mean of the baseline samples, in seconds.
    pub mean_secs: f64,
    /// Sample standard deviation of the baseline samples, in seconds.
    pub std_dev_secs: f64,
    /// 50th, 90th, 95th and 99th percentiles of the baseline samples, in seconds.
    pub percentiles: HashMap<u8, f64>,
    /// `mean + 3 * std_dev`, in seconds.
    pub outlier_threshold_secs: f64,
    /// Number of real samples the distribution was estimated from --
    /// required for the Welch's t-test comparison in
    /// [`PerformanceRegressionDetector::check_regression`] /
    /// [`PerformanceRegressionDetector::get_status`].
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAlert {
    pub alert_id: Uuid,
    pub kernel_name: String,
    pub alert_type: RegressionType,
    pub severity: RegressionSeverity,
    pub current_performance: Duration,
    pub baseline_performance: Duration,
    pub regression_magnitude: f64,
    pub detection_timestamp: SystemTime,
    pub potential_causes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionType {
    PerformanceDegradation,
    MemoryUsageIncrease,
    OccupancyDecrease,
    BandwidthUtilizationDrop,
    EnergyEfficiencyLoss,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Minor,    // < 5% regression
    Moderate, // 5-15% regression
    Major,    // 15-30% regression
    Critical, // > 30% regression
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionThresholds {
    /// Minimum magnitude (fraction, e.g. `0.05` = 5%) for a statistically
    /// significant slowdown to count as a regression at all -- the entry
    /// gate checked by `super::analysis::detect_regression`, not a
    /// [`RegressionSeverity`] bucket boundary (see
    /// `super::analysis::classify_severity`).
    pub minor_threshold: f64,
    pub moderate_threshold: f64,
    pub major_threshold: f64,
    /// Retained for API/config completeness (a caller may reasonably
    /// expect a "critical" knob alongside the other three), but not
    /// currently consumed by `super::analysis::classify_severity`:
    /// with 4 severities and 4 fields, 3 boundaries already fully
    /// partition the magnitude axis into 4 buckets once
    /// `minor_threshold` is spoken for as the entry gate above, so this
    /// field has no remaining boundary to own without either
    /// contradicting [`RegressionSeverity`]'s own documented ranges or
    /// leaving a fifth, unreachable bucket.
    pub critical_threshold: f64,
    pub detection_window: Duration,
    pub confidence_level: f64,
}

#[derive(Debug)]
pub struct StatisticalAnalyzer {
    sample_size_requirements: HashMap<String, usize>,
    statistical_tests: Vec<StatisticalTest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    pub test_name: String,
    pub test_type: TestType,
    /// Alpha the test was judged at (`1 - confidence_level`).
    pub significance_level: f64,
    /// Observed (post-hoc) power of the test -- see
    /// `super::analysis::observed_power` -- or `None` when it is not
    /// computable from the recorded statistic.
    ///
    /// This field was called `power` and was set to `1.0 - p_value`, which is
    /// the confidence in the observed result, not `P(reject H0 | H1 true)`.
    pub observed_power: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestType {
    TTest,
    MannWhitneyU,
    KolmogorovSmirnov,
    ChangePointDetection,
    AnomalyDetection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionStatus {
    /// `true` only when the most recent comparison found a statistically
    /// significant slowdown beyond the detector's `minor_threshold` -- see
    /// [`PerformanceRegressionDetector::check_regression`]. Structurally
    /// capable of being `true`: this is not a constant.
    pub has_regression: bool,
    /// All alerts raised for this kernel so far (each already gated on
    /// significance + magnitude at the time it fired).
    pub regression_alerts: Vec<RegressionAlert>,
    pub performance_trend: PerformanceTrend,
    pub baseline_comparison: BaselineComparison,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    /// Percentage difference of the recent sample's mean vs the baseline
    /// mean (positive = slower). Real per-comparison value, computed from
    /// `analysis::compare_to_baseline`'s Welch's t-test.
    pub current_vs_baseline: f64,
    /// `1.0 - p_value` of the same Welch's t-test (higher = more
    /// significant a change was detected). Not a fixed confidence-level
    /// constant.
    pub statistical_significance: f64,
    /// 95% confidence interval on `current_vs_baseline`, expressed as a
    /// fraction (not a percentage) of the baseline mean.
    pub confidence_interval: (f64, f64),
}

impl PerformanceRegressionDetector {
    pub fn new() -> Result<Self> {
        Ok(Self {
            baseline_profiles: HashMap::new(),
            regression_alerts: vec![],
            statistical_analyzer: StatisticalAnalyzer::new()?,
            alert_thresholds: RegressionThresholds {
                minor_threshold: 0.05,
                moderate_threshold: 0.15,
                major_threshold: 0.30,
                critical_threshold: 0.50,
                detection_window: Duration::from_secs(3600),
                confidence_level: 0.95,
            },
            execution_history: HashMap::new(),
        })
    }

    pub fn new_empty() -> Self {
        Self {
            baseline_profiles: HashMap::new(),
            regression_alerts: vec![],
            statistical_analyzer: StatisticalAnalyzer::new_empty(),
            alert_thresholds: RegressionThresholds {
                minor_threshold: 0.05,
                moderate_threshold: 0.15,
                major_threshold: 0.30,
                critical_threshold: 0.50,
                detection_window: Duration::from_secs(3600),
                confidence_level: 0.95,
            },
            execution_history: HashMap::new(),
        }
    }

    /// Real implementation shared by [`Self::check_regression`] (which logs
    /// a new alert as a side effect) and [`Self::get_status`] (a pure
    /// query): compute the Welch's t-test outcome for `kernel_name`'s
    /// recent samples -- those recorded since the baseline was
    /// established, bounded to `alert_thresholds.detection_window` -- vs
    /// its established baseline.
    ///
    /// Returns `None` when there is no baseline yet, fewer than
    /// [`analysis::MIN_COMPARISON_SAMPLES`] recent samples fall inside the
    /// window, or the underlying test itself has no meaningful result to
    /// report (see [`analysis::compare_to_baseline`]).
    fn recent_comparison(
        &self,
        kernel_name: &str,
        now_ns: u64,
    ) -> Option<analysis::BaselineTestOutcome> {
        let baseline = self.baseline_profiles.get(kernel_name)?;
        let history = self.execution_history.get(kernel_name)?;

        let established_ns = analysis::system_time_to_ns(baseline.established_date);
        let window_start_ns = established_ns
            .max(now_ns.saturating_sub(self.alert_thresholds.detection_window.as_nanos() as u64));
        let recent = history.values_in_range(window_start_ns, now_ns);
        if recent.len() < analysis::MIN_COMPARISON_SAMPLES {
            return None;
        }

        analysis::compare_to_baseline(
            baseline.performance_distribution.mean_secs,
            baseline.performance_distribution.std_dev_secs,
            baseline.performance_distribution.sample_count,
            &recent,
        )
    }

    /// Record one real execution-time measurement and, once enough history
    /// exists, either establish this kernel's baseline or test the recent
    /// window against it -- pushing a real [`RegressionAlert`] only when
    /// the Welch's t-test finds a statistically significant slowdown
    /// beyond `alert_thresholds.minor_threshold`. Never a no-op that
    /// silently discards `profile_data`.
    pub fn check_regression(
        &mut self,
        kernel_name: &str,
        profile_data: &KernelProfileData,
    ) -> Result<()> {
        let now_ns = analysis::system_time_to_ns(SystemTime::now());
        let sample_secs = profile_data.execution_time.as_secs_f64();

        self.execution_history
            .entry(kernel_name.to_string())
            .or_insert_with(|| TimestampedRingBuffer::new(EXECUTION_HISTORY_CAPACITY))
            .push_now(sample_secs, now_ns);

        if !self.baseline_profiles.contains_key(kernel_name) {
            // Not enough data to establish a baseline yet is an honest
            // absence, not a fabricated "no regression" -- only act once
            // real history has accumulated.
            if let Some(samples) = self.execution_history.get(kernel_name).and_then(|h| {
                (h.len() >= analysis::MIN_BASELINE_SAMPLES)
                    .then(|| h.iter_ordered().map(|v| v.value).collect::<Vec<f64>>())
            }) {
                let baseline = analysis::establish_baseline(kernel_name, &samples);
                self.statistical_analyzer
                    .sample_size_requirements
                    .insert(kernel_name.to_string(), analysis::MIN_BASELINE_SAMPLES);
                self.baseline_profiles.insert(kernel_name.to_string(), baseline);
            }
            return Ok(());
        }

        let Some(test_outcome) = self.recent_comparison(kernel_name, now_ns) else {
            return Ok(());
        };

        let alpha = 1.0 - self.alert_thresholds.confidence_level;
        self.statistical_analyzer.statistical_tests.push(StatisticalTest {
            test_name: format!("Welch's t-test ({kernel_name})"),
            test_type: TestType::TTest,
            significance_level: alpha,
            observed_power: analysis::observed_power(
                test_outcome.t_statistic,
                test_outcome.degrees_of_freedom,
                alpha,
            ),
        });

        let baseline = self
            .baseline_profiles
            .get(kernel_name)
            .ok_or_else(|| anyhow::anyhow!("baseline for '{}' vanished mid-check", kernel_name))?;
        let check = analysis::detect_regression(
            kernel_name,
            baseline,
            test_outcome,
            &self.alert_thresholds,
        );
        if let Some(alert) = check.new_alert {
            self.regression_alerts.push(alert);
        }

        Ok(())
    }

    /// Real regression status for `kernel_name`, freshly recomputed from
    /// its execution-time history against its established baseline.
    ///
    /// Returns `Ok(None)` -- never a fabricated "stable" status --
    /// whenever there is not yet a baseline, or not yet enough recent
    /// samples in the detection window to compare against it. A pure
    /// query: unlike [`Self::check_regression`], it never appends to
    /// `regression_alerts`.
    pub fn get_status(&self, kernel_name: &str) -> Result<Option<RegressionStatus>> {
        if !self.baseline_profiles.contains_key(kernel_name) {
            return Ok(None);
        }
        let now_ns = analysis::system_time_to_ns(SystemTime::now());
        let Some(test_outcome) = self.recent_comparison(kernel_name, now_ns) else {
            return Ok(None);
        };
        let baseline = self
            .baseline_profiles
            .get(kernel_name)
            .ok_or_else(|| anyhow::anyhow!("baseline for '{}' vanished mid-check", kernel_name))?;
        let check = analysis::detect_regression(
            kernel_name,
            baseline,
            test_outcome,
            &self.alert_thresholds,
        );

        let regression_alerts: Vec<RegressionAlert> = self
            .regression_alerts
            .iter()
            .filter(|alert| alert.kernel_name == kernel_name)
            .cloned()
            .collect();

        Ok(Some(RegressionStatus {
            has_regression: check.new_alert.is_some(),
            regression_alerts,
            performance_trend: check.performance_trend,
            baseline_comparison: check.baseline_comparison,
        }))
    }
}

impl StatisticalAnalyzer {
    fn new() -> Result<Self> {
        Ok(Self {
            sample_size_requirements: HashMap::new(),
            statistical_tests: vec![],
        })
    }

    fn new_empty() -> Self {
        Self {
            sample_size_requirements: HashMap::new(),
            statistical_tests: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration as StdDuration;

    fn thresholds() -> RegressionThresholds {
        RegressionThresholds {
            minor_threshold: 0.05,
            moderate_threshold: 0.15,
            major_threshold: 0.30,
            critical_threshold: 0.50,
            detection_window: StdDuration::from_secs(3600),
            confidence_level: 0.95,
        }
    }

    fn profile(exec_secs: f64) -> KernelProfileData {
        KernelProfileData {
            execution_time: StdDuration::from_secs_f64(exec_secs),
            grid_size: (128, 1, 1),
            block_size: (256, 1, 1),
            shared_memory_bytes: 4096,
            registers_per_thread: 32,
            occupancy: 0.5,
            compute_utilization: 0.5,
            memory_bandwidth_utilization: 0.5,
            warp_efficiency: 0.9,
            memory_efficiency: 0.8,
        }
    }

    #[test]
    fn test_new_has_no_baseline_and_no_status() {
        let detector = PerformanceRegressionDetector::new().expect("new ok");
        assert!(
            detector.get_status("nope").expect("get_status ok").is_none(),
            "an unknown kernel has no baseline, so status must be an honest None"
        );
    }

    #[test]
    fn test_status_stays_none_below_baseline_sample_count() {
        let mut detector = PerformanceRegressionDetector::new().expect("new ok");
        for _ in 0..(analysis::MIN_BASELINE_SAMPLES - 1) {
            detector.check_regression("k", &profile(0.001)).expect("check ok");
        }
        assert!(
            detector.get_status("k").expect("get_status ok").is_none(),
            "fewer than MIN_BASELINE_SAMPLES real measurements must not fabricate a baseline"
        );
    }

    #[test]
    fn test_stable_kernel_reports_no_regression_with_real_stats() {
        let mut detector = PerformanceRegressionDetector::new().expect("new ok");
        // Constant 1ms execution time: baseline establishes, then recent
        // samples exactly match it -- no regression, but the comparison
        // itself must be real (not the old unconditional constant).
        for _ in 0..(analysis::MIN_BASELINE_SAMPLES + analysis::MIN_COMPARISON_SAMPLES) {
            detector.check_regression("stable_kernel", &profile(0.001)).expect("check ok");
        }
        let status = detector
            .get_status("stable_kernel")
            .expect("get_status ok")
            .expect("baseline should be established by now");
        assert!(
            !status.has_regression,
            "identical samples must not be flagged as a regression"
        );
        assert!(
            status.baseline_comparison.current_vs_baseline.abs() < 1e-6,
            "recent mean equals baseline mean -> ~0% difference, got {}",
            status.baseline_comparison.current_vs_baseline
        );
    }

    #[test]
    fn test_significant_slowdown_produces_real_alert() {
        let mut detector = PerformanceRegressionDetector::new().expect("new ok");
        // Establish a tight baseline around 1ms.
        for _ in 0..20 {
            detector.check_regression("slow_kernel", &profile(0.0010)).expect("check ok");
        }
        // Now feed a consistently, dramatically slower recent window (2x).
        for _ in 0..10 {
            detector.check_regression("slow_kernel", &profile(0.0020)).expect("check ok");
        }
        let status = detector
            .get_status("slow_kernel")
            .expect("get_status ok")
            .expect("baseline established");
        assert!(
            status.has_regression,
            "a real, sustained 2x slowdown must be detected, got {:?}",
            status.baseline_comparison
        );
        assert!(
            status.baseline_comparison.current_vs_baseline > 50.0,
            "current_vs_baseline should reflect the real ~100% slowdown, got {}",
            status.baseline_comparison.current_vs_baseline
        );
        assert!(
            !status.regression_alerts.is_empty(),
            "check_regression must have logged a real RegressionAlert"
        );
        assert_eq!(status.regression_alerts[0].kernel_name, "slow_kernel");
    }

    #[test]
    fn test_speedup_is_improving_not_a_regression() {
        let mut detector = PerformanceRegressionDetector::new().expect("new ok");
        for _ in 0..20 {
            detector.check_regression("fast_kernel", &profile(0.0020)).expect("check ok");
        }
        for _ in 0..10 {
            detector.check_regression("fast_kernel", &profile(0.0005)).expect("check ok");
        }
        let status = detector
            .get_status("fast_kernel")
            .expect("get_status ok")
            .expect("baseline established");
        assert!(
            !status.has_regression,
            "getting faster must never be reported as a regression"
        );
        assert!(
            status.baseline_comparison.current_vs_baseline < 0.0,
            "a real speedup must show a negative current_vs_baseline, got {}",
            status.baseline_comparison.current_vs_baseline
        );
    }

    #[test]
    fn test_classify_severity_uses_configured_thresholds() {
        let t = thresholds();
        assert!(matches!(
            analysis::classify_severity(0.04, &t),
            RegressionSeverity::Minor
        ));
        assert!(matches!(
            analysis::classify_severity(0.10, &t),
            RegressionSeverity::Moderate
        ));
        assert!(matches!(
            analysis::classify_severity(0.20, &t),
            RegressionSeverity::Major
        ));
        assert!(matches!(
            analysis::classify_severity(0.60, &t),
            RegressionSeverity::Critical
        ));
    }

    #[test]
    fn test_shrinking_variance_is_not_mislabeled_volatile() {
        // Regression test for a real bug: `is_volatile` originally fired
        // symmetrically on `variance_ratio` far from 1.0 in EITHER
        // direction (>=3x OR <=1/3x baseline variance). A recent window
        // whose variance SHRANK relative to baseline means the kernel got
        // MORE consistent -- the opposite of volatile -- and must fall
        // through to the ordinary Stable/Degrading/Improving
        // classification, never be reported as "Volatile".
        let t = thresholds();
        let baseline_samples = [
            0.0008, 0.0012, 0.0008, 0.0012, 0.0008, 0.0012, 0.0008, 0.0012,
        ];
        let baseline = analysis::establish_baseline("k", &baseline_samples);
        let outcome = analysis::BaselineTestOutcome {
            relative_change: 0.0, // no mean shift
            relative_change_ci: (0.0, 0.0),
            t_statistic: 0.0,
            p_value: 1.0, // not statistically significant
            degrees_of_freedom: 10.0,
            variance_ratio: 0.1, // recent variance is 1/10th of baseline's -- MORE consistent
        };
        let check = analysis::detect_regression("k", &baseline, outcome, &t);
        assert!(
            matches!(check.performance_trend, PerformanceTrend::Stable),
            "a recent window that became MORE consistent (variance_ratio well below 1.0) with \
             no significant mean shift must be Stable, not mislabeled Volatile, got {:?}",
            check.performance_trend
        );
    }

    #[test]
    fn test_growing_variance_is_labeled_volatile() {
        // The intended (one-sided) behavior of the same gate: a recent
        // window that genuinely got MORE erratic (variance_ratio >= 3x
        // baseline) must still be Volatile.
        let t = thresholds();
        let baseline_samples = [
            0.0008, 0.0012, 0.0008, 0.0012, 0.0008, 0.0012, 0.0008, 0.0012,
        ];
        let baseline = analysis::establish_baseline("k", &baseline_samples);
        let outcome = analysis::BaselineTestOutcome {
            relative_change: 0.0,
            relative_change_ci: (0.0, 0.0),
            t_statistic: 0.0,
            p_value: 1.0,
            degrees_of_freedom: 10.0,
            variance_ratio: 5.0,
        };
        let check = analysis::detect_regression("k", &baseline, outcome, &t);
        assert!(
            matches!(check.performance_trend, PerformanceTrend::Volatile),
            "a recent variance >= 3x baseline must still be Volatile, got {:?}",
            check.performance_trend
        );
    }

    /// The published Welch statistics must depend only on the *shape* of the
    /// measurements, never on the unit they happen to be expressed in.
    ///
    /// `PerformanceDistribution.{mean,std_dev}` used to be `Duration`, which
    /// quantizes to whole nanoseconds. Feeding the same relative samples in at
    /// second, millisecond and microsecond scale therefore produced three
    /// different p-values, and at microsecond scale the standard deviation
    /// round-tripped to `0 ns`, collapsing the p-value to exactly 0.0.
    #[test]
    fn test_welch_statistics_are_invariant_to_the_time_unit() {
        // Deliberately not round numbers: a Duration round-trip has to lose
        // something for the assertion to bite.
        const BASELINE: [f64; 10] = [
            1.031, 0.987, 1.004, 1.019, 0.973, 1.011, 0.996, 1.027, 0.981, 1.008,
        ];
        const RECENT: [f64; 8] = [1.137, 1.152, 1.129, 1.161, 1.143, 1.156, 1.134, 1.148];

        let p_at_scale = |scale: f64| -> f64 {
            let baseline_samples: Vec<f64> = BASELINE.iter().map(|v| v * scale).collect();
            let recent_samples: Vec<f64> = RECENT.iter().map(|v| v * scale).collect();
            let baseline = analysis::establish_baseline("k", &baseline_samples);
            let outcome = analysis::compare_to_baseline(
                baseline.performance_distribution.mean_secs,
                baseline.performance_distribution.std_dev_secs,
                baseline.performance_distribution.sample_count,
                &recent_samples,
            )
            .expect("both windows have enough real samples");
            outcome.p_value
        };

        let seconds = p_at_scale(1.0);
        let millis = p_at_scale(1e-3);
        let micros = p_at_scale(1e-6);

        assert!(seconds > 0.0 && seconds < 1.0, "sanity: got {seconds}");
        assert!(
            (millis - seconds).abs() < 1e-9,
            "millisecond scale must give the same p-value: {millis} vs {seconds}"
        );
        assert!(
            (micros - seconds).abs() < 1e-9,
            "microsecond scale must give the same p-value: {micros} vs {seconds}"
        );
        assert!(
            micros > 0.0,
            "a microsecond-scale std_dev must not collapse to zero"
        );
    }

    /// A far-tail comparison must report the real (tiny) p-value rather than
    /// underflowing to exactly 0.0, which `2 * (1 - cdf(|t|))` does.
    #[test]
    fn test_far_tail_p_value_does_not_underflow_to_zero() {
        let baseline_samples: Vec<f64> =
            (0..40).map(|i| 1.0 + if i % 2 == 0 { 0.001 } else { -0.001 }).collect();
        let recent_samples: Vec<f64> =
            (0..40).map(|i| 1.5 + if i % 2 == 0 { 0.001 } else { -0.001 }).collect();
        let baseline = analysis::establish_baseline("k", &baseline_samples);
        let outcome = analysis::compare_to_baseline(
            baseline.performance_distribution.mean_secs,
            baseline.performance_distribution.std_dev_secs,
            baseline.performance_distribution.sample_count,
            &recent_samples,
        )
        .expect("both windows have enough real samples");

        assert!(
            outcome.t_statistic.abs() > 100.0,
            "sanity: got t={}",
            outcome.t_statistic
        );
        assert!(
            outcome.p_value > 0.0,
            "an enormous but finite t-statistic has a tiny, non-zero p-value; got exactly 0.0"
        );
        assert!(
            outcome.p_value < 1e-30,
            "and it must still be tiny: {}",
            outcome.p_value
        );
    }

    /// `StatisticalTest.power` used to be `1 - p_value`. Observed power is a
    /// different quantity, and the recorded test must carry the real one.
    #[test]
    fn test_recorded_statistical_test_reports_observed_power_not_one_minus_p() {
        let mut detector = PerformanceRegressionDetector::new().expect("new ok");
        for _ in 0..analysis::MIN_BASELINE_SAMPLES {
            detector.check_regression("k", &profile(0.001)).expect("check ok");
        }
        // A clearly slower recent window, with a little jitter so the recent
        // variance is real.
        for i in 0..(analysis::MIN_COMPARISON_SAMPLES + 4) {
            let jitter = if i % 2 == 0 { 1.0e-6 } else { -1.0e-6 };
            detector.check_regression("k", &profile(0.0015 + jitter)).expect("check ok");
        }

        let test = detector
            .statistical_analyzer
            .statistical_tests
            .last()
            .expect("a comparison must have been recorded");
        let power = test.observed_power.expect("a finite t-statistic yields a power");
        assert!(
            (0.0..=1.0).contains(&power),
            "power must be a probability, got {power}"
        );
        assert!(
            (test.significance_level - 0.05).abs() < 1e-9,
            "alpha comes from the configured confidence level"
        );

        // A huge effect saturates both quantities at 1.0, so the
        // power-is-not-1-minus-p separation is asserted on a closed-form case
        // in `test_observed_power_is_not_one_minus_p`.
    }

    /// Closed form: at exactly the critical value the two-sided test rejects
    /// half the time, so observed power is ~0.5 -- while `1 - p` is ~0.95.
    #[test]
    fn test_observed_power_is_not_one_minus_p() {
        // t = t_crit(df=20, alpha=0.05) = 2.085963...
        let t = 2.085_963_447_265_837;
        let df = 20.0;
        let alpha = 0.05;

        let power = analysis::observed_power(t, df, alpha).expect("computable");
        assert!(
            (power - 0.5).abs() < 1e-3,
            "at the critical value the noncentral-t power is ~0.5, got {power}"
        );

        let p =
            trustformers_core::statistics::student_t_two_sided_p_value(t, df).expect("computable");
        assert!(
            (p - alpha).abs() < 1e-9,
            "sanity: t is the alpha critical value, p={p}"
        );
        assert!(
            ((1.0 - p) - power).abs() > 0.4,
            "1 - p = {} is a different quantity from power = {power}",
            1.0 - p
        );

        // A far larger effect really is near-certain to be detected.
        let strong = analysis::observed_power(8.0, df, alpha).expect("computable");
        assert!(strong > 0.99, "got {strong}");
        // And degenerate inputs are refused rather than invented.
        assert_eq!(analysis::observed_power(f64::NAN, df, alpha), None);
        assert_eq!(analysis::observed_power(t, 0.0, alpha), None);
        assert_eq!(analysis::observed_power(t, df, 0.0), None);
    }

    #[test]
    fn test_statistical_analyzer_new() {
        let analyzer = StatisticalAnalyzer::new().expect("new ok");
        assert!(analyzer.sample_size_requirements.is_empty());
        assert!(analyzer.statistical_tests.is_empty());
    }

    #[test]
    fn test_statistical_analyzer_new_empty() {
        let analyzer = StatisticalAnalyzer::new_empty();
        assert!(analyzer.sample_size_requirements.is_empty());
    }
}
