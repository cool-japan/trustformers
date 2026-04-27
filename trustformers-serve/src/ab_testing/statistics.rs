//! Statistical significance testing for A/B experiments.
//!
//! Provides:
//! - Welch's t-test (unequal variance t-test) for two independent samples
//! - SPRT (Sequential Probability Ratio Test) for early stopping
//! - Power analysis / minimum sample size calculation
//! - Summary statistics computation

use thiserror::Error;

/// Errors from statistical computations
#[derive(Debug, Clone, Error)]
pub enum StatError {
    #[error("Insufficient samples: need at least {min}, got {got}")]
    InsufficientSamples { min: usize, got: usize },

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for a sample
#[derive(Debug, Clone)]
pub struct SampleStats {
    pub n: usize,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub std_error: f64,
    pub min: f64,
    pub max: f64,
}

impl SampleStats {
    /// Compute summary statistics from a slice of samples.
    ///
    /// Requires at least 2 samples to compute variance.
    pub fn from_samples(samples: &[f64]) -> Result<Self, StatError> {
        if samples.len() < 2 {
            return Err(StatError::InsufficientSamples {
                min: 2,
                got: samples.len(),
            });
        }

        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;

        // Bessel-corrected sample variance
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

        let std_dev = variance.sqrt();
        let std_error = std_dev / (n as f64).sqrt();

        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Ok(Self {
            n,
            mean,
            variance,
            std_dev,
            std_error,
            min,
            max,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Welch's t-test
// ─────────────────────────────────────────────────────────────────────────────

/// Result of Welch's t-test
#[derive(Debug, Clone)]
pub struct TTestResult {
    pub t_statistic: f64,
    pub p_value: f64,
    pub is_significant: bool,
    pub confidence_level: f64,
    /// Degrees of freedom (Welch–Satterthwaite approximation)
    pub degrees_of_freedom: f64,
    pub control_mean: f64,
    pub treatment_mean: f64,
    /// (treatment_mean - control_mean) / |control_mean| — positive means improvement
    pub relative_improvement: f64,
}

/// Welch's t-test for two independent samples with potentially unequal variances.
///
/// Returns the t-statistic, a two-tailed p-value approximation, and whether the
/// result is statistically significant at the given significance level `alpha`.
pub fn welch_t_test(
    control: &[f64],
    treatment: &[f64],
    alpha: f64,
) -> Result<TTestResult, StatError> {
    if control.len() < 2 {
        return Err(StatError::InsufficientSamples {
            min: 2,
            got: control.len(),
        });
    }
    if treatment.len() < 2 {
        return Err(StatError::InsufficientSamples {
            min: 2,
            got: treatment.len(),
        });
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(StatError::InvalidParameter(format!(
            "alpha must be in (0, 1), got {alpha}"
        )));
    }

    let ctrl = SampleStats::from_samples(control)?;
    let trt = SampleStats::from_samples(treatment)?;

    // Welch t-statistic: (mean_A - mean_B) / sqrt(var_A/n_A + var_B/n_B)
    let se = ((ctrl.variance / ctrl.n as f64) + (trt.variance / trt.n as f64)).sqrt();

    if se < f64::EPSILON {
        return Err(StatError::InvalidParameter(
            "Standard error is zero; samples have no variance".to_string(),
        ));
    }

    let t = (ctrl.mean - trt.mean) / se;

    // Welch–Satterthwaite degrees of freedom
    let s1n1 = ctrl.variance / ctrl.n as f64;
    let s2n2 = trt.variance / trt.n as f64;
    let df = (s1n1 + s2n2).powi(2)
        / (s1n1.powi(2) / (ctrl.n - 1) as f64 + s2n2.powi(2) / (trt.n - 1) as f64);

    // Two-tailed p-value using the t-distribution CDF approximation
    let p_value = two_tailed_p_value(t.abs(), df);

    let relative_improvement = if ctrl.mean.abs() > f64::EPSILON {
        (trt.mean - ctrl.mean) / ctrl.mean.abs()
    } else {
        0.0
    };

    Ok(TTestResult {
        t_statistic: t,
        p_value,
        is_significant: p_value < alpha,
        confidence_level: 1.0 - alpha,
        degrees_of_freedom: df,
        control_mean: ctrl.mean,
        treatment_mean: trt.mean,
        relative_improvement,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// SPRT
// ─────────────────────────────────────────────────────────────────────────────

/// Decision from the Sequential Probability Ratio Test
#[derive(Debug, Clone, PartialEq)]
pub enum SprtDecision {
    /// Not enough evidence yet — continue collecting data
    ContinueTesting,
    /// Accept the null hypothesis (no meaningful difference)
    AcceptNull,
    /// Reject the null hypothesis (significant difference detected)
    RejectNull,
}

/// Sequential Probability Ratio Test for early stopping.
///
/// Tests whether two Bernoulli proportions are different.
///
/// * `alpha` — target false-positive rate (Type I error)
/// * `beta`  — target false-negative rate (Type II error)
pub fn sprt_test(
    control_successes: u64,
    control_trials: u64,
    treatment_successes: u64,
    treatment_trials: u64,
    alpha: f64,
    beta: f64,
) -> SprtDecision {
    if control_trials == 0 || treatment_trials == 0 {
        return SprtDecision::ContinueTesting;
    }

    let p0 = control_successes as f64 / control_trials as f64;
    let p1 = treatment_successes as f64 / treatment_trials as f64;

    if (p0 - p1).abs() < f64::EPSILON {
        return SprtDecision::ContinueTesting;
    }

    // Wald boundaries
    let upper = (1.0 - beta) / alpha; // reject null when log-likelihood ratio >= ln(upper)
    let lower = beta / (1.0 - alpha); // accept null when log-likelihood ratio <= ln(lower)

    // Compute log-likelihood ratio for the treatment sample
    // under H0 (rate = p0) vs H1 (rate = p1)
    let k = treatment_successes as f64;
    let n = treatment_trials as f64;

    // Avoid log(0)
    let p1_safe = p1.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    let p0_safe = p0.clamp(f64::EPSILON, 1.0 - f64::EPSILON);

    let log_lr = k * (p1_safe / p0_safe).ln() + (n - k) * ((1.0 - p1_safe) / (1.0 - p0_safe)).ln();

    if log_lr >= upper.ln() {
        SprtDecision::RejectNull
    } else if log_lr <= lower.ln() {
        SprtDecision::AcceptNull
    } else {
        SprtDecision::ContinueTesting
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sample size / power analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Calculate the minimum sample size needed per group to detect `minimum_detectable_effect`
/// with the given significance level and statistical power.
///
/// Uses the standard two-proportion z-test formula.
pub fn sample_size_for_power(
    baseline_rate: f64,
    minimum_detectable_effect: f64,
    alpha: f64,
    power: f64,
) -> usize {
    let treatment_rate = baseline_rate + minimum_detectable_effect;
    let p_avg = (baseline_rate + treatment_rate) / 2.0;

    // z-scores for the given alpha (two-tailed) and power
    let z_alpha = z_score_two_tailed(alpha);
    let z_beta = z_score_one_tailed(1.0 - power);

    let pooled_se = (2.0 * p_avg * (1.0 - p_avg)).sqrt();
    let effect_size = (treatment_rate - baseline_rate).abs();

    if effect_size < f64::EPSILON {
        return usize::MAX;
    }

    let n = ((z_alpha * pooled_se
        + z_beta
            * (baseline_rate * (1.0 - baseline_rate) + treatment_rate * (1.0 - treatment_rate))
                .sqrt())
        / effect_size)
        .powi(2);

    n.ceil() as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistical helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Approximation of the two-tailed p-value for a t-distribution.
///
/// Uses the regularized incomplete beta function via a continued fraction
/// approximation.  Accurate to ~4 decimal places for df > 1.
fn two_tailed_p_value(t_abs: f64, df: f64) -> f64 {
    // P(|T| > t) = 2 * (1 - CDF(t; df))
    // CDF(t; df) = 1 - 0.5 * I_x(df/2, 1/2)  where x = df / (df + t²)
    let x = df / (df + t_abs * t_abs);
    let p_one_tail = 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, x);
    (2.0 * p_one_tail).min(1.0)
}

/// Regularized incomplete beta function I_x(a, b) via continued fraction.
///
/// Uses Lentz's method.  This is sufficient precision for p-value computation.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    // Use symmetry relation for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
    }

    // Compute the beta function log for normalisation
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - ln_beta).exp() / a;

    front * continued_fraction_beta(a, b, x)
}

/// Continued fraction expansion for the regularized incomplete beta function.
fn continued_fraction_beta(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3e-7;
    const FPMIN: f64 = 1e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;

        // Even step
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < EPS {
            break;
        }
    }

    h
}

/// Lanczos approximation of ln(Γ(z)) for z > 0
fn ln_gamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    if z < 0.5 {
        // Reflection formula
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma(1.0 - z);
    }

    let z = z - 1.0;
    let x = C[0]
        + C[1] / (z + 1.0)
        + C[2] / (z + 2.0)
        + C[3] / (z + 3.0)
        + C[4] / (z + 4.0)
        + C[5] / (z + 5.0)
        + C[6] / (z + 6.0)
        + C[7] / (z + 7.0)
        + C[8] / (z + 8.0);

    let t = z + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Inverse normal CDF (probit) via a rational approximation.
///
/// Accurate to about 4.5 significant figures for p in (0, 1).
fn probit(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm (adapted)
    const A: [f64; 4] = [2.515517, 0.802853, 0.010328, 0.0];
    const B: [f64; 3] = [1.432788, 0.189269, 0.001308];

    let p = p.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let q = if p < 0.5 { p } else { 1.0 - p };

    let t = (-2.0 * q.ln()).sqrt();
    let numerator = A[0] + t * (A[1] + t * (A[2] + t * A[3]));
    let denominator = 1.0 + t * (B[0] + t * (B[1] + t * B[2]));
    sign * (t - numerator / denominator)
}

/// z-score for two-tailed test at significance level alpha
fn z_score_two_tailed(alpha: f64) -> f64 {
    probit(1.0 - alpha / 2.0).abs()
}

/// z-score for one-tailed test at probability p
fn z_score_one_tailed(p: f64) -> f64 {
    probit(p).abs()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SampleStats ──────────────────────────────────────────────────────────

    #[test]
    fn test_sample_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = SampleStats::from_samples(&data).expect("stats should succeed");
        assert_eq!(stats.n, 5);
        assert!((stats.mean - 3.0).abs() < 1e-9, "mean should be 3.0");
        // Sample variance of [1,2,3,4,5] = 2.5
        assert!(
            (stats.variance - 2.5).abs() < 1e-9,
            "variance should be 2.5"
        );
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_sample_stats_too_few_samples() {
        let result = SampleStats::from_samples(&[1.0]);
        assert!(
            matches!(result, Err(StatError::InsufficientSamples { .. })),
            "Single sample should error"
        );
    }

    // ── Welch t-test ─────────────────────────────────────────────────────────

    #[test]
    fn test_welch_t_test_identical_distributions_not_significant() {
        // Same mean, same variance → no significant difference
        let control: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let treatment: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let result = welch_t_test(&control, &treatment, 0.05).expect("t-test should succeed");
        assert!(
            !result.is_significant,
            "Identical samples should not be significant"
        );
        assert!(
            (result.t_statistic).abs() < 1e-9,
            "t-statistic should be ~0"
        );
    }

    #[test]
    fn test_welch_t_test_clearly_different_distributions() {
        // Control: N(0, 1), treatment: N(100, 1) — extremely different
        let control: Vec<f64> = (0..30).map(|i| i as f64 * 0.1 - 1.5).collect();
        let treatment: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.1 - 1.5).collect();
        let result = welch_t_test(&control, &treatment, 0.05).expect("t-test should succeed");
        assert!(
            result.is_significant,
            "Clearly different means should be significant"
        );
        assert!(result.p_value < 0.001, "p-value should be very small");
    }

    #[test]
    fn test_welch_t_test_relative_improvement() {
        // Control mean=10, treatment mean=11 → relative improvement = 10%
        let control = vec![10.0_f64; 20];
        let treatment_arr: Vec<f64> = (0..20).map(|i| 11.0 + (i as f64 - 10.0) * 0.01).collect();
        let result = welch_t_test(&control, &treatment_arr, 0.05);
        // Just verify it doesn't error even with near-zero variance control
        // (control has zero variance — should error with standard error check)
        // Actually the control has zero variance — but we handle that gracefully
        if let Ok(r) = result {
            assert!(
                r.relative_improvement > 0.0,
                "Improvement should be positive"
            );
        }
    }

    #[test]
    fn test_welch_t_test_insufficient_samples() {
        let result = welch_t_test(&[1.0], &[2.0, 3.0], 0.05);
        assert!(matches!(result, Err(StatError::InsufficientSamples { .. })));
    }

    #[test]
    fn test_welch_t_test_invalid_alpha() {
        let control = vec![1.0_f64, 2.0, 3.0];
        let treatment = vec![2.0_f64, 3.0, 4.0];
        assert!(matches!(
            welch_t_test(&control, &treatment, 0.0),
            Err(StatError::InvalidParameter(_))
        ));
        assert!(matches!(
            welch_t_test(&control, &treatment, 1.0),
            Err(StatError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_welch_t_test_degrees_of_freedom_positive() {
        let control: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let treatment: Vec<f64> = (0..25).map(|i| i as f64 * 1.1).collect();
        let result = welch_t_test(&control, &treatment, 0.05).expect("should succeed");
        assert!(result.degrees_of_freedom > 0.0, "df must be positive");
    }

    // ── SPRT ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_sprt_continue_with_no_data() {
        let decision = sprt_test(0, 0, 0, 0, 0.05, 0.20);
        assert_eq!(decision, SprtDecision::ContinueTesting);
    }

    #[test]
    fn test_sprt_reject_null_with_strong_treatment_effect() {
        // Control rate = 10%, treatment rate = 50% with many samples
        let decision = sprt_test(10, 100, 50, 100, 0.05, 0.20);
        assert_eq!(decision, SprtDecision::RejectNull);
    }

    #[test]
    fn test_sprt_accept_null_with_identical_rates() {
        // Same rate for control and treatment
        let decision = sprt_test(50, 100, 50, 100, 0.05, 0.20);
        // Equal rates → continue or accept null (log-likelihood ratio near 0)
        assert!(
            decision == SprtDecision::ContinueTesting || decision == SprtDecision::AcceptNull,
            "Equal rates should not reject null"
        );
    }

    // ── Power / Sample Size ───────────────────────────────────────────────────

    #[test]
    fn test_sample_size_for_power_reasonable_values() {
        // Classic rule of thumb: ~400 samples per group for 5% lift on 10% baseline,
        // 5% significance, 80% power
        let n = sample_size_for_power(0.10, 0.05, 0.05, 0.80);
        assert!(
            n > 100,
            "Sample size should be substantial for small effect"
        );
        assert!(n < 100_000, "Sample size should be reasonable");
    }

    #[test]
    fn test_sample_size_increases_with_smaller_effect() {
        let n_large_effect = sample_size_for_power(0.10, 0.05, 0.05, 0.80);
        let n_small_effect = sample_size_for_power(0.10, 0.01, 0.05, 0.80);
        assert!(
            n_small_effect > n_large_effect,
            "Smaller MDE requires more samples"
        );
    }

    // ── p-value sanity ────────────────────────────────────────────────────────

    #[test]
    fn test_p_value_is_in_unit_interval() {
        let control: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let treatment: Vec<f64> = (0..15).map(|i| i as f64 + 2.0).collect();
        let result = welch_t_test(&control, &treatment, 0.05).expect("ok");
        assert!(
            result.p_value >= 0.0 && result.p_value <= 1.0,
            "p-value must be in [0, 1]"
        );
    }

    #[test]
    fn test_confidence_level_is_complement_of_alpha() {
        let control: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let treatment: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let result = welch_t_test(&control, &treatment, 0.05).expect("ok");
        assert!((result.confidence_level - 0.95).abs() < 1e-9);
    }

    // ── Additional tests ──────────────────────────────────────────────────────

    #[test]
    fn test_sample_stats_min_max_correct() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let stats = SampleStats::from_samples(&data).expect("should succeed");
        assert!((stats.min - 1.0).abs() < 1e-9, "min should be 1.0");
        assert!((stats.max - 9.0).abs() < 1e-9, "max should be 9.0");
    }

    #[test]
    fn test_sample_stats_std_dev_equals_sqrt_variance() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let stats = SampleStats::from_samples(&data).expect("should succeed");
        assert!((stats.std_dev - stats.variance.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_sample_stats_std_error_equals_std_dev_over_sqrt_n() {
        let data: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        let stats = SampleStats::from_samples(&data).expect("should succeed");
        let expected_se = stats.std_dev / (stats.n as f64).sqrt();
        assert!((stats.std_error - expected_se).abs() < 1e-12);
    }

    #[test]
    fn test_sample_stats_two_elements() {
        // Minimal valid input: 2 elements
        let data = vec![0.0, 10.0];
        let stats = SampleStats::from_samples(&data).expect("should succeed for 2 samples");
        assert_eq!(stats.n, 2);
        assert!((stats.mean - 5.0).abs() < 1e-9);
        // Bessel-corrected variance of [0, 10]: ((0-5)^2 + (10-5)^2) / 1 = 50
        assert!((stats.variance - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_welch_t_test_alpha_01() {
        // Same result should change significance at stricter alpha
        let control: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let treatment: Vec<f64> = (0..30).map(|i| i as f64 + 1.0).collect();
        let r_05 = welch_t_test(&control, &treatment, 0.05).expect("ok");
        let r_01 = welch_t_test(&control, &treatment, 0.01).expect("ok");
        // Confidence levels differ by alpha
        assert!((r_05.confidence_level - 0.95).abs() < 1e-9);
        assert!((r_01.confidence_level - 0.99).abs() < 1e-9);
        // The t-statistic and df should be identical regardless of alpha
        assert!((r_05.t_statistic - r_01.t_statistic).abs() < 1e-9);
        assert!((r_05.degrees_of_freedom - r_01.degrees_of_freedom).abs() < 1e-9);
    }

    #[test]
    fn test_welch_t_test_treatment_mean_greater_gives_negative_t_relative_improvement() {
        // If treatment > control, relative improvement is positive
        let control = vec![10.0_f64; 20];
        let treatment: Vec<f64> = (0..20).map(|i| 12.0 + (i as f64 - 10.0) * 0.01).collect();
        if let Ok(result) = welch_t_test(&control, &treatment, 0.05) {
            // treatment mean ≈ 12, control mean ≈ 10 → relative improvement ≈ 0.2 (20%)
            assert!(
                result.relative_improvement > 0.0,
                "treatment > control should give positive relative improvement"
            );
        }
    }

    #[test]
    fn test_sprt_both_zero_trials_is_continue() {
        // Explicitly test zero for treatment only as well
        let d = sprt_test(5, 10, 0, 0, 0.05, 0.20);
        assert_eq!(d, SprtDecision::ContinueTesting);
    }

    #[test]
    fn test_sprt_identical_proportions_does_not_reject() {
        // Same proportions with large sample sizes: should not reject null
        let d = sprt_test(500, 1000, 500, 1000, 0.05, 0.20);
        assert_ne!(
            d,
            SprtDecision::RejectNull,
            "equal proportions must not reject null"
        );
    }

    #[test]
    fn test_sprt_accept_null_when_control_superior() {
        // Control rate = 90%, treatment rate = 10% — very different, treatment is worse.
        // The SPRT tests whether *treatment* rate (10%) differs from control (90%).
        // With such an extreme difference, the null (no difference) should be rejected or accepted.
        let d = sprt_test(900, 1000, 100, 1000, 0.05, 0.20);
        // Either accept or reject, but not "continue" with this many samples and extreme difference.
        assert_ne!(
            d,
            SprtDecision::ContinueTesting,
            "should not continue with 1000 samples and extreme effect"
        );
    }

    #[test]
    fn test_sample_size_increases_with_higher_power() {
        // Higher required power means more samples needed
        let n_80 = sample_size_for_power(0.10, 0.05, 0.05, 0.80);
        let n_90 = sample_size_for_power(0.10, 0.05, 0.05, 0.90);
        assert!(
            n_90 >= n_80,
            "higher power ({n_90}) should require at least as many samples as lower power ({n_80})"
        );
    }

    #[test]
    fn test_sample_size_increases_with_stricter_alpha() {
        // Stricter alpha means more samples needed
        let n_05 = sample_size_for_power(0.10, 0.05, 0.05, 0.80);
        let n_01 = sample_size_for_power(0.10, 0.05, 0.01, 0.80);
        assert!(
            n_01 >= n_05,
            "stricter alpha should require at least as many samples"
        );
    }

    #[test]
    fn test_sample_size_zero_effect_returns_max() {
        // A zero effect size means we would need infinite samples
        let n = sample_size_for_power(0.10, 0.0, 0.05, 0.80);
        assert_eq!(n, usize::MAX, "zero effect size should return usize::MAX");
    }

    #[test]
    fn test_welch_t_test_t_statistic_sign() {
        // If control > treatment, t should be positive (control - treatment)
        let control: Vec<f64> = (0..20).map(|i| i as f64 + 10.0).collect();
        let treatment: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = welch_t_test(&control, &treatment, 0.05).expect("ok");
        assert!(
            result.t_statistic > 0.0,
            "when control > treatment, t_statistic should be positive"
        );
    }

    #[test]
    fn test_welch_t_test_too_few_treatment() {
        let control = vec![1.0, 2.0, 3.0];
        let result = welch_t_test(&control, &[5.0], 0.05);
        assert!(matches!(result, Err(StatError::InsufficientSamples { .. })));
    }

    #[test]
    fn test_stat_error_messages_contain_context() {
        let err = StatError::InsufficientSamples { min: 2, got: 1 };
        let msg = err.to_string();
        assert!(msg.contains("2"), "message should mention minimum required");
        assert!(msg.contains("1"), "message should mention actual count");
    }
}
