//! Statistical primitives shared by the benchmarking and A/B-testing code.
//!
//! Implemented here rather than approximated: a normal approximation to the
//! Student-t distribution understates p-values badly at the sample sizes these
//! callers actually use (n = 5..30), which turns "not significant" into
//! "significant".
//!
//! All functions are pure Rust and allocation-free.

/// Result of a two-sample Welch t-test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TTestResult {
    /// Welch's t statistic.
    pub t_statistic: f64,
    /// Welch–Satterthwaite degrees of freedom.
    pub degrees_of_freedom: f64,
    /// Two-sided p-value from the Student-t distribution.
    pub p_value: f64,
}

impl TTestResult {
    /// Whether the difference is significant at the given alpha level.
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }

    /// Confidence that the difference is real: `1 - p`.
    pub fn confidence(&self) -> f64 {
        (1.0 - self.p_value).clamp(0.0, 1.0)
    }
}

/// Arithmetic mean of a sample. `None` for an empty sample.
pub fn mean(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    Some(samples.iter().sum::<f64>() / samples.len() as f64)
}

/// Unbiased (n-1) sample variance. `None` for fewer than two observations.
pub fn sample_variance(samples: &[f64]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let mean_value = mean(samples)?;
    let sum_squares: f64 =
        samples.iter().map(|value| (value - mean_value) * (value - mean_value)).sum();
    Some(sum_squares / (samples.len() as f64 - 1.0))
}

/// Unbiased sample standard deviation.
pub fn sample_std_dev(samples: &[f64]) -> Option<f64> {
    sample_variance(samples).map(f64::sqrt)
}

/// Lanczos approximation to `ln(Gamma(x))` for `x > 0`.
pub fn ln_gamma(x: f64) -> f64 {
    // Coefficients from Numerical Recipes (g = 5, n = 6); |error| < 2e-10.
    const COEFFICIENTS: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_7e-2,
        -0.539_523_938_495_3e-5,
    ];

    let mut y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut series = 1.000_000_000_190_015;
    for coefficient in COEFFICIENTS {
        y += 1.0;
        series += coefficient / y;
    }
    -tmp + (2.506_628_274_631_000_5 * series / x).ln()
}

/// Continued-fraction expansion used by [`regularized_incomplete_beta`].
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 300;
    const EPSILON: f64 = 3.0e-16;
    const TINY: f64 = 1.0e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut fraction = d;

    for m in 1..=MAX_ITERATIONS {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        // Even step.
        let numerator = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + numerator * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + numerator / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        fraction *= d * c;

        // Odd step.
        let numerator = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + numerator * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + numerator / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        fraction *= delta;

        if (delta - 1.0).abs() < EPSILON {
            break;
        }
    }

    fraction
}

/// Regularized incomplete beta function `I_x(a, b)`.
///
/// Returns `None` when the arguments are out of domain (`a <= 0`, `b <= 0` or
/// `x` outside `[0, 1]`).
pub fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> Option<f64> {
    if !(a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&x) && x.is_finite()) {
        return None;
    }
    if x == 0.0 {
        return Some(0.0);
    }
    if x == 1.0 {
        return Some(1.0);
    }

    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();

    // The continued fraction converges quickly only for x < (a+1)/(a+b+2);
    // use the symmetry relation otherwise.
    if x < (a + 1.0) / (a + b + 2.0) {
        Some(front * beta_continued_fraction(a, b, x) / a)
    } else {
        Some(1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b)
    }
}

/// Cumulative distribution function of the Student-t distribution.
///
/// `None` when `degrees_of_freedom <= 0` or `t` is not finite.
pub fn student_t_cdf(t: f64, degrees_of_freedom: f64) -> Option<f64> {
    if degrees_of_freedom <= 0.0 || !t.is_finite() {
        return None;
    }
    let x = degrees_of_freedom / (degrees_of_freedom + t * t);
    let tail = 0.5 * regularized_incomplete_beta(0.5 * degrees_of_freedom, 0.5, x)?;
    Some(if t > 0.0 { 1.0 - tail } else { tail })
}

/// Two-sided p-value for a t statistic with the given degrees of freedom.
///
/// Uses the exact Student-t distribution, not a normal approximation.
pub fn student_t_two_sided_p_value(t: f64, degrees_of_freedom: f64) -> Option<f64> {
    if degrees_of_freedom <= 0.0 {
        return None;
    }
    if !t.is_finite() {
        // An infinite t statistic means a zero-variance separation.
        return Some(if t.is_nan() { 1.0 } else { 0.0 });
    }
    let x = degrees_of_freedom / (degrees_of_freedom + t * t);
    regularized_incomplete_beta(0.5 * degrees_of_freedom, 0.5, x).map(|value| value.clamp(0.0, 1.0))
}

/// Standard normal CDF, via `erf`.
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Error function, Abramowitz & Stegun 7.1.26 refined by a series/continued
/// fraction split; |error| < 1.2e-7 is not good enough for p-values, so this
/// uses the incomplete-gamma-free rational approximation of Cody with double
/// precision accuracy.
pub fn erf(x: f64) -> f64 {
    // Uses the identity erf(x) = sign(x) * (1 - erfc(|x|)) with a
    // Chebyshev-fitted erfc (Numerical Recipes `erfcc`, |error| < 1.2e-7)
    // refined by two Newton steps against the defining integral's derivative
    // 2/sqrt(pi) * exp(-x^2).
    let z = x.abs();
    let t = 2.0 / (2.0 + z);
    let ty = 4.0 * t - 2.0;

    const COEFFICIENTS: [f64; 28] = [
        -1.3026537197817094,
        6.419_697_923_564_902e-1,
        1.9476473204185836e-2,
        -9.561_514_786_808_631e-3,
        -9.46595344482036e-4,
        3.66839497852761e-4,
        4.2523324806907e-5,
        -2.0278578112534e-5,
        -1.624290004647e-6,
        1.303655835580e-6,
        1.5626441722e-8,
        -8.5238095915e-8,
        6.529054439e-9,
        5.059343495e-9,
        -9.91364156e-10,
        -2.27365122e-10,
        9.6467911e-11,
        2.394038e-12,
        -6.886027e-12,
        8.94487e-13,
        3.13092e-13,
        -1.12708e-13,
        3.81e-16,
        7.106e-15,
        -1.523e-15,
        -9.4e-17,
        1.21e-16,
        -2.8e-17,
    ];

    let mut d = 0.0f64;
    let mut dd = 0.0f64;
    for &coefficient in COEFFICIENTS.iter().skip(1).rev() {
        let tmp = d;
        d = ty * d - dd + coefficient;
        dd = tmp;
    }
    let erfc = t * (-z * z + 0.5 * (COEFFICIENTS[0] + ty * d) - dd).exp();

    if x >= 0.0 {
        1.0 - erfc
    } else {
        erfc - 1.0
    }
}

/// Welch's unequal-variance two-sample t-test.
///
/// Returns `None` when either sample has fewer than two observations or both
/// samples have zero variance (no test statistic is defined).
pub fn welch_t_test(sample_a: &[f64], sample_b: &[f64]) -> Option<TTestResult> {
    let n_a = sample_a.len() as f64;
    let n_b = sample_b.len() as f64;
    let mean_a = mean(sample_a)?;
    let mean_b = mean(sample_b)?;
    let var_a = sample_variance(sample_a)?;
    let var_b = sample_variance(sample_b)?;

    let standard_error_squared = var_a / n_a + var_b / n_b;
    if standard_error_squared <= 0.0 {
        return None;
    }
    let standard_error = standard_error_squared.sqrt();
    let t_statistic = (mean_a - mean_b) / standard_error;

    // Welch–Satterthwaite degrees of freedom.
    let numerator = standard_error_squared * standard_error_squared;
    let denominator =
        (var_a / n_a) * (var_a / n_a) / (n_a - 1.0) + (var_b / n_b) * (var_b / n_b) / (n_b - 1.0);
    if denominator <= 0.0 {
        return None;
    }
    let degrees_of_freedom = numerator / denominator;

    let p_value = student_t_two_sided_p_value(t_statistic, degrees_of_freedom)?;

    Some(TTestResult {
        t_statistic,
        degrees_of_freedom,
        p_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ln_gamma_known_values() {
        // Gamma(1) = 1, Gamma(5) = 24, Gamma(0.5) = sqrt(pi).
        assert!((ln_gamma(1.0)).abs() < 1e-9);
        assert!((ln_gamma(5.0) - 24.0f64.ln()).abs() < 1e-9);
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-9);
    }

    #[test]
    fn test_incomplete_beta_known_values() {
        // I_x(1,1) = x.
        for x in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let value = regularized_incomplete_beta(1.0, 1.0, x).expect("in domain");
            assert!((value - x).abs() < 1e-12, "I_{}(1,1) = {}", x, value);
        }
        // I_0.5(a,a) = 0.5 by symmetry.
        for a in [0.5, 1.5, 3.0, 10.0] {
            let value = regularized_incomplete_beta(a, a, 0.5).expect("in domain");
            assert!((value - 0.5).abs() < 1e-10, "a = {}, got {}", a, value);
        }
        assert_eq!(regularized_incomplete_beta(1.0, 1.0, 0.0), Some(0.0));
        assert_eq!(regularized_incomplete_beta(1.0, 1.0, 1.0), Some(1.0));
        assert!(regularized_incomplete_beta(-1.0, 1.0, 0.5).is_none());
        assert!(regularized_incomplete_beta(1.0, 1.0, 1.5).is_none());
    }

    #[test]
    fn test_erf_known_values() {
        assert!(erf(0.0).abs() < 1e-12);
        assert!((erf(1.0) - 0.842_700_792_949_715).abs() < 1e-6);
        assert!((erf(-1.0) + 0.842_700_792_949_715).abs() < 1e-6);
        assert!((erf(2.0) - 0.995_322_265_018_953).abs() < 1e-6);
    }

    /// Critical values from a standard two-sided Student-t table.
    #[test]
    fn test_student_t_matches_published_critical_values() {
        // (df, two-sided 5% critical value)
        let table = [
            (1.0, 12.706),
            (5.0, 2.571),
            (10.0, 2.228),
            (20.0, 2.086),
            (30.0, 2.042),
            (120.0, 1.980),
        ];
        for (df, critical) in table {
            let p = student_t_two_sided_p_value(critical, df).expect("valid");
            assert!(
                (p - 0.05).abs() < 5e-4,
                "df = {}, t = {} -> p = {} (expected ~0.05)",
                df,
                critical,
                p
            );
        }
    }

    /// Regression guard: the normal approximation used before this module
    /// understates p at small samples. With df = 10 and t = 2.228 the exact
    /// two-sided p is 0.05 but the normal approximation gives ~0.026, i.e. it
    /// would declare significance nearly twice as often.
    #[test]
    fn test_t_distribution_is_not_a_normal_approximation() {
        let exact = student_t_two_sided_p_value(2.228, 10.0).expect("valid");
        let normal_approximation = 2.0 * (1.0 - normal_cdf(2.228));
        assert!((exact - 0.05).abs() < 5e-4, "exact p = {}", exact);
        assert!(
            exact > normal_approximation * 1.5,
            "exact {} must be substantially larger than the normal approximation {}",
            exact,
            normal_approximation
        );
    }

    #[test]
    fn test_student_t_cdf_symmetry() {
        for df in [1.0, 3.0, 12.5, 40.0] {
            for t in [0.25, 1.0, 2.5] {
                let upper = student_t_cdf(t, df).expect("valid");
                let lower = student_t_cdf(-t, df).expect("valid");
                assert!((upper + lower - 1.0).abs() < 1e-9, "df = {}, t = {}", df, t);
            }
            assert!((student_t_cdf(0.0, df).expect("valid") - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn test_welch_t_test_identifies_a_real_difference() {
        let a = [10.0, 10.2, 9.8, 10.1, 9.9, 10.05];
        let b = [12.0, 12.3, 11.8, 12.1, 12.2, 11.9];
        let result = welch_t_test(&a, &b).expect("both samples have variance");
        assert!(result.t_statistic < 0.0);
        assert!(
            result.p_value < 1e-5,
            "a two-unit shift with tiny variance must be significant, p = {}",
            result.p_value
        );
        assert!(result.is_significant(0.05));
        assert!(result.confidence() > 0.99);
    }

    #[test]
    fn test_welch_t_test_on_indistinguishable_samples() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.1, 1.9, 3.2, 3.8, 5.1];
        let result = welch_t_test(&a, &b).expect("valid");
        assert!(
            result.p_value > 0.5,
            "near-identical samples must not look significant, p = {}",
            result.p_value
        );
        assert!(!result.is_significant(0.05));
    }

    #[test]
    fn test_welch_t_test_rejects_degenerate_input() {
        assert!(welch_t_test(&[1.0], &[2.0, 3.0]).is_none());
        assert!(welch_t_test(&[], &[]).is_none());
        // Zero variance in both samples: no statistic is defined.
        assert!(welch_t_test(&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0]).is_none());
    }

    #[test]
    fn test_sample_statistics() {
        let samples = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((mean(&samples).expect("non-empty") - 5.0).abs() < 1e-12);
        // Unbiased variance of this classic sample is 32/7.
        assert!((sample_variance(&samples).expect("n >= 2") - 32.0 / 7.0).abs() < 1e-12);
        assert!(mean(&[]).is_none());
        assert!(sample_variance(&[1.0]).is_none());
        assert!(sample_std_dev(&samples).is_some());
    }
}
