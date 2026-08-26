//! Statistical primitives for the testing framework.
//!
//! Pure-Rust implementations of the distributions the fairness audit needs:
//! the regularized incomplete gamma function (and therefore the chi-square
//! survival function), the normal CDF, and the two-proportion z-test. No
//! constants are hard-coded — every value returned here is computed from the
//! caller's data.
//!
//! The series / continued-fraction split for the incomplete gamma function
//! follows the classical formulation (Press et al., *Numerical Recipes*,
//! §6.2): the power series converges quickly for `x < a + 1`, the Lentz
//! continued fraction elsewhere.

use anyhow::{anyhow, Result};

/// Absolute convergence tolerance for the iterative expansions.
const EPSILON: f64 = 1e-14;
/// Iteration cap for the series / continued fraction.
const MAX_ITERATIONS: usize = 500;
/// Smallest positive number used to prime the Lentz continued fraction.
const TINY: f64 = 1e-300;

/// Natural logarithm of the gamma function (Lanczos approximation, g = 7).
///
/// Accurate to ~15 significant digits for `x > 0`.
pub fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_1,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Γ(x)Γ(1-x) = π / sin(πx)
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEFFICIENTS[0];
        let t = x + 7.5;
        for (i, &coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
            a += coefficient / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Regularized lower incomplete gamma function `P(a, x)`.
///
/// # Errors
///
/// Returns an error for `a <= 0` or `x < 0`, and when the expansion fails to
/// converge (which would otherwise silently return a wrong probability).
pub fn regularized_gamma_p(a: f64, x: f64) -> Result<f64> {
    if a <= 0.0 {
        return Err(anyhow!("regularized_gamma_p requires a > 0, got {a}"));
    }
    if x < 0.0 {
        return Err(anyhow!("regularized_gamma_p requires x >= 0, got {x}"));
    }
    if x == 0.0 {
        return Ok(0.0);
    }

    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        Ok(1.0 - gamma_continued_fraction(a, x)?)
    }
}

/// Regularized upper incomplete gamma function `Q(a, x) = 1 - P(a, x)`.
pub fn regularized_gamma_q(a: f64, x: f64) -> Result<f64> {
    Ok(1.0 - regularized_gamma_p(a, x)?)
}

/// Series expansion of `P(a, x)`, used for `x < a + 1`.
fn gamma_series(a: f64, x: f64) -> Result<f64> {
    let ln_gamma_a = ln_gamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut delta = sum;

    for _ in 0..MAX_ITERATIONS {
        ap += 1.0;
        delta *= x / ap;
        sum += delta;
        if delta.abs() < sum.abs() * EPSILON {
            return Ok(sum * (-x + a * x.ln() - ln_gamma_a).exp());
        }
    }

    Err(anyhow!(
        "the incomplete gamma series did not converge for a={a}, x={x}"
    ))
}

/// Lentz continued fraction for `Q(a, x)`, used for `x >= a + 1`.
fn gamma_continued_fraction(a: f64, x: f64) -> Result<f64> {
    let ln_gamma_a = ln_gamma(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=MAX_ITERATIONS {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPSILON {
            return Ok(h * (-x + a * x.ln() - ln_gamma_a).exp());
        }
    }

    Err(anyhow!(
        "the incomplete gamma continued fraction did not converge for a={a}, x={x}"
    ))
}

/// Survival function of the chi-square distribution: `P(X² >= statistic)`.
///
/// This is the p-value of a chi-square test with `degrees_of_freedom` degrees of
/// freedom.
pub fn chi_square_p_value(statistic: f64, degrees_of_freedom: usize) -> Result<f64> {
    if degrees_of_freedom == 0 {
        return Err(anyhow!(
            "a chi-square test needs at least one degree of freedom"
        ));
    }
    if statistic < 0.0 {
        return Err(anyhow!(
            "a chi-square statistic cannot be negative, got {statistic}"
        ));
    }
    if statistic == 0.0 {
        return Ok(1.0);
    }
    Ok(regularized_gamma_q(degrees_of_freedom as f64 / 2.0, statistic / 2.0)?.clamp(0.0, 1.0))
}

/// Result of a chi-square test of independence on a contingency table.
#[derive(Debug, Clone, PartialEq)]
pub struct ChiSquareResult {
    /// Pearson's chi-square statistic.
    pub statistic: f64,
    /// Degrees of freedom, `(rows - 1) * (columns - 1)`.
    pub degrees_of_freedom: usize,
    /// Probability of a statistic at least this extreme under independence.
    pub p_value: f64,
    /// Smallest expected cell count (the usual validity check: should be >= 5).
    pub min_expected_count: f64,
}

/// Pearson's chi-square test of independence over a contingency table.
///
/// `table[row][column]` holds observed counts. Rows are typically the protected
/// groups and columns the model's decisions.
///
/// # Errors
///
/// Returns an error for degenerate tables (fewer than two rows or columns,
/// ragged rows, negative counts, an empty table, or a row/column that is
/// entirely zero — the expected counts would be zero and the statistic
/// undefined).
pub fn chi_square_test(table: &[Vec<f64>]) -> Result<ChiSquareResult> {
    let rows = table.len();
    if rows < 2 {
        return Err(anyhow!(
            "a chi-square test of independence needs at least two rows, got {rows}"
        ));
    }
    let columns = table[0].len();
    if columns < 2 {
        return Err(anyhow!(
            "a chi-square test of independence needs at least two columns, got {columns}"
        ));
    }
    if table.iter().any(|row| row.len() != columns) {
        return Err(anyhow!("the contingency table is ragged"));
    }
    if table.iter().flatten().any(|&count| count < 0.0 || !count.is_finite()) {
        return Err(anyhow!(
            "contingency counts must be finite and non-negative"
        ));
    }

    let total: f64 = table.iter().flatten().sum();
    if total <= 0.0 {
        return Err(anyhow!("the contingency table is empty"));
    }

    let row_totals: Vec<f64> = table.iter().map(|row| row.iter().sum()).collect();
    let column_totals: Vec<f64> =
        (0..columns).map(|c| table.iter().map(|row| row[c]).sum()).collect();

    if row_totals.iter().any(|&t| t <= 0.0) || column_totals.iter().any(|&t| t <= 0.0) {
        return Err(anyhow!(
            "every row and column of the contingency table must contain at least one observation"
        ));
    }

    let mut statistic = 0.0;
    let mut min_expected = f64::INFINITY;
    for (r, row) in table.iter().enumerate() {
        for (c, &observed) in row.iter().enumerate() {
            let expected = row_totals[r] * column_totals[c] / total;
            min_expected = min_expected.min(expected);
            let difference = observed - expected;
            statistic += difference * difference / expected;
        }
    }

    let degrees_of_freedom = (rows - 1) * (columns - 1);
    let p_value = chi_square_p_value(statistic, degrees_of_freedom)?;

    Ok(ChiSquareResult {
        statistic,
        degrees_of_freedom,
        p_value,
        min_expected_count: min_expected,
    })
}

/// Error function, evaluated through the regularized incomplete gamma function.
///
/// `erf(x) = sign(x) · P(1/2, x²)`, which reuses the series / continued-fraction
/// machinery above and is therefore accurate to full double precision instead of
/// the ~1e-7 of the usual Chebyshev fits.
pub fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    match regularized_gamma_p(0.5, x * x) {
        Ok(p) => {
            if x > 0.0 {
                p
            } else {
                -p
            }
        },
        // The expansion only fails for non-finite inputs, where the limits are exact.
        Err(_) => {
            if x > 0.0 {
                1.0
            } else {
                -1.0
            }
        },
    }
}

/// Complementary error function, `1 - erf(x)`.
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Cumulative distribution function of the standard normal distribution.
pub fn normal_cdf(z: f64) -> f64 {
    if !z.is_finite() {
        return if z > 0.0 { 1.0 } else { 0.0 };
    }
    (0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))).clamp(0.0, 1.0)
}

/// Two-sided p-value for a standard-normal test statistic.
pub fn normal_two_sided_p_value(z: f64) -> f64 {
    (2.0 * (1.0 - normal_cdf(z.abs()))).clamp(0.0, 1.0)
}

/// Result of a two-proportion z-test.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoProportionTest {
    /// Difference of the two sample proportions, `p1 - p2`.
    pub difference: f64,
    /// z statistic under the pooled-variance null hypothesis.
    pub z: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Wald confidence interval for the difference at the requested level.
    pub confidence_interval: (f64, f64),
}

/// Two-proportion z-test with a Wald confidence interval for the difference.
///
/// `successes_a` of `total_a` and `successes_b` of `total_b`. `confidence_level`
/// is e.g. `0.95`.
///
/// # Errors
///
/// Returns an error when a sample is empty, when the successes exceed the
/// sample size, or when the confidence level is not in `(0, 1)`.
pub fn two_proportion_z_test(
    successes_a: f64,
    total_a: f64,
    successes_b: f64,
    total_b: f64,
    confidence_level: f64,
) -> Result<TwoProportionTest> {
    if total_a <= 0.0 || total_b <= 0.0 {
        return Err(anyhow!(
            "a two-proportion test needs a non-empty sample in both groups"
        ));
    }
    if successes_a < 0.0 || successes_b < 0.0 || successes_a > total_a || successes_b > total_b {
        return Err(anyhow!(
            "the number of successes must lie between 0 and the sample size"
        ));
    }
    if !(0.0..1.0).contains(&confidence_level) || confidence_level <= 0.0 {
        return Err(anyhow!(
            "the confidence level must lie strictly between 0 and 1, got {confidence_level}"
        ));
    }

    let p_a = successes_a / total_a;
    let p_b = successes_b / total_b;
    let difference = p_a - p_b;

    // Pooled proportion under H0: p_a == p_b.
    let pooled = (successes_a + successes_b) / (total_a + total_b);
    let pooled_variance = pooled * (1.0 - pooled) * (1.0 / total_a + 1.0 / total_b);

    let z = if pooled_variance > 0.0 { difference / pooled_variance.sqrt() } else { 0.0 };
    let p_value = if pooled_variance > 0.0 { normal_two_sided_p_value(z) } else { 1.0 };

    // Unpooled (Wald) standard error for the interval.
    let standard_error = (p_a * (1.0 - p_a) / total_a + p_b * (1.0 - p_b) / total_b).sqrt();
    let critical = normal_quantile(0.5 + confidence_level / 2.0);
    let margin = critical * standard_error;

    Ok(TwoProportionTest {
        difference,
        z,
        p_value,
        confidence_interval: (difference - margin, difference + margin),
    })
}

/// Quantile (inverse CDF) of the standard normal distribution.
///
/// Acklam's rational approximation refined with one Halley step, giving full
/// double precision.
pub fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    const P_LOW: f64 = 0.02425;
    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley refinement step against the exact CDF.
    let e = normal_cdf(x) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
    x - u / (1.0 + x * u / 2.0)
}

/// Expected calibration error of probabilistic predictions.
///
/// Predictions are bucketed into `bins` equal-width bins over `[0, 1]`; the ECE
/// is the sample-weighted mean absolute gap between the average predicted
/// probability and the observed positive rate in each bin.
///
/// # Errors
///
/// Returns an error when the inputs have different lengths, are empty, or when
/// `bins` is zero.
pub fn expected_calibration_error(predictions: &[f32], labels: &[i32], bins: usize) -> Result<f64> {
    if predictions.len() != labels.len() {
        return Err(anyhow!(
            "calibration needs one label per prediction, got {} predictions and {} labels",
            predictions.len(),
            labels.len()
        ));
    }
    if predictions.is_empty() {
        return Err(anyhow!("calibration needs at least one prediction"));
    }
    if bins == 0 {
        return Err(anyhow!("calibration needs at least one bin"));
    }

    let mut bin_counts = vec![0usize; bins];
    let mut bin_confidence = vec![0.0f64; bins];
    let mut bin_positives = vec![0.0f64; bins];

    for (&prediction, &label) in predictions.iter().zip(labels.iter()) {
        let probability = f64::from(prediction).clamp(0.0, 1.0);
        let index = ((probability * bins as f64).floor() as usize).min(bins - 1);
        bin_counts[index] += 1;
        bin_confidence[index] += probability;
        if label > 0 {
            bin_positives[index] += 1.0;
        }
    }

    let total = predictions.len() as f64;
    let mut ece = 0.0;
    for bin in 0..bins {
        if bin_counts[bin] == 0 {
            continue;
        }
        let count = bin_counts[bin] as f64;
        let average_confidence = bin_confidence[bin] / count;
        let observed_rate = bin_positives[bin] / count;
        ece += (count / total) * (average_confidence - observed_rate).abs();
    }

    Ok(ece)
}

/// Upper-tail critical value of the chi-square distribution.
///
/// Returns the `x` for which `P(X² >= x) = tail_probability` with the given
/// degrees of freedom, found by bisection on [`chi_square_p_value`]. Returns
/// `f64::NAN` for a degenerate request (zero degrees of freedom, or a tail
/// probability outside `(0, 1)`).
pub fn chi_square_quantile(degrees_of_freedom: usize, tail_probability: f64) -> f64 {
    if degrees_of_freedom == 0 || !(0.0..1.0).contains(&tail_probability) {
        return f64::NAN;
    }
    if tail_probability == 0.0 {
        return f64::INFINITY;
    }

    let mut low = 0.0f64;
    let mut high = 2.0f64.max(degrees_of_freedom as f64);
    // Expand until the upper bound is beyond the requested quantile.
    for _ in 0..200 {
        match chi_square_p_value(high, degrees_of_freedom) {
            Ok(p) if p > tail_probability => high *= 2.0,
            _ => break,
        }
    }

    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        match chi_square_p_value(mid, degrees_of_freedom) {
            Ok(p) if p > tail_probability => low = mid,
            Ok(_) => high = mid,
            Err(_) => return f64::NAN,
        }
    }

    0.5 * (low + high)
}
