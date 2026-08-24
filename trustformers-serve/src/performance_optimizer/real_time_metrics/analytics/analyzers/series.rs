//! Numeric primitives shared by the analytics analyzers.
//!
//! Everything in this module is a closed-form computation over data the caller
//! actually supplied. No function here substitutes a default when the input is
//! insufficient: the callers check length pre-conditions and return a
//! structured error naming what was missing.

use std::collections::HashMap;

use super::super::super::types::data_structures::TimestampedMetrics;

/// A named numeric series extracted from a window of timestamped metrics.
#[derive(Debug, Clone)]
pub struct MetricSeries {
    /// Name of the metric this series carries.
    pub name: &'static str,
    /// One observation per input sample, in input order.
    pub values: Vec<f64>,
}

/// The metric names every extraction produces, in a stable order.
pub const SERIES_NAMES: [&str; 7] = [
    "throughput",
    "latency_seconds",
    "cpu_utilization",
    "memory_utilization",
    "io_usage",
    "network_usage",
    "error_rate",
];

/// Extract the seven numeric series carried by `RealTimeMetrics`.
///
/// Every series has exactly `data.len()` entries so the series stay aligned for
/// correlation analysis. Non-finite readings are replaced by the last finite
/// value of the same series when one exists; if a series has no finite reading
/// at all it is returned empty and its consumers skip it.
pub fn extract_series(data: &[TimestampedMetrics]) -> Vec<MetricSeries> {
    let mut series: Vec<MetricSeries> = SERIES_NAMES
        .iter()
        .map(|name| MetricSeries {
            name: *name,
            values: Vec::with_capacity(data.len()),
        })
        .collect();

    for sample in data {
        let metrics = &sample.metrics;
        let raw = [
            metrics.throughput,
            metrics.latency.as_secs_f64(),
            metrics.resource_usage.cpu_usage as f64,
            metrics.resource_usage.memory_usage,
            metrics.resource_usage.io_usage as f64,
            metrics.resource_usage.network_usage as f64,
            metrics.error_rate as f64,
        ];
        for (slot, value) in series.iter_mut().zip(raw.iter()) {
            if value.is_finite() {
                slot.values.push(*value);
            } else if let Some(last) = slot.values.last().copied() {
                slot.values.push(last);
            }
        }
    }

    // A series that never saw a finite reading is shorter than the window; drop
    // its partial contents rather than pretend it is aligned with the others.
    for slot in series.iter_mut() {
        if slot.values.len() != data.len() {
            slot.values.clear();
        }
    }

    series
}

/// Arithmetic mean. Returns `None` for an empty slice.
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Unbiased (n-1) sample variance. Returns `None` for fewer than two values.
pub fn sample_variance(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let m = mean(values)?;
    let n = values.len() as f64;
    Some(values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0))
}

/// Unbiased sample standard deviation.
pub fn sample_std_dev(values: &[f64]) -> Option<f64> {
    sample_variance(values).map(f64::sqrt)
}

/// Sort a copy of `values` ascending, dropping non-finite entries.
pub fn sorted_finite(values: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Linear-interpolated percentile of an ascending, non-empty slice.
///
/// `p` is a fraction in `[0, 1]`. Returns `None` for an empty slice.
pub fn percentile_sorted(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    if sorted.len() == 1 {
        return sorted.first().copied();
    }
    let clamped = p.clamp(0.0, 1.0);
    let rank = clamped * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let lo = *sorted.get(lower)?;
    let hi = *sorted.get(upper)?;
    if lower == upper {
        Some(lo)
    } else {
        Some(lo + (hi - lo) * (rank - lower as f64))
    }
}

/// Sample skewness (third standardised moment, `1/n` convention).
pub fn skewness(values: &[f64]) -> Option<f64> {
    if values.len() < 3 {
        return None;
    }
    let m = mean(values)?;
    let sd = sample_std_dev(values)?;
    if sd <= 0.0 {
        return Some(0.0);
    }
    let n = values.len() as f64;
    Some(values.iter().map(|v| ((v - m) / sd).powi(3)).sum::<f64>() / n)
}

/// Excess kurtosis (fourth standardised moment minus three).
pub fn excess_kurtosis(values: &[f64]) -> Option<f64> {
    if values.len() < 4 {
        return None;
    }
    let m = mean(values)?;
    let sd = sample_std_dev(values)?;
    if sd <= 0.0 {
        return Some(0.0);
    }
    let n = values.len() as f64;
    Some(values.iter().map(|v| ((v - m) / sd).powi(4)).sum::<f64>() / n - 3.0)
}

/// Pearson product-moment correlation between two equal-length series.
///
/// Returns `None` when the series differ in length, are shorter than three
/// points, or either has zero variance (the coefficient is undefined there).
pub fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 3 {
        return None;
    }
    let ma = mean(a)?;
    let mb = mean(b)?;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = x - ma;
        let dy = y - mb;
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    if da <= 0.0 || db <= 0.0 {
        return None;
    }
    Some((num / (da.sqrt() * db.sqrt())).clamp(-1.0, 1.0))
}

/// Ordinary-least-squares fit of `y` against the sample index `0..n`.
#[derive(Debug, Clone, Copy)]
pub struct LinearFit {
    /// Slope per sample step.
    pub slope: f64,
    /// Intercept at index zero.
    pub intercept: f64,
    /// Coefficient of determination.
    pub r_squared: f64,
    /// Standard error of the slope estimate.
    pub slope_std_error: f64,
    /// Residual standard deviation.
    pub residual_std_dev: f64,
}

/// Fit `y` against its own index by ordinary least squares.
///
/// Returns `None` for fewer than three points (the residual degrees of freedom
/// would be zero and the standard error undefined).
pub fn linear_fit(y: &[f64]) -> Option<LinearFit> {
    let n = y.len();
    if n < 3 {
        return None;
    }
    let nf = n as f64;
    let mean_x = (nf - 1.0) / 2.0;
    let mean_y = mean(y)?;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (i, value) in y.iter().enumerate() {
        let dx = i as f64 - mean_x;
        sxx += dx * dx;
        sxy += dx * (value - mean_y);
    }
    if sxx <= 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (i, value) in y.iter().enumerate() {
        let predicted = intercept + slope * i as f64;
        ss_res += (value - predicted).powi(2);
        ss_tot += (value - mean_y).powi(2);
    }
    let r_squared = if ss_tot > 0.0 {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    } else {
        // A constant series is perfectly described by a zero-slope line.
        1.0
    };
    let residual_variance = ss_res / (nf - 2.0);
    Some(LinearFit {
        slope,
        intercept,
        r_squared,
        slope_std_error: (residual_variance / sxx).sqrt(),
        residual_std_dev: residual_variance.max(0.0).sqrt(),
    })
}

/// Lag-`k` autocorrelation of a series about its own mean.
pub fn autocorrelation(values: &[f64], lag: usize) -> Option<f64> {
    if lag == 0 || values.len() <= lag + 2 {
        return None;
    }
    let m = mean(values)?;
    let denom: f64 = values.iter().map(|v| (v - m).powi(2)).sum();
    if denom <= 0.0 {
        return None;
    }
    let mut num = 0.0;
    for i in lag..values.len() {
        let (Some(current), Some(previous)) = (values.get(i), values.get(i - lag)) else {
            continue;
        };
        num += (current - m) * (previous - m);
    }
    Some((num / denom).clamp(-1.0, 1.0))
}

/// Freedman-Diaconis bin count, falling back to Sturges when the IQR is zero.
pub fn suggested_bin_count(sorted: &[f64]) -> usize {
    let n = sorted.len();
    if n < 2 {
        return 1;
    }
    let sturges = ((n as f64).log2().ceil() as usize + 1).clamp(1, 128);
    let (Some(q1), Some(q3)) = (
        percentile_sorted(sorted, 0.25),
        percentile_sorted(sorted, 0.75),
    ) else {
        return sturges;
    };
    let iqr = q3 - q1;
    if iqr <= 0.0 {
        return sturges;
    }
    let width = 2.0 * iqr / (n as f64).cbrt();
    let (Some(min), Some(max)) = (sorted.first(), sorted.last()) else {
        return sturges;
    };
    if width <= 0.0 || max <= min {
        return sturges;
    }
    (((max - min) / width).ceil() as usize).clamp(1, 128)
}

/// A histogram over `sorted` with `bins` equal-width buckets.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// `bins + 1` edges, ascending.
    pub edges: Vec<f64>,
    /// Observations per bucket.
    pub counts: Vec<u64>,
    /// Midpoint of each bucket.
    pub centers: Vec<f64>,
    /// `counts` normalised by the sample size.
    pub frequencies: Vec<f64>,
    /// Running sum of `frequencies`.
    pub cumulative: Vec<f64>,
}

/// Build an equal-width histogram of an ascending, non-empty slice.
pub fn histogram(sorted: &[f64], bins: usize) -> Option<Histogram> {
    if sorted.is_empty() || bins == 0 {
        return None;
    }
    let min = *sorted.first()?;
    let max = *sorted.last()?;
    // A degenerate range still deserves one honest bucket holding everything.
    let width = if max > min { (max - min) / bins as f64 } else { 1.0 };
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push(min + width * i as f64);
    }
    let mut counts = vec![0u64; bins];
    for value in sorted {
        let mut index = ((value - min) / width).floor() as isize;
        if index < 0 {
            index = 0;
        }
        let index = (index as usize).min(bins - 1);
        if let Some(slot) = counts.get_mut(index) {
            *slot += 1;
        }
    }
    let total = sorted.len() as f64;
    let frequencies: Vec<f64> = counts.iter().map(|c| *c as f64 / total).collect();
    let mut cumulative = Vec::with_capacity(bins);
    let mut running = 0.0;
    for frequency in &frequencies {
        running += frequency;
        cumulative.push(running);
    }
    let centers: Vec<f64> = (0..bins).map(|i| min + width * (i as f64 + 0.5)).collect();
    Some(Histogram {
        edges,
        counts,
        centers,
        frequencies,
        cumulative,
    })
}

/// Standard normal cumulative distribution function.
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Abramowitz & Stegun 7.1.26 error-function approximation (|error| < 1.5e-7).
pub fn erf(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Natural logarithm of the gamma function (Lanczos approximation, g = 7).
pub fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula keeps the approximation on its valid half-line.
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEFFICIENTS[0];
        let t = x + 7.5;
        for (i, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
            a += coefficient / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Regularised upper incomplete gamma function `Q(a, x)`.
///
/// Series expansion below the continued-fraction crossover, Lentz's continued
/// fraction above it; both iterate to a 1e-14 relative tolerance.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_continued_fraction(a, x)
    }
}

fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..500 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-14 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_q_continued_fraction(a: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..500 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-14 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Upper-tail probability of a chi-square variate with `df` degrees of freedom.
pub fn chi_square_sf(statistic: f64, df: f64) -> f64 {
    if !statistic.is_finite() || statistic <= 0.0 || df <= 0.0 {
        return 1.0;
    }
    gamma_q(df / 2.0, statistic / 2.0).clamp(0.0, 1.0)
}

/// Regularised incomplete beta function `I_x(a, b)`.
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // `front` is symmetric under (a, b, x) -> (b, a, 1 - x), so the mirrored
    // branch reuses it unchanged.
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-14 {
            break;
        }
    }
    h
}

/// Two-sided p-value of a Student-t statistic with `df` degrees of freedom.
pub fn student_t_two_sided(t: f64, df: f64) -> f64 {
    if !t.is_finite() || df <= 0.0 {
        return 1.0;
    }
    let x = df / (df + t * t);
    beta_inc(df / 2.0, 0.5, x).clamp(0.0, 1.0)
}

/// Two-sided p-value for a Pearson correlation coefficient over `n` pairs.
pub fn pearson_p_value(r: f64, n: usize) -> f64 {
    if n < 3 {
        return 1.0;
    }
    let df = n as f64 - 2.0;
    let denom = 1.0 - r * r;
    if denom <= f64::EPSILON {
        return 0.0;
    }
    let t = r * (df / denom).sqrt();
    student_t_two_sided(t, df)
}

/// Kolmogorov-Smirnov one-sample statistic against a caller-supplied CDF.
pub fn ks_statistic<F: Fn(f64) -> f64>(sorted: &[f64], cdf: F) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let n = sorted.len() as f64;
    let mut d = 0.0f64;
    for (i, value) in sorted.iter().enumerate() {
        let theoretical = cdf(*value);
        let empirical_low = i as f64 / n;
        let empirical_high = (i as f64 + 1.0) / n;
        d = d.max((theoretical - empirical_low).abs());
        d = d.max((empirical_high - theoretical).abs());
    }
    Some(d)
}

/// Asymptotic p-value of the Kolmogorov-Smirnov statistic (Kolmogorov series).
pub fn ks_p_value(d: f64, n: usize) -> f64 {
    if n == 0 || !d.is_finite() || d <= 0.0 {
        return 1.0;
    }
    let en = (n as f64).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d;
    let mut sum = 0.0;
    for j in 1..=100 {
        let jf = j as f64;
        let term = (-2.0 * jf * jf * lambda * lambda).exp();
        sum += if j % 2 == 1 { term } else { -term };
        if term < 1e-12 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

/// Collect the descriptive numbers a report needs from an ascending slice.
pub fn percentile_map(sorted: &[f64], points: &[u8]) -> HashMap<u8, f64> {
    let mut out = HashMap::new();
    for point in points {
        if let Some(value) = percentile_sorted(sorted, *point as f64 / 100.0) {
            out.insert(*point, value);
        }
    }
    out
}
