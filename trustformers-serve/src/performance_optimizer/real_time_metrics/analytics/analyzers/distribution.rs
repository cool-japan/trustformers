//! Distribution analysis over an observed metric series.
//!
//! Replaces the `create_analyzer_placeholder!`-generated `DistributionAnalyzer`
//! that ignored its input and returned a hand-written "normal-like" report with
//! invented Shapiro-Wilk / Jarque-Bera / D'Agostino statistics. Every number
//! below is computed from the samples the caller passed in; the two statistics
//! this crate cannot compute honestly are `Option` and reported as `None`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};

use super::super::super::types::data_structures::TimestampedMetrics;
use super::super::types::{
    DistributionAnalysisResult, DistributionCharacteristics, DistributionFit,
    GoodnessOfFitStatistics, HistogramAnalysis, NormalityAssessment, NormalityTestResult,
    ShapeAssessment,
};
use super::super::types::{DistributionComparison, HistogramData, HistogramPeak};
use super::series::{
    chi_square_sf, excess_kurtosis, extract_series, histogram, ks_p_value, ks_statistic, mean,
    normal_cdf, percentile_sorted, sample_std_dev, skewness, sorted_finite, suggested_bin_count,
    Histogram,
};

/// The series this analyzer characterises.
///
/// Latency is the metric whose distributional shape (heavy tails, multi-modality)
/// drives serving behaviour; the remaining series are summarised by
/// `StatisticalAnalyzer` and `PerformanceAnalyzer`.
const ANALYSED_SERIES: &str = "latency_seconds";

/// Minimum sample count before a distribution fit means anything.
const MIN_SAMPLES: usize = 8;

/// Fits parametric distributions to an observed metric series.
#[derive(Clone, Debug)]
pub struct DistributionAnalyzer {
    shutdown: Arc<AtomicBool>,
}

impl DistributionAnalyzer {
    /// Create a new distribution analyzer.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Characterise the latency distribution of `data`.
    ///
    /// Returns an error naming the shortfall when the window is too small to
    /// support a fit, rather than reporting a default-valued distribution.
    pub async fn analyze(&self, data: &[TimestampedMetrics]) -> Result<DistributionAnalysisResult> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(anyhow!("Distribution analyzer is shut down"));
        }
        let values = self.series_values(data)?;
        let sorted = sorted_finite(&values);
        analyse_sorted(ANALYSED_SERIES, &sorted)
    }

    /// Stop accepting analyses.
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// True once `shutdown` has been called.
    pub fn is_shut_down(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    fn series_values(&self, data: &[TimestampedMetrics]) -> Result<Vec<f64>> {
        extract_series(data)
            .into_iter()
            .find(|s| s.name == ANALYSED_SERIES)
            .map(|s| s.values)
            .ok_or_else(|| anyhow!("Metric series `{}` is not present", ANALYSED_SERIES))
    }
}

/// Characterise an ascending sample of a named series.
///
/// Shared by `DistributionAnalyzer` and by the anomaly detector, which runs it
/// over its own score distribution.
pub(crate) fn analyse_sorted(series: &str, sorted: &[f64]) -> Result<DistributionAnalysisResult> {
    if sorted.len() < MIN_SAMPLES {
        return Err(anyhow!(
            "Distribution analysis needs at least {} finite `{}` samples, got {}",
            MIN_SAMPLES,
            series,
            sorted.len()
        ));
    }
    let fits = fit_candidates(sorted);
    let best_fit = fits
        .iter()
        .max_by(|a, b| a.fit_score.partial_cmp(&b.fit_score).unwrap_or(std::cmp::Ordering::Equal))
        .cloned();
    let normality_assessment = assess_normality(sorted);
    let bins = suggested_bin_count(sorted);
    let hist = histogram(sorted, bins)
        .ok_or_else(|| anyhow!("Histogram construction failed for `{}`", series))?;
    let histogram_analysis = analyse_histogram(sorted, &hist, bins);
    let characteristics = describe(sorted, best_fit.as_ref(), &hist, &normality_assessment);
    let comparison_results = compare_fits(&fits);
    Ok(DistributionAnalysisResult {
        series: series.to_string(),
        distribution_fits: fits,
        best_fit,
        normality_assessment,
        characteristics,
        histogram_analysis,
        comparison_results,
    })
}

/// Convert a `HistogramData`-shaped view out of the computed histogram.
pub(crate) fn to_histogram_data(hist: &Histogram) -> HistogramData {
    HistogramData {
        bin_edges: hist.edges.clone(),
        bin_counts: hist.counts.clone(),
        bin_centers: hist.centers.clone(),
        frequencies: hist.frequencies.clone(),
        cumulative_frequencies: hist.cumulative.clone(),
    }
}

fn fit_candidates(sorted: &[f64]) -> Vec<DistributionFit> {
    let mut fits = Vec::new();
    if let Some(fit) = fit_normal(sorted) {
        fits.push(fit);
    }
    if sorted.first().is_some_and(|v| *v > 0.0) {
        if let Some(fit) = fit_exponential(sorted) {
            fits.push(fit);
        }
        if let Some(fit) = fit_lognormal(sorted) {
            fits.push(fit);
        }
    }
    fits
}

fn goodness_of_fit<F: Fn(f64) -> f64>(
    sorted: &[f64],
    cdf: F,
    log_likelihood: f64,
    parameter_count: usize,
) -> Option<GoodnessOfFitStatistics> {
    let n = sorted.len();
    let ks = ks_statistic(sorted, &cdf)?;
    let (chi_square_statistic, chi_square_p_value) =
        chi_square_goodness(sorted, &cdf, parameter_count)?;
    let k = parameter_count as f64;
    Some(GoodnessOfFitStatistics {
        ks_statistic: ks,
        ks_p_value: ks_p_value(ks, n),
        ad_statistic: anderson_darling(sorted, &cdf)?,
        // The Anderson-Darling null distribution depends on which family was
        // fitted and on n; no critical-value table is linked in this crate, so
        // only the statistic is reported.
        ad_p_value: None,
        chi_square_statistic,
        chi_square_p_value,
        log_likelihood,
        aic: 2.0 * k - 2.0 * log_likelihood,
        bic: k * (n as f64).ln() - 2.0 * log_likelihood,
    })
}

fn anderson_darling<F: Fn(f64) -> f64>(sorted: &[f64], cdf: &F) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    let nf = n as f64;
    let mut sum = 0.0;
    for i in 0..n {
        let lower = cdf(*sorted.get(i)?).clamp(1e-12, 1.0 - 1e-12);
        let upper = cdf(*sorted.get(n - 1 - i)?).clamp(1e-12, 1.0 - 1e-12);
        sum += (2.0 * (i as f64 + 1.0) - 1.0) * (lower.ln() + (1.0 - upper).ln());
    }
    Some(-nf - sum / nf)
}

fn chi_square_goodness<F: Fn(f64) -> f64>(
    sorted: &[f64],
    cdf: &F,
    parameter_count: usize,
) -> Option<(f64, f64)> {
    let bins = suggested_bin_count(sorted).max(2);
    let hist = histogram(sorted, bins)?;
    let n = sorted.len() as f64;
    let mut statistic = 0.0;
    let mut used_bins = 0usize;
    for index in 0..hist.counts.len() {
        let low = *hist.edges.get(index)?;
        let high = *hist.edges.get(index + 1)?;
        let expected = (cdf(high) - cdf(low)).max(0.0) * n;
        if expected < 1.0 {
            // Pearson's approximation is invalid for sparse cells; skip them
            // and shrink the degrees of freedom accordingly.
            continue;
        }
        let observed = *hist.counts.get(index)? as f64;
        statistic += (observed - expected).powi(2) / expected;
        used_bins += 1;
    }
    let df = used_bins as isize - 1 - parameter_count as isize;
    if df <= 0 {
        return None;
    }
    Some((statistic, chi_square_sf(statistic, df as f64)))
}

fn fit_normal(sorted: &[f64]) -> Option<DistributionFit> {
    let mu = mean(sorted)?;
    let sigma = sample_std_dev(sorted)?;
    if sigma <= 0.0 {
        return None;
    }
    let cdf = move |x: f64| normal_cdf((x - mu) / sigma);
    let log_likelihood: f64 = sorted
        .iter()
        .map(|x| {
            let z = (x - mu) / sigma;
            -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
        })
        .sum();
    let stats = goodness_of_fit(sorted, cdf, log_likelihood, 2)?;
    Some(build_fit(
        "normal",
        [("mean", mu), ("std_dev", sigma)],
        stats,
        sorted.len(),
        sigma,
    ))
}

fn fit_exponential(sorted: &[f64]) -> Option<DistributionFit> {
    let m = mean(sorted)?;
    if m <= 0.0 {
        return None;
    }
    let rate = 1.0 / m;
    let cdf = move |x: f64| if x <= 0.0 { 0.0 } else { 1.0 - (-rate * x).exp() };
    let log_likelihood: f64 = sorted.iter().map(|x| rate.ln() - rate * x).sum();
    let stats = goodness_of_fit(sorted, cdf, log_likelihood, 1)?;
    Some(build_fit(
        "exponential",
        [("rate", rate), ("mean", m)],
        stats,
        sorted.len(),
        m,
    ))
}

fn fit_lognormal(sorted: &[f64]) -> Option<DistributionFit> {
    let logs: Vec<f64> = sorted.iter().map(|x| x.ln()).collect();
    let mu = mean(&logs)?;
    let sigma = sample_std_dev(&logs)?;
    if sigma <= 0.0 {
        return None;
    }
    let cdf = move |x: f64| {
        if x <= 0.0 {
            0.0
        } else {
            normal_cdf((x.ln() - mu) / sigma)
        }
    };
    let log_likelihood: f64 = sorted
        .iter()
        .zip(logs.iter())
        .map(|(x, l)| {
            let z = (l - mu) / sigma;
            -0.5 * z * z - sigma.ln() - x.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
        })
        .sum();
    let stats = goodness_of_fit(sorted, cdf, log_likelihood, 2)?;
    Some(build_fit(
        "lognormal",
        [("log_mean", mu), ("log_std_dev", sigma)],
        stats,
        sorted.len(),
        sigma,
    ))
}

/// Assemble a `DistributionFit`, deriving the fit score from the KS distance.
///
/// `scale` is the fitted spread parameter; it sets the width of the reported
/// parameter confidence intervals via the large-sample normal approximation.
fn build_fit(
    name: &str,
    parameters: [(&str, f64); 2],
    stats: GoodnessOfFitStatistics,
    n: usize,
    scale: f64,
) -> DistributionFit {
    let standard_error = scale / (n as f64).sqrt();
    let mut parameter_map = HashMap::new();
    let mut intervals = HashMap::new();
    for (key, value) in parameters {
        parameter_map.insert(key.to_string(), value);
        intervals.insert(
            key.to_string(),
            (value - 1.96 * standard_error, value + 1.96 * standard_error),
        );
    }
    DistributionFit {
        distribution_name: name.to_string(),
        parameters: parameter_map,
        // Fit score is `1 - D`, the Kolmogorov-Smirnov supremum distance
        // subtracted from one, so that 1.0 is an exact empirical match.
        fit_score: (1.0 - stats.ks_statistic).clamp(0.0, 1.0),
        fit_statistics: stats,
        parameter_confidence_intervals: intervals,
    }
}

fn assess_normality(sorted: &[f64]) -> NormalityAssessment {
    let jarque_bera = jarque_bera_test(sorted);
    let dagostino = dagostino_k2_test(sorted);
    let mut executed = Vec::new();
    if let Some(test) = jarque_bera.as_ref() {
        executed.push(test.is_normal);
    }
    if let Some(test) = dagostino.as_ref() {
        executed.push(test.is_normal);
    }
    let normal_votes = executed.iter().filter(|v| **v).count();
    let is_normal = !executed.is_empty() && normal_votes * 2 > executed.len();
    let confidence = if executed.is_empty() {
        0.0
    } else {
        let agreeing = if is_normal { normal_votes } else { executed.len() - normal_votes };
        agreeing as f64 / executed.len() as f64
    };
    NormalityAssessment {
        // Shapiro-Wilk needs the Royston coefficient tables, which this crate
        // does not carry; reporting a substitute statistic would be a fiction.
        shapiro_wilk: None,
        jarque_bera,
        dagostino,
        is_normal,
        confidence,
    }
}

fn jarque_bera_test(values: &[f64]) -> Option<NormalityTestResult> {
    let n = values.len();
    if n < 8 {
        return None;
    }
    let s = skewness(values)?;
    let k = excess_kurtosis(values)?;
    let statistic = (n as f64 / 6.0) * (s * s + k * k / 4.0);
    let p_value = chi_square_sf(statistic, 2.0);
    Some(NormalityTestResult {
        statistic,
        p_value,
        is_normal: p_value > 0.05,
        significance_level: 0.05,
    })
}

/// D'Agostino-Pearson omnibus K-squared test.
///
/// Requires n >= 20; below that the skewness and kurtosis transformations are
/// not usable and the test is reported as unavailable rather than approximated.
fn dagostino_k2_test(values: &[f64]) -> Option<NormalityTestResult> {
    let n = values.len();
    if n < 20 {
        return None;
    }
    let nf = n as f64;
    let g1 = skewness(values)?;
    let g2 = excess_kurtosis(values)?;

    // Skewness transformation (D'Agostino 1970).
    let y = g1 * ((nf + 1.0) * (nf + 3.0) / (6.0 * (nf - 2.0))).sqrt();
    let beta2 = 3.0 * (nf * nf + 27.0 * nf - 70.0) * (nf + 1.0) * (nf + 3.0)
        / ((nf - 2.0) * (nf + 5.0) * (nf + 7.0) * (nf + 9.0));
    let w_squared = -1.0 + (2.0 * (beta2 - 1.0)).sqrt();
    if w_squared <= 0.0 {
        return None;
    }
    let w = w_squared.sqrt();
    if w <= 1.0 {
        return None;
    }
    let delta = 1.0 / w.ln().sqrt();
    let alpha = (2.0 / (w_squared - 1.0)).sqrt();
    let ratio = y / alpha;
    let z1 = delta * (ratio + (ratio * ratio + 1.0).sqrt()).ln();

    // Kurtosis transformation (Anscombe-Glynn 1983).
    let expected = 3.0 * (nf - 1.0) / (nf + 1.0);
    let variance =
        24.0 * nf * (nf - 2.0) * (nf - 3.0) / ((nf + 1.0) * (nf + 1.0) * (nf + 3.0) * (nf + 5.0));
    if variance <= 0.0 {
        return None;
    }
    let x = (g2 + 3.0 - expected) / variance.sqrt();
    let sqrt_beta1 = 6.0 * (nf * nf - 5.0 * nf + 2.0) / ((nf + 7.0) * (nf + 9.0))
        * (6.0 * (nf + 3.0) * (nf + 5.0) / (nf * (nf - 2.0) * (nf - 3.0))).sqrt();
    if sqrt_beta1 <= 0.0 {
        return None;
    }
    let a = 6.0
        + (8.0 / sqrt_beta1) * (2.0 / sqrt_beta1 + (1.0 + 4.0 / (sqrt_beta1 * sqrt_beta1)).sqrt());
    if a <= 4.0 {
        return None;
    }
    let inner = (1.0 - 2.0 / a) / (1.0 + x * (2.0 / (a - 4.0)).sqrt());
    if inner <= 0.0 {
        return None;
    }
    let z2 = ((1.0 - 2.0 / (9.0 * a)) - inner.cbrt()) / (2.0 / (9.0 * a)).sqrt();

    let statistic = z1 * z1 + z2 * z2;
    let p_value = chi_square_sf(statistic, 2.0);
    Some(NormalityTestResult {
        statistic,
        p_value,
        is_normal: p_value > 0.05,
        significance_level: 0.05,
    })
}

fn analyse_histogram(sorted: &[f64], hist: &Histogram, bins: usize) -> HistogramAnalysis {
    let peaks = detect_peaks(hist);
    let skew = skewness(sorted).unwrap_or(0.0);
    let kurt = excess_kurtosis(sorted).unwrap_or(0.0);
    let is_unimodal = peaks.len() <= 1;
    let is_symmetric = skew.abs() < 0.5;
    let has_heavy_tails = kurt > 1.0;
    let shape_description = format!(
        "{} peak(s), skewness {:.3}, excess kurtosis {:.3}",
        peaks.len(),
        skew,
        kurt
    );
    HistogramAnalysis {
        optimal_bins: bins as u32,
        histogram: to_histogram_data(hist),
        peaks,
        shape_assessment: ShapeAssessment {
            is_unimodal,
            is_symmetric,
            has_heavy_tails,
            shape_description,
        },
    }
}

/// Local maxima of the histogram, with prominence relative to the lower of the
/// two adjacent valleys.
fn detect_peaks(hist: &Histogram) -> Vec<HistogramPeak> {
    let counts = &hist.counts;
    if counts.len() < 3 {
        return Vec::new();
    }
    let width = match (hist.edges.first(), hist.edges.get(1)) {
        (Some(a), Some(b)) => b - a,
        _ => 0.0,
    };
    let mut peaks = Vec::new();
    for index in 1..counts.len() - 1 {
        let (Some(previous), Some(current), Some(next)) = (
            counts.get(index - 1),
            counts.get(index),
            counts.get(index + 1),
        ) else {
            continue;
        };
        if current > previous && current >= next {
            let valley = (*previous).min(*next) as f64;
            let Some(location) = hist.centers.get(index) else {
                continue;
            };
            peaks.push(HistogramPeak {
                location: *location,
                height: *current as f64,
                prominence: *current as f64 - valley,
                width,
            });
        }
    }
    peaks
}

fn describe(
    sorted: &[f64],
    best_fit: Option<&DistributionFit>,
    hist: &Histogram,
    normality: &NormalityAssessment,
) -> DistributionCharacteristics {
    let mut normality_tests = HashMap::new();
    if let Some(test) = normality.jarque_bera.as_ref() {
        normality_tests.insert("jarque_bera".to_string(), test.clone());
    }
    if let Some(test) = normality.dagostino.as_ref() {
        normality_tests.insert("dagostino".to_string(), test.clone());
    }
    DistributionCharacteristics {
        distribution_type: best_fit
            .map(|f| f.distribution_name.clone())
            .unwrap_or_else(|| "unfitted".to_string()),
        parameters: best_fit.map(|f| f.parameters.clone()).unwrap_or_default(),
        goodness_of_fit: best_fit.map(|f| f.fit_score).unwrap_or(0.0),
        normality_tests,
        histogram: to_histogram_data(hist),
        // Reported as the sample skewness: zero means symmetric, positive means
        // a right tail. This is not a 0..1 "symmetry score".
        symmetry: skewness(sorted).unwrap_or(0.0),
        // Reported as excess kurtosis: zero matches the normal distribution.
        peakedness: excess_kurtosis(sorted).unwrap_or(0.0),
    }
}

/// Compare every fitted family against every other by the largest gap between
/// their fitted CDFs at the observed quantiles.
fn compare_fits(fits: &[DistributionFit]) -> Vec<DistributionComparison> {
    let mut out = Vec::new();
    for (i, reference) in fits.iter().enumerate() {
        for candidate in fits.iter().skip(i + 1) {
            let mut distances = HashMap::new();
            distances.insert(
                "ks_statistic_delta".to_string(),
                (reference.fit_statistics.ks_statistic - candidate.fit_statistics.ks_statistic)
                    .abs(),
            );
            distances.insert(
                "aic_delta".to_string(),
                (reference.fit_statistics.aic - candidate.fit_statistics.aic).abs(),
            );
            distances.insert(
                "log_likelihood_delta".to_string(),
                (reference.fit_statistics.log_likelihood - candidate.fit_statistics.log_likelihood)
                    .abs(),
            );
            let similarity = 1.0 - (reference.fit_score - candidate.fit_score).abs();
            out.push(DistributionComparison {
                reference: reference.distribution_name.clone(),
                comparison: candidate.distribution_name.clone(),
                distance_measures: distances,
                similarity_score: similarity.clamp(0.0, 1.0),
                // Both fits used the same sample, so the comparison is as
                // reliable as the weaker of the two fits.
                confidence: reference.fit_score.min(candidate.fit_score),
            });
        }
    }
    out
}

/// Best-fitting parametric family for an ascending sample, with its parameters.
///
/// Returns `None` when no candidate family could be fitted (a constant series,
/// or fewer points than a fit needs).
pub(crate) fn best_family(sorted: &[f64]) -> Option<(String, HashMap<String, f64>, f64)> {
    let fits = fit_candidates(sorted);
    let best = fits.into_iter().max_by(|a, b| {
        a.fit_score.partial_cmp(&b.fit_score).unwrap_or(std::cmp::Ordering::Equal)
    })?;
    Some((best.distribution_name, best.parameters, best.fit_score))
}

/// Quartile summary used by the performance analyzer's outlier reporting.
pub(crate) fn quartiles(sorted: &[f64]) -> Option<(f64, f64, f64)> {
    Some((
        percentile_sorted(sorted, 0.25)?,
        percentile_sorted(sorted, 0.50)?,
        percentile_sorted(sorted, 0.75)?,
    ))
}
