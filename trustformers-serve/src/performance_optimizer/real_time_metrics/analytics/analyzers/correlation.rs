//! Correlation analysis across the observed metric series.
//!
//! Replaces the `create_analyzer_placeholder!`-generated `CorrelationAnalyzer`,
//! which ignored its input and returned an empty matrix with a hardcoded
//! determinant and condition number of 1.0. Everything here is computed from
//! the samples; relationships this crate cannot establish (causality, lagged
//! leadership beyond cross-correlation) are reported empty rather than invented.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};

use super::super::super::types::data_structures::TimestampedMetrics;
use super::super::types::{ConditionalDependency, CorrelationMeasure, CorrelationStrength};
use super::super::types::{
    CorrelationAnalysisResult, CorrelationDirection, CorrelationMatrix, CorrelationPattern,
    DependencyAnalysis, LeadLagRelationship, SignificantCorrelation,
};
use super::series::{extract_series, mean, pearson, pearson_p_value, MetricSeries};

/// Minimum aligned samples before a correlation is reportable.
const MIN_SAMPLES: usize = 5;

/// Largest lag, in samples, searched for a lead-lag relationship.
const MAX_LAG: usize = 8;

/// Computes pairwise and partial correlations between metric series.
#[derive(Clone, Debug)]
pub struct CorrelationAnalyzer {
    shutdown: Arc<AtomicBool>,
}

impl CorrelationAnalyzer {
    /// Create a new correlation analyzer.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Correlate every pair of metric series present in `data`.
    pub async fn analyze(&self, data: &[TimestampedMetrics]) -> Result<CorrelationAnalysisResult> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(anyhow!("Correlation analyzer is shut down"));
        }
        if data.len() < MIN_SAMPLES {
            return Err(anyhow!(
                "Correlation analysis needs at least {} samples, got {}",
                MIN_SAMPLES,
                data.len()
            ));
        }
        // A constant series has undefined correlation; keep only series that
        // actually vary so no pair is reported with a substituted coefficient.
        let series: Vec<MetricSeries> =
            extract_series(data).into_iter().filter(|s| varies(&s.values)).collect();
        if series.len() < 2 {
            return Err(anyhow!(
                "Correlation analysis needs at least two varying metric series; {} of {} varied",
                series.len(),
                super::series::SERIES_NAMES.len()
            ));
        }

        let n = series.first().map(|s| s.values.len()).unwrap_or(0);
        let mut pairwise = HashMap::new();
        let mut significant = Vec::new();
        for (i, left) in series.iter().enumerate() {
            for right in series.iter().skip(i + 1) {
                let Some(r) = pearson(&left.values, &right.values) else {
                    continue;
                };
                let p_value = pearson_p_value(r, n);
                let measure = measure_for(r, p_value, n, &left.values, &right.values);
                if p_value < 0.05 {
                    significant.push(SignificantCorrelation {
                        variables: (left.name.to_string(), right.name.to_string()),
                        correlation: r,
                        p_value,
                        // Pearson's r is itself the effect size for a linear
                        // bivariate relationship (Cohen's convention).
                        effect_size: r.abs(),
                        strength: classify_strength(r),
                        direction: if r >= 0.0 {
                            CorrelationDirection::Positive
                        } else {
                            CorrelationDirection::Negative
                        },
                    });
                }
                pairwise.insert((left.name.to_string(), right.name.to_string()), measure);
            }
        }

        let correlation_matrix = build_matrix(&series, n);
        let partial_correlations = partial_correlations(&series);
        let dependency_analysis = dependency_analysis(&series, n);
        let patterns = correlation_patterns(&significant);

        Ok(CorrelationAnalysisResult {
            pairwise_correlations: pairwise,
            correlation_matrix,
            significant_correlations: significant,
            partial_correlations,
            dependency_analysis,
            patterns,
        })
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
}

fn varies(values: &[f64]) -> bool {
    values.len() >= MIN_SAMPLES && super::series::varies_materially(values)
}

fn classify_strength(r: f64) -> CorrelationStrength {
    match r.abs() {
        x if x >= 0.8 => CorrelationStrength::VeryStrong,
        x if x >= 0.6 => CorrelationStrength::Strong,
        x if x >= 0.3 => CorrelationStrength::Moderate,
        _ => CorrelationStrength::Weak,
    }
}

fn measure_for(r: f64, p_value: f64, n: usize, left: &[f64], right: &[f64]) -> CorrelationMeasure {
    let spearman = spearman(left, right).unwrap_or(f64::NAN);
    let kendall_tau = kendall_tau(left, right).unwrap_or(f64::NAN);
    CorrelationMeasure {
        pearson: r,
        spearman,
        kendall_tau,
        // Gaussian mutual information implied by the linear correlation:
        // I = -0.5 * ln(1 - r^2). Exact only under bivariate normality, which
        // the accompanying normality tests report on.
        mutual_information: if r.abs() < 1.0 { -0.5 * (1.0 - r * r).ln() } else { f64::INFINITY },
        // Reported as |r|; a true distance correlation needs the full pairwise
        // distance matrices, which are not retained for a streaming window.
        distance_correlation: r.abs(),
        p_value,
        confidence_interval: fisher_interval(r, n),
    }
}

/// 95% confidence interval for `r` via Fisher's z transform.
fn fisher_interval(r: f64, n: usize) -> (f64, f64) {
    if n <= 3 || r.abs() >= 1.0 {
        return (r, r);
    }
    let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
    let se = 1.0 / ((n as f64 - 3.0).sqrt());
    let lower = z - 1.96 * se;
    let upper = z + 1.96 * se;
    (lower.tanh(), upper.tanh())
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0; values.len()];
    let mut i = 0;
    while i < indexed.len() {
        let mut j = i;
        while j + 1 < indexed.len() {
            let (Some(current), Some(next)) = (indexed.get(j), indexed.get(j + 1)) else {
                break;
            };
            if (current.1 - next.1).abs() > f64::EPSILON {
                break;
            }
            j += 1;
        }
        // Average rank across the tie group keeps Spearman unbiased.
        let average = (i + j) as f64 / 2.0 + 1.0;
        for entry in indexed.iter().take(j + 1).skip(i) {
            if let Some(slot) = out.get_mut(entry.0) {
                *slot = average;
            }
        }
        i = j + 1;
    }
    out
}

fn spearman(a: &[f64], b: &[f64]) -> Option<f64> {
    pearson(&ranks(a), &ranks(b))
}

fn kendall_tau(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 3 {
        return None;
    }
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    for i in 0..a.len() {
        for j in (i + 1)..a.len() {
            let (Some(ai), Some(aj), Some(bi), Some(bj)) = (a.get(i), a.get(j), b.get(i), b.get(j))
            else {
                continue;
            };
            let sign = (aj - ai) * (bj - bi);
            if sign > 0.0 {
                concordant += 1;
            } else if sign < 0.0 {
                discordant += 1;
            }
        }
    }
    let total = concordant + discordant;
    if total == 0 {
        return None;
    }
    Some((concordant - discordant) as f64 / total as f64)
}

fn build_matrix(series: &[MetricSeries], n: usize) -> CorrelationMatrix {
    let variables: Vec<String> = series.iter().map(|s| s.name.to_string()).collect();
    let size = series.len();
    let mut values = vec![vec![0.0; size]; size];
    let mut p_values = vec![vec![1.0; size]; size];
    for i in 0..size {
        for j in 0..size {
            let r = if i == j {
                1.0
            } else {
                match (series.get(i), series.get(j)) {
                    (Some(a), Some(b)) => pearson(&a.values, &b.values).unwrap_or(0.0),
                    _ => 0.0,
                }
            };
            if let Some(row) = values.get_mut(i) {
                if let Some(cell) = row.get_mut(j) {
                    *cell = r;
                }
            }
            if let Some(row) = p_values.get_mut(i) {
                if let Some(cell) = row.get_mut(j) {
                    *cell = if i == j { 0.0 } else { pearson_p_value(r, n) };
                }
            }
        }
    }
    let eigenvalues = symmetric_eigenvalues(&values);
    let determinant = eigenvalues.iter().product::<f64>();
    let max = eigenvalues.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let condition_number = if min.abs() > 1e-12 && max.is_finite() {
        (max / min).abs()
    } else {
        f64::INFINITY
    };
    CorrelationMatrix {
        variables,
        values,
        p_values,
        determinant,
        condition_number,
    }
}

/// Eigenvalues of a small symmetric matrix by the cyclic Jacobi method.
fn symmetric_eigenvalues(matrix: &[Vec<f64>]) -> Vec<f64> {
    let size = matrix.len();
    let mut a = matrix.to_vec();
    for _ in 0..100 {
        let mut off_diagonal = 0.0;
        for i in 0..size {
            for j in (i + 1)..size {
                if let Some(value) = a.get(i).and_then(|row| row.get(j)) {
                    off_diagonal += value * value;
                }
            }
        }
        if off_diagonal < 1e-18 {
            break;
        }
        for p in 0..size {
            for q in (p + 1)..size {
                let apq = a.get(p).and_then(|row| row.get(q)).copied().unwrap_or(0.0);
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = a.get(p).and_then(|row| row.get(p)).copied().unwrap_or(0.0);
                let aqq = a.get(q).and_then(|row| row.get(q)).copied().unwrap_or(0.0);
                let theta = 0.5 * (aqq - app) / apq;
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..size {
                    let akp = a.get(k).and_then(|row| row.get(p)).copied().unwrap_or(0.0);
                    let akq = a.get(k).and_then(|row| row.get(q)).copied().unwrap_or(0.0);
                    if let Some(row) = a.get_mut(k) {
                        if let Some(cell) = row.get_mut(p) {
                            *cell = c * akp - s * akq;
                        }
                        if let Some(cell) = row.get_mut(q) {
                            *cell = s * akp + c * akq;
                        }
                    }
                }
                for k in 0..size {
                    let apk = a.get(p).and_then(|row| row.get(k)).copied().unwrap_or(0.0);
                    let aqk = a.get(q).and_then(|row| row.get(k)).copied().unwrap_or(0.0);
                    if let Some(row) = a.get_mut(p) {
                        if let Some(cell) = row.get_mut(k) {
                            *cell = c * apk - s * aqk;
                        }
                    }
                    if let Some(row) = a.get_mut(q) {
                        if let Some(cell) = row.get_mut(k) {
                            *cell = s * apk + c * aqk;
                        }
                    }
                }
            }
        }
    }
    (0..size)
        .map(|i| a.get(i).and_then(|row| row.get(i)).copied().unwrap_or(0.0))
        .collect()
}

/// First-order partial correlations, each controlling for one third series.
fn partial_correlations(series: &[MetricSeries]) -> HashMap<(String, String), f64> {
    let mut out = HashMap::new();
    for (i, left) in series.iter().enumerate() {
        for right in series.iter().skip(i + 1) {
            let Some(rxy) = pearson(&left.values, &right.values) else {
                continue;
            };
            // Control for the third series most strongly tied to both.
            let mut best: Option<(f64, f64, f64)> = None;
            for control in series.iter() {
                if control.name == left.name || control.name == right.name {
                    continue;
                }
                let (Some(rxz), Some(ryz)) = (
                    pearson(&left.values, &control.values),
                    pearson(&right.values, &control.values),
                ) else {
                    continue;
                };
                let score = rxz.abs().min(ryz.abs());
                if best.map(|(s, _, _)| score > s).unwrap_or(true) {
                    best = Some((score, rxz, ryz));
                }
            }
            let partial = match best {
                Some((_, rxz, ryz)) => {
                    let denominator = ((1.0 - rxz * rxz) * (1.0 - ryz * ryz)).sqrt();
                    if denominator > 1e-12 {
                        (rxy - rxz * ryz) / denominator
                    } else {
                        continue;
                    }
                },
                // With no third series to control for, the partial correlation
                // is undefined; omit the entry rather than echo `rxy`.
                None => continue,
            };
            out.insert(
                (left.name.to_string(), right.name.to_string()),
                partial.clamp(-1.0, 1.0),
            );
        }
    }
    out
}

fn dependency_analysis(series: &[MetricSeries], n: usize) -> DependencyAnalysis {
    let mut lead_lag = Vec::new();
    let mut dependency_scores = HashMap::new();
    let mut conditional = Vec::new();

    for (i, left) in series.iter().enumerate() {
        for right in series.iter().skip(i + 1) {
            if let Some((lag, r)) = best_lag(&left.values, &right.values) {
                if lag != 0 {
                    let (leading, lagging) =
                        if lag > 0 { (left.name, right.name) } else { (right.name, left.name) };
                    lead_lag.push(LeadLagRelationship {
                        leading_variable: leading.to_string(),
                        lagging_variable: lagging.to_string(),
                        // Lag is measured in samples; the caller knows the
                        // sampling interval, so it is reported as a sample
                        // count expressed in whole "sample" units.
                        optimal_lag: std::time::Duration::from_secs(lag.unsigned_abs() as u64),
                        cross_correlation: r,
                        confidence: 1.0 - pearson_p_value(r, n),
                    });
                }
            }
            if let Some(r) = pearson(&left.values, &right.values) {
                dependency_scores.insert(format!("{}~{}", left.name, right.name), r.abs());
            }
            for control in series.iter() {
                if control.name == left.name || control.name == right.name {
                    continue;
                }
                let (Some(rxy), Some(rxz), Some(ryz)) = (
                    pearson(&left.values, &right.values),
                    pearson(&left.values, &control.values),
                    pearson(&right.values, &control.values),
                ) else {
                    continue;
                };
                let denominator = ((1.0 - rxz * rxz) * (1.0 - ryz * ryz)).sqrt();
                if denominator <= 1e-12 {
                    continue;
                }
                let partial = ((rxy - rxz * ryz) / denominator).clamp(-1.0, 1.0);
                // Report only where conditioning changed the picture materially.
                if (partial - rxy).abs() < 0.1 {
                    continue;
                }
                conditional.push(ConditionalDependency {
                    primary_variables: (left.name.to_string(), right.name.to_string()),
                    conditioning_variables: vec![control.name.to_string()],
                    conditional_correlation: partial,
                    test_result: partial - rxy,
                    p_value: pearson_p_value(partial, n.saturating_sub(1)),
                });
            }
        }
    }

    DependencyAnalysis {
        // Causality cannot be established from a correlation window; this crate
        // runs no intervention or Granger test, so the list stays empty rather
        // than restating correlations as causes.
        causal_relationships: Vec::new(),
        lead_lag_relationships: lead_lag,
        conditional_dependencies: conditional,
        dependency_scores,
    }
}

/// Lag in `[-MAX_LAG, MAX_LAG]` maximising |cross-correlation|.
fn best_lag(a: &[f64], b: &[f64]) -> Option<(isize, f64)> {
    let mean_a = mean(a)?;
    let mean_b = mean(b)?;
    let mut best: Option<(isize, f64)> = None;
    let max_lag = MAX_LAG.min(a.len() / 3);
    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let mut num = 0.0;
        let mut da = 0.0;
        let mut db = 0.0;
        let mut count = 0usize;
        for i in 0..a.len() {
            let j = i as isize + lag;
            if j < 0 || j as usize >= b.len() {
                continue;
            }
            let (Some(x), Some(y)) = (a.get(i), b.get(j as usize)) else {
                continue;
            };
            let dx = x - mean_a;
            let dy = y - mean_b;
            num += dx * dy;
            da += dx * dx;
            db += dy * dy;
            count += 1;
        }
        if count < 3 || da <= 0.0 || db <= 0.0 {
            continue;
        }
        let r = (num / (da.sqrt() * db.sqrt())).clamp(-1.0, 1.0);
        if best.map(|(_, br)| r.abs() > br.abs()).unwrap_or(true) {
            best = Some((lag, r));
        }
    }
    best
}

fn correlation_patterns(significant: &[SignificantCorrelation]) -> Vec<CorrelationPattern> {
    let mut clusters: HashMap<&'static str, Vec<String>> = HashMap::new();
    let mut strengths: HashMap<&'static str, f64> = HashMap::new();
    for correlation in significant {
        let label = match correlation.direction {
            CorrelationDirection::Positive => "co-varying",
            CorrelationDirection::Negative => "counter-varying",
        };
        let entry = clusters.entry(label).or_default();
        for name in [&correlation.variables.0, &correlation.variables.1] {
            if !entry.contains(name) {
                entry.push(name.clone());
            }
        }
        let slot = strengths.entry(label).or_insert(0.0);
        *slot = slot.max(correlation.correlation.abs());
    }
    clusters
        .into_iter()
        .map(|(label, variables)| {
            let strength = strengths.get(label).copied().unwrap_or(0.0);
            CorrelationPattern {
                pattern_type: label.to_string(),
                description: format!(
                    "{} metrics move {} (strongest |r| = {:.3})",
                    variables.len(),
                    label,
                    strength
                ),
                variables,
                strength,
                confidence: strength,
            }
        })
        .collect()
}
