//! Trend analysis over the observed metric series.
//!
//! Replaces the `TrendAnalyzer` placeholder that lived in
//! `analytics/types_analysis.rs`, whose `analyze` ignored its input and returned
//! `overall_trend: Stable, trend_strength: 0.5, trend_significance: 0.8` with
//! empty trend, seasonality and change-point lists. `AnalyticsEngine` then fed
//! that invented `trend_significance` straight into its overall confidence.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};

use super::super::super::types::data_structures::TimestampedMetrics;
use super::super::super::types::enums::TrendDirection;
use super::super::types::{
    ChangeDirection, ChangePoint, ChangeType, CyclicalPattern, ForecastPoint, SeasonalComponent,
    TrendForecast,
};
use super::super::types::{TrendAnalysisResult, TrendComponent, TrendDataPoint};
use super::series::{
    autocorrelation, extract_series, linear_fit, mean, sample_std_dev, student_t_two_sided,
    MetricSeries,
};

/// Minimum samples before a slope estimate has residual degrees of freedom.
const MIN_SAMPLES: usize = 6;

/// Autocorrelation above which a lag counts as a cycle.
const CYCLE_THRESHOLD: f64 = 0.5;

/// Fits and tests linear trends, cycles and change points.
#[derive(Clone, Debug)]
pub struct TrendAnalyzer {
    shutdown: Arc<AtomicBool>,
}

impl TrendAnalyzer {
    /// Create a new trend analyzer.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Fit a trend to every metric series in `data`.
    pub async fn analyze(&self, data: &[TimestampedMetrics]) -> Result<TrendAnalysisResult> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(anyhow!("Trend analyzer is shut down"));
        }
        if data.len() < MIN_SAMPLES {
            return Err(anyhow!(
                "Trend analysis needs at least {} samples, got {}",
                MIN_SAMPLES,
                data.len()
            ));
        }
        let timestamps: Vec<DateTime<Utc>> = data.iter().map(|s| s.timestamp).collect();
        let step = median_step(&timestamps);
        let (Some(start), Some(end)) = (timestamps.first().copied(), timestamps.last().copied())
        else {
            return Err(anyhow!("Window carried no timestamps"));
        };
        let series: Vec<MetricSeries> = extract_series(data)
            .into_iter()
            .filter(|s| s.values.len() == data.len())
            .collect();
        if series.is_empty() {
            return Err(anyhow!(
                "No metric series in the window carried a finite reading for every sample"
            ));
        }

        let mut trends = Vec::new();
        let mut seasonal_components = Vec::new();
        let mut cyclical_patterns = Vec::new();
        let mut change_points = Vec::new();
        let mut forecasts = Vec::new();

        for entry in &series {
            let Some(fit) = linear_fit(&entry.values) else {
                continue;
            };
            if fit.slope_std_error <= 0.0 {
                continue;
            }
            let t = fit.slope / fit.slope_std_error;
            let significance = 1.0 - student_t_two_sided(t, entry.values.len() as f64 - 2.0);
            let direction = if significance < 0.95 {
                TrendDirection::Stable
            } else if fit.slope > 0.0 {
                TrendDirection::Increasing
            } else {
                TrendDirection::Decreasing
            };
            let data_points: Vec<TrendDataPoint> = entry
                .values
                .iter()
                .enumerate()
                .filter_map(|(index, value)| {
                    let predicted = fit.intercept + fit.slope * index as f64;
                    Some(TrendDataPoint {
                        timestamp: timestamps.get(index).copied()?,
                        value: *value,
                        predicted_value: predicted,
                        residual: value - predicted,
                        // Ordinary least squares weights every observation
                        // equally; no weighting scheme is applied here.
                        weight: 1.0,
                    })
                })
                .collect();
            trends.push(TrendComponent {
                metric: entry.name.to_string(),
                direction,
                slope: fit.slope,
                // Strength is the coefficient of determination: the share of the
                // series' variance the fitted line accounts for.
                strength: fit.r_squared,
                significance,
                start_time: start,
                end_time: end,
                data_points,
                confidence: significance,
            });

            if let Some((lag, r)) = strongest_cycle(&entry.values) {
                let amplitude = sample_std_dev(&entry.values).unwrap_or(0.0) * r;
                seasonal_components.push(SeasonalComponent {
                    period: step * lag as u32,
                    strength: r,
                    // Phase is not estimated: the autocorrelation peak locates a
                    // period, not an offset within it.
                    phase: 0.0,
                    amplitude,
                    confidence: r,
                });
                cyclical_patterns.push(CyclicalPattern {
                    period: step * lag as u32,
                    strength: r,
                    confidence: r,
                    description: format!(
                        "`{}` repeats every {} samples (autocorrelation {:.3})",
                        entry.name, lag, r
                    ),
                });
            }

            change_points.extend(detect_change_points(entry, &timestamps));

            forecasts.push(TrendForecast {
                horizon: step * 5,
                predicted_values: (1..=5)
                    .filter_map(|k| {
                        let value = fit.intercept + fit.slope * (entry.values.len() + k - 1) as f64;
                        let margin = 1.96 * fit.residual_std_dev * (k as f64).sqrt();
                        Some(ForecastPoint {
                            timestamp: end + chrono::Duration::from_std(step * k as u32).ok()?,
                            value,
                            lower_bound: value - margin,
                            upper_bound: value + margin,
                            confidence: fit.r_squared,
                        })
                    })
                    .collect(),
                confidence: fit.r_squared,
                method: format!("ordinary_least_squares:{}", entry.name),
            });
        }

        if trends.is_empty() {
            return Err(anyhow!(
                "No metric series in the window supported a slope estimate"
            ));
        }

        // The overall direction is the one the strongest significant trend takes.
        let strongest = trends.iter().max_by(|a, b| {
            a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal)
        });
        let overall_trend = strongest
            .map(|component| component.direction.clone())
            .unwrap_or(TrendDirection::Unknown);
        let strengths: Vec<f64> = trends.iter().map(|t| t.strength).collect();
        let significances: Vec<f64> = trends.iter().map(|t| t.significance).collect();

        Ok(TrendAnalysisResult {
            overall_trend,
            trend_strength: mean(&strengths).unwrap_or(0.0),
            trend_significance: mean(&significances).unwrap_or(0.0),
            trends,
            seasonal_components,
            cyclical_patterns,
            change_points,
            forecasts,
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

fn median_step(timestamps: &[DateTime<Utc>]) -> Duration {
    let mut gaps: Vec<i64> = timestamps
        .windows(2)
        .filter_map(|pair| {
            let (Some(first), Some(second)) = (pair.first(), pair.get(1)) else {
                return None;
            };
            let delta = second.signed_duration_since(*first).num_milliseconds();
            if delta > 0 {
                Some(delta)
            } else {
                None
            }
        })
        .collect();
    if gaps.is_empty() {
        return Duration::from_secs(1);
    }
    gaps.sort_unstable();
    Duration::from_millis(gaps.get(gaps.len() / 2).copied().unwrap_or(1000).max(1) as u64)
}

fn strongest_cycle(values: &[f64]) -> Option<(usize, f64)> {
    let max_lag = (values.len() / 3).min(32);
    let mut best: Option<(usize, f64)> = None;
    for lag in 2..=max_lag {
        let Some(r) = autocorrelation(values, lag) else {
            continue;
        };
        if r >= CYCLE_THRESHOLD && best.map(|(_, br)| r > br).unwrap_or(true) {
            best = Some((lag, r));
        }
    }
    best
}

/// Change points located by a sliding Welch t-test between adjacent halves.
fn detect_change_points(entry: &MetricSeries, timestamps: &[DateTime<Utc>]) -> Vec<ChangePoint> {
    let values = &entry.values;
    let n = values.len();
    let window = (n / 4).max(3);
    if n < window * 2 + 1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for split in window..(n - window) {
        let (Some(left), Some(right)) = (
            values.get(split - window..split),
            values.get(split..split + window),
        ) else {
            continue;
        };
        let (Some(m1), Some(m2)) = (mean(left), mean(right)) else {
            continue;
        };
        let (Some(sd1), Some(sd2)) = (sample_std_dev(left), sample_std_dev(right)) else {
            continue;
        };
        let nf = window as f64;
        let v1 = sd1 * sd1 / nf;
        let v2 = sd2 * sd2 / nf;
        if v1 + v2 <= 0.0 {
            continue;
        }
        let t = (m1 - m2) / (v1 + v2).sqrt();
        let df = (v1 + v2).powi(2) / (v1 * v1 / (nf - 1.0) + v2 * v2 / (nf - 1.0));
        let p_value = student_t_two_sided(t, df);
        if p_value >= 0.001 {
            continue;
        }
        let Some(timestamp) = timestamps.get(split).copied() else {
            continue;
        };
        // Only keep the locally strongest split so one shift is not reported
        // once per offset inside the window.
        if out
            .last()
            .and_then(|previous: &ChangePoint| {
                timestamps
                    .get(split.saturating_sub(window))
                    .map(|edge| previous.timestamp >= *edge)
            })
            .unwrap_or(false)
        {
            continue;
        }
        // Which moment moved further decides how the change is labelled.
        let variance_dominates = (sd2 - sd1).abs() > (m2 - m1).abs();
        out.push(ChangePoint {
            timestamp,
            magnitude: (m2 - m1).abs(),
            direction: if variance_dominates {
                ChangeDirection::VarianceChange
            } else if m2 > m1 {
                ChangeDirection::Increase
            } else {
                ChangeDirection::Decrease
            },
            confidence: 1.0 - p_value,
            change_type: if variance_dominates { ChangeType::Variance } else { ChangeType::Mean },
        });
    }
    out
}
