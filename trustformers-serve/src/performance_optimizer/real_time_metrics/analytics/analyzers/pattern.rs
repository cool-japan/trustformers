//! Pattern detection over the observed metric series.
//!
//! Replaces the `create_analyzer_placeholder!`-generated `PatternAnalyzer`,
//! which ignored its input and always classified the window as `"normal"` with
//! `confidence: 0.8` and no patterns. Every pattern below is a statistical
//! finding about the samples the caller supplied.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};

use super::super::super::types::data_structures::TimestampedMetrics;
use super::super::types::{DetectedPattern, PatternAnalysisResult, PatternType};
use super::super::types::{PatternClassification, PatternRelationship, RelationshipType};
use super::series::{
    autocorrelation, extract_series, linear_fit, mean, pearson, sample_std_dev,
    student_t_two_sided, MetricSeries,
};

/// Minimum samples before any pattern claim is defensible.
const MIN_SAMPLES: usize = 10;

/// Autocorrelation above which a lag is called periodic.
const PERIODIC_THRESHOLD: f64 = 0.5;

/// Detects trend, periodicity, spike and level-shift patterns.
#[derive(Clone, Debug)]
pub struct PatternAnalyzer {
    shutdown: Arc<AtomicBool>,
}

impl PatternAnalyzer {
    /// Create a new pattern analyzer.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Detect patterns across every varying metric series in `data`.
    pub async fn analyze(&self, data: &[TimestampedMetrics]) -> Result<PatternAnalysisResult> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(anyhow!("Pattern analyzer is shut down"));
        }
        if data.len() < MIN_SAMPLES {
            return Err(anyhow!(
                "Pattern analysis needs at least {} samples, got {}",
                MIN_SAMPLES,
                data.len()
            ));
        }
        let timestamps: Vec<DateTime<Utc>> = data.iter().map(|s| s.timestamp).collect();
        let step = step_duration(&timestamps);
        let series: Vec<MetricSeries> = extract_series(data)
            .into_iter()
            .filter(|s| s.values.len() == data.len())
            .collect();

        let mut patterns = Vec::new();
        let mut strength_scores = HashMap::new();
        for entry in &series {
            let found = detect_for_series(entry, &timestamps, step);
            for pattern in &found {
                let slot = strength_scores.entry(entry.name.to_string()).or_insert(0.0f64);
                *slot = slot.max(pattern.strength);
            }
            patterns.extend(found);
        }

        let relationships = relate(&series, &patterns);
        let classification = classify(&patterns);
        let confidence =
            patterns.iter().map(|p| p.confidence).fold(0.0f64, f64::max).clamp(0.0, 1.0);

        Ok(PatternAnalysisResult {
            patterns,
            classification,
            relationships,
            strength_scores,
            confidence,
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

fn step_duration(timestamps: &[DateTime<Utc>]) -> Duration {
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

fn window_span(
    timestamps: &[DateTime<Utc>],
    from: usize,
    to: usize,
) -> (DateTime<Utc>, DateTime<Utc>) {
    let start = timestamps.get(from).copied().unwrap_or_else(Utc::now);
    let end = timestamps
        .get(to.min(timestamps.len().saturating_sub(1)))
        .copied()
        .unwrap_or(start);
    (start, end)
}

fn detect_for_series(
    series: &MetricSeries,
    timestamps: &[DateTime<Utc>],
    step: Duration,
) -> Vec<DetectedPattern> {
    let mut out = Vec::new();
    let values = &series.values;
    let span = window_span(timestamps, 0, values.len().saturating_sub(1));
    let total_duration = step * values.len() as u32;

    if let Some(fit) = linear_fit(values) {
        if fit.slope_std_error > 0.0 {
            let t = fit.slope / fit.slope_std_error;
            let p_value = student_t_two_sided(t, values.len() as f64 - 2.0);
            if p_value < 0.05 && fit.r_squared > 0.1 {
                let mut characteristics = HashMap::new();
                characteristics.insert("slope_per_sample".to_string(), fit.slope);
                characteristics.insert("r_squared".to_string(), fit.r_squared);
                characteristics.insert("p_value".to_string(), p_value);
                out.push(DetectedPattern {
                    name: format!("{}_trend", series.name),
                    pattern_type: PatternType::Trending,
                    characteristics,
                    // A monotone trend spans the whole window exactly once.
                    frequency: 1.0,
                    duration: total_duration,
                    strength: fit.r_squared,
                    occurrences: vec![span],
                    confidence: 1.0 - p_value,
                });
            }
        }
    }

    let max_lag = (values.len() / 3).min(32);
    let mut best_lag: Option<(usize, f64)> = None;
    for lag in 2..=max_lag {
        let Some(r) = autocorrelation(values, lag) else {
            continue;
        };
        if best_lag.map(|(_, best)| r > best).unwrap_or(true) {
            best_lag = Some((lag, r));
        }
    }
    if let Some((lag, r)) = best_lag {
        if r >= PERIODIC_THRESHOLD {
            let mut characteristics = HashMap::new();
            characteristics.insert("lag_samples".to_string(), lag as f64);
            characteristics.insert("autocorrelation".to_string(), r);
            let cycles = values.len() as f64 / lag as f64;
            out.push(DetectedPattern {
                name: format!("{}_periodicity", series.name),
                pattern_type: PatternType::Periodic,
                characteristics,
                frequency: cycles,
                duration: step * lag as u32,
                strength: r,
                occurrences: vec![span],
                confidence: r,
            });
        }
    }

    if let (Some(m), Some(sd)) = (mean(values), sample_std_dev(values)) {
        if sd > 0.0 {
            let mut occurrences = Vec::new();
            let mut peak_z = 0.0f64;
            for (index, value) in values.iter().enumerate() {
                let z = (value - m).abs() / sd;
                if z > 3.0 {
                    occurrences.push(window_span(timestamps, index, index));
                    peak_z = peak_z.max(z);
                }
            }
            if !occurrences.is_empty() {
                let mut characteristics = HashMap::new();
                characteristics.insert("max_z_score".to_string(), peak_z);
                characteristics.insert("mean".to_string(), m);
                characteristics.insert("std_dev".to_string(), sd);
                let count = occurrences.len();
                out.push(DetectedPattern {
                    name: format!("{}_spikes", series.name),
                    pattern_type: PatternType::Spike,
                    characteristics,
                    frequency: count as f64 / values.len() as f64,
                    duration: step,
                    // Strength grows with how far past three sigma the worst
                    // sample reached, saturating at six sigma.
                    strength: ((peak_z - 3.0) / 3.0).clamp(0.0, 1.0),
                    occurrences,
                    confidence: ((peak_z - 3.0) / 3.0).clamp(0.0, 1.0),
                });
            }
        }
    }

    if let Some(shift) = level_shift(values) {
        let midpoint = values.len() / 2;
        let mut characteristics = HashMap::new();
        characteristics.insert("mean_before".to_string(), shift.before);
        characteristics.insert("mean_after".to_string(), shift.after);
        characteristics.insert("welch_p_value".to_string(), shift.p_value);
        out.push(DetectedPattern {
            name: format!("{}_level_shift", series.name),
            pattern_type: PatternType::Anomalous,
            characteristics,
            frequency: 1.0,
            duration: total_duration,
            strength: (1.0 - shift.p_value).clamp(0.0, 1.0),
            occurrences: vec![window_span(timestamps, midpoint, values.len() - 1)],
            confidence: (1.0 - shift.p_value).clamp(0.0, 1.0),
        });
    }

    out
}

struct LevelShift {
    before: f64,
    after: f64,
    p_value: f64,
}

/// Welch's t-test between the window's two halves.
fn level_shift(values: &[f64]) -> Option<LevelShift> {
    if values.len() < 8 {
        return None;
    }
    let midpoint = values.len() / 2;
    let first = values.get(..midpoint)?;
    let second = values.get(midpoint..)?;
    let (m1, m2) = (mean(first)?, mean(second)?);
    let (sd1, sd2) = (sample_std_dev(first)?, sample_std_dev(second)?);
    let (n1, n2) = (first.len() as f64, second.len() as f64);
    let v1 = sd1 * sd1 / n1;
    let v2 = sd2 * sd2 / n2;
    if v1 + v2 <= 0.0 {
        return None;
    }
    let t = (m1 - m2) / (v1 + v2).sqrt();
    // Welch-Satterthwaite degrees of freedom.
    let df = (v1 + v2).powi(2) / (v1 * v1 / (n1 - 1.0) + v2 * v2 / (n2 - 1.0));
    let p_value = student_t_two_sided(t, df);
    if p_value >= 0.01 {
        return None;
    }
    Some(LevelShift {
        before: m1,
        after: m2,
        p_value,
    })
}

fn relate(series: &[MetricSeries], patterns: &[DetectedPattern]) -> Vec<PatternRelationship> {
    let mut out = Vec::new();
    for (i, left) in patterns.iter().enumerate() {
        for right in patterns.iter().skip(i + 1) {
            if !same_kind(&left.pattern_type, &right.pattern_type) {
                continue;
            }
            let left_series = series.iter().find(|s| left.name.starts_with(s.name));
            let right_series = series.iter().find(|s| right.name.starts_with(s.name));
            let (Some(a), Some(b)) = (left_series, right_series) else {
                continue;
            };
            if a.name == b.name {
                continue;
            }
            let Some(r) = pearson(&a.values, &b.values) else {
                continue;
            };
            out.push(PatternRelationship {
                source_pattern: left.name.clone(),
                target_pattern: right.name.clone(),
                // Co-occurrence is what a shared window supports; causal and
                // hierarchical links are not testable from these samples.
                relationship_type: RelationshipType::CoOccurrence,
                strength: r.abs(),
                // No lagged search is performed for pattern relationships, so
                // there is no offset to report.
                temporal_offset: None,
                confidence: r.abs(),
            });
        }
    }
    out
}

fn same_kind(a: &PatternType, b: &PatternType) -> bool {
    matches!(
        (a, b),
        (PatternType::Trending, PatternType::Trending)
            | (PatternType::Periodic, PatternType::Periodic)
            | (PatternType::Spike, PatternType::Spike)
            | (PatternType::Anomalous, PatternType::Anomalous)
            | (PatternType::Cyclical, PatternType::Cyclical)
            | (PatternType::Behavioral, PatternType::Behavioral)
            | (PatternType::Performance, PatternType::Performance)
    )
}

fn classify(patterns: &[DetectedPattern]) -> PatternClassification {
    let mut features = HashMap::new();
    for pattern in patterns {
        let key = type_label(&pattern.pattern_type).to_string();
        let slot = features.entry(key).or_insert(0.0f64);
        *slot = slot.max(pattern.strength);
    }
    let strongest = patterns
        .iter()
        .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal));
    let primary_class = strongest
        .map(|p| type_label(&p.pattern_type).to_string())
        // No detector fired: the window is featureless, which is a finding, not
        // a fallback label.
        .unwrap_or_else(|| "no_pattern_detected".to_string());
    let mut secondary_classes: Vec<String> =
        features.keys().filter(|k| **k != primary_class).cloned().collect();
    secondary_classes.sort();
    PatternClassification {
        primary_class,
        secondary_classes,
        confidence: strongest.map(|p| p.confidence).unwrap_or(0.0),
        features,
    }
}

fn type_label(pattern_type: &PatternType) -> &'static str {
    match pattern_type {
        PatternType::Periodic => "periodic",
        PatternType::Trending => "trending",
        PatternType::Cyclical => "cyclical",
        PatternType::Spike => "spike",
        PatternType::Anomalous => "anomalous",
        PatternType::Behavioral => "behavioral",
        PatternType::Performance => "performance",
    }
}
