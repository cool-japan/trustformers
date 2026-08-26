//! Anomaly detection over the observed metric series.
//!
//! Replaces the hand-written `AnomalyDetector` placeholder in `analytics/types.rs`,
//! which ignored its input and returned `anomaly_rate: 0.02`,
//! `detection_confidence: 0.9` and a `baseline_performance` block claiming
//! `accuracy: 0.95 / precision: 0.9 / recall: 0.85 / auc_roc: 0.92` for a model
//! that was never trained or evaluated. Detection here is a robust modified
//! z-score against the window's own median and MAD; the classifier-quality
//! block is now `Option` and reported as `None` because no labelled ground
//! truth exists to score against.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};

use super::super::super::types::data_structures::TimestampedMetrics;
use super::super::types::{
    AnomalyAnalysisResult, AnomalyDetection, AnomalyPattern, AnomalySeverity, AnomalyType,
};
use super::distribution::analyse_sorted;
use super::series::{extract_series, percentile_sorted, sorted_finite, MetricSeries};

/// Minimum samples before a median/MAD baseline means anything.
const MIN_SAMPLES: usize = 8;

/// Modified z-score above which a reading is called anomalous (Iglewicz-Hoaglin).
const MODIFIED_Z_THRESHOLD: f64 = 3.5;

/// Consecutive flagged samples that constitute a collective anomaly.
const COLLECTIVE_RUN: usize = 3;

/// Flags readings that deviate from the window's own robust centre.
#[derive(Clone, Debug)]
pub struct AnomalyDetector {
    shutdown: Arc<AtomicBool>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Score every sample in `data` and report the anomalous ones.
    pub async fn analyze(&self, data: &[TimestampedMetrics]) -> Result<AnomalyAnalysisResult> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(anyhow!("Anomaly detector is shut down"));
        }
        if data.len() < MIN_SAMPLES {
            return Err(anyhow!(
                "Anomaly detection needs at least {} samples to establish a baseline, got {}",
                MIN_SAMPLES,
                data.len()
            ));
        }
        let timestamps: Vec<DateTime<Utc>> = data.iter().map(|s| s.timestamp).collect();
        let series: Vec<MetricSeries> = extract_series(data)
            .into_iter()
            .filter(|s| s.values.len() == data.len())
            .collect();
        let baselines: Vec<(&MetricSeries, RobustBaseline)> = series
            .iter()
            .filter_map(|entry| robust_baseline(&entry.values).map(|b| (entry, b)))
            .collect();
        if baselines.is_empty() {
            return Err(anyhow!(
                "No metric series in the window had a non-zero median absolute deviation; \
                 nothing can be scored against a robust baseline"
            ));
        }

        // Per-sample score: the largest modified z-score across scoreable series.
        let mut scores = Vec::with_capacity(data.len());
        let mut per_sample_metrics: Vec<Vec<String>> = vec![Vec::new(); data.len()];
        for index in 0..data.len() {
            let mut worst = 0.0f64;
            for (entry, baseline) in &baselines {
                let Some(value) = entry.values.get(index) else {
                    continue;
                };
                let z = baseline.modified_z(*value);
                if z > MODIFIED_Z_THRESHOLD {
                    if let Some(slot) = per_sample_metrics.get_mut(index) {
                        slot.push(entry.name.to_string());
                    }
                }
                worst = worst.max(z);
            }
            scores.push(worst);
        }

        let anomalies = self.collect_anomalies(&scores, &per_sample_metrics, &timestamps);
        let anomaly_rate = anomalies.len() as f64 / data.len() as f64;
        let patterns = anomaly_patterns(&scores, &timestamps);
        // The score distribution is itself a real sample; characterise it with
        // the same machinery, and report the failure honestly if it is too
        // degenerate to fit (for example an all-zero score vector).
        let score_distribution = analyse_sorted("anomaly_score", &sorted_finite(&scores))?;
        // Confidence is the share of the seven metric series that carried a
        // usable robust baseline: detection saw that fraction of the picture.
        let detection_confidence =
            baselines.len() as f64 / super::series::SERIES_NAMES.len() as f64;

        Ok(AnomalyAnalysisResult {
            anomalies,
            score_distribution,
            patterns,
            // Precision/recall/AUC need labelled anomalies. None are available,
            // so no classifier-quality block is reported.
            baseline_performance: None,
            anomaly_rate,
            detection_confidence,
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

    fn collect_anomalies(
        &self,
        scores: &[f64],
        per_sample_metrics: &[Vec<String>],
        timestamps: &[DateTime<Utc>],
    ) -> Vec<AnomalyDetection> {
        let runs = flagged_runs(scores);
        let mut out = Vec::new();
        for (start, length) in &runs {
            for offset in 0..*length {
                let index = start + offset;
                let (Some(score), Some(timestamp)) = (scores.get(index), timestamps.get(index))
                else {
                    continue;
                };
                let affected = per_sample_metrics.get(index).cloned().unwrap_or_default();
                let anomaly_type = if *length >= COLLECTIVE_RUN {
                    AnomalyType::CollectiveAnomaly
                } else {
                    AnomalyType::StatisticalOutlier
                };
                out.push(AnomalyDetection {
                    timestamp: *timestamp,
                    score: *score,
                    severity: severity_for(*score),
                    anomaly_type,
                    description: format!(
                        "modified z-score {:.2} exceeds the {:.1} threshold on {}",
                        score,
                        MODIFIED_Z_THRESHOLD,
                        if affected.is_empty() {
                            "an unnamed series".to_string()
                        } else {
                            affected.join(", ")
                        }
                    ),
                    // A modified z-score of 7 or more is as certain as this
                    // estimator gets; below that, confidence scales linearly.
                    confidence: ((score - MODIFIED_Z_THRESHOLD) / MODIFIED_Z_THRESHOLD)
                        .clamp(0.0, 1.0),
                    contributing_factors: affected
                        .iter()
                        .map(|metric| {
                            format!(
                                "`{}` deviated from its median by more than {:.1} MAD-scaled units",
                                metric, MODIFIED_Z_THRESHOLD
                            )
                        })
                        .collect(),
                    // Detection observes deviation; it has no model of what to
                    // do about it, so no remediation is suggested.
                    suggested_actions: Vec::new(),
                    affected_metrics: affected,
                });
            }
        }
        out
    }
}

struct RobustBaseline {
    median: f64,
    mad: f64,
}

impl RobustBaseline {
    /// Iglewicz-Hoaglin modified z-score.
    fn modified_z(&self, value: f64) -> f64 {
        if self.mad <= 0.0 {
            return 0.0;
        }
        (0.6745 * (value - self.median) / self.mad).abs()
    }
}

fn robust_baseline(values: &[f64]) -> Option<RobustBaseline> {
    let sorted = sorted_finite(values);
    if sorted.len() < MIN_SAMPLES {
        return None;
    }
    let median = percentile_sorted(&sorted, 0.5)?;
    let deviations =
        sorted_finite(&sorted.iter().map(|v| (v - median).abs()).collect::<Vec<f64>>());
    let mad = percentile_sorted(&deviations, 0.5)?;
    if mad <= super::series::CONSTANT_SERIES_TOLERANCE * median.abs().max(1.0) {
        // A series whose MAD is zero -- or is only rounding noise -- cannot
        // distinguish an outlier from its own constant level; excluding it is
        // more honest than scoring against it.
        return None;
    }
    Some(RobustBaseline { median, mad })
}

/// `(start_index, length)` for every maximal run of flagged samples.
fn flagged_runs(scores: &[f64]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start: Option<usize> = None;
    for (index, score) in scores.iter().enumerate() {
        if *score > MODIFIED_Z_THRESHOLD {
            start.get_or_insert(index);
        } else if let Some(begin) = start.take() {
            runs.push((begin, index - begin));
        }
    }
    if let Some(begin) = start {
        runs.push((begin, scores.len() - begin));
    }
    runs
}

fn anomaly_patterns(scores: &[f64], timestamps: &[DateTime<Utc>]) -> Vec<AnomalyPattern> {
    let runs = flagged_runs(scores);
    if runs.is_empty() {
        return Vec::new();
    }
    let clustered: Vec<&(usize, usize)> =
        runs.iter().filter(|(_, length)| *length >= COLLECTIVE_RUN).collect();
    let mut out = Vec::new();
    if !clustered.is_empty() {
        let time_periods: Vec<(DateTime<Utc>, DateTime<Utc>)> = clustered
            .iter()
            .filter_map(|(start, length)| {
                let begin = timestamps.get(*start).copied()?;
                let end = timestamps.get((start + length).saturating_sub(1)).copied()?;
                Some((begin, end))
            })
            .collect();
        let flagged: usize = clustered.iter().map(|(_, length)| *length).sum();
        out.push(AnomalyPattern {
            pattern_type: "clustered_burst".to_string(),
            frequency: clustered.len() as f64 / scores.len() as f64,
            strength: flagged as f64 / scores.len() as f64,
            time_periods,
            confidence: (flagged as f64 / scores.len() as f64).clamp(0.0, 1.0),
        });
    }
    let isolated: Vec<&(usize, usize)> =
        runs.iter().filter(|(_, length)| *length < COLLECTIVE_RUN).collect();
    if !isolated.is_empty() {
        let time_periods: Vec<(DateTime<Utc>, DateTime<Utc>)> = isolated
            .iter()
            .filter_map(|(start, length)| {
                let begin = timestamps.get(*start).copied()?;
                let end = timestamps.get((start + length).saturating_sub(1)).copied()?;
                Some((begin, end))
            })
            .collect();
        out.push(AnomalyPattern {
            pattern_type: "isolated_spike".to_string(),
            frequency: isolated.len() as f64 / scores.len() as f64,
            strength: isolated.len() as f64 / scores.len() as f64,
            time_periods,
            confidence: (isolated.len() as f64 / scores.len() as f64).clamp(0.0, 1.0),
        });
    }
    out
}

fn severity_for(score: f64) -> AnomalySeverity {
    match score {
        s if s >= 10.0 => AnomalySeverity::Critical,
        s if s >= 7.0 => AnomalySeverity::High,
        s if s >= 5.0 => AnomalySeverity::Medium,
        _ => AnomalySeverity::Low,
    }
}
