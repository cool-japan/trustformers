//! Data-quality analysis of an observed metrics window.
//!
//! Replaces the `create_analyzer_placeholder!`-generated `QualityAnalyzer`,
//! which ignored its input and always reported `overall_score: 0.9` with seven
//! invented dimension scores. Each dimension below is either measured from the
//! window or reported as `None` because this crate has nothing to measure it
//! against.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use chrono::Utc;

use super::super::super::types::data_structures::TimestampedMetrics;
use super::super::super::types::enums::{QualityIssueType, TrendDirection};
use super::super::types::{
    DataQualityIssue, QualityAnalysisResult, QualityDimensions, QualityTrend,
};
use super::super::types::{QualityRecommendation, QualityThresholds};
use super::series::{extract_series, mean, sample_std_dev, SERIES_NAMES};
use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

/// Assesses the quality of a metrics window against configured thresholds.
#[derive(Clone, Debug)]
pub struct QualityAnalyzer {
    thresholds: QualityThresholds,
    shutdown: Arc<AtomicBool>,
}

impl QualityAnalyzer {
    /// Create a quality analyzer bound to the caller's thresholds.
    pub async fn new(thresholds: QualityThresholds) -> Result<Self> {
        Ok(Self {
            thresholds,
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Reject a window that fails the hard input pre-conditions.
    ///
    /// Called before the expensive analyses so a bad window fails fast with a
    /// message naming what was wrong.
    pub async fn validate_input_data(&self, data: &[TimestampedMetrics]) -> Result<()> {
        if data.is_empty() {
            return Err(anyhow!("No data provided for validation"));
        }
        let invalid = data.iter().filter(|s| !sample_is_valid(s)).count();
        if invalid == data.len() {
            return Err(anyhow!(
                "All {} samples carry out-of-domain metric values (negative throughput/latency \
                 or an error rate outside 0..=1)",
                data.len()
            ));
        }
        Ok(())
    }

    /// Measure the quality dimensions of `data`.
    pub async fn analyze(&self, data: &[TimestampedMetrics]) -> Result<QualityAnalysisResult> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(anyhow!("Quality analyzer is shut down"));
        }
        if data.is_empty() {
            return Err(anyhow!("Quality analysis requires at least one sample"));
        }
        let dimensions = self.measure(data);
        let issues = self.collect_issues(data, &dimensions);
        let trends = self.measure_trends(data);
        let recommendations = self.recommend(&dimensions, &issues);
        let measured: Vec<f64> = [
            Some(dimensions.completeness),
            Some(dimensions.consistency),
            Some(dimensions.timeliness),
            Some(dimensions.validity),
            Some(dimensions.uniqueness),
            dimensions.accuracy,
            dimensions.integrity,
        ]
        .into_iter()
        .flatten()
        .collect();
        let overall_score = mean(&measured).unwrap_or(0.0);
        // Confidence saturates once the window reaches the configured minimum
        // sample size; a short window supports only a weak claim.
        let confidence = if self.thresholds.min_sample_size == 0 {
            1.0
        } else {
            (data.len() as f64 / self.thresholds.min_sample_size as f64).clamp(0.0, 1.0)
        };
        Ok(QualityAnalysisResult {
            overall_score,
            dimensions,
            issues,
            trends,
            recommendations,
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

    fn measure(&self, data: &[TimestampedMetrics]) -> QualityDimensions {
        let series = extract_series(data);
        let complete = series.iter().filter(|s| s.values.len() == data.len()).count();
        let completeness = complete as f64 / SERIES_NAMES.len() as f64;

        let now = Utc::now();
        let fresh = data
            .iter()
            .filter(|s| {
                now.signed_duration_since(s.timestamp)
                    .to_std()
                    .map(|age| age <= self.thresholds.max_staleness)
                    .unwrap_or(false)
            })
            .count();
        let timeliness = fresh as f64 / data.len() as f64;

        let valid = data.iter().filter(|s| sample_is_valid(s)).count();
        let validity = valid as f64 / data.len() as f64;

        let mut seen = std::collections::BTreeSet::new();
        for sample in data {
            seen.insert(sample.timestamp.timestamp_nanos_opt().unwrap_or(0));
        }
        let uniqueness = seen.len() as f64 / data.len() as f64;

        QualityDimensions {
            completeness,
            // No reference measurement exists to compare these readings against,
            // so accuracy is not measurable from the window alone.
            accuracy: None,
            consistency: self.sampling_regularity(data),
            timeliness,
            validity,
            uniqueness,
            // Integrity would need cross-record referential constraints; a
            // metrics window declares none.
            integrity: None,
        }
    }

    /// One minus the coefficient of variation of the inter-sample interval.
    fn sampling_regularity(&self, data: &[TimestampedMetrics]) -> f64 {
        let gaps: Vec<f64> = data
            .windows(2)
            .filter_map(|pair| {
                let (Some(first), Some(second)) = (pair.first(), pair.get(1)) else {
                    return None;
                };
                let millis =
                    second.timestamp.signed_duration_since(first.timestamp).num_milliseconds();
                if millis > 0 {
                    Some(millis as f64)
                } else {
                    None
                }
            })
            .collect();
        let (Some(m), Some(sd)) = (mean(&gaps), sample_std_dev(&gaps)) else {
            // A single-sample window has no interval to judge; report the
            // neutral 1.0 only when there is genuinely nothing irregular.
            return if gaps.is_empty() { 1.0 } else { 0.0 };
        };
        if m <= 0.0 {
            return 0.0;
        }
        (1.0 - sd / m).clamp(0.0, 1.0)
    }

    fn collect_issues(
        &self,
        data: &[TimestampedMetrics],
        dimensions: &QualityDimensions,
    ) -> Vec<DataQualityIssue> {
        let mut issues = Vec::new();
        let now = Utc::now();
        let stale = data
            .iter()
            .filter(|s| {
                now.signed_duration_since(s.timestamp)
                    .to_std()
                    .map(|age| age > self.thresholds.max_staleness)
                    .unwrap_or(true)
            })
            .count();
        if stale > 0 {
            issues.push(DataQualityIssue {
                issue_type: QualityIssueType::StaleData,
                severity: severity_for(stale as f64 / data.len() as f64),
                description: format!(
                    "{} of {} samples are older than the {:?} staleness threshold",
                    stale,
                    data.len(),
                    self.thresholds.max_staleness
                ),
                affected_count: stale as u64,
                location: "timestamp".to_string(),
                remediation: "Shorten the collection interval or widen max_staleness".to_string(),
            });
        }
        let invalid = data.iter().filter(|s| !sample_is_valid(s)).count();
        if invalid > 0 {
            issues.push(DataQualityIssue {
                issue_type: QualityIssueType::CorruptedData,
                severity: severity_for(invalid as f64 / data.len() as f64),
                description: format!(
                    "{} of {} samples carry a negative throughput/latency or an error rate \
                     outside 0..=1",
                    invalid,
                    data.len()
                ),
                affected_count: invalid as u64,
                location: "metrics".to_string(),
                remediation: "Fix the collector emitting out-of-domain readings".to_string(),
            });
        }
        if dimensions.uniqueness < 1.0 {
            let duplicates =
                data.len() - (dimensions.uniqueness * data.len() as f64).round().max(0.0) as usize;
            issues.push(DataQualityIssue {
                issue_type: QualityIssueType::DuplicateData,
                severity: severity_for(1.0 - dimensions.uniqueness),
                description: format!(
                    "{} samples share a timestamp with another sample",
                    duplicates
                ),
                affected_count: duplicates as u64,
                location: "timestamp".to_string(),
                remediation: "De-duplicate on timestamp before analysis".to_string(),
            });
        }
        if dimensions.completeness < self.thresholds.min_completeness {
            let missing = extract_series(data)
                .into_iter()
                .filter(|s| s.values.len() != data.len())
                .map(|s| s.name)
                .collect::<Vec<_>>()
                .join(", ");
            issues.push(DataQualityIssue {
                issue_type: QualityIssueType::MissingData,
                severity: severity_for(1.0 - dimensions.completeness),
                description: format!("Metric series with no finite readings: {}", missing),
                affected_count: data.len() as u64,
                location: "metric series".to_string(),
                remediation: "Enable the collectors for the missing series".to_string(),
            });
        }
        if dimensions.consistency < 0.5 {
            issues.push(DataQualityIssue {
                issue_type: QualityIssueType::InconsistentData,
                severity: severity_for(1.0 - dimensions.consistency),
                description: format!(
                    "Sampling interval varies widely (regularity {:.3})",
                    dimensions.consistency
                ),
                affected_count: data.len() as u64,
                location: "timestamp".to_string(),
                remediation: "Stabilise the collection loop's tick".to_string(),
            });
        }
        issues
    }

    /// Compare each measurable dimension between the window's two halves.
    fn measure_trends(&self, data: &[TimestampedMetrics]) -> Vec<QualityTrend> {
        if data.len() < 4 {
            return Vec::new();
        }
        let midpoint = data.len() / 2;
        let (Some(first), Some(second)) = (data.get(..midpoint), data.get(midpoint..)) else {
            return Vec::new();
        };
        let before = self.measure(first);
        let after = self.measure(second);
        let (Some(start), Some(end)) = (data.first(), data.last()) else {
            return Vec::new();
        };
        let period = (start.timestamp, end.timestamp);
        [
            ("completeness", before.completeness, after.completeness),
            ("consistency", before.consistency, after.consistency),
            ("timeliness", before.timeliness, after.timeliness),
            ("validity", before.validity, after.validity),
            ("uniqueness", before.uniqueness, after.uniqueness),
        ]
        .into_iter()
        .map(|(dimension, from, to)| {
            let delta = to - from;
            QualityTrend {
                dimension: dimension.to_string(),
                direction: if delta > 0.01 {
                    TrendDirection::Increasing
                } else if delta < -0.01 {
                    TrendDirection::Decreasing
                } else {
                    TrendDirection::Stable
                },
                strength: delta.abs().clamp(0.0, 1.0),
                time_period: period,
                // Half-window comparison over `n` samples: the significance a
                // single split supports is the magnitude of the change itself.
                significance: delta.abs().clamp(0.0, 1.0),
            }
        })
        .collect()
    }

    fn recommend(
        &self,
        dimensions: &QualityDimensions,
        issues: &[DataQualityIssue],
    ) -> Vec<QualityRecommendation> {
        let mut out = Vec::new();
        if dimensions.completeness < self.thresholds.min_completeness {
            out.push(QualityRecommendation {
                recommendation_type: "completeness".to_string(),
                priority: 1,
                description: format!(
                    "Enable the missing collectors: completeness is {:.3}, threshold {:.3}",
                    dimensions.completeness, self.thresholds.min_completeness
                ),
                expected_improvement: self.thresholds.min_completeness - dimensions.completeness,
                implementation_effort: "collector configuration".to_string(),
                // Cost is not observable from a metrics window.
                cost_benefit_ratio: None,
            });
        }
        if dimensions.validity < 1.0 {
            out.push(QualityRecommendation {
                recommendation_type: "validity".to_string(),
                priority: 1,
                description: format!(
                    "Reject out-of-domain readings at the collector: validity is {:.3}",
                    dimensions.validity
                ),
                expected_improvement: 1.0 - dimensions.validity,
                implementation_effort: "collector validation".to_string(),
                cost_benefit_ratio: None,
            });
        }
        if dimensions.timeliness < 1.0 {
            out.push(QualityRecommendation {
                recommendation_type: "timeliness".to_string(),
                priority: 2,
                description: format!(
                    "Reduce collection latency: timeliness is {:.3}",
                    dimensions.timeliness
                ),
                expected_improvement: 1.0 - dimensions.timeliness,
                implementation_effort: "collection interval".to_string(),
                cost_benefit_ratio: None,
            });
        }
        if out.is_empty() && !issues.is_empty() {
            out.push(QualityRecommendation {
                recommendation_type: "review".to_string(),
                priority: 3,
                description: format!(
                    "{} quality issue(s) recorded without breaching a configured threshold",
                    issues.len()
                ),
                expected_improvement: 0.0,
                implementation_effort: "review".to_string(),
                cost_benefit_ratio: None,
            });
        }
        out
    }
}

fn sample_is_valid(sample: &TimestampedMetrics) -> bool {
    let metrics = &sample.metrics;
    metrics.throughput >= 0.0
        && metrics.throughput.is_finite()
        && (0.0..=1.0).contains(&metrics.error_rate)
        && metrics.resource_usage.cpu_usage >= 0.0
        && metrics.resource_usage.memory_usage >= 0.0
}

fn severity_for(fraction: f64) -> SeverityLevel {
    match fraction {
        f if f >= 0.5 => SeverityLevel::Critical,
        f if f >= 0.25 => SeverityLevel::High,
        f if f >= 0.1 => SeverityLevel::Medium,
        f if f > 0.0 => SeverityLevel::Low,
        _ => SeverityLevel::Info,
    }
}
