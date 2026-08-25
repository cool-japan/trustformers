//! Advanced Gradient Anomaly Detection System
//!
//! This module provides sophisticated anomaly detection capabilities for gradient
//! analysis, including baseline establishment, pattern recognition, and contextual
//! anomaly classification.

use crate::anomaly_detector::{Anomaly, AnomalySeverity};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Advanced gradient anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientAnomalyDetector {
    pub enabled: bool,
    pub sensitivity: f64,
    pub detection_window: usize,
    pub anomaly_history: VecDeque<GradientAnomaly>,
    pub baseline_statistics: HashMap<String, BaselineGradientStats>,
}

impl Default for GradientAnomalyDetector {
    fn default() -> Self {
        Self {
            enabled: true,
            sensitivity: 0.8,
            detection_window: 50,
            anomaly_history: VecDeque::with_capacity(1000),
            baseline_statistics: HashMap::new(),
        }
    }
}

impl GradientAnomalyDetector {
    pub fn new(sensitivity: f64, window_size: usize) -> Self {
        Self {
            enabled: true,
            sensitivity,
            detection_window: window_size,
            anomaly_history: VecDeque::with_capacity(1000),
            baseline_statistics: HashMap::new(),
        }
    }

    pub fn establish_baseline(&mut self, layer_name: &str, gradient_history: &[f64]) {
        if gradient_history.len() < 10 {
            return;
        }

        let mean = gradient_history.iter().sum::<f64>() / gradient_history.len() as f64;
        let variance = gradient_history.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / gradient_history.len() as f64;
        let std = variance.sqrt();

        let mut sorted_values = gradient_history.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_idx = sorted_values.len() / 2;
        let median = if sorted_values.len().is_multiple_of(2) {
            (sorted_values[median_idx - 1] + sorted_values[median_idx]) / 2.0
        } else {
            sorted_values[median_idx]
        };

        let percentile_5_idx = (sorted_values.len() as f64 * 0.05) as usize;
        let percentile_95_idx = (sorted_values.len() as f64 * 0.95) as usize;

        let baseline = BaselineGradientStats {
            mean,
            std,
            median,
            percentile_95: sorted_values[percentile_95_idx.min(sorted_values.len() - 1)],
            percentile_5: sorted_values[percentile_5_idx],
            samples: gradient_history.len(),
        };

        self.baseline_statistics.insert(layer_name.to_string(), baseline);
    }

    pub fn detect_anomalies(
        &mut self,
        layer_name: &str,
        gradient_norm: f64,
        step: usize,
    ) -> Vec<GradientAnomaly> {
        if !self.enabled {
            return Vec::new();
        }

        let baseline = match self.baseline_statistics.get(layer_name) {
            Some(baseline) => baseline,
            None => return Vec::new(), // No baseline established yet
        };

        let mut anomalies = Vec::new();

        // Statistical anomaly detection
        if let Some(anomaly) =
            self.detect_statistical_anomaly(layer_name, gradient_norm, step, baseline)
        {
            anomalies.push(anomaly);
        }

        // Pattern-based anomaly detection
        if let Some(anomaly) = self.detect_pattern_anomaly(layer_name, gradient_norm, step) {
            anomalies.push(anomaly);
        }

        // Add to history
        for anomaly in &anomalies {
            if self.anomaly_history.len() >= 1000 {
                self.anomaly_history.pop_front();
            }
            self.anomaly_history.push_back(anomaly.clone());
        }

        anomalies
    }

    fn detect_statistical_anomaly(
        &self,
        layer_name: &str,
        gradient_norm: f64,
        step: usize,
        baseline: &BaselineGradientStats,
    ) -> Option<GradientAnomaly> {
        let z_score = (gradient_norm - baseline.mean) / baseline.std;
        let threshold = 2.0 + (1.0 - self.sensitivity) * 2.0; // Threshold between 2-4 based on sensitivity

        if z_score.abs() > threshold {
            let anomaly_type = if z_score > 0.0 {
                if z_score > threshold * 1.5 {
                    AnomalyType::SuddenSpike
                } else {
                    AnomalyType::SuddenSpike
                }
            } else {
                AnomalyType::SuddenDrop
            };

            let severity = (z_score.abs() / threshold).min(1.0);

            Some(GradientAnomaly {
                layer_name: layer_name.to_string(),
                anomaly_type,
                severity,
                timestamp: Utc::now(),
                context: AnomalyContext {
                    step,
                    gradient_norm,
                    expected_range: (baseline.percentile_5, baseline.percentile_95),
                    deviation_magnitude: z_score.abs(),
                },
            })
        } else {
            None
        }
    }

    /// Real "expected range" for `layer_name`'s gradient norm, for the
    /// [`AnomalyContext`] attached to pattern-based (as opposed to
    /// single-sample statistical) anomalies. [`Self::detect_anomalies`]
    /// only ever calls [`Self::detect_pattern_anomaly`] after already
    /// confirming a baseline exists for `layer_name` (it early-returns
    /// otherwise), so the same real
    /// `(percentile_5, percentile_95)` used by
    /// [`Self::detect_statistical_anomaly`] is available here too --
    /// reusing it keeps both anomaly kinds reporting the SAME real
    /// "normal" band for a layer, rather than one carrying a measured
    /// range and the other a fabricated `(0.0, 1.0)` regardless of the
    /// layer's actual gradient scale (which is very often << 1.0 or >>
    /// 1.0). The `None` branch is defensive only -- reachable if this
    /// method is ever called directly without going through
    /// `detect_anomalies`'s baseline check -- and falls back to the real
    /// observed min/max gradient norm across the recent pattern window
    /// (still genuine data, never an invented constant).
    fn expected_range_for(
        &self,
        layer_name: &str,
        recent_anomalies: &[&GradientAnomaly],
    ) -> (f64, f64) {
        if let Some(baseline) = self.baseline_statistics.get(layer_name) {
            return (baseline.percentile_5, baseline.percentile_95);
        }
        let norms: Vec<f64> = recent_anomalies.iter().map(|a| a.context.gradient_norm).collect();
        match (
            norms.iter().cloned().fold(f64::INFINITY, f64::min),
            norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        ) {
            (lo, hi) if lo.is_finite() && hi.is_finite() => (lo, hi),
            _ => (0.0, 0.0),
        }
    }

    fn detect_pattern_anomaly(
        &self,
        layer_name: &str,
        gradient_norm: f64,
        step: usize,
    ) -> Option<GradientAnomaly> {
        // Look for patterns in recent anomaly history for this layer
        let recent_anomalies: Vec<&GradientAnomaly> = self
            .anomaly_history
            .iter()
            .filter(|a| a.layer_name == layer_name)
            .rev()
            .take(10)
            .collect();

        if recent_anomalies.len() >= 3 {
            // Check for oscillation pattern
            let oscillation_count = recent_anomalies
                .windows(2)
                .filter(|pair| {
                    matches!(
                        (&pair[0].anomaly_type, &pair[1].anomaly_type),
                        (AnomalyType::SuddenSpike, AnomalyType::SuddenDrop)
                            | (AnomalyType::SuddenDrop, AnomalyType::SuddenSpike)
                    )
                })
                .count();

            if oscillation_count >= 2 {
                return Some(GradientAnomaly {
                    layer_name: layer_name.to_string(),
                    anomaly_type: AnomalyType::Oscillation,
                    severity: 0.7,
                    timestamp: Utc::now(),
                    context: AnomalyContext {
                        step,
                        gradient_norm,
                        expected_range: self.expected_range_for(layer_name, &recent_anomalies),
                        deviation_magnitude: oscillation_count as f64,
                    },
                });
            }
        }

        // Check for stagnation
        if recent_anomalies.len() >= 5 {
            let all_similar = recent_anomalies.windows(2).all(|pair| {
                (pair[0].context.gradient_norm - pair[1].context.gradient_norm).abs() < 1e-6
            });

            if all_similar {
                return Some(GradientAnomaly {
                    layer_name: layer_name.to_string(),
                    anomaly_type: AnomalyType::Stagnation,
                    severity: 0.8,
                    timestamp: Utc::now(),
                    context: AnomalyContext {
                        step,
                        gradient_norm,
                        expected_range: self.expected_range_for(layer_name, &recent_anomalies),
                        deviation_magnitude: 0.0,
                    },
                });
            }
        }

        None
    }

    pub fn get_anomaly_summary(&self, layer_name: Option<&str>) -> AnomalySummary {
        let filtered_anomalies: Vec<&GradientAnomaly> = match layer_name {
            Some(name) => self.anomaly_history.iter().filter(|a| a.layer_name == name).collect(),
            None => self.anomaly_history.iter().collect(),
        };

        let total_anomalies = filtered_anomalies.len();
        let mut anomaly_type_counts = HashMap::new();
        let mut severity_sum = 0.0;

        for anomaly in &filtered_anomalies {
            *anomaly_type_counts.entry(anomaly.anomaly_type.clone()).or_insert(0) += 1;
            severity_sum += anomaly.severity;
        }

        let average_severity =
            if total_anomalies > 0 { severity_sum / total_anomalies as f64 } else { 0.0 };

        // Convert GradientAnomaly to Anomaly objects
        let anomalies: Vec<Anomaly> = filtered_anomalies
            .iter()
            .map(|gradient_anomaly| {
                let severity = if gradient_anomaly.severity >= 0.8 {
                    AnomalySeverity::Critical
                } else if gradient_anomaly.severity >= 0.6 {
                    AnomalySeverity::High
                } else if gradient_anomaly.severity >= 0.3 {
                    AnomalySeverity::Medium
                } else {
                    AnomalySeverity::Low
                };

                // Convert gradient-specific anomaly type to general anomaly type
                let general_anomaly_type = match gradient_anomaly.anomaly_type {
                    AnomalyType::SuddenSpike => {
                        crate::anomaly_detector::AnomalyType::GradientExplosion
                    },
                    AnomalyType::SuddenDrop => {
                        crate::anomaly_detector::AnomalyType::GradientVanishing
                    },
                    AnomalyType::Oscillation => {
                        crate::anomaly_detector::AnomalyType::NumericalInstability
                    },
                    AnomalyType::Stagnation => {
                        crate::anomaly_detector::AnomalyType::GradientVanishing
                    },
                    AnomalyType::Chaos => {
                        crate::anomaly_detector::AnomalyType::NumericalInstability
                    },
                };

                let description = format!(
                    "Gradient anomaly of type {:?} detected with severity {:.2}",
                    gradient_anomaly.anomaly_type, gradient_anomaly.severity
                );

                let mut metadata = HashMap::new();
                metadata.insert(
                    "step".to_string(),
                    gradient_anomaly.context.step.to_string(),
                );
                metadata.insert(
                    "gradient_norm".to_string(),
                    gradient_anomaly.context.gradient_norm.to_string(),
                );
                metadata.insert(
                    "expected_range_min".to_string(),
                    gradient_anomaly.context.expected_range.0.to_string(),
                );
                metadata.insert(
                    "expected_range_max".to_string(),
                    gradient_anomaly.context.expected_range.1.to_string(),
                );
                metadata.insert(
                    "deviation_magnitude".to_string(),
                    gradient_anomaly.context.deviation_magnitude.to_string(),
                );
                metadata.insert(
                    "original_anomaly_type".to_string(),
                    format!("{:?}", gradient_anomaly.anomaly_type),
                );

                Anomaly {
                    anomaly_type: general_anomaly_type,
                    timestamp: gradient_anomaly.timestamp,
                    location: gradient_anomaly.layer_name.clone(),
                    description,
                    severity,
                    metadata,
                }
            })
            .collect();

        AnomalySummary {
            layer_name: layer_name.map(|s| s.to_string()),
            total_anomalies,
            anomaly_type_counts,
            average_severity,
            recent_trend: self.analyze_recent_trend(&filtered_anomalies),
            recommendations: self.generate_anomaly_recommendations(&filtered_anomalies),
            anomalies,
        }
    }

    fn analyze_recent_trend(&self, anomalies: &[&GradientAnomaly]) -> AnomalyTrend {
        if anomalies.len() < 5 {
            return AnomalyTrend::Stable;
        }

        let recent_anomalies: Vec<&GradientAnomaly> =
            anomalies.iter().rev().take(10).cloned().collect();
        let older_anomalies: Vec<&GradientAnomaly> =
            anomalies.iter().rev().skip(10).take(10).cloned().collect();

        if older_anomalies.is_empty() {
            return AnomalyTrend::Stable;
        }

        let recent_avg_severity: f64 = recent_anomalies.iter().map(|a| a.severity).sum::<f64>()
            / recent_anomalies.len() as f64;
        let older_avg_severity: f64 =
            older_anomalies.iter().map(|a| a.severity).sum::<f64>() / older_anomalies.len() as f64;

        let trend_threshold = 0.1;
        if recent_avg_severity > older_avg_severity + trend_threshold {
            AnomalyTrend::Increasing
        } else if recent_avg_severity < older_avg_severity - trend_threshold {
            AnomalyTrend::Decreasing
        } else {
            AnomalyTrend::Stable
        }
    }

    fn generate_anomaly_recommendations(&self, anomalies: &[&GradientAnomaly]) -> Vec<String> {
        let mut recommendations = Vec::new();

        let spike_count = anomalies
            .iter()
            .filter(|a| matches!(a.anomaly_type, AnomalyType::SuddenSpike))
            .count();
        let drop_count = anomalies
            .iter()
            .filter(|a| matches!(a.anomaly_type, AnomalyType::SuddenDrop))
            .count();
        let oscillation_count = anomalies
            .iter()
            .filter(|a| matches!(a.anomaly_type, AnomalyType::Oscillation))
            .count();
        let stagnation_count = anomalies
            .iter()
            .filter(|a| matches!(a.anomaly_type, AnomalyType::Stagnation))
            .count();

        if spike_count > 3 {
            recommendations
                .push("Consider reducing learning rate to prevent gradient explosion".to_string());
            recommendations.push("Add gradient clipping to stabilize training".to_string());
        }

        if drop_count > 3 {
            recommendations.push("Check for vanishing gradient issues".to_string());
            recommendations
                .push("Consider using residual connections or better initialization".to_string());
        }

        if oscillation_count > 2 {
            recommendations.push("Reduce learning rate to dampen oscillations".to_string());
            recommendations
                .push("Consider using momentum or adaptive learning rate methods".to_string());
        }

        if stagnation_count > 2 {
            recommendations.push(
                "Learning may have plateaued - consider learning rate scheduling".to_string(),
            );
            recommendations
                .push("Check for potential convergence or training data issues".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Gradient behavior appears normal".to_string());
        }

        recommendations
    }
}

/// Gradient anomaly event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientAnomaly {
    pub layer_name: String,
    pub anomaly_type: AnomalyType,
    pub severity: f64,
    pub timestamp: DateTime<Utc>,
    pub context: AnomalyContext,
}

/// Types of gradient anomalies
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyType {
    SuddenSpike,
    SuddenDrop,
    Oscillation,
    Stagnation,
    Chaos,
}

/// Context information for anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyContext {
    pub step: usize,
    pub gradient_norm: f64,
    pub expected_range: (f64, f64),
    pub deviation_magnitude: f64,
}

/// Baseline statistics for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineGradientStats {
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub percentile_95: f64,
    pub percentile_5: f64,
    pub samples: usize,
}

/// Summary of anomaly detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalySummary {
    pub layer_name: Option<String>,
    pub total_anomalies: usize,
    pub anomaly_type_counts: HashMap<AnomalyType, usize>,
    pub average_severity: f64,
    pub recent_trend: AnomalyTrend,
    pub recommendations: Vec<String>,
    pub anomalies: Vec<Anomaly>,
}

/// Trend in anomaly occurrence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyTrend {
    Increasing,
    Stable,
    Decreasing,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_anomaly(layer: &str, anomaly_type: AnomalyType, norm: f64) -> GradientAnomaly {
        GradientAnomaly {
            layer_name: layer.to_string(),
            anomaly_type,
            severity: 0.5,
            timestamp: Utc::now(),
            context: AnomalyContext {
                step: 0,
                gradient_norm: norm,
                expected_range: (0.0, 0.0),
                deviation_magnitude: 0.0,
            },
        }
    }

    #[test]
    fn test_oscillation_uses_real_baseline_expected_range_not_placeholder() {
        let mut detector = GradientAnomalyDetector::new(0.8, 50);
        // A baseline whose real "normal" band is nowhere near (0.0, 1.0) --
        // exactly the case the old hardcoded placeholder always got wrong.
        let history: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64 % 3.0)).collect();
        detector.establish_baseline("layer0", &history);
        let baseline = detector.baseline_statistics.get("layer0").cloned().expect("established");

        // Seed a Spike/Drop/Spike history (push order == chronological
        // order, oldest first) so `detect_pattern_anomaly` sees 2 real
        // oscillation transitions.
        detector
            .anomaly_history
            .push_back(make_anomaly("layer0", AnomalyType::SuddenSpike, 500.0));
        detector
            .anomaly_history
            .push_back(make_anomaly("layer0", AnomalyType::SuddenDrop, 1.0));
        detector
            .anomaly_history
            .push_back(make_anomaly("layer0", AnomalyType::SuddenSpike, 500.0));

        let anomaly = detector
            .detect_pattern_anomaly("layer0", 500.0, 99)
            .expect("a 3-entry Spike/Drop/Spike history must trigger an oscillation anomaly");
        assert!(matches!(anomaly.anomaly_type, AnomalyType::Oscillation));
        assert_eq!(
            anomaly.context.expected_range,
            (baseline.percentile_5, baseline.percentile_95),
            "expected_range must be the REAL baseline band, not the old hardcoded (0.0, 1.0)"
        );
        assert_ne!(anomaly.context.expected_range, (0.0, 1.0));
    }

    #[test]
    fn test_stagnation_uses_real_baseline_expected_range_not_placeholder() {
        let mut detector = GradientAnomalyDetector::new(0.8, 50);
        let history: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64 % 3.0)).collect();
        detector.establish_baseline("layer0", &history);
        let baseline = detector.baseline_statistics.get("layer0").cloned().expect("established");

        // 5 near-identical recent gradient norms -> real stagnation
        // pattern (all SuddenDrop, so no oscillation transitions fire
        // first).
        for _ in 0..5 {
            detector.anomaly_history.push_back(make_anomaly(
                "layer0",
                AnomalyType::SuddenDrop,
                42.0,
            ));
        }

        let anomaly = detector
            .detect_pattern_anomaly("layer0", 42.0, 99)
            .expect("5 near-identical recent norms must trigger a stagnation anomaly");
        assert!(matches!(anomaly.anomaly_type, AnomalyType::Stagnation));
        assert_eq!(
            anomaly.context.expected_range,
            (baseline.percentile_5, baseline.percentile_95),
            "expected_range must be the REAL baseline band, not the old hardcoded (0.0, 1.0)"
        );
        assert_ne!(anomaly.context.expected_range, (0.0, 1.0));
    }

    #[test]
    fn test_expected_range_for_falls_back_to_observed_bounds_without_baseline() {
        // Defensive fallback path: no baseline established for this layer
        // (only reachable if `expected_range_for` were ever called outside
        // `detect_anomalies`'s own baseline gate). Must still be real
        // data -- the min/max of what was actually observed -- never a
        // constant.
        let detector = GradientAnomalyDetector::new(0.8, 50);
        let a1 = make_anomaly("orphan", AnomalyType::SuddenSpike, 5.0);
        let a2 = make_anomaly("orphan", AnomalyType::SuddenDrop, 1.0);
        let range = detector.expected_range_for("orphan", &[&a1, &a2]);
        assert_eq!(range, (1.0, 5.0));
    }

    #[test]
    fn test_detect_anomalies_end_to_end_publishes_real_expected_range() {
        // Full public-API path (not the private helpers directly): builds
        // up real oscillation history purely through repeated
        // `detect_anomalies` calls, the way a real caller would.
        let mut detector = GradientAnomalyDetector::new(0.8, 50);
        let history: Vec<f64> = (0..20).map(|i| 10.0 + (i as f64 % 2.0)).collect();
        detector.establish_baseline("layer0", &history);
        let baseline = detector.baseline_statistics.get("layer0").cloned().expect("established");

        let mut last_oscillation = None;
        for (step, norm) in [500.0, 0.001, 500.0, 0.001].into_iter().enumerate() {
            for anomaly in detector.detect_anomalies("layer0", norm, step) {
                if matches!(anomaly.anomaly_type, AnomalyType::Oscillation) {
                    last_oscillation = Some(anomaly);
                }
            }
        }

        let oscillation = last_oscillation.expect(
            "alternating far-above/far-below-baseline norms must \
                 eventually trigger a real oscillation anomaly through the public API",
        );
        assert_eq!(
            oscillation.context.expected_range,
            (baseline.percentile_5, baseline.percentile_95)
        );
        assert_ne!(oscillation.context.expected_range, (0.0, 1.0));
    }
}
