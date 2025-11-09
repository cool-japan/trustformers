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
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_idx = sorted_values.len() / 2;
        let median = if sorted_values.len() % 2 == 0 {
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
                        expected_range: (0.0, 1.0), // Placeholder
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
                        expected_range: (0.0, 1.0), // Placeholder
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
