//! Quality and validation types for test characterization

use super::super::analysis::AnomalyInfo;
use super::super::quality::SafetyValidationRule;
use super::enums::TestCharacterizationResult;

#[derive(Debug, Clone)]
pub struct IsolationSafetyRule {
    pub isolation_level: String,
    pub enforce_boundaries: bool,
    pub cross_contamination_check: bool,
}

impl IsolationSafetyRule {
    pub fn new() -> Self {
        Self {
            isolation_level: "default".to_string(),
            enforce_boundaries: true,
            cross_contamination_check: true,
        }
    }
}

impl Default for IsolationSafetyRule {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetyValidationRule for IsolationSafetyRule {
    fn validate(&self) -> bool {
        self.enforce_boundaries && self.cross_contamination_check
    }

    fn name(&self) -> &str {
        "IsolationSafetyRule"
    }
}

/// Flags readings outside a configured band.
///
/// The `anomalies_detected` counter was removed in 0.2.1: nothing ever
/// incremented it, so it reported zero detections forever while
/// `detect_anomalies` returned an empty vector regardless of its input.
#[derive(Debug, Clone)]
pub struct ThresholdAnomalyDetector {
    /// Upper threshold
    pub upper_threshold: f64,
    /// Lower threshold
    pub lower_threshold: f64,
}

impl ThresholdAnomalyDetector {
    /// Create a new ThresholdAnomalyDetector with default thresholds
    pub fn new() -> Self {
        Self {
            upper_threshold: 100.0,
            lower_threshold: 0.0,
        }
    }
}

impl Default for ThresholdAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl super::super::analysis::AnomalyDetector for ThresholdAnomalyDetector {
    fn describe(&self) -> String {
        format!(
            "Threshold anomaly detector: flags readings outside [{:.2}, {:.2}]",
            self.lower_threshold, self.upper_threshold
        )
    }

    fn detect_anomalies(
        &self,
        observations: super::super::analysis::InsightObservations<'_>,
        _baseline: &super::BaselineModel,
    ) -> TestCharacterizationResult<Vec<AnomalyInfo>> {
        if self.upper_threshold <= self.lower_threshold {
            return Err(super::TestCharacterizationError::InvalidInput {
                message: "upper threshold must exceed the lower threshold".to_string(),
                field: "upper_threshold".to_string(),
                value: self.upper_threshold.to_string(),
            });
        }
        let band = self.upper_threshold - self.lower_threshold;
        let mut anomalies = Vec::new();
        for key in observations.keys() {
            for value in observations.series(&key) {
                let excess = if value > self.upper_threshold {
                    value - self.upper_threshold
                } else if value < self.lower_threshold {
                    self.lower_threshold - value
                } else {
                    continue;
                };
                anomalies.push(super::super::analysis::anomaly_from_deviation(
                    "threshold",
                    super::super::analysis::AnomalyType::Statistical,
                    &key,
                    1.0 + excess / band,
                    format!(
                        "`{}` read {:.4}, outside the configured band [{:.2}, {:.2}]",
                        key, value, self.lower_threshold, self.upper_threshold
                    ),
                ));
            }
        }
        Ok(anomalies)
    }
}

pub struct IndicatorStatus {
    pub status: String,
    pub health_score: f64,
}

pub struct ThresholdDirection {
    pub direction: String,
    pub is_upper_bound: bool,
    pub is_lower_bound: bool,
}

pub struct ThresholdEvaluatorType {
    pub evaluator_type: String,
    pub algorithm: String,
    pub sensitivity: f64,
}

pub struct CriticalIssue {
    pub issue_type: String,
    pub severity: u8,
    pub description: String,
}

pub struct CriticalityLevel {
    pub level: u8,
    pub level_name: String,
}

pub struct TestStatus {
    pub status: String,
    pub passed: bool,
    pub failed: bool,
    pub skipped: bool,
    pub error_message: Option<String>,
}

// Prevention action type that was incorrectly placed
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreventionAction {
    pub action_id: String,
    pub action_type: String,
    pub description: String,
    pub priority: super::enums::PriorityLevel,
    pub urgency: super::enums::UrgencyLevel,
    pub estimated_effort: String,
    pub expected_impact: f64,
    pub implementation_steps: Vec<String>,
    pub verification_steps: Vec<String>,
    pub rollback_plan: String,
    pub dependencies: Vec<String>,
    pub constraints: Vec<String>,
    #[serde(skip)]
    pub estimated_completion_time: std::time::Duration,
    pub risk_mitigation_score: f64,
}
