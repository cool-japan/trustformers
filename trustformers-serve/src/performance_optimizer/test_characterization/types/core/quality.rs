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

#[derive(Debug, Clone)]
pub struct ThresholdAnomalyDetector {
    /// Upper threshold
    pub upper_threshold: f64,
    /// Lower threshold
    pub lower_threshold: f64,
    /// Anomalies detected
    pub anomalies_detected: u64,
}

impl ThresholdAnomalyDetector {
    /// Create a new ThresholdAnomalyDetector with default thresholds
    pub fn new() -> Self {
        Self {
            upper_threshold: 100.0,
            lower_threshold: 0.0,
            anomalies_detected: 0,
        }
    }
}

impl Default for ThresholdAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl super::super::analysis::AnomalyDetector for ThresholdAnomalyDetector {
    fn detect(&self) -> String {
        format!(
            "Threshold anomaly detector (upper={:.2}, lower={:.2}, detected={})",
            self.upper_threshold, self.lower_threshold, self.anomalies_detected
        )
    }

    fn detect_anomalies(&self) -> TestCharacterizationResult<Vec<AnomalyInfo>> {
        // Placeholder implementation - in real use, this would check values against thresholds
        // For now, return empty vec indicating no anomalies detected
        Ok(Vec::new())
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
