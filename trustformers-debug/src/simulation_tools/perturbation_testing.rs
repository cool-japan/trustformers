//! Perturbation Testing and Robustness Assessment
//!
//! This module provides comprehensive perturbation testing capabilities including
//! robustness assessment, sensitivity hotspot identification, and failure mode analysis.

use super::types::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Perturbation testing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationTestResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Base input
    pub base_input: HashMap<String, f64>,
    /// Perturbation test results by intensity
    pub results_by_intensity: HashMap<String, PerturbationIntensityResult>,
    /// Overall robustness assessment
    pub robustness_assessment: RobustnessAssessment,
    /// Sensitivity hotspots
    pub sensitivity_hotspots: Vec<SensitivityHotspot>,
    /// Failure modes analysis
    pub failure_modes: FailureModesAnalysis,
}

/// Perturbation test results for a specific intensity level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationIntensityResult {
    /// Perturbation intensity
    pub intensity: f64,
    /// Number of perturbations tested
    pub num_perturbations: usize,
    /// Successful perturbations (model behavior unchanged)
    pub successful_perturbations: usize,
    /// Failed perturbations (significant behavior change)
    pub failed_perturbations: usize,
    /// Average prediction change
    pub avg_prediction_change: f64,
    /// Maximum prediction change
    pub max_prediction_change: f64,
    /// Standard deviation of prediction changes
    pub std_prediction_change: f64,
    /// Detailed perturbation results
    pub perturbation_details: Vec<PerturbationDetail>,
}

/// Detailed result for a single perturbation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationDetail {
    /// Perturbation ID
    pub id: String,
    /// Original input
    pub original_input: HashMap<String, f64>,
    /// Perturbed input
    pub perturbed_input: HashMap<String, f64>,
    /// Original prediction
    pub original_prediction: f64,
    /// Perturbed prediction
    pub perturbed_prediction: f64,
    /// Prediction change
    pub prediction_change: f64,
    /// Perturbation vector
    pub perturbation_vector: HashMap<String, f64>,
    /// Perturbation magnitude
    pub perturbation_magnitude: f64,
    /// Success/failure classification
    pub is_successful: bool,
}

/// Overall robustness assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessAssessment {
    /// Overall robustness score (0-1)
    pub robustness_score: f64,
    /// Robustness classification
    pub robustness_class: RobustnessClass,
    /// Robustness by feature
    pub feature_robustness: HashMap<String, f64>,
    /// Critical intensity threshold
    pub critical_threshold: f64,
    /// Recommendations for improvement
    pub improvement_recommendations: Vec<String>,
}

/// Sensitivity hotspot identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityHotspot {
    /// Hotspot location in feature space
    pub location: HashMap<String, f64>,
    /// Sensitivity score
    pub sensitivity_score: f64,
    /// Radius of sensitivity
    pub sensitivity_radius: f64,
    /// Primary sensitive features
    pub sensitive_features: Vec<String>,
    /// Hotspot type
    pub hotspot_type: HotspotType,
}

/// Analysis of model failure modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureModesAnalysis {
    /// Identified failure modes
    pub failure_modes: Vec<FailureMode>,
    /// Failure frequency analysis
    pub failure_frequency: FailureFrequencyAnalysis,
    /// Failure severity analysis
    pub failure_severity: FailureSeverityAnalysis,
    /// Failure mitigation strategies
    pub mitigation_strategies: Vec<MitigationStrategy>,
}

/// Individual failure mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureMode {
    /// Failure mode ID
    pub id: String,
    /// Description of the failure
    pub description: String,
    /// Triggering conditions
    pub triggering_conditions: Vec<TriggeringCondition>,
    /// Failure severity
    pub severity: FailureSeverity,
    /// Frequency of occurrence
    pub frequency: f64,
    /// Example failing inputs
    pub example_inputs: Vec<HashMap<String, f64>>,
}

/// Condition that triggers a failure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggeringCondition {
    /// Feature involved
    pub feature: String,
    /// Condition type
    pub condition_type: ConditionType,
    /// Threshold value
    pub threshold: f64,
    /// Condition description
    pub description: String,
}

/// Analysis of failure frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureFrequencyAnalysis {
    /// Overall failure rate
    pub overall_failure_rate: f64,
    /// Failure rate by intensity
    pub failure_rate_by_intensity: HashMap<String, f64>,
    /// Failure rate by feature
    pub failure_rate_by_feature: HashMap<String, f64>,
    /// Time-to-failure analysis
    pub time_to_failure: TimeToFailureAnalysis,
}

/// Analysis of time to failure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeToFailureAnalysis {
    /// Average time to failure
    pub avg_time_to_failure: f64,
    /// Median time to failure
    pub median_time_to_failure: f64,
    /// Time to failure distribution
    pub distribution_parameters: HashMap<String, f64>,
}

/// Analysis of failure severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSeverityAnalysis {
    /// Average failure severity
    pub avg_severity: f64,
    /// Severity distribution
    pub severity_distribution: HashMap<FailureSeverity, usize>,
    /// Most severe failure modes
    pub most_severe_modes: Vec<String>,
    /// Cascading failure analysis
    pub cascading_failures: CascadingFailureAnalysis,
}

/// Analysis of cascading failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadingFailureAnalysis {
    /// Number of cascading events
    pub cascading_events: usize,
    /// Average cascade length
    pub avg_cascade_length: f64,
    /// Cascade triggers
    pub cascade_triggers: Vec<String>,
    /// Cascade amplification factors
    pub amplification_factors: HashMap<String, f64>,
}

/// Strategy for mitigating failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationStrategy {
    /// Strategy name
    pub name: String,
    /// Strategy description
    pub description: String,
    /// Target failure modes
    pub target_failure_modes: Vec<String>,
    /// Expected effectiveness
    pub effectiveness: f64,
    /// Implementation cost
    pub implementation_cost: ImplementationCost,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
}
