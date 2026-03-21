//! Statistical Types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

// Import common types

// Import types from sibling modules
use super::enums::{ObjectiveType, TrendDirection};
use super::metrics::ImpactAssessment;

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

// ADDITIONAL OPTIMIZATION AND STATISTICAL TYPES
// =============================================================================

/// Threshold evaluation result
///
/// Result from threshold evaluation including violation status
/// and contextual information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdEvaluation {
    /// Evaluation timestamp
    pub timestamp: DateTime<Utc>,

    /// Threshold violated
    pub violated: bool,

    /// Violation severity
    pub severity: SeverityLevel,

    /// Current value
    pub current_value: f64,

    /// Threshold value
    pub threshold_value: f64,

    /// Evaluation confidence
    pub confidence: f32,

    /// Evaluation metadata
    pub metadata: HashMap<String, String>,
}

impl ThresholdEvaluation {
    /// Create new threshold evaluation
    pub fn new(violated: bool, current_value: f64, threshold_value: f64) -> Self {
        let severity = if violated {
            if (current_value - threshold_value).abs() > threshold_value * 0.5 {
                SeverityLevel::Critical
            } else if (current_value - threshold_value).abs() > threshold_value * 0.2 {
                SeverityLevel::High
            } else {
                SeverityLevel::Medium
            }
        } else {
            SeverityLevel::Info
        };

        Self {
            timestamp: Utc::now(),
            violated,
            severity,
            current_value,
            threshold_value,
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Get violation magnitude
    pub fn violation_magnitude(&self) -> f64 {
        if self.violated {
            (self.current_value - self.threshold_value).abs()
        } else {
            0.0
        }
    }

    /// Get violation percentage
    pub fn violation_percentage(&self) -> f64 {
        if self.threshold_value != 0.0 {
            self.violation_magnitude() / self.threshold_value.abs() * 100.0
        } else {
            0.0
        }
    }
}

/// Optimization context for algorithms
///
/// Context information provided to optimization algorithms including
/// system state, constraints, and optimization objectives.
#[derive(Debug, Clone)]
pub struct OptimizationContext {
    /// Current system state
    pub system_state: SystemState,

    /// Test characteristics
    pub test_characteristics: TestCharacteristics,

    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,

    /// Constraints
    pub constraints: HashMap<String, f64>,

    /// Historical performance
    pub historical_performance: Vec<PerformanceDataPoint>,

    /// Context metadata
    pub metadata: HashMap<String, String>,

    /// Optimization window
    pub optimization_window: Duration,

    /// Maximum optimization time
    pub max_optimization_time: Duration,
}

impl OptimizationContext {
    /// Create new optimization context
    pub fn new(system_state: SystemState, test_characteristics: TestCharacteristics) -> Self {
        Self {
            system_state,
            test_characteristics,
            objectives: Vec::new(),
            constraints: HashMap::new(),
            historical_performance: Vec::new(),
            metadata: HashMap::new(),
            optimization_window: Duration::from_secs(300),
            max_optimization_time: Duration::from_secs(60),
        }
    }

    /// Add optimization objective
    pub fn add_objective(&mut self, objective: OptimizationObjective) {
        self.objectives.push(objective);
    }

    /// Add constraint
    pub fn add_constraint(&mut self, name: String, value: f64) {
        self.constraints.insert(name, value);
    }

    /// Check if context has conflicting objectives
    pub fn has_conflicting_objectives(&self) -> bool {
        // Simplified conflict detection
        let has_minimize = self.objectives.iter().any(|obj| {
            matches!(
                obj.objective_type,
                ObjectiveType::MinimizeLatency
                    | ObjectiveType::MinimizeResourceUsage
                    | ObjectiveType::MinimizeCost
            )
        });
        let has_maximize = self.objectives.iter().any(|obj| {
            matches!(
                obj.objective_type,
                ObjectiveType::MaximizeThroughput
                    | ObjectiveType::MaximizeEfficiency
                    | ObjectiveType::MaximizeReliability
            )
        });
        has_minimize && has_maximize
    }
}

/// Optimization objective
///
/// Individual optimization objective with weight, target, and constraints
/// for multi-objective optimization scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationObjective {
    /// Objective name
    pub name: String,

    /// Objective type
    pub objective_type: ObjectiveType,

    /// Target value
    pub target: f64,

    /// Objective weight
    pub weight: f32,

    /// Constraints
    pub constraints: HashMap<String, f64>,

    /// Priority level
    pub priority: u8,

    /// Tolerance range
    pub tolerance: f64,
}

impl OptimizationObjective {
    /// Create new optimization objective
    pub fn new(name: String, objective_type: ObjectiveType, target: f64, weight: f32) -> Self {
        Self {
            name,
            objective_type,
            target,
            weight,
            constraints: HashMap::new(),
            priority: 1,
            tolerance: 0.05, // 5% tolerance
        }
    }

    /// Check if current value meets objective
    pub fn is_met(&self, current_value: f64) -> bool {
        let diff = (current_value - self.target).abs();
        diff <= self.tolerance * self.target.abs()
    }

    /// Calculate objective score
    pub fn score(&self, current_value: f64) -> f32 {
        let diff = (current_value - self.target).abs();
        let normalized_diff = if self.target != 0.0 { diff / self.target.abs() } else { diff };

        (1.0 - normalized_diff as f32).clamp(0.0, 1.0)
    }
}

/// Optimization recommendation
///
/// Recommendation generated by optimization algorithms including
/// actions, expected impact, and confidence scores.
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation ID
    pub id: String,

    /// Recommendation timestamp
    pub timestamp: DateTime<Utc>,

    /// Recommended actions
    pub actions: Vec<RecommendedAction>,

    /// Expected performance impact
    pub expected_impact: ImpactAssessment,

    /// Recommendation confidence
    pub confidence: f32,

    /// Supporting analysis
    pub analysis: String,

    /// Risk assessment
    pub risks: Vec<RiskFactor>,

    /// Implementation priority
    pub priority: u8,

    /// Expected implementation time
    pub implementation_time: Duration,
}

impl OptimizationRecommendation {
    /// Create new optimization recommendation
    pub fn new(id: String, actions: Vec<RecommendedAction>, confidence: f32) -> Self {
        Self {
            id,
            timestamp: Utc::now(),
            actions,
            expected_impact: ImpactAssessment::default(),
            confidence,
            analysis: String::new(),
            risks: Vec::new(),
            priority: 1,
            implementation_time: Duration::from_secs(300),
        }
    }

    /// Add risk factor
    pub fn add_risk(&mut self, risk: RiskFactor) {
        self.risks.push(risk);
    }

    /// Check if recommendation is high priority
    pub fn is_high_priority(&self) -> bool {
        self.priority >= 3 && self.confidence >= 0.8
    }

    /// Calculate overall recommendation score
    pub fn overall_score(&self) -> f32 {
        let impact_score = self.expected_impact.overall_score();
        let confidence_weight = self.confidence;
        let priority_weight = self.priority as f32 / 5.0;

        (impact_score * confidence_weight * priority_weight).min(1.0)
    }
}

/// Risk factor for recommendations
///
/// Risk factor associated with implementing optimization recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Risk type
    pub risk_type: String,

    /// Risk description
    pub description: String,

    /// Risk probability (0.0 to 1.0)
    pub probability: f32,

    /// Risk impact (0.0 to 1.0)
    pub impact: f32,

    /// Risk severity
    pub severity: SeverityLevel,

    /// Mitigation strategies
    pub mitigation: Vec<String>,
}

impl RiskFactor {
    /// Create new risk factor
    pub fn new(risk_type: String, description: String, probability: f32, impact: f32) -> Self {
        let severity = match probability * impact {
            x if x >= 0.7 => SeverityLevel::Critical,
            x if x >= 0.5 => SeverityLevel::High,
            x if x >= 0.3 => SeverityLevel::Medium,
            x if x >= 0.1 => SeverityLevel::Low,
            _ => SeverityLevel::Info,
        };

        Self {
            risk_type,
            description,
            probability,
            impact,
            severity,
            mitigation: Vec::new(),
        }
    }

    /// Calculate risk score
    pub fn risk_score(&self) -> f32 {
        self.probability * self.impact
    }

    /// Add mitigation strategy
    pub fn add_mitigation(&mut self, strategy: String) {
        self.mitigation.push(strategy);
    }
}

/// Statistical result from processing
///
/// Statistical analysis result including various statistical measures
/// and analysis metadata.
#[derive(Debug, Clone)]
pub struct StatisticalResult {
    /// Result timestamp
    pub timestamp: DateTime<Utc>,

    /// Statistical measures
    pub measures: HashMap<String, f64>,

    /// Distribution analysis
    pub distribution: DistributionAnalysis,

    /// Correlation analysis
    pub correlations: HashMap<String, f32>,

    /// Trend analysis
    pub trends: Vec<TrendAnalysis>,

    /// Analysis confidence
    pub confidence: f32,

    /// Sample size
    pub sample_size: usize,

    /// Analysis window
    pub analysis_window: Duration,
}

impl StatisticalResult {
    /// Create new statistical result
    pub fn new(sample_size: usize, analysis_window: Duration) -> Self {
        Self {
            timestamp: Utc::now(),
            measures: HashMap::new(),
            distribution: DistributionAnalysis::default(),
            correlations: HashMap::new(),
            trends: Vec::new(),
            confidence: 0.0,
            sample_size,
            analysis_window,
        }
    }

    /// Add statistical measure
    pub fn add_measure(&mut self, name: String, value: f64) {
        self.measures.insert(name, value);
    }
}

impl Default for StatisticalResult {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            measures: HashMap::new(),
            distribution: DistributionAnalysis::default(),
            correlations: HashMap::new(),
            trends: Vec::new(),
            confidence: 0.0,
            sample_size: 0,
            analysis_window: Duration::from_secs(60),
        }
    }
}

impl StatisticalResult {
    /// Get measure value
    pub fn get_measure(&self, name: &str) -> Option<f64> {
        self.measures.get(name).copied()
    }

    /// Check if result is statistically significant
    pub fn is_significant(&self) -> bool {
        self.confidence >= 0.95 && self.sample_size >= 30
    }
}

/// Distribution analysis results
///
/// Analysis of data distribution including type, parameters, and
/// goodness of fit measures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    /// Distribution type
    pub distribution_type: String,

    /// Distribution parameters
    pub parameters: HashMap<String, f64>,

    /// Goodness of fit score
    pub goodness_of_fit: f32,

    /// Statistical tests
    pub tests: HashMap<String, f64>,

    /// Confidence level
    pub confidence_level: f32,
}

impl Default for DistributionAnalysis {
    fn default() -> Self {
        Self {
            distribution_type: "normal".to_string(),
            parameters: HashMap::new(),
            goodness_of_fit: 0.0,
            tests: HashMap::new(),
            confidence_level: 0.95,
        }
    }
}

impl DistributionAnalysis {
    /// Check if distribution is normal
    pub fn is_normal(&self) -> bool {
        self.distribution_type == "normal" && self.goodness_of_fit >= 0.8
    }

    /// Get distribution parameter
    pub fn get_parameter(&self, name: &str) -> Option<f64> {
        self.parameters.get(name).copied()
    }

    /// Add statistical test result
    pub fn add_test(&mut self, test_name: String, p_value: f64) {
        self.tests.insert(test_name, p_value);
    }
}

/// Trend analysis results
///
/// Analysis of trends in the data including direction, strength,
/// and statistical significance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend metric
    pub metric: String,

    /// Trend direction
    pub direction: TrendDirection,

    /// Trend strength
    pub strength: f32,

    /// Statistical significance
    pub significance: f32,

    /// Trend duration
    #[serde(skip)]
    pub duration: Duration,

    /// Slope coefficient
    pub slope: f64,

    /// R-squared value
    pub r_squared: f32,

    /// Confidence level of the trend analysis
    pub confidence: f32,
}

impl Default for TrendAnalysis {
    fn default() -> Self {
        Self {
            metric: String::new(),
            direction: TrendDirection::Unknown,
            strength: 0.0,
            significance: 0.0,
            duration: Duration::from_secs(0),
            slope: 0.0,
            r_squared: 0.0,
            confidence: 0.0,
        }
    }
}

impl TrendAnalysis {
    /// Create new trend analysis
    pub fn new(
        metric: String,
        direction: TrendDirection,
        strength: f32,
        duration: Duration,
    ) -> Self {
        Self {
            metric,
            direction,
            strength,
            significance: 0.0,
            duration,
            slope: 0.0,
            r_squared: 0.0,
            confidence: 0.5,
        }
    }

    /// Check if trend is statistically significant
    pub fn is_significant(&self) -> bool {
        self.significance >= 0.95 && self.strength >= 0.3
    }

    /// Check if trend is strong
    pub fn is_strong(&self) -> bool {
        self.strength >= 0.7 && self.r_squared >= 0.5
    }
}

// =============================================================================
