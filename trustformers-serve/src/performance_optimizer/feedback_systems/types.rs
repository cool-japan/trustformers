//! Core Types for Enhanced Feedback Systems
//!
//! This module provides comprehensive type definitions for advanced feedback processing
//! including quality assessment, validation, risk analysis, action recommendations,
//! and multi-objective optimization strategies. These types form the foundation for
//! all feedback system operations and enable sophisticated performance analysis.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

use crate::performance_optimizer::types::{FeedbackSource, RecommendedAction};

// =============================================================================
// FEEDBACK QUALITY AND VALIDATION TYPES
// =============================================================================

/// Feedback quality assessment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackQualityMetrics {
    /// Reliability score (0.0 to 1.0)
    pub reliability: f32,
    /// Relevance score (0.0 to 1.0)
    pub relevance: f32,
    /// Timeliness score (0.0 to 1.0)
    pub timeliness: f32,
    /// Completeness score (0.0 to 1.0)
    pub completeness: f32,
    /// Consistency score (0.0 to 1.0)
    pub consistency: f32,
    /// Overall quality score (0.0 to 1.0)
    pub overall_quality: f32,
    /// Quality assessment timestamp
    pub assessed_at: DateTime<Utc>,
}

/// Feedback validation result
#[derive(Debug, Clone)]
pub struct FeedbackValidationResult {
    /// Validation passed
    pub valid: bool,
    /// Validation score (0.0 to 1.0)
    pub score: f32,
    /// Validation issues
    pub issues: Vec<ValidationIssue>,
    /// Quality metrics
    pub quality_metrics: FeedbackQualityMetrics,
    /// Recommended corrections
    pub recommended_corrections: Vec<FeedbackCorrection>,
}

/// Feedback validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Issue type
    pub issue_type: ValidationIssueType,
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue description
    pub description: String,
    /// Suggested resolution
    pub suggested_resolution: Option<String>,
}

/// Types of validation issues
#[derive(Debug, Clone)]
pub enum ValidationIssueType {
    /// Data integrity issue
    DataIntegrity,
    /// Temporal inconsistency
    TemporalInconsistency,
    /// Value out of range
    ValueOutOfRange,
    /// Missing context
    MissingContext,
    /// Source unreliable
    SourceUnreliable,
    /// Format error
    FormatError,
    /// Custom issue
    Custom(String),
}

/// Issue severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Feedback correction recommendation
#[derive(Debug, Clone)]
pub struct FeedbackCorrection {
    /// Correction type
    pub correction_type: CorrectionType,
    /// Original value
    pub original_value: f64,
    /// Corrected value
    pub corrected_value: f64,
    /// Correction confidence
    pub confidence: f32,
    /// Correction rationale
    pub rationale: String,
}

/// Types of feedback corrections
#[derive(Debug, Clone)]
pub enum CorrectionType {
    /// Outlier removal
    OutlierRemoval,
    /// Smoothing
    Smoothing,
    /// Normalization
    Normalization,
    /// Gap filling
    GapFilling,
    /// Noise reduction
    NoiseReduction,
    /// Custom correction
    Custom(String),
}

// =============================================================================
// ACTION RECOMMENDATION TYPES
// =============================================================================

/// Action recommendation priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ActionPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Enhanced action recommendation with detailed analysis
#[derive(Debug, Clone)]
pub struct EnhancedRecommendedAction {
    /// Base action
    pub base_action: RecommendedAction,
    /// Action category
    pub category: ActionCategory,
    /// Priority level
    pub priority_level: ActionPriority,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Implementation complexity
    pub complexity: ActionComplexity,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    /// Success probability
    pub success_probability: f32,
    /// Alternative actions
    pub alternatives: Vec<RecommendedAction>,
    /// Dependencies
    pub dependencies: Vec<String>,
    /// Contraindications
    pub contraindications: Vec<String>,
    /// Action description
    pub action: String,
    /// Confidence level
    pub confidence: f32,
    /// Rationale for action
    pub rationale: String,
}

/// Action categories for better organization
#[derive(Debug, Clone)]
pub enum ActionCategory {
    /// Performance optimization
    PerformanceOptimization,
    /// Resource management
    ResourceManagement,
    /// Configuration adjustment
    ConfigurationAdjustment,
    /// System maintenance
    SystemMaintenance,
    /// Emergency response
    EmergencyResponse,
    /// Experimental
    Experimental,
}

/// Resource requirements for action execution
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// CPU requirements
    pub cpu_requirements: f32,
    /// Memory requirements (MB)
    pub memory_requirements: u64,
    /// I/O requirements
    pub io_requirements: f32,
    /// Network requirements
    pub network_requirements: f32,
    /// Execution time estimate
    pub execution_time: Duration,
    /// Additional resources
    pub additional_resources: HashMap<String, f64>,
}

/// Action implementation complexity
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ActionComplexity {
    /// Simple action
    Simple,
    /// Moderate complexity
    Moderate,
    /// Complex action
    Complex,
    /// Very complex
    VeryComplex,
}

// =============================================================================
// RISK ASSESSMENT TYPES
// =============================================================================

/// Risk assessment for action execution
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// Overall risk level
    pub risk_level: RiskLevel,
    /// Potential risks
    pub potential_risks: Vec<Risk>,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<MitigationStrategy>,
    /// Rollback plan available
    pub rollback_available: bool,
    /// Recovery time estimate
    pub recovery_time: Option<Duration>,
}

/// Risk levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    /// Minimal risk
    Minimal,
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Critical risk
    Critical,
}

/// Individual risk specification
#[derive(Debug, Clone)]
pub struct Risk {
    /// Risk type
    pub risk_type: RiskType,
    /// Probability (0.0 to 1.0)
    pub probability: f32,
    /// Impact severity
    pub impact: RiskImpact,
    /// Description
    pub description: String,
}

/// Types of risks
#[derive(Debug, Clone)]
pub enum RiskType {
    /// Performance degradation
    PerformanceDegradation,
    /// Resource contention
    ResourceContention,
    /// System instability
    SystemInstability,
    /// Data loss
    DataLoss,
    /// Service interruption
    ServiceInterruption,
    /// Configuration corruption
    ConfigurationCorruption,
    /// Custom risk
    Custom(String),
}

/// Risk impact levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskImpact {
    /// Negligible impact
    Negligible,
    /// Minor impact
    Minor,
    /// Moderate impact
    Moderate,
    /// Major impact
    Major,
    /// Severe impact
    Severe,
}

/// Risk mitigation strategy
#[derive(Debug, Clone)]
pub struct MitigationStrategy {
    /// Strategy type
    pub strategy_type: MitigationType,
    /// Implementation steps
    pub steps: Vec<String>,
    /// Effectiveness
    pub effectiveness: f32,
    /// Implementation cost
    pub implementation_cost: f32,
}

/// Types of mitigation strategies
#[derive(Debug, Clone)]
pub enum MitigationType {
    /// Preventive measures
    Preventive,
    /// Monitoring and alerts
    Monitoring,
    /// Gradual rollout
    GradualRollout,
    /// Backup and recovery
    BackupAndRecovery,
    /// Resource isolation
    ResourceIsolation,
    /// Custom mitigation
    Custom(String),
}

// =============================================================================
// AGGREGATION STRATEGY TYPES
// =============================================================================

/// Weighted confidence aggregation strategy
#[derive(Debug, Clone)]
pub struct WeightedConfidenceAggregationStrategy {
    /// Confidence weight factor
    pub confidence_weight: f32,
    /// Recency weight factor
    pub recency_weight: f32,
    /// Source reliability weights
    pub source_weights: HashMap<FeedbackSource, f32>,
}

/// Time-series aggregation strategy
#[derive(Debug, Clone)]
pub struct TimeSeriesAggregationStrategy {
    /// Time window for aggregation
    pub time_window: Duration,
    /// Smoothing factor for exponential smoothing
    pub smoothing_factor: f32,
    /// Trend detection enabled
    pub trend_detection: bool,
    /// Seasonal adjustment enabled
    pub seasonal_adjustment: bool,
}

/// Consensus-based aggregation strategy
#[derive(Debug, Clone)]
pub struct ConsensusAggregationStrategy {
    /// Minimum consensus threshold
    pub consensus_threshold: f32,
    /// Outlier detection enabled
    pub outlier_detection: bool,
    /// Outlier threshold
    pub outlier_threshold: f32,
    /// Voting mechanism
    pub voting_mechanism: VotingMechanism,
}

/// Voting mechanisms for consensus
#[derive(Debug, Clone)]
pub enum VotingMechanism {
    /// Simple majority
    SimpleMajority,
    /// Weighted voting
    WeightedVoting,
    /// Ranked choice
    RankedChoice,
    /// Fuzzy consensus
    FuzzyConsensus,
}

/// Multi-objective aggregation strategy
#[derive(Debug, Clone)]
pub struct MultiObjectiveAggregationStrategy {
    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,
    /// Objective weights
    pub weights: Vec<f32>,
    /// Pareto optimization enabled
    pub pareto_optimization: bool,
    /// Constraint handling
    pub constraints: Vec<OptimizationConstraint>,
}

// =============================================================================
// OPTIMIZATION TYPES
// =============================================================================

/// Optimization objectives
#[derive(Debug, Clone)]
pub struct OptimizationObjective {
    /// Objective name
    pub name: String,
    /// Objective type
    pub objective_type: ObjectiveType,
    /// Target value
    pub target_value: Option<f64>,
    /// Minimize or maximize
    pub direction: OptimizationDirection,
}

/// Types of objectives
#[derive(Debug, Clone)]
pub enum ObjectiveType {
    /// Performance objective
    Performance,
    /// Resource efficiency
    ResourceEfficiency,
    /// Stability
    Stability,
    /// Reliability
    Reliability,
    /// Custom objective
    Custom(String),
}

/// Optimization direction
#[derive(Debug, Clone)]
pub enum OptimizationDirection {
    /// Minimize objective
    Minimize,
    /// Maximize objective
    Maximize,
}

/// Optimization constraints
#[derive(Debug, Clone)]
pub struct OptimizationConstraint {
    /// Constraint name
    pub name: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Constraint value
    pub value: f64,
    /// Constraint operator
    pub operator: ConstraintOperator,
}

/// Types of constraints
#[derive(Debug, Clone)]
pub enum ConstraintType {
    /// Resource constraint
    Resource,
    /// Performance constraint
    Performance,
    /// Safety constraint
    Safety,
    /// Policy constraint
    Policy,
    /// Custom constraint
    Custom(String),
}

/// Constraint operators
#[derive(Debug, Clone)]
pub enum ConstraintOperator {
    /// Less than
    LessThan,
    /// Less than or equal
    LessThanOrEqual,
    /// Equal
    Equal,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Greater than
    GreaterThan,
    /// Not equal
    NotEqual,
}

// =============================================================================
// DEFAULT IMPLEMENTATIONS
// =============================================================================

impl Default for FeedbackQualityMetrics {
    fn default() -> Self {
        Self {
            reliability: 0.5,
            relevance: 0.5,
            timeliness: 0.5,
            completeness: 0.5,
            consistency: 0.5,
            overall_quality: 0.5,
            assessed_at: Utc::now(),
        }
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_requirements: 0.1,
            memory_requirements: 100,
            io_requirements: 0.1,
            network_requirements: 0.1,
            execution_time: Duration::from_secs(30),
            additional_resources: HashMap::new(),
        }
    }
}

impl Default for RiskAssessment {
    fn default() -> Self {
        Self {
            risk_level: RiskLevel::Low,
            potential_risks: Vec::new(),
            mitigation_strategies: Vec::new(),
            rollback_available: true,
            recovery_time: Some(Duration::from_secs(300)),
        }
    }
}

impl Default for WeightedConfidenceAggregationStrategy {
    fn default() -> Self {
        Self {
            confidence_weight: 0.7,
            recency_weight: 0.3,
            source_weights: HashMap::new(),
        }
    }
}

impl Default for TimeSeriesAggregationStrategy {
    fn default() -> Self {
        Self {
            time_window: Duration::from_secs(300),
            smoothing_factor: 0.3,
            trend_detection: true,
            seasonal_adjustment: false,
        }
    }
}

impl Default for ConsensusAggregationStrategy {
    fn default() -> Self {
        Self {
            consensus_threshold: 0.7,
            outlier_detection: true,
            outlier_threshold: 2.0,
            voting_mechanism: VotingMechanism::WeightedVoting,
        }
    }
}

impl Default for MultiObjectiveAggregationStrategy {
    fn default() -> Self {
        Self {
            objectives: vec![
                OptimizationObjective {
                    name: "performance".to_string(),
                    objective_type: ObjectiveType::Performance,
                    target_value: None,
                    direction: OptimizationDirection::Maximize,
                },
                OptimizationObjective {
                    name: "efficiency".to_string(),
                    objective_type: ObjectiveType::ResourceEfficiency,
                    target_value: None,
                    direction: OptimizationDirection::Maximize,
                },
            ],
            weights: vec![0.6, 0.4],
            pareto_optimization: false,
            constraints: Vec::new(),
        }
    }
}

/// Aggregation strategy type for feedback aggregation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategyType {
    /// Simple averaging
    Average,
    /// Weighted average
    WeightedAverage,
    /// Exponential moving average
    ExponentialMovingAverage,
    /// Median-based
    Median,
    /// Maximum value
    Maximum,
    /// Minimum value
    Minimum,
}

impl Default for AggregationStrategyType {
    fn default() -> Self {
        Self::Average
    }
}

// =============================================================================
// PROCESSOR CONFIGURATION TYPES
// =============================================================================

/// Configuration for throughput processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputProcessorConfig {
    /// Minimum throughput threshold
    pub min_throughput: f64,
    /// Maximum throughput threshold
    pub max_throughput: f64,
    /// Smoothing factor for exponential moving average
    pub smoothing_factor: f64,
}

impl Default for ThroughputProcessorConfig {
    fn default() -> Self {
        Self {
            min_throughput: 0.0,
            max_throughput: f64::MAX,
            smoothing_factor: 0.3,
        }
    }
}

/// Configuration for latency processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyProcessorConfig {
    /// Maximum acceptable latency (milliseconds)
    pub max_latency_ms: u64,
    /// Target latency percentile
    pub target_percentile: f64,
    /// Enable outlier detection
    pub outlier_detection: bool,
}

impl Default for LatencyProcessorConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 1000,
            target_percentile: 0.95,
            outlier_detection: true,
        }
    }
}

/// Configuration for resource utilization processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceProcessorConfig {
    /// CPU utilization threshold
    pub cpu_threshold: f64,
    /// Memory utilization threshold
    pub memory_threshold: f64,
    /// Disk I/O threshold
    pub disk_threshold: f64,
}

impl Default for ResourceProcessorConfig {
    fn default() -> Self {
        Self {
            cpu_threshold: 0.8,
            memory_threshold: 0.9,
            disk_threshold: 0.85,
        }
    }
}

/// Configuration for validation engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationEngineConfig {
    /// Enable strict validation
    pub strict_mode: bool,
    /// Validation timeout (seconds)
    pub timeout_secs: u64,
    /// Required confidence threshold
    pub min_confidence: f32,
}

impl Default for ValidationEngineConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            timeout_secs: 30,
            min_confidence: 0.7,
        }
    }
}

/// Configuration for aggregation manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationManagerConfig {
    /// Aggregation strategy
    pub strategy: AggregationStrategyType,
    /// Weight decay factor
    pub weight_decay: f64,
    /// Minimum samples required
    pub min_samples: usize,
}

impl Default for AggregationManagerConfig {
    fn default() -> Self {
        Self {
            strategy: AggregationStrategyType::WeightedAverage,
            weight_decay: 0.95,
            min_samples: 3,
        }
    }
}

// =============================================================================
// AGGREGATION STRATEGY IMPLEMENTATIONS
// =============================================================================

impl WeightedConfidenceAggregationStrategy {
    /// Aggregate processed feedbacks using the specified strategy
    pub async fn aggregate_with_strategy(
        &self,
        feedbacks: &[crate::performance_optimizer::types::ProcessedFeedback],
        strategy: AggregationStrategyType,
    ) -> anyhow::Result<crate::performance_optimizer::types::AggregatedFeedback> {
        use crate::performance_optimizer::types::AggregatedFeedback;

        if feedbacks.is_empty() {
            anyhow::bail!("Cannot aggregate empty feedback list");
        }

        // Calculate aggregated value based on strategy
        let aggregated_value = match strategy {
            AggregationStrategyType::Average => {
                let sum: f64 = feedbacks.iter().map(|f| f.processed_value).sum();
                sum / feedbacks.len() as f64
            },
            AggregationStrategyType::WeightedAverage => {
                let total_weight: f32 = feedbacks.iter().map(|f| f.confidence).sum();
                if total_weight == 0.0 {
                    feedbacks.iter().map(|f| f.processed_value).sum::<f64>()
                        / feedbacks.len() as f64
                } else {
                    feedbacks.iter().map(|f| f.processed_value * f.confidence as f64).sum::<f64>()
                        / total_weight as f64
                }
            },
            AggregationStrategyType::ExponentialMovingAverage => {
                let alpha = 0.3; // Smoothing factor
                let mut ema = feedbacks[0].processed_value;
                for feedback in &feedbacks[1..] {
                    ema = alpha * feedback.processed_value + (1.0 - alpha) * ema;
                }
                ema
            },
            AggregationStrategyType::Median => {
                let mut values: Vec<f64> = feedbacks.iter().map(|f| f.processed_value).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if values.len() % 2 == 0 {
                    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                } else {
                    values[values.len() / 2]
                }
            },
            AggregationStrategyType::Maximum => {
                feedbacks.iter().map(|f| f.processed_value).fold(f64::NEG_INFINITY, f64::max)
            },
            AggregationStrategyType::Minimum => {
                feedbacks.iter().map(|f| f.processed_value).fold(f64::INFINITY, f64::min)
            },
        };

        // Calculate aggregated confidence
        let confidence = if feedbacks.is_empty() {
            0.0
        } else {
            feedbacks.iter().map(|f| f.confidence).sum::<f32>() / feedbacks.len() as f32
        };

        // Collect all recommended actions
        let recommended_actions =
            feedbacks.iter().filter_map(|f| f.recommended_action.clone()).collect();

        Ok(AggregatedFeedback {
            aggregated_value,
            confidence,
            contributing_count: feedbacks.len(),
            contributing_feedback_count: feedbacks.len(),
            aggregation_method: format!("{:?}", strategy),
            timestamp: Utc::now(),
            recommended_actions,
        })
    }
}
