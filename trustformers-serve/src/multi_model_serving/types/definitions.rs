//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::types_3::{
    ABTestExperiment, AlertingThresholds, CascadeStage, ExitCondition, FallbackConfig,
    ModelSelectionCriteria, ModelSize, ModelStatus, PerformanceStats, QualityAssessmentConfig,
    ResourceBudget, RoutingCondition, RoutingStrategy,
};

/// Voting strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingStrategy {
    /// Simple majority
    SimpleMajority,
    /// Weighted majority
    WeightedMajority,
    /// Unanimous consensus
    Unanimous,
    /// Threshold-based
    Threshold { threshold: f64 },
    /// Rank-based
    RankBased,
}
/// Cascade chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeChain {
    pub id: String,
    pub name: String,
    pub stages: Vec<CascadeStage>,
    pub default_chain: bool,
}
/// Statistical significance thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalThresholds {
    pub p_value: f64,
    pub confidence_level: f64,
    pub minimum_sample_size: u32,
    pub minimum_effect_size: f64,
}
/// Performance metrics for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    Latency,
    Throughput,
    Accuracy,
    ErrorRate,
    ResourceUsage,
    Cost,
}
/// Fallback triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackTrigger {
    ModelUnavailable,
    HighLatency(Duration),
    HighErrorRate(f64),
    ResourceExhaustion,
    QualityBelowThreshold,
}
/// Traffic split
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficSplit {
    pub splits: HashMap<String, f64>,
    pub sticky_sessions: bool,
}
/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub characteristics: Vec<ModelCharacteristic>,
    pub capabilities: Vec<String>,
    pub status: ModelStatus,
    pub performance_stats: PerformanceStats,
    pub resource_usage: ResourceUsage,
    pub metadata: HashMap<String, String>,
}
/// Resource usage
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub cpu_usage: f64,
    pub memory_usage: u64,
    pub gpu_memory_usage: u64,
    pub network_io: f64,
}
/// A/B test metrics
#[derive(Debug, Clone, Default)]
pub struct ABTestMetrics {
    pub control_requests: u64,
    pub variant_requests: HashMap<String, u64>,
    pub control_performance: PerformanceStats,
    pub variant_performance: HashMap<String, PerformanceStats>,
    pub statistical_significance: Option<f64>,
}
/// Ensemble information
#[derive(Debug, Clone)]
pub struct EnsembleInfo {
    pub id: String,
    pub method: EnsembleMethod,
    pub participating_models: Vec<String>,
    pub performance_stats: PerformanceStats,
}
/// Ensemble optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleOptimizationConfig {
    /// Enable optimization
    pub enabled: bool,
    /// Optimization strategies
    pub strategies: Vec<OptimizationStrategy>,
    /// Resource budget
    pub resource_budget: ResourceBudget,
}
/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    pub max_memory_usage: Option<u64>,
    pub max_gpu_memory: Option<u64>,
    pub max_cpu_usage: Option<f64>,
    pub required_gpu_count: Option<u32>,
}
/// Comparison operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equals,
    NotEquals,
    Contains,
    StartsWith,
    EndsWith,
    Regex,
    GreaterThan,
    LessThan,
}
/// Model characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelCharacteristic {
    Size(ModelSize),
    Accuracy(f64),
    Latency(Duration),
    Language(String),
    Domain(String),
    Task(String),
}
/// Traffic splitting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficSplittingConfig {
    /// Enable traffic splitting
    pub enabled: bool,
    /// Split rules
    pub split_rules: Vec<TrafficSplitRule>,
    /// Default split
    pub default_split: TrafficSplit,
}
/// Quality assessment methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityAssessmentMethod {
    ConfidenceScoring,
    ConsistencyChecking,
    CrossValidation,
    UncertaintyQuantification,
    EnsembleAgreement,
}
/// Exit strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitStrategy {
    pub condition: ExitCondition,
    pub action: ExitAction,
}
/// Ensemble state
#[derive(Debug)]
pub(crate) struct EnsembleState {
    pub(crate) _active_ensembles: HashMap<String, EnsembleInfo>,
    pub(crate) _quality_scores: HashMap<String, f64>,
}
/// Traffic split rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficSplitRule {
    pub id: String,
    pub condition: RoutingCondition,
    pub split: TrafficSplit,
}
/// Exit action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitAction {
    ReturnResult,
    ContinueToNext,
    FallbackToDefault,
    RaiseError,
}
/// Inference request
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub input_text: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub user_id: Option<String>,
    pub metadata: HashMap<String, String>,
}
/// Model routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRoutingConfig {
    /// Default routing strategy
    pub default_strategy: RoutingStrategy,
    /// Route-specific strategies
    pub route_strategies: HashMap<String, RoutingStrategy>,
    /// Model selection criteria
    pub selection_criteria: ModelSelectionCriteria,
    /// Fallback behavior
    pub fallback: FallbackConfig,
}
/// Routing result
#[derive(Debug, Clone)]
pub enum RoutingResult {
    SingleModel {
        model_id: String,
    },
    Ensemble {
        method: EnsembleMethod,
        models: Vec<String>,
    },
}
/// Optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    EarlyExit,
    ParallelExecution,
    SequentialExecution,
    AdaptiveSelection,
    ResourceAwareScheduling,
}
/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Metrics to monitor
    pub monitored_metrics: Vec<MonitoredMetric>,
    /// Alerting thresholds
    pub alerting_thresholds: AlertingThresholds,
}
/// Monitored metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitoredMetric {
    ModelLatency,
    ModelAccuracy,
    ModelThroughput,
    ResourceUtilization,
    ErrorRates,
    QueueLengths,
    EnsembleAgreement,
    ABTestMetrics,
}
/// Ensemble methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    /// Majority voting
    MajorityVoting {
        models: Vec<String>,
        weights: Option<HashMap<String, f64>>,
    },
    /// Weighted averaging
    WeightedAveraging {
        models: Vec<String>,
        weights: HashMap<String, f64>,
    },
    /// Stacking ensemble
    Stacking {
        base_models: Vec<String>,
        meta_model: String,
    },
    /// Boosting ensemble
    Boosting {
        models: Vec<String>,
        boost_weights: Vec<f64>,
    },
    /// Bagging ensemble
    Bagging {
        models: Vec<String>,
        sample_size: f64,
    },
    /// Mixture of experts
    MixtureOfExperts {
        experts: Vec<String>,
        gating_network: String,
    },
    /// Cascading ensemble
    Cascading { stages: Vec<CascadeStage> },
}
/// Content routing rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentRoutingRule {
    /// Rule name
    pub name: String,
    /// Condition to match
    pub condition: RoutingCondition,
    /// Target model ID
    pub target_model: String,
    /// Rule priority (higher = more priority)
    pub priority: u32,
}
/// A/B testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestingConfig {
    /// Enable A/B testing
    pub enabled: bool,
    /// A/B test experiments
    pub experiments: Vec<ABTestExperiment>,
    /// Statistical significance thresholds
    pub significance_thresholds: StatisticalThresholds,
}
/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Enable ensemble serving
    pub enabled: bool,
    /// Ensemble methods
    pub methods: Vec<EnsembleMethod>,
    /// Voting strategy
    pub voting_strategy: VotingStrategy,
    /// Quality assessment
    pub quality_assessment: QualityAssessmentConfig,
    /// Performance optimization
    pub optimization: EnsembleOptimizationConfig,
}
