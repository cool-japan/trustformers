//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::definitions::{
    ABTestMetrics, ABTestingConfig, CascadeChain, ComparisonOperator, ContentRoutingRule,
    EnsembleConfig, EnsembleState, ExitStrategy, FallbackTrigger, ModelCharacteristic, ModelInfo,
    ModelRoutingConfig, PerformanceMetric, PerformanceMonitoringConfig, QualityAssessmentMethod,
    ResourceConstraints, ResourceUsage, TrafficSplittingConfig,
};

/// Model cascading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCascadingConfig {
    /// Enable model cascading
    pub enabled: bool,
    /// Cascade chains
    pub cascade_chains: Vec<CascadeChain>,
    /// Exit strategies
    pub exit_strategies: Vec<ExitStrategy>,
}
/// Model size categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSize {
    Small,
    Medium,
    Large,
    XLarge,
}
/// A/B test experiment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestResult {
    pub experiment_id: String,
    pub is_significant: bool,
    pub winner: Option<String>,
    pub control_performance: PerformanceStats,
    pub variant_results: Vec<ABTestVariantResult>,
    pub total_requests: u64,
    pub duration: chrono::Duration,
    pub confidence_level: f64,
}
/// Model selection criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionCriteria {
    /// Preferred model characteristics
    pub preferred_characteristics: Vec<ModelCharacteristic>,
    /// Quality thresholds
    pub quality_thresholds: QualityThresholds,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}
/// Multi-model server state
#[derive(Debug)]
pub(crate) struct MultiModelState {
    /// Registered models
    pub(crate) models: HashMap<String, ModelInfo>,
    /// Active experiments
    pub(crate) active_experiments: HashMap<String, ABTestExperiment>,
    /// Routing state
    pub(crate) _routing_state: RoutingState,
    /// Ensemble state
    pub(crate) _ensemble_state: EnsembleState,
    /// Performance history
    pub(crate) _performance_history: HashMap<String, Vec<PerformanceRecord>>,
}
/// Exit condition for cascading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitCondition {
    ConfidenceAboveThreshold,
    QualityMet,
    ResourceBudgetExceeded,
    TimeoutReached,
}
/// Routing state
#[derive(Debug)]
pub(crate) struct RoutingState {
    pub(crate) _round_robin_index: usize,
    pub(crate) _model_weights: HashMap<String, f64>,
    pub(crate) _request_counts: HashMap<String, u64>,
}
/// Success metrics for A/B testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuccessMetric {
    Accuracy,
    Latency,
    UserSatisfaction,
    ConversionRate,
    ErrorRate,
    Custom(String),
}
/// Routing condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingCondition {
    /// Text length condition
    TextLength {
        min: Option<usize>,
        max: Option<usize>,
    },
    /// Language detection
    Language { languages: Vec<String> },
    /// Keywords presence
    Keywords {
        keywords: Vec<String>,
        match_all: bool,
    },
    /// Request headers
    Header {
        name: String,
        value: String,
        operator: ComparisonOperator,
    },
    /// Request path pattern
    PathPattern { pattern: String },
    /// Content type
    ContentType { content_types: Vec<String> },
    /// Custom condition
    Custom {
        name: String,
        parameters: HashMap<String, String>,
    },
}
/// Size threshold for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeThreshold {
    pub max_size: usize,
    pub target_model: String,
}
/// Alerting thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingThresholds {
    pub latency_threshold: Duration,
    pub error_rate_threshold: f64,
    pub accuracy_threshold: f64,
    pub resource_threshold: f64,
}
/// Multi-model metrics
#[derive(Debug, Default)]
pub struct MultiModelMetrics {
    /// Total requests
    pub total_requests: u64,
    /// Requests per model
    pub model_request_counts: HashMap<String, u64>,
    /// Ensemble request counts
    pub ensemble_request_counts: HashMap<String, u64>,
    /// A/B test metrics
    pub ab_test_metrics: HashMap<String, ABTestMetrics>,
    /// Average routing time
    pub avg_routing_time: Duration,
    /// Fallback trigger counts
    pub fallback_triggers: HashMap<String, u64>,
}
/// Cascade stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeStage {
    pub model: String,
    pub confidence_threshold: f64,
    pub exit_condition: ExitCondition,
}
/// Routing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route based on request content
    ContentBased { rules: Vec<ContentRoutingRule> },
    /// Route based on model performance
    PerformanceBased {
        metrics: Vec<PerformanceMetric>,
        weights: HashMap<String, f64>,
    },
    /// Route based on model capabilities
    CapabilityBased {
        capability_map: HashMap<String, Vec<String>>,
    },
    /// Route based on resource utilization
    ResourceBased {
        cpu_threshold: f64,
        memory_threshold: f64,
        gpu_threshold: f64,
    },
    /// Route based on user/tenant
    UserBased {
        user_model_map: HashMap<String, String>,
        default_model: String,
    },
    /// Route based on request size
    SizeBased { size_thresholds: Vec<SizeThreshold> },
    /// Round-robin routing
    RoundRobin,
    /// Weighted round-robin
    WeightedRoundRobin { weights: HashMap<String, f64> },
    /// Random routing
    Random,
    /// Custom routing logic
    Custom {
        name: String,
        parameters: HashMap<String, String>,
    },
}
/// Allocation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationMethod {
    Random,
    UserId,
    SessionId,
    IPAddress,
    Custom(String),
}
/// A/B test experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestExperiment {
    /// Experiment ID
    pub id: String,
    /// Experiment name
    pub name: String,
    /// Control model
    pub control_model: String,
    /// Variant models
    pub variant_models: Vec<ABTestVariant>,
    /// Traffic allocation
    pub traffic_allocation: TrafficAllocation,
    /// Success metrics
    pub success_metrics: Vec<SuccessMetric>,
    /// Experiment duration
    pub duration: Duration,
    /// Statistical power
    pub statistical_power: f64,
}
/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub avg_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub throughput: f64,
    pub error_rate: f64,
    pub accuracy: Option<f64>,
    pub request_count: u64,
}
/// Quality thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    pub min_accuracy: Option<f64>,
    pub max_latency: Option<Duration>,
    pub max_error_rate: Option<f64>,
    pub min_throughput: Option<f64>,
}
/// Traffic allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficAllocation {
    pub control_percentage: f64,
    pub variant_percentages: HashMap<String, f64>,
    pub allocation_method: AllocationMethod,
}
/// Multi-model serving configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModelConfig {
    /// Model routing configuration
    pub routing: ModelRoutingConfig,
    /// Ensemble serving configuration
    pub ensemble: EnsembleConfig,
    /// A/B testing configuration
    pub ab_testing: ABTestingConfig,
    /// Traffic splitting configuration
    pub traffic_splitting: TrafficSplittingConfig,
    /// Model cascading configuration
    pub model_cascading: ModelCascadingConfig,
    /// Performance monitoring
    pub performance_monitoring: PerformanceMonitoringConfig,
}
/// A/B test variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestVariant {
    pub id: String,
    pub model: String,
    pub traffic_percentage: f64,
    pub configuration: HashMap<String, String>,
}
/// Quality assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessmentConfig {
    /// Enable quality assessment
    pub enabled: bool,
    /// Assessment methods
    pub methods: Vec<QualityAssessmentMethod>,
    /// Quality thresholds
    pub thresholds: QualityThresholds,
}
/// Performance record
#[derive(Debug, Clone)]
pub(crate) struct PerformanceRecord {
    _timestamp: Instant,
    _latency: Duration,
    _accuracy: Option<f64>,
    _error_rate: f64,
    _resource_usage: ResourceUsage,
}
/// Resource budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub max_latency: Duration,
    pub max_memory: u64,
    pub max_compute_cost: f64,
}
/// Fallback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackConfig {
    /// Fallback model ID
    pub fallback_model: String,
    /// Enable fallback
    pub enabled: bool,
    /// Fallback triggers
    pub triggers: Vec<FallbackTrigger>,
}
/// Model status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Available,
    Loading,
    Unavailable,
    Maintenance,
    Deprecated,
}
/// A/B test variant result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestVariantResult {
    pub variant_id: String,
    pub model_id: String,
    pub performance: PerformanceStats,
    pub statistical_significance: f64,
    pub is_better_than_control: bool,
    pub confidence_interval: (f64, f64),
}
