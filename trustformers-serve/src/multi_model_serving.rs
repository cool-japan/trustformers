// Allow dead code for infrastructure under development
#![allow(dead_code)]

use anyhow::{anyhow, Result};
use chrono;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::info;

/// Multi-model serving system for model routing and ensemble inference
#[derive(Debug, Clone)]
pub struct MultiModelServer {
    config: MultiModelConfig,
    state: Arc<RwLock<MultiModelState>>,
    metrics: Arc<Mutex<MultiModelMetrics>>,
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

/// Model size categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSize {
    Small,  // < 1B parameters
    Medium, // 1B - 10B parameters
    Large,  // 10B - 100B parameters
    XLarge, // > 100B parameters
}

/// Quality thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    pub min_accuracy: Option<f64>,
    pub max_latency: Option<Duration>,
    pub max_error_rate: Option<f64>,
    pub min_throughput: Option<f64>,
}

/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    pub max_memory_usage: Option<u64>,
    pub max_gpu_memory: Option<u64>,
    pub max_cpu_usage: Option<f64>,
    pub required_gpu_count: Option<u32>,
}

/// Size threshold for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeThreshold {
    pub max_size: usize,
    pub target_model: String,
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

/// Fallback triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackTrigger {
    ModelUnavailable,
    HighLatency(Duration),
    HighErrorRate(f64),
    ResourceExhaustion,
    QualityBelowThreshold,
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

/// Cascade stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeStage {
    pub model: String,
    pub confidence_threshold: f64,
    pub exit_condition: ExitCondition,
}

/// Exit condition for cascading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitCondition {
    ConfidenceAboveThreshold,
    QualityMet,
    ResourceBudgetExceeded,
    TimeoutReached,
}

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

/// Quality assessment methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityAssessmentMethod {
    ConfidenceScoring,
    ConsistencyChecking,
    CrossValidation,
    UncertaintyQuantification,
    EnsembleAgreement,
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

/// Optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    EarlyExit,
    ParallelExecution,
    SequentialExecution,
    AdaptiveSelection,
    ResourceAwareScheduling,
}

/// Resource budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub max_latency: Duration,
    pub max_memory: u64,
    pub max_compute_cost: f64,
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

/// A/B test variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestVariant {
    pub id: String,
    pub model: String,
    pub traffic_percentage: f64,
    pub configuration: HashMap<String, String>,
}

/// Traffic allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficAllocation {
    pub control_percentage: f64,
    pub variant_percentages: HashMap<String, f64>,
    pub allocation_method: AllocationMethod,
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

/// Statistical significance thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalThresholds {
    pub p_value: f64,
    pub confidence_level: f64,
    pub minimum_sample_size: u32,
    pub minimum_effect_size: f64,
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

/// Traffic split rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficSplitRule {
    pub id: String,
    pub condition: RoutingCondition,
    pub split: TrafficSplit,
}

/// Traffic split
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficSplit {
    pub splits: HashMap<String, f64>,
    pub sticky_sessions: bool,
}

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

/// Cascade chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeChain {
    pub id: String,
    pub name: String,
    pub stages: Vec<CascadeStage>,
    pub default_chain: bool,
}

/// Exit strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitStrategy {
    pub condition: ExitCondition,
    pub action: ExitAction,
}

/// Exit action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitAction {
    ReturnResult,
    ContinueToNext,
    FallbackToDefault,
    RaiseError,
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

/// Alerting thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingThresholds {
    pub latency_threshold: Duration,
    pub error_rate_threshold: f64,
    pub accuracy_threshold: f64,
    pub resource_threshold: f64,
}

/// Multi-model server state
#[derive(Debug)]
struct MultiModelState {
    /// Registered models
    models: HashMap<String, ModelInfo>,
    /// Active experiments
    active_experiments: HashMap<String, ABTestExperiment>,
    /// Routing state
    routing_state: RoutingState,
    /// Ensemble state
    ensemble_state: EnsembleState,
    /// Performance history
    performance_history: HashMap<String, Vec<PerformanceRecord>>,
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

/// Model status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Available,
    Loading,
    Unavailable,
    Maintenance,
    Deprecated,
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

/// Resource usage
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub cpu_usage: f64,
    pub memory_usage: u64,
    pub gpu_memory_usage: u64,
    pub network_io: f64,
}

/// Routing state
#[derive(Debug)]
struct RoutingState {
    round_robin_index: usize,
    model_weights: HashMap<String, f64>,
    request_counts: HashMap<String, u64>,
}

/// Ensemble state
#[derive(Debug)]
struct EnsembleState {
    active_ensembles: HashMap<String, EnsembleInfo>,
    quality_scores: HashMap<String, f64>,
}

/// Ensemble information
#[derive(Debug, Clone)]
pub struct EnsembleInfo {
    pub id: String,
    pub method: EnsembleMethod,
    pub participating_models: Vec<String>,
    pub performance_stats: PerformanceStats,
}

/// Performance record
#[derive(Debug, Clone)]
struct PerformanceRecord {
    timestamp: Instant,
    latency: Duration,
    accuracy: Option<f64>,
    error_rate: f64,
    resource_usage: ResourceUsage,
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

/// A/B test metrics
#[derive(Debug, Clone, Default)]
pub struct ABTestMetrics {
    pub control_requests: u64,
    pub variant_requests: HashMap<String, u64>,
    pub control_performance: PerformanceStats,
    pub variant_performance: HashMap<String, PerformanceStats>,
    pub statistical_significance: Option<f64>,
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

impl MultiModelServer {
    /// Create a new multi-model server
    pub fn new(config: MultiModelConfig) -> Self {
        let state = MultiModelState {
            models: HashMap::new(),
            active_experiments: HashMap::new(),
            routing_state: RoutingState {
                round_robin_index: 0,
                model_weights: HashMap::new(),
                request_counts: HashMap::new(),
            },
            ensemble_state: EnsembleState {
                active_ensembles: HashMap::new(),
                quality_scores: HashMap::new(),
            },
            performance_history: HashMap::new(),
        };

        Self {
            config,
            state: Arc::new(RwLock::new(state)),
            metrics: Arc::new(Mutex::new(MultiModelMetrics::default())),
        }
    }

    /// Register a model
    pub async fn register_model(&self, model_info: ModelInfo) -> Result<()> {
        let mut state = self.state.write().await;
        state.models.insert(model_info.id.clone(), model_info.clone());

        info!("Registered model: {} ({})", model_info.name, model_info.id);
        Ok(())
    }

    /// Unregister a model
    pub async fn unregister_model(&self, model_id: &str) -> Result<()> {
        let mut state = self.state.write().await;
        if state.models.remove(model_id).is_some() {
            info!("Unregistered model: {}", model_id);
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}", model_id))
        }
    }

    /// Route a request to appropriate model(s)
    pub async fn route_request(&self, request: &InferenceRequest) -> Result<RoutingResult> {
        let state = self.state.read().await;
        let mut metrics = self.metrics.lock().await;

        let start_time = Instant::now();

        // Check if ensemble is required
        if self.config.ensemble.enabled {
            if let Some(ensemble_result) = self.try_ensemble_routing(request, &state).await? {
                metrics.total_requests += 1;
                metrics.avg_routing_time = Duration::from_nanos(
                    (metrics.avg_routing_time.as_nanos() as u64 * (metrics.total_requests - 1)
                        + start_time.elapsed().as_nanos() as u64)
                        / metrics.total_requests,
                );
                return Ok(ensemble_result);
            }
        }

        // Single model routing
        let selected_model = self.select_model(request, &state).await?;

        metrics.total_requests += 1;
        *metrics.model_request_counts.entry(selected_model.clone()).or_insert(0) += 1;
        metrics.avg_routing_time = Duration::from_nanos(
            (metrics.avg_routing_time.as_nanos() as u64 * (metrics.total_requests - 1)
                + start_time.elapsed().as_nanos() as u64)
                / metrics.total_requests,
        );

        Ok(RoutingResult::SingleModel {
            model_id: selected_model,
        })
    }

    /// Select a model based on routing strategy
    async fn select_model(
        &self,
        request: &InferenceRequest,
        state: &MultiModelState,
    ) -> Result<String> {
        let strategy = self.get_routing_strategy(request);

        match strategy {
            RoutingStrategy::ContentBased { rules } => {
                self.select_content_based(request, &rules, state).await
            },
            RoutingStrategy::PerformanceBased { metrics, weights } => {
                self.select_performance_based(&metrics, &weights, state).await
            },
            RoutingStrategy::CapabilityBased { capability_map } => {
                self.select_capability_based(request, &capability_map, state).await
            },
            RoutingStrategy::ResourceBased {
                cpu_threshold,
                memory_threshold,
                gpu_threshold,
            } => {
                self.select_resource_based(*cpu_threshold, *memory_threshold, *gpu_threshold, state)
                    .await
            },
            RoutingStrategy::UserBased {
                user_model_map,
                default_model,
            } => self.select_user_based(request, &user_model_map, &default_model, state).await,
            RoutingStrategy::SizeBased { size_thresholds } => {
                self.select_size_based(request, &size_thresholds, state).await
            },
            RoutingStrategy::RoundRobin => self.select_round_robin(state).await,
            RoutingStrategy::WeightedRoundRobin { weights } => {
                self.select_weighted_round_robin(&weights, state).await
            },
            RoutingStrategy::Random => self.select_random(state).await,
            RoutingStrategy::Custom { name, parameters } => {
                self.select_custom(request, &name, &parameters, state).await
            },
        }
    }

    /// Try ensemble routing
    async fn try_ensemble_routing(
        &self,
        request: &InferenceRequest,
        state: &MultiModelState,
    ) -> Result<Option<RoutingResult>> {
        for method in &self.config.ensemble.methods {
            if self.should_use_ensemble(request, method).await {
                return Ok(Some(RoutingResult::Ensemble {
                    method: method.clone(),
                    models: self.get_ensemble_models(method, state).await,
                }));
            }
        }
        Ok(None)
    }

    /// Check if ensemble should be used
    async fn should_use_ensemble(
        &self,
        _request: &InferenceRequest,
        _method: &EnsembleMethod,
    ) -> bool {
        // Implement ensemble selection logic
        true // Placeholder
    }

    /// Get models for ensemble
    async fn get_ensemble_models(
        &self,
        method: &EnsembleMethod,
        _state: &MultiModelState,
    ) -> Vec<String> {
        match method {
            EnsembleMethod::MajorityVoting { models, .. } => models.clone(),
            EnsembleMethod::WeightedAveraging { models, .. } => models.clone(),
            EnsembleMethod::Stacking { base_models, .. } => base_models.clone(),
            EnsembleMethod::Boosting { models, .. } => models.clone(),
            EnsembleMethod::Bagging { models, .. } => models.clone(),
            EnsembleMethod::MixtureOfExperts { experts, .. } => experts.clone(),
            EnsembleMethod::Cascading { stages } => {
                stages.iter().map(|s| s.model.clone()).collect()
            },
        }
    }

    /// Get routing strategy for request
    fn get_routing_strategy(&self, request: &InferenceRequest) -> &RoutingStrategy {
        // Check for route-specific strategies
        for (route_pattern, strategy) in &self.config.routing.route_strategies {
            if request.path.contains(route_pattern) {
                return strategy;
            }
        }

        &self.config.routing.default_strategy
    }

    // Model selection implementations
    async fn select_content_based(
        &self,
        request: &InferenceRequest,
        rules: &[ContentRoutingRule],
        state: &MultiModelState,
    ) -> Result<String> {
        // Sort rules by priority
        let mut sorted_rules: Vec<_> = rules.iter().collect();
        sorted_rules.sort_by(|a, b| b.priority.cmp(&a.priority));

        for rule in sorted_rules {
            if self.matches_condition(request, &rule.condition).await {
                if state.models.contains_key(&rule.target_model) {
                    return Ok(rule.target_model.clone());
                }
            }
        }

        // Fallback
        self.get_fallback_model(state).await
    }

    async fn select_performance_based(
        &self,
        _metrics: &[PerformanceMetric],
        _weights: &HashMap<String, f64>,
        state: &MultiModelState,
    ) -> Result<String> {
        // Find model with best weighted performance score
        let mut best_model = None;
        let mut best_score = f64::NEG_INFINITY;

        for (model_id, model_info) in &state.models {
            if matches!(model_info.status, ModelStatus::Available) {
                let score = self.calculate_performance_score(model_info, _metrics, _weights).await;
                if score > best_score {
                    best_score = score;
                    best_model = Some(model_id.clone());
                }
            }
        }

        best_model.ok_or_else(|| anyhow!("No available models"))
    }

    async fn select_capability_based(
        &self,
        request: &InferenceRequest,
        capability_map: &HashMap<String, Vec<String>>,
        state: &MultiModelState,
    ) -> Result<String> {
        // Extract required capabilities from request
        let required_capabilities = self.extract_required_capabilities(request).await;

        for (model_id, capabilities) in capability_map {
            if state.models.contains_key(model_id)
                && matches!(state.models[model_id].status, ModelStatus::Available)
            {
                if required_capabilities.iter().all(|req| capabilities.contains(req)) {
                    return Ok(model_id.clone());
                }
            }
        }

        self.get_fallback_model(state).await
    }

    async fn select_resource_based(
        &self,
        cpu_threshold: f64,
        memory_threshold: f64,
        gpu_threshold: f64,
        state: &MultiModelState,
    ) -> Result<String> {
        for (model_id, model_info) in &state.models {
            if matches!(model_info.status, ModelStatus::Available) {
                let usage = &model_info.resource_usage;
                if usage.cpu_usage < cpu_threshold
                    && usage.memory_usage < memory_threshold as u64
                    && usage.gpu_memory_usage < gpu_threshold as u64
                {
                    return Ok(model_id.clone());
                }
            }
        }

        self.get_fallback_model(state).await
    }

    async fn select_user_based(
        &self,
        request: &InferenceRequest,
        user_model_map: &HashMap<String, String>,
        default_model: &str,
        state: &MultiModelState,
    ) -> Result<String> {
        if let Some(user_id) = &request.user_id {
            if let Some(model_id) = user_model_map.get(user_id) {
                if state.models.contains_key(model_id)
                    && matches!(state.models[model_id].status, ModelStatus::Available)
                {
                    return Ok(model_id.clone());
                }
            }
        }

        if state.models.contains_key(default_model)
            && matches!(state.models[default_model].status, ModelStatus::Available)
        {
            Ok(default_model.to_string())
        } else {
            self.get_fallback_model(state).await
        }
    }

    async fn select_size_based(
        &self,
        request: &InferenceRequest,
        size_thresholds: &[SizeThreshold],
        state: &MultiModelState,
    ) -> Result<String> {
        let request_size = request.input_text.len();

        for threshold in size_thresholds {
            if request_size <= threshold.max_size {
                if state.models.contains_key(&threshold.target_model)
                    && matches!(
                        state.models[&threshold.target_model].status,
                        ModelStatus::Available
                    )
                {
                    return Ok(threshold.target_model.clone());
                }
            }
        }

        self.get_fallback_model(state).await
    }

    async fn select_round_robin(&self, state: &MultiModelState) -> Result<String> {
        let available_models: Vec<_> = state
            .models
            .iter()
            .filter(|(_, info)| matches!(info.status, ModelStatus::Available))
            .map(|(id, _)| id.clone())
            .collect();

        if available_models.is_empty() {
            return Err(anyhow!("No available models"));
        }

        // This would need mutable access to state for real implementation
        let index = 0; // Placeholder
        Ok(available_models[index % available_models.len()].clone())
    }

    async fn select_weighted_round_robin(
        &self,
        weights: &HashMap<String, f64>,
        state: &MultiModelState,
    ) -> Result<String> {
        let available_models: Vec<_> = state
            .models
            .iter()
            .filter(|(_, info)| matches!(info.status, ModelStatus::Available))
            .collect();

        if available_models.is_empty() {
            return Err(anyhow!("No available models"));
        }

        // Weighted selection
        let total_weight: f64 =
            available_models.iter().map(|(id, _)| weights.get(*id).unwrap_or(&1.0)).sum();

        let mut rand_val = fastrand::f64() * total_weight;
        for (model_id, _) in available_models.iter() {
            let weight = weights.get(*model_id).unwrap_or(&1.0);
            rand_val -= weight;
            if rand_val <= 0.0 {
                return Ok((*model_id).clone());
            }
        }

        // Fallback to first available
        Ok(available_models[0].0.clone())
    }

    async fn select_random(&self, state: &MultiModelState) -> Result<String> {
        let available_models: Vec<_> = state
            .models
            .iter()
            .filter(|(_, info)| matches!(info.status, ModelStatus::Available))
            .map(|(id, _)| id.clone())
            .collect();

        if available_models.is_empty() {
            return Err(anyhow!("No available models"));
        }

        let index = fastrand::usize(..available_models.len());
        Ok(available_models[index].clone())
    }

    async fn select_custom(
        &self,
        _request: &InferenceRequest,
        _name: &str,
        _parameters: &HashMap<String, String>,
        state: &MultiModelState,
    ) -> Result<String> {
        // Placeholder for custom selection logic
        self.get_fallback_model(state).await
    }

    // Helper methods
    async fn matches_condition(
        &self,
        request: &InferenceRequest,
        condition: &RoutingCondition,
    ) -> bool {
        match condition {
            RoutingCondition::TextLength { min, max } => {
                let len = request.input_text.len();
                min.map_or(true, |m| len >= m) && max.map_or(true, |m| len <= m)
            },
            RoutingCondition::Language { languages } => {
                // Placeholder for language detection
                languages.contains(&"en".to_string())
            },
            RoutingCondition::Keywords {
                keywords,
                match_all,
            } => {
                if *match_all {
                    keywords.iter().all(|k| request.input_text.contains(k))
                } else {
                    keywords.iter().any(|k| request.input_text.contains(k))
                }
            },
            RoutingCondition::Header {
                name,
                value,
                operator,
            } => {
                if let Some(header_value) = request.headers.get(name) {
                    self.matches_operator(header_value, value, operator)
                } else {
                    false
                }
            },
            RoutingCondition::PathPattern { pattern } => request.path.contains(pattern),
            RoutingCondition::ContentType { content_types } => {
                if let Some(content_type) = request.headers.get("content-type") {
                    content_types.iter().any(|ct| content_type.contains(ct))
                } else {
                    false
                }
            },
            RoutingCondition::Custom { .. } => {
                // Placeholder for custom condition logic
                true
            },
        }
    }

    fn matches_operator(
        &self,
        actual: &str,
        expected: &str,
        operator: &ComparisonOperator,
    ) -> bool {
        match operator {
            ComparisonOperator::Equals => actual == expected,
            ComparisonOperator::NotEquals => actual != expected,
            ComparisonOperator::Contains => actual.contains(expected),
            ComparisonOperator::StartsWith => actual.starts_with(expected),
            ComparisonOperator::EndsWith => actual.ends_with(expected),
            ComparisonOperator::Regex => {
                // Would use regex crate in real implementation
                actual.contains(expected)
            },
            ComparisonOperator::GreaterThan => {
                actual.parse::<f64>().unwrap_or(0.0) > expected.parse::<f64>().unwrap_or(0.0)
            },
            ComparisonOperator::LessThan => {
                actual.parse::<f64>().unwrap_or(0.0) < expected.parse::<f64>().unwrap_or(0.0)
            },
        }
    }

    async fn calculate_performance_score(
        &self,
        model_info: &ModelInfo,
        metrics: &[PerformanceMetric],
        weights: &HashMap<String, f64>,
    ) -> f64 {
        let mut score = 0.0;
        let model_weight = weights.get(&model_info.id).unwrap_or(&1.0);

        for metric in metrics {
            let metric_score = match metric {
                PerformanceMetric::Latency => {
                    1.0 / (model_info.performance_stats.avg_latency.as_millis() as f64 + 1.0)
                },
                PerformanceMetric::Throughput => model_info.performance_stats.throughput,
                PerformanceMetric::Accuracy => model_info.performance_stats.accuracy.unwrap_or(0.0),
                PerformanceMetric::ErrorRate => 1.0 - model_info.performance_stats.error_rate,
                PerformanceMetric::ResourceUsage => {
                    1.0 / (model_info.resource_usage.cpu_usage + 1.0)
                },
                PerformanceMetric::Cost => {
                    // Placeholder for cost calculation
                    1.0
                },
            };
            score += metric_score * model_weight;
        }

        score
    }

    async fn extract_required_capabilities(&self, _request: &InferenceRequest) -> Vec<String> {
        // Placeholder for capability extraction
        vec!["text-generation".to_string()]
    }

    async fn get_fallback_model(&self, state: &MultiModelState) -> Result<String> {
        if self.config.routing.fallback.enabled {
            let fallback_model = &self.config.routing.fallback.fallback_model;
            if state.models.contains_key(fallback_model)
                && matches!(state.models[fallback_model].status, ModelStatus::Available)
            {
                return Ok(fallback_model.clone());
            }
        }

        // Return first available model
        for (model_id, model_info) in &state.models {
            if matches!(model_info.status, ModelStatus::Available) {
                return Ok(model_id.clone());
            }
        }

        Err(anyhow!("No available models"))
    }

    /// Get server metrics
    pub async fn get_metrics(&self) -> MultiModelMetrics {
        let metrics = self.metrics.lock().await;
        MultiModelMetrics {
            total_requests: metrics.total_requests,
            model_request_counts: metrics.model_request_counts.clone(),
            ensemble_request_counts: metrics.ensemble_request_counts.clone(),
            ab_test_metrics: metrics.ab_test_metrics.clone(),
            avg_routing_time: metrics.avg_routing_time,
            fallback_triggers: metrics.fallback_triggers.clone(),
        }
    }

    /// Start an A/B test experiment
    pub async fn start_ab_test(&self, experiment: ABTestExperiment) -> Result<()> {
        if !self.config.ab_testing.enabled {
            return Err(anyhow::anyhow!("A/B testing is not enabled"));
        }

        // Validate experiment configuration
        self.validate_experiment(&experiment).await?;

        let mut state = self.state.write().await;
        state.active_experiments.insert(experiment.id.clone(), experiment.clone());

        // Initialize metrics for the experiment
        let mut metrics = self.metrics.lock().await;
        metrics.ab_test_metrics.insert(experiment.id.clone(), ABTestMetrics::default());

        info!(
            "Started A/B test experiment: {} with {} variants",
            experiment.id,
            experiment.variant_models.len()
        );
        Ok(())
    }

    /// Stop an A/B test experiment and return results
    pub async fn stop_ab_test(&self, experiment_id: &str) -> Result<ABTestResult> {
        let mut state = self.state.write().await;
        let experiment = state
            .active_experiments
            .remove(experiment_id)
            .ok_or_else(|| anyhow::anyhow!("Experiment not found: {}", experiment_id))?;

        let mut metrics = self.metrics.lock().await;
        let test_metrics = metrics.ab_test_metrics.remove(experiment_id).unwrap_or_default();

        // Analyze experiment results
        let result = self.analyze_ab_test_results(&experiment, &test_metrics).await?;

        info!(
            "Stopped A/B test experiment: {} - Significant: {}",
            experiment_id, result.is_significant
        );
        Ok(result)
    }

    /// Route request for A/B testing
    pub async fn route_ab_test_request(
        &self,
        request_id: &str,
        user_id: Option<&str>,
    ) -> Result<String> {
        let state = self.state.read().await;

        if state.active_experiments.is_empty() {
            return Err(anyhow::anyhow!("No active A/B test experiments"));
        }

        // Select the first active experiment (in production, you might want more sophisticated selection)
        let experiment = state.active_experiments.values().next().unwrap();

        // Determine which variant to use for this request
        let selected_variant = self.select_ab_test_variant(request_id, user_id, experiment).await?;

        // Update metrics
        let mut metrics = self.metrics.lock().await;
        if let Some(ab_metrics) = metrics.ab_test_metrics.get_mut(&experiment.id) {
            if selected_variant == experiment.control_model {
                ab_metrics.control_requests += 1;
            } else {
                *ab_metrics.variant_requests.entry(selected_variant.clone()).or_insert(0) += 1;
            }
        }

        Ok(selected_variant)
    }

    /// Select A/B test variant based on traffic allocation strategy
    async fn select_ab_test_variant(
        &self,
        request_id: &str,
        user_id: Option<&str>,
        experiment: &ABTestExperiment,
    ) -> Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_input = match experiment.traffic_allocation.allocation_method {
            AllocationMethod::Random => request_id.to_string(),
            AllocationMethod::UserId => user_id.unwrap_or(request_id).to_string(),
            AllocationMethod::SessionId => request_id.to_string(), // Use request_id as session fallback
            AllocationMethod::IPAddress => request_id.to_string(), // Use request_id as IP fallback
            AllocationMethod::Custom(ref key) => key.clone(),
        };

        let mut hasher = DefaultHasher::new();
        hash_input.hash(&mut hasher);
        let hash_value = hasher.finish();

        // Convert hash to percentage (0.0 to 1.0)
        let percentage = (hash_value as f64) / (u64::MAX as f64);

        // Determine which bucket this falls into
        let mut cumulative_percentage = 0.0;

        // Check control group first
        cumulative_percentage += experiment.traffic_allocation.control_percentage;
        if percentage < cumulative_percentage {
            return Ok(experiment.control_model.clone());
        }

        // Check variant groups
        for variant in &experiment.variant_models {
            cumulative_percentage += variant.traffic_percentage;
            if percentage < cumulative_percentage {
                return Ok(variant.model.clone());
            }
        }

        // Fallback to control if somehow we didn't match anything
        Ok(experiment.control_model.clone())
    }

    /// Record A/B test performance metrics
    pub async fn record_ab_test_metrics(
        &self,
        experiment_id: &str,
        model_id: &str,
        latency: Duration,
        success: bool,
        quality_score: Option<f64>,
    ) -> Result<()> {
        let mut metrics = self.metrics.lock().await;
        if let Some(ab_metrics) = metrics.ab_test_metrics.get_mut(experiment_id) {
            let perf_stats = if model_id == self.get_control_model(experiment_id).await? {
                &mut ab_metrics.control_performance
            } else {
                ab_metrics
                    .variant_performance
                    .entry(model_id.to_string())
                    .or_insert_with(PerformanceStats::default)
            };

            // Update latency statistics (simplified moving average)
            if perf_stats.avg_latency == Duration::from_millis(0) {
                perf_stats.avg_latency = latency;
            } else {
                // Simple exponential moving average
                let current_ms = perf_stats.avg_latency.as_millis() as f64;
                let new_ms = latency.as_millis() as f64;
                let updated_ms = (current_ms * 0.9 + new_ms * 0.1) as u64;
                perf_stats.avg_latency = Duration::from_millis(updated_ms);
            }

            // Update error rate (simplified)
            if !success {
                perf_stats.error_rate = (perf_stats.error_rate * 0.9) + 0.1;
            } else {
                perf_stats.error_rate = perf_stats.error_rate * 0.9;
            }

            // Update accuracy if quality score provided
            if let Some(score) = quality_score {
                perf_stats.accuracy = Some(score);
            }
        }
        Ok(())
    }

    /// Validate A/B test experiment configuration
    async fn validate_experiment(&self, experiment: &ABTestExperiment) -> Result<()> {
        // Check that control model exists
        let state = self.state.read().await;
        if !state.models.contains_key(&experiment.control_model) {
            return Err(anyhow::anyhow!(
                "Control model not found: {}",
                experiment.control_model
            ));
        }

        // Check that all variant models exist
        for variant in &experiment.variant_models {
            if !state.models.contains_key(&variant.model) {
                return Err(anyhow::anyhow!(
                    "Variant model not found: {}",
                    variant.model
                ));
            }
        }

        // Validate traffic percentages sum to 100%
        let total_percentage = experiment.traffic_allocation.control_percentage
            + experiment.variant_models.iter().map(|v| v.traffic_percentage).sum::<f64>();

        if (total_percentage - 1.0).abs() > 0.001 {
            return Err(anyhow::anyhow!(
                "Traffic percentages must sum to 100%, got: {:.1}%",
                total_percentage * 100.0
            ));
        }

        Ok(())
    }

    /// Get control model for an experiment
    async fn get_control_model(&self, experiment_id: &str) -> Result<String> {
        let state = self.state.read().await;
        let experiment = state
            .active_experiments
            .get(experiment_id)
            .ok_or_else(|| anyhow::anyhow!("Experiment not found: {}", experiment_id))?;
        Ok(experiment.control_model.clone())
    }

    /// Analyze A/B test results and calculate statistical significance
    async fn analyze_ab_test_results(
        &self,
        experiment: &ABTestExperiment,
        metrics: &ABTestMetrics,
    ) -> Result<ABTestResult> {
        let control_perf = &metrics.control_performance;
        let total_control = metrics.control_requests;

        let mut variant_results = Vec::new();

        for variant in &experiment.variant_models {
            if let Some(variant_perf) = metrics.variant_performance.get(&variant.model) {
                let total_variant = *metrics.variant_requests.get(&variant.model).unwrap_or(&0);

                // Calculate statistical significance using z-test for proportions (simplified)
                let significance = self
                    .calculate_statistical_significance(
                        control_perf,
                        total_control,
                        variant_perf,
                        total_variant,
                    )
                    .await?;

                // Determine winner based on primary success metric (use first one)
                let primary_metric =
                    experiment.success_metrics.get(0).unwrap_or(&SuccessMetric::Accuracy);
                let is_better = match primary_metric {
                    SuccessMetric::Accuracy => {
                        let variant_acc = variant_perf.accuracy.unwrap_or(0.0);
                        let control_acc = control_perf.accuracy.unwrap_or(0.0);
                        variant_acc > control_acc
                    },
                    SuccessMetric::Latency => variant_perf.avg_latency < control_perf.avg_latency,
                    SuccessMetric::UserSatisfaction => {
                        variant_perf.error_rate < control_perf.error_rate
                    }, // Lower error rate = better satisfaction
                    SuccessMetric::ConversionRate => {
                        variant_perf.error_rate < control_perf.error_rate
                    }, // Lower error rate = better conversion
                    SuccessMetric::ErrorRate => variant_perf.error_rate < control_perf.error_rate, // Lower is better
                    SuccessMetric::Custom(_) => {
                        let variant_acc = variant_perf.accuracy.unwrap_or(0.0);
                        let control_acc = control_perf.accuracy.unwrap_or(0.0);
                        variant_acc > control_acc
                    },
                };

                variant_results.push(ABTestVariantResult {
                    variant_id: variant.id.clone(),
                    model_id: variant.model.clone(),
                    performance: variant_perf.clone(),
                    statistical_significance: significance,
                    is_better_than_control: is_better,
                    confidence_interval: self
                        .calculate_confidence_interval(variant_perf, total_variant)
                        .await?,
                });
            }
        }

        // Get significance thresholds from config
        let significance_thresholds = &self.config.ab_testing.significance_thresholds;

        // Determine overall significance
        let is_significant = variant_results
            .iter()
            .any(|r| r.statistical_significance < significance_thresholds.p_value);

        let winner = if is_significant {
            variant_results
                .iter()
                .filter(|r| {
                    r.is_better_than_control
                        && r.statistical_significance < significance_thresholds.p_value
                })
                .min_by(|a, b| {
                    a.statistical_significance.partial_cmp(&b.statistical_significance).unwrap()
                })
                .map(|r| r.model_id.clone())
        } else {
            None
        };

        Ok(ABTestResult {
            experiment_id: experiment.id.clone(),
            is_significant,
            winner,
            control_performance: control_perf.clone(),
            variant_results,
            total_requests: total_control + metrics.variant_requests.values().sum::<u64>(),
            duration: chrono::Duration::from_std(experiment.duration)
                .unwrap_or_else(|_| chrono::Duration::zero()), // Convert to chrono::Duration
            confidence_level: significance_thresholds.confidence_level,
        })
    }

    /// Calculate statistical significance using error rate comparison
    async fn calculate_statistical_significance(
        &self,
        control: &PerformanceStats,
        control_n: u64,
        variant: &PerformanceStats,
        variant_n: u64,
    ) -> Result<f64> {
        if control_n == 0 || variant_n == 0 {
            return Ok(1.0); // No significance if no data
        }

        let p1 = 1.0 - control.error_rate; // Success rate = 1 - error rate
        let p2 = 1.0 - variant.error_rate; // Success rate = 1 - error rate
        let n1 = control_n as f64;
        let n2 = variant_n as f64;

        // If sample sizes are too small, return no significance
        if n1 < 30.0 || n2 < 30.0 {
            return Ok(0.5);
        }

        // Pooled proportion
        let p_pool = (n1 * p1 + n2 * p2) / (n1 + n2);

        // Standard error
        let se = (p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)).sqrt();

        if se == 0.0 {
            return Ok(1.0);
        }

        // Z-score
        let z = (p2 - p1).abs() / se;

        // Convert z-score to p-value (two-tailed test, simplified)
        // This is a simplified calculation - in production you'd use a proper statistics library
        let p_value = if z > 2.576 {
            0.01
        } else if z > 1.96 {
            0.05
        } else if z > 1.645 {
            0.1
        } else {
            0.5
        };

        Ok(p_value)
    }

    /// Calculate confidence interval for performance metric
    async fn calculate_confidence_interval(
        &self,
        perf: &PerformanceStats,
        n: u64,
    ) -> Result<(f64, f64)> {
        if n == 0 {
            return Ok((0.0, 0.0));
        }

        let p = 1.0 - perf.error_rate; // Success rate = 1 - error rate
        let n_f = n as f64;

        // 95% confidence interval for proportion
        let z_score = 1.96; // 95% confidence
        let margin_error = z_score * (p * (1.0 - p) / n_f).sqrt();

        Ok(((p - margin_error).max(0.0), (p + margin_error).min(1.0)))
    }

    /// Get registered models
    pub async fn get_models(&self) -> HashMap<String, ModelInfo> {
        let state = self.state.read().await;
        state.models.clone()
    }
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

impl Default for MultiModelConfig {
    fn default() -> Self {
        Self {
            routing: ModelRoutingConfig {
                default_strategy: RoutingStrategy::RoundRobin,
                route_strategies: HashMap::new(),
                selection_criteria: ModelSelectionCriteria {
                    preferred_characteristics: Vec::new(),
                    quality_thresholds: QualityThresholds {
                        min_accuracy: None,
                        max_latency: None,
                        max_error_rate: None,
                        min_throughput: None,
                    },
                    resource_constraints: ResourceConstraints {
                        max_memory_usage: None,
                        max_gpu_memory: None,
                        max_cpu_usage: None,
                        required_gpu_count: None,
                    },
                },
                fallback: FallbackConfig {
                    fallback_model: "default".to_string(),
                    enabled: true,
                    triggers: vec![
                        FallbackTrigger::ModelUnavailable,
                        FallbackTrigger::HighErrorRate(0.1),
                    ],
                },
            },
            ensemble: EnsembleConfig {
                enabled: false,
                methods: Vec::new(),
                voting_strategy: VotingStrategy::SimpleMajority,
                quality_assessment: QualityAssessmentConfig {
                    enabled: false,
                    methods: Vec::new(),
                    thresholds: QualityThresholds {
                        min_accuracy: None,
                        max_latency: None,
                        max_error_rate: None,
                        min_throughput: None,
                    },
                },
                optimization: EnsembleOptimizationConfig {
                    enabled: false,
                    strategies: Vec::new(),
                    resource_budget: ResourceBudget {
                        max_latency: Duration::from_secs(10),
                        max_memory: 1024 * 1024 * 1024, // 1GB
                        max_compute_cost: 1.0,
                    },
                },
            },
            ab_testing: ABTestingConfig {
                enabled: false,
                experiments: Vec::new(),
                significance_thresholds: StatisticalThresholds {
                    p_value: 0.05,
                    confidence_level: 0.95,
                    minimum_sample_size: 1000,
                    minimum_effect_size: 0.1,
                },
            },
            traffic_splitting: TrafficSplittingConfig {
                enabled: false,
                split_rules: Vec::new(),
                default_split: TrafficSplit {
                    splits: HashMap::from([("default".to_string(), 100.0)]),
                    sticky_sessions: false,
                },
            },
            model_cascading: ModelCascadingConfig {
                enabled: false,
                cascade_chains: Vec::new(),
                exit_strategies: Vec::new(),
            },
            performance_monitoring: PerformanceMonitoringConfig {
                enabled: true,
                collection_interval: Duration::from_secs(60),
                monitored_metrics: vec![
                    MonitoredMetric::ModelLatency,
                    MonitoredMetric::ErrorRates,
                    MonitoredMetric::ResourceUtilization,
                ],
                alerting_thresholds: AlertingThresholds {
                    latency_threshold: Duration::from_secs(5),
                    error_rate_threshold: 0.05,
                    accuracy_threshold: 0.9,
                    resource_threshold: 0.8,
                },
            },
        }
    }
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            avg_latency: Duration::from_millis(0),
            p95_latency: Duration::from_millis(0),
            p99_latency: Duration::from_millis(0),
            throughput: 0.0,
            error_rate: 0.0,
            accuracy: None,
            request_count: 0,
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            gpu_memory_usage: 0,
            network_io: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multi_model_server_creation() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);

        let models = server.get_models().await;
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_model_registration() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);

        let model_info = ModelInfo {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            version: "1.0".to_string(),
            characteristics: vec![ModelCharacteristic::Size(ModelSize::Small)],
            capabilities: vec!["text-generation".to_string()],
            status: ModelStatus::Available,
            performance_stats: PerformanceStats::default(),
            resource_usage: ResourceUsage::default(),
            metadata: HashMap::new(),
        };

        let result = server.register_model(model_info).await;
        assert!(result.is_ok());

        let models = server.get_models().await;
        assert_eq!(models.len(), 1);
    }

    #[tokio::test]
    async fn test_request_routing() {
        let config = MultiModelConfig::default();
        let server = MultiModelServer::new(config);

        // Register a model
        let model_info = ModelInfo {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            version: "1.0".to_string(),
            characteristics: vec![ModelCharacteristic::Size(ModelSize::Small)],
            capabilities: vec!["text-generation".to_string()],
            status: ModelStatus::Available,
            performance_stats: PerformanceStats::default(),
            resource_usage: ResourceUsage::default(),
            metadata: HashMap::new(),
        };

        server.register_model(model_info).await.unwrap();

        let request = InferenceRequest {
            input_text: "Hello world".to_string(),
            path: "/v1/inference".to_string(),
            headers: HashMap::new(),
            user_id: None,
            metadata: HashMap::new(),
        };

        let result = server.route_request(&request).await;
        assert!(result.is_ok());

        match result.unwrap() {
            RoutingResult::SingleModel { model_id } => {
                assert_eq!(model_id, "test-model");
            },
            other => {
                // Use assert! to provide better error message in tests
                assert!(
                    matches!(other, RoutingResult::SingleModel { .. }),
                    "Expected single model routing, got: {:?}",
                    other
                );
            },
        }
    }
}
