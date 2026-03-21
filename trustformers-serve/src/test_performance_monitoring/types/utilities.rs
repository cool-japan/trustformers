//! Utilities Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

// Import types from sibling modules
use super::config::TestPerformanceMonitoringConfig;
use super::enums::{
    AnalysisDepth, BaselineType, ComparisonMethod, ComparisonOperator, GroupStatus, GroupType,
    MetricValue, OptimizationStrategy, RecoveryConditionType, SensitivityLevel, SuppressionLevel,
    Trend, UpdateType,
};
use super::storage::ArchivalFilter;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CleanupSchedule {
    /// Cleanup interval
    pub interval: Duration,
    /// Enable automatic cleanup
    pub auto_cleanup: bool,
}

impl Default for CleanupSchedule {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(24 * 3600), // Daily
            auto_cleanup: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionEvaluator {
    config: TestPerformanceMonitoringConfig,
}

impl ConditionEvaluator {
    pub fn new(config: &TestPerformanceMonitoringConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationContext {
    pub variables: HashMap<String, MetricValue>,
    pub timestamp: Option<DateTime<Utc>>,
}

impl Default for EvaluationContext {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
            timestamp: None,
        }
    }
}

impl EvaluationContext {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecoveryTracker {
    pub recovery_attempts: u32,
    pub success_count: u32,
    pub last_recovery: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnhancedLatencyProcessor {
    pub processing_times: Vec<Duration>,
    pub avg_latency: Duration,
    pub max_latency: Duration,
}

impl Default for EnhancedLatencyProcessor {
    fn default() -> Self {
        Self {
            processing_times: Vec::new(),
            avg_latency: Duration::from_secs(0),
            max_latency: Duration::from_secs(0),
        }
    }
}

impl EnhancedLatencyProcessor {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnhancedResourceUtilizationProcessor {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub gpu_utilization: f64,
    pub network_utilization: f64,
}

impl Default for EnhancedResourceUtilizationProcessor {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            gpu_utilization: 0.0,
            network_utilization: 0.0,
        }
    }
}

impl EnhancedResourceUtilizationProcessor {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionContext {
    pub variables: std::collections::HashMap<String, f64>,
    pub timestamp: DateTime<Utc>,
}

impl Default for ConditionContext {
    fn default() -> Self {
        Self {
            variables: std::collections::HashMap::new(),
            timestamp: Utc::now(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub min_improvement: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RealTimeProcessor {
    pub enabled: bool,
    pub buffer_size: usize,
    pub flush_interval: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationCost {
    pub cpu_ms: f64,
    pub memory_bytes: u64,
    pub io_ops: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationMetadata {
    pub evaluator_type: String,
    pub evaluation_time: DateTime<Utc>,
    pub data_points_evaluated: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImpactAssessment {
    pub severity: String,
    pub affected_systems: Vec<String>,
    pub estimated_cost: f64,
    pub impact_level: String,
    pub affected_users: usize,
    pub business_impact: String,
    pub estimated_downtime: Duration,
    pub financial_impact: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuppressionCondition {
    pub condition_type: String,
    pub parameters: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EstimatedEffort {
    pub person_hours: f64,
    pub complexity: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpectedImpact {
    pub improvement_percentage: f64,
    pub affected_metrics: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutomaticAction {
    pub action_type: String,
    pub parameters: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RateLimits {
    pub requests_per_second: u32,
    pub burst_size: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationResult {
    pub verified: bool,
    pub checksum_match: bool,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeBucket {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub bucket_size: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Percentiles {
    pub p1: f64,
    pub p5: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChangePoint {
    pub timestamp: DateTime<Utc>,
    pub confidence: f64,
    pub change_magnitude: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeSeriesFilter {
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub metric_names: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchiveRequest {
    pub request_id: String,
    pub filter: ArchivalFilter,
    pub destination: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiskFactor {
    pub factor_name: String,
    pub severity: f64,
    pub probability: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StabilityAssessment {
    pub stability_score: f64,
    pub variance: f64,
    pub trend: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubscriptionFilter {
    pub filter_type: String,
    pub patterns: Vec<String>,
    pub exclude_patterns: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RateLimit {
    pub max_requests: u32,
    pub time_window: Duration,
    pub burst_size: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalCorrelator {
    pub correlation_threshold: f64,
    pub time_window: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpatialCorrelator {
    pub correlation_threshold: f64,
    pub scope: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveSuppression {
    pub suppression_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationCheck {
    pub check_name: String,
    pub check_type: String,
    pub enabled: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConnectionCharacteristics {
    pub connection_type: String,
    pub latency_ms: f64,
    pub throughput_mbps: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BandwidthUtilization {
    pub current_mbps: f64,
    pub available_mbps: f64,
    pub utilization_percent: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NetworkLatencyProfile {
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NetworkReliability {
    pub packet_loss_rate: f64,
    pub retransmission_rate: f64,
    pub connection_stability: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecoveryCharacteristics {
    pub recovery_time_ms: f64,
    pub success_rate: f64,
    pub retry_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StabilityIndicators {
    pub stability_score: f64,
    pub variance: f64,
    pub trend: String,
}

impl Default for ComparisonOperator {
    fn default() -> Self {
        Self::Equal
    }
}

impl Default for GroupType {
    fn default() -> Self {
        Self::BySource
    }
}

impl Default for GroupStatus {
    fn default() -> Self {
        Self::Active
    }
}

impl Default for SuppressionLevel {
    fn default() -> Self {
        Self::Low
    }
}

impl Default for RecoveryConditionType {
    fn default() -> Self {
        Self::MetricReturnsToNormal
    }
}

impl Default for Trend {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeakIndicator {
    pub detected: bool,
    pub leak_rate: f64,
    pub confidence: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ThreadUtilization {
    pub thread_count: usize,
    pub utilization_percent: f64,
    pub blocked_threads: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuBoundPhase {
    pub duration: Duration,
    pub cpu_usage: f64,
    pub start_time: DateTime<Utc>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LatencyCharacteristics {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderingConstraint {
    pub before_test: String,
    pub after_test: String,
    pub constraint_type: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedSharingCapability {
    pub can_share: bool,
    pub sharing_mode: String,
    pub cache_timestamp: DateTime<Utc>,
    /// Sharing capability result
    pub result: String,
    /// Cached at timestamp
    pub cached_at: DateTime<Utc>,
    /// Confidence score
    pub confidence: f64,
}

impl CachedSharingCapability {
    /// Check if cached data is still valid (within 5 minutes)
    pub fn is_valid(&self) -> bool {
        let now = chrono::Utc::now();
        let age = now.signed_duration_since(self.cached_at);
        age.num_seconds() < 300 // Valid for 5 minutes
    }
}

impl Default for SensitivityLevel {
    fn default() -> Self {
        Self::Medium
    }
}

impl Default for AnalysisDepth {
    fn default() -> Self {
        Self::Normal
    }
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        Self::Balanced
    }
}

impl Default for ComparisonMethod {
    fn default() -> Self {
        Self::Relative
    }
}

impl Default for BaselineType {
    fn default() -> Self {
        Self::Rolling
    }
}

impl Default for OrderingConstraint {
    fn default() -> Self {
        Self {
            before_test: String::new(),
            after_test: String::new(),
            constraint_type: "temporal".to_string(),
            reason: String::new(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DataSource {
    pub source_type: String,
    pub query: String,
    pub parameters: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SeverityDistribution {
    pub critical: u64,
    pub high: u64,
    pub medium: u64,
    pub low: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FalsePositiveAssessment {
    pub false_positive_rate: f64,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImprovementOpportunity {
    pub opportunity_type: String,
    pub potential_gain: f64,
    pub implementation_cost: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StatisticalSignificance {
    pub p_value: f64,
    pub confidence_level: f64,
    pub is_significant: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Evidence {
    pub evidence_type: String,
    pub data: serde_json::Value,
    pub confidence: f64,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImplementationRoadmap {
    pub phases: Vec<String>,
    pub timeline: Duration,
    pub dependencies: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DependencyAnalyzer {
    pub analysis_depth: u32,
    pub detected_dependencies: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynchronizationAnalyzer {
    pub sync_points: Vec<String>,
    pub contention_detected: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateData {
    pub update_type: UpdateType,
    pub data: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessingRule {
    pub rule_id: String,
    pub condition: String,
    pub action: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthIndicator {
    pub indicator_name: String,
    pub current_value: f64,
    pub threshold: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComplianceFlag {
    pub flag_name: String,
    pub is_compliant: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackoffStrategy {
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub multiplier: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetryPredicate {
    pub max_attempts: u32,
    pub backoff: BackoffStrategy,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubscriptionError {
    pub error_code: String,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchingAlgorithm {
    pub algorithm_name: String,
    pub parameters: HashMap<String, String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimeConstraint {
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ContextRequirement {
    pub required_fields: Vec<String>,
    pub optional_fields: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregationRule {
    pub rule_id: String,
    pub aggregation_type: String,
    pub window_size: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregationWindow {
    pub window_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub window_size: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeletionCriteria {
    pub age: Duration,
    pub size_threshold: usize,
    pub access_count: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeletionStrategy {
    pub strategy_type: String,
    pub criteria: Vec<String>,
    pub dry_run: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComplianceRequirement {
    pub requirement_id: String,
    pub description: String,
    pub enforcement_level: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccessFrequency {
    pub access_count: usize,
    pub last_access: DateTime<Utc>,
    pub frequency_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LifecycleCondition {
    pub condition_type: String,
    pub threshold: f64,
    pub duration: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LifecycleAction {
    pub action_type: String,
    pub parameters: HashMap<String, String>,
    pub priority: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgorithmSelection {
    pub algorithm: String,
    pub level: i32,
    pub auto_select: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalCache {
    pub cache_id: String,
    pub max_size: usize,
    pub ttl: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetadataPreservation {
    pub preserve_timestamps: bool,
    pub preserve_permissions: bool,
    pub preserve_attributes: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalOptions {
    pub priority: i32,
    pub timeout: Duration,
    pub decompress: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LifecycleStateTracker {
    pub state_id: String,
    pub current_state: String,
    pub last_transition: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostOptimizer {
    pub optimizer_id: String,
    pub optimization_strategy: String,
    pub target_cost: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransitionRule {
    pub rule_id: String,
    pub source_state: String,
    pub target_state: String,
    pub conditions: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostConstraint {
    pub constraint_type: String,
    pub max_cost: f64,
    pub enforcement_level: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComplianceRule {
    pub rule_id: String,
    pub description: String,
    pub enforcement_level: String,
    pub audit_required: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostTargets {
    pub max_monthly_cost: f64,
    pub cost_per_gb: f64,
    pub optimization_priority: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityRequirements {
    pub min_quality_score: f64,
    pub validation_required: bool,
    pub integrity_checks: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SortingSpec {
    pub field: String,
    pub order: String,
    pub nulls_first: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HavingCondition {
    pub field: String,
    pub operator: String,
    pub value: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionalDisplay {
    pub condition: String,
    pub show_if_true: bool,
    pub fallback_content: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataAggregator {
    pub aggregator_id: String,
    pub aggregation_type: String,
    pub group_by: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContentProcessor {
    pub processor_id: String,
    pub transformations: Vec<String>,
    pub filters: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentHealthSummary {
    pub component_id: String,
    pub health_status: String,
    pub issues_count: usize,
    pub last_check: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CriticalIssue {
    pub issue_id: String,
    pub severity: String,
    pub description: String,
    pub affected_components: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceOperationMetadata {
    pub operation_id: String,
    pub operation_type: String,
    pub timestamp: DateTime<Utc>,
    pub duration: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuietHours {
    pub enabled: bool,
    pub start_time: String,
    pub end_time: String,
    pub timezone: String,
    pub days_of_week: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoodnessOfFit {
    pub test_statistic: f64,
    pub p_value: f64,
    pub test_type: String,
    pub distribution: String,
    pub sample_size: usize,
}
