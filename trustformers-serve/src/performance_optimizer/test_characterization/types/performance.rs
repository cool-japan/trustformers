use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

// Import commonly used types from core
use super::core::{PriorityLevel, TestCharacterizationResult};

// Import cross-module types
use super::analysis::InsightEngine;
use super::patterns::PatternOutcome;
use super::quality::QualityAssessment;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EfficiencyRating {
    /// Very low efficiency
    VeryLow,
    /// Low efficiency
    Low,
    /// Medium efficiency
    Medium,
    /// High efficiency
    High,
    /// Very high efficiency
    VeryHigh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProfilingDepth {
    /// Surface level profiling
    Surface,
    /// Basic profiling
    Basic,
    /// Standard profiling
    Standard,
    /// Detailed profiling
    Detailed,
    /// Deep profiling
    Deep,
    /// Comprehensive profiling
    Comprehensive,
    /// Exhaustive profiling
    Exhaustive,
}

impl Default for ProfilingDepth {
    fn default() -> Self {
        ProfilingDepth::Standard
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkExecutionState {
    pub state: String,
    pub current_benchmark: String,
    pub progress_percentage: f64,
    pub start_time: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkOrchestrator {
    pub orchestration_enabled: bool,
    pub benchmarks: Vec<String>,
    pub execution_order: Vec<String>,
    pub parallel_execution: bool,
}

#[derive(Debug, Clone)]
pub struct BenchmarkSuiteDefinition {
    pub suite_name: String,
    pub suite_version: String,
    pub benchmarks: Vec<String>,
    pub configuration: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkSuiteResults {
    pub suite_name: String,
    pub results: HashMap<String, f64>,
    pub overall_score: f64,
    pub execution_time: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct EffectivenessComparison {
    pub baseline_id: String,
    pub comparison_id: String,
    pub effectiveness_delta: f64,
    pub statistical_significance: f64,
    pub confidence_level: f64,
}

#[derive(Debug, Clone)]
pub struct EffectivenessContext {
    /// System configuration
    pub system_config: HashMap<String, String>,
    /// Test environment
    pub test_environment: String,
    /// Load conditions
    pub load_conditions: HashMap<String, f64>,
    /// Resource availability
    pub resource_availability: HashMap<String, f64>,
    /// Performance constraints
    pub constraints: Vec<String>,
    /// Measurement objectives
    pub objectives: Vec<String>,
    /// Environmental factors
    pub environmental_factors: HashMap<String, f64>,
    /// Time period
    pub time_period: (Instant, Instant),
    /// External dependencies
    pub external_dependencies: Vec<String>,
}

impl Default for EffectivenessContext {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            system_config: HashMap::new(),
            test_environment: "default".to_string(),
            load_conditions: HashMap::new(),
            resource_availability: HashMap::new(),
            constraints: Vec::new(),
            objectives: Vec::new(),
            environmental_factors: HashMap::new(),
            time_period: (now, now),
            external_dependencies: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectivenessMetrics {
    pub overall_effectiveness: f64,
    pub performance_improvement: f64,
    pub resource_efficiency: f64,
    pub cost_effectiveness: f64,
    pub success_rate: f64,
}

impl Default for EffectivenessMetrics {
    fn default() -> Self {
        Self {
            overall_effectiveness: 0.0,
            performance_improvement: 0.0,
            resource_efficiency: 0.0,
            cost_effectiveness: 0.0,
            success_rate: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EffectivenessRecord {
    /// Pattern identifier
    pub pattern_id: String,
    /// Effectiveness context
    pub context: EffectivenessContext,
    /// Measured outcomes
    pub outcomes: Vec<PatternOutcome>,
    /// Effectiveness score
    pub effectiveness_score: f64,
    /// Measurement timestamp
    pub measured_at: Instant,
    /// Measurement duration
    pub measurement_duration: Duration,
    /// Baseline comparison
    pub baseline_comparison: f64,
    /// Statistical significance
    pub significance: f64,
    /// Confidence level
    pub confidence: f64,
    /// Validation status
    pub validated: bool,
}

#[derive(Debug, Clone)]
pub struct EffectivenessTracker {
    /// Effectiveness records
    pub records: HashMap<String, EffectivenessRecord>,
    pub effectiveness_records: HashMap<String, EffectivenessRecord>,
    /// Tracking metrics
    pub metrics: EffectivenessMetrics,
    /// Trend analysis
    pub trends: HashMap<String, EffectivenessTrend>,
    /// Baseline measurements
    pub baselines: HashMap<String, f64>,
    /// Comparison results
    pub comparisons: Vec<EffectivenessComparison>,
    /// Quality assessments
    pub quality_assessments: Vec<QualityAssessment>,
    /// Performance improvements
    pub improvements: HashMap<String, f64>,
    /// ROI calculations
    pub roi_calculations: HashMap<String, f64>,
    /// Success criteria
    pub success_criteria: Vec<String>,
    /// Effectiveness context
    pub context: EffectivenessContext,
    /// Pattern outcomes
    pub outcomes: Vec<PatternOutcome>,
    /// Overall effectiveness score
    pub effectiveness_score: f64,
    /// Measurement timestamp
    pub measurement_timestamp: Instant,
    /// Tracking metadata
    pub tracking_metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct EffectivenessTrend {
    pub trend_direction: String,
    pub trend_strength: f64,
    pub historical_data: Vec<(chrono::DateTime<chrono::Utc>, f64)>,
    pub predicted_values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EfficiencyCalculator {
    pub calculation_method: String,
    pub normalization_enabled: bool,
    pub weighting_factors: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct EfficiencyCurve {
    pub data_points: Vec<(f64, f64)>,
    pub curve_type: String,
    pub peak_efficiency: f64,
    pub optimal_point: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct EfficiencyFunction {
    pub function_type: String,
    pub parameters: Vec<f64>,
    pub domain: (f64, f64),
    pub range: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    pub cpu_efficiency: f64,
    pub memory_efficiency: f64,
    pub io_efficiency: f64,
    pub overall_efficiency: f64,
    pub efficiency_rating: EfficiencyRating,
}

#[derive(Debug, Clone)]
pub struct EfficiencyTrends {
    pub trend_data: Vec<(chrono::DateTime<chrono::Utc>, f64)>,
    pub trend_direction: String,
    pub average_efficiency: f64,
    pub efficiency_volatility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceMetrics {
    pub failure_rate: f64,
    #[serde(skip)]
    pub recovery_time: std::time::Duration,
    #[serde(skip)]
    pub fault_detection_time: std::time::Duration,
    pub resilience_score: f64,
}

#[derive(Debug, Clone)]
pub struct GcImpactMetrics {
    pub gc_pause_time: std::time::Duration,
    pub gc_frequency: f64,
    pub memory_reclaimed: usize,
    pub performance_impact: f64,
}

#[derive(Debug, Clone)]
pub struct MetricCriteria {
    pub metric_name: String,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub target_value: Option<f64>,
    pub tolerance: f64,
}

#[derive(Debug, Clone)]
pub struct MetricIndex {
    pub index_name: String,
    pub metrics: Vec<String>,
    pub index_value: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct MetricPrediction {
    pub metric_name: String,
    pub predicted_value: f64,
    pub prediction_confidence: f64,
    pub prediction_horizon: std::time::Duration,
    pub prediction_method: String,
}

#[derive(Debug, Clone)]
pub struct MetricsCollectorConfig {
    pub collection_enabled: bool,
    pub collection_interval: std::time::Duration,
    pub metrics_to_collect: Vec<String>,
    pub storage_retention: std::time::Duration,
    /// Enable CPU monitoring
    pub monitor_cpu: bool,
    /// Enable memory monitoring
    pub monitor_memory: bool,
    /// Enable I/O monitoring
    pub monitor_io: bool,
    /// Enable network monitoring
    pub monitor_network: bool,
}

impl Default for MetricsCollectorConfig {
    fn default() -> Self {
        Self {
            collection_enabled: true,
            collection_interval: std::time::Duration::from_secs(1),
            metrics_to_collect: vec![],
            storage_retention: std::time::Duration::from_secs(3600),
            monitor_cpu: true,
            monitor_memory: true,
            monitor_io: true,
            monitor_network: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceAnalysisPipeline {
    /// Pipeline stages
    pub stages: Vec<String>,
    /// Processing enabled
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct PerformanceAnalysisReport {
    pub report_id: String,
    pub analysis_summary: String,
    pub key_findings: Vec<String>,
    pub recommendations: Vec<String>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub baseline_id: String,
    pub baseline_metrics: HashMap<String, f64>,
    pub established_at: chrono::DateTime<chrono::Utc>,
    pub baseline_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConstraints {
    /// Maximum execution time
    #[serde(skip)]
    pub max_execution_time: Option<Duration>,
    /// Maximum memory usage
    pub max_memory_usage: Option<usize>,
    /// Maximum CPU utilization
    pub max_cpu_utilization: Option<f64>,
    /// Maximum I/O rate
    pub max_io_rate: Option<f64>,
    /// Maximum network bandwidth
    pub max_network_bandwidth: Option<f64>,
    /// Quality thresholds
    pub quality_thresholds: HashMap<String, f64>,
    /// Performance benchmarks
    pub benchmarks: HashMap<String, f64>,
    /// SLA requirements
    pub sla_requirements: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PerformanceCorrelations {
    pub metric_pairs: Vec<(String, String)>,
    pub correlation_coefficients: HashMap<(String, String), f64>,
    pub statistical_significance: HashMap<(String, String), f64>,
}

#[derive(Debug, Clone)]
pub struct PerformanceData {
    pub data_points: Vec<(chrono::DateTime<chrono::Utc>, HashMap<String, f64>)>,
    pub metrics: Vec<String>,
    pub collection_period: std::time::Duration,
    /// Overall performance score
    pub overall_score: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceDelta {
    /// Metric name
    pub metric: String,
    /// Before value
    pub before_value: f64,
    /// After value
    pub after_value: f64,
    /// Absolute change
    pub absolute_change: f64,
    /// Percentage change
    pub percentage_change: f64,
    /// Improvement indicator
    pub improvement: bool,
    /// Statistical significance
    pub significance: f64,
    /// Measurement confidence
    pub confidence: f64,
    /// Measurement method
    pub measurement_method: String,
    /// Time period
    pub time_period: Duration,
}

#[derive(Debug, Clone)]
pub struct PerformanceGuarantees {
    pub guaranteed_metrics: HashMap<String, f64>,
    pub confidence_level: f64,
    pub validity_period: std::time::Duration,
    pub penalty_conditions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PerformanceImpactAnalysis {
    pub degradation: f64,
    pub impact_areas: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PerformanceInsightEngine {
    /// Insights generated
    pub insights_generated: u64,
    /// Insight quality score
    pub quality_score: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub throughput: f64,
    pub latency: std::time::Duration,
    pub response_time: std::time::Duration,
    pub error_rate: f64,
    pub resource_utilization: HashMap<String, f64>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            throughput: 0.0,
            latency: Duration::default(),
            response_time: Duration::default(),
            error_rate: 0.0,
            resource_utilization: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformancePatterns {
    pub patterns: Vec<String>,
    pub pattern_frequencies: HashMap<String, f64>,
    pub dominant_pattern: String,
}

#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    pub predicted_metrics: HashMap<String, f64>,
    pub prediction_confidence: f64,
    pub prediction_time_horizon: std::time::Duration,
    pub prediction_model: String,
    pub horizon: std::time::Duration,
    pub predictions: HashMap<String, f64>,
    pub confidence_level: f64,
    pub generated_at: std::time::Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// Average execution time
    #[serde(skip)]
    pub average_execution_time: Duration,
    /// Execution time variance
    #[serde(skip)]
    pub execution_time_variance: Duration,
    /// Resource usage peaks
    pub resource_peaks: HashMap<String, f64>,
    /// Throughput characteristics
    pub throughput: f64,
    /// Latency characteristics
    #[serde(skip)]
    pub latency_distribution: Vec<Duration>,
    /// Scalability characteristics
    pub scalability_factors: HashMap<String, f64>,
    /// Efficiency metrics
    pub efficiency_metrics: HashMap<String, f64>,
    /// Performance trends
    pub trends: Vec<PerformanceTrend>,
    /// Baseline comparisons
    pub baseline_comparisons: HashMap<String, f64>,
    /// Performance predictability
    pub predictability_score: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceReportGenerator {
    pub report_interval_ms: u64,
    pub include_charts: bool,
}

impl Default for PerformanceReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceReportGenerator {
    pub fn new() -> Self {
        Self {
            report_interval_ms: 1000,
            include_charts: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceSample {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metrics: HashMap<String, f64>,
    pub sample_quality: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    pub target_metrics: HashMap<String, f64>,
    pub target_achievement_deadline: chrono::DateTime<chrono::Utc>,
    pub priority: PriorityLevel,
}

#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub warning_thresholds: HashMap<String, f64>,
    pub critical_thresholds: HashMap<String, f64>,
    pub threshold_actions: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub trend_type: String,
    pub trend_direction: String,
    pub trend_strength: f64,
    pub historical_data: Vec<(chrono::DateTime<chrono::Utc>, f64)>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfilingContext {
    pub context_id: String,
    pub profiling_depth: ProfilingDepth,
    pub environment: HashMap<String, String>,
    pub start_time: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct ProfilingInput {
    pub test_name: String,
    pub input_data: Vec<u8>,
    pub configuration: HashMap<String, String>,
    pub profiling_options: ProfilingOptions,
}

#[derive(Debug, Clone, Default)]
pub struct ProfilingOptions {
    pub profiling_depth: ProfilingDepth,
    pub sample_rate: f64,
    pub include_memory_profiling: bool,
    pub include_cpu_profiling: bool,
    pub include_io_profiling: bool,
}

#[derive(Debug, Clone)]
pub struct ProfilingOutput {
    pub profiling_results: ProfilingResults,
    pub execution_time: std::time::Duration,
    pub resource_usage: HashMap<String, f64>,
    pub output_data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ProfilingResults {
    pub performance_metrics: PerformanceMetrics,
    pub profiling_statistics: ProfilingStatistics,
    pub bottlenecks: Vec<String>,
    pub optimization_opportunities: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ProfilingStatistics {
    pub total_samples: usize,
    pub sampling_duration: std::time::Duration,
    pub average_sample_interval: std::time::Duration,
    pub data_quality_score: f64,
    pub active_sessions: usize,
    pub data_points_processed: usize,
    pub anomalies_detected: usize,
    pub insights_generated: usize,
    pub buffer_utilization: f32,
    pub processing_rate: f64,
    pub last_updated: std::time::Instant,
}

#[derive(Debug, Clone)]
pub struct ProfilingStatus {
    pub is_active: bool,
    pub progress_percentage: f64,
    pub current_stage: String,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

impl ProfilingStatus {
    /// Create an active profiling status
    pub fn active() -> Self {
        Self {
            is_active: true,
            progress_percentage: 0.0,
            current_stage: String::from("active"),
            started_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScalingBehavior {
    pub scaling_type: String,
    pub scaling_efficiency: f64,
    pub optimal_scale: usize,
    pub scaling_limits: (usize, usize),
}

#[derive(Debug, Clone)]
pub struct ScalingCharacteristics {
    pub horizontal_scaling: ScalingBehavior,
    pub vertical_scaling: ScalingBehavior,
    pub scaling_overhead: f64,
    pub recommended_scaling_strategy: String,
}

#[derive(Debug, Clone)]
pub struct ThroughputAnalysis {
    pub average_throughput: f64,
    pub peak_throughput: f64,
    pub throughput_variance: f64,
    pub throughput_trend: String,
}

#[derive(Debug, Clone)]
pub struct ThroughputProcessorConfig {
    pub processing_enabled: bool,
    pub aggregation_window: std::time::Duration,
    pub smoothing_factor: f64,
}

/// Profiling stage trait for pipeline stages
pub trait ProfilingStage: std::fmt::Debug + Send + Sync {
    /// Execute the profiling stage
    fn execute(&self, input: ProfilingInput) -> TestCharacterizationResult<ProfilingOutput>;

    /// Get stage name
    fn name(&self) -> &str;

    /// Get stage dependencies
    fn dependencies(&self) -> Vec<String>;

    /// Get estimated execution time
    fn estimated_duration(&self, input: &ProfilingInput) -> Duration;

    /// Validate stage input
    fn validate_input(&self, input: &ProfilingInput) -> TestCharacterizationResult<()>;
}

// Trait implementations

#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    pub required_throughput: f64,
    pub max_latency: std::time::Duration,
    pub min_availability: f64,
    pub resource_constraints: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct MonitoringRequirements {
    pub metrics_to_monitor: Vec<String>,
    pub monitoring_interval: std::time::Duration,
    pub alert_thresholds: HashMap<String, f64>,
    pub retention_period: std::time::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAlgorithmResult {
    pub algorithm: String,
    pub patterns: Vec<String>,
    #[serde(skip)]
    pub detection_duration: std::time::Duration,
    pub confidence: f64,
}

#[async_trait]
pub trait ProfilingStrategy: std::fmt::Debug + Send + Sync {
    fn profile(&self) -> String;
    fn name(&self) -> &str;
    async fn activate(&self) -> Result<()>;
    async fn deactivate(&self) -> Result<()>;
}

// Implementations

impl PerformanceImpactAnalysis {
    pub fn new(degradation: f64, impact_areas: Vec<String>) -> Self {
        Self {
            degradation,
            impact_areas,
        }
    }
}

impl Default for PerformanceImpactAnalysis {
    fn default() -> Self {
        Self {
            degradation: 0.0,
            impact_areas: Vec::new(),
        }
    }
}

impl PerformanceAnalysisPipeline {
    /// Create a new PerformanceAnalysisPipeline with default settings
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            enabled: true,
        }
    }
}

impl Default for PerformanceAnalysisPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl super::core::StreamingPipeline for PerformanceAnalysisPipeline {
    fn process(
        &self,
        _sample: super::core::ProfileSample,
    ) -> TestCharacterizationResult<super::core::StreamingResult> {
        // Process the sample through the pipeline
        Ok(super::core::StreamingResult {
            timestamp: Instant::now(),
            data: super::analysis::AnalysisResultData::Custom(
                "performance_analysis".to_string(),
                serde_json::json!({"processed": true}),
            ),
            anomalies: Vec::new(),
            quality: Default::default(),
            trend: Default::default(),
            recommendations: Vec::new(),
            confidence: 1.0,
            analysis_duration: Duration::from_millis(1),
            data_points_analyzed: 1,
            alert_conditions: Vec::new(),
        })
    }

    fn name(&self) -> &str {
        "PerformanceAnalysisPipeline"
    }

    fn latency(&self) -> Duration {
        Duration::from_millis(10)
    }

    fn throughput_capacity(&self) -> f64 {
        1000.0 // samples per second
    }

    fn flush(&self) -> TestCharacterizationResult<Vec<super::core::StreamingResult>> {
        Ok(Vec::new())
    }
}

impl PerformanceInsightEngine {
    /// Create a new PerformanceInsightEngine with default settings
    pub fn new() -> Self {
        Self {
            insights_generated: 0,
            quality_score: 0.0,
        }
    }
}

impl Default for PerformanceInsightEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl InsightEngine for PerformanceInsightEngine {
    fn generate(&self) -> String {
        format!(
            "Performance Insight Engine (insights_generated={}, quality_score={:.2})",
            self.insights_generated, self.quality_score
        )
    }

    fn generate_test_insights(&self, test_id: &str) -> TestCharacterizationResult<Vec<String>> {
        // Placeholder implementation - in production, this would analyze test-specific performance data
        Ok(vec![
            format!(
                "Test '{}' performance analysis: {} insights generated with quality score {:.2}",
                test_id, self.insights_generated, self.quality_score
            ),
            format!(
                "Performance metrics suggest {} optimization opportunities",
                if self.quality_score > 0.7 { "few" } else { "several" }
            ),
        ])
    }

    fn generate_insights(&self) -> TestCharacterizationResult<Vec<String>> {
        // Placeholder implementation - in production, this would generate comprehensive insights
        Ok(vec![
            format!("Total insights generated: {}", self.insights_generated),
            format!("Overall insight quality: {:.2}", self.quality_score),
            "Performance analysis engine active".to_string(),
        ])
    }
}

impl Default for PerformanceConstraints {
    fn default() -> Self {
        Self {
            max_execution_time: None,
            max_memory_usage: None,
            max_cpu_utilization: None,
            max_io_rate: None,
            max_network_bandwidth: None,
            quality_thresholds: HashMap::new(),
            benchmarks: HashMap::new(),
            sla_requirements: Vec::new(),
        }
    }
}

// Trait implementations for E0277 fixes

impl super::patterns::ThreadAnalysisAlgorithm for PerformanceImpactAnalysis {
    fn analyze(&self) -> String {
        let score = 1.0 - self.degradation.min(1.0);
        format!(
            "Performance degradation: {:.2}%, score: {:.2}",
            self.degradation * 100.0,
            score
        )
    }

    fn name(&self) -> &str {
        "PerformanceImpactAnalysis"
    }
}

impl super::reporting::ReportGenerator for PerformanceReportGenerator {
    fn generate(&self) -> String {
        format!(
            "Performance Report (interval={}ms, charts={})",
            self.report_interval_ms, self.include_charts
        )
    }
}
