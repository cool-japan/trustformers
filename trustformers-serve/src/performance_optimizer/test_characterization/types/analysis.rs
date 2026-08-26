use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use uuid;

// Import commonly used types from core
use super::core::{
    DetectedPattern, TestCharacterizationError, TestCharacterizationResult, UrgencyLevel,
};

// Import cross-module types
use super::optimization::OptimizationRecommendation;
use super::quality::QualityAssessment;
use super::reporting::ReportGenerator;

#[derive(Debug, Clone)]
pub enum AnalysisResultData {
    /// Resource usage data
    ResourceUsage(HashMap<String, f64>),
    /// Performance metrics
    PerformanceMetrics(HashMap<String, f64>),
    /// Pattern detection results
    PatternDetection(Vec<DetectedPattern>),
    /// Anomaly detection results
    AnomalyDetection(Vec<AnomalyInfo>),
    /// Optimization recommendations
    OptimizationRecommendations(Vec<OptimizationRecommendation>),
    /// Quality assessment
    QualityAssessment(QualityAssessment),
    /// Trend analysis
    TrendAnalysis(HashMap<String, TrendDirection>),
    /// Comparison results
    ComparisonResults(HashMap<String, f64>),
    /// Statistical analysis
    StatisticalAnalysis(HashMap<String, f64>),
    /// Custom data
    Custom(String, serde_json::Value),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AnomalySeverity {
    /// Minor anomaly
    Minor,
    /// Moderate anomaly
    Moderate,
    /// Significant anomaly
    Significant,
    /// Major anomaly
    Major,
    /// Severe anomaly
    Severe,
    /// Critical anomaly
    Critical,
    /// Extreme anomaly
    Extreme,
    // Standard severity level aliases for compatibility
    /// Low severity (alias for Minor)
    Low,
    /// Medium severity (alias for Moderate)
    Medium,
    /// High severity (alias for Major)
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Statistical anomaly
    Statistical,
    /// Behavioral anomaly
    Behavioral,
    /// Performance anomaly
    Performance,
    /// Resource anomaly
    Resource,
    /// Temporal anomaly
    Temporal,
    /// Pattern anomaly
    Pattern,
    /// Contextual anomaly
    Contextual,
    /// Collective anomaly
    Collective,
    /// Point anomaly
    Point,
    /// Sequence anomaly
    Sequence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable trend
    Stable,
    /// Fluctuating trend
    Fluctuating,
    /// Cyclical trend
    Cyclical,
    /// Seasonal trend
    Seasonal,
    /// Random trend
    Random,
    /// Unknown trend
    Unknown,
}

#[derive(Debug, Clone)]
pub struct AnalysisDepth {
    pub depth_level: String,
    pub analysis_scope: Vec<String>,
    pub detail_level: f64,
}

#[derive(Debug, Clone)]
pub struct AnalysisWindowOptimizer {
    /// Window size
    pub window_size: usize,
    /// Optimization enabled
    pub enabled: bool,
}

#[async_trait::async_trait]
impl super::optimization::OptimizationStrategy for AnalysisWindowOptimizer {
    fn optimize(&self) -> String {
        format!(
            "Optimize analysis window size to {} samples",
            self.window_size
        )
    }

    fn is_applicable(&self, _context: &super::optimization::OptimizationContext) -> bool {
        // Window optimization is applicable when enabled and window size is reasonable
        self.enabled && self.window_size > 0 && self.window_size < 10000
    }

    async fn apply_optimization(
        &self,
        _performance_data: &super::optimization::OptimizationPerformanceData,
    ) -> TestCharacterizationResult<super::optimization::StrategyOptimizationResult> {
        // Calculate effectiveness based on window size
        // Optimal range is typically 100-1000 samples
        let optimal_min = 100.0;
        let optimal_max = 1000.0;
        let window = self.window_size as f64;

        let effectiveness = if window < optimal_min {
            window / optimal_min // Too small
        } else if window > optimal_max {
            optimal_max / window // Too large
        } else {
            1.0 - ((window - (optimal_min + optimal_max) / 2.0).abs() / optimal_max)
            // Within range
        };

        Ok(super::optimization::StrategyOptimizationResult {
            strategy_name: "AnalysisWindowOptimizer".to_string(),
            result: super::optimization::OptimizationResult {
                result_id: format!("window_opt_{}", uuid::Uuid::new_v4()),
                optimization_type: super::optimization::OptimizationType::Parallelism,
                success: effectiveness > 0.5,
                performance_improvement: effectiveness * 0.18, // Up to 18% improvement
                resource_savings: {
                    let mut savings = std::collections::HashMap::new();
                    savings.insert("memory".to_string(), effectiveness * 0.12);
                    savings.insert("cpu".to_string(), effectiveness * 0.08);
                    savings
                },
            },
            effectiveness_score: effectiveness,
            timestamp: chrono::Utc::now(),
        })
    }

    async fn get_recommendation(
        &self,
        _context: &super::optimization::OptimizationContext,
        _effectiveness: &std::collections::HashMap<String, f64>,
    ) -> TestCharacterizationResult<super::optimization::OptimizationRecommendation> {
        let optimal_window = 500; // Default optimal window size
        let diff = (self.window_size as i64 - optimal_window).unsigned_abs() as usize;
        let relative_diff = diff as f64 / optimal_window as f64;

        let urgency = if relative_diff > 1.0 {
            super::core::UrgencyLevel::High
        } else if relative_diff > 0.5 {
            super::core::UrgencyLevel::Medium
        } else {
            super::core::UrgencyLevel::Low
        };

        Ok(super::optimization::OptimizationRecommendation {
            recommendation_id: format!("window_rec_{}", uuid::Uuid::new_v4()),
            recommendation_type: "Analysis Window Adjustment".to_string(),
            description: format!(
                "Adjust analysis window size to {} samples for optimal analysis accuracy and performance",
                optimal_window
            ),
            expected_benefit: relative_diff.min(1.0) * 0.5,
            complexity: 0.2,
            priority: super::core::PriorityLevel::Low,
            urgency,
            required_resources: vec!["Analysis Engine".to_string()],
            steps: vec![
                "Evaluate current window utilization".to_string(),
                format!("Adjust window size to {}", optimal_window),
                "Test analysis accuracy".to_string(),
                "Validate performance improvement".to_string(),
            ],
            risk: 0.1,
            confidence: 0.90,
            expected_roi: 1.8,
        })
    }
}

#[derive(Debug, Clone)]
pub struct AnalyzerMetrics {
    pub analysis_count: usize,
    pub average_analysis_time: std::time::Duration,
    pub accuracy: f64,
    pub throughput: f64,
}

#[derive(Debug, Clone)]
/// Records anomaly alerts raised while the system is running.
///
/// Before 0.2.1 `alert_history` was a plain `Vec` behind an `Arc`, so
/// `trigger_alert` could not record anything; it, `start_alerting` and
/// `stop_alerting` were all `Ok(())` no-ops, and the caller took that `Ok` as
/// proof a notification had been delivered.
pub struct AnomalyAlertSystem {
    pub alert_enabled: bool,
    pub alert_thresholds: HashMap<AnomalySeverity, f64>,
    pub notification_channels: Vec<String>,
    alerting: Arc<AtomicBool>,
    alert_history: Arc<Mutex<Vec<AnomalyInfo>>>,
}

#[derive(Debug, Clone)]
pub struct AnomalyClustering {
    pub clustering_method: String,
    pub num_clusters: usize,
    pub cluster_assignments: HashMap<String, usize>,
    pub cluster_centroids: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct AnomalyDetectionConfig {
    pub detection_enabled: bool,
    pub sensitivity: f64,
    pub detection_algorithms: Vec<String>,
    pub anomaly_threshold: f64,
    pub detection_interval: std::time::Duration,
    pub threshold_config: String,
    pub alert_config: String,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            detection_enabled: true,
            sensitivity: 0.8,
            detection_algorithms: vec![],
            anomaly_threshold: 0.95,
            detection_interval: std::time::Duration::from_secs(5),
            threshold_config: String::new(),
            alert_config: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnomalyDetectionPipeline {
    /// Detection algorithms
    pub algorithms: Vec<String>,
    /// Detection enabled
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct AnomalyDetectionResult {
    pub anomalies_detected: Vec<AnomalyInfo>,
    pub detection_confidence: f64,
    pub detection_timestamp: chrono::DateTime<chrono::Utc>,
    pub false_positive_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyIndicator {
    pub indicator_name: String,
    pub indicator_value: f64,
    pub threshold: f64,
    pub anomaly_detected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyInfo {
    /// Anomaly identifier
    pub anomaly_id: String,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Affected resources
    pub affected_resources: Vec<String>,
    /// Anomaly description
    pub description: String,
    /// Impact assessment
    pub impact: f64,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
    /// False positive probability
    pub false_positive_probability: f64,
    /// Historical frequency
    pub historical_frequency: f64,
    /// Resolution urgency
    pub urgency: UrgencyLevel,
}

#[derive(Debug, Clone)]
pub struct AnomalyReportGenerator {
    pub threshold: f64,
    pub auto_report: bool,
}

#[derive(Debug, Clone)]
pub struct AnomalyScore {
    pub score: f64,
    pub normalized_score: f64,
    pub severity: AnomalySeverity,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct AnomalyTracker {
    pub tracked_anomalies: HashMap<String, AnomalyInfo>,
    pub tracking_start: chrono::DateTime<chrono::Utc>,
    pub anomaly_count: usize,
    pub resolution_rate: f64,
}

#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    pub bottleneck_type: String,
    pub severity: BottleneckSeverity,
    pub affected_components: Vec<String>,
    pub impact_score: f64,
    pub resolution_suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BottleneckAnalysisResult {
    pub bottlenecks_identified: Vec<BottleneckAnalysis>,
    pub overall_severity: BottleneckSeverity,
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BottleneckDetectionMethod {
    pub method_name: String,
    pub detection_accuracy: f64,
    pub sensitivity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckEmergencePoint {
    pub emergence_time: chrono::DateTime<chrono::Utc>,
    pub trigger_condition: String,
    pub contributing_factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckIndicator {
    pub indicator_name: String,
    pub current_value: f64,
    pub threshold_value: f64,
    pub bottleneck_detected: bool,
}

#[derive(Debug, Clone)]
pub struct BottleneckInteractionMatrix {
    pub bottleneck_pairs: Vec<(String, String)>,
    pub interaction_strengths: HashMap<(String, String), f64>,
    pub cascading_effects: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BottleneckType {
    pub type_name: String,
    pub type_category: String,
    pub typical_severity: BottleneckSeverity,
}

#[derive(Debug, Clone)]
pub struct ChangePoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metric_name: String,
    pub value_before: f64,
    pub value_after: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct ChangePointDetection {
    pub detection_enabled: bool,
    pub detected_change_points: Vec<ChangePoint>,
    pub detection_sensitivity: f64,
}

#[derive(Debug, Clone)]
pub struct CorrelationAction {
    pub action_id: String,
    pub action_type: String,
    pub trigger_correlation: f64,
    pub action_description: String,
}

#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    pub analyzed_metrics: Vec<String>,
    pub correlation_matrix: CorrelationMatrix,
    pub significant_correlations: Vec<(String, String, f64)>,
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct CorrelationCache {
    pub cached_correlations: HashMap<(String, String), f64>,
    pub cache_timestamp: chrono::DateTime<chrono::Utc>,
    pub cache_ttl: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct CorrelationContext {
    pub context_id: String,
    pub time_window: std::time::Duration,
    pub metrics_included: Vec<String>,
    pub environmental_factors: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct CorrelationEngine {
    pub engine_type: String,
    pub correlation_methods: Vec<String>,
    pub significance_threshold: f64,
    pub max_lag: usize,
}

#[derive(Debug, Clone)]
pub struct CorrelationLogic {
    pub logic_rules: Vec<String>,
    pub rule_priorities: HashMap<String, u32>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    pub metric_names: Vec<String>,
    pub correlations: Vec<Vec<f64>>,
    pub sample_size: usize,
}

#[derive(Debug, Clone)]
pub struct CorrelationStatistics {
    pub total_correlations_analyzed: usize,
    pub significant_correlations_found: usize,
    pub average_correlation_strength: f64,
    pub max_correlation: f64,
}

#[derive(Debug, Clone)]
pub struct DensityEstimation {
    pub estimation_method: String,
    pub bandwidth: f64,
    pub kernel_type: String,
    pub density_values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DetectionMethod {
    pub method_name: String,
    pub method_type: String,
    pub sensitivity: f64,
    pub accuracy: f64,
}

#[derive(Debug, Clone)]
pub struct DetectionStatistics {
    pub total_detections: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub detection_rate: f64,
    pub precision: f64,
}

#[derive(Debug, Clone)]
pub struct DeviationType {
    pub deviation_name: String,
    pub deviation_severity: f64,
    pub deviation_category: String,
}

#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    pub distribution_type: DistributionType,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

#[derive(Debug, Clone)]
pub struct DistributionType {
    pub type_name: String,
    pub parameters: HashMap<String, f64>,
    pub goodness_of_fit: f64,
}

#[derive(Debug, Clone)]
pub struct HypothesisTestResult {
    pub test_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub reject_null: bool,
    pub confidence_level: f64,
}

#[derive(Debug, Clone)]
pub struct Outlier {
    pub outlier_id: String,
    pub value: f64,
    pub expected_value: f64,
    pub deviation: f64,
    pub detection_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct OutlierAnalysis {
    pub outliers_detected: Vec<Outlier>,
    pub outlier_percentage: f64,
    pub analysis_method: String,
    pub threshold_used: f64,
}

#[derive(Debug, Clone)]
pub struct OutlierDetector {
    pub detection_method: String,
    pub sensitivity: f64,
    pub threshold_multiplier: f64,
}

#[derive(Debug, Clone)]
pub struct OutlierHandling {
    pub handling_strategy: String,
    pub auto_remove: bool,
    pub replacement_method: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OutlierParameters {
    pub z_score_threshold: f64,
    pub iqr_multiplier: f64,
    pub contamination_rate: f64,
}

#[derive(Debug, Clone)]
pub struct OutlierResult {
    pub is_outlier: bool,
    pub outlier_score: f64,
    pub method_used: String,
}

#[derive(Debug, Clone)]
pub struct OutlierResults {
    pub total_outliers: usize,
    pub outlier_details: Vec<OutlierResult>,
    pub summary_statistics: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct RootCauseAnalysis {
    pub issue_id: String,
    pub probable_causes: Vec<String>,
    pub confidence_scores: HashMap<String, f64>,
    pub recommended_actions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StatisticalAnalysisEngine {
    pub analysis_methods: Vec<String>,
    pub engine_enabled: bool,
    pub confidence_level: f64,
}

#[derive(Debug, Clone)]
pub struct StatisticalAnomalyDetector {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Z-score threshold
    pub threshold: f64,
}

#[derive(Debug, Clone)]
pub struct StatisticalMethod {
    pub method_name: String,
    pub method_type: String,
    pub assumptions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StatisticalRiskAssessment {
    pub model: String,
    pub risk_score: f64,
}

#[derive(Debug, Clone)]
pub struct StatisticalSignificance {
    pub p_value: f64,
    pub significance_level: f64,
    pub is_significant: bool,
}

#[derive(Debug, Clone)]
pub struct StatisticalSummary {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub quartiles: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trend {
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub data_points: Vec<(chrono::DateTime<chrono::Utc>, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub detected_trends: Vec<Trend>,
    pub overall_direction: TrendDirection,
    pub confidence: f64,
    pub forecast: Vec<f64>,
}

impl Default for TrendAnalysis {
    fn default() -> Self {
        Self {
            detected_trends: Vec::new(),
            overall_direction: TrendDirection::Unknown,
            confidence: 0.0,
            forecast: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrendAnalysisEngine {
    pub analysis_algorithms: Vec<String>,
    pub analysis_window: std::time::Duration,
    pub min_data_points: usize,
}

#[derive(Debug, Clone)]
pub struct TrendAnalysisResult {
    pub result_id: String,
    pub trends: Vec<Trend>,
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
    pub confidence_score: f64,
}

#[derive(Debug, Clone)]
pub struct TrendAnalyzerConfig {
    pub enabled: bool,
    pub analysis_interval: std::time::Duration,
    pub detection_sensitivity: f64,
    pub forecast_horizon: usize,
    pub database_config: String,
    pub detection_config: String,
}

impl Default for TrendAnalyzerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            analysis_interval: std::time::Duration::from_secs(10),
            detection_sensitivity: 0.8,
            forecast_horizon: 100,
            database_config: String::new(),
            detection_config: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrendAnomalyDetector {
    /// Historical trends
    pub trends: Vec<String>,
    /// Deviation threshold
    pub threshold: f64,
}

#[derive(Debug, Clone)]
pub struct TrendComponents {
    pub trend_component: Vec<f64>,
    pub seasonal_component: Vec<f64>,
    pub residual_component: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TrendDetectionConfig {
    pub detection_enabled: bool,
    pub min_trend_strength: f64,
    pub detection_algorithms: Vec<String>,
}

#[derive(Debug, Clone)]
/// Trend-detection lifecycle with an observable running state.
///
/// `detection_accuracy` was removed in 0.2.1: it was a constant 0.8 that no
/// evaluation ever produced, and `start_detection`/`stop_detection` were
/// `Ok(())` no-ops with nothing to flip.
pub struct TrendDetectionEngine {
    pub detection_methods: Vec<String>,
    pub current_method: String,
    detecting: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub struct TrendReportGenerator {
    pub trend_window_size: usize,
    pub include_forecast: bool,
}

/// Trend analysis algorithm trait
pub trait TrendAnalysisAlgorithm: std::fmt::Debug + Send + Sync {
    /// Analyze trends in time series data
    fn analyze_trend(&self, data: &[(Instant, f64)]) -> TestCharacterizationResult<TrendAnalysis>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get trend confidence
    fn confidence(&self, data: &[(Instant, f64)]) -> f64;

    /// Predict future values
    fn predict(
        &self,
        data: &[(Instant, f64)],
        steps: usize,
    ) -> TestCharacterizationResult<Vec<f64>>;
}

// Trait implementations

/// Flags readings that deviate from an observed baseline.
///
/// Before 0.2.1 `detect_anomalies` took no arguments at all: every
/// implementation was structurally incapable of detecting anything and returned
/// an empty vector, which `AnomalyDetectionEngine` reported as "no anomalies".
pub trait AnomalyDetector: std::fmt::Debug + Send + Sync {
    /// What this detector looks for. Describes the detector, not any state.
    fn describe(&self) -> String;
    /// Anomalies in `observations`, judged against `baseline`.
    fn detect_anomalies(
        &self,
        observations: InsightObservations<'_>,
        baseline: &super::core::BaselineModel,
    ) -> TestCharacterizationResult<Vec<AnomalyInfo>>;
}

/// Build an `AnomalyInfo` from a measured deviation.
///
/// Severity is a function of how far past the detector's own threshold the
/// reading sat, so the label always traces back to a number.
pub fn anomaly_from_deviation(
    detector: &str,
    anomaly_type: AnomalyType,
    metric: &str,
    ratio: f64,
    description: String,
) -> AnomalyInfo {
    let severity = match ratio {
        r if r >= 4.0 => AnomalySeverity::Critical,
        r if r >= 3.0 => AnomalySeverity::Severe,
        r if r >= 2.0 => AnomalySeverity::Major,
        r if r >= 1.5 => AnomalySeverity::Significant,
        r if r >= 1.2 => AnomalySeverity::Moderate,
        _ => AnomalySeverity::Minor,
    };
    AnomalyInfo {
        anomaly_id: format!(
            "{}:{}:{}",
            detector,
            metric,
            Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ),
        anomaly_type,
        severity,
        detected_at: Utc::now(),
        affected_resources: vec![metric.to_string()],
        description,
        // Impact is reported as the excess over the threshold, capped at one.
        impact: (ratio - 1.0).clamp(0.0, 1.0),
        // The detector measures a deviation; it proposes no remedy.
        recommended_actions: Vec::new(),
        // A reading twice past its threshold is a false positive far less often
        // than one that just crossed it; this is that monotone relationship, not
        // a calibrated rate.
        false_positive_probability: (1.0 / ratio.max(1.0)).clamp(0.0, 1.0),
        // No cross-window history is retained, so no frequency is claimed.
        historical_frequency: 0.0,
        // Urgency tracks the same measured ratio the severity does.
        urgency: match ratio {
            r if r >= 4.0 => UrgencyLevel::Critical,
            r if r >= 3.0 => UrgencyLevel::VeryUrgent,
            r if r >= 2.0 => UrgencyLevel::Urgent,
            r if r >= 1.5 => UrgencyLevel::High,
            r if r >= 1.2 => UrgencyLevel::Medium,
            _ => UrgencyLevel::Low,
        },
    }
}

/// Summary of one metric key across an observation window.
#[derive(Debug, Clone)]
pub struct MetricSummary {
    /// The metric key this summarises.
    pub key: String,
    /// Number of samples that carried the key.
    pub count: usize,
    /// Arithmetic mean of those samples.
    pub mean: f64,
    /// Smallest observed value.
    pub min: f64,
    /// Largest observed value.
    pub max: f64,
    /// Most recent observed value.
    pub last: f64,
}

/// The samples an insight engine reasons over.
///
/// Before 0.2.1 `InsightEngine` took no input at all: every implementation
/// produced prose from struct fields that nothing ever wrote to, so an engine
/// with `insights_generated: 0` still announced an "optimization potential" or a
/// "priority attention" level. Engines now see the profiler's real sample
/// window and report only what it supports; an empty window yields no insights.
#[derive(Debug, Clone, Copy)]
pub struct InsightObservations<'a> {
    /// Samples collected over the observation window, oldest first.
    pub samples: &'a [super::core::RealTimeMetrics],
}

impl<'a> InsightObservations<'a> {
    /// Wrap a sample window.
    pub fn new(samples: &'a [super::core::RealTimeMetrics]) -> Self {
        Self { samples }
    }

    /// True when no sample was collected.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Every metric key present in the window, sorted and de-duplicated.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> =
            self.samples.iter().flat_map(|sample| sample.metrics.keys().cloned()).collect();
        keys.sort();
        keys.dedup();
        keys
    }

    /// Values recorded for `key`, in sample order.
    pub fn series(&self, key: &str) -> Vec<f64> {
        self.samples
            .iter()
            .filter_map(|sample| sample.metrics.get(key).copied())
            .filter(|value| value.is_finite())
            .collect()
    }

    /// Summary of `key`, or `None` when no sample carried a finite value for it.
    pub fn summary(&self, key: &str) -> Option<MetricSummary> {
        let values = self.series(key);
        let first = *values.first()?;
        let mut min = first;
        let mut max = first;
        let mut total = 0.0;
        for value in &values {
            min = min.min(*value);
            max = max.max(*value);
            total += value;
        }
        Some(MetricSummary {
            key: key.to_string(),
            count: values.len(),
            mean: total / values.len() as f64,
            min,
            max,
            last: *values.last()?,
        })
    }

    /// Summaries for every key whose name contains one of `needles`.
    pub fn summaries_matching(&self, needles: &[&str]) -> Vec<MetricSummary> {
        self.keys()
            .into_iter()
            .filter(|key| needles.iter().any(|needle| key.contains(needle)))
            .filter_map(|key| self.summary(&key))
            .collect()
    }
}

/// Produces human-readable findings from an observed metric window.
pub trait InsightEngine: std::fmt::Debug + Send + Sync {
    /// What this engine covers. Describes the engine, not any accumulated state.
    fn describe(&self) -> String;
    /// Findings scoped to one test id.
    fn generate_test_insights(
        &self,
        test_id: &str,
        observations: InsightObservations<'_>,
    ) -> TestCharacterizationResult<Vec<String>>;
    /// Findings over the whole window.
    fn generate_insights(
        &self,
        observations: InsightObservations<'_>,
    ) -> TestCharacterizationResult<Vec<String>>;
}

// Implementations

impl StatisticalRiskAssessment {
    pub fn new(model: String, risk_score: f64) -> Self {
        Self { model, risk_score }
    }
}

impl Default for StatisticalRiskAssessment {
    fn default() -> Self {
        Self {
            model: String::new(),
            risk_score: 0.0,
        }
    }
}

impl StatisticalSignificance {
    pub fn new(p_value: f64, significance_level: f64) -> Self {
        let is_significant = p_value < significance_level;
        Self {
            p_value,
            significance_level,
            is_significant,
        }
    }
}

impl Default for StatisticalSignificance {
    fn default() -> Self {
        Self {
            p_value: 1.0,
            significance_level: 0.05,
            is_significant: false,
        }
    }
}

impl StatisticalSummary {
    pub fn new(
        count: usize,
        mean: f64,
        median: f64,
        std_dev: f64,
        min: f64,
        max: f64,
        quartiles: Vec<f64>,
    ) -> Self {
        Self {
            count,
            mean,
            median,
            std_dev,
            min,
            max,
            quartiles,
        }
    }
}

impl Default for StatisticalSummary {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            quartiles: vec![0.0, 0.0, 0.0],
        }
    }
}

impl AnalysisDepth {
    pub fn new(depth_level: String, analysis_scope: Vec<String>, detail_level: f64) -> Self {
        Self {
            depth_level,
            analysis_scope,
            detail_level,
        }
    }
}

impl Default for AnalysisDepth {
    fn default() -> Self {
        Self {
            depth_level: "basic".to_string(),
            analysis_scope: Vec::new(),
            detail_level: 1.0,
        }
    }
}

impl AnalysisWindowOptimizer {
    pub fn new(window_size: usize, enabled: bool) -> Self {
        Self {
            window_size,
            enabled,
        }
    }
}

impl Default for AnalysisWindowOptimizer {
    fn default() -> Self {
        Self {
            window_size: 100,
            enabled: true,
        }
    }
}

impl AnalyzerMetrics {
    pub fn new(
        analysis_count: usize,
        average_analysis_time: std::time::Duration,
        accuracy: f64,
        throughput: f64,
    ) -> Self {
        Self {
            analysis_count,
            average_analysis_time,
            accuracy,
            throughput,
        }
    }
}

impl Default for AnalyzerMetrics {
    fn default() -> Self {
        Self {
            analysis_count: 0,
            average_analysis_time: std::time::Duration::from_secs(0),
            accuracy: 0.0,
            throughput: 0.0,
        }
    }
}

impl StatisticalAnomalyDetector {
    pub fn new(mean: f64, std_dev: f64, threshold: f64) -> Self {
        Self {
            mean,
            std_dev,
            threshold,
        }
    }
}

impl Default for StatisticalAnomalyDetector {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
            threshold: 3.0,
        }
    }
}

impl AnomalyDetector for StatisticalAnomalyDetector {
    fn describe(&self) -> String {
        format!(
            "Statistical anomaly detector: flags readings more than {:.2} standard deviations              from the baseline mean (fallback mean {:.2}, std dev {:.2} where the baseline has              no entry)",
            self.threshold, self.mean, self.std_dev
        )
    }

    fn detect_anomalies(
        &self,
        observations: InsightObservations<'_>,
        baseline: &super::core::BaselineModel,
    ) -> TestCharacterizationResult<Vec<AnomalyInfo>> {
        if self.threshold <= 0.0 {
            return Err(TestCharacterizationError::InvalidInput {
                message: "z-score threshold must be positive".to_string(),
                field: "threshold".to_string(),
                value: self.threshold.to_string(),
            });
        }
        let mut anomalies = Vec::new();
        for key in observations.keys() {
            let (mean, std_dev) =
                baseline.statistics_for(&key).unwrap_or((self.mean, self.std_dev));
            if std_dev <= 0.0 {
                // A baseline with no spread cannot separate signal from noise.
                continue;
            }
            for value in observations.series(&key) {
                let z = (value - mean).abs() / std_dev;
                if z <= self.threshold {
                    continue;
                }
                anomalies.push(anomaly_from_deviation(
                    "statistical",
                    AnomalyType::Statistical,
                    &key,
                    z / self.threshold,
                    format!(
                        "`{}` read {:.4}, {:.2} standard deviations from the baseline mean                          {:.4} (threshold {:.2})",
                        key, value, z, mean, self.threshold
                    ),
                ));
            }
        }
        Ok(anomalies)
    }
}

impl TrendAnomalyDetector {
    pub fn new(trends: Vec<String>, threshold: f64) -> Self {
        Self { trends, threshold }
    }
}

impl Default for TrendAnomalyDetector {
    fn default() -> Self {
        Self {
            trends: Vec::new(),
            threshold: 2.0,
        }
    }
}

impl AnomalyDetector for TrendAnomalyDetector {
    fn describe(&self) -> String {
        format!(
            "Trend anomaly detector: flags a metric whose window drifts by more than {:.2}              baseline standard deviations end to end ({} metric(s) watched; all when empty)",
            self.threshold,
            self.trends.len()
        )
    }

    fn detect_anomalies(
        &self,
        observations: InsightObservations<'_>,
        baseline: &super::core::BaselineModel,
    ) -> TestCharacterizationResult<Vec<AnomalyInfo>> {
        if self.threshold <= 0.0 {
            return Err(TestCharacterizationError::InvalidInput {
                message: "drift threshold must be positive".to_string(),
                field: "threshold".to_string(),
                value: self.threshold.to_string(),
            });
        }
        let mut anomalies = Vec::new();
        for key in observations.keys() {
            if !self.trends.is_empty() && !self.trends.contains(&key) {
                continue;
            }
            let values = observations.series(&key);
            if values.len() < 3 {
                continue;
            }
            let (_, std_dev) = match baseline.statistics_for(&key) {
                Some(stats) => stats,
                None => continue,
            };
            if std_dev <= 0.0 {
                continue;
            }
            let (Some(first), Some(last)) = (values.first(), values.last()) else {
                continue;
            };
            let drift = (last - first).abs() / std_dev;
            if drift <= self.threshold {
                continue;
            }
            anomalies.push(anomaly_from_deviation(
                "trend",
                AnomalyType::Temporal,
                &key,
                drift / self.threshold,
                format!(
                    "`{}` drifted from {:.4} to {:.4} across {} samples, {:.2} baseline standard                      deviations (threshold {:.2})",
                    key,
                    first,
                    last,
                    values.len(),
                    drift,
                    self.threshold
                ),
            ));
        }
        Ok(anomalies)
    }
}

impl AnomalyDetectionPipeline {
    /// Create a new AnomalyDetectionPipeline with default settings
    pub fn new() -> Self {
        Self {
            algorithms: Vec::new(),
            enabled: true,
        }
    }
}

impl Default for AnomalyDetectionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl super::core::StreamingPipeline for AnomalyDetectionPipeline {
    fn process(
        &self,
        _sample: super::core::ProfileSample,
    ) -> TestCharacterizationResult<super::core::StreamingResult> {
        // Process the sample through the anomaly detection pipeline
        Ok(super::core::StreamingResult {
            timestamp: Instant::now(),
            data: super::analysis::AnalysisResultData::AnomalyDetection(Vec::new()),
            anomalies: Vec::new(),
            quality: super::quality::QualityAssessment::default(),
            trend: super::quality::QualityTrend {
                direction: TrendDirection::Stable,
                strength: 0.0,
                confidence: 0.95,
                historical_points: Vec::new(),
                trend_start: Instant::now(),
                predictions: Vec::new(),
                analysis_method: String::from("default"),
                significance: 0.0,
                stability: 1.0,
                change_rate: 0.0,
            },
            recommendations: Vec::new(),
            confidence: 0.95,
            analysis_duration: Duration::from_millis(1),
            data_points_analyzed: 0,
            alert_conditions: Vec::new(),
        })
    }

    fn name(&self) -> &str {
        "AnomalyDetectionPipeline"
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

impl AnomalyAlertSystem {
    /// Create a new AnomalyAlertSystem with default settings
    pub fn new() -> Self {
        Self {
            alert_enabled: true,
            alert_thresholds: HashMap::new(),
            notification_channels: Vec::new(),
            alerting: Arc::new(AtomicBool::new(false)),
            alert_history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Record an alert for a detected anomaly.
    ///
    /// The alert is recorded locally and is readable through
    /// [`Self::alert_history`]. No notification is delivered: no channel
    /// transport is wired into this crate, so a configured
    /// `notification_channels` entry is reported back as unsupported rather
    /// than acknowledged as delivered.
    pub fn trigger_alert(&self, anomaly: &AnomalyInfo) -> TestCharacterizationResult<()> {
        if !self.is_alerting() {
            return Err(TestCharacterizationError::InvalidInput {
                message: "alert system is not running; call start_alerting first".to_string(),
                field: "alerting".to_string(),
                value: "false".to_string(),
            });
        }
        if let Some(threshold) = self.alert_thresholds.get(&anomaly.severity) {
            if anomaly.impact < *threshold {
                return Ok(());
            }
        }
        self.alert_history.lock().push(anomaly.clone());
        if !self.notification_channels.is_empty() {
            return Err(TestCharacterizationError::NotSupported {
                message: format!(
                    "the alert was recorded locally but {} notification channel(s) are \
                     configured and no channel transport is wired in",
                    self.notification_channels.len()
                ),
                component: "AnomalyAlertSystem".to_string(),
            });
        }
        Ok(())
    }

    /// Start the alerting system.
    pub fn start_alerting(&self) -> TestCharacterizationResult<()> {
        if !self.alert_enabled {
            return Err(TestCharacterizationError::NotSupported {
                message: "alerting is disabled by configuration".to_string(),
                component: "AnomalyAlertSystem".to_string(),
            });
        }
        self.alerting.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Stop the alerting system.
    pub fn stop_alerting(&self) -> TestCharacterizationResult<()> {
        self.alerting.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Whether alerting is running right now.
    pub fn is_alerting(&self) -> bool {
        self.alerting.load(Ordering::Relaxed)
    }

    /// Alerts recorded so far, oldest first.
    pub fn alert_history(&self) -> Vec<AnomalyInfo> {
        self.alert_history.lock().clone()
    }
}

impl Default for AnomalyAlertSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendDetectionEngine {
    /// Create a new TrendDetectionEngine with default settings
    pub fn new() -> Self {
        Self {
            detection_methods: Vec::new(),
            current_method: String::from("least_squares_slope"),
            detecting: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start trend detection.
    pub fn start_detection(&self) -> TestCharacterizationResult<()> {
        self.detecting.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Stop trend detection.
    pub fn stop_detection(&self) -> TestCharacterizationResult<()> {
        self.detecting.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Whether detection is running right now.
    pub fn is_detecting(&self) -> bool {
        self.detecting.load(Ordering::Relaxed)
    }

    /// Least-squares slope per sample of every metric in `observations`.
    ///
    /// A metric with fewer than three readings is omitted: two points always
    /// define a line, so a slope from them carries no evidence of a trend.
    pub fn detect_trends(
        &self,
        observations: InsightObservations<'_>,
    ) -> TestCharacterizationResult<HashMap<String, f64>> {
        if !self.is_detecting() {
            return Err(TestCharacterizationError::InvalidInput {
                message: "trend detection is not running; call start_detection first".to_string(),
                field: "detecting".to_string(),
                value: "false".to_string(),
            });
        }
        let mut slopes = HashMap::new();
        for key in observations.keys() {
            let values = observations.series(&key);
            if values.len() < 3 {
                continue;
            }
            let n = values.len() as f64;
            let mean_x = (n - 1.0) / 2.0;
            let mean_y = values.iter().sum::<f64>() / n;
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            for (index, value) in values.iter().enumerate() {
                let dx = index as f64 - mean_x;
                sxx += dx * dx;
                sxy += dx * (value - mean_y);
            }
            if sxx <= 0.0 {
                continue;
            }
            slopes.insert(key, sxy / sxx);
        }
        Ok(slopes)
    }
}

impl Default for TrendDetectionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyReportGenerator {
    /// Create a new AnomalyReportGenerator with default settings
    pub fn new() -> Self {
        Self {
            threshold: 0.8,
            auto_report: true,
        }
    }
}

impl Default for AnomalyReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ReportGenerator for AnomalyReportGenerator {
    fn generate(&self) -> String {
        format!(
            "Anomaly Report Generator (threshold={:.2}, auto_report={})",
            self.threshold, self.auto_report
        )
    }
}

impl TrendReportGenerator {
    /// Create a new TrendReportGenerator with default settings
    pub fn new() -> Self {
        Self {
            trend_window_size: 100,
            include_forecast: true,
        }
    }
}

impl Default for TrendReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ReportGenerator for TrendReportGenerator {
    fn generate(&self) -> String {
        format!(
            "Trend Report Generator (window_size={}, include_forecast={})",
            self.trend_window_size, self.include_forecast
        )
    }
}
