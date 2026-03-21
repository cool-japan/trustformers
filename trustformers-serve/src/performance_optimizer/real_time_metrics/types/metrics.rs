//! Metrics and Analysis Types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

// Import common types

// Import types from sibling modules
use super::aggregators::ConfidenceMethod;
use super::data_structures::WindowStatistics;
use super::enums::{InsightType, TrendDirection};

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

// METRICS AND ANALYSIS TYPES
// =============================================================================

/// Data quality assessment metrics
///
/// Comprehensive assessment of data quality including completeness, accuracy,
/// consistency, timeliness, and trend analysis for informed decision making.
/// This is the canonical QualityMetrics used in WindowStatistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,

    /// Data completeness score (0.0 to 1.0)
    pub completeness_score: f32,

    /// Data accuracy score (0.0 to 1.0)
    pub accuracy_score: f32,

    /// Data consistency score (0.0 to 1.0)
    pub consistency_score: f32,

    /// Data timeliness score (0.0 to 1.0)
    pub timeliness_score: f32,

    /// Outlier percentage
    pub outlier_percentage: f32,

    /// Missing data percentage
    pub missing_data_percentage: f32,

    /// Quality trend direction
    pub quality_trend: TrendDirection,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            overall_score: 1.0,
            completeness_score: 1.0,
            accuracy_score: 1.0,
            consistency_score: 1.0,
            timeliness_score: 1.0,
            outlier_percentage: 0.0,
            missing_data_percentage: 0.0,
            quality_trend: TrendDirection::Stable,
        }
    }
}

/// Aggregation result with comprehensive analysis
///
/// Complete aggregation result including statistical analysis, trends,
/// insights, and recommendations based on real-time data processing.
#[derive(Debug, Clone)]
pub struct AggregationResult {
    /// Aggregation timestamp
    pub timestamp: DateTime<Utc>,

    /// Aggregation window
    pub window: Duration,

    /// Statistical summary
    pub statistics: WindowStatistics,

    /// Trend analysis
    pub trends: Vec<PerformanceTrend>,

    /// Performance insights
    pub insights: Vec<PerformanceInsight>,

    /// Recommendations
    pub recommendations: Vec<RecommendedAction>,

    /// Confidence score
    pub confidence: f32,

    /// Processing metadata
    pub metadata: HashMap<String, String>,

    /// Window duration
    pub window_duration: Duration,

    /// Data point count
    pub data_point_count: usize,

    /// Quality score
    pub quality_score: f32,

    /// Trend analysis (detailed)
    pub trend_analysis: String,
}

impl AggregationResult {
    /// Create new aggregation result
    pub fn new(window: Duration, statistics: WindowStatistics) -> Self {
        Self {
            timestamp: Utc::now(),
            window,
            statistics,
            trends: Vec::new(),
            insights: Vec::new(),
            recommendations: Vec::new(),
            confidence: 1.0,
            metadata: HashMap::new(),
            window_duration: window,
            data_point_count: 0, // Will be updated as data is aggregated
            quality_score: 1.0,  // High quality by default
            trend_analysis: String::new(), // Will be populated by trend analysis
        }
    }

    /// Add insight to result
    pub fn add_insight(&mut self, insight: PerformanceInsight) {
        self.insights.push(insight);
    }

    /// Add recommendation to result
    pub fn add_recommendation(&mut self, recommendation: RecommendedAction) {
        self.recommendations.push(recommendation);
    }

    /// Check if result has critical insights
    pub fn has_critical_insights(&self) -> bool {
        self.insights.iter().any(|i| i.severity == SeverityLevel::Critical)
    }
}

/// Performance insight from real-time analysis
///
/// Actionable performance insight derived from real-time data analysis
/// with severity assessment and recommended actions.
#[derive(Debug, Clone)]
pub struct PerformanceInsight {
    /// Insight type
    pub insight_type: InsightType,

    /// Insight description
    pub description: String,

    /// Severity level
    pub severity: SeverityLevel,

    /// Confidence score
    pub confidence: f32,

    /// Supporting data
    pub supporting_data: HashMap<String, f64>,

    /// Recommended actions
    pub actions: Vec<RecommendedAction>,

    /// Impact assessment
    pub impact: ImpactAssessment,
}

impl PerformanceInsight {
    /// Create new performance insight
    pub fn new(insight_type: InsightType, description: String, severity: SeverityLevel) -> Self {
        Self {
            insight_type,
            description,
            severity,
            confidence: 0.0,
            supporting_data: HashMap::new(),
            actions: Vec::new(),
            impact: ImpactAssessment::default(),
        }
    }

    /// Add supporting data
    pub fn add_data(&mut self, key: String, value: f64) {
        self.supporting_data.insert(key, value);
    }

    /// Add recommended action
    pub fn add_action(&mut self, action: RecommendedAction) {
        self.actions.push(action);
    }

    /// Check if insight requires immediate attention
    pub fn requires_immediate_attention(&self) -> bool {
        matches!(self.severity, SeverityLevel::High | SeverityLevel::Critical)
    }
}

/// Impact assessment for insights and recommendations
///
/// Comprehensive assessment of potential impact including performance,
/// resources, costs, and risks for informed decision making.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    /// Performance impact estimate
    pub performance_impact: f32,

    /// Resource impact estimate
    pub resource_impact: f32,

    /// Implementation complexity
    pub complexity: f32,

    /// Risk assessment
    pub risk_level: f32,

    /// Estimated benefit
    pub estimated_benefit: f32,

    /// Time to implementation
    pub implementation_time: Duration,
}

impl Default for ImpactAssessment {
    fn default() -> Self {
        Self {
            performance_impact: 0.0,
            resource_impact: 0.0,
            complexity: 0.5,
            risk_level: 0.3,
            estimated_benefit: 0.0,
            implementation_time: Duration::from_secs(300),
        }
    }
}

impl ImpactAssessment {
    /// Calculate overall impact score
    pub fn overall_score(&self) -> f32 {
        let benefit_score = self.estimated_benefit;
        let cost_score = (self.complexity + self.risk_level + self.resource_impact.abs()) / 3.0;
        (benefit_score - cost_score).max(0.0)
    }

    /// Check if implementation is recommended
    pub fn is_recommended(&self) -> bool {
        self.overall_score() > 0.5 && self.risk_level < 0.7
    }
}

/// Performance baseline for comparison
///
/// Established performance baseline with statistical characteristics
/// for detecting deviations and performance changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Baseline timestamp
    pub timestamp: DateTime<Utc>,

    /// Baseline throughput
    pub baseline_throughput: f64,

    /// Baseline latency
    pub baseline_latency: Duration,

    /// Baseline CPU utilization
    pub baseline_cpu: f32,

    /// Baseline memory utilization
    pub baseline_memory: f32,

    /// Baseline variability
    pub variability_bounds: VariabilityBounds,

    /// Confidence intervals
    pub confidence_intervals: ConfidenceIntervals,

    /// Baseline quality score
    pub quality_score: f32,
}

impl PerformanceBaseline {
    /// Create new baseline from window statistics
    pub fn from_statistics(stats: &WindowStatistics) -> Self {
        Self {
            timestamp: Utc::now(),
            baseline_throughput: stats.mean_throughput,
            baseline_latency: stats.mean_latency,
            baseline_cpu: stats.mean_cpu_utilization,
            baseline_memory: stats.mean_memory_utilization,
            variability_bounds: VariabilityBounds::from_statistics(stats),
            confidence_intervals: ConfidenceIntervals::from_statistics(stats),
            quality_score: stats.quality_metrics.overall_score,
        }
    }

    /// Check if current metrics deviate from baseline
    pub fn check_deviation(&self, metrics: &RealTimeMetrics) -> bool {
        let throughput_ok = self.variability_bounds.throughput_bounds.0 <= metrics.throughput
            && metrics.throughput <= self.variability_bounds.throughput_bounds.1;

        let cpu_ok = self.variability_bounds.cpu_bounds.0 <= metrics.cpu_utilization
            && metrics.cpu_utilization <= self.variability_bounds.cpu_bounds.1;

        let memory_ok = self.variability_bounds.memory_bounds.0 <= metrics.memory_utilization
            && metrics.memory_utilization <= self.variability_bounds.memory_bounds.1;

        !(throughput_ok && cpu_ok && memory_ok)
    }

    /// Get baseline age
    pub fn age(&self) -> Duration {
        let now = Utc::now();
        (now - self.timestamp).to_std().unwrap_or(Duration::from_secs(0))
    }

    /// Check if baseline needs update
    pub fn needs_update(&self, max_age: Duration) -> bool {
        self.age() > max_age
    }
}

/// Variability bounds for performance baseline
///
/// Statistical bounds defining normal variability ranges for performance
/// metrics to distinguish normal fluctuations from significant changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariabilityBounds {
    /// Throughput bounds (min, max)
    pub throughput_bounds: (f64, f64),

    /// Latency bounds (min, max)
    pub latency_bounds: (Duration, Duration),

    /// CPU utilization bounds (min, max)
    pub cpu_bounds: (f32, f32),

    /// Memory utilization bounds (min, max)
    pub memory_bounds: (f32, f32),

    /// Efficiency bounds (min, max)
    pub efficiency_bounds: (f32, f32),
}

impl VariabilityBounds {
    /// Create bounds from window statistics
    pub fn from_statistics(stats: &WindowStatistics) -> Self {
        let throughput_margin = stats.throughput_std_dev * 2.0; // 2 sigma
        let cpu_margin = stats.mean_cpu_utilization * 0.1; // 10% margin
        let memory_margin = stats.mean_memory_utilization * 0.1; // 10% margin
        let latency_margin = stats.mean_latency / 10; // 10% margin

        Self {
            throughput_bounds: (
                (stats.mean_throughput - throughput_margin).max(0.0),
                stats.mean_throughput + throughput_margin,
            ),
            latency_bounds: (
                stats.mean_latency.saturating_sub(latency_margin),
                stats.mean_latency + latency_margin,
            ),
            cpu_bounds: (
                (stats.mean_cpu_utilization - cpu_margin).max(0.0),
                (stats.mean_cpu_utilization + cpu_margin).min(100.0),
            ),
            memory_bounds: (
                (stats.mean_memory_utilization - memory_margin).max(0.0),
                (stats.mean_memory_utilization + memory_margin).min(100.0),
            ),
            efficiency_bounds: (0.5, 1.0), // Simplified
        }
    }
}

/// Confidence intervals for baseline metrics
///
/// Statistical confidence intervals for baseline performance metrics
/// to support reliable anomaly detection and performance comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f32,

    /// Throughput confidence interval
    pub throughput_interval: (f64, f64),

    /// Latency confidence interval
    pub latency_interval: (Duration, Duration),

    /// CPU utilization confidence interval
    pub cpu_interval: (f32, f32),

    /// Memory utilization confidence interval
    pub memory_interval: (f32, f32),

    /// Network throughput confidence interval
    pub network_interval: (f64, f64),

    /// I/O operations confidence interval
    pub io_interval: (f64, f64),

    /// Response time confidence interval
    pub response_time_interval: (Duration, Duration),

    /// Error rate confidence interval
    pub error_rate_interval: (f32, f32),

    /// Statistical method used for calculation
    pub method: ConfidenceMethod,

    /// Mean lower bound
    pub mean_lower: f64,

    /// Mean upper bound
    pub mean_upper: f64,

    /// Variance lower bound
    pub variance_lower: f64,

    /// Variance upper bound
    pub variance_upper: f64,
}

impl ConfidenceIntervals {
    /// Create confidence intervals from window statistics
    pub fn from_statistics(stats: &WindowStatistics) -> Self {
        // Simplified confidence interval calculation (assuming normal distribution)
        let confidence_level = 95.0;
        let z_score = 1.96; // for 95% confidence

        let throughput_margin = (stats.throughput_std_dev / (stats.count as f64).sqrt()) * z_score;
        let cpu_margin = stats.mean_cpu_utilization * 0.05; // Simplified
        let memory_margin = stats.mean_memory_utilization * 0.05; // Simplified
        let latency_margin = stats.mean_latency / 20; // Simplified

        Self {
            confidence_level,
            throughput_interval: (
                stats.mean_throughput - throughput_margin,
                stats.mean_throughput + throughput_margin,
            ),
            latency_interval: (
                stats.mean_latency.saturating_sub(latency_margin),
                stats.mean_latency + latency_margin,
            ),
            cpu_interval: (
                (stats.mean_cpu_utilization - cpu_margin).max(0.0),
                (stats.mean_cpu_utilization + cpu_margin).min(100.0),
            ),
            memory_interval: (
                (stats.mean_memory_utilization - memory_margin).max(0.0),
                (stats.mean_memory_utilization + memory_margin).min(100.0),
            ),
            network_interval: (2_000_000.0, 8_000_000.0), // Simplified default
            io_interval: (200.0, 800.0),                  // Simplified default
            response_time_interval: (Duration::from_millis(15), Duration::from_millis(85)),
            error_rate_interval: (0.0, 3.0),
            method: ConfidenceMethod::TDistribution,
            mean_lower: stats.mean_throughput - throughput_margin,
            mean_upper: stats.mean_throughput + throughput_margin,
            variance_lower: (stats.throughput_std_dev * stats.throughput_std_dev * 0.8),
            variance_upper: (stats.throughput_std_dev * stats.throughput_std_dev * 1.2),
        }
    }
}

impl Default for ConfidenceIntervals {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            throughput_interval: (90.0, 110.0),
            latency_interval: (Duration::from_millis(45), Duration::from_millis(55)),
            cpu_interval: (0.35, 0.65),
            memory_interval: (0.55, 0.75),
            network_interval: (2_000_000.0, 8_000_000.0), // 2MB/s to 8MB/s
            io_interval: (200.0, 800.0),                  // 200 to 800 IOPS
            response_time_interval: (Duration::from_millis(15), Duration::from_millis(85)),
            error_rate_interval: (0.0, 3.0), // 0% to 3%
            method: ConfidenceMethod::TDistribution,
            mean_lower: 90.0,
            mean_upper: 110.0,
            variance_lower: 1.0,
            variance_upper: 5.0,
        }
    }
}

// =============================================================================
