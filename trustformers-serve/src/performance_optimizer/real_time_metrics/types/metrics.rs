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

    /// Whether `metrics` fall outside any bound this baseline actually has.
    ///
    /// A dimension with no measured bound (`None`) is skipped rather than
    /// counted as conforming: the baseline has nothing to say about it.
    pub fn check_deviation(&self, metrics: &RealTimeMetrics) -> bool {
        let bounds = &self.variability_bounds;
        let throughput_out = metrics.throughput < bounds.throughput_bounds.0
            || metrics.throughput > bounds.throughput_bounds.1;

        let cpu_out = bounds.cpu_bounds.is_some_and(|(low, high)| {
            metrics.cpu_utilization < low || metrics.cpu_utilization > high
        });

        let memory_out = bounds.memory_bounds.is_some_and(|(low, high)| {
            metrics.memory_utilization < low || metrics.memory_utilization > high
        });

        throughput_out || cpu_out || memory_out
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
    /// Throughput bounds (min, max), from the window's measured standard
    /// deviation.
    pub throughput_bounds: (f64, f64),

    /// Latency bounds (min, max), or `None` when the window carries no latency
    /// dispersion to derive them from.
    pub latency_bounds: Option<(Duration, Duration)>,

    /// CPU utilization bounds (min, max), or `None` when the window carries no
    /// CPU dispersion.
    pub cpu_bounds: Option<(f32, f32)>,

    /// Memory utilization bounds (min, max), or `None` when the window carries
    /// no memory dispersion.
    pub memory_bounds: Option<(f32, f32)>,

    /// Efficiency bounds (min, max), or `None` -- which is always, because
    /// `WindowStatistics` records no efficiency measurement at all.
    pub efficiency_bounds: Option<(f32, f32)>,
}

impl VariabilityBounds {
    /// Create bounds from window statistics.
    ///
    /// ## Changed in 0.2.1
    ///
    /// Only the throughput bound was ever derived from the window: the others
    /// were `mean * 0.1` ("10% margin") for CPU and memory, `mean_latency / 10`
    /// for latency, and the constant `(0.5, 1.0)` for efficiency. A ten-percent
    /// band around the mean is not a variability bound -- it is the same band
    /// whether the metric was rock steady or all over the place.
    /// `WindowStatistics` carries a standard deviation for throughput and for
    /// nothing else, so the other three are now `None`.
    pub fn from_statistics(stats: &WindowStatistics) -> Self {
        let throughput_margin = stats.throughput_std_dev * 2.0; // 2 sigma

        Self {
            throughput_bounds: (
                (stats.mean_throughput - throughput_margin).max(0.0),
                stats.mean_throughput + throughput_margin,
            ),
            latency_bounds: None,
            cpu_bounds: None,
            memory_bounds: None,
            efficiency_bounds: None,
        }
    }
}

/// Confidence intervals for baseline metrics
///
/// Statistical confidence intervals for baseline performance metrics
/// to support reliable anomaly detection and performance comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    /// Confidence level as a percentage (`95.0` means 95%).
    pub confidence_level: f32,

    /// Confidence interval for the mean throughput.
    pub throughput_interval: (f64, f64),

    /// Latency confidence interval, or `None` when the window carries no
    /// latency dispersion to build one from.
    pub latency_interval: Option<(Duration, Duration)>,

    /// CPU utilization confidence interval, or `None` when unmeasured.
    pub cpu_interval: Option<(f32, f32)>,

    /// Memory utilization confidence interval, or `None` when unmeasured.
    pub memory_interval: Option<(f32, f32)>,

    /// Network throughput confidence interval, or `None` -- the window carries
    /// no network measurement at all.
    pub network_interval: Option<(f64, f64)>,

    /// I/O operations confidence interval, or `None` -- likewise unmeasured.
    pub io_interval: Option<(f64, f64)>,

    /// Response time confidence interval, or `None` when unmeasured.
    pub response_time_interval: Option<(Duration, Duration)>,

    /// Error rate confidence interval, or `None` when unmeasured.
    pub error_rate_interval: Option<(f32, f32)>,

    /// Statistical method used for calculation
    pub method: ConfidenceMethod,

    /// Lower bound of the mean throughput interval.
    pub mean_lower: f64,

    /// Upper bound of the mean throughput interval.
    pub mean_upper: f64,

    /// Lower bound of the throughput variance interval, or `None` when the
    /// window holds too few samples to build one.
    pub variance_lower: Option<f64>,

    /// Upper bound of the throughput variance interval, or `None`.
    pub variance_upper: Option<f64>,
}

impl ConfidenceIntervals {
    /// Build intervals from a window's measured statistics.
    ///
    /// ## Changed in 0.2.1
    ///
    /// Only the throughput interval was ever computed. CPU and memory used a
    /// flat `mean * 0.05`, latency `mean / 20`, and network, I/O, response time
    /// and error rate were the literal constants `(2_000_000.0, 8_000_000.0)`,
    /// `(200.0, 800.0)`, `(15ms, 85ms)` and `(0.0, 3.0)` -- the same numbers
    /// for every window ever measured. The variance interval was the point
    /// estimate multiplied by 0.8 and 1.2. `method` claimed
    /// `TDistribution` while the code used the normal z-score 1.96.
    ///
    /// `WindowStatistics` carries a standard deviation for throughput and for
    /// nothing else, so throughput (mean and variance) is what can be reported;
    /// the rest are `None`.
    pub fn from_statistics(stats: &WindowStatistics) -> Self {
        // 95% two-sided normal quantile.
        let z_score = 1.96;
        let count = stats.count as f64;

        let throughput_margin = if count > 0.0 {
            (stats.throughput_std_dev / count.sqrt()) * z_score
        } else {
            0.0
        };
        let lower = stats.mean_throughput - throughput_margin;
        let upper = stats.mean_throughput + throughput_margin;

        // Large-sample interval for the variance: Var(s^2) ~ 2*sigma^4/(n-1),
        // so the margin is z * s^2 * sqrt(2/(n-1)). It needs at least two
        // samples; with fewer there is no interval to report.
        let variance = stats.throughput_std_dev * stats.throughput_std_dev;
        let variance_bounds = if stats.count >= 2 {
            let margin = z_score * variance * (2.0 / (count - 1.0)).sqrt();
            Some(((variance - margin).max(0.0), variance + margin))
        } else {
            None
        };

        Self {
            confidence_level: 95.0,
            throughput_interval: (lower, upper),
            latency_interval: None,
            cpu_interval: None,
            memory_interval: None,
            network_interval: None,
            io_interval: None,
            response_time_interval: None,
            error_rate_interval: None,
            method: ConfidenceMethod::Normal,
            mean_lower: lower,
            mean_upper: upper,
            variance_lower: variance_bounds.map(|(low, _)| low),
            variance_upper: variance_bounds.map(|(_, high)| high),
        }
    }
}

impl Default for ConfidenceIntervals {
    /// Intervals with nothing measured in them.
    ///
    /// ## Changed in 0.2.1
    ///
    /// The default used to be a full set of plausible-looking numbers --
    /// throughput `(90.0, 110.0)`, latency `(45ms, 55ms)`, CPU `(0.35, 0.65)`,
    /// network `(2MB/s, 8MB/s)`, and so on -- so anything that fell back to
    /// `Default::default()` published invented measurements that were
    /// indistinguishable from computed ones.
    fn default() -> Self {
        Self {
            confidence_level: 95.0,
            throughput_interval: (0.0, 0.0),
            latency_interval: None,
            cpu_interval: None,
            memory_interval: None,
            network_interval: None,
            io_interval: None,
            response_time_interval: None,
            error_rate_interval: None,
            method: ConfidenceMethod::Normal,
            mean_lower: 0.0,
            mean_upper: 0.0,
            variance_lower: None,
            variance_upper: None,
        }
    }
}

// =============================================================================

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod metrics_tests;
