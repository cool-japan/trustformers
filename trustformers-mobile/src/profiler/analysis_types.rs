//! Analysis result types: bottlenecks, optimization suggestions, patterns, regressions, trends and alerts.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}
/// Alert types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    HighCpuUsage,
    HighMemoryUsage,
    HighLatency,
    HighTemperature,
    LowBattery,
    HighPowerConsumption,
    PerformanceRegression,
    SystemOverload,
}
/// Bottleneck severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}
/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    GPU,
    IO,
    Network,
    Thermal,
    Battery,
    Backend,
}
/// Optimization categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationCategory {
    ModelCompression,
    MemoryOptimization,
    ComputeOptimization,
    ThermalManagement,
    PowerOptimization,
    NetworkOptimization,
    CacheOptimization,
}
/// Optimization difficulty levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationDifficulty {
    Easy,
    Medium,
    Hard,
    Expert,
}
/// Optimization priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
}
/// Optimization suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Optimization category
    pub category: OptimizationCategory,
    /// Description of the optimization
    pub description: String,
    /// Expected performance improvement (%)
    pub expected_improvement_percent: f32,
    /// Implementation difficulty
    pub difficulty: OptimizationDifficulty,
    /// Priority level
    pub priority: OptimizationPriority,
}
/// Types of performance patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    MemoryLeak,
    CpuSpike,
    ThermalThrottling,
    BatteryDrain,
    NetworkCongestion,
    LoadBalanceIssue,
    CacheInefficiency,
}
/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Metric value that triggered alert
    pub trigger_value: f32,
    /// Threshold that was exceeded
    pub threshold_value: f32,
    /// Timestamp when alert was triggered
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    /// Duration of the condition
    pub duration_ms: u64,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}
/// Performance bottleneck information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Description of the bottleneck
    pub description: String,
    /// Severity level
    pub severity: BottleneckSeverity,
    /// Duration of bottleneck (ms)
    pub duration_ms: u64,
    /// Impact on performance (%)
    pub performance_impact_percent: f32,
    /// Suggested optimizations
    pub optimizations: Vec<OptimizationSuggestion>,
    /// Detection confidence (0.0-1.0)
    pub confidence: f32,
}
/// Performance pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern description
    pub description: String,
    /// Frequency of occurrence
    pub frequency: f32,
    /// Performance impact
    pub impact: f32,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}
/// Performance regression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    /// Metric that regressed
    pub metric_name: String,
    /// Baseline value
    pub baseline_value: f32,
    /// Current value
    pub current_value: f32,
    /// Regression percentage
    pub regression_percent: f32,
    /// Regression severity
    pub severity: RegressionSeverity,
}
/// Performance trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend magnitude
    pub magnitude: f32,
    /// Trend confidence
    pub confidence: f32,
    /// Prediction for next period
    pub prediction: Option<f32>,
}
/// Regression severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Minor,
    Moderate,
    Major,
    Critical,
}
/// Trend directions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}
