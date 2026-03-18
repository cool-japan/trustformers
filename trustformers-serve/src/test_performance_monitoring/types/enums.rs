//! Enums Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt, time::Duration};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum PressureLevel {
    #[default]
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for PressureLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PressureLevel::Low => write!(f, "Low"),
            PressureLevel::Medium => write!(f, "Medium"),
            PressureLevel::High => write!(f, "High"),
            PressureLevel::Critical => write!(f, "Critical"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    #[default]
    Unknown,
    Increasing,
    Decreasing,
}

impl fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrendDirection::Improving => write!(f, "Improving"),
            TrendDirection::Stable => write!(f, "Stable"),
            TrendDirection::Degrading => write!(f, "Degrading"),
            TrendDirection::Unknown => write!(f, "Unknown"),
            TrendDirection::Increasing => write!(f, "Increasing"),
            TrendDirection::Decreasing => write!(f, "Decreasing"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum EventSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl fmt::Display for EventSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventSeverity::Info => write!(f, "Info"),
            EventSeverity::Warning => write!(f, "Warning"),
            EventSeverity::Error => write!(f, "Error"),
            EventSeverity::Critical => write!(f, "Critical"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum RateLimitingStrategy {
    /// Token bucket algorithm
    TokenBucket { capacity: u64, refill_rate: u64 },
    /// Fixed window rate limiting
    FixedWindow {
        window_size: Duration,
        max_requests: u64,
    },
    /// Sliding window rate limiting
    SlidingWindow {
        window_size: Duration,
        max_requests: u64,
    },
    /// No rate limiting
    None,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ThresholdAction {
    /// Log the event
    Log,
    /// Send an alert
    Alert,
    /// Trigger optimization
    Optimize,
    /// Scale resources
    Scale,
    /// Stop monitoring
    Stop,
    /// Custom action with command
    Custom(String),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    /// Statistical outlier detection
    StatisticalOutlier,
    /// Isolation forest
    IsolationForest,
    /// Moving average based
    MovingAverage,
    /// Seasonal decomposition
    SeasonalDecomposition,
    /// Machine learning based
    MachineLearning,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AnomalyScoringMethod {
    /// Z-score based scoring
    ZScore,
    /// Percentile based scoring
    Percentile,
    /// IQR based scoring
    IQR,
    /// Custom scoring function
    Custom,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Snappy,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ArchivalStorageType {
    LocalFileSystem,
    S3,
    GCS,
    Azure,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ArchivalSchedule {
    Daily,
    Weekly,
    Monthly,
    Manual,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ArchivalFormat {
    Json,
    CompressedJson,
    Parquet,
    Csv,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SubscriptionPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ReportSection {
    Summary,
    TestExecution,
    ResourceUsage,
    Performance,
    Alerts,
    Trends,
    Recommendations,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ReportType {
    Summary,
    Detailed,
    Performance,
    Historical,
    Comparative,
    Custom(String),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Html,
    Pdf,
    Csv,
    Xml,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AggregationLevel {
    Raw,
    Minute,
    Hour,
    Day,
    Week,
    Month,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MetricValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Duration(Duration),
    Timestamp(DateTime<Utc>),
    Array(Vec<MetricValue>),
    Object(HashMap<String, MetricValue>),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ThresholdType {
    Absolute,
    Percentage,
    Relative,
    Trend,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AnomalyType {
    Outlier,
    Drift,
    Spike,
    Drop,
    Seasonal,
    Trend,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AlertType {
    Performance,
    Resource,
    Error,
    Threshold,
    Anomaly,
    System,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum EventStorageType {
    /// In-memory storage
    Memory,
    /// File-based storage
    File,
    /// Database storage
    Database,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SecurityClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeSeriesDataType {
    Numeric,
    String,
    Boolean,
    Struct,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ResolutionMethod {
    Manual,
    Automatic,
    Escalation,
    Timeout,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AlertCategory {
    Performance,
    Resource,
    Error,
    Security,
    Availability,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum OutlierType {
    Low,
    High,
    Anomalous,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum RateLimitState {
    Normal,
    Limited,
    Throttled,
    Blocked,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContextValue {
    String(String),
    Number(f64),
    Boolean(bool),
    List(Vec<ContextValue>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RecoveryAction {
    Restart,
    Scale,
    Notify,
    Custom(String),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AlertStatus {
    /// Alert is active
    Active,
    /// Alert is acknowledged
    Acknowledged,
    /// Alert is resolved
    Resolved,
    /// Alert is suppressed
    Suppressed,
    /// Alert is expired
    Expired,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System is degraded
    Degraded,
    /// System is unhealthy
    Unhealthy,
    /// System is critical
    Critical,
    /// Status unknown
    Unknown,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// Low impact
    Low,
    /// Medium impact
    Medium,
    /// High impact
    High,
    /// Critical impact
    Critical,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum BusinessImpact {
    /// No business impact
    None,
    /// Minor business impact
    Minor,
    /// Moderate business impact
    Moderate,
    /// Major business impact
    Major,
    /// Severe business impact
    Severe,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum InsightType {
    /// Performance insight
    Performance,
    /// Anomaly insight
    Anomaly,
    /// Trend insight
    Trend,
    /// Pattern insight
    Pattern,
    /// Optimization insight
    Optimization,
    /// Risk insight
    Risk,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Statistical evidence
    Statistical,
    /// Historical evidence
    Historical,
    /// Comparative evidence
    Comparative,
    /// Experimental evidence
    Experimental,
    /// Observational evidence
    Observational,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum StatisticalMethod {
    /// Mean calculation
    Mean,
    /// Median calculation
    Median,
    /// Standard deviation
    StandardDeviation,
    /// Percentile calculation
    Percentile,
    /// Regression analysis
    Regression,
    /// Correlation analysis
    Correlation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OutlierDetectionMethod {
    /// Z-score method
    ZScore,
    /// IQR method
    IQR,
    /// MAD method (Median Absolute Deviation)
    MAD,
    /// Isolation forest
    IsolationForest,
    /// LOF (Local Outlier Factor)
    LOF,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum TestPhase {
    /// Setup phase
    Setup,
    /// Execution phase
    Execution,
    /// Teardown phase
    Teardown,
    /// Validation phase
    Validation,
    /// Cleanup phase
    Cleanup,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ThresholdValue {
    Absolute(f64),
    Percentage(f64),
    Dynamic(String),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AggregationMethod {
    Sum,
    Average,
    Min,
    Max,
    Count,
    Percentile,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AggregationScope {
    Test,
    Suite,
    Global,
    Custom,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum DynamicThresholdMethod {
    Statistical,
    MachineLearning,
    Adaptive,
    Baseline,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum LearningAlgorithm {
    LinearRegression,
    ARIMA,
    ExponentialSmoothing,
    NeuralNetwork,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum OutlierHandling {
    Ignore,
    Remove,
    Dampen,
    Flag,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum DeviationType {
    Absolute,
    Relative,
    StandardDeviation,
    Percentage,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum BaselineUpdateStrategy {
    Fixed,
    RollingWindow,
    ExponentialDecay,
    Adaptive,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ThresholdEvaluatorType {
    Static,
    Dynamic,
    Adaptive,
    ML,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum DependencyStatus {
    Resolved,
    Pending,
    Failed,
    Unknown,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ChangeType {
    Deployment,
    Configuration,
    Infrastructure,
    Code,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ActionType {
    Restart,
    Scale,
    Rollback,
    Notify,
    Custom,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ActionPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum NotificationType {
    Email,
    Sms,
    Slack,
    Webhook,
    PagerDuty,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum NotificationPriority {
    Low,
    Normal,
    High,
    Urgent,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum DeliveryResult {
    Success,
    Failed,
    Pending,
    Throttled,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum NotificationChannelType {
    Email,
    Sms,
    Slack,
    Webhook,
    PagerDuty,
    Custom,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum PayloadFormat {
    Json,
    Xml,
    FormUrlEncoded,
    Custom,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Range(f64, f64),
}

impl Default for FilterValue {
    fn default() -> Self {
        Self::String(String::new())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum QualityIssueType {
    MissingData,
    InvalidData,
    DuplicateData,
    OutOfRange,
    Inconsistent,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum TestStatus {
    Pending,
    Running,
    Passed,
    Failed,
    Skipped,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum OptimizationCategory {
    Performance,
    Resource,
    Reliability,
    Efficiency,
    Quality,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum EffortLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    IO,
    Network,
    Lock,
    Database,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum DetectionMethod {
    Statistical,
    MachineLearning,
    RuleBased,
    Hybrid,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum FailureType {
    Timeout,
    Error,
    Crash,
    AssertionFailure,
    ResourceExhaustion,
    Unknown,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum IndicatorStatus {
    Normal,
    Warning,
    Critical,
    Unknown,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum CriticalityLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SubscriptionType {
    Alert,
    Event,
    Metric,
    Report,
    All,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum GroupType {
    BySource,
    BySeverity,
    ByCategory,
    ByTimeWindow,
    Custom,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum GroupStatus {
    Active,
    Resolved,
    Suppressed,
    Escalated,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SuppressionLevel {
    Low,
    Medium,
    High,
    Complete,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum RecoveryConditionType {
    MetricReturnsToNormal,
    ManualResolution,
    Timeout,
    AutoRecovery,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Trend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Unknown,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SensitivityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AnalysisDepth {
    Shallow,
    Normal,
    Deep,
    Comprehensive,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    Performance,
    ResourceEfficiency,
    Balanced,
    Custom,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ComparisonMethod {
    Absolute,
    Relative,
    Percentage,
    Statistical,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum BaselineType {
    Static,
    Rolling,
    Adaptive,
    Seasonal,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum FilterOperator {
    #[default]
    Equals,
    NotEquals,
    Contains,
    GreaterThan,
    LessThan,
    Between,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum ComparisonType {
    #[default]
    Absolute,
    Relative,
    Percentage,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum UpdateType {
    #[default]
    Add,
    Update,
    Delete,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum ThresholdDirection {
    #[default]
    Above,
    Below,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum AvailabilityStatus {
    #[default]
    Available,
    Unavailable,
    Degraded,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum PredicateType {
    #[default]
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    #[default]
    RoundRobin,
    LeastConnections,
    Random,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum PartitionStatus {
    #[default]
    Active,
    Archived,
    Deleting,
    Error,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum StorageClass {
    #[default]
    Hot,
    Warm,
    Cold,
    Archive,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum AvailabilityLevel {
    #[default]
    High,
    Medium,
    Low,
    Archive,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum CompressionLevel {
    #[default]
    None,
    Fast,
    Standard,
    Maximum,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum LifecycleStageType {
    #[default]
    Active,
    Transitioning,
    Archived,
    Deleted,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum OutputFormat {
    #[default]
    Json,
    Csv,
    Parquet,
    Avro,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum AggregationType {
    #[default]
    Sum,
    Count,
    Avg,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum SectionType {
    #[default]
    Summary,
    Details,
    Chart,
    Table,
    Text,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum DeliveryMethod {
    #[default]
    Email,
    Slack,
    Webhook,
    FileSystem,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum NotificationFrequency {
    #[default]
    Immediate,
    Batched,
    Hourly,
    Daily,
    Weekly,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum UserRole {
    #[default]
    Admin,
    Developer,
    Viewer,
    Guest,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum SubscriptionTemplateType {
    #[default]
    Custom,
    Alert,
    Report,
    Notification,
}
