//! Enumeration Types

use serde::{Deserialize, Serialize};

// Import common types

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

// =============================================================================
// EARLY ENUM DEFINITIONS
// =============================================================================

/// Feedback type classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackType {
    Performance,
    Resource,
    Quality,
    Error,
    Warning,
    Success,
}

/// Trend direction classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Unknown,
}

/// Cleanup priority levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CleanupPriority {
    Low,
    Normal,
    High,
    Critical,
    Emergency,
}

/// Risk type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskType {
    Performance,
    Security,
    Resource,
    Operational,
    Financial,
    Technical,
}

/// Memory type classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    Heap,
    Stack,
    Static,
    Shared,
    GPU,
    Cache,
}

/// Impact area classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpactArea {
    Performance,
    Reliability,
    Security,
    UserExperience,
    ResourceUtilization,
    Cost,
}

// =============================================================================
// MAIN ENUMS
// =============================================================================

/// Types of performance insights
///
/// Classification of different types of performance insights that can be
/// generated from real-time data analysis.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InsightType {
    /// Performance degradation detected
    PerformanceDegradation,

    /// Resource bottleneck identified
    ResourceBottleneck,

    /// Optimization opportunity found
    OptimizationOpportunity,

    /// Anomalous behavior detected
    AnomalousBehavior,

    /// Trend change identified
    TrendChange,

    /// Threshold violation
    ThresholdViolation,

    /// Capacity planning insight
    CapacityPlanning,

    /// Efficiency improvement
    EfficiencyImprovement,

    /// Custom insight
    Custom(String),
}

impl std::fmt::Display for InsightType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InsightType::PerformanceDegradation => write!(f, "Performance Degradation"),
            InsightType::ResourceBottleneck => write!(f, "Resource Bottleneck"),
            InsightType::OptimizationOpportunity => write!(f, "Optimization Opportunity"),
            InsightType::AnomalousBehavior => write!(f, "Anomalous Behavior"),
            InsightType::TrendChange => write!(f, "Trend Change"),
            InsightType::ThresholdViolation => write!(f, "Threshold Violation"),
            InsightType::CapacityPlanning => write!(f, "Capacity Planning"),
            InsightType::EfficiencyImprovement => write!(f, "Efficiency Improvement"),
            InsightType::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}

// SeverityLevel is imported from pattern_engine module - see line 115

/// Scope of monitoring for individual threads
///
/// Defines the specific monitoring responsibilities and data collection
/// scope for individual monitoring threads.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MonitoringScope {
    /// CPU performance monitoring
    CpuMonitoring,

    /// Memory performance monitoring
    MemoryMonitoring,

    /// I/O performance monitoring
    IoMonitoring,

    /// Network performance monitoring
    NetworkMonitoring,

    /// Application-level monitoring
    ApplicationMonitoring,

    /// System-level monitoring
    SystemMonitoring,

    /// Thread-level monitoring
    ThreadMonitoring,

    /// Process-level monitoring
    ProcessMonitoring,

    /// Custom monitoring scope
    Custom(String),
}

impl std::fmt::Display for MonitoringScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MonitoringScope::CpuMonitoring => write!(f, "CPU Monitoring"),
            MonitoringScope::MemoryMonitoring => write!(f, "Memory Monitoring"),
            MonitoringScope::IoMonitoring => write!(f, "I/O Monitoring"),
            MonitoringScope::NetworkMonitoring => write!(f, "Network Monitoring"),
            MonitoringScope::ApplicationMonitoring => write!(f, "Application Monitoring"),
            MonitoringScope::SystemMonitoring => write!(f, "System Monitoring"),
            MonitoringScope::ThreadMonitoring => write!(f, "Thread Monitoring"),
            MonitoringScope::ProcessMonitoring => write!(f, "Process Monitoring"),
            MonitoringScope::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}

/// Types of monitoring events
///
/// Classification of different monitoring events for system communication
/// and event-driven processing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MonitoringEventType {
    /// Metrics collected
    MetricsCollected,

    /// Threshold exceeded
    ThresholdExceeded,

    /// Anomaly detected
    AnomalyDetected,

    /// System state changed
    SystemStateChanged,

    /// Performance degraded
    PerformanceDegraded,

    /// Performance improved
    PerformanceImproved,

    /// Optimization applied
    OptimizationApplied,

    /// Configuration changed
    ConfigurationChanged,

    /// Error occurred
    ErrorOccurred,

    /// Warning issued
    WarningIssued,

    /// System started
    SystemStarted,

    /// System stopped
    SystemStopped,

    /// Monitoring started
    MonitoringStarted,

    /// Monitoring shutdown
    MonitoringShutdown,

    /// Custom event
    Custom(String),
}

impl std::fmt::Display for MonitoringEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MonitoringEventType::MetricsCollected => write!(f, "Metrics Collected"),
            MonitoringEventType::ThresholdExceeded => write!(f, "Threshold Exceeded"),
            MonitoringEventType::AnomalyDetected => write!(f, "Anomaly Detected"),
            MonitoringEventType::SystemStateChanged => write!(f, "System State Changed"),
            MonitoringEventType::PerformanceDegraded => write!(f, "Performance Degraded"),
            MonitoringEventType::PerformanceImproved => write!(f, "Performance Improved"),
            MonitoringEventType::OptimizationApplied => write!(f, "Optimization Applied"),
            MonitoringEventType::ConfigurationChanged => write!(f, "Configuration Changed"),
            MonitoringEventType::ErrorOccurred => write!(f, "Error Occurred"),
            MonitoringEventType::WarningIssued => write!(f, "Warning Issued"),
            MonitoringEventType::SystemStarted => write!(f, "System Started"),
            MonitoringEventType::SystemStopped => write!(f, "System Stopped"),
            MonitoringEventType::MonitoringStarted => write!(f, "Monitoring Started"),
            MonitoringEventType::MonitoringShutdown => write!(f, "Monitoring Shutdown"),
            MonitoringEventType::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}

/// Direction for threshold evaluation
///
/// Specifies whether threshold violations occur when values are above
/// or below the configured threshold levels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThresholdDirection {
    /// Violation when value exceeds threshold
    Above,

    /// Violation when value falls below threshold
    Below,

    /// Violation when value is outside range
    OutsideRange { min: f64, max: f64 },

    /// Violation when value is inside range (for inverted logic)
    InsideRange { min: f64, max: f64 },
}

impl std::fmt::Display for ThresholdDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThresholdDirection::Above => write!(f, "Above"),
            ThresholdDirection::Below => write!(f, "Below"),
            ThresholdDirection::OutsideRange { min, max } => {
                write!(f, "Outside [{}, {}]", min, max)
            },
            ThresholdDirection::InsideRange { min, max } => write!(f, "Inside [{}, {}]", min, max),
        }
    }
}

impl ThresholdDirection {
    /// Check if value violates threshold
    pub fn is_violation(&self, value: f64, threshold: f64) -> bool {
        match self {
            ThresholdDirection::Above => value > threshold,
            ThresholdDirection::Below => value < threshold,
            ThresholdDirection::OutsideRange { min, max } => value < *min || value > *max,
            ThresholdDirection::InsideRange { min, max } => value >= *min && value <= *max,
        }
    }
}

// ActionType is imported from parent types module - see line 86

/// Types of quality issues
///
/// Classification of different types of quality issues that can be
/// detected during quality checking.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualityIssueType {
    /// Missing data
    MissingData,

    /// Inconsistent data
    InconsistentData,

    /// Outlier data
    OutlierData,

    /// Stale data
    StaleData,

    /// Corrupted data
    CorruptedData,

    /// Duplicate data
    DuplicateData,

    /// Invalid format
    InvalidFormat,

    /// Schema mismatch
    SchemaMismatch,

    /// Custom issue
    Custom(String),
}

impl std::fmt::Display for QualityIssueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QualityIssueType::MissingData => write!(f, "Missing Data"),
            QualityIssueType::InconsistentData => write!(f, "Inconsistent Data"),
            QualityIssueType::OutlierData => write!(f, "Outlier Data"),
            QualityIssueType::StaleData => write!(f, "Stale Data"),
            QualityIssueType::CorruptedData => write!(f, "Corrupted Data"),
            QualityIssueType::DuplicateData => write!(f, "Duplicate Data"),
            QualityIssueType::InvalidFormat => write!(f, "Invalid Format"),
            QualityIssueType::SchemaMismatch => write!(f, "Schema Mismatch"),
            QualityIssueType::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}

/// Quality enforcement levels
///
/// Different levels of quality enforcement with varying strictness
/// and impact on processing operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Advisory only (warnings)
    Advisory,

    /// Moderate enforcement (warnings and corrections)
    Moderate,

    /// Strict enforcement (block processing on violations)
    Strict,

    /// Emergency mode (immediate escalation)
    Emergency,
}

impl std::fmt::Display for EnforcementLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnforcementLevel::Advisory => write!(f, "Advisory"),
            EnforcementLevel::Moderate => write!(f, "Moderate"),
            EnforcementLevel::Strict => write!(f, "Strict"),
            EnforcementLevel::Emergency => write!(f, "Emergency"),
        }
    }
}

/// Types of optimization objectives
///
/// Different types of optimization objectives that can be pursued
/// simultaneously in multi-objective optimization.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectiveType {
    /// Maximize throughput
    MaximizeThroughput,

    /// Minimize latency
    MinimizeLatency,

    /// Minimize resource usage
    MinimizeResourceUsage,

    /// Maximize efficiency
    MaximizeEfficiency,

    /// Minimize cost
    MinimizeCost,

    /// Minimize energy consumption
    MinimizeEnergy,

    /// Maximize reliability
    MaximizeReliability,

    /// Custom objective
    Custom {
        name: String,
        direction: OptimizationDirection,
    },
}

impl std::fmt::Display for ObjectiveType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjectiveType::MaximizeThroughput => write!(f, "Maximize Throughput"),
            ObjectiveType::MinimizeLatency => write!(f, "Minimize Latency"),
            ObjectiveType::MinimizeResourceUsage => write!(f, "Minimize Resource Usage"),
            ObjectiveType::MaximizeEfficiency => write!(f, "Maximize Efficiency"),
            ObjectiveType::MinimizeCost => write!(f, "Minimize Cost"),
            ObjectiveType::MinimizeEnergy => write!(f, "Minimize Energy"),
            ObjectiveType::MaximizeReliability => write!(f, "Maximize Reliability"),
            ObjectiveType::Custom { name, direction } => {
                write!(f, "Custom: {} ({:?})", name, direction)
            },
        }
    }
}

/// Optimization direction
///
/// Direction for optimization objectives (minimize or maximize).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationDirection {
    /// Minimize the objective
    Minimize,

    /// Maximize the objective
    Maximize,
}

impl std::fmt::Display for OptimizationDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationDirection::Minimize => write!(f, "Minimize"),
            OptimizationDirection::Maximize => write!(f, "Maximize"),
        }
    }
}

// =============================================================================
