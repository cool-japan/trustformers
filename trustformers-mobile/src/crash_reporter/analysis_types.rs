//! Crash analysis, pattern-matching and risk-assessment types.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Crash pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern description
    pub description: String,
    /// Frequency
    pub frequency: usize,
    /// Confidence
    pub confidence: f32,
}
/// Error log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorLogEntry {
    /// Log level
    pub level: LogLevel,
    /// Log message
    pub message: String,
    /// Timestamp
    pub timestamp: u64,
    /// Source file
    pub source: Option<String>,
    /// Line number
    pub line: Option<u32>,
}
/// Impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    /// User impact
    pub user_impact: ImpactLevel,
    /// Business impact
    pub business_impact: ImpactLevel,
    /// Technical impact
    pub technical_impact: ImpactLevel,
    /// Security impact
    pub security_impact: ImpactLevel,
}
/// Impact levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpactLevel {
    Minimal,
    Low,
    Medium,
    High,
    Severe,
}
/// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warning,
    Info,
    Debug,
    Trace,
}
/// Memory permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPermissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}
/// Memory region types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryRegionType {
    Code,
    Data,
    Heap,
    Stack,
    Shared,
    Unknown,
}
/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    /// Inference time
    pub inference_time: Duration,
    /// Memory usage
    pub memory_usage: usize,
    /// GPU utilization
    pub gpu_utilization: Option<f32>,
    /// Throughput
    pub throughput: f32,
}
/// Network connection types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkConnectionType {
    WiFi,
    Cellular,
    Ethernet,
    Bluetooth,
    None,
    Unknown,
}
/// Pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    MemoryLeak,
    StackOverflow,
    DeadLock,
    RaceCondition,
    ResourceExhaustion,
    InvalidState,
    ConfigurationError,
    Unknown,
}
/// Recovery impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryImpact {
    /// User experience impact
    pub user_experience: ImpactLevel,
    /// Performance impact
    pub performance: ImpactLevel,
    /// Data loss risk
    pub data_loss_risk: RiskLevel,
    /// Recovery time estimate
    pub recovery_time_estimate: Duration,
}
/// Resolution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionStatus {
    Unresolved,
    InProgress,
    Resolved,
    WontFix,
}
/// Risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk level
    pub risk_level: RiskLevel,
    /// Likelihood of recurrence
    pub recurrence_likelihood: f32,
    /// Impact assessment
    pub impact_assessment: ImpactAssessment,
    /// Mitigation urgency
    pub mitigation_urgency: UrgencyLevel,
}
/// Risk levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}
/// Similar crash
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarCrash {
    /// Crash report ID
    pub report_id: String,
    /// Similarity score (0-1)
    pub similarity_score: f32,
    /// Matching patterns
    pub matching_patterns: Vec<String>,
    /// Resolution status
    pub resolution_status: ResolutionStatus,
}
/// System event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemEvent {
    /// Event type
    pub event_type: String,
    /// Timestamp
    pub timestamp: u64,
    /// Event data
    pub data: HashMap<String, String>,
}
/// Thread states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreadState {
    Running,
    Blocked,
    Waiting,
    Terminated,
    Unknown,
}
/// Urgency levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}
/// User action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAction {
    /// Action type
    pub action_type: String,
    /// Timestamp
    pub timestamp: u64,
    /// Action details
    pub details: HashMap<String, String>,
}
