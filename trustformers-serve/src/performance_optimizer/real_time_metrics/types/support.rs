//! Support Types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

// Import common types
use super::common::AtomicF32;
use super::errors::ErrorHandlingPolicy;

use num_cpus;

// Import types from sibling modules
use super::data_structures::TimestampedMetrics;
use super::enums::{EnforcementLevel, QualityIssueType};
use super::errors::ProcessingError;
use super::metrics::QualityMetrics;

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

// ADDITIONAL SUPPORT TYPES
// =============================================================================

/// Statistics for statistical processors
#[derive(Debug, Clone, Default)]
pub struct ProcessorStatistics {
    /// Total data points processed
    pub total_processed: u64,

    /// Processing errors
    pub errors: u64,

    /// Average processing time (microseconds)
    pub avg_processing_time: f32,

    /// Last processing timestamp
    pub last_processing: Option<DateTime<Utc>>,
}

/// Statistics for optimization algorithms
#[derive(Debug, Clone, Default)]
pub struct AlgorithmStatistics {
    /// Total recommendations generated
    pub recommendations_generated: u64,

    /// Average confidence score
    pub avg_confidence: f32,

    /// Algorithm accuracy (successful recommendations)
    pub accuracy: f32,

    /// Processing time statistics
    pub processing_time: ProcessorStatistics,
}

/// Statistics for quality checkers
#[derive(Debug, Clone, Default)]
pub struct CheckerStatistics {
    /// Total checks performed
    pub total_checks: u64,

    /// Quality issues found
    pub issues_found: u64,

    /// Average quality score
    pub avg_quality_score: f32,

    /// Check processing time
    pub avg_check_time: f32,
}

/// Quality requirements for processing
///
/// Quality requirements and thresholds for data processing operations
/// to ensure reliable and accurate results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    /// Minimum accuracy required
    pub min_accuracy: f32,

    /// Maximum acceptable error rate
    pub max_error_rate: f32,

    /// Required completeness
    pub required_completeness: f32,

    /// Consistency requirements
    pub consistency_requirements: HashMap<String, f32>,

    /// Maximum data age allowed
    pub max_data_age: Duration,
}

impl Default for QualityRequirements {
    fn default() -> Self {
        Self {
            min_accuracy: 0.95,
            max_error_rate: 0.05,
            required_completeness: 0.9,
            consistency_requirements: HashMap::new(),
            max_data_age: Duration::from_secs(300),
        }
    }
}

/// Pipeline configuration
///
/// Configuration for the processing pipeline including parallelism,
/// buffer sizes, and processing parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Parallel processing threads
    pub parallel_threads: usize,

    /// Buffer size per stage
    pub buffer_size: usize,

    /// Processing timeout
    pub timeout: Duration,

    /// Error handling policy
    pub error_handling: ErrorHandlingPolicy,

    /// Quality control enabled
    pub quality_control: bool,

    /// Pipeline stages configuration
    pub stages: Vec<String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            parallel_threads: num_cpus::get(),
            buffer_size: 1000,
            timeout: Duration::from_secs(30),
            error_handling: ErrorHandlingPolicy::default(),
            quality_control: true,
            stages: vec![
                "validation".to_string(),
                "filtering".to_string(),
                "aggregation".to_string(),
                "analysis".to_string(),
            ],
        }
    }
}

/// Pipeline performance statistics
///
/// Performance statistics for the processing pipeline including throughput,
/// latency, and resource utilization metrics.
#[derive(Debug, Default)]
pub struct PipelineStatistics {
    /// Total items processed
    pub items_processed: AtomicU64,

    /// Processing throughput (items/sec)
    pub throughput: AtomicF32,

    /// Average processing latency (microseconds)
    pub avg_latency: AtomicF32,

    /// Error rate
    pub error_rate: AtomicF32,

    /// Resource utilization
    pub resource_utilization: AtomicF32,

    /// Pipeline uptime
    pub uptime: AtomicU64,
}

impl PipelineStatistics {
    /// Get current throughput
    pub fn current_throughput(&self) -> f32 {
        self.throughput.load(Ordering::Acquire)
    }

    /// Get current error rate
    pub fn current_error_rate(&self) -> f32 {
        self.error_rate.load(Ordering::Acquire)
    }

    /// Update throughput
    pub fn update_throughput(&self, throughput: f32) {
        self.throughput.store(throughput, Ordering::Release);
    }
}

/// Pipeline input data structure
///
/// Input data structure for pipeline stages containing metrics data
/// and processing context.
#[derive(Debug, Clone)]
pub struct PipelineInput {
    /// Input data
    pub data: Vec<TimestampedMetrics>,

    /// Processing context
    pub context: ProcessingContext,

    /// Input metadata
    pub metadata: HashMap<String, String>,
}

/// Pipeline output data structure
///
/// Output data structure from pipeline stages containing processed results
/// and updated context.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// Output data
    pub data: Vec<TimestampedMetrics>,

    /// Processing results
    pub results: ProcessingResults,

    /// Output metadata
    pub metadata: HashMap<String, String>,
}

/// Processing context for pipeline operations
///
/// Context information maintained throughout pipeline processing including
/// configuration, state, and processing metadata.
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    /// Processing configuration
    pub config: PipelineConfig,

    /// Current processing stage
    pub current_stage: String,

    /// Processing timestamp
    pub timestamp: DateTime<Utc>,

    /// Quality requirements
    pub quality_requirements: QualityRequirements,

    /// Context metadata
    pub metadata: HashMap<String, String>,
}

impl ProcessingContext {
    /// Create new processing context
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            current_stage: "initialization".to_string(),
            timestamp: Utc::now(),
            quality_requirements: QualityRequirements::default(),
            metadata: HashMap::new(),
        }
    }

    /// Update current stage
    pub fn update_stage(&mut self, stage: String) {
        self.current_stage = stage;
        self.timestamp = Utc::now();
    }
}

/// Processing results from pipeline operations
///
/// Results from pipeline processing including statistics, quality metrics,
/// and any processing errors or warnings.
#[derive(Debug, Clone)]
pub struct ProcessingResults {
    /// Processing statistics
    pub statistics: ProcessorStatistics,

    /// Quality metrics
    pub quality_metrics: QualityMetrics,

    /// Processing errors
    pub errors: Vec<ProcessingError>,

    /// Result metadata
    pub metadata: HashMap<String, String>,
}

impl Default for ProcessingResults {
    fn default() -> Self {
        Self {
            statistics: ProcessorStatistics::default(),
            quality_metrics: QualityMetrics::default(),
            errors: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Quality check result
///
/// Result from quality checking including pass/fail status, quality scores,
/// and detailed findings.
#[derive(Debug, Clone)]
pub struct QualityCheckResult {
    /// Check passed
    pub passed: bool,

    /// Quality score (0.0 to 1.0)
    pub score: f32,

    /// Quality metrics
    pub metrics: QualityMetrics,

    /// Quality issues found
    pub issues: Vec<QualityIssue>,

    /// Check metadata
    pub metadata: HashMap<String, String>,
}

impl QualityCheckResult {
    /// Create successful quality check result
    pub fn success(score: f32, metrics: QualityMetrics) -> Self {
        Self {
            passed: true,
            score,
            metrics,
            issues: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create failed quality check result
    pub fn failure(score: f32, metrics: QualityMetrics, issues: Vec<QualityIssue>) -> Self {
        Self {
            passed: false,
            score,
            metrics,
            issues,
            metadata: HashMap::new(),
        }
    }

    /// Check if result has critical issues
    pub fn has_critical_issues(&self) -> bool {
        self.issues.iter().any(|issue| issue.severity == SeverityLevel::Critical)
    }
}

/// Quality issue information
///
/// Information about quality issues found during quality checking
/// including type, severity, and resolution recommendations.
#[derive(Debug, Clone)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: QualityIssueType,

    /// Issue severity
    pub severity: SeverityLevel,

    /// Issue description
    pub description: String,

    /// Affected data count
    pub affected_count: usize,

    /// Resolution recommendations
    pub recommendations: Vec<String>,

    /// Issue timestamp
    pub timestamp: DateTime<Utc>,
}

impl QualityIssue {
    /// Create new quality issue
    pub fn new(
        issue_type: QualityIssueType,
        severity: SeverityLevel,
        description: String,
        affected_count: usize,
    ) -> Self {
        Self {
            issue_type,
            severity,
            description,
            affected_count,
            recommendations: Vec::new(),
            timestamp: Utc::now(),
        }
    }

    /// Add recommendation
    pub fn add_recommendation(&mut self, recommendation: String) {
        self.recommendations.push(recommendation);
    }

    /// Check if issue requires immediate attention
    pub fn requires_attention(&self) -> bool {
        matches!(self.severity, SeverityLevel::High | SeverityLevel::Critical)
    }
}

/// Quality standards for data processing
///
/// Standards and thresholds for data quality assessment and validation
/// throughout the processing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStandards {
    /// Minimum completeness threshold
    pub min_completeness: f32,

    /// Maximum staleness allowed
    pub max_staleness: Duration,

    /// Outlier detection threshold
    pub outlier_threshold: f32,

    /// Consistency requirements
    pub consistency_threshold: f32,

    /// Accuracy requirements
    pub accuracy_threshold: f32,

    /// Maximum error rate allowed
    pub max_error_rate: f32,
}

impl Default for QualityStandards {
    fn default() -> Self {
        Self {
            min_completeness: 0.9,
            max_staleness: Duration::from_secs(300),
            outlier_threshold: 2.5, // 2.5 standard deviations
            consistency_threshold: 0.95,
            accuracy_threshold: 0.95,
            max_error_rate: 0.05,
        }
    }
}

/// Quality violation information
///
/// Information about quality violations including details, impact,
/// and corrective actions taken.
#[derive(Debug, Clone)]
pub struct QualityViolation {
    /// Violation timestamp
    pub timestamp: DateTime<Utc>,

    /// Violation type
    pub violation_type: QualityIssueType,

    /// Violation severity
    pub severity: SeverityLevel,

    /// Impact assessment
    pub impact: f32,

    /// Corrective actions taken
    pub actions_taken: Vec<String>,

    /// Resolution status
    pub resolved: bool,

    /// Resolution timestamp
    pub resolution_timestamp: Option<DateTime<Utc>>,
}

impl QualityViolation {
    /// Create new quality violation
    pub fn new(violation_type: QualityIssueType, severity: SeverityLevel, impact: f32) -> Self {
        Self {
            timestamp: Utc::now(),
            violation_type,
            severity,
            impact,
            actions_taken: Vec::new(),
            resolved: false,
            resolution_timestamp: None,
        }
    }

    /// Add corrective action
    pub fn add_action(&mut self, action: String) {
        self.actions_taken.push(action);
    }

    /// Mark violation as resolved
    pub fn resolve(&mut self) {
        self.resolved = true;
        self.resolution_timestamp = Some(Utc::now());
    }

    /// Get time to resolution
    pub fn time_to_resolution(&self) -> Option<Duration> {
        self.resolution_timestamp.map(|resolution| {
            (resolution - self.timestamp).to_std().unwrap_or(Duration::from_secs(0))
        })
    }
}

/// Quality control configuration
///
/// Configuration for quality control operations including checking
/// frequency, standards enforcement, and violation handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityControlConfig {
    /// Quality checking enabled
    pub enabled: bool,

    /// Checking frequency
    pub check_frequency: Duration,

    /// Automatic correction enabled
    pub auto_correction: bool,

    /// Violation alert threshold
    pub alert_threshold: f32,

    /// Standards enforcement level
    pub enforcement_level: EnforcementLevel,

    /// Maximum violations before escalation
    pub max_violations: usize,
}

impl Default for QualityControlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_frequency: Duration::from_secs(30),
            auto_correction: true,
            alert_threshold: 0.8,
            enforcement_level: EnforcementLevel::Moderate,
            max_violations: 10,
        }
    }
}

// =============================================================================
