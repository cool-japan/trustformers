//! Error Types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

// Import common types

use thiserror;

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

// ERROR TYPES
// =============================================================================

/// Processing error information
///
/// Information about processing errors including type, severity,
/// and recovery suggestions for robust error handling.
#[derive(Debug, Clone, thiserror::Error)]
#[error("Processing error: {message}")]
pub struct ProcessingError {
    /// Error type
    pub error_type: String,

    /// Error message
    pub message: String,

    /// Error severity
    pub severity: SeverityLevel,

    /// Recovery suggestions
    pub recovery: Vec<String>,

    /// Error timestamp
    pub timestamp: DateTime<Utc>,

    /// Error context
    pub context: HashMap<String, String>,
}

impl ProcessingError {
    /// Create new processing error
    pub fn new(error_type: String, message: String, severity: SeverityLevel) -> Self {
        Self {
            error_type,
            message,
            severity,
            recovery: Vec::new(),
            timestamp: Utc::now(),
            context: HashMap::new(),
        }
    }

    /// Add recovery suggestion
    pub fn add_recovery(&mut self, suggestion: String) {
        self.recovery.push(suggestion);
    }

    /// Add context information
    pub fn add_context(&mut self, key: String, value: String) {
        self.context.insert(key, value);
    }

    /// Check if error is critical
    pub fn is_critical(&self) -> bool {
        matches!(self.severity, SeverityLevel::High | SeverityLevel::Critical)
    }
}

/// Comprehensive error types for the real-time metrics system
#[derive(Debug, thiserror::Error)]
pub enum RealTimeMetricsError {
    /// Configuration error
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Data collection error
    #[error("Data collection error: {message}")]
    DataCollection { message: String },

    /// Processing error
    #[error("Processing error: {message}")]
    Processing { message: String },

    /// Threshold evaluation error
    #[error("Threshold evaluation error: {message}")]
    ThresholdEvaluation { message: String },

    /// Quality control error
    #[error("Quality control error: {message}")]
    QualityControl { message: String },

    /// Optimization error
    #[error("Optimization error: {message}")]
    Optimization { message: String },

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Timeout error
    #[error("Timeout error: operation timed out after {duration:?}")]
    Timeout { duration: Duration },

    /// Resource exhaustion error
    #[error("Resource exhaustion: {resource} limit exceeded")]
    ResourceExhaustion { resource: String },

    /// Invalid state error
    #[error("Invalid state: {message}")]
    InvalidState { message: String },
}

impl RealTimeMetricsError {
    /// Get error severity level
    pub fn severity(&self) -> SeverityLevel {
        match self {
            RealTimeMetricsError::Configuration { .. } => SeverityLevel::High,
            RealTimeMetricsError::DataCollection { .. } => SeverityLevel::Medium,
            RealTimeMetricsError::Processing { .. } => SeverityLevel::Medium,
            RealTimeMetricsError::ThresholdEvaluation { .. } => SeverityLevel::Low,
            RealTimeMetricsError::QualityControl { .. } => SeverityLevel::Medium,
            RealTimeMetricsError::Optimization { .. } => SeverityLevel::Low,
            RealTimeMetricsError::Io(_) => SeverityLevel::High,
            RealTimeMetricsError::Serialization(_) => SeverityLevel::Medium,
            RealTimeMetricsError::Timeout { .. } => SeverityLevel::Medium,
            RealTimeMetricsError::ResourceExhaustion { .. } => SeverityLevel::Critical,
            RealTimeMetricsError::InvalidState { .. } => SeverityLevel::High,
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        !matches!(
            self,
            RealTimeMetricsError::Configuration { .. }
                | RealTimeMetricsError::InvalidState { .. }
                | RealTimeMetricsError::ResourceExhaustion { .. }
        )
    }
}

/// Error handling policy for processing
///
/// Policy for handling errors during processing including retry strategies,
/// fallback mechanisms, and error escalation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandlingPolicy {
    /// Fail fast on any error
    FailFast,

    /// Retry with exponential backoff
    RetryWithBackoff {
        max_retries: usize,
        initial_delay: Duration,
        max_delay: Duration,
    },

    /// Continue processing with logging
    ContinueWithLogging,

    /// Circuit breaker pattern
    CircuitBreaker {
        failure_threshold: usize,
        timeout: Duration,
    },

    /// Custom error handling
    Custom(String),
}

impl Default for ErrorHandlingPolicy {
    fn default() -> Self {
        ErrorHandlingPolicy::RetryWithBackoff {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
        }
    }
}

// =============================================================================
