//! Error types for threshold monitoring and alerting.

/// Result type for threshold operations
pub type Result<T> = std::result::Result<T, ThresholdError>;

/// Errors that can occur during threshold operations
#[derive(Debug, thiserror::Error)]
pub enum ThresholdError {
    #[error("Threshold configuration error: {0}")]
    ConfigurationError(String),

    #[error("Threshold evaluation error: {0}")]
    EvaluationError(String),

    #[error("Alert processing error: {0}")]
    AlertProcessingError(String),

    #[error("Escalation error: {0}")]
    EscalationError(String),

    #[error("Notification error: {0}")]
    NotificationError(String),

    #[error("Performance monitoring error: {0}")]
    PerformanceError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}
