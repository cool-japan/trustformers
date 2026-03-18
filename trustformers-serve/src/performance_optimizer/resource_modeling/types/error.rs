//! Error types for resource modeling
//!
//! Comprehensive error types for resource modeling operations including
//! detection, profiling, monitoring, and analysis errors.

use std::time::Duration;

/// Resource modeling error types
///
/// Comprehensive error types for resource modeling operations
/// including detection, profiling, monitoring, and analysis errors.
#[derive(Debug, thiserror::Error)]
pub enum ResourceModelingError {
    /// Hardware detection error
    #[error("Hardware detection failed: {0}")]
    HardwareDetection(String),

    /// Performance profiling error
    #[error("Performance profiling failed: {0}")]
    PerformanceProfiling(String),

    /// Temperature monitoring error
    #[error("Temperature monitoring failed: {0}")]
    TemperatureMonitoring(String),

    /// Topology analysis error
    #[error("Topology analysis failed: {0}")]
    TopologyAnalysis(String),

    /// Utilization tracking error
    #[error("Utilization tracking failed: {0}")]
    UtilizationTracking(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Cache operation error
    #[error("Cache operation failed: {0}")]
    CacheOperation(String),

    /// Vendor detection error
    #[error("Vendor-specific detection failed: {vendor}, error: {error}")]
    VendorDetection { vendor: String, error: String },

    /// Benchmark execution error
    #[error("Benchmark execution failed: {0}")]
    BenchmarkExecution(String),

    /// System resource error
    #[error("System resource error: {0}")]
    SystemResource(String),
}

/// Detection operation errors
///
/// Specific errors related to hardware detection operations
/// including timeout, access, and capability errors.
#[derive(Debug, thiserror::Error)]
pub enum DetectionError {
    /// Timeout during detection
    #[error("Detection timeout after {duration:?}")]
    Timeout { duration: Duration },

    /// Access denied to hardware
    #[error("Access denied to hardware resource: {resource}")]
    AccessDenied { resource: String },

    /// Unsupported hardware
    #[error("Unsupported hardware: {hardware}")]
    UnsupportedHardware { hardware: String },

    /// Invalid detection result
    #[error("Invalid detection result: {reason}")]
    InvalidResult { reason: String },

    /// Missing required capability
    #[error("Missing required capability: {capability}")]
    MissingCapability { capability: String },
}

/// Profiling operation errors
///
/// Specific errors related to performance profiling operations
/// including benchmark failures, measurement errors, and validation issues.
#[derive(Debug, thiserror::Error)]
pub enum ProfilingError {
    /// Benchmark execution failed
    #[error("Benchmark {benchmark} failed: {reason}")]
    BenchmarkFailed { benchmark: String, reason: String },

    /// Measurement validation failed
    #[error("Measurement validation failed: {reason}")]
    ValidationFailed { reason: String },

    /// Insufficient data for profiling
    #[error("Insufficient data for profiling: {component}")]
    InsufficientData { component: String },

    /// Profiling timeout
    #[error("Profiling timeout for {component} after {duration:?}")]
    ProfilingTimeout {
        component: String,
        duration: Duration,
    },

    /// Resource unavailable for profiling
    #[error("Resource unavailable for profiling: {resource}")]
    ResourceUnavailable { resource: String },
}

/// Monitoring operation errors
///
/// Specific errors related to monitoring operations including
/// sensor access, data collection, and threshold violations.
#[derive(Debug, thiserror::Error)]
pub enum MonitoringError {
    /// Sensor access failed
    #[error("Sensor access failed: {sensor}")]
    SensorAccess { sensor: String },

    /// Data collection error
    #[error("Data collection error: {reason}")]
    DataCollection { reason: String },

    /// Threshold violation
    #[error("Threshold violation: {metric} = {value}, threshold = {threshold}")]
    ThresholdViolation {
        metric: String,
        value: f32,
        threshold: f32,
    },

    /// Monitoring initialization failed
    #[error("Monitoring initialization failed: {reason}")]
    InitializationFailed { reason: String },

    /// Alert generation failed
    #[error("Alert generation failed: {reason}")]
    AlertFailed { reason: String },
}

/// Analysis operation errors
///
/// Specific errors related to topology and performance analysis
/// including calculation errors, data inconsistency, and analysis failures.
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    /// Calculation error
    #[error("Analysis calculation error: {calculation}")]
    CalculationError { calculation: String },

    /// Data inconsistency
    #[error("Data inconsistency detected: {description}")]
    DataInconsistency { description: String },

    /// Analysis convergence failed
    #[error("Analysis failed to converge: {analysis}")]
    ConvergenceFailed { analysis: String },

    /// Insufficient analysis data
    #[error("Insufficient data for analysis: {analysis}")]
    InsufficientData { analysis: String },

    /// Analysis timeout
    #[error("Analysis timeout: {analysis} after {duration:?}")]
    AnalysisTimeout {
        analysis: String,
        duration: Duration,
    },
}
