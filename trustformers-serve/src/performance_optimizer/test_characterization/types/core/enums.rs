//! Enumeration types for test characterization

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApplicationResult {
    /// Application succeeded
    Success,
    /// Application failed
    Failure,
    /// Partial application
    Partial,
    /// Application cancelled
    Cancelled,
    /// Application timed out
    Timeout,
    /// Application skipped
    Skipped,
    /// Application deferred
    Deferred,
    /// Application rolled back
    RolledBack,
    /// Application pending
    Pending,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplexityLevel {
    /// Very simple
    VerySimple,
    /// Simple
    Simple,
    /// Medium complexity
    Medium,
    /// Complex
    Complex,
    /// Very complex
    VeryComplex,
    /// Highly complex
    HighlyComplex,
    /// Extremely complex
    ExtremelyComplex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FunctionType {
    Pure,
    Impure,
    Async,
    Callback,
    Logarithmic,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntensityCalculationMethod {
    /// Simple moving average
    MovingAverage,
    /// Exponential weighted moving average
    ExponentialWeighted,
    /// Percentile-based calculation
    Percentile,
    /// Peak-based calculation
    Peak,
    /// Variance-based calculation
    Variance,
    /// Fourier transform based
    FourierTransform,
    /// Machine learning based
    MachineLearning,
    /// Statistical model based
    Statistical,
    /// Hybrid approach
    Hybrid,
    /// Custom algorithm
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IsolationLevel {
    /// No isolation
    None,
    /// Read uncommitted
    ReadUncommitted,
    /// Read committed
    ReadCommitted,
    /// Repeatable read
    RepeatableRead,
    /// Serializable
    Serializable,
    /// Snapshot isolation
    Snapshot,
    /// Moderate isolation
    Moderate,
    /// Custom isolation
    Custom(u8),
}

impl Default for IsolationLevel {
    fn default() -> Self {
        IsolationLevel::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectiveType {
    /// Minimize execution time
    MinimizeTime,
    /// Maximize throughput
    MaximizeThroughput,
    /// Minimize resource usage
    MinimizeResourceUsage,
    /// Maximize reliability
    MaximizeReliability,
    /// Minimize cost
    MinimizeCost,
    /// Maximize quality
    MaximizeQuality,
    /// Minimize latency
    MinimizeLatency,
    /// Maximize availability
    MaximizeAvailability,
    /// Minimize errors
    MinimizeErrors,
    /// Maximize efficiency
    MaximizeEfficiency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationResult {
    /// Operation succeeded
    Success,
    /// Operation failed
    Failure,
    /// Operation timed out
    Timeout,
    /// Operation was cancelled
    Cancelled,
    /// Operation is pending
    Pending,
    /// Operation was retried
    Retried,
    /// Operation was skipped
    Skipped,
    /// Partial success
    PartialSuccess,
    /// Unknown result
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    /// Memory allocation
    MemoryAllocation,
    /// Memory deallocation
    MemoryDeallocation,
    /// File I/O operation
    FileIo,
    /// Network operation
    NetworkOperation,
    /// Database operation
    DatabaseOperation,
    /// CPU computation
    Computation,
    /// Lock acquisition
    LockAcquisition,
    /// Lock acquire (alias for LockAcquisition)
    LockAcquire,
    /// Lock release
    LockRelease,
    /// Read operation
    Read,
    /// Write operation
    Write,
    /// Thread creation
    ThreadCreation,
    /// Thread termination
    ThreadTermination,
    /// System call
    SystemCall,
    /// API call
    ApiCall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PriorityLevel {
    /// Lowest priority
    Lowest,
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Highest priority
    Highest,
    /// Critical priority
    Critical,
    /// Urgent priority
    Urgent,
    /// Immediate priority
    Immediate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Retry operation
    Retry,
    /// Fallback to alternative
    Fallback,
    /// Skip and continue
    Skip,
    /// Fail fast
    FailFast,
    /// Rollback changes
    Rollback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResolutionType {
    /// Avoidance strategy
    Avoidance,
    /// Mitigation strategy
    Mitigation,
    /// Isolation strategy
    Isolation,
    /// Scheduling strategy
    Scheduling,
    /// Resource allocation
    ResourceAllocation,
    /// Configuration change
    ConfigurationChange,
    /// Algorithm modification
    AlgorithmModification,
    /// Infrastructure upgrade
    InfrastructureUpgrade,
    /// Process optimization
    ProcessOptimization,
    /// Manual intervention
    ManualIntervention,
    /// Serialization strategy
    Serialization,
    /// Timeout strategy
    Timeout,
    /// Optimization strategy
    Optimization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamStatus {
    /// Stream is active
    Active,
    /// Stream is inactive
    Inactive,
    /// Stream is paused
    Paused,
    /// Stream is stopped
    Stopped,
    /// Stream has error
    Error,
    /// Stream is initializing
    Initializing,
    /// Stream is terminating
    Terminating,
    /// Stream is buffering
    Buffering,
    /// Stream is draining
    Draining,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamType {
    /// Real-time stream
    RealTime,
    /// Batch stream
    Batch,
    /// Event stream
    Event,
    /// Metric stream
    Metric,
    /// Log stream
    Log,
    /// Trace stream
    Trace,
    /// Performance stream
    Performance,
    /// Resource stream
    Resource,
    /// Diagnostic stream
    Diagnostic,
    /// Custom stream
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwitchReason {
    HighLoad,
    LowLoad,
    ResourceConstraint,
    AccuracyRequirement,
    UserRequest,
    PerformanceOptimization,
}

#[derive(Debug, thiserror::Error)]
pub enum TestCharacterizationError {
    /// Configuration error
    #[error("Configuration error: {message}")]
    Configuration {
        message: String,
        context: HashMap<String, String>,
    },

    /// Resource analysis error
    #[error("Resource analysis failed: {message}")]
    ResourceAnalysis {
        message: String,
        resource_type: String,
        context: HashMap<String, String>,
    },

    /// Concurrency analysis error
    #[error("Concurrency analysis failed: {message}")]
    ConcurrencyAnalysis {
        message: String,
        test_id: String,
        context: HashMap<String, String>,
    },

    /// Pattern recognition error
    #[error("Pattern recognition failed: {message}")]
    PatternRecognition {
        message: String,
        pattern_type: String,
        context: HashMap<String, String>,
    },

    /// Profiling error
    #[error("Profiling failed: {message}")]
    Profiling {
        message: String,
        profiler_type: String,
        context: HashMap<String, String>,
    },

    /// I/O error
    #[error("I/O operation failed: {message}")]
    Io {
        message: String,
        operation: String,
        path: Option<String>,
    },

    /// Serialization error
    #[error("Serialization failed: {message}")]
    Serialization { message: String, data_type: String },

    /// Database error
    #[error("Database operation failed: {message}")]
    Database {
        message: String,
        operation: String,
        table: Option<String>,
    },

    /// Network error
    #[error("Network operation failed: {message}")]
    Network {
        message: String,
        endpoint: Option<String>,
        operation: String,
    },

    /// Timeout error
    #[error("Operation timed out: {message}")]
    Timeout {
        message: String,
        operation: String,
        timeout_duration: Duration,
    },

    /// Invalid input error
    #[error("Invalid input: {message}")]
    InvalidInput {
        message: String,
        field: String,
        value: String,
    },

    /// System resource error
    #[error("System resource error: {message}")]
    SystemResource {
        message: String,
        resource_type: String,
        available: Option<usize>,
    },

    /// Internal error
    #[error("Internal system error: {message}")]
    Internal {
        message: String,
        component: String,
        details: HashMap<String, String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TestPhase {
    /// Test setup phase
    Setup,
    /// Test execution phase
    Execution,
    /// Test teardown phase
    Teardown,
    /// Test initialization
    Initialization,
    /// Test validation
    Validation,
    /// Test cleanup
    Cleanup,
    /// Test pre-processing
    PreProcessing,
    /// Test post-processing
    PostProcessing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UrgencyLevel {
    /// No urgency
    None,
    /// Low urgency
    Low,
    /// Medium urgency
    Medium,
    /// High urgency
    High,
    /// Urgent
    Urgent,
    /// Very urgent
    VeryUrgent,
    /// Critical urgency
    Critical,
    /// Emergency
    Emergency,
}

/// Result type alias for test characterization operations
pub type TestCharacterizationResult<T> = Result<T, TestCharacterizationError>;
