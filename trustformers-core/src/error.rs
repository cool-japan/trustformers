use std::time::{Duration, Instant};
use thiserror::Error;

/// Error code for categorizing errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// Shape-related errors (E1000 series)
    E1001, // Shape mismatch
    E1002, // Dimension mismatch
    E1003, // Invalid shape

    /// Tensor operation errors (E2000 series)
    E2001, // Incompatible tensor types
    E2002, // Invalid tensor operation
    E2003, // Tensor data corruption

    /// Memory errors (E3000 series)
    E3001, // Out of memory
    E3002, // Memory allocation failed
    E3003, // Memory access violation

    /// Configuration errors (E4000 series)
    E4001, // Invalid configuration
    E4002, // Missing required parameter
    E4003, // Conflicting parameters

    /// Hardware errors (E5000 series)
    E5001, // GPU unavailable
    E5002, // Unsupported operation on hardware
    E5003, // Hardware capability insufficient

    /// Performance errors (E6000 series)
    E6001, // Operation timeout
    E6002, // Performance regression detected
    E6003, // Resource limit exceeded,

    /// General errors (E9000 series)
    E9001, // Unknown error
    E9002, // Not implemented
    E9003, // Internal error
}

impl ErrorCode {
    /// Get the numeric code as a string
    pub fn code(&self) -> &'static str {
        match self {
            ErrorCode::E1001 => "E1001",
            ErrorCode::E1002 => "E1002",
            ErrorCode::E1003 => "E1003",
            ErrorCode::E2001 => "E2001",
            ErrorCode::E2002 => "E2002",
            ErrorCode::E2003 => "E2003",
            ErrorCode::E3001 => "E3001",
            ErrorCode::E3002 => "E3002",
            ErrorCode::E3003 => "E3003",
            ErrorCode::E4001 => "E4001",
            ErrorCode::E4002 => "E4002",
            ErrorCode::E4003 => "E4003",
            ErrorCode::E5001 => "E5001",
            ErrorCode::E5002 => "E5002",
            ErrorCode::E5003 => "E5003",
            ErrorCode::E6001 => "E6001",
            ErrorCode::E6002 => "E6002",
            ErrorCode::E6003 => "E6003",
            ErrorCode::E9001 => "E9001",
            ErrorCode::E9002 => "E9002",
            ErrorCode::E9003 => "E9003",
        }
    }

    /// Get helpful suggestions for resolving the error
    pub fn suggestions(&self) -> Vec<&'static str> {
        match self {
            ErrorCode::E1001 => vec![
                "Check that tensor dimensions are compatible for the operation",
                "Use reshape() to adjust tensor dimensions",
                "Verify input data shapes match expected model dimensions",
            ],
            ErrorCode::E1002 => vec![
                "Ensure tensors have the same number of dimensions",
                "Use expand_dims() or squeeze() to adjust dimensionality",
                "Check documentation for required input dimensions",
            ],
            ErrorCode::E2001 => vec![
                "Convert tensors to compatible types using to_dtype()",
                "Check if operation supports mixed precision",
                "Use explicit casting before the operation",
            ],
            ErrorCode::E3001 => vec![
                "Reduce batch size or model size",
                "Enable gradient checkpointing",
                "Use model parallelism for large models",
                "Clear unused tensors from memory",
            ],
            ErrorCode::E4001 => vec![
                "Check configuration file syntax",
                "Verify all required parameters are present",
                "Consult documentation for valid parameter values",
            ],
            ErrorCode::E5001 => vec![
                "Install appropriate GPU drivers",
                "Check CUDA/ROCm installation",
                "Fall back to CPU execution",
                "Verify GPU is not being used by another process",
            ],
            ErrorCode::E6001 => vec![
                "Increase timeout duration",
                "Use smaller batch sizes",
                "Enable operation optimization",
                "Check for infinite loops in custom operations",
            ],
            _ => vec!["Check documentation for troubleshooting guide"],
        }
    }
}

/// Enhanced error context with profiling and recovery information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Error code for categorization
    pub code: ErrorCode,
    /// Timestamp when error occurred
    pub timestamp: Instant,
    /// Operation that caused the error
    pub operation: String,
    /// Stack trace or call context
    pub context: Vec<String>,
    /// Performance metrics when error occurred
    pub performance_info: Option<PerformanceInfo>,
    /// Suggested recovery actions
    pub recovery_suggestions: Vec<String>,
}

/// Performance information captured during error
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceInfo {
    /// Memory usage at time of error
    pub memory_usage_mb: Option<u64>,
    /// Operation duration before error
    pub operation_duration: Option<Duration>,
    /// GPU utilization if available
    pub gpu_utilization: Option<f32>,
    /// CPU usage percentage
    pub cpu_usage: Option<f32>,
}

impl ErrorContext {
    pub fn new(code: ErrorCode, operation: String) -> Self {
        Self {
            code,
            timestamp: Instant::now(),
            operation,
            context: Vec::new(),
            performance_info: None,
            recovery_suggestions: code.suggestions().iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn with_context(mut self, context: String) -> Self {
        self.context.push(context);
        self
    }

    pub fn with_performance(mut self, perf: PerformanceInfo) -> Self {
        self.performance_info = Some(perf);
        self
    }

    pub fn add_recovery_suggestion(mut self, suggestion: String) -> Self {
        self.recovery_suggestions.push(suggestion);
        self
    }
}

impl PartialEq for ErrorContext {
    fn eq(&self, other: &Self) -> bool {
        self.code == other.code
            && self.operation == other.operation
            && self.context == other.context
            && self.performance_info == other.performance_info
            && self.recovery_suggestions == other.recovery_suggestions
        // Note: We deliberately ignore timestamp for equality comparison
    }
}

#[deprecated(
    since = "0.4.0",
    note = "CoreError is deprecated. Use TrustformersError from crate::errors for better error handling with rich context and suggestions."
)]
#[derive(Error, Debug)]
pub enum CoreError {
    /// Enhanced shape mismatch error with detailed context
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
        context: ErrorContext,
    },

    /// Enhanced dimension mismatch error
    #[error("Dimension mismatch")]
    DimensionMismatch { context: ErrorContext },

    /// Enhanced tensor operation error
    #[error("Tensor operation failed: {message}")]
    TensorOpError {
        message: String,
        context: ErrorContext,
    },

    /// Enhanced memory error
    #[error("Memory error: {message}")]
    MemoryError {
        message: String,
        context: ErrorContext,
    },

    /// Enhanced hardware error
    #[error("Hardware error: {message}")]
    HardwareError {
        message: String,
        context: ErrorContext,
    },

    /// Device not found error
    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    /// Enhanced configuration error
    #[error("Configuration error: {message}")]
    ConfigError {
        message: String,
        context: ErrorContext,
    },

    /// Enhanced performance error
    #[error("Performance error: {message}")]
    PerformanceError {
        message: String,
        context: ErrorContext,
    },

    // Legacy errors for backward compatibility
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Weight loading error: {0}")]
    WeightLoadError(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Shape error: {0}")]
    ShapeError(String),

    #[error("SafeTensors error: {0}")]
    SafeTensorsError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Image processing error: {0}")]
    ImageProcessingError(String),

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("IO error: {0}")]
    IoError(String),

    // ConfigError is defined above as a structured error
    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Quantization error: {0}")]
    QuantizationError(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Formatting error: {0}")]
    FormattingError(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Lock error: {0}")]
    LockError(String),

    #[error("Plugin error: {0}")]
    PluginError(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Autodiff error: {0}")]
    AutodiffError(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

#[allow(deprecated)] // Backward compatibility implementation for CoreError
impl CoreError {
    /// Create a shape mismatch error with enhanced context
    pub fn shape_mismatch(expected: Vec<usize>, got: Vec<usize>, operation: &str) -> Self {
        let context = ErrorContext::new(ErrorCode::E1001, operation.to_string())
            .with_context(format!("Expected shape {:?}, got {:?}", expected, got));

        Self::ShapeMismatch {
            expected,
            got,
            context,
        }
    }

    /// Create a dimension mismatch error with enhanced context
    pub fn dimension_mismatch(operation: &str) -> Self {
        let context = ErrorContext::new(ErrorCode::E1002, operation.to_string());
        Self::DimensionMismatch { context }
    }

    /// Create a tensor operation error with enhanced context
    pub fn tensor_op_error(message: &str, operation: &str) -> Self {
        let context = ErrorContext::new(ErrorCode::E2002, operation.to_string());
        Self::TensorOpError {
            message: message.to_string(),
            context,
        }
    }

    /// Create a memory error with enhanced context
    pub fn memory_error(message: &str, operation: &str) -> Self {
        let context = ErrorContext::new(ErrorCode::E3001, operation.to_string());
        Self::MemoryError {
            message: message.to_string(),
            context,
        }
    }

    /// Create a hardware error with enhanced context
    pub fn hardware_error(message: &str, operation: &str) -> Self {
        let context = ErrorContext::new(ErrorCode::E5001, operation.to_string());
        Self::HardwareError {
            message: message.to_string(),
            context,
        }
    }

    /// Create a configuration error with enhanced context
    pub fn config_error(message: &str, operation: &str) -> Self {
        let context = ErrorContext::new(ErrorCode::E4001, operation.to_string());
        Self::ConfigError {
            message: message.to_string(),
            context,
        }
    }

    /// Create a performance error with enhanced context
    pub fn performance_error(message: &str, operation: &str) -> Self {
        let context = ErrorContext::new(ErrorCode::E6001, operation.to_string());
        Self::PerformanceError {
            message: message.to_string(),
            context,
        }
    }

    /// Get the error code if available
    pub fn error_code(&self) -> Option<ErrorCode> {
        match self {
            Self::ShapeMismatch { context, .. } => Some(context.code),
            Self::DimensionMismatch { context, .. } => Some(context.code),
            Self::TensorOpError { context, .. } => Some(context.code),
            Self::MemoryError { context, .. } => Some(context.code),
            Self::HardwareError { context, .. } => Some(context.code),
            Self::ConfigError { context, .. } => Some(context.code),
            Self::PerformanceError { context, .. } => Some(context.code),
            _ => None,
        }
    }

    /// Get recovery suggestions if available
    pub fn recovery_suggestions(&self) -> Option<&Vec<String>> {
        match self {
            Self::ShapeMismatch { context, .. } => Some(&context.recovery_suggestions),
            Self::DimensionMismatch { context, .. } => Some(&context.recovery_suggestions),
            Self::TensorOpError { context, .. } => Some(&context.recovery_suggestions),
            Self::MemoryError { context, .. } => Some(&context.recovery_suggestions),
            Self::HardwareError { context, .. } => Some(&context.recovery_suggestions),
            Self::ConfigError { context, .. } => Some(&context.recovery_suggestions),
            Self::PerformanceError { context, .. } => Some(&context.recovery_suggestions),
            _ => None,
        }
    }

    /// Get performance information if available
    pub fn performance_info(&self) -> Option<&PerformanceInfo> {
        match self {
            Self::ShapeMismatch { context, .. } => context.performance_info.as_ref(),
            Self::DimensionMismatch { context, .. } => context.performance_info.as_ref(),
            Self::TensorOpError { context, .. } => context.performance_info.as_ref(),
            Self::MemoryError { context, .. } => context.performance_info.as_ref(),
            Self::HardwareError { context, .. } => context.performance_info.as_ref(),
            Self::ConfigError { context, .. } => context.performance_info.as_ref(),
            Self::PerformanceError { context, .. } => context.performance_info.as_ref(),
            _ => None,
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::ShapeMismatch { .. } => true, // Can be fixed by reshaping
            Self::DimensionMismatch { .. } => true, // Can be fixed by dimension adjustment
            Self::TensorOpError { .. } => false, // Usually indicates fundamental issue
            Self::MemoryError { .. } => true,   // Can be fixed by freeing memory
            Self::HardwareError { .. } => true, // Can fallback to CPU
            Self::ConfigError { .. } => true,   // Can be fixed by config adjustment
            Self::PerformanceError { .. } => true, // Can be fixed by optimization
            _ => false,
        }
    }

    /// Attempt to recover from the error automatically
    pub fn try_recover(&self) -> std::result::Result<RecoveryAction, Self> {
        match self {
            Self::MemoryError { .. } => Ok(RecoveryAction::ClearMemory),
            Self::HardwareError { .. } => Ok(RecoveryAction::FallbackToCpu),
            Self::ShapeMismatch {
                expected,
                got,
                context,
            } => {
                if can_reshape(got, expected) {
                    Ok(RecoveryAction::Reshape(expected.clone()))
                } else {
                    Err(Self::ShapeMismatch {
                        expected: expected.clone(),
                        got: got.clone(),
                        context: context.clone(),
                    })
                }
            },
            _ => Err(Self::tensor_op_error(
                "Recovery not supported for this error type",
                "try_recover",
            )),
        }
    }
}

/// Recovery actions that can be taken automatically
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Clear memory and retry
    ClearMemory,
    /// Fallback to CPU execution
    FallbackToCpu,
    /// Reshape tensor to expected dimensions
    Reshape(Vec<usize>),
    /// Reduce batch size
    ReduceBatchSize(usize),
    /// Enable gradient checkpointing
    EnableCheckpointing,
}

/// Check if a tensor can be reshaped from one shape to another
fn can_reshape(from: &[usize], to: &[usize]) -> bool {
    let from_size: usize = from.iter().product();
    let to_size: usize = to.iter().product();
    from_size == to_size
}

// Removed manual Display implementation - using derive(Error) instead
// impl std::fmt::Display for CoreError {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
/*        match self {
            Self::ShapeMismatch { expected, got, context } => {
                write!(f, "[{}] Shape mismatch in '{}': expected {:?}, got {:?}",
                       context.code.code(), context.operation, expected, got)?;
                if !context.recovery_suggestions.is_empty() {
                    write!(f, "\n\nSuggestions:")?;
                    for suggestion in &context.recovery_suggestions {
                        write!(f, "\n  • {}", suggestion)?;
                    }
                }
                if let Some(perf) = &context.performance_info {
                    write!(f, "\n\nPerformance Info:")?;
                    if let Some(mem) = perf.memory_usage_mb {
                        write!(f, "\n  Memory usage: {} MB", mem)?;
                    }
                    if let Some(dur) = perf.operation_duration {
                        write!(f, "\n  Operation duration: {:?}", dur)?;
                    }
                }
                Ok(())
            }
            Self::DimensionMismatch { context } => {
                write!(f, "[{}] Dimension mismatch in '{}'",
                       context.code.code(), context.operation)?;
                if !context.recovery_suggestions.is_empty() {
                    write!(f, "\n\nSuggestions:")?;
                    for suggestion in &context.recovery_suggestions {
                        write!(f, "\n  • {}", suggestion)?;
                    }
                }
                Ok(())
            }
            Self::TensorOpError { message, context } => {
                write!(f, "[{}] Tensor operation failed in '{}': {}",
                       context.code.code(), context.operation, message)?;
                if !context.recovery_suggestions.is_empty() {
                    write!(f, "\n\nSuggestions:")?;
                    for suggestion in &context.recovery_suggestions {
                        write!(f, "\n  • {}", suggestion)?;
                    }
                }
                Ok(())
            }
            Self::MemoryError { message, context } => {
                write!(f, "[{}] Memory error in '{}': {}",
                       context.code.code(), context.operation, message)?;
                if !context.recovery_suggestions.is_empty() {
                    write!(f, "\n\nSuggestions:")?;
                    for suggestion in &context.recovery_suggestions {
                        write!(f, "\n  • {}", suggestion)?;
                    }
                }
                if let Some(perf) = &context.performance_info {
                    if let Some(mem) = perf.memory_usage_mb {
                        write!(f, "\n\nMemory usage at error: {} MB", mem)?;
                    }
                }
                Ok(())
            }
            Self::HardwareError { message, context } => {
                write!(f, "[{}] Hardware error in '{}': {}",
                       context.code.code(), context.operation, message)?;
                if !context.recovery_suggestions.is_empty() {
                    write!(f, "\n\nSuggestions:")?;
                    for suggestion in &context.recovery_suggestions {
                        write!(f, "\n  • {}", suggestion)?;
                    }
                }
                Ok(())
            }
            Self::ConfigError { message, context } => {
                write!(f, "[{}] Configuration error in '{}': {}",
                       context.code.code(), context.operation, message)?;
                if !context.recovery_suggestions.is_empty() {
                    write!(f, "\n\nSuggestions:")?;
                    for suggestion in &context.recovery_suggestions {
                        write!(f, "\n  • {}", suggestion)?;
                    }
                }
                Ok(())
            }
            Self::PerformanceError { message, context } => {
                write!(f, "[{}] Performance error in '{}': {}",
                       context.code.code(), context.operation, message)?;
                if !context.recovery_suggestions.is_empty() {
                    write!(f, "\n\nSuggestions:")?;
                    for suggestion in &context.recovery_suggestions {
                        write!(f, "\n  • {}", suggestion)?;
                    }
                }
                if let Some(perf) = &context.performance_info {
                    write!(f, "\n\nPerformance Info:")?;
                    if let Some(dur) = perf.operation_duration {
                        write!(f, "\n  Operation duration: {:?}", dur)?;
                    }
                    if let Some(cpu) = perf.cpu_usage {
                        write!(f, "\n  CPU usage: {:.1}%", cpu)?;
                    }
                    if let Some(gpu) = perf.gpu_utilization {
                        write!(f, "\n  GPU utilization: {:.1}%", gpu)?;
                    }
                }
                Ok(())
            }
            // Fall back to default Error display for legacy errors
            _ => write!(f, "{}", self),
        }
    }
} */

#[allow(deprecated)] // Backward compatibility implementation for CoreError
impl From<std::fmt::Error> for CoreError {
    fn from(err: std::fmt::Error) -> Self {
        CoreError::FormattingError(err.to_string())
    }
}

#[allow(deprecated)] // Backward compatibility implementation for CoreError
impl From<ndarray::ShapeError> for CoreError {
    fn from(err: ndarray::ShapeError) -> Self {
        CoreError::ShapeError(err.to_string())
    }
}

#[allow(deprecated)] // Backward compatibility implementation for CoreError
impl Clone for CoreError {
    fn clone(&self) -> Self {
        match self {
            CoreError::ShapeMismatch {
                expected,
                got,
                context,
            } => CoreError::ShapeMismatch {
                expected: expected.clone(),
                got: got.clone(),
                context: context.clone(),
            },
            CoreError::DimensionMismatch { context } => CoreError::DimensionMismatch {
                context: context.clone(),
            },
            CoreError::InvalidArgument(msg) => CoreError::InvalidArgument(msg.clone()),
            CoreError::InvalidConfig(msg) => CoreError::InvalidConfig(msg.clone()),
            CoreError::NotImplemented(msg) => CoreError::NotImplemented(msg.clone()),
            CoreError::TensorOpError { message, context } => CoreError::TensorOpError {
                message: message.clone(),
                context: context.clone(),
            },
            CoreError::MemoryError { message, context } => CoreError::MemoryError {
                message: message.clone(),
                context: context.clone(),
            },
            CoreError::HardwareError { message, context } => CoreError::HardwareError {
                message: message.clone(),
                context: context.clone(),
            },
            CoreError::ConfigError { message, context } => CoreError::ConfigError {
                message: message.clone(),
                context: context.clone(),
            },
            CoreError::PerformanceError { message, context } => CoreError::PerformanceError {
                message: message.clone(),
                context: context.clone(),
            },
            CoreError::WeightLoadError(msg) => CoreError::WeightLoadError(msg.clone()),
            CoreError::ShapeError(msg) => CoreError::ShapeError(msg.clone()),
            // For non-cloneable types, create new instances with the string representation
            CoreError::Io(io_err) => CoreError::InvalidConfig(format!("IO Error: {}", io_err)),
            CoreError::Serialization(serde_err) => {
                CoreError::InvalidConfig(format!("Serialization Error: {}", serde_err))
            },
            CoreError::Other(anyhow_err) => {
                CoreError::InvalidConfig(format!("Other Error: {}", anyhow_err))
            },
            // Handle all other string-based variants
            _ => {
                // For any other variant, convert to string and wrap as InvalidConfig
                CoreError::InvalidConfig(format!("{}", self))
            },
        }
    }
}

#[deprecated(
    since = "0.4.0",
    note = "Use Result from crate::errors instead for standardized error handling"
)]
#[allow(deprecated)]
pub type Result<T> = std::result::Result<T, CoreError>;
