//! Enhanced error handling with contextual information and recovery suggestions
//!
//! This module provides comprehensive error types with rich context and actionable
//! recovery suggestions to improve developer experience.

use std::fmt;
use thiserror::Error;

mod conversions;
mod standardization;

pub use conversions::{
    acceleration_error, checkpoint_error, compute_error, dimension_mismatch, file_not_found,
    hardware_error, invalid_config, invalid_format, invalid_input, memory_error,
    model_compatibility_error, model_not_found, not_implemented, out_of_memory, performance_error,
    quantization_error, resource_exhausted, runtime_error, shape_mismatch, tensor_op_error,
    timed_error, timeout_error, unsupported_operation, ResultExt, TimedResultExt,
};

pub use standardization::{ErrorMigrationHelper, ResultStandardization, StandardError};

/// Core error type with context and recovery suggestions
#[derive(Debug, Error)]
pub struct TrustformersError {
    /// The underlying error kind
    #[source]
    pub kind: ErrorKind,

    /// Contextual information about where the error occurred
    pub context: ErrorContext,

    /// Suggested recovery actions
    pub suggestions: Vec<String>,

    /// Error code for documentation lookup
    pub code: ErrorCode,
}

impl TrustformersError {
    /// Create a new error with context
    pub fn new(kind: ErrorKind) -> Self {
        let code = ErrorCode::from_kind(&kind);
        let suggestions = Self::default_suggestions(&kind);

        Self {
            kind,
            context: ErrorContext::default(),
            suggestions,
            code,
        }
    }

    /// Add contextual information
    pub fn with_context(mut self, key: &str, value: String) -> Self {
        self.context.add(key, value);
        self
    }

    /// Add a recovery suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Set the operation that failed
    pub fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.context.operation = Some(operation.into());
        self
    }

    /// Set the model or component involved
    pub fn with_component(mut self, component: impl Into<String>) -> Self {
        self.context.component = Some(component.into());
        self
    }

    /// Get default suggestions based on error kind
    fn default_suggestions(kind: &ErrorKind) -> Vec<String> {
        match kind {
            ErrorKind::DimensionMismatch { expected, actual } => vec![
                format!(
                    "Check that input tensors have shape {}, not {}",
                    expected, actual
                ),
                "Verify the model configuration matches your input dimensions".to_string(),
                "Use .view() or .reshape() to adjust tensor dimensions".to_string(),
            ],

            ErrorKind::ShapeMismatch { expected, actual } => vec![
                format!(
                    "Check that tensor shapes match: expected {:?}, got {:?}",
                    expected, actual
                ),
                "Use .reshape() to adjust tensor dimensions".to_string(),
                "Verify input data shapes match expected model dimensions".to_string(),
                "Consider using broadcasting operations if appropriate".to_string(),
            ],

            ErrorKind::OutOfMemory {
                required,
                available: _available,
            } => vec![
                "Try reducing batch size".to_string(),
                "Enable gradient checkpointing to trade compute for memory".to_string(),
                "Use mixed precision training (fp16/bf16) to reduce memory usage".to_string(),
                format!(
                    "Consider using model parallelism if model requires >{}GB",
                    required / 1_000_000_000
                ),
            ],

            ErrorKind::InvalidConfiguration { field, reason } => vec![
                format!("Check the '{}' field in your configuration", field),
                format!("Reason: {}", reason),
                "Refer to the model's configuration documentation".to_string(),
                "Use Model::from_pretrained() for validated configurations".to_string(),
            ],

            ErrorKind::ModelNotFound { name } => vec![
                format!("Verify the model name '{}' is correct", name),
                "Check available models with Model::list_available()".to_string(),
                "Ensure you have internet connectivity for downloading".to_string(),
                "Try specifying a revision if the model was recently updated".to_string(),
            ],

            ErrorKind::QuantizationError { reason } => vec![
                "Ensure the model supports the requested quantization type".to_string(),
                format!("Issue: {}", reason),
                "Try a different quantization method (int8, int4, gptq, awq)".to_string(),
                "Check if calibration data is required for this quantization".to_string(),
            ],

            ErrorKind::DeviceError { device, reason } => vec![
                format!("Check that {} is available and properly configured", device),
                format!("Error: {}", reason),
                "Try running on CPU as a fallback".to_string(),
                "Verify driver installation and versions".to_string(),
            ],

            ErrorKind::SerializationError { format, reason } => vec![
                format!("Check the {} file format", format),
                format!("Issue: {}", reason),
                "Ensure the file is not corrupted".to_string(),
                "Try converting to a different format".to_string(),
            ],

            ErrorKind::ComputeError { operation, reason } => vec![
                format!("The {} operation failed: {}", operation, reason),
                "Check for numerical instability (NaN/Inf values)".to_string(),
                "Try using different precision (fp32 instead of fp16)".to_string(),
                "Enable debug mode for detailed tensor information".to_string(),
            ],

            ErrorKind::TensorOpError { operation, reason } => vec![
                format!("Tensor operation '{}' failed: {}", operation, reason),
                "Check tensor dimensions and compatibility".to_string(),
                "Verify data types are compatible".to_string(),
                "Enable tensor debugging to see intermediate values".to_string(),
            ],

            ErrorKind::MemoryError { reason } => vec![
                format!("Memory operation failed: {}", reason),
                "Try reducing memory usage by clearing unused tensors".to_string(),
                "Enable memory optimization settings".to_string(),
                "Consider using CPU offloading for large tensors".to_string(),
            ],

            ErrorKind::HardwareError { device, reason } => vec![
                format!("Hardware error on {}: {}", device, reason),
                "Check device drivers and installation".to_string(),
                "Verify hardware is properly connected".to_string(),
                "Try falling back to CPU execution".to_string(),
            ],

            ErrorKind::PerformanceError { reason } => vec![
                format!("Performance issue: {}", reason),
                "Try optimizing batch size or model parameters".to_string(),
                "Enable performance profiling to identify bottlenecks".to_string(),
                "Consider using more efficient operations".to_string(),
            ],

            ErrorKind::InvalidInput { reason } => vec![
                format!("Invalid input: {}", reason),
                "Check input data format and types".to_string(),
                "Verify input shapes match model expectations".to_string(),
                "Ensure input data is properly preprocessed".to_string(),
            ],

            ErrorKind::RuntimeError { reason } => vec![
                format!("Runtime error: {}", reason),
                "Check system resources and dependencies".to_string(),
                "Verify configuration settings".to_string(),
                "Try restarting the operation".to_string(),
            ],

            ErrorKind::ResourceExhausted { resource, reason } => vec![
                format!("Resource '{}' exhausted: {}", resource, reason),
                "Reduce resource usage by optimizing operations".to_string(),
                "Consider using resource pooling or management".to_string(),
                "Check system resource limits".to_string(),
            ],

            ErrorKind::TimeoutError {
                operation,
                timeout_ms,
            } => vec![
                format!("Operation '{}' timed out after {}ms", operation, timeout_ms),
                "Increase timeout duration if operation is expected to take longer".to_string(),
                "Optimize the operation for better performance".to_string(),
                "Check for deadlocks or infinite loops".to_string(),
            ],

            ErrorKind::FileNotFound { path } => vec![
                format!("File not found: {}", path),
                "Check that the file path is correct".to_string(),
                "Verify file permissions".to_string(),
                "Ensure the file exists in the expected location".to_string(),
            ],

            ErrorKind::InvalidFormat { expected, actual } => vec![
                format!("Invalid format: expected {}, got {}", expected, actual),
                "Check the file format and conversion requirements".to_string(),
                "Verify the data is in the expected format".to_string(),
                "Try using format conversion utilities".to_string(),
            ],

            ErrorKind::UnsupportedOperation { operation, target } => vec![
                format!("Operation '{}' not supported on {}", operation, target),
                "Check if the operation is available for this target".to_string(),
                "Try using an alternative operation or target".to_string(),
                "Verify feature compatibility".to_string(),
            ],

            ErrorKind::NotImplemented { feature } => vec![
                format!("Feature '{}' is not yet implemented", feature),
                "Check the roadmap for planned features".to_string(),
                "Consider using alternative approaches".to_string(),
                "Submit a feature request if needed".to_string(),
            ],

            ErrorKind::AutodiffError { reason } => vec![
                format!("Automatic differentiation failed: {}", reason),
                "Check that all operations support gradient computation".to_string(),
                "Verify the computational graph is correctly built".to_string(),
                "Enable gradient checking to validate gradients".to_string(),
            ],

            _ => vec!["Check the error details and context for more information".to_string()],
        }
    }

    // Convenience methods for common error patterns
    pub fn hardware_error(message: &str, operation: &str) -> Self {
        TrustformersError::new(ErrorKind::HardwareError {
            device: "unknown".to_string(),
            reason: message.to_string(),
        })
        .with_operation(operation)
    }

    pub fn tensor_op_error(message: &str, operation: &str) -> Self {
        TrustformersError::new(ErrorKind::TensorOpError {
            operation: operation.to_string(),
            reason: message.to_string(),
        })
        .with_operation(operation)
    }

    pub fn autodiff_error(message: String) -> Self {
        TrustformersError::new(ErrorKind::AutodiffError { reason: message })
    }

    pub fn invalid_input(message: String) -> Self {
        TrustformersError::new(ErrorKind::InvalidInput { reason: message })
    }

    pub fn config_error(message: &str, field: &str) -> Self {
        TrustformersError::new(ErrorKind::InvalidConfiguration {
            field: field.to_string(),
            reason: message.to_string(),
        })
    }

    pub fn invalid_config(message: String) -> Self {
        TrustformersError::new(ErrorKind::InvalidConfiguration {
            field: "config".to_string(),
            reason: message,
        })
    }

    pub fn model_error(message: String) -> Self {
        TrustformersError::new(ErrorKind::ModelNotFound { name: message })
    }

    pub fn weight_load_error(message: String) -> Self {
        TrustformersError::new(ErrorKind::WeightLoadingError { reason: message })
    }

    pub fn runtime_error(message: String) -> Self {
        TrustformersError::new(ErrorKind::RuntimeError { reason: message })
    }

    pub fn io_error(message: String) -> Self {
        TrustformersError::new(ErrorKind::IoError(std::io::Error::new(
            std::io::ErrorKind::Other,
            message,
        )))
    }

    pub fn shape_error(message: String) -> Self {
        TrustformersError::new(ErrorKind::ShapeError { reason: message })
    }

    pub fn safe_tensors_error(message: String) -> Self {
        TrustformersError::new(ErrorKind::SafeTensorsError { reason: message })
    }

    pub fn dimension_mismatch(expected: String, actual: String) -> Self {
        TrustformersError::new(ErrorKind::DimensionMismatch { expected, actual })
    }

    pub fn invalid_format(expected: String, actual: String) -> Self {
        TrustformersError::new(ErrorKind::InvalidFormat { expected, actual })
    }

    pub fn invalid_format_simple(message: String) -> Self {
        TrustformersError::new(ErrorKind::InvalidFormat {
            expected: "valid format".to_string(),
            actual: message,
        })
    }

    pub fn not_implemented(feature: String) -> Self {
        TrustformersError::new(ErrorKind::NotImplemented { feature })
    }

    pub fn invalid_input_simple(reason: String) -> Self {
        TrustformersError::new(ErrorKind::InvalidInput { reason })
    }

    pub fn invalid_state(reason: String) -> Self {
        TrustformersError::new(ErrorKind::InvalidState { reason })
    }

    pub fn invalid_operation(message: String) -> Self {
        TrustformersError::new(ErrorKind::InvalidInput { reason: message })
    }

    pub fn other(message: String) -> Self {
        TrustformersError::new(ErrorKind::Other(message))
    }

    pub fn resource_exhausted(message: String) -> Self {
        TrustformersError::new(ErrorKind::ResourceExhausted {
            resource: "memory".to_string(),
            reason: message,
        })
    }

    pub fn lock_error(message: String) -> Self {
        TrustformersError::new(ErrorKind::Other(format!("Lock error: {}", message)))
    }

    pub fn serialization_error(message: String) -> Self {
        TrustformersError::new(ErrorKind::SerializationError {
            format: "unknown".to_string(),
            reason: message,
        })
    }

    pub fn plugin_error(message: String) -> Self {
        TrustformersError::new(ErrorKind::Other(format!("Plugin error: {}", message)))
    }

    pub fn quantization_error(message: String) -> Self {
        TrustformersError::new(ErrorKind::Other(format!("Quantization error: {}", message)))
    }

    pub fn invalid_argument(message: String) -> Self {
        TrustformersError::new(ErrorKind::InvalidInput { reason: message })
    }

    pub fn file_not_found(message: String) -> Self {
        TrustformersError::new(ErrorKind::FileNotFound { path: message })
    }
}

impl fmt::Display for TrustformersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header with error code
        writeln!(f, "\n❌ Error [{}]", self.code)?;
        writeln!(f, "{}", "─".repeat(60))?;

        // Error description
        writeln!(f, "📍 {}", self.kind)?;

        // Context information
        if self.context.has_info() {
            writeln!(f, "\n📋 Context:")?;
            if let Some(op) = &self.context.operation {
                writeln!(f, "   Operation: {}", op)?;
            }
            if let Some(comp) = &self.context.component {
                writeln!(f, "   Component: {}", comp)?;
            }
            for (key, value) in &self.context.info {
                writeln!(f, "   {}: {}", key, value)?;
            }
        }

        // Recovery suggestions
        if !self.suggestions.is_empty() {
            writeln!(f, "\n💡 Suggestions:")?;
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                writeln!(f, "   {}. {}", i + 1, suggestion)?;
            }
        }

        // Documentation link
        writeln!(
            f,
            "\n📚 For more information, see: https://docs.trustformers.ai/errors/{}",
            self.code
        )?;
        writeln!(f, "{}", "─".repeat(60))?;

        Ok(())
    }
}

/// Specific error kinds
#[derive(Debug, Error)]
pub enum ErrorKind {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Out of memory: required {required} bytes, available {available} bytes")]
    OutOfMemory { required: usize, available: usize },

    #[error("Invalid configuration: field '{field}' {reason}")]
    InvalidConfiguration { field: String, reason: String },

    #[error("Model not found: '{name}'")]
    ModelNotFound { name: String },

    #[error("Weight loading failed: {reason}")]
    WeightLoadingError { reason: String },

    #[error("Tokenization error: {reason}")]
    TokenizationError { reason: String },

    #[error("Quantization error: {reason}")]
    QuantizationError { reason: String },

    #[error("Device error on {device}: {reason}")]
    DeviceError { device: String, reason: String },

    #[error("Serialization error for {format}: {reason}")]
    SerializationError { format: String, reason: String },

    #[error("Compute error in {operation}: {reason}")]
    ComputeError { operation: String, reason: String },

    #[error("Training error: {reason}")]
    TrainingError { reason: String },

    #[error("Pipeline error: {reason}")]
    PipelineError { reason: String },

    #[error("Attention error: {reason}")]
    AttentionError { reason: String },

    #[error("Optimization error: {reason}")]
    OptimizationError { reason: String },

    #[error("Autodiff error: {reason}")]
    AutodiffError { reason: String },

    #[error("Tensor operation error: {operation} failed with {reason}")]
    TensorOpError { operation: String, reason: String },

    #[error("Memory allocation error: {reason}")]
    MemoryError { reason: String },

    #[error("Hardware error: {device} - {reason}")]
    HardwareError { device: String, reason: String },

    #[error("Performance error: {reason}")]
    PerformanceError { reason: String },

    #[error("Invalid input: {reason}")]
    InvalidInput { reason: String },

    #[error("Image processing error: {reason}")]
    ImageProcessingError { reason: String },

    #[error("Runtime error: {reason}")]
    RuntimeError { reason: String },

    #[error("Resource exhausted: {resource} - {reason}")]
    ResourceExhausted { resource: String, reason: String },

    #[error("Plugin error: {plugin} - {reason}")]
    PluginError { plugin: String, reason: String },

    #[error("Timeout error: operation '{operation}' exceeded {timeout_ms}ms")]
    TimeoutError { operation: String, timeout_ms: u64 },

    #[error("Network error: {reason}")]
    NetworkError { reason: String },

    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error("Invalid format: expected {expected}, got {actual}")]
    InvalidFormat { expected: String, actual: String },

    #[error("Invalid state: {reason}")]
    InvalidState { reason: String },

    #[error("Unsupported operation: {operation} on {target}")]
    UnsupportedOperation { operation: String, target: String },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Not implemented: {feature}")]
    NotImplemented { feature: String },

    #[error("Shape error: {reason}")]
    ShapeError { reason: String },

    #[error("SafeTensors error: {reason}")]
    SafeTensorsError { reason: String },

    #[error("Other error: {0}")]
    Other(String),
}

/// Error context information
#[derive(Debug, Default)]
pub struct ErrorContext {
    /// The operation being performed
    pub operation: Option<String>,

    /// The component or model involved
    pub component: Option<String>,

    /// Additional key-value information
    pub info: Vec<(String, String)>,
}

impl ErrorContext {
    /// Add contextual information
    pub fn add(&mut self, key: &str, value: String) {
        self.info.push((key.to_string(), value));
    }

    /// Check if context has any information
    pub fn has_info(&self) -> bool {
        self.operation.is_some() || self.component.is_some() || !self.info.is_empty()
    }
}

/// Error codes for documentation
#[derive(Debug, Clone, Copy)]
pub enum ErrorCode {
    E0001, // DimensionMismatch
    E0002, // ShapeMismatch
    E0003, // OutOfMemory
    E0004, // InvalidConfiguration
    E0005, // ModelNotFound
    E0006, // WeightLoadingError
    E0007, // TokenizationError
    E0008, // QuantizationError
    E0009, // DeviceError
    E0010, // SerializationError
    E0011, // ComputeError
    E0012, // TrainingError
    E0013, // PipelineError
    E0014, // AttentionError
    E0015, // OptimizationError
    E0016, // TensorOpError
    E0017, // MemoryError
    E0018, // HardwareError
    E0019, // PerformanceError
    E0020, // InvalidInput
    E0021, // ImageProcessingError
    E0022, // RuntimeError
    E0023, // ResourceExhausted
    E0024, // PluginError
    E0025, // TimeoutError
    E0026, // NetworkError
    E0027, // FileNotFound
    E0028, // InvalidFormat
    E0029, // InvalidState
    E0030, // UnsupportedOperation
    E0031, // IoError
    E0032, // NotImplemented
    E0033, // AutodiffError
    E9999, // Other
}

impl ErrorCode {
    /// Get error code from error kind
    pub fn from_kind(kind: &ErrorKind) -> Self {
        match kind {
            ErrorKind::DimensionMismatch { .. } => ErrorCode::E0001,
            ErrorKind::ShapeMismatch { .. } => ErrorCode::E0002,
            ErrorKind::OutOfMemory { .. } => ErrorCode::E0003,
            ErrorKind::InvalidConfiguration { .. } => ErrorCode::E0004,
            ErrorKind::ModelNotFound { .. } => ErrorCode::E0005,
            ErrorKind::WeightLoadingError { .. } => ErrorCode::E0006,
            ErrorKind::TokenizationError { .. } => ErrorCode::E0007,
            ErrorKind::QuantizationError { .. } => ErrorCode::E0008,
            ErrorKind::DeviceError { .. } => ErrorCode::E0009,
            ErrorKind::SerializationError { .. } => ErrorCode::E0010,
            ErrorKind::ComputeError { .. } => ErrorCode::E0011,
            ErrorKind::TrainingError { .. } => ErrorCode::E0012,
            ErrorKind::PipelineError { .. } => ErrorCode::E0013,
            ErrorKind::AttentionError { .. } => ErrorCode::E0014,
            ErrorKind::OptimizationError { .. } => ErrorCode::E0015,
            ErrorKind::AutodiffError { .. } => ErrorCode::E0033,
            ErrorKind::TensorOpError { .. } => ErrorCode::E0016,
            ErrorKind::MemoryError { .. } => ErrorCode::E0017,
            ErrorKind::HardwareError { .. } => ErrorCode::E0018,
            ErrorKind::PerformanceError { .. } => ErrorCode::E0019,
            ErrorKind::InvalidInput { .. } => ErrorCode::E0020,
            ErrorKind::ImageProcessingError { .. } => ErrorCode::E0021,
            ErrorKind::RuntimeError { .. } => ErrorCode::E0022,
            ErrorKind::ResourceExhausted { .. } => ErrorCode::E0023,
            ErrorKind::PluginError { .. } => ErrorCode::E0024,
            ErrorKind::TimeoutError { .. } => ErrorCode::E0025,
            ErrorKind::NetworkError { .. } => ErrorCode::E0026,
            ErrorKind::FileNotFound { .. } => ErrorCode::E0027,
            ErrorKind::InvalidFormat { .. } => ErrorCode::E0028,
            ErrorKind::InvalidState { .. } => ErrorCode::E0029,
            ErrorKind::UnsupportedOperation { .. } => ErrorCode::E0030,
            ErrorKind::IoError { .. } => ErrorCode::E0031,
            ErrorKind::NotImplemented { .. } => ErrorCode::E0032,
            _ => ErrorCode::E9999,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Result type alias
pub type Result<T> = std::result::Result<T, TrustformersError>;

/// Helper macro for creating errors with context
#[macro_export]
macro_rules! tf_error {
    ($kind:expr) => {
        $crate::errors::TrustformersError::new($kind)
    };

    ($kind:expr, operation = $op:expr) => {
        $crate::errors::TrustformersError::new($kind).with_operation($op)
    };

    ($kind:expr, component = $comp:expr) => {
        $crate::errors::TrustformersError::new($kind).with_component($comp)
    };

    ($kind:expr, operation = $op:expr, component = $comp:expr) => {
        $crate::errors::TrustformersError::new($kind)
            .with_operation($op)
            .with_component($comp)
    };
}

/// Helper macro for adding context to existing errors
#[macro_export]
macro_rules! tf_context {
    ($err:expr, $key:expr => $value:expr) => {
        $err.with_context($key, $value.to_string())
    };

    ($err:expr, $key:expr => $value:expr, $($rest_key:expr => $rest_value:expr),+) => {
        tf_context!($err.with_context($key, $value.to_string()), $($rest_key => $rest_value),+)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = TrustformersError::new(ErrorKind::DimensionMismatch {
            expected: "[batch_size, 512, 768]".to_string(),
            actual: "[batch_size, 256, 768]".to_string(),
        })
        .with_operation("MultiHeadAttention.forward")
        .with_component("BERT")
        .with_context("layer", "12".to_string())
        .with_context("head_count", "12".to_string());

        let display = format!("{}", error);
        assert!(display.contains("Error [E0001]"));
        assert!(display.contains("MultiHeadAttention.forward"));
        assert!(display.contains("BERT"));
        assert!(display.contains("layer: 12"));
    }

    #[test]
    fn test_error_suggestions() {
        let error = TrustformersError::new(ErrorKind::OutOfMemory {
            required: 8_000_000_000,
            available: 4_000_000_000,
        });

        assert!(!error.suggestions.is_empty());
        assert!(error.suggestions.iter().any(|s| s.contains("batch size")));
        assert!(error.suggestions.iter().any(|s| s.contains("mixed precision")));
    }

    #[test]
    fn test_error_macros() {
        let error = tf_error!(
            ErrorKind::ModelNotFound {
                name: "gpt-5".to_string()
            },
            operation = "Model::from_pretrained",
            component = "ModelLoader"
        );

        assert_eq!(
            error.context.operation,
            Some("Model::from_pretrained".to_string())
        );
        assert_eq!(error.context.component, Some("ModelLoader".to_string()));
    }
}
