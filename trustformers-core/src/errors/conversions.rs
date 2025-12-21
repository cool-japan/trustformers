//! Error conversion utilities for seamless integration with existing code

#![allow(deprecated)] // Backward compatibility conversions for deprecated CoreError
#![allow(unused_variables)] // Error conversions

use super::{ErrorKind, TrustformersError};
use crate::error::CoreError;
use anyhow::Error as AnyhowError;
use scirs2_core::ndarray::ShapeError;
use std::time::Instant;

impl From<CoreError> for TrustformersError {
    fn from(err: CoreError) -> Self {
        match err {
            CoreError::DimensionMismatch { context: _ } => {
                TrustformersError::new(ErrorKind::DimensionMismatch {
                    expected: "unknown".to_string(),
                    actual: "unknown".to_string(),
                })
            },

            CoreError::ShapeMismatch {
                expected,
                got,
                context: _,
            } => TrustformersError::new(ErrorKind::DimensionMismatch {
                expected: format!("{:?}", expected),
                actual: format!("{:?}", got),
            }),

            CoreError::InvalidArgument(msg) => {
                TrustformersError::new(ErrorKind::InvalidConfiguration {
                    field: "argument".to_string(),
                    reason: msg,
                })
            },

            CoreError::InvalidConfig(msg) => {
                TrustformersError::new(ErrorKind::InvalidConfiguration {
                    field: "config".to_string(),
                    reason: msg,
                })
            },

            CoreError::NotImplemented(msg) => TrustformersError::new(ErrorKind::Other(msg))
                .with_suggestion("This feature is not yet implemented")
                .with_suggestion("Check the roadmap for planned features"),

            CoreError::Io(io_err) => TrustformersError::new(ErrorKind::IoError(io_err)),
            CoreError::Serialization(serde_err) => {
                TrustformersError::new(ErrorKind::SerializationError {
                    format: "JSON".to_string(),
                    reason: serde_err.to_string(),
                })
            },

            CoreError::WeightLoadError(msg) => TrustformersError::new(ErrorKind::Other(msg)),
            CoreError::TensorOpError {
                message,
                context: _,
            } => TrustformersError::new(ErrorKind::ComputeError {
                operation: "tensor_op".to_string(),
                reason: message,
            }),
            CoreError::ModelError(msg) => {
                TrustformersError::new(ErrorKind::ModelNotFound { name: msg })
            },
            CoreError::ShapeError(msg) => TrustformersError::new(ErrorKind::DimensionMismatch {
                expected: "valid_shape".to_string(),
                actual: msg,
            }),
            CoreError::SafeTensorsError(msg) => TrustformersError::new(ErrorKind::Other(msg)),
            CoreError::InvalidInput(msg) => {
                TrustformersError::new(ErrorKind::InvalidConfiguration {
                    field: "input".to_string(),
                    reason: msg,
                })
            },
            CoreError::TokenizerError(msg) => TrustformersError::new(ErrorKind::Other(msg)),
            CoreError::RuntimeError(msg) => TrustformersError::new(ErrorKind::Other(msg)),
            CoreError::IoError(msg) => {
                TrustformersError::new(ErrorKind::IoError(std::io::Error::other(msg)))
            },
            CoreError::ConfigError {
                message,
                context: _,
            } => TrustformersError::new(ErrorKind::InvalidConfiguration {
                field: "config".to_string(),
                reason: message,
            }),
            CoreError::ComputationError(msg) => TrustformersError::new(ErrorKind::ComputeError {
                operation: "computation".to_string(),
                reason: msg,
            }),
            CoreError::SerializationError(msg) => {
                TrustformersError::new(ErrorKind::SerializationError {
                    format: "unknown".to_string(),
                    reason: msg,
                })
            },
            CoreError::QuantizationError(msg) => TrustformersError::new(ErrorKind::Other(msg)),
            CoreError::ResourceExhausted(msg) => TrustformersError::new(ErrorKind::OutOfMemory {
                required: 0,
                available: 0,
            })
            .with_context("details", msg),
            CoreError::FormattingError(msg) => TrustformersError::new(ErrorKind::Other(msg)),
            CoreError::ImageProcessingError(msg) => TrustformersError::new(ErrorKind::Other(msg)),
            CoreError::LockError(msg) => TrustformersError::new(ErrorKind::Other(msg)),
            CoreError::PluginError(msg) => TrustformersError::new(ErrorKind::Other(msg)),
            CoreError::MemoryError {
                message,
                context: _,
            } => TrustformersError::new(ErrorKind::OutOfMemory {
                required: 0,
                available: 0,
            })
            .with_context("details", message),
            CoreError::HardwareError {
                message,
                context: _,
            } => TrustformersError::new(ErrorKind::HardwareError {
                device: "unknown".to_string(),
                reason: message,
            }),
            CoreError::PerformanceError {
                message,
                context: _,
            } => TrustformersError::new(ErrorKind::PerformanceError { reason: message }),
            CoreError::Timeout(msg) => TrustformersError::new(ErrorKind::TimeoutError {
                operation: "unknown".to_string(),
                timeout_ms: 0,
            })
            .with_context("details", msg),
            CoreError::NetworkError(msg) => TrustformersError::new(ErrorKind::Other(msg)),
            CoreError::FileNotFound(msg) => {
                TrustformersError::new(ErrorKind::FileNotFound { path: msg })
            },
            CoreError::TensorNotFound(msg) => TrustformersError::new(ErrorKind::Other(msg)),
            CoreError::InvalidFormat(msg) => TrustformersError::new(ErrorKind::InvalidFormat {
                expected: "valid_format".to_string(),
                actual: msg,
            }),
            CoreError::InvalidState(msg) => TrustformersError::new(ErrorKind::Other(msg)),
            CoreError::UnsupportedFormat(msg) => {
                TrustformersError::new(ErrorKind::UnsupportedOperation {
                    operation: "format_parsing".to_string(),
                    target: msg,
                })
            },
            CoreError::AutodiffError(msg) => TrustformersError::new(ErrorKind::ComputeError {
                operation: "autodiff".to_string(),
                reason: msg,
            }),
            CoreError::InvalidOperation(msg) => {
                TrustformersError::new(ErrorKind::UnsupportedOperation {
                    operation: "tensor_operation".to_string(),
                    target: msg,
                })
            },
            CoreError::InternalError(msg) => TrustformersError::new(ErrorKind::Other(msg)),

            CoreError::Other(msg) => TrustformersError::new(ErrorKind::Other(msg.to_string())),
            CoreError::DeviceNotFound(device_id) => {
                TrustformersError::new(ErrorKind::HardwareError {
                    device: device_id,
                    reason: "Device not found in registry".to_string(),
                })
            },
        }
    }
}

impl From<std::io::Error> for TrustformersError {
    fn from(err: std::io::Error) -> Self {
        TrustformersError::new(ErrorKind::IoError(err))
            .with_suggestion("Check file permissions and path existence")
            .with_suggestion("Ensure sufficient disk space")
    }
}

impl From<std::fmt::Error> for TrustformersError {
    fn from(err: std::fmt::Error) -> Self {
        TrustformersError::new(ErrorKind::Other(format!("Format error: {}", err)))
            .with_suggestion("Check string formatting operations")
    }
}

impl From<serde_json::Error> for TrustformersError {
    fn from(err: serde_json::Error) -> Self {
        TrustformersError::new(ErrorKind::SerializationError {
            format: "JSON".to_string(),
            reason: err.to_string(),
        })
    }
}

// Backward compatibility: TrustformersError -> CoreError conversion
impl From<TrustformersError> for CoreError {
    fn from(err: TrustformersError) -> Self {
        match err.kind {
            ErrorKind::DimensionMismatch { expected, actual } => CoreError::DimensionMismatch {
                context: crate::error::ErrorContext::new(
                    crate::error::ErrorCode::E1002,
                    "dimension_mismatch".to_string(),
                ),
            },
            ErrorKind::ShapeMismatch { expected, actual } => CoreError::ShapeMismatch {
                expected,
                got: actual,
                context: crate::error::ErrorContext::new(
                    crate::error::ErrorCode::E1001,
                    "shape_mismatch".to_string(),
                ),
            },
            ErrorKind::OutOfMemory { .. } => CoreError::MemoryError {
                message: "Out of memory".to_string(),
                context: crate::error::ErrorContext::new(
                    crate::error::ErrorCode::E3001,
                    "memory_allocation".to_string(),
                ),
            },
            ErrorKind::InvalidConfiguration { field, reason } => {
                CoreError::InvalidConfig(format!("{}: {}", field, reason))
            },
            ErrorKind::ModelNotFound { name } => {
                CoreError::ModelError(format!("Model not found: {}", name))
            },
            ErrorKind::TensorOpError { operation, reason } => CoreError::TensorOpError {
                message: format!("{}: {}", operation, reason),
                context: crate::error::ErrorContext::new(crate::error::ErrorCode::E2002, operation),
            },
            ErrorKind::IoError(io_err) => CoreError::Io(io_err),
            _ => CoreError::Other(anyhow::anyhow!(err.to_string())),
        }
    }
}

impl From<ShapeError> for TrustformersError {
    fn from(err: ShapeError) -> Self {
        TrustformersError::new(ErrorKind::DimensionMismatch {
            expected: "valid shape".to_string(),
            actual: format!("invalid shape: {}", err),
        })
        .with_suggestion("Check tensor dimensions and shape compatibility")
        .with_suggestion("Ensure tensor shapes match operation requirements")
    }
}

impl From<AnyhowError> for TrustformersError {
    fn from(err: AnyhowError) -> Self {
        // Try to downcast to known error types
        if let Some(core_err) = err.downcast_ref::<CoreError>() {
            // Convert without cloning by matching on the error type
            return match core_err {
                CoreError::DimensionMismatch { context: _ } => {
                    TrustformersError::new(ErrorKind::DimensionMismatch {
                        expected: "unknown".to_string(),
                        actual: "unknown".to_string(),
                    })
                },
                _ => TrustformersError::new(ErrorKind::Other(core_err.to_string())),
            };
        }

        if let Some(io_err) = err.downcast_ref::<std::io::Error>() {
            return TrustformersError::new(ErrorKind::IoError(io_err.kind().into()))
                .with_context("source", err.to_string());
        }

        // Generic conversion
        TrustformersError::new(ErrorKind::Other(err.to_string()))
    }
}

/// Extension trait for adding context to Results
pub trait ResultExt<T> {
    /// Add operation context to an error
    fn with_operation(self, operation: impl Into<String>) -> Result<T, TrustformersError>;

    /// Add component context to an error
    fn with_component(self, component: impl Into<String>) -> Result<T, TrustformersError>;

    /// Add arbitrary context to an error
    fn with_context_key(self, key: &str, value: impl Into<String>) -> Result<T, TrustformersError>;
}

impl<T, E> ResultExt<T> for Result<T, E>
where
    E: Into<TrustformersError>,
{
    fn with_operation(self, operation: impl Into<String>) -> Result<T, TrustformersError> {
        self.map_err(|e| e.into().with_operation(operation))
    }

    fn with_component(self, component: impl Into<String>) -> Result<T, TrustformersError> {
        self.map_err(|e| e.into().with_component(component))
    }

    fn with_context_key(self, key: &str, value: impl Into<String>) -> Result<T, TrustformersError> {
        self.map_err(|e| e.into().with_context(key, value.into()))
    }
}

/// Helper function for dimension mismatch errors
pub fn dimension_mismatch(expected: impl ToString, actual: impl ToString) -> TrustformersError {
    TrustformersError::new(ErrorKind::DimensionMismatch {
        expected: expected.to_string(),
        actual: actual.to_string(),
    })
}

/// Helper function for OOM errors
pub fn out_of_memory(required: usize, available: usize) -> TrustformersError {
    TrustformersError::new(ErrorKind::OutOfMemory {
        required,
        available,
    })
}

/// Helper function for configuration errors
pub fn invalid_config(field: impl Into<String>, reason: impl Into<String>) -> TrustformersError {
    TrustformersError::new(ErrorKind::InvalidConfiguration {
        field: field.into(),
        reason: reason.into(),
    })
}

/// Helper function for model not found errors
pub fn model_not_found(name: impl Into<String>) -> TrustformersError {
    TrustformersError::new(ErrorKind::ModelNotFound { name: name.into() })
}

/// Helper function for compute errors
pub fn compute_error(operation: impl Into<String>, reason: impl Into<String>) -> TrustformersError {
    TrustformersError::new(ErrorKind::ComputeError {
        operation: operation.into(),
        reason: reason.into(),
    })
}

/// Helper function for shape mismatch errors
pub fn shape_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> TrustformersError {
    TrustformersError::new(ErrorKind::ShapeMismatch { expected, actual })
}

/// Helper function for tensor operation errors
pub fn tensor_op_error(
    operation: impl Into<String>,
    reason: impl Into<String>,
) -> TrustformersError {
    TrustformersError::new(ErrorKind::TensorOpError {
        operation: operation.into(),
        reason: reason.into(),
    })
}

/// Helper function for memory errors
pub fn memory_error(reason: impl Into<String>) -> TrustformersError {
    TrustformersError::new(ErrorKind::MemoryError {
        reason: reason.into(),
    })
}

/// Helper function for hardware errors
pub fn hardware_error(device: impl Into<String>, reason: impl Into<String>) -> TrustformersError {
    TrustformersError::new(ErrorKind::HardwareError {
        device: device.into(),
        reason: reason.into(),
    })
}

/// Helper function for performance errors
pub fn performance_error(reason: impl Into<String>) -> TrustformersError {
    TrustformersError::new(ErrorKind::PerformanceError {
        reason: reason.into(),
    })
}

/// Helper function for invalid input errors
pub fn invalid_input(reason: impl Into<String>) -> TrustformersError {
    TrustformersError::new(ErrorKind::InvalidInput {
        reason: reason.into(),
    })
}

/// Helper function for runtime errors
pub fn runtime_error(reason: impl Into<String>) -> TrustformersError {
    TrustformersError::new(ErrorKind::RuntimeError {
        reason: reason.into(),
    })
}

/// Helper function for resource exhausted errors
pub fn resource_exhausted(
    resource: impl Into<String>,
    reason: impl Into<String>,
) -> TrustformersError {
    TrustformersError::new(ErrorKind::ResourceExhausted {
        resource: resource.into(),
        reason: reason.into(),
    })
}

/// Helper function for timeout errors
pub fn timeout_error(operation: impl Into<String>, timeout_ms: u64) -> TrustformersError {
    TrustformersError::new(ErrorKind::TimeoutError {
        operation: operation.into(),
        timeout_ms,
    })
}

/// Helper function for file not found errors
pub fn file_not_found(path: impl Into<String>) -> TrustformersError {
    TrustformersError::new(ErrorKind::FileNotFound { path: path.into() })
}

/// Helper function for invalid format errors
pub fn invalid_format(expected: impl Into<String>, actual: impl Into<String>) -> TrustformersError {
    TrustformersError::new(ErrorKind::InvalidFormat {
        expected: expected.into(),
        actual: actual.into(),
    })
}

/// Helper function for unsupported operation errors
pub fn unsupported_operation(
    operation: impl Into<String>,
    target: impl Into<String>,
) -> TrustformersError {
    TrustformersError::new(ErrorKind::UnsupportedOperation {
        operation: operation.into(),
        target: target.into(),
    })
}

/// Helper function for not implemented errors
pub fn not_implemented(feature: impl Into<String>) -> TrustformersError {
    TrustformersError::new(ErrorKind::NotImplemented {
        feature: feature.into(),
    })
}

/// Helper function for model compatibility errors
pub fn model_compatibility_error(
    model_type: impl Into<String>,
    required_version: impl Into<String>,
) -> TrustformersError {
    TrustformersError::new(ErrorKind::InvalidConfiguration {
        field: "model_compatibility".to_string(),
        reason: format!(
            "Model type '{}' requires version '{}'",
            model_type.into(),
            required_version.into()
        ),
    })
    .with_suggestion("Update to a compatible model version")
    .with_suggestion("Check the model documentation for compatibility requirements")
}

/// Helper function for quantization errors
pub fn quantization_error(
    operation: impl Into<String>,
    reason: impl Into<String>,
) -> TrustformersError {
    TrustformersError::new(ErrorKind::ComputeError {
        operation: operation.into(),
        reason: reason.into(),
    })
    .with_suggestion("Try a different quantization scheme")
    .with_suggestion("Check if the model supports the requested quantization")
    .with_suggestion("Verify quantization parameters are within valid ranges")
}

/// Helper function for hardware acceleration errors
pub fn acceleration_error(
    backend: impl Into<String>,
    reason: impl Into<String>,
) -> TrustformersError {
    TrustformersError::new(ErrorKind::HardwareError {
        device: backend.into(),
        reason: reason.into(),
    })
    .with_suggestion("Check hardware drivers are installed and up to date")
    .with_suggestion("Verify hardware compatibility with the operation")
    .with_suggestion("Try falling back to CPU execution")
}

/// Helper function for checkpoint loading errors
pub fn checkpoint_error(path: impl Into<String>, reason: impl Into<String>) -> TrustformersError {
    TrustformersError::new(ErrorKind::IoError(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("Checkpoint error at {}: {}", path.into(), reason.into()),
    )))
    .with_suggestion("Verify the checkpoint file is not corrupted")
    .with_suggestion("Check if the checkpoint format is supported")
    .with_suggestion("Ensure sufficient disk space and permissions")
}

/// Helper function for creating errors with timing information
pub fn timed_error(
    kind: ErrorKind,
    operation_start: Instant,
    operation_name: impl Into<String>,
) -> TrustformersError {
    let duration = operation_start.elapsed();
    TrustformersError::new(kind)
        .with_operation(operation_name)
        .with_context("duration_ms", duration.as_millis().to_string())
        .with_suggestion(format!(
            "Operation took {:.2}ms - consider optimization if this is slow",
            duration.as_millis()
        ))
}

/// Result extension trait for adding error context with timing
pub trait TimedResultExt<T> {
    /// Add timing context to an error result
    fn with_timing(
        self,
        operation_start: Instant,
        operation_name: impl Into<String>,
    ) -> Result<T, TrustformersError>;
}

impl<T, E> TimedResultExt<T> for Result<T, E>
where
    E: Into<TrustformersError>,
{
    fn with_timing(
        self,
        operation_start: Instant,
        operation_name: impl Into<String>,
    ) -> Result<T, TrustformersError> {
        self.map_err(|err| {
            let duration = operation_start.elapsed();
            err.into()
                .with_operation(operation_name)
                .with_context("duration_ms", duration.as_millis().to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_error_conversion() {
        let tf_err =
            TrustformersError::dimension_mismatch("expected".to_string(), "actual".to_string());

        match &tf_err.kind {
            ErrorKind::DimensionMismatch { .. } => {},
            _ => panic!("Wrong error kind"),
        }
    }

    #[test]
    fn test_result_extension() {
        fn failing_operation() -> Result<(), TrustformersError> {
            Err(TrustformersError::invalid_argument("test".to_string()))
        }

        let result = failing_operation()
            .with_operation("test_operation")
            .with_component("TestComponent");

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.context.operation, Some("test_operation".to_string()));
        assert_eq!(err.context.component, Some("TestComponent".to_string()));
    }
}
