//! Error standardization utilities for migrating modules to TrustformersError
//!
//! This module provides utilities to help modules migrate from the legacy CoreError
//! system to the new TrustformersError system with rich context and suggestions.

use super::{ErrorKind, TrustformersError};

#[allow(deprecated)]
use crate::error::CoreError;

/// Standard error interface that all modules should use
pub trait StandardError {
    /// Convert any error to a standardized TrustformersError
    fn standardize(self) -> TrustformersError;

    /// Convert with operation context
    fn standardize_with_operation(self, operation: &str) -> TrustformersError;

    /// Convert with component context
    fn standardize_with_component(self, component: &str) -> TrustformersError;

    /// Convert with full context
    fn standardize_with_context(self, operation: &str, component: &str) -> TrustformersError;
}

#[allow(deprecated)]
impl StandardError for CoreError {
    fn standardize(self) -> TrustformersError {
        self.into()
    }

    fn standardize_with_operation(self, operation: &str) -> TrustformersError {
        TrustformersError::from(self).with_operation(operation)
    }

    fn standardize_with_component(self, component: &str) -> TrustformersError {
        TrustformersError::from(self).with_component(component)
    }

    fn standardize_with_context(self, operation: &str, component: &str) -> TrustformersError {
        TrustformersError::from(self)
            .with_operation(operation)
            .with_component(component)
    }
}

impl StandardError for String {
    fn standardize(self) -> TrustformersError {
        TrustformersError::new(ErrorKind::Other(self))
    }

    fn standardize_with_operation(self, operation: &str) -> TrustformersError {
        TrustformersError::new(ErrorKind::Other(self)).with_operation(operation)
    }

    fn standardize_with_component(self, component: &str) -> TrustformersError {
        TrustformersError::new(ErrorKind::Other(self)).with_component(component)
    }

    fn standardize_with_context(self, operation: &str, component: &str) -> TrustformersError {
        TrustformersError::new(ErrorKind::Other(self))
            .with_operation(operation)
            .with_component(component)
    }
}

impl StandardError for &str {
    fn standardize(self) -> TrustformersError {
        TrustformersError::new(ErrorKind::Other(self.to_string()))
    }

    fn standardize_with_operation(self, operation: &str) -> TrustformersError {
        TrustformersError::new(ErrorKind::Other(self.to_string())).with_operation(operation)
    }

    fn standardize_with_component(self, component: &str) -> TrustformersError {
        TrustformersError::new(ErrorKind::Other(self.to_string())).with_component(component)
    }

    fn standardize_with_context(self, operation: &str, component: &str) -> TrustformersError {
        TrustformersError::new(ErrorKind::Other(self.to_string()))
            .with_operation(operation)
            .with_component(component)
    }
}

impl StandardError for std::io::Error {
    fn standardize(self) -> TrustformersError {
        TrustformersError::new(ErrorKind::IoError(self))
    }

    fn standardize_with_operation(self, operation: &str) -> TrustformersError {
        TrustformersError::new(ErrorKind::IoError(self)).with_operation(operation)
    }

    fn standardize_with_component(self, component: &str) -> TrustformersError {
        TrustformersError::new(ErrorKind::IoError(self)).with_component(component)
    }

    fn standardize_with_context(self, operation: &str, component: &str) -> TrustformersError {
        TrustformersError::new(ErrorKind::IoError(self))
            .with_operation(operation)
            .with_component(component)
    }
}

/// Macro for easy error standardization with automatic context
#[macro_export]
macro_rules! std_error {
    ($err:expr) => {
        $crate::errors::standardization::StandardError::standardize($err)
    };

    ($err:expr, operation = $op:expr) => {
        $crate::errors::standardization::StandardError::standardize_with_operation($err, $op)
    };

    ($err:expr, component = $comp:expr) => {
        $crate::errors::standardization::StandardError::standardize_with_component($err, $comp)
    };

    ($err:expr, operation = $op:expr, component = $comp:expr) => {
        $crate::errors::standardization::StandardError::standardize_with_context($err, $op, $comp)
    };
}

/// Migration utilities for common error patterns
pub struct ErrorMigrationHelper;

impl ErrorMigrationHelper {
    /// Convert legacy shape error pattern to new system
    pub fn shape_error(
        expected: Vec<usize>,
        actual: Vec<usize>,
        operation: &str,
    ) -> TrustformersError {
        TrustformersError::new(ErrorKind::ShapeMismatch { expected, actual })
            .with_operation(operation)
            .with_suggestion("Check tensor dimensions before operations")
            .with_suggestion("Use .reshape() or broadcasting to fix dimension mismatches")
    }

    /// Convert legacy tensor operation error to new system
    pub fn tensor_operation_error(
        operation: &str,
        reason: &str,
        component: &str,
    ) -> TrustformersError {
        TrustformersError::new(ErrorKind::TensorOpError {
            operation: operation.to_string(),
            reason: reason.to_string(),
        })
        .with_component(component)
        .with_suggestion("Check tensor compatibility and data types")
        .with_suggestion("Enable tensor debugging for more information")
    }

    /// Convert legacy memory error to new system
    pub fn memory_allocation_error(reason: &str, operation: &str) -> TrustformersError {
        TrustformersError::new(ErrorKind::MemoryError {
            reason: reason.to_string(),
        })
        .with_operation(operation)
        .with_suggestion("Try reducing batch size or model complexity")
        .with_suggestion("Enable memory optimization settings")
    }

    /// Convert legacy hardware error to new system
    pub fn hardware_unavailable_error(
        device: &str,
        reason: &str,
        component: &str,
    ) -> TrustformersError {
        TrustformersError::new(ErrorKind::HardwareError {
            device: device.to_string(),
            reason: reason.to_string(),
        })
        .with_component(component)
        .with_suggestion("Check device drivers and installation")
        .with_suggestion("Try falling back to CPU execution")
    }

    /// Convert legacy configuration error to new system
    pub fn invalid_configuration_error(
        field: &str,
        reason: &str,
        component: &str,
    ) -> TrustformersError {
        TrustformersError::new(ErrorKind::InvalidConfiguration {
            field: field.to_string(),
            reason: reason.to_string(),
        })
        .with_component(component)
        .with_suggestion("Check configuration file syntax and values")
        .with_suggestion("Refer to documentation for valid parameter ranges")
    }
}

/// Extension trait for Result types to add standardization
pub trait ResultStandardization<T> {
    /// Standardize any error in a Result
    fn standardize_err(self) -> Result<T, TrustformersError>;

    /// Standardize with operation context
    fn standardize_err_with_operation(self, operation: &str) -> Result<T, TrustformersError>;

    /// Standardize with component context
    fn standardize_err_with_component(self, component: &str) -> Result<T, TrustformersError>;

    /// Standardize with full context
    fn standardize_err_with_context(
        self,
        operation: &str,
        component: &str,
    ) -> Result<T, TrustformersError>;
}

impl<T, E> ResultStandardization<T> for Result<T, E>
where
    E: StandardError,
{
    fn standardize_err(self) -> Result<T, TrustformersError> {
        self.map_err(|e| e.standardize())
    }

    fn standardize_err_with_operation(self, operation: &str) -> Result<T, TrustformersError> {
        self.map_err(|e| e.standardize_with_operation(operation))
    }

    fn standardize_err_with_component(self, component: &str) -> Result<T, TrustformersError> {
        self.map_err(|e| e.standardize_with_component(component))
    }

    fn standardize_err_with_context(
        self,
        operation: &str,
        component: &str,
    ) -> Result<T, TrustformersError> {
        self.map_err(|e| e.standardize_with_context(operation, component))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn test_core_error_standardization() {
        let core_err = CoreError::InvalidInput("test".to_string());
        let std_err = core_err.standardize_with_operation("test_operation");

        assert_eq!(
            std_err.context.operation,
            Some("test_operation".to_string())
        );
    }

    #[test]
    fn test_string_error_standardization() {
        let str_err = "Something went wrong";
        let std_err = str_err.standardize_with_component("TestComponent");

        assert_eq!(std_err.context.component, Some("TestComponent".to_string()));
    }

    #[test]
    #[allow(deprecated)]
    fn test_result_standardization() {
        fn failing_function() -> Result<(), CoreError> {
            Err(CoreError::InvalidArgument("test".to_string()))
        }

        let result = failing_function().standardize_err_with_context("test_op", "test_component");

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.context.operation, Some("test_op".to_string()));
        assert_eq!(err.context.component, Some("test_component".to_string()));
    }

    #[test]
    fn test_migration_helper() {
        let err =
            ErrorMigrationHelper::shape_error(vec![2, 3, 4], vec![2, 3, 5], "matrix_multiply");

        match &err.kind {
            ErrorKind::ShapeMismatch { expected, actual } => {
                assert_eq!(expected, &vec![2, 3, 4]);
                assert_eq!(actual, &vec![2, 3, 5]);
            },
            _ => panic!("Wrong error kind"),
        }

        assert_eq!(err.context.operation, Some("matrix_multiply".to_string()));
        assert!(err.suggestions.len() >= 2);
    }

    #[test]
    #[allow(deprecated)]
    fn test_std_error_macro() {
        let core_err = CoreError::TensorOpError {
            message: "test".to_string(),
            context: crate::error::ErrorContext::new(
                crate::error::ErrorCode::E2002,
                "test_operation".to_string(),
            ),
        };

        let err1 = std_error!(core_err);
        assert!(matches!(err1.kind, ErrorKind::ComputeError { .. }));

        let str_err = "test error";
        let err2 = std_error!(str_err, operation = "test_op");
        assert_eq!(err2.context.operation, Some("test_op".to_string()));
    }
}
