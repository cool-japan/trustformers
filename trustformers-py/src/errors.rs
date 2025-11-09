use pyo3::{exceptions::PyException, prelude::*, types::PyType};
use thiserror::Error;

/// Custom error types for TrustformeRS Python bindings
#[derive(Error, Debug)]
pub enum TrustformersPyError {
    #[error("Tensor operation failed: {message}")]
    TensorError { message: String },

    #[error("Model loading failed: {message}")]
    ModelLoadError { message: String },

    #[error("Tokenizer error: {message}")]
    TokenizerError { message: String },

    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    #[error("Memory allocation failed: {message}")]
    MemoryError { message: String },

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatchError {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid input: {message}")]
    InvalidInputError { message: String },

    #[error("Core library error: {0}")]
    CoreError(#[from] trustformers_core::TrustformersError),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl From<TrustformersPyError> for PyErr {
    fn from(err: TrustformersPyError) -> PyErr {
        match err {
            TrustformersPyError::TensorError { message } => TensorError::new_err(message),
            TrustformersPyError::ModelLoadError { message } => ModelLoadError::new_err(message),
            TrustformersPyError::TokenizerError { message } => TokenizerError::new_err(message),
            TrustformersPyError::ConfigError { message } => ConfigError::new_err(message),
            TrustformersPyError::MemoryError { message } => MemoryError::new_err(message),
            TrustformersPyError::ShapeMismatchError { expected, actual } => {
                ShapeMismatchError::new_err(format!(
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected, actual
                ))
            },
            TrustformersPyError::InvalidInputError { message } => {
                InvalidInputError::new_err(message)
            },
            _ => PyException::new_err(err.to_string()),
        }
    }
}

// Custom Python exception types
pyo3::create_exception!(trustformers, TensorError, PyException);
pyo3::create_exception!(trustformers, ModelLoadError, PyException);
pyo3::create_exception!(trustformers, TokenizerError, PyException);
pyo3::create_exception!(trustformers, ConfigError, PyException);
pyo3::create_exception!(trustformers, MemoryError, PyException);
pyo3::create_exception!(trustformers, ShapeMismatchError, PyException);
pyo3::create_exception!(trustformers, InvalidInputError, PyException);

/// Result type alias for TrustformeRS Python operations
pub type TrustformersPyResult<T> = Result<T, TrustformersPyError>;

/// Helper macros for creating errors
#[macro_export]
macro_rules! tensor_error {
    ($msg:expr) => {
        $crate::errors::TrustformersPyError::TensorError {
            message: $msg.to_string(),
        }
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::errors::TrustformersPyError::TensorError {
            message: format!($fmt, $($arg)*),
        }
    };
}

#[macro_export]
macro_rules! model_load_error {
    ($msg:expr) => {
        $crate::errors::TrustformersPyError::ModelLoadError {
            message: $msg.to_string(),
        }
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::errors::TrustformersPyError::ModelLoadError {
            message: format!($fmt, $($arg)*),
        }
    };
}

#[macro_export]
macro_rules! tokenizer_error {
    ($msg:expr) => {
        $crate::errors::TrustformersPyError::TokenizerError {
            message: $msg.to_string(),
        }
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::errors::TrustformersPyError::TokenizerError {
            message: format!($fmt, $($arg)*),
        }
    };
}

#[macro_export]
macro_rules! shape_mismatch_error {
    ($expected:expr, $actual:expr) => {
        $crate::errors::TrustformersPyError::ShapeMismatchError {
            expected: $expected,
            actual: $actual,
        }
    };
}

/// Add exception classes to Python module
pub fn add_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("TensorError", m.py().get_type::<TensorError>())?;
    m.add("ModelLoadError", m.py().get_type::<ModelLoadError>())?;
    m.add("TokenizerError", m.py().get_type::<TokenizerError>())?;
    m.add("ConfigError", m.py().get_type::<ConfigError>())?;
    m.add("MemoryError", m.py().get_type::<MemoryError>())?;
    m.add(
        "ShapeMismatchError",
        m.py().get_type::<ShapeMismatchError>(),
    )?;
    m.add("InvalidInputError", m.py().get_type::<InvalidInputError>())?;
    Ok(())
}
