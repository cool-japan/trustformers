use tonic::{Code, Status};
#![allow(unused_variables)]

#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Inference error: {0}")]
    InferenceError(#[from] anyhow::Error),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

impl From<ServiceError> for Status {
    fn from(error: ServiceError) -> Self {
        match error {
            ServiceError::ModelNotFound(msg) => Status::not_found(msg),
            ServiceError::ModelNotLoaded(msg) => Status::failed_precondition(msg),
            ServiceError::InvalidInput(msg) => Status::invalid_argument(msg),
            ServiceError::InferenceError(e) => Status::internal(e.to_string()),
            ServiceError::TokenizationError(msg) => Status::invalid_argument(msg),
            ServiceError::ResourceExhausted(msg) => Status::resource_exhausted(msg),
            ServiceError::Internal(msg) => Status::internal(msg),
            ServiceError::Unsupported(msg) => Status::unimplemented(msg),
        }
    }
}

pub type ServiceResult<T> = Result<T, ServiceError>;