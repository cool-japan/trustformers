//! Service-level errors and their gRPC status mapping.
//!
//! Every variant carries the real reason as a string built at the failure site,
//! so a client always learns what actually went wrong instead of receiving a
//! generic "internal error".

use tonic::Status;

#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    /// No model is registered under this id.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// The id is not registered, so it cannot serve inference.
    #[error("model not loaded: {0}")]
    ModelNotLoaded(String),

    /// The id is already taken by a loaded model.
    #[error("model already loaded: {0}")]
    ModelAlreadyLoaded(String),

    /// The request itself is malformed or asks for something nonsensical.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Loading a checkpoint (config, weights or tokenizer) failed.
    #[error("failed to load {0}")]
    Load(String),

    /// A generation run failed.
    #[error("generation failed: {0}")]
    Generation(String),

    /// A server-side limit was hit.
    #[error("resource exhausted: {0}")]
    ResourceExhausted(String),

    /// The request asked for a capability this build genuinely does not have.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl From<ServiceError> for Status {
    fn from(error: ServiceError) -> Self {
        let message = error.to_string();
        match error {
            ServiceError::ModelNotFound(_) => Status::not_found(message),
            ServiceError::ModelNotLoaded(_) => Status::failed_precondition(message),
            ServiceError::ModelAlreadyLoaded(_) => Status::already_exists(message),
            ServiceError::InvalidInput(_) => Status::invalid_argument(message),
            ServiceError::Load(_) => Status::internal(message),
            ServiceError::Generation(_) => Status::internal(message),
            ServiceError::ResourceExhausted(_) => Status::resource_exhausted(message),
            ServiceError::Unsupported(_) => Status::unimplemented(message),
        }
    }
}

pub type ServiceResult<T> = Result<T, ServiceError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_match_the_failure_kind() {
        let cases = [
            (
                ServiceError::ModelNotFound("m".into()),
                tonic::Code::NotFound,
            ),
            (
                ServiceError::ModelNotLoaded("m".into()),
                tonic::Code::FailedPrecondition,
            ),
            (
                ServiceError::ModelAlreadyLoaded("m".into()),
                tonic::Code::AlreadyExists,
            ),
            (
                ServiceError::InvalidInput("x".into()),
                tonic::Code::InvalidArgument,
            ),
            (ServiceError::Load("x".into()), tonic::Code::Internal),
            (ServiceError::Generation("x".into()), tonic::Code::Internal),
            (
                ServiceError::ResourceExhausted("x".into()),
                tonic::Code::ResourceExhausted,
            ),
            (
                ServiceError::Unsupported("x".into()),
                tonic::Code::Unimplemented,
            ),
        ];

        for (error, expected) in cases {
            let text = error.to_string();
            let status = Status::from(error);
            assert_eq!(status.code(), expected);
            // The real reason must survive the conversion; a client that only
            // reads `status.message()` still learns what happened.
            assert_eq!(status.message(), text);
        }
    }
}
