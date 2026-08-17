//! Structured errors for the cloud inference providers.
//!
//! A provider that cannot reach its API returns one of these variants. None of
//! them is ever replaced by a fabricated completion, a synthetic endpoint URL
//! or an invented billing figure.

use thiserror::Error;

/// Errors produced by the cloud provider implementations.
#[derive(Debug, Error)]
pub enum CloudProviderError {
    /// The provider was never initialized with a [`super::ProviderConfig`].
    #[error(
        "{provider} was not initialized: call `initialize(&ProviderConfig)` before `{operation}`"
    )]
    NotInitialized {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
    },

    /// The configuration carries no usable credential for this provider.
    #[error("{provider} has no credentials for `{operation}`: {detail}")]
    MissingCredentials {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
        /// Which credential field must be set.
        detail: String,
    },

    /// A required non-credential configuration value is missing.
    #[error("{provider} cannot run `{operation}`: {detail}")]
    MissingConfiguration {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
        /// What is missing.
        detail: String,
    },

    /// The request could not be sent at all.
    #[error("{provider} `{operation}` transport failure: {detail}")]
    Transport {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
        /// Underlying transport error.
        detail: String,
    },

    /// The API answered with a non-success status.
    #[error("{provider} `{operation}` failed with HTTP {status}: {body}")]
    ApiError {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
        /// HTTP status code returned by the API.
        status: u16,
        /// Response body, truncated for logging.
        body: String,
    },

    /// The API answered successfully but the payload could not be interpreted.
    #[error("{provider} `{operation}` returned an unusable response: {detail}")]
    InvalidResponse {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
        /// What was wrong with the payload.
        detail: String,
    },

    /// The request used an input type this provider's API cannot accept.
    #[error("{provider} does not accept {input_kind} input for `{operation}`")]
    UnsupportedInput {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
        /// The rejected input variant.
        input_kind: &'static str,
    },

    /// The provider's API has no equivalent for this operation.
    #[error(
        "{provider} has no API for `{operation}`; it will not report an invented result instead"
    )]
    UnsupportedOperation {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
    },
}

impl CloudProviderError {
    /// Convenience constructor for [`CloudProviderError::MissingCredentials`].
    pub fn missing_credentials(
        provider: &'static str,
        operation: &'static str,
        detail: impl Into<String>,
    ) -> Self {
        Self::MissingCredentials {
            provider,
            operation,
            detail: detail.into(),
        }
    }

    /// Convenience constructor for [`CloudProviderError::UnsupportedOperation`].
    pub fn unsupported(provider: &'static str, operation: &'static str) -> Self {
        Self::UnsupportedOperation {
            provider,
            operation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_credentials_names_the_field() {
        let err = CloudProviderError::missing_credentials(
            "OpenAiProvider",
            "inference",
            "set credentials.api_key",
        );
        let message = err.to_string();
        assert!(message.contains("OpenAiProvider"));
        assert!(message.contains("api_key"));
    }

    #[test]
    fn test_unsupported_operation_states_it_will_not_invent() {
        let err = CloudProviderError::unsupported("AnthropicProvider", "deploy_model");
        assert!(err.to_string().contains("invented"));
    }

    #[test]
    fn test_errors_survive_anyhow_downcast() {
        let err: anyhow::Error = CloudProviderError::NotInitialized {
            provider: "OpenAiProvider",
            operation: "inference",
        }
        .into();
        assert!(matches!(
            err.downcast_ref::<CloudProviderError>(),
            Some(CloudProviderError::NotInitialized { .. })
        ));
    }
}
