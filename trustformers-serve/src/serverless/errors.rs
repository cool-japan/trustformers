//! Structured errors for the serverless providers.
//!
//! Every provider operation that cannot reach its cloud API returns one of
//! these variants. None of them is ever swapped for a fabricated success value:
//! an unconfigured provider fails loudly so a deployment pipeline stops instead
//! of believing it shipped a function that does not exist.

use thiserror::Error;

/// Errors produced by the serverless provider implementations.
#[derive(Debug, Error)]
pub enum ServerlessError {
    /// No SDK client is installed on the provider, so no API call can be made.
    #[error(
        "{provider} is not configured with credentials: build the provider with \
         `with_aws_config()` or `with_clients(..)` before calling `{operation}`"
    )]
    MissingCredentials {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
    },

    /// A required piece of configuration is absent.
    #[error("{provider} cannot run `{operation}`: {detail}")]
    MissingConfiguration {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
        /// What is missing and how to supply it.
        detail: String,
    },

    /// The cloud API rejected the request or was unreachable.
    #[error("{provider} `{operation}` failed: {source_message}")]
    ApiError {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
        /// Message reported by the SDK.
        source_message: String,
    },

    /// The API answered, but with a payload this code cannot interpret.
    #[error("{provider} `{operation}` returned an unusable response: {detail}")]
    InvalidResponse {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
        /// What was wrong with the response.
        detail: String,
    },

    /// The provider has no implementation for this operation against its cloud.
    #[error(
        "{provider} does not implement `{operation}` against the real {provider} API yet; \
         it will not report fabricated results instead"
    )]
    NotImplemented {
        /// Provider that was called.
        provider: &'static str,
        /// Operation that was attempted.
        operation: &'static str,
    },
}

impl ServerlessError {
    /// Convenience constructor for [`ServerlessError::MissingCredentials`].
    pub fn missing_credentials(provider: &'static str, operation: &'static str) -> Self {
        Self::MissingCredentials {
            provider,
            operation,
        }
    }

    /// Convenience constructor for [`ServerlessError::ApiError`].
    pub fn api(
        provider: &'static str,
        operation: &'static str,
        source_message: impl std::fmt::Display,
    ) -> Self {
        Self::ApiError {
            provider,
            operation,
            source_message: source_message.to_string(),
        }
    }

    /// Convenience constructor for [`ServerlessError::NotImplemented`].
    pub fn not_implemented(provider: &'static str, operation: &'static str) -> Self {
        Self::NotImplemented {
            provider,
            operation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_credentials_message_names_the_fix() {
        let err = ServerlessError::missing_credentials("AwsLambdaProvider", "deploy");
        let message = err.to_string();
        assert!(message.contains("AwsLambdaProvider"));
        assert!(message.contains("deploy"));
        assert!(message.contains("with_clients"));
    }

    #[test]
    fn test_errors_survive_anyhow_downcast() {
        let err: anyhow::Error =
            ServerlessError::missing_credentials("AwsLambdaProvider", "invoke").into();
        let downcast = err.downcast_ref::<ServerlessError>();
        assert!(matches!(
            downcast,
            Some(ServerlessError::MissingCredentials { .. })
        ));
    }

    #[test]
    fn test_not_implemented_states_it_will_not_fabricate() {
        let err = ServerlessError::not_implemented("AzureFunctionsProvider", "get_metrics");
        assert!(err.to_string().contains("fabricated"));
    }
}
