//! Auto-generated module structure

#[cfg(test)]
mod aws_lambda_tests;
pub mod awslambdaprovider_traits;
pub mod azurefunctionsprovider_traits;
pub mod errors;
pub mod functions;
pub mod googlecloudfunctionsprovider_traits;
pub mod serverlessorchestrator_traits;
pub mod types;

// Re-export all types
pub use errors::ServerlessError;
pub use functions::*;
pub use types::*;
