//! # DefaultPublishErrorHandler - Trait Implementations
//!
//! This module contains trait implementations for `DefaultPublishErrorHandler`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `PublishErrorHandler`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::functions::PublishErrorHandler;
use super::types::{
    DefaultPublishErrorHandler, PublishError, PublishErrorType, PublishRecoveryAction,
    RetryStrategy,
};

impl Default for DefaultPublishErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl PublishErrorHandler for DefaultPublishErrorHandler {
    fn handle_error(&self, _error: PublishError) -> Result<PublishRecoveryAction> {
        Ok(PublishRecoveryAction::Retry)
    }
    fn retry_strategy(&self, _error_type: PublishErrorType) -> RetryStrategy {
        RetryStrategy::ExponentialBackoff
    }
    fn report_failure(&self, _error: PublishError) -> Result<()> {
        Ok(())
    }
}
