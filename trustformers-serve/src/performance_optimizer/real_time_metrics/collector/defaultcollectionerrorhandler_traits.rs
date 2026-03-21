//! # DefaultCollectionErrorHandler - Trait Implementations
//!
//! This module contains trait implementations for `DefaultCollectionErrorHandler`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `CollectionErrorHandler`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::functions::CollectionErrorHandler;
use super::types::{
    CollectionError, DefaultCollectionErrorHandler, ErrorRecoveryAction, ErrorType,
    RecoveryStrategy,
};

impl Default for DefaultCollectionErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CollectionErrorHandler for DefaultCollectionErrorHandler {
    fn handle_error(&self, _error: CollectionError) -> Result<ErrorRecoveryAction> {
        Ok(ErrorRecoveryAction::Retry)
    }
    fn recovery_strategy(&self, _error_type: ErrorType) -> RecoveryStrategy {
        RecoveryStrategy::Immediate
    }
    fn report_error(&self, _error: CollectionError) -> Result<()> {
        Ok(())
    }
}
