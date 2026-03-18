//! # HistoricalDataError - Trait Implementations
//!
//! This module contains trait implementations for `HistoricalDataError`.
//!
//! ## Implemented Traits
//!
//! - `From`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::HistoricalDataError;

impl From<crate::test_performance_monitoring::service::ServiceError> for HistoricalDataError {
    fn from(err: crate::test_performance_monitoring::service::ServiceError) -> Self {
        HistoricalDataError::StorageError {
            reason: format!("{:?}", err),
        }
    }
}
