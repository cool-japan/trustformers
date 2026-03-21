//! # ReportingConfig - Trait Implementations
//!
//! This module contains trait implementations for `ReportingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ReportingConfig;

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            report_retention_days: 30,
        }
    }
}
