//! # RealTimeAggregator - Trait Implementations
//!
//! This module contains trait implementations for `RealTimeAggregator`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl std::fmt::Debug for RealTimeAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealTimeAggregator")
            .field("windows", &self.windows)
            .field("config", &self.config)
            .field(
                "custom_rules",
                &format!("{} custom rules", self.custom_rules.len()),
            )
            .finish()
    }
}
