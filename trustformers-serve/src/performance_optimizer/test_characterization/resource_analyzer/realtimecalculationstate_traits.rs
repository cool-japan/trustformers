//! # RealTimeCalculationState - Trait Implementations
//!
//! This module contains trait implementations for `RealTimeCalculationState`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Instant;

use super::types::RealTimeCalculationState;

impl Default for RealTimeCalculationState {
    fn default() -> Self {
        Self {
            last_update: Instant::now(),
            update_count: 0,
            calculation_load: 0.0,
        }
    }
}
