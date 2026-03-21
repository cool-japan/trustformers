//! # PrioritySchedulingConfig - Trait Implementations
//!
//! This module contains trait implementations for `PrioritySchedulingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{PriorityFactors, PrioritySchedulingConfig};

impl Default for PrioritySchedulingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            priority_factors: PriorityFactors::default(),
            dynamic_adjustment: true,
            failure_priority_boost: 1.5,
        }
    }
}
