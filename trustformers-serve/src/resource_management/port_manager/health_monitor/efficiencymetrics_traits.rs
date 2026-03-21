//! # EfficiencyMetrics - Trait Implementations
//!
//! This module contains trait implementations for `EfficiencyMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

use super::types::EfficiencyMetrics;

impl Default for EfficiencyMetrics {
    fn default() -> Self {
        Self {
            allocation_efficiency: 1.0,
            utilization_efficiency: 1.0,
            conflict_resolution_efficiency: 1.0,
            overall_efficiency: 1.0,
            fragmentation_level: 0.0,
            waste_percentage: 0.0,
        }
    }
}

