//! # PortHealthThresholds - Trait Implementations
//!
//! This module contains trait implementations for `PortHealthThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

use super::types::PortHealthThresholds;

impl Default for PortHealthThresholds {
    fn default() -> Self {
        Self {
            utilization_warning: 80.0,
            utilization_critical: 95.0,
            conflicts_per_minute_warning: 10.0,
            conflicts_per_minute_critical: 50.0,
            allocation_time_warning_ms: 100.0,
            allocation_time_critical_ms: 500.0,
            health_score_warning: 70.0,
            health_score_critical: 50.0,
            error_rate_warning: 5.0,
            error_rate_critical: 15.0,
            fragmentation_warning: 20.0,
            fragmentation_critical: 40.0,
        }
    }
}

