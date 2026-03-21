//! # SchedulingConfig - Trait Implementations
//!
//! This module contains trait implementations for `SchedulingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    EarlyTerminationStrategy, FailureHandlingStrategy, LoadBalancingConfig,
    PrioritySchedulingConfig, SchedulingConfig, SchedulingStrategy,
};

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            strategy: SchedulingStrategy::ResourceAware,
            load_balancing: LoadBalancingConfig::default(),
            priority_scheduling: PrioritySchedulingConfig::default(),
            adaptive_scheduling: true,
            failure_handling: FailureHandlingStrategy::StopDependent,
            early_termination: EarlyTerminationStrategy::ErrorRateThreshold(0.2),
        }
    }
}
