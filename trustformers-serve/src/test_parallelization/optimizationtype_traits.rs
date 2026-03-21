//! # OptimizationType - Trait Implementations
//!
//! This module contains trait implementations for `OptimizationType`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::OptimizationType;

impl fmt::Display for OptimizationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizationType::IncreaseParallelism => write!(f, "Increase Parallelism"),
            OptimizationType::DecreaseParallelism => write!(f, "Decrease Parallelism"),
            OptimizationType::AdjustResourceAllocation => {
                write!(f, "Adjust Resource Allocation")
            },
            OptimizationType::ChangeSchedulingStrategy => {
                write!(f, "Change Scheduling Strategy")
            },
            OptimizationType::OptimizeTestBatching => write!(f, "Optimize Test Batching"),
            OptimizationType::ImproveResourceSharing => {
                write!(f, "Improve Resource Sharing")
            },
            OptimizationType::AddCaching => write!(f, "Add Caching"),
            OptimizationType::OptimizeTestOrder => write!(f, "Optimize Test Order"),
            OptimizationType::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}
