//! # WorkloadType - Trait Implementations
//!
//! This module contains trait implementations for `WorkloadType`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::WorkloadType;

impl std::fmt::Display for WorkloadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Training => write!(f, "Training"),
            Self::Inference => write!(f, "Inference"),
            Self::Simulation => write!(f, "Simulation"),
            Self::DataProcessing => write!(f, "Data Processing"),
            Self::Rendering => write!(f, "Rendering"),
            Self::General => write!(f, "General"),
        }
    }
}
