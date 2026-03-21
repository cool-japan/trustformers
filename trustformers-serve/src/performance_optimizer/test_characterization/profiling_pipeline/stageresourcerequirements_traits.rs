//! # StageResourceRequirements - Trait Implementations
//!
//! This module contains trait implementations for `StageResourceRequirements`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{ResourceIntensityLevel, StageResourceRequirements};

impl Default for StageResourceRequirements {
    fn default() -> Self {
        Self {
            min_cpu_cores: 1,
            min_memory_bytes: 256 * 1024 * 1024,
            min_io_capacity: 5 * 1024 * 1024,
            estimated_duration: Duration::from_secs(30),
            intensity_level: ResourceIntensityLevel::Medium,
        }
    }
}
