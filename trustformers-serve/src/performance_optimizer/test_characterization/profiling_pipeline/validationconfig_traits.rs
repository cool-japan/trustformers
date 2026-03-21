//! # ValidationConfig - Trait Implementations
//!
//! This module contains trait implementations for `ValidationConfig`.
//!
//! ## Implemented Traits
//!
//! - `From`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ProfilingPipelineConfig, ValidationConfig};

impl From<&ProfilingPipelineConfig> for ValidationConfig {
    fn from(config: &ProfilingPipelineConfig) -> Self {
        Self {
            enable_validation: config.enable_validation,
            strictness: config.validation_strictness,
        }
    }
}
