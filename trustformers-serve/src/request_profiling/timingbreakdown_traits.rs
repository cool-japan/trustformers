//! # TimingBreakdown - Trait Implementations
//!
//! This module contains trait implementations for `TimingBreakdown`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::TimingBreakdown;

impl Default for TimingBreakdown {
    fn default() -> Self {
        Self {
            parsing_duration: None,
            auth_duration: None,
            model_load_duration: None,
            preprocessing_duration: None,
            inference_duration: None,
            postprocessing_duration: None,
            serialization_duration: None,
            network_io_duration: None,
            database_duration: None,
            cache_duration: None,
            custom_timings: HashMap::new(),
        }
    }
}
