//! # ProfileDataCollector - Trait Implementations
//!
//! This module contains trait implementations for `ProfileDataCollector`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ProfileDataCollector;

impl std::fmt::Debug for ProfileDataCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProfileDataCollector")
            .field("config", &self.config)
            .field("collectors", &"<trait objects>")
            .field("strategies", &"<trait objects>")
            .field("data_buffer", &self.data_buffer)
            .field("metrics", &self.metrics)
            .field("state", &self.state)
            .field("shutdown", &self.shutdown)
            .finish()
    }
}
