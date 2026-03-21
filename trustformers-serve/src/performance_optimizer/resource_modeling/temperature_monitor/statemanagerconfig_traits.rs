//! # StateManagerConfig - Trait Implementations
//!
//! This module contains trait implementations for `StateManagerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::StateManagerConfig;

impl Default for StateManagerConfig {
    fn default() -> Self {
        Self {
            max_state_history: 1000,
        }
    }
}
