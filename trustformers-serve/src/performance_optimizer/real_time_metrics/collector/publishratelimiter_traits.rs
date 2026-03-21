//! # PublishRateLimiter - Trait Implementations
//!
//! This module contains trait implementations for `PublishRateLimiter`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PublishRateLimiter;

impl Default for PublishRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}
