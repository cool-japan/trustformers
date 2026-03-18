//! # CacheMetadata - Trait Implementations
//!
//! This module contains trait implementations for `CacheMetadata`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Instant;

use super::types::CacheMetadata;

impl Default for CacheMetadata {
    fn default() -> Self {
        Self {
            creation_time: Instant::now(),
            last_access: Instant::now(),
            access_count: 0,
            size_bytes: 0,
        }
    }
}
