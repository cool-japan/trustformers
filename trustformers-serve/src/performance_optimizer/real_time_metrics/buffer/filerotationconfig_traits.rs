//! # FileRotationConfig - Trait Implementations
//!
//! This module contains trait implementations for `FileRotationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::FileRotationConfig;

impl Default for FileRotationConfig {
    fn default() -> Self {
        Self {
            max_file_size: 100 * 1024 * 1024,
            max_files: 10,
            check_interval: Duration::from_secs(60),
            compress_rotated: true,
        }
    }
}
