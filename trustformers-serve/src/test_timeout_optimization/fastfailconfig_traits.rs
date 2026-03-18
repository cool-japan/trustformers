//! # FastFailConfig - Trait Implementations
//!
//! This module contains trait implementations for `FastFailConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::FastFailConfig;

impl Default for FastFailConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            error_patterns: vec![
                "connection refused".to_string(),
                "network unreachable".to_string(),
                "permission denied".to_string(),
                "file not found".to_string(),
            ],
            confirmation_timeout: Duration::from_millis(500),
        }
    }
}
