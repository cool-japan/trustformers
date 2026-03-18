//! # VersionControl - Trait Implementations
//!
//! This module contains trait implementations for `VersionControl`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Instant;

use super::types::VersionControl;

impl Default for VersionControl {
    fn default() -> Self {
        Self {
            current_version: String::from("1.0.0"),
            version_history: Vec::new(),
            last_update: Instant::now(),
            compatibility_info: HashMap::new(),
        }
    }
}
