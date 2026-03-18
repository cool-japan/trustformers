//! # IsolationRequirements - Trait Implementations
//!
//! This module contains trait implementations for `IsolationRequirements`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::IsolationRequirements;

impl Default for IsolationRequirements {
    fn default() -> Self {
        Self {
            process_isolation: false,
            network_isolation: false,
            filesystem_isolation: false,
            database_isolation: false,
            gpu_isolation: false,
            custom_isolation: HashMap::new(),
        }
    }
}
