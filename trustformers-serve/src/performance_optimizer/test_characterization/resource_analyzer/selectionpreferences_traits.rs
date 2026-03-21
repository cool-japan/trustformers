//! # SelectionPreferences - Trait Implementations
//!
//! This module contains trait implementations for `SelectionPreferences`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

use super::types::SelectionPreferences;

impl Default for SelectionPreferences {
    fn default() -> Self {
        Self {
            primary_strategy: "auto".to_string(),
            total_selections: 0,
            algorithm_usage: HashMap::new(),
            quality_threshold: 0.7,
            performance_threshold: Duration::from_millis(100),
        }
    }
}
