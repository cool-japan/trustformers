//! # ClassificationRuleSet - Trait Implementations
//!
//! This module contains trait implementations for `ClassificationRuleSet`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::ClassificationRuleSet;

impl Default for ClassificationRuleSet {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            priorities: HashMap::new(),
            effectiveness: HashMap::new(),
        }
    }
}
