//! # TestGroupingConfig - Trait Implementations
//!
//! This module contains trait implementations for `TestGroupingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{GroupFormationCriteria, TestGroupingConfig, TestGroupingStrategy};

impl Default for TestGroupingConfig {
    fn default() -> Self {
        Self {
            strategy: TestGroupingStrategy::ByResource,
            max_tests_per_group: 10,
            formation_criteria: GroupFormationCriteria::default(),
            dynamic_regrouping: true,
        }
    }
}
