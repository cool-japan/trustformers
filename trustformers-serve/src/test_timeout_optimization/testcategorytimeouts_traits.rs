//! # TestCategoryTimeouts - Trait Implementations
//!
//! This module contains trait implementations for `TestCategoryTimeouts`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::TestCategoryTimeouts;

impl Default for TestCategoryTimeouts {
    fn default() -> Self {
        Self {
            unit_tests: Duration::from_secs(5),
            integration_tests: Duration::from_secs(30),
            e2e_tests: Duration::from_secs(120),
            stress_tests: Duration::from_secs(300),
            property_tests: Duration::from_secs(60),
            chaos_tests: Duration::from_secs(180),
            long_running_tests: Duration::from_secs(600),
        }
    }
}
