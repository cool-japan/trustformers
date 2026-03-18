//! # TestSuiteOrganizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `TestSuiteOrganizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    SuiteDefinitionConfig, SuiteExecutionOrder, TestGroupingConfig, TestSuiteOrganizationConfig,
};

impl Default for TestSuiteOrganizationConfig {
    fn default() -> Self {
        Self {
            suite_definition: SuiteDefinitionConfig::default(),
            test_grouping: TestGroupingConfig::default(),
            execution_order: SuiteExecutionOrder::Priority,
            suite_dependencies: true,
        }
    }
}
