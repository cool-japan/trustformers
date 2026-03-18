//! # SuiteDefinitionConfig - Trait Implementations
//!
//! This module contains trait implementations for `SuiteDefinitionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::{SuiteDefinitionConfig, SuiteNamingConvention};

impl Default for SuiteDefinitionConfig {
    fn default() -> Self {
        Self {
            auto_discovery: true,
            manual_suites: HashMap::new(),
            naming_convention: SuiteNamingConvention::DirectoryBased,
            metadata_requirements: vec![],
        }
    }
}
