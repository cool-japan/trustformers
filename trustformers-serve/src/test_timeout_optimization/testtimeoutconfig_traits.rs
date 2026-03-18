//! # TestTimeoutConfig - Trait Implementations
//!
//! This module contains trait implementations for `TestTimeoutConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::{
    AdaptiveTimeoutConfig, EarlyTerminationConfig, MonitoringConfig, TestCategoryTimeouts,
    TestTimeoutConfig,
};

impl Default for TestTimeoutConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_timeouts: TestCategoryTimeouts::default(),
            adaptive: AdaptiveTimeoutConfig::default(),
            early_termination: EarlyTerminationConfig::default(),
            monitoring: MonitoringConfig::default(),
            environment_overrides: HashMap::new(),
        }
    }
}
