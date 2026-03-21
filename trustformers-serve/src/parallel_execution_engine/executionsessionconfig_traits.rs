//! # ExecutionSessionConfig - Trait Implementations
//!
//! This module contains trait implementations for `ExecutionSessionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use crate::test_parallelization::{EarlyTerminationStrategy, FailureHandlingStrategy};

use super::types::ExecutionSessionConfig;

impl Default for ExecutionSessionConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tests: num_cpus::get(),
            session_timeout: Duration::from_secs(7200),
            failure_handling: FailureHandlingStrategy::StopDependent,
            early_termination: EarlyTerminationStrategy::ErrorRateThreshold(0.2),
        }
    }
}
