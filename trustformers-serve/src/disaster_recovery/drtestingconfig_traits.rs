//! # DRTestingConfig - Trait Implementations
//!
//! This module contains trait implementations for `DRTestingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DRTestingConfig, TestEnvironmentConfig, TestScenario, TestSchedule};

impl Default for DRTestingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            test_schedule: TestSchedule::Monthly,
            test_scenarios: vec![
                TestScenario::FailoverTest,
                TestScenario::BackupRestoreTest,
                TestScenario::DataConsistencyTest,
                TestScenario::PerformanceTest,
            ],
            test_environment: TestEnvironmentConfig {
                isolated_environment: true,
                test_data_size_gb: 100,
                max_test_duration_seconds: 3600,
            },
        }
    }
}
