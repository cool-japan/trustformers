//! # CalibratorConfig - Trait Implementations
//!
//! This module contains trait implementations for `CalibratorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::CalibratorConfig;

impl Default for CalibratorConfig {
    fn default() -> Self {
        Self {
            calibration_interval: Duration::from_secs(86400),
        }
    }
}
