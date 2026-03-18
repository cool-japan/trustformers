//! # SensorManagerConfig - Trait Implementations
//!
//! This module contains trait implementations for `SensorManagerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{SensorConfig, SensorManagerConfig};

impl Default for SensorManagerConfig {
    fn default() -> Self {
        Self {
            max_history_size: 3600,
            sensor_config: SensorConfig::default(),
        }
    }
}
