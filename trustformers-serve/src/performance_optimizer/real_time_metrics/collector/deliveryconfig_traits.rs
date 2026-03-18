//! # DeliveryConfig - Trait Implementations
//!
//! This module contains trait implementations for `DeliveryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{DeliveryConfig, DeliveryGuarantee};

impl Default for DeliveryConfig {
    fn default() -> Self {
        Self {
            guarantee: DeliveryGuarantee::BestEffort,
            retry_attempts: 3,
            retry_delay: Duration::from_millis(1000),
            batch_size: 1,
            compression: false,
            timeout: Duration::from_secs(30),
            rate_limiting_enabled: false,
            max_throughput: 1000.0,
        }
    }
}
