//! # PublishingConfig - Trait Implementations
//!
//! This module contains trait implementations for `PublishingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::collector::*;

use super::types::PublishingConfig;

impl Default for PublishingConfig {
    fn default() -> Self {
        Self {
            enable_publishing: false,
            delivery_guarantee: DeliveryGuarantee::BestEffort,
            batch_size: 100,
            retry_attempts: 3,
            compression_enabled: false,
        }
    }
}
