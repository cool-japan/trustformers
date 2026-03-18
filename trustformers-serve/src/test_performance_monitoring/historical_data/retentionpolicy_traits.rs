//! # RetentionPolicy - Trait Implementations
//!
//! This module contains trait implementations for `RetentionPolicy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::{Duration, SystemTime};

use super::types::RetentionPolicy;

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            policy_id: String::from("default_policy"),
            policy_name: String::from("Default Retention Policy"),
            description: String::from("Default retention policy with 90 days retention"),
            retention_period: Duration::from_secs(90 * 24 * 60 * 60),
            data_tiers: vec![],
            deletion_strategy: DeletionStrategy::default(),
            compliance_requirements: vec![],
            cost_optimization: CostOptimization::default(),
            created_at: SystemTime::now(),
            last_modified: SystemTime::now(),
        }
    }
}
