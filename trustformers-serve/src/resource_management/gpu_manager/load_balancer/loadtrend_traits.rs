//! # LoadTrend - Trait Implementations
//!
//! This module contains trait implementations for `LoadTrend`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::VecDeque;

use super::types::{LoadBalancerTrendDirection, LoadTrend};

impl Default for LoadTrend {
    fn default() -> Self {
        Self {
            history: VecDeque::new(),
            predicted_load: 0.0,
            trend_direction: LoadBalancerTrendDirection::Stable,
            confidence: 0.0,
        }
    }
}
