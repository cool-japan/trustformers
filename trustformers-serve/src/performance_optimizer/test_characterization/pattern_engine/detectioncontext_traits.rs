//! # DetectionContext - Trait Implementations
//!
//! This module contains trait implementations for `DetectionContext`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Instant;

use super::types::{DetectionContext, DetectionMode, PerformanceConstraints, SystemState};

impl Default for DetectionContext {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            historical_patterns: Vec::new(),
            system_state: SystemState::default(),
            constraints: PerformanceConstraints::default(),
            mode: DetectionMode::Hybrid,
        }
    }
}
