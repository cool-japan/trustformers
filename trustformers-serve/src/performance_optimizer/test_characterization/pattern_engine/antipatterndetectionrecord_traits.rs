//! # AntiPatternDetectionRecord - Trait Implementations
//!
//! This module contains trait implementations for `AntiPatternDetectionRecord`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Instant;

use super::types::{AntiPatternDetectionRecord, DetectionContext};

impl Default for AntiPatternDetectionRecord {
    fn default() -> Self {
        Self {
            pattern_id: String::new(),
            detected_at: Instant::now(),
            confidence: 0.0,
            severity: 0.0,
            context: DetectionContext::default(),
        }
    }
}
