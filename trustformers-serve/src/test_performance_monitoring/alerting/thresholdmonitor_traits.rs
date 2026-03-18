//! # ThresholdMonitor - Trait Implementations
//!
//! This module contains trait implementations for `ThresholdMonitor`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ThresholdMonitor;

impl std::fmt::Debug for ThresholdMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThresholdMonitor")
            .field(
                "threshold_evaluators",
                &format!("<{} evaluators>", self.threshold_evaluators.len()),
            )
            .field("monitoring_scheduler", &self.monitoring_scheduler)
            .field("threshold_cache", &self.threshold_cache)
            .field("evaluation_metrics", &self.evaluation_metrics)
            .field("real_time_processor", &self.real_time_processor)
            .finish()
    }
}

impl Default for ThresholdMonitor {
    fn default() -> Self {
        Self::new()
    }
}
