//! # ErrorTrackingConfig - Trait Implementations
//!
//! This module contains trait implementations for `ErrorTrackingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ErrorTrackingConfig, NotificationThresholds};

impl Default for ErrorTrackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_errors_in_memory: 10000,
            retention_hours: 72,
            enable_grouping: true,
            enable_deduplication: true,
            enable_severity_classification: true,
            enable_trend_analysis: true,
            enable_notifications: true,
            notification_thresholds: NotificationThresholds::default(),
            enable_sampling: false,
            sampling_rate: 1.0,
            enable_stack_traces: true,
            max_stack_trace_depth: 50,
            enable_context_capture: true,
            export_interval_seconds: 300,
            export_endpoints: Vec::new(),
        }
    }
}
