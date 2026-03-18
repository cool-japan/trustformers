//! # GpuMonitoringSystem - Trait Implementations
//!
//! This module contains trait implementations for `GpuMonitoringSystem`.
//!
//! ## Implemented Traits
//!
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::Ordering;
use tracing::warn;

use super::types::GpuMonitoringSystem;

impl Drop for GpuMonitoringSystem {
    fn drop(&mut self) {
        if self.monitoring_active.load(Ordering::Acquire) {
            warn!(
                "GpuMonitoringSystem dropped while still active - this may indicate improper shutdown"
            );
        }
    }
}
