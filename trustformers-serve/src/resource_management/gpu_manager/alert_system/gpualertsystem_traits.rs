//! # GpuAlertSystem - Trait Implementations
//!
//! This module contains trait implementations for `GpuAlertSystem`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::Ordering;

use super::types::GpuAlertSystem;

impl std::fmt::Debug for GpuAlertSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuAlertSystem")
            .field("config", &self.config)
            .field("active_alerts", &self.active_alerts)
            .field("alert_history", &self.alert_history)
            .field(
                "alert_handlers",
                &format!("<{} handlers>", self.alert_handlers.read().len()),
            )
            .field("alert_queue", &self.alert_queue)
            .field("active", &self.active.load(Ordering::SeqCst))
            .field(
                "background_tasks",
                &format!("<{} tasks>", self.background_tasks.lock().len()),
            )
            .field("escalation_tracking", &self.escalation_tracking)
            .field("alert_statistics", &self.alert_statistics)
            .finish()
    }
}

impl Drop for GpuAlertSystem {
    fn drop(&mut self) {
        if self.active.load(Ordering::Acquire) {
            self.active.store(false, Ordering::Release);
            let tasks = self.background_tasks.lock();
            for task in tasks.iter() {
                task.abort();
            }
        }
    }
}
