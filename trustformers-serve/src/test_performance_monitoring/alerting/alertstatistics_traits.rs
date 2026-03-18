//! # AlertStatistics - Trait Implementations
//!
//! This module contains trait implementations for `AlertStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::types::AlertStatistics;

impl Clone for AlertStatistics {
    fn clone(&self) -> Self {
        Self {
            total_alerts_generated: AtomicU64::new(
                self.total_alerts_generated.load(Ordering::Relaxed),
            ),
            alerts_by_severity: Arc::clone(&self.alerts_by_severity),
            alerts_by_category: Arc::clone(&self.alerts_by_category),
            average_resolution_time: Arc::clone(&self.average_resolution_time),
            false_positive_rate: Arc::clone(&self.false_positive_rate),
            escalation_rates: Arc::clone(&self.escalation_rates),
            notification_delivery_rates: Arc::clone(&self.notification_delivery_rates),
            alert_frequency: Arc::clone(&self.alert_frequency),
        }
    }
}
