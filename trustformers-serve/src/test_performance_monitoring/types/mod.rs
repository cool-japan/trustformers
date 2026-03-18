//! Type Definitions Module
//!
//! This module contains all type definitions used throughout the test performance
//! monitoring system, organized into logical sub-modules for better maintainability.
//!
//! ## Module Organization
//!
//! - `enums` - All enumeration types (100 types)
//! - `config` - Configuration structures (50 types)
//! - `utilities` - Utility types and common structures (93 types)
//! - `events` - Event-related types (30 types)
//! - `storage` - Storage and retention types (29 types)
//! - `metrics` - Metrics and measurement types (24 types)
//! - `reporting` - Report generation types (20 types)
//! - `analysis` - Analysis and detection types (17 types)
//! - `alerts` - Alert system types (15 types)
//! - `managers` - Manager and orchestrator types (14 types)
//! - `dashboard` - Dashboard and visualization types (12 types)
//! - `notifications` - Notification system types (11 types)
//! - `thresholds` - Threshold definition types (6 types)
//! - `execution` - Test execution types (4 types)

pub mod alerts;
pub mod analysis;
pub mod config;
pub mod dashboard;
pub mod enums;
pub mod events;
pub mod execution;
pub mod managers;
pub mod metrics;
pub mod notifications;
pub mod reporting;
pub mod storage;
pub mod thresholds;
pub mod utilities;

// Re-export all types for backward compatibility
pub use alerts::*;
pub use analysis::*;
pub use config::*;
pub use dashboard::*;
pub use enums::*;
pub use events::*;
pub use execution::*;
pub use managers::*;
pub use metrics::*;
pub use notifications::*;
pub use reporting::*;
pub use storage::*;
pub use thresholds::*;
pub use utilities::*;

// Re-export types from other modules
pub use super::historical_data::RetentionPolicy;
pub use super::reporting::Report;
pub use crate::performance_optimizer::real_time_metrics::TimestampedMetrics;

// Utility trait for metric conversion
pub trait IntoMetricValue {
    fn into_metric_value(self) -> MetricValue;
}

impl IntoMetricValue for i64 {
    fn into_metric_value(self) -> MetricValue {
        MetricValue::Integer(self)
    }
}

impl IntoMetricValue for f64 {
    fn into_metric_value(self) -> MetricValue {
        MetricValue::Float(self)
    }
}

impl IntoMetricValue for bool {
    fn into_metric_value(self) -> MetricValue {
        MetricValue::Boolean(self)
    }
}

impl IntoMetricValue for String {
    fn into_metric_value(self) -> MetricValue {
        MetricValue::String(self)
    }
}

impl IntoMetricValue for std::time::Duration {
    fn into_metric_value(self) -> MetricValue {
        MetricValue::Duration(self)
    }
}

impl IntoMetricValue for chrono::DateTime<chrono::Utc> {
    fn into_metric_value(self) -> MetricValue {
        MetricValue::Timestamp(self)
    }
}
