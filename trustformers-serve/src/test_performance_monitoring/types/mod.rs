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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_f64(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn test_into_metric_value_i64_positive() {
        let val: i64 = 42;
        let mv = val.into_metric_value();
        if let MetricValue::Integer(v) = mv {
            assert_eq!(v, 42);
        } else {
            panic!("expected Integer variant");
        }
    }

    #[test]
    fn test_into_metric_value_i64_negative() {
        let val: i64 = -100;
        let mv = val.into_metric_value();
        if let MetricValue::Integer(v) = mv {
            assert_eq!(v, -100);
        } else {
            panic!("expected Integer variant");
        }
    }

    #[test]
    fn test_into_metric_value_f64_positive() {
        let val: f64 = 3.14;
        let mv = val.into_metric_value();
        if let MetricValue::Float(v) = mv {
            assert!((v - 3.14).abs() < f64::EPSILON);
        } else {
            panic!("expected Float variant");
        }
    }

    #[test]
    fn test_into_metric_value_f64_zero() {
        let val: f64 = 0.0;
        let mv = val.into_metric_value();
        if let MetricValue::Float(v) = mv {
            assert_eq!(v, 0.0);
        } else {
            panic!("expected Float variant");
        }
    }

    #[test]
    fn test_into_metric_value_bool_true() {
        let val: bool = true;
        let mv = val.into_metric_value();
        if let MetricValue::Boolean(v) = mv {
            assert!(v);
        } else {
            panic!("expected Boolean variant");
        }
    }

    #[test]
    fn test_into_metric_value_bool_false() {
        let val: bool = false;
        let mv = val.into_metric_value();
        if let MetricValue::Boolean(v) = mv {
            assert!(!v);
        } else {
            panic!("expected Boolean variant");
        }
    }

    #[test]
    fn test_into_metric_value_string() {
        let val: String = "hello_metric".to_string();
        let mv = val.into_metric_value();
        if let MetricValue::String(s) = mv {
            assert_eq!(s, "hello_metric");
        } else {
            panic!("expected String variant");
        }
    }

    #[test]
    fn test_into_metric_value_empty_string() {
        let val: String = String::new();
        let mv = val.into_metric_value();
        if let MetricValue::String(s) = mv {
            assert!(s.is_empty());
        } else {
            panic!("expected String variant");
        }
    }

    #[test]
    fn test_into_metric_value_duration() {
        let val = Duration::from_secs(60);
        let mv = val.into_metric_value();
        if let MetricValue::Duration(d) = mv {
            assert_eq!(d, Duration::from_secs(60));
        } else {
            panic!("expected Duration variant");
        }
    }

    #[test]
    fn test_into_metric_value_duration_zero() {
        let val = Duration::from_secs(0);
        let mv = val.into_metric_value();
        if let MetricValue::Duration(d) = mv {
            assert_eq!(d, Duration::from_secs(0));
        } else {
            panic!("expected Duration variant");
        }
    }

    #[test]
    fn test_into_metric_value_timestamp() {
        let now = chrono::Utc::now();
        let mv = now.into_metric_value();
        if let MetricValue::Timestamp(_ts) = mv {
            // valid timestamp
        } else {
            panic!("expected Timestamp variant");
        }
    }

    #[test]
    fn test_metric_value_integer_debug_format() {
        let mv = MetricValue::Integer(99);
        let debug_str = format!("{:?}", mv);
        assert!(debug_str.contains("99"));
    }

    #[test]
    fn test_metric_value_float_debug_format() {
        let mv = MetricValue::Float(2.718);
        let debug_str = format!("{:?}", mv);
        assert!(debug_str.contains("2.718") || debug_str.contains("Float"));
    }

    #[test]
    fn test_metric_value_array_construction() {
        let arr = MetricValue::Array(vec![
            MetricValue::Integer(1),
            MetricValue::Integer(2),
            MetricValue::Integer(3),
        ]);
        if let MetricValue::Array(items) = arr {
            assert_eq!(items.len(), 3);
        } else {
            panic!("expected Array variant");
        }
    }

    #[test]
    fn test_metric_value_object_construction() {
        let mut map = std::collections::HashMap::new();
        map.insert("key1".to_string(), MetricValue::Integer(42));
        map.insert("key2".to_string(), MetricValue::Float(1.5));
        let obj = MetricValue::Object(map);
        if let MetricValue::Object(m) = obj {
            assert_eq!(m.len(), 2);
        } else {
            panic!("expected Object variant");
        }
    }

    #[test]
    fn test_multiple_conversions_i64_sequence() {
        let mut lcg = Lcg::new(12345);
        for _ in 0..10 {
            let val = (lcg.next_f64() * 1000.0) as i64;
            let mv = val.into_metric_value();
            if let MetricValue::Integer(v) = mv {
                assert_eq!(v, val);
            } else {
                panic!("expected Integer variant");
            }
        }
    }

    #[test]
    fn test_multiple_conversions_f64_sequence() {
        let mut lcg = Lcg::new(67890);
        for _ in 0..10 {
            let val = lcg.next_f64() * 100.0;
            let mv = val.into_metric_value();
            if let MetricValue::Float(v) = mv {
                assert!((v - val).abs() < 1e-9);
            } else {
                panic!("expected Float variant");
            }
        }
    }

    #[test]
    fn test_into_metric_value_large_i64() {
        let val: i64 = i64::MAX;
        let mv = val.into_metric_value();
        if let MetricValue::Integer(v) = mv {
            assert_eq!(v, i64::MAX);
        } else {
            panic!("expected Integer variant");
        }
    }

    #[test]
    fn test_into_metric_value_min_i64() {
        let val: i64 = i64::MIN;
        let mv = val.into_metric_value();
        if let MetricValue::Integer(v) = mv {
            assert_eq!(v, i64::MIN);
        } else {
            panic!("expected Integer variant");
        }
    }

    #[test]
    fn test_into_metric_value_duration_millis() {
        let val = Duration::from_millis(500);
        let mv = val.into_metric_value();
        if let MetricValue::Duration(d) = mv {
            assert_eq!(d.as_millis(), 500);
        } else {
            panic!("expected Duration variant");
        }
    }
}
