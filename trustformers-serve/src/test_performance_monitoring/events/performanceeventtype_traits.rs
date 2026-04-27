//! # PerformanceEventType - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceEventType`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::PerformanceEventType;

impl fmt::Display for PerformanceEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PerformanceEventType::TestStarted => write!(f, "TestStarted"),
            PerformanceEventType::TestCompleted => write!(f, "TestCompleted"),
            PerformanceEventType::TestFailed => write!(f, "TestFailed"),
            PerformanceEventType::TestTimeout => write!(f, "TestTimeout"),
            PerformanceEventType::MetricThresholdBreached => {
                write!(f, "MetricThresholdBreached")
            },
            PerformanceEventType::AnomalyDetected => write!(f, "AnomalyDetected"),
            PerformanceEventType::PerformanceRegression => {
                write!(f, "PerformanceRegression")
            },
            PerformanceEventType::ResourcePressure => write!(f, "ResourcePressure"),
            PerformanceEventType::SystemAlert => write!(f, "SystemAlert"),
            PerformanceEventType::ConfigurationChange => write!(f, "ConfigurationChange"),
            PerformanceEventType::BaselineUpdate => write!(f, "BaselineUpdate"),
            PerformanceEventType::TrendDetected => write!(f, "TrendDetected"),
            PerformanceEventType::PatternRecognized => write!(f, "PatternRecognized"),
            PerformanceEventType::OptimizationOpportunity => {
                write!(f, "OptimizationOpportunity")
            },
            PerformanceEventType::Custom { event_name } => write!(f, "{}", event_name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_test_started() {
        let et = PerformanceEventType::TestStarted;
        assert_eq!(format!("{}", et), "TestStarted");
    }

    #[test]
    fn test_display_test_completed() {
        let et = PerformanceEventType::TestCompleted;
        assert_eq!(format!("{}", et), "TestCompleted");
    }

    #[test]
    fn test_display_test_failed() {
        let et = PerformanceEventType::TestFailed;
        assert_eq!(format!("{}", et), "TestFailed");
    }

    #[test]
    fn test_display_test_timeout() {
        let et = PerformanceEventType::TestTimeout;
        assert_eq!(format!("{}", et), "TestTimeout");
    }

    #[test]
    fn test_display_metric_threshold_breached() {
        let et = PerformanceEventType::MetricThresholdBreached;
        assert_eq!(format!("{}", et), "MetricThresholdBreached");
    }

    #[test]
    fn test_display_anomaly_detected() {
        let et = PerformanceEventType::AnomalyDetected;
        assert_eq!(format!("{}", et), "AnomalyDetected");
    }

    #[test]
    fn test_display_performance_regression() {
        let et = PerformanceEventType::PerformanceRegression;
        assert_eq!(format!("{}", et), "PerformanceRegression");
    }

    #[test]
    fn test_display_resource_pressure() {
        let et = PerformanceEventType::ResourcePressure;
        assert_eq!(format!("{}", et), "ResourcePressure");
    }

    #[test]
    fn test_display_system_alert() {
        let et = PerformanceEventType::SystemAlert;
        assert_eq!(format!("{}", et), "SystemAlert");
    }

    #[test]
    fn test_display_configuration_change() {
        let et = PerformanceEventType::ConfigurationChange;
        assert_eq!(format!("{}", et), "ConfigurationChange");
    }

    #[test]
    fn test_display_baseline_update() {
        let et = PerformanceEventType::BaselineUpdate;
        assert_eq!(format!("{}", et), "BaselineUpdate");
    }

    #[test]
    fn test_display_trend_detected() {
        let et = PerformanceEventType::TrendDetected;
        assert_eq!(format!("{}", et), "TrendDetected");
    }

    #[test]
    fn test_display_pattern_recognized() {
        let et = PerformanceEventType::PatternRecognized;
        assert_eq!(format!("{}", et), "PatternRecognized");
    }

    #[test]
    fn test_display_optimization_opportunity() {
        let et = PerformanceEventType::OptimizationOpportunity;
        assert_eq!(format!("{}", et), "OptimizationOpportunity");
    }

    #[test]
    fn test_display_custom_event() {
        let et = PerformanceEventType::Custom {
            event_name: "my_custom_event".to_string(),
        };
        assert_eq!(format!("{}", et), "my_custom_event");
    }

    #[test]
    fn test_display_custom_event_empty_name() {
        let et = PerformanceEventType::Custom {
            event_name: String::new(),
        };
        assert_eq!(format!("{}", et), "");
    }

    #[test]
    fn test_display_custom_event_with_spaces() {
        let et = PerformanceEventType::Custom {
            event_name: "Custom Deploy Event".to_string(),
        };
        assert_eq!(format!("{}", et), "Custom Deploy Event");
    }

    #[test]
    fn test_all_non_custom_variants_non_empty() {
        let variants = vec![
            PerformanceEventType::TestStarted,
            PerformanceEventType::TestCompleted,
            PerformanceEventType::TestFailed,
            PerformanceEventType::TestTimeout,
            PerformanceEventType::MetricThresholdBreached,
            PerformanceEventType::AnomalyDetected,
            PerformanceEventType::PerformanceRegression,
            PerformanceEventType::ResourcePressure,
            PerformanceEventType::SystemAlert,
            PerformanceEventType::ConfigurationChange,
            PerformanceEventType::BaselineUpdate,
            PerformanceEventType::TrendDetected,
            PerformanceEventType::PatternRecognized,
            PerformanceEventType::OptimizationOpportunity,
        ];
        for v in &variants {
            let s = format!("{}", v);
            assert!(!s.is_empty(), "variant display should not be empty");
        }
    }

    #[test]
    fn test_variant_display_uniqueness() {
        let variants = vec![
            format!("{}", PerformanceEventType::TestStarted),
            format!("{}", PerformanceEventType::TestCompleted),
            format!("{}", PerformanceEventType::TestFailed),
            format!("{}", PerformanceEventType::AnomalyDetected),
            format!("{}", PerformanceEventType::SystemAlert),
        ];
        let unique: std::collections::HashSet<_> = variants.iter().collect();
        assert_eq!(variants.len(), unique.len());
    }

    #[test]
    fn test_custom_event_name_preserved() {
        let name = "special_monitoring_event_v2".to_string();
        let et = PerformanceEventType::Custom {
            event_name: name.clone(),
        };
        assert_eq!(format!("{}", et), name);
    }

    #[test]
    fn test_display_does_not_include_braces() {
        let et = PerformanceEventType::TestStarted;
        let s = format!("{}", et);
        assert!(!s.contains('{'));
        assert!(!s.contains('}'));
    }

    #[test]
    fn test_all_variants_are_string_representations() {
        let variants: Vec<(PerformanceEventType, &str)> = vec![
            (PerformanceEventType::TestStarted, "TestStarted"),
            (PerformanceEventType::TestFailed, "TestFailed"),
            (PerformanceEventType::ResourcePressure, "ResourcePressure"),
            (PerformanceEventType::TrendDetected, "TrendDetected"),
            (
                PerformanceEventType::OptimizationOpportunity,
                "OptimizationOpportunity",
            ),
        ];
        for (variant, expected) in variants {
            assert_eq!(format!("{}", variant), expected);
        }
    }
}
