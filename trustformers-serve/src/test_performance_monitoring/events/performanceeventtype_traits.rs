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
