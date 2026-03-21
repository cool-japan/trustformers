//! Thresholds Type Definitions

use serde::{Deserialize, Serialize};
use std::time::Duration;

// Import types from sibling modules
use super::enums::{PressureLevel, ThresholdAction};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_usage_threshold: f64,
    pub memory_usage_threshold: f64,
    pub execution_time_threshold: Duration,
    pub failure_rate_threshold: f64,
    pub throughput_threshold: f64,
    pub resource_pressure_threshold: PressureLevel,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_threshold: 0.8,                           // 80%
            memory_usage_threshold: 0.85,                       // 85%
            execution_time_threshold: Duration::from_secs(300), // 5 minutes
            failure_rate_threshold: 0.1,                        // 10%
            throughput_threshold: 1.0,                          // 1 test per second minimum
            resource_pressure_threshold: PressureLevel::High,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceThreshold {
    /// Name of the metric to monitor
    pub metric_name: String,
    /// Threshold value
    pub threshold_value: f64,
    /// Comparison operator (greater_than, less_than, equals)
    pub comparison_operator: String,
    /// Action to take when threshold is exceeded
    pub action: ThresholdAction,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResourceThresholds {
    pub cpu_threshold: f64,
    pub memory_threshold: f64,
    pub disk_threshold: f64,
    pub network_threshold: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdCache {
    pub enabled: bool,
    pub ttl: Duration,
    pub max_entries: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub cpu_threshold: f64,
    pub memory_threshold: f64,
    pub latency_threshold_ms: f64,
    pub max_execution_time: Duration,
    pub max_memory_usage: u64,
    pub max_cpu_usage: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 0.8,
            memory_threshold: 0.85,
            latency_threshold_ms: 1000.0,
            max_execution_time: Duration::from_secs(300),
            max_memory_usage: 1024 * 1024 * 1024,
            max_cpu_usage: 85.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemPressureThresholds {
    pub cpu_pressure_threshold: f64,
    pub memory_pressure_threshold: f64,
    pub disk_pressure_threshold: f64,
    pub file_descriptor_threshold: u32,
}

impl Default for SystemPressureThresholds {
    fn default() -> Self {
        Self {
            cpu_pressure_threshold: 80.0,
            memory_pressure_threshold: 0.85,
            disk_pressure_threshold: 80.0,
            file_descriptor_threshold: 10_000,
        }
    }
}
