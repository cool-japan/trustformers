//! # GpuProfilerConfig - Trait Implementations
//!
//! This module contains trait implementations for `GpuProfilerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{GpuAlertThresholds, GpuMonitorConfig, GpuProfilerConfig};

impl Default for GpuProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            profiling_interval_seconds: 10,
            enable_memory_profiling: true,
            enable_performance_profiling: true,
            enable_thermal_monitoring: true,
            enable_power_monitoring: true,
            data_retention_hours: 24,
            max_profile_samples: 8640,
            export_interval_seconds: 300,
            alert_thresholds: GpuAlertThresholds::default(),
            gpu_configs: vec![GpuMonitorConfig {
                gpu_id: 0,
                enabled: true,
                max_temperature_celsius: 85.0,
                max_power_watts: 300.0,
                max_memory_utilization: 0.95,
                max_compute_utilization: 0.95,
            }],
        }
    }
}
