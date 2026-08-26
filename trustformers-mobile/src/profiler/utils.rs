//! Utility functions for mobile performance profiling.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::device_info::{MobileDeviceInfo, PerformanceTier};
use serde_json::json;
use trustformers_core::error::Result;

use super::collector::ProfileSnapshot;
use super::config_types::ProfilerConfig;
use super::metrics_types::{MemoryPressureLevel, MetricsSnapshot};

/// Utility functions for mobile performance profiling
pub struct MobileProfilerUtils;
impl MobileProfilerUtils {
    /// Create optimized profiler configuration for device
    pub fn create_optimized_config(device_info: &MobileDeviceInfo) -> ProfilerConfig {
        let mut config = ProfilerConfig::default();

        // Adjust based on device performance tier
        match device_info.performance_scores.overall_tier {
            PerformanceTier::VeryLow => {
                config.profiling_interval_ms = 10000; // 10 seconds
                config.metrics_config.sampling_rate_hz = 0; // Disabled
                config.max_history_size = 50;
                config.metrics_config.detailed_collection = false;
            },
            PerformanceTier::Low => {
                config.profiling_interval_ms = 8000; // 8 seconds
                config.metrics_config.sampling_rate_hz = 0; // Disabled
                config.max_history_size = 75;
                config.metrics_config.detailed_collection = false;
            },
            PerformanceTier::Budget => {
                config.profiling_interval_ms = 5000; // 5 seconds
                config.metrics_config.sampling_rate_hz = 1; // 1 Hz
                config.max_history_size = 100;
                config.metrics_config.detailed_collection = false;
            },
            PerformanceTier::Medium => {
                config.profiling_interval_ms = 3000; // 3 seconds
                config.metrics_config.sampling_rate_hz = 2; // 2 Hz
                config.max_history_size = 300;
                config.metrics_config.detailed_collection = false;
            },
            PerformanceTier::Mid => {
                config.profiling_interval_ms = 2000; // 2 seconds
                config.metrics_config.sampling_rate_hz = 5; // 5 Hz
                config.max_history_size = 500;
            },
            PerformanceTier::High => {
                config.profiling_interval_ms = 1000; // 1 second
                config.metrics_config.sampling_rate_hz = 10; // 10 Hz
                config.max_history_size = 1000;
                config.metrics_config.detailed_collection = true;
            },
            PerformanceTier::VeryHigh => {
                config.profiling_interval_ms = 750; // 750ms
                config.metrics_config.sampling_rate_hz = 15; // 15 Hz
                config.max_history_size = 1500;
                config.metrics_config.detailed_collection = true;
            },
            PerformanceTier::Flagship => {
                config.profiling_interval_ms = 500; // 500ms
                config.metrics_config.sampling_rate_hz = 20; // 20 Hz
                config.max_history_size = 2000;
                config.metrics_config.detailed_collection = true;
            },
        }

        // Enable GPU profiling if available
        if device_info.gpu_info.is_some() {
            config.metrics_config.collect_gpu = true;
        }

        config
    }

    /// Calculate performance efficiency score
    pub fn calculate_efficiency_score(metrics: &MetricsSnapshot) -> f32 {
        let cpu_efficiency = 100.0 - metrics.platform_metrics.cpu_metrics.utilization_percent;
        let memory_efficiency = match metrics.platform_metrics.memory_metrics.pressure_level {
            MemoryPressureLevel::Low => 100.0,
            MemoryPressureLevel::Medium => 75.0,
            MemoryPressureLevel::High => 50.0,
            MemoryPressureLevel::Critical => 25.0,
        };

        let inference_efficiency = if metrics.inference_metrics.latency_ms > 0.0 {
            (1000.0 / metrics.inference_metrics.latency_ms).min(100.0)
        } else {
            100.0
        };

        (cpu_efficiency + memory_efficiency + inference_efficiency) / 3.0
    }

    /// Export profiling data to Chrome trace format
    pub fn export_to_chrome_trace(snapshots: &[ProfileSnapshot]) -> Result<String> {
        // Implementation for Chrome trace format export
        let trace_data = json!({
            "traceEvents": snapshots.iter().map(|snapshot| {
                json!({
                    "name": "Performance Snapshot",
                    "ph": "X",
                    "ts": 0, // Would convert Instant to microseconds
                    "dur": 1000,
                    "pid": 1,
                    "tid": 1,
                    "args": {
                        "performance_score": snapshot.performance_score,
                        "cpu_usage": snapshot.metrics.platform_metrics.cpu_metrics.utilization_percent,
                        "memory_pressure": snapshot.metrics.platform_metrics.memory_metrics.pressure_level,
                        "inference_latency": snapshot.metrics.inference_metrics.latency_ms
                    }
                })
            }).collect::<Vec<_>>()
        });

        Ok(trace_data.to_string())
    }
}
