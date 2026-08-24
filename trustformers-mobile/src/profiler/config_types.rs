//! Profiler, metrics-collection, bottleneck-detection and alert-threshold configuration.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};

use super::collector::ExportFormat;

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// CPU utilization alert threshold (%)
    pub cpu_threshold_percent: f32,
    /// Memory utilization alert threshold (%)
    pub memory_threshold_percent: f32,
    /// Latency alert threshold (ms)
    pub latency_threshold_ms: f32,
    /// Temperature alert threshold (°C)
    pub temperature_threshold_celsius: f32,
    /// Battery level alert threshold (%)
    pub battery_threshold_percent: u8,
    /// Power consumption alert threshold (mW)
    pub power_threshold_mw: f32,
}
impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold_percent: 90.0,
            memory_threshold_percent: 90.0,
            latency_threshold_ms: 500.0,
            temperature_threshold_celsius: 85.0,
            battery_threshold_percent: 20,
            power_threshold_mw: 5000.0, // 5W
        }
    }
}
/// Bottleneck detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckConfig {
    /// Enable CPU bottleneck detection
    pub detect_cpu_bottlenecks: bool,
    /// Enable memory bottleneck detection
    pub detect_memory_bottlenecks: bool,
    /// Enable I/O bottleneck detection
    pub detect_io_bottlenecks: bool,
    /// Enable thermal bottleneck detection
    pub detect_thermal_bottlenecks: bool,
    /// CPU threshold for bottleneck (%)
    pub cpu_threshold_percent: f32,
    /// Memory threshold for bottleneck (%)
    pub memory_threshold_percent: f32,
    /// Analysis window size
    pub analysis_window_samples: usize,
}
impl Default for BottleneckConfig {
    fn default() -> Self {
        Self {
            detect_cpu_bottlenecks: true,
            detect_memory_bottlenecks: true,
            detect_io_bottlenecks: true,
            detect_thermal_bottlenecks: true,
            cpu_threshold_percent: 80.0,
            memory_threshold_percent: 85.0,
            analysis_window_samples: 100,
        }
    }
}
/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Collect CPU metrics
    pub collect_cpu: bool,
    /// Collect GPU metrics
    pub collect_gpu: bool,
    /// Collect memory metrics
    pub collect_memory: bool,
    /// Collect network metrics
    pub collect_network: bool,
    /// Collect inference-specific metrics
    pub collect_inference: bool,
    /// Sampling rate (Hz)
    pub sampling_rate_hz: u32,
    /// Collect detailed metrics
    pub detailed_collection: bool,
}
impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            collect_cpu: true,
            collect_gpu: true,
            collect_memory: true,
            collect_network: true,
            collect_inference: true,
            sampling_rate_hz: 10, // 10 Hz
            detailed_collection: false,
        }
    }
}
/// Performance profiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable real-time profiling
    pub enable_realtime_profiling: bool,
    /// Profiling interval (ms)
    pub profiling_interval_ms: u64,
    /// Enable platform-specific profiler integration
    pub enable_platform_integration: bool,
    /// Maximum profile history size
    pub max_history_size: usize,
    /// Performance metrics to collect
    pub metrics_config: MetricsConfig,
    /// Bottleneck detection configuration
    pub bottleneck_config: BottleneckConfig,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Export profiling data
    pub enable_export: bool,
    /// Export format
    pub export_format: ExportFormat,
}
impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_realtime_profiling: true,
            profiling_interval_ms: 1000, // 1 second
            enable_platform_integration: true,
            max_history_size: 1000,
            metrics_config: MetricsConfig::default(),
            bottleneck_config: BottleneckConfig::default(),
            alert_thresholds: AlertThresholds::default(),
            enable_export: true,
            export_format: ExportFormat::JSON,
        }
    }
}
