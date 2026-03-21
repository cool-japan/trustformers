//! Configuration management for mobile performance profiling.
//!
//! This module provides comprehensive configuration options for controlling
//! all aspects of mobile performance profiling including sampling rates,
//! monitoring parameters, export settings, and alert thresholds.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::{ExportFormat, ProfilingMode};

/// Main configuration for mobile performance profiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileProfilerConfig {
    /// Enable profiling
    pub enabled: bool,
    /// Profiling mode
    pub mode: ProfilingMode,
    /// Sampling configuration
    pub sampling: SamplingConfig,
    /// Memory profiling configuration
    pub memory_profiling: MemoryProfilingConfig,
    /// CPU profiling configuration
    pub cpu_profiling: CpuProfilingConfig,
    /// GPU profiling configuration
    pub gpu_profiling: GpuProfilingConfig,
    /// Network profiling configuration
    pub network_profiling: NetworkProfilingConfig,
    /// Real-time monitoring configuration
    pub real_time_monitoring: RealTimeConfig,
    /// Export configuration
    pub export_config: ExportConfig,
}

/// Sampling configuration for profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Sampling interval in milliseconds
    pub interval_ms: u64,
    /// Maximum samples to keep in memory
    pub max_samples: usize,
    /// Enable adaptive sampling based on activity
    pub adaptive_sampling: bool,
    /// High frequency sampling threshold in milliseconds
    pub high_freq_threshold_ms: u64,
    /// Low frequency sampling threshold in milliseconds
    pub low_freq_threshold_ms: u64,
    /// Enable burst sampling for critical events
    pub burst_sampling: bool,
    /// Burst duration in milliseconds
    pub burst_duration_ms: u64,
    /// Thermal sampling interval in milliseconds
    pub thermal_sampling_interval_ms: u64,
    /// Memory sampling interval in milliseconds
    pub memory_sampling_interval_ms: u64,
    /// CPU sampling interval in milliseconds
    pub cpu_sampling_interval_ms: u64,
    /// Battery sampling interval in milliseconds
    pub battery_sampling_interval_ms: u64,
}

/// Memory profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfilingConfig {
    /// Enable memory profiling
    pub enabled: bool,
    /// Track memory allocations
    pub track_allocations: bool,
    /// Track memory deallocations
    pub track_deallocations: bool,
    /// Enable memory leak detection
    pub leak_detection: bool,
    /// Memory pressure monitoring
    pub pressure_monitoring: bool,
    /// Heap analysis
    pub heap_analysis: bool,
    /// Stack trace depth for allocations
    pub stack_trace_depth: usize,
    /// Memory allocation size threshold for tracking
    pub allocation_threshold_bytes: usize,
    /// Enable garbage collection monitoring
    pub gc_monitoring: bool,
}

/// CPU profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfilingConfig {
    /// Enable CPU profiling
    pub enabled: bool,
    /// Track CPU usage by thread
    pub per_thread_tracking: bool,
    /// Enable thermal monitoring
    pub thermal_monitoring: bool,
    /// CPU frequency monitoring
    pub frequency_monitoring: bool,
    /// Core utilization tracking
    pub core_utilization: bool,
    /// Power consumption estimation
    pub power_estimation: bool,
    /// Sampling rate for CPU metrics (Hz)
    pub sampling_rate_hz: u32,
    /// Track CPU cache performance
    pub cache_performance: bool,
    /// Monitor instruction sets usage
    pub instruction_profiling: bool,
}

/// GPU profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfilingConfig {
    /// Enable GPU profiling
    pub enabled: bool,
    /// Track GPU memory usage
    pub memory_tracking: bool,
    /// GPU utilization monitoring
    pub utilization_monitoring: bool,
    /// Shader performance tracking
    pub shader_tracking: bool,
    /// GPU thermal monitoring
    pub thermal_monitoring: bool,
    /// Power consumption tracking
    pub power_tracking: bool,
    /// GPU frequency monitoring
    pub frequency_monitoring: bool,
    /// Track GPU memory bandwidth
    pub memory_bandwidth: bool,
    /// Monitor GPU command queue
    pub command_queue_monitoring: bool,
    /// GPU sampling interval in milliseconds
    pub sampling_interval_ms: u64,
    /// Enable GPU metrics collection
    pub enable_gpu_metrics: bool,
    /// Enable GPU memory tracking
    pub enable_memory_tracking: bool,
    /// Enable GPU performance counters
    pub enable_performance_counters: bool,
}

/// Network profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkProfilingConfig {
    /// Enable network profiling
    pub enabled: bool,
    /// Track bandwidth usage
    pub bandwidth_tracking: bool,
    /// Latency monitoring
    pub latency_monitoring: bool,
    /// Connection pool monitoring
    pub connection_monitoring: bool,
    /// Request/response analysis
    pub request_analysis: bool,
    /// Error rate tracking
    pub error_tracking: bool,
    /// Track specific protocols
    pub protocol_analysis: bool,
    /// Monitor DNS performance
    pub dns_monitoring: bool,
    /// SSL/TLS handshake monitoring
    pub ssl_monitoring: bool,
}

/// Real-time monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeConfig {
    /// Enable real-time monitoring
    pub enabled: bool,
    /// Update interval in milliseconds
    pub update_interval_ms: u64,
    /// Enable performance alerts
    pub performance_alerts: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Maximum history points for real-time display
    pub max_history_points: usize,
    /// Enable live streaming of metrics
    pub live_streaming: bool,
    /// WebSocket port for live streaming
    pub websocket_port: u16,
    /// Buffer size for real-time data
    pub buffer_size: usize,
}

/// Alert thresholds for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Memory usage threshold percentage
    pub memory_threshold_percent: f32,
    /// CPU usage threshold percentage
    pub cpu_threshold_percent: f32,
    /// GPU usage threshold percentage
    pub gpu_threshold_percent: f32,
    /// Temperature threshold in Celsius
    pub temperature_threshold_c: f32,
    /// Inference latency threshold in milliseconds
    pub latency_threshold_ms: f32,
    /// Battery level threshold percentage
    pub battery_threshold_percent: f32,
    /// Network error rate threshold percentage
    pub network_error_threshold_percent: f32,
    /// Frame rate threshold (FPS)
    pub frame_rate_threshold_fps: f32,
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Enable automatic export
    pub auto_export: bool,
    /// Export format
    pub format: ExportFormat,
    /// Export directory path
    pub export_directory: String,
    /// Include raw profiling data
    pub include_raw_data: bool,
    /// Include visualizations
    pub include_visualizations: bool,
    /// Compression level (0-9)
    pub compression_level: u8,
    /// Maximum file size for exports (MB)
    pub max_file_size_mb: u32,
    /// Export interval in seconds
    pub export_interval_sec: u64,
    /// Include system information
    pub include_system_info: bool,
    /// Custom export metadata
    pub custom_metadata: HashMap<String, String>,
}

/// Thermal monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalConfig {
    /// Enable thermal monitoring
    pub enabled: bool,
    /// Temperature sampling interval in milliseconds
    pub sampling_interval_ms: u64,
    /// Critical temperature threshold in Celsius
    pub critical_temp_c: f32,
    /// Warning temperature threshold in Celsius
    pub warning_temp_c: f32,
    /// Enable throttling detection
    pub throttling_detection: bool,
    /// Temperature sensors to monitor
    pub monitored_sensors: Vec<String>,
}

/// Battery monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryConfig {
    /// Enable battery monitoring
    pub enabled: bool,
    /// Battery sampling interval in milliseconds
    pub sampling_interval_ms: u64,
    /// Low battery threshold percentage
    pub low_battery_threshold: f32,
    /// Critical battery threshold percentage
    pub critical_battery_threshold: f32,
    /// Monitor charging state
    pub monitor_charging: bool,
    /// Monitor power consumption estimation
    pub power_estimation: bool,
}

/// Platform-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlatformConfig {
    /// iOS-specific configuration
    pub ios_config: Option<IosConfig>,
    /// Android-specific configuration
    pub android_config: Option<AndroidConfig>,
    /// Generic mobile configuration
    pub generic_config: GenericConfig,
}

/// iOS-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IosConfig {
    /// Enable Metal performance shaders monitoring
    pub metal_monitoring: bool,
    /// Core ML performance tracking
    pub coreml_tracking: bool,
    /// Instruments integration
    pub instruments_integration: bool,
    /// Xcode profiling support
    pub xcode_profiling: bool,
}

/// Android-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidConfig {
    /// Enable systrace integration
    pub systrace_integration: bool,
    /// GPU profiling via Adreno/Mali tools
    pub gpu_vendor_tools: bool,
    /// Android performance monitoring
    pub android_profiler: bool,
    /// Enable method tracing
    pub method_tracing: bool,
}

/// Generic mobile configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericConfig {
    /// Enable cross-platform features
    pub cross_platform_features: bool,
    /// Generic performance counters
    pub generic_counters: bool,
    /// Standard mobile optimizations
    pub standard_optimizations: bool,
}

/// Default implementations
impl Default for MobileProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: ProfilingMode::Development,
            sampling: SamplingConfig::default(),
            memory_profiling: MemoryProfilingConfig::default(),
            cpu_profiling: CpuProfilingConfig::default(),
            gpu_profiling: GpuProfilingConfig::default(),
            network_profiling: NetworkProfilingConfig::default(),
            real_time_monitoring: RealTimeConfig::default(),
            export_config: ExportConfig::default(),
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            interval_ms: 100, // 10 Hz sampling rate
            max_samples: 10000,
            adaptive_sampling: true,
            high_freq_threshold_ms: 50,
            low_freq_threshold_ms: 1000,
            burst_sampling: true,
            burst_duration_ms: 500,
            thermal_sampling_interval_ms: 1000, // 1 second
            memory_sampling_interval_ms: 500,   // 0.5 seconds
            cpu_sampling_interval_ms: 200,      // 0.2 seconds
            battery_sampling_interval_ms: 5000, // 5 seconds
        }
    }
}

impl Default for MemoryProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            track_allocations: true,
            track_deallocations: true,
            leak_detection: true,
            pressure_monitoring: true,
            heap_analysis: false, // Expensive operation
            stack_trace_depth: 10,
            allocation_threshold_bytes: 1024, // Track allocations > 1KB
            gc_monitoring: true,
        }
    }
}

impl Default for CpuProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            per_thread_tracking: false, // Can be expensive
            thermal_monitoring: true,
            frequency_monitoring: true,
            core_utilization: true,
            power_estimation: true,
            sampling_rate_hz: 10,
            cache_performance: false,
            instruction_profiling: false,
        }
    }
}

impl Default for GpuProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_tracking: true,
            utilization_monitoring: true,
            shader_tracking: false, // Platform dependent
            thermal_monitoring: true,
            power_tracking: true,
            frequency_monitoring: true,
            memory_bandwidth: false,
            command_queue_monitoring: false,
            sampling_interval_ms: 200, // 0.2 seconds
            enable_gpu_metrics: true,
            enable_memory_tracking: true,
            enable_performance_counters: true,
        }
    }
}

impl Default for NetworkProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bandwidth_tracking: true,
            latency_monitoring: true,
            connection_monitoring: true,
            request_analysis: false, // Privacy sensitive
            error_tracking: true,
            protocol_analysis: false,
            dns_monitoring: true,
            ssl_monitoring: false,
        }
    }
}

impl Default for RealTimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_interval_ms: 1000, // 1 second updates
            performance_alerts: true,
            alert_thresholds: AlertThresholds::default(),
            max_history_points: 300, // 5 minutes at 1 second intervals
            live_streaming: false,
            websocket_port: 8080,
            buffer_size: 1024,
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            memory_threshold_percent: 80.0,
            cpu_threshold_percent: 80.0,
            gpu_threshold_percent: 80.0,
            temperature_threshold_c: 70.0,
            latency_threshold_ms: 100.0,
            battery_threshold_percent: 20.0,
            network_error_threshold_percent: 5.0,
            frame_rate_threshold_fps: 30.0,
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            auto_export: false,
            format: ExportFormat::JSON,
            export_directory: "/tmp/profiling_data".to_string(),
            include_raw_data: true,
            include_visualizations: false,
            compression_level: 6,
            max_file_size_mb: 100,
            export_interval_sec: 300, // 5 minutes
            include_system_info: true,
            custom_metadata: HashMap::new(),
        }
    }
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sampling_interval_ms: 1000,
            critical_temp_c: 85.0,
            warning_temp_c: 70.0,
            throttling_detection: true,
            monitored_sensors: vec!["cpu".to_string(), "gpu".to_string()],
        }
    }
}

impl Default for BatteryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sampling_interval_ms: 5000, // 5 seconds
            low_battery_threshold: 20.0,
            critical_battery_threshold: 10.0,
            monitor_charging: true,
            power_estimation: true,
        }
    }
}

impl Default for GenericConfig {
    fn default() -> Self {
        Self {
            cross_platform_features: true,
            generic_counters: true,
            standard_optimizations: true,
        }
    }
}
