//! Core configuration types for resource modeling
//!
//! Configuration structures for performance profiling, temperature monitoring,
//! utilization tracking, and hardware detection.

use serde::{Deserialize, Serialize};
use std::time::Duration;

// ResourceModelingConfig is defined in manager.rs and should be imported from there
// This avoids duplicate type definitions that cause E0308 errors

/// Configuration for performance profiling operations
///
/// Settings for controlling performance profiling behavior including
/// benchmark parameters, timeout values, and result caching options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Number of CPU benchmark iterations
    pub cpu_benchmark_iterations: usize,

    /// Enable GPU profiling
    pub enable_gpu_profiling: bool,

    /// Cache profiling results
    pub cache_results: bool,

    /// Profiling timeout
    pub profiling_timeout: Duration,
}

/// Temperature monitoring thresholds and limits
///
/// Threshold configuration for thermal monitoring including warning levels,
/// critical temperatures, and emergency shutdown points for system protection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureThresholds {
    /// Warning temperature threshold (Celsius)
    pub warning_temperature: f32,

    /// Critical temperature threshold (Celsius)
    pub critical_temperature: f32,

    /// Shutdown temperature threshold (Celsius)
    pub shutdown_temperature: f32,
}

/// Configuration for resource utilization tracking
///
/// Settings for tracking system resource utilization including sample rates,
/// history retention, and detailed monitoring options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilizationTrackingConfig {
    /// Sample interval for measurements
    pub sample_interval: Duration,

    /// History size (number of samples to keep)
    pub history_size: usize,

    /// Enable detailed tracking
    pub detailed_tracking: bool,
}

/// Configuration for hardware detection processes
///
/// Settings for hardware detection including vendor-specific detection,
/// caching options, and timeout values for detection operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareDetectionConfig {
    /// Enable Intel-specific detection
    pub enable_intel_detection: bool,

    /// Enable AMD-specific detection
    pub enable_amd_detection: bool,

    /// Enable NVIDIA-specific detection
    pub enable_nvidia_detection: bool,

    /// Cache detection results
    pub cache_detection_results: bool,

    /// Detection timeout
    pub detection_timeout: Duration,
}
