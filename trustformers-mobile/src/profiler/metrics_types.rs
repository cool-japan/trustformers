//! Raw performance metrics: CPU/GPU/memory/network/thermal/battery/inference snapshots.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Backend utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendUtilization {
    /// CPU backend utilization (%)
    pub cpu_percent: f32,
    /// GPU backend utilization (%)
    pub gpu_percent: Option<f32>,
    /// NPU backend utilization (%)
    pub npu_percent: Option<f32>,
    /// Custom backend utilization (%)
    pub custom_percent: Option<f32>,
}
impl Default for BackendUtilization {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            gpu_percent: None,
            npu_percent: None,
            custom_percent: None,
        }
    }
}
/// Battery and power metrics for profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryPowerMetrics {
    /// Current power consumption (mW)
    pub power_consumption_mw: f32,
    /// Battery drain rate (%/hour)
    pub drain_rate_percent_per_hour: f32,
    /// Power efficiency (inferences per mWh)
    pub power_efficiency: f32,
    /// Estimated battery life (minutes)
    pub estimated_life_minutes: u32,
}
/// CPU performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    /// CPU utilization percentage
    pub utilization_percent: f32,
    /// Per-core utilization
    pub per_core_utilization: Vec<f32>,
    /// CPU frequency (MHz)
    pub frequency_mhz: Vec<u32>,
    /// Context switches per second
    pub context_switches_per_sec: u32,
    /// CPU load average
    pub load_average: [f32; 3],
    /// Time spent in user mode (%)
    pub user_time_percent: f32,
    /// Time spent in kernel mode (%)
    pub kernel_time_percent: f32,
    /// Time spent idle (%)
    pub idle_time_percent: f32,
}
impl Default for CpuMetrics {
    fn default() -> Self {
        Self {
            utilization_percent: 0.0,
            per_core_utilization: vec![],
            frequency_mhz: vec![],
            context_switches_per_sec: 0,
            load_average: [0.0, 0.0, 0.0],
            user_time_percent: 0.0,
            kernel_time_percent: 0.0,
            idle_time_percent: 100.0,
        }
    }
}
/// Garbage collection metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcMetrics {
    /// Total GC time (ms)
    pub total_gc_time_ms: u64,
    /// GC frequency (per minute)
    pub gc_frequency_per_min: f32,
    /// Average GC pause time (ms)
    pub avg_pause_time_ms: f32,
    /// Memory freed by GC (MB)
    pub memory_freed_mb: usize,
}
/// GPU performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU utilization percentage
    pub utilization_percent: f32,
    /// GPU memory utilization (%)
    pub memory_utilization_percent: f32,
    /// GPU frequency (MHz)
    pub frequency_mhz: u32,
    /// GPU temperature (°C)
    pub temperature_celsius: f32,
    /// GPU power consumption (mW)
    pub power_consumption_mw: f32,
    /// Number of active shaders
    pub active_shaders: u32,
}
/// Inference-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    /// Inference latency (ms)
    pub latency_ms: f32,
    /// Throughput (inferences per second)
    pub throughput_ips: f32,
    /// Queue depth
    pub queue_depth: usize,
    /// Model loading time (ms)
    pub model_load_time_ms: f32,
    /// Memory footprint (MB)
    pub memory_footprint_mb: usize,
    /// Accuracy score
    pub accuracy_score: Option<f32>,
    /// Backend utilization
    pub backend_utilization: BackendUtilization,
}
impl Default for InferenceMetrics {
    fn default() -> Self {
        Self {
            latency_ms: 0.0,
            throughput_ips: 0.0,
            queue_depth: 0,
            model_load_time_ms: 0.0,
            memory_footprint_mb: 0,
            accuracy_score: None,
            backend_utilization: BackendUtilization::default(),
        }
    }
}
/// Memory performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Total memory usage (MB)
    pub total_usage_mb: usize,
    /// Available memory (MB)
    pub available_mb: usize,
    /// Memory pressure level
    pub pressure_level: MemoryPressureLevel,
    /// Page faults per second
    pub page_faults_per_sec: u32,
    /// Memory allocations per second
    pub allocations_per_sec: u32,
    /// Memory deallocations per second
    pub deallocations_per_sec: u32,
    /// Garbage collection metrics
    pub gc_metrics: Option<GcMetrics>,
}
impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            total_usage_mb: 0,
            available_mb: 0,
            pressure_level: MemoryPressureLevel::Low,
            page_faults_per_sec: 0,
            allocations_per_sec: 0,
            deallocations_per_sec: 0,
            gc_metrics: None,
        }
    }
}
/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    Low,
    Medium,
    High,
    Critical,
}
/// Metrics snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub platform_metrics: PlatformMetrics,
    pub inference_metrics: InferenceMetrics,
    pub thermal_metrics: Option<ThermalMetrics>,
    pub battery_metrics: Option<BatteryPowerMetrics>,
}
/// Network connection types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkConnectionType {
    WiFi,
    Cellular4G,
    Cellular5G,
    Ethernet,
    Offline,
    Unknown,
}
/// Network performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Network bytes sent (per second)
    pub bytes_sent_per_sec: u64,
    /// Network bytes received (per second)
    pub bytes_received_per_sec: u64,
    /// Network latency (ms)
    pub latency_ms: f32,
    /// Connection count
    pub connection_count: u32,
    /// Network errors per second
    pub errors_per_sec: u32,
    /// Connection type
    pub connection_type: NetworkConnectionType,
    /// Signal strength (dBm)
    pub signal_strength_dbm: Option<i32>,
}
impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            bytes_sent_per_sec: 0,
            bytes_received_per_sec: 0,
            latency_ms: 0.0,
            connection_count: 0,
            errors_per_sec: 0,
            connection_type: NetworkConnectionType::Unknown,
            signal_strength_dbm: None,
        }
    }
}
/// Platform-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlatformMetrics {
    /// CPU metrics
    pub cpu_metrics: CpuMetrics,
    /// GPU metrics
    pub gpu_metrics: Option<GpuMetrics>,
    /// Memory metrics
    pub memory_metrics: MemoryMetrics,
    /// Network metrics
    pub network_metrics: NetworkMetrics,
    /// Platform-specific metrics
    pub platform_specific: HashMap<String, f64>,
}
/// Thermal performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalMetrics {
    /// Current temperature (°C)
    pub temperature_celsius: f32,
    /// Thermal throttling level
    pub throttling_level: u8,
    /// Thermal pressure
    pub thermal_pressure: f32,
    /// Cooling effectiveness
    pub cooling_effectiveness: f32,
}
