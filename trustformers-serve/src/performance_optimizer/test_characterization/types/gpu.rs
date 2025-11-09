use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Import commonly used types from core

#[derive(Debug, Clone)]
pub struct GpuAlertEscalation {
    pub alert_level: String,
    pub escalation_threshold: f64,
    pub notification_recipients: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GpuAlertStatistics {
    pub total_alerts: usize,
    pub alerts_by_severity: HashMap<String, usize>,
    pub average_response_time: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct GpuAlertSystem {
    pub enabled: bool,
    pub alert_thresholds: HashMap<String, f64>,
    pub notification_channels: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GpuCapabilityInfo {
    pub compute_capability: String,
    pub max_threads_per_block: usize,
    pub shared_memory_per_block: usize,
    pub warp_size: usize,
}

#[derive(Debug, Clone)]
pub struct GpuComputeBenchmarks {
    pub flops: f64,
    pub memory_bandwidth_gbps: f64,
    pub kernel_launch_latency_us: f64,
}

#[derive(Debug, Clone)]
pub struct GpuComputePerformance {
    pub compute_throughput: f64,
    pub occupancy: f64,
    pub kernel_efficiency: f64,
}

#[derive(Debug, Clone)]
pub struct GpuDeviceModel {
    pub vendor: String,
    pub model_name: String,
    pub device_id: String,
    pub pci_bus_id: String,
}

#[derive(Debug, Clone)]
pub struct GpuKernelAnalysis {
    pub kernel_name: String,
    pub execution_time: std::time::Duration,
    pub occupancy: f64,
    pub register_usage: usize,
}

#[derive(Debug, Clone)]
pub struct GpuKernelAnalyzer {
    pub analysis_enabled: bool,
    pub kernels_analyzed: Vec<String>,
    pub optimization_suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GpuMemoryPerformance {
    pub bandwidth_utilization: f64,
    pub memory_throughput_gbps: f64,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Clone)]
pub struct GpuMemoryTester {
    pub test_size_bytes: usize,
    pub bandwidth_results: Vec<f64>,
    pub latency_results: Vec<std::time::Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU utilization percentage
    pub utilization: f64,
    /// GPU memory usage
    pub memory_usage: usize,
    /// GPU memory utilization
    pub memory_utilization: f64,
    /// GPU temperature
    pub temperature: f64,
    /// GPU power usage
    pub power_usage: f64,
    /// Compute unit utilization
    pub compute_utilization: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
    /// GPU frequency
    pub frequency: f64,
    /// Throttling indicators
    pub throttling: bool,
    /// Performance efficiency
    pub efficiency: f64,
}

#[derive(Debug, Clone)]
pub struct GpuProfilingState {
    pub profiling_active: bool,
    pub samples_collected: usize,
    pub profiling_start_time: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct GpuThermalAnalysis {
    pub current_temperature: f64,
    pub max_temperature: f64,
    pub thermal_throttling_detected: bool,
    pub cooling_efficiency: f64,
}

#[derive(Debug, Clone)]
pub struct GpuUtilizationCharacteristics {
    pub average_utilization: f64,
    pub peak_utilization: f64,
    pub utilization_variance: f64,
    pub idle_percentage: f64,
}

#[derive(Debug, Clone)]
pub struct GpuVendorDetector {
    pub detected_vendor: String,
    pub vendor_id: u32,
    pub detection_confidence: f64,
}

#[derive(Debug, Clone)]
pub struct GpuVendorOptimizations {
    pub vendor: String,
    pub optimizations_enabled: Vec<String>,
    pub performance_gain: f64,
}

// Trait implementations
