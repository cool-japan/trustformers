//! Report structures for profiling results

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::events::{CpuBottleneckAnalysis, PerformanceBottleneck, ProfileStats};
use super::gpu::GpuKernelSummary;
use super::io_monitor::IoDeviceType;
use super::memory::{MemoryEfficiencyAnalysis, MemoryStats};

/// Profiler report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerReport {
    pub total_events: usize,
    pub total_runtime: Duration,
    pub statistics: HashMap<String, ProfileStats>,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub slowest_layers: Vec<(String, Duration)>,
    pub memory_efficiency: MemoryEfficiencyAnalysis,
    pub recommendations: Vec<String>,
}

/// Layer latency analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerLatencyAnalysis {
    pub layer_name: String,
    pub layer_type: String,
    pub total_time: Duration,
    pub cpu_percentage: f64,
    pub gpu_percentage: f64,
    pub memory_copy_percentage: f64,
    pub flops_per_second: f64,
    pub memory_bandwidth_utilization: f64,
    pub bottleneck_type: String,
}

/// Comprehensive performance analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub memory_stats: Option<MemoryStats>,
    pub io_bandwidth_stats: HashMap<IoDeviceType, f64>,
    pub layer_analysis: Vec<LayerLatencyAnalysis>,
    pub gpu_utilization: Option<f64>,
    pub cpu_bottlenecks: Vec<CpuBottleneckAnalysis>,
    pub total_gpu_kernels: usize,
    pub total_io_operations: usize,
    pub performance_score: f64,
    pub recommendations: Vec<String>,
}

/// Enhanced profiler report
#[derive(Debug, Serialize, Deserialize)]
pub struct EnhancedProfilerReport {
    pub basic_report: ProfilerReport,
    pub performance_analysis: PerformanceAnalysis,
    pub gpu_kernel_summary: GpuKernelSummary,
    pub memory_allocation_summary: MemoryAllocationSummary,
    pub io_performance_summary: super::io_monitor::IoPerformanceSummary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryAllocationSummary {
    pub total_allocations: usize,
    pub peak_memory_usage: usize,
    pub memory_efficiency: f64,
    pub largest_allocations: Vec<String>,
    pub memory_leaks: usize,
}
