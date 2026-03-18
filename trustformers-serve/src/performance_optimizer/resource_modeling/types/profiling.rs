//! Profiling types for resource modeling
//!
//! Types for performance profiling including CPU, memory, I/O, network,
//! and GPU performance characterization.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// Comprehensive performance profile results
///
/// Complete performance characterization across all system components
/// with timestamp and comparative analysis capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfileResults {
    /// CPU performance profile
    pub cpu_profile: CpuProfile,

    /// Memory performance profile
    pub memory_profile: MemoryProfile,

    /// I/O performance profile
    pub io_profile: IoProfile,

    /// Network performance profile
    pub network_profile: NetworkProfile,

    /// GPU performance profile
    pub gpu_profile: Option<GpuProfile>,

    /// Profiling timestamp
    pub timestamp: DateTime<Utc>,
}

/// CPU performance characterization profile
///
/// Detailed CPU performance analysis including instruction throughput,
/// context switching costs, cache performance, and floating-point capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProfile {
    /// Instructions per second
    pub instructions_per_second: f64,

    /// Context switch overhead
    #[serde(skip)]
    pub context_switch_overhead: Duration,

    /// Thread creation overhead
    #[serde(skip)]
    pub thread_creation_overhead: Duration,

    /// Cache performance metrics
    pub cache_performance: CachePerformanceMetrics,

    /// Branch prediction accuracy
    pub branch_prediction_accuracy: f32,

    /// Floating point performance
    pub floating_point_performance: f64,

    /// Profiling duration
    #[serde(skip)]
    pub profiling_duration: Duration,

    /// Profile timestamp
    pub timestamp: DateTime<Utc>,
}

impl Default for CpuProfile {
    fn default() -> Self {
        Self {
            instructions_per_second: 0.0,
            context_switch_overhead: Duration::from_nanos(0),
            thread_creation_overhead: Duration::from_nanos(0),
            cache_performance: CachePerformanceMetrics::default(),
            branch_prediction_accuracy: 0.0,
            floating_point_performance: 0.0,
            profiling_duration: Duration::from_nanos(0),
            timestamp: Utc::now(),
        }
    }
}

/// Memory subsystem performance profile
///
/// Comprehensive memory performance analysis including bandwidth,
/// latency, cache hierarchy performance, and allocation characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    /// Memory bandwidth (GB/s)
    pub bandwidth_gbps: f32,

    /// Memory latency
    #[serde(skip)]
    pub latency: Duration,

    /// Cache performance
    pub cache_performance: CachePerformanceMetrics,

    /// Page fault overhead
    #[serde(skip)]
    pub page_fault_overhead: Duration,

    /// Memory allocation overhead
    #[serde(skip)]
    pub memory_allocation_overhead: Duration,

    /// Profiling duration
    #[serde(skip)]
    pub profiling_duration: Duration,

    /// Profile timestamp
    pub timestamp: DateTime<Utc>,
}

impl Default for MemoryProfile {
    fn default() -> Self {
        Self {
            bandwidth_gbps: 0.0,
            latency: Duration::from_nanos(0),
            cache_performance: CachePerformanceMetrics::default(),
            page_fault_overhead: Duration::from_nanos(0),
            memory_allocation_overhead: Duration::from_nanos(0),
            profiling_duration: Duration::from_nanos(0),
            timestamp: Utc::now(),
        }
    }
}

/// I/O subsystem performance profile
///
/// Detailed I/O performance analysis including sequential and random
/// performance, latency characteristics, and queue depth optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoProfile {
    /// Sequential read performance (MB/s)
    pub sequential_read_mbps: f32,

    /// Sequential write performance (MB/s)
    pub sequential_write_mbps: f32,

    /// Random read IOPS
    pub random_read_iops: u32,

    /// Random write IOPS
    pub random_write_iops: u32,

    /// Average I/O latency
    #[serde(skip)]
    pub average_latency: Duration,

    /// Queue depth performance
    pub queue_depth_performance: QueueDepthMetrics,

    /// Profiling duration
    #[serde(skip)]
    pub profiling_duration: Duration,

    /// Profile timestamp
    pub timestamp: DateTime<Utc>,
}

impl Default for IoProfile {
    fn default() -> Self {
        Self {
            sequential_read_mbps: 0.0,
            sequential_write_mbps: 0.0,
            random_read_iops: 0,
            random_write_iops: 0,
            average_latency: Duration::from_nanos(0),
            queue_depth_performance: QueueDepthMetrics::default(),
            profiling_duration: Duration::from_nanos(0),
            timestamp: Utc::now(),
        }
    }
}

/// Network subsystem performance profile
///
/// Network performance analysis including bandwidth, latency, packet loss,
/// MTU optimization, and connection establishment costs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkProfile {
    /// Network bandwidth (Mbps)
    pub bandwidth_mbps: f32,

    /// Network latency
    #[serde(skip)]
    pub latency: Duration,

    /// Packet loss rate
    pub packet_loss_rate: f32,

    /// MTU optimization
    pub mtu_optimization: MtuOptimizationMetrics,

    /// Connection overhead
    #[serde(skip)]
    pub connection_overhead: Duration,

    /// Profiling duration
    #[serde(skip)]
    pub profiling_duration: Duration,

    /// Profile timestamp
    pub timestamp: DateTime<Utc>,
}

impl Default for NetworkProfile {
    fn default() -> Self {
        Self {
            bandwidth_mbps: 0.0,
            latency: Duration::from_nanos(0),
            packet_loss_rate: 0.0,
            mtu_optimization: MtuOptimizationMetrics::default(),
            connection_overhead: Duration::from_nanos(0),
            profiling_duration: Duration::from_nanos(0),
            timestamp: Utc::now(),
        }
    }
}

/// GPU performance characterization profile
///
/// Comprehensive GPU performance analysis including compute capabilities,
/// memory bandwidth, kernel launch costs, and context switching overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfile {
    /// Compute performance (GFLOPS)
    pub compute_performance: f64,

    /// Memory bandwidth (GB/s)
    pub memory_bandwidth_gbps: f32,

    /// Kernel launch overhead
    #[serde(skip)]
    pub kernel_launch_overhead: Duration,

    /// Context switch overhead
    #[serde(skip)]
    pub context_switch_overhead: Duration,

    /// Memory transfer overhead
    #[serde(skip)]
    pub memory_transfer_overhead: Duration,

    /// Profiling duration
    #[serde(skip)]
    pub profiling_duration: Duration,

    /// Profile timestamp
    pub timestamp: DateTime<Utc>,
}

impl Default for GpuProfile {
    fn default() -> Self {
        Self {
            compute_performance: 0.0,
            memory_bandwidth_gbps: 0.0,
            kernel_launch_overhead: Duration::from_nanos(0),
            context_switch_overhead: Duration::from_nanos(0),
            memory_transfer_overhead: Duration::from_nanos(0),
            profiling_duration: Duration::from_nanos(0),
            timestamp: Utc::now(),
        }
    }
}

/// Cache hierarchy performance metrics
///
/// Detailed cache performance analysis across all cache levels
/// with hit rates, access latencies, and efficiency measurements.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CachePerformanceMetrics {
    /// L1 cache metrics
    #[serde(default)]
    pub l1_metrics: Option<CacheLevelMetrics>,

    /// L2 cache metrics
    #[serde(default)]
    pub l2_metrics: Option<CacheLevelMetrics>,

    /// L3 cache metrics
    #[serde(default)]
    pub l3_metrics: Option<CacheLevelMetrics>,

    /// Cache line size
    #[serde(default)]
    pub cache_line_size: u32,
}

/// Individual cache level performance metrics
///
/// Performance characteristics for a specific cache level including
/// size, access latency, and hit rate statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLevelMetrics {
    /// Cache size (KB)
    pub size_kb: u32,

    /// Access latency
    #[serde(skip)]
    pub access_latency: Duration,

    /// Hit rate
    pub hit_rate: f32,
}

impl Default for CacheLevelMetrics {
    fn default() -> Self {
        Self {
            size_kb: 0,
            access_latency: Duration::from_nanos(0),
            hit_rate: 0.0,
        }
    }
}

/// Queue depth optimization metrics
///
/// Analysis of I/O queue depth performance characteristics
/// for optimal queue depth determination and performance tuning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueueDepthMetrics {
    /// Optimal queue depth
    #[serde(default)]
    pub optimal_queue_depth: usize,

    /// Performance by queue depth
    #[serde(default)]
    pub performance_by_depth: HashMap<usize, f32>,
}

/// MTU optimization analysis metrics
///
/// Network MTU (Maximum Transmission Unit) optimization analysis
/// for optimal packet size determination and network performance tuning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MtuOptimizationMetrics {
    /// Optimal MTU size
    #[serde(default)]
    pub optimal_mtu: u32,

    /// Performance by MTU size
    #[serde(default)]
    pub performance_by_mtu: HashMap<u32, f32>,
}

/// Resource constraints and limitations
///
/// System resource constraints including limits, quotas, and
/// operational boundaries for optimization decision making.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// CPU usage limit (percentage)
    pub cpu_limit: Option<f32>,

    /// Memory usage limit (MB)
    pub memory_limit: Option<u64>,

    /// I/O bandwidth limit (MB/s)
    pub io_bandwidth_limit: Option<f32>,

    /// Network bandwidth limit (Mbps)
    pub network_bandwidth_limit: Option<f32>,

    /// Custom constraints
    pub custom_constraints: HashMap<String, f32>,
}
