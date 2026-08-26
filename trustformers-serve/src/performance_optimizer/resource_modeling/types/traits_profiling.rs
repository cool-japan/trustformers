//! Profiling types for resource modeling traits
//!
//! CPU, Memory, I/O, Network, and GPU profiling types used by
//! the resource modeling trait implementations.

use super::traits_analysis::MeasurementUnavailable;
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CPU Profiling Types
// ============================================================================

/// CPU vendor detector for identifying CPU manufacturer and features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuVendorDetector {
    /// Detected vendor
    pub vendor: String,
    /// CPU model
    pub model: String,
    /// Feature flags
    pub features: Vec<String>,
}

impl Default for CpuVendorDetector {
    fn default() -> Self {
        Self {
            vendor: String::from("Unknown"),
            model: String::from("Unknown"),
            features: Vec::new(),
        }
    }
}

impl CpuVendorDetector {
    /// Create new CPU vendor detector
    pub fn new() -> Self {
        Self::default()
    }

    /// Detect CPU capabilities and features
    pub fn detect_cpu_capabilities(&self) -> anyhow::Result<Vec<String>> {
        // Return detected features or default set
        Ok(self.features.clone())
    }
}

/// CPU benchmark suite for performance testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuBenchmarkSuite {
    /// Single-core score
    pub single_core_score: f64,
    /// Multi-core score
    pub multi_core_score: f64,
    /// Integer performance
    pub integer_score: f64,
    /// Floating point performance
    pub float_score: f64,
}

impl Default for CpuBenchmarkSuite {
    fn default() -> Self {
        Self {
            single_core_score: 0.0,
            multi_core_score: 0.0,
            integer_score: 0.0,
            float_score: 0.0,
        }
    }
}

impl CpuBenchmarkSuite {
    /// Create new CPU benchmark suite
    pub fn new() -> Self {
        Self::default()
    }

    /// Execute comprehensive CPU benchmarks.
    ///
    /// Returns [`MeasurementUnavailable`]. This used to assign the literals
    /// `1000.0`, `8000.0`, `1200.0` and `900.0` to the four score fields and
    /// report success — the same "benchmark result" on a laptop and on a
    /// 128-core server, obtained without executing a single instruction of
    /// benchmark. The fields keep their `0.0` defaults instead.
    pub fn execute_comprehensive_benchmarks(&mut self) -> anyhow::Result<()> {
        Err(MeasurementUnavailable::raise(
            "CPU benchmark scores",
            "no CPU benchmark harness is linked into trustformers-serve",
        ))
    }
}

/// CPU profiling state
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CpuProfilingState {
    /// Profiling active
    pub active: bool,
    /// Start time
    #[serde(skip)]
    pub start_time: std::time::Instant,
    /// Samples collected
    pub samples_collected: u64,
}

impl Default for CpuProfilingState {
    fn default() -> Self {
        Self {
            active: false,
            start_time: std::time::Instant::now(),
            samples_collected: 0,
        }
    }
}

// ============================================================================
// Memory Profiling Types
// ============================================================================

/// Memory hierarchy analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHierarchyAnalyzer {
    /// L1 cache size
    pub l1_size: usize,
    /// L2 cache size
    pub l2_size: usize,
    /// L3 cache size
    pub l3_size: usize,
    /// Main memory size
    pub main_memory_size: usize,
}

impl Default for MemoryHierarchyAnalyzer {
    fn default() -> Self {
        Self {
            l1_size: 32768,
            l2_size: 262144,
            l3_size: 8388608,
            main_memory_size: 8589934592,
        }
    }
}

impl MemoryHierarchyAnalyzer {
    /// Create a new MemoryHierarchyAnalyzer with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze memory hierarchy
    pub fn analyze_memory_hierarchy(&self) -> anyhow::Result<(usize, usize, usize, usize)> {
        Ok((
            self.l1_size,
            self.l2_size,
            self.l3_size,
            self.main_memory_size,
        ))
    }
}

/// Memory bandwidth tester
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBandwidthTester {
    /// Read bandwidth (GB/s)
    pub read_bandwidth: f64,
    /// Write bandwidth (GB/s)
    pub write_bandwidth: f64,
    /// Copy bandwidth (GB/s)
    pub copy_bandwidth: f64,
}

impl Default for MemoryBandwidthTester {
    fn default() -> Self {
        Self {
            read_bandwidth: 0.0,
            write_bandwidth: 0.0,
            copy_bandwidth: 0.0,
        }
    }
}

impl MemoryBandwidthTester {
    /// Create a new MemoryBandwidthTester with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Test comprehensive memory bandwidth.
    ///
    /// Returns [`MeasurementUnavailable`]. The `25.0 / 20.0 / 22.0` GB/s this
    /// used to store and return were literals; no memory traffic was ever
    /// generated to measure them.
    pub fn test_comprehensive_bandwidth(&mut self) -> anyhow::Result<(f64, f64, f64)> {
        Err(MeasurementUnavailable::raise(
            "memory bandwidth",
            "no memory bandwidth benchmark is linked into trustformers-serve",
        ))
    }
}

/// Memory latency tester
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLatencyTester {
    /// L1 latency (ns)
    pub l1_latency: f64,
    /// L2 latency (ns)
    pub l2_latency: f64,
    /// L3 latency (ns)
    pub l3_latency: f64,
    /// Main memory latency (ns)
    pub main_latency: f64,
}

impl Default for MemoryLatencyTester {
    fn default() -> Self {
        Self {
            l1_latency: 1.0,
            l2_latency: 4.0,
            l3_latency: 12.0,
            main_latency: 80.0,
        }
    }
}

impl MemoryLatencyTester {
    /// Create a new MemoryLatencyTester with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Measure comprehensive memory latency.
    ///
    /// Returns [`MeasurementUnavailable`]. This echoed back the struct's own
    /// `Default` values (`1.0 / 4.0 / 12.0 / 80.0` ns) as though they had been
    /// measured; a pointer-chase harness is needed to obtain them for real and
    /// this crate carries none.
    pub fn measure_comprehensive_latency(&mut self) -> anyhow::Result<(f64, f64, f64, f64)> {
        Err(MeasurementUnavailable::raise(
            "memory latency",
            "no pointer-chase latency harness is linked into trustformers-serve",
        ))
    }
}

/// NUMA topology analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaTopologyAnalyzer {
    /// Number of NUMA nodes
    pub node_count: usize,
    /// CPUs per node
    pub cpus_per_node: Vec<usize>,
    /// Memory per node (bytes)
    pub memory_per_node: Vec<usize>,
}

impl Default for NumaTopologyAnalyzer {
    fn default() -> Self {
        Self {
            node_count: 1,
            cpus_per_node: vec![],
            memory_per_node: vec![],
        }
    }
}

impl NumaTopologyAnalyzer {
    /// Create a new NumaTopologyAnalyzer with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze NUMA performance characteristics
    pub fn analyze_numa_performance(&self) -> anyhow::Result<(usize, Vec<usize>, Vec<usize>)> {
        Ok((
            self.node_count,
            self.cpus_per_node.clone(),
            self.memory_per_node.clone(),
        ))
    }
}

/// Memory profiling state
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryProfilingState {
    /// Profiling active
    pub active: bool,
    /// Start time
    #[serde(skip)]
    pub start_time: std::time::Instant,
    /// Allocations tracked
    pub allocations_tracked: u64,
}

impl Default for MemoryProfilingState {
    fn default() -> Self {
        Self {
            active: false,
            start_time: std::time::Instant::now(),
            allocations_tracked: 0,
        }
    }
}

// ============================================================================
// I/O Profiling Types
// ============================================================================

/// Storage device analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageDeviceAnalyzer {
    /// Device type (SSD, HDD, NVMe)
    pub device_type: String,
    /// Read throughput (MB/s)
    pub read_throughput: f64,
    /// Write throughput (MB/s)
    pub write_throughput: f64,
    /// IOPS
    pub iops: u64,
}

impl Default for StorageDeviceAnalyzer {
    fn default() -> Self {
        Self {
            device_type: String::from("Unknown"),
            read_throughput: 0.0,
            write_throughput: 0.0,
            iops: 0,
        }
    }
}

impl StorageDeviceAnalyzer {
    /// Create a new StorageDeviceAnalyzer with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze storage devices.
    ///
    /// Returns [`MeasurementUnavailable`]. This declared every machine to have
    /// an `"NVMe"` device doing 3500 MB/s read, 3000 MB/s write and 500,000
    /// IOPS — a specification sheet, not a measurement, and wrong on any
    /// spinning disk. Enumerating the devices themselves is possible through
    /// `sysinfo::Disks` (see
    /// [`hardware_detector::StorageDetector`](crate::performance_optimizer::resource_modeling::hardware_detector::StorageDetector)),
    /// but their throughput and IOPS require running I/O against them.
    pub fn analyze_storage_devices(&mut self) -> anyhow::Result<(String, f64, f64, u64)> {
        Err(MeasurementUnavailable::raise(
            "storage device throughput and IOPS",
            "measuring them requires running I/O against the device",
        ))
    }
}

/// I/O pattern analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoPatternAnalyzer {
    /// Sequential access ratio
    pub sequential_ratio: f64,
    /// Random access ratio
    pub random_ratio: f64,
    /// Average request size
    pub avg_request_size: usize,
}

impl Default for IoPatternAnalyzer {
    fn default() -> Self {
        Self {
            sequential_ratio: 0.5,
            random_ratio: 0.5,
            avg_request_size: 4096,
        }
    }
}

impl IoPatternAnalyzer {
    /// Create a new IoPatternAnalyzer with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze I/O patterns.
    ///
    /// Returns [`MeasurementUnavailable`]. This returned the struct's own
    /// `Default` (an exact 50/50 sequential-to-random split at a 4 KiB request
    /// size) as a finding. Classifying access patterns needs a block-layer
    /// trace, which nothing here collects.
    pub fn analyze_io_patterns(&mut self) -> anyhow::Result<(f64, f64, usize)> {
        Err(MeasurementUnavailable::raise(
            "I/O access patterns",
            "no block-layer tracing source is wired into this analyzer",
        ))
    }
}

/// Queue depth optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueDepthOptimizer {
    /// Optimal queue depth
    pub optimal_depth: u32,
    /// Current queue depth
    pub current_depth: u32,
    /// Throughput at optimal depth
    pub optimal_throughput: f64,
}

impl Default for QueueDepthOptimizer {
    fn default() -> Self {
        Self {
            optimal_depth: 32,
            current_depth: 1,
            optimal_throughput: 0.0,
        }
    }
}

impl QueueDepthOptimizer {
    /// Create a new QueueDepthOptimizer with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Optimize queue depths.
    ///
    /// Returns [`MeasurementUnavailable`]. An "optimal" depth of 64 at 5000
    /// units of throughput was asserted for every device without sweeping a
    /// single depth; finding an optimum means measuring at several.
    pub fn optimize_queue_depths(&mut self) -> anyhow::Result<(u32, f64)> {
        Err(MeasurementUnavailable::raise(
            "the optimal I/O queue depth",
            "finding it requires sweeping depths against the real device",
        ))
    }
}

/// I/O latency analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoLatencyAnalyzer {
    /// Average read latency (ms)
    pub avg_read_latency: f64,
    /// Average write latency (ms)
    pub avg_write_latency: f64,
    /// P99 latency (ms)
    pub p99_latency: f64,
}

impl Default for IoLatencyAnalyzer {
    fn default() -> Self {
        Self {
            avg_read_latency: 0.0,
            avg_write_latency: 0.0,
            p99_latency: 0.0,
        }
    }
}

impl IoLatencyAnalyzer {
    /// Create a new IoLatencyAnalyzer with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze comprehensive I/O latency.
    ///
    /// Returns [`MeasurementUnavailable`]. The `0.5 / 0.7 / 2.5` ms figures
    /// were literals, and a p99 in particular cannot exist without a
    /// distribution of samples to take a percentile of.
    pub fn analyze_comprehensive_latency(&mut self) -> anyhow::Result<(f64, f64, f64)> {
        Err(MeasurementUnavailable::raise(
            "I/O latency percentiles",
            "no I/O latency samples are collected on this build",
        ))
    }
}

/// I/O profiling state
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IoProfilingState {
    /// Profiling active
    pub active: bool,
    /// Start time
    #[serde(skip)]
    pub start_time: std::time::Instant,
    /// I/O operations tracked
    pub operations_tracked: u64,
}

impl Default for IoProfilingState {
    fn default() -> Self {
        Self {
            active: false,
            start_time: std::time::Instant::now(),
            operations_tracked: 0,
        }
    }
}

// ============================================================================
// Network Profiling Types
// ============================================================================

/// Network interface analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterfaceAnalyzer {
    /// Interface name
    pub interface_name: String,
    /// Link speed (Mbps)
    pub link_speed: u64,
    /// MTU size
    pub mtu: u32,
}

impl Default for NetworkInterfaceAnalyzer {
    fn default() -> Self {
        Self {
            interface_name: String::from("eth0"),
            link_speed: 1000,
            mtu: 1500,
        }
    }
}

impl NetworkInterfaceAnalyzer {
    /// Create a new NetworkInterfaceAnalyzer with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze network interfaces.
    ///
    /// Returns [`MeasurementUnavailable`]. This echoed the struct's `Default`
    /// — an interface literally named `"eth0"` at 1000 Mbps with a 1500-byte
    /// MTU — regardless of what the machine actually has, and would have named
    /// `eth0` on a host with no such interface.
    pub fn analyze_network_interfaces(&mut self) -> anyhow::Result<(String, u64, u32)> {
        Err(MeasurementUnavailable::raise(
            "network interface link speed and MTU",
            "reading them requires a platform interface-query API this build does not use",
        ))
    }
}

/// Network bandwidth tester
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkBandwidthTester {
    /// Upload bandwidth (Mbps)
    pub upload_bandwidth: f64,
    /// Download bandwidth (Mbps)
    pub download_bandwidth: f64,
}

impl Default for NetworkBandwidthTester {
    fn default() -> Self {
        Self {
            upload_bandwidth: 0.0,
            download_bandwidth: 0.0,
        }
    }
}

impl NetworkBandwidthTester {
    /// Create a new NetworkBandwidthTester with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Test comprehensive network bandwidth.
    ///
    /// Returns [`MeasurementUnavailable`]. Reporting 950/980 Mbps required a
    /// peer to transfer against; none was ever contacted. Measuring bandwidth
    /// for real also means sending traffic, which a profiling call has no
    /// mandate to do unasked.
    pub fn test_comprehensive_bandwidth(&mut self) -> anyhow::Result<(f64, f64)> {
        Err(MeasurementUnavailable::raise(
            "network bandwidth",
            "measuring it requires transferring data against a peer",
        ))
    }
}

/// Network latency tester
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLatencyTester {
    /// Average latency (ms)
    pub avg_latency: f64,
    /// Jitter (ms)
    pub jitter: f64,
    /// Packet loss rate
    pub packet_loss: f64,
}

impl Default for NetworkLatencyTester {
    fn default() -> Self {
        Self {
            avg_latency: 0.0,
            jitter: 0.0,
            packet_loss: 0.0,
        }
    }
}

impl NetworkLatencyTester {
    /// Create a new NetworkLatencyTester with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze comprehensive network latency.
    ///
    /// Returns [`MeasurementUnavailable`]. Latency, jitter and packet loss all
    /// describe a path to a peer; the `1.5 ms / 0.3 ms / 0.1%` this reported
    /// described no path, because no packet was sent.
    pub fn analyze_comprehensive_latency(&mut self) -> anyhow::Result<(f64, f64, f64)> {
        Err(MeasurementUnavailable::raise(
            "network latency, jitter and packet loss",
            "measuring them requires probing a peer over the network",
        ))
    }
}

/// MTU optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtuOptimizer {
    /// Optimal MTU size
    pub optimal_mtu: u32,
    /// Current MTU size
    pub current_mtu: u32,
}

impl Default for MtuOptimizer {
    fn default() -> Self {
        Self {
            optimal_mtu: 1500,
            current_mtu: 1500,
        }
    }
}

impl MtuOptimizer {
    /// Create a new MtuOptimizer with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Optimize MTU settings based on interface analysis.
    ///
    /// Returns [`MeasurementUnavailable`]. This recommended 9000-byte jumbo
    /// frames unconditionally — without reading `interface_analysis`, without
    /// checking whether the interface or the path supports them, and with a
    /// promised "15% throughput improvement, 5% latency reduction" that no
    /// experiment produced. On a path with a 1500-byte MTU that advice
    /// black-holes traffic.
    pub fn optimize_mtu_settings(
        &mut self,
        _interface_analysis: &NetworkInterfaceAnalysisResults,
    ) -> anyhow::Result<MtuOptimizationResults> {
        Err(MeasurementUnavailable::raise(
            "an optimal MTU",
            "choosing one requires path-MTU discovery and a throughput sweep",
        ))
    }
}

/// Network profiling state
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NetworkProfilingState {
    /// Profiling active
    pub active: bool,
    /// Start time
    #[serde(skip)]
    pub start_time: std::time::Instant,
    /// Packets analyzed
    pub packets_analyzed: u64,
}

impl Default for NetworkProfilingState {
    fn default() -> Self {
        Self {
            active: false,
            start_time: std::time::Instant::now(),
            packets_analyzed: 0,
        }
    }
}

// ============================================================================
// GPU Profiling Types
// ============================================================================

/// GPU vendor detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuVendorDetector {
    /// Detected vendor
    pub vendor: GpuVendor,
    /// GPU model
    pub model: String,
    /// Compute capability
    pub compute_capability: String,
}

impl Default for GpuVendorDetector {
    fn default() -> Self {
        Self {
            vendor: GpuVendor::Unknown,
            model: String::from("Unknown"),
            compute_capability: String::from("0.0"),
        }
    }
}

impl GpuVendorDetector {
    /// Create a new GpuVendorDetector with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Detect GPU capabilities.
    ///
    /// Returns [`MeasurementUnavailable`]. This returned the struct's
    /// `Default` — vendor `Unknown`, model `"Unknown"`, compute capability
    /// `"0.0"` — as a detection result, so a caller could not tell a machine
    /// with no GPU from a machine whose GPU was never looked at.
    pub fn detect_gpu_capabilities(&mut self) -> anyhow::Result<(GpuVendor, String, String)> {
        Err(MeasurementUnavailable::raise(
            "GPU vendor and compute capability",
            "no GPU driver query is linked into trustformers-serve",
        ))
    }
}

/// GPU vendor enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuVendor {
    /// NVIDIA GPU
    Nvidia,
    /// AMD GPU
    Amd,
    /// Intel GPU
    Intel,
    /// Apple GPU
    Apple,
    /// Unknown vendor
    Unknown,
    /// Other vendor
    Other,
}

impl Default for GpuVendor {
    fn default() -> Self {
        Self::Unknown
    }
}

/// GPU compute benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuComputeBenchmarks {
    /// Compute score
    pub compute_score: f64,
    /// Memory bandwidth score
    pub memory_bandwidth_score: f64,
    /// Texture processing score
    pub texture_score: f64,
}

impl Default for GpuComputeBenchmarks {
    fn default() -> Self {
        Self {
            compute_score: 0.0,
            memory_bandwidth_score: 0.0,
            texture_score: 0.0,
        }
    }
}

impl GpuComputeBenchmarks {
    /// Create a new GpuComputeBenchmarks with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Run comprehensive GPU benchmarks.
    ///
    /// Returns [`MeasurementUnavailable`]. The reported 15,000 GFLOPS,
    /// 600 GB/s and "peak = measured × 1.2" were literals and arithmetic on
    /// literals; `gpu_caps` was never read, so the same numbers came back for
    /// every device, including machines with no GPU at all.
    pub fn run_comprehensive_benchmarks(
        &mut self,
        _gpu_caps: &GpuCapabilityInfo,
    ) -> anyhow::Result<GpuComputePerformance> {
        Err(MeasurementUnavailable::raise(
            "GPU compute throughput",
            "no GPU compute benchmark is linked into trustformers-serve",
        ))
    }
}

/// GPU memory tester
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryTester {
    /// Total memory (bytes)
    pub total_memory: u64,
    /// Available memory (bytes)
    pub available_memory: u64,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
}

impl Default for GpuMemoryTester {
    fn default() -> Self {
        Self {
            total_memory: 0,
            available_memory: 0,
            memory_bandwidth: 0.0,
        }
    }
}

impl GpuMemoryTester {
    /// Create a new GpuMemoryTester with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Test comprehensive GPU memory performance.
    ///
    /// Returns [`MeasurementUnavailable`]. This asserted a 12 GiB device with
    /// 10 GiB free on every host — so the `utilization` it computed was
    /// arithmetic over two constants — plus a 600 GB/s bandwidth, a 100 ns
    /// latency and a 75 ns transfer overhead that nothing timed.
    pub fn test_comprehensive_memory_performance(
        &mut self,
    ) -> anyhow::Result<GpuMemoryPerformance> {
        Err(MeasurementUnavailable::raise(
            "GPU memory capacity and bandwidth",
            "no GPU driver query or memory benchmark is linked into trustformers-serve",
        ))
    }
}

/// GPU kernel analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuKernelAnalyzer {
    /// Kernel execution time (ms)
    pub execution_time: f64,
    /// Occupancy percentage
    pub occupancy: f64,
    /// Memory efficiency
    pub memory_efficiency: f64,
}

impl Default for GpuKernelAnalyzer {
    fn default() -> Self {
        Self {
            execution_time: 0.0,
            occupancy: 0.0,
            memory_efficiency: 0.0,
        }
    }
}

impl GpuKernelAnalyzer {
    /// Create a new GpuKernelAnalyzer with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze GPU kernel performance.
    ///
    /// Returns [`MeasurementUnavailable`]. There was no kernel: the analysis
    /// named one `"default_kernel"` and reported 85% occupancy, 90% memory
    /// efficiency and "typical" launch and context-switch overheads copied
    /// from vendor documentation rather than measured on this device.
    pub fn analyze_kernel_performance(&mut self) -> anyhow::Result<GpuKernelAnalysis> {
        Err(MeasurementUnavailable::raise(
            "GPU kernel occupancy and launch overhead",
            "no GPU kernel is launched or profiled on this build",
        ))
    }
}

/// GPU profiling state
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuProfilingState {
    /// Profiling active
    pub active: bool,
    /// Start time
    #[serde(skip)]
    pub start_time: std::time::Instant,
    /// Kernels profiled
    pub kernels_profiled: u64,
}

impl Default for GpuProfilingState {
    fn default() -> Self {
        Self {
            active: false,
            start_time: std::time::Instant::now(),
            kernels_profiled: 0,
        }
    }
}

/// GPU vendor-specific optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuVendorOptimizations {
    /// Vendor
    pub vendor: GpuVendor,
    /// Optimization flags
    pub optimization_flags: Vec<String>,
    /// Recommended settings
    pub recommended_settings: std::collections::HashMap<String, String>,
    /// Recommended CUDA version
    pub recommended_cuda_version: String,
    /// Optimal block sizes
    pub optimal_block_sizes: Vec<usize>,
    /// Memory coalescing hints
    pub memory_coalescing_hints: Vec<String>,
    /// Tensor core optimization
    pub tensor_core_optimization: bool,
    /// RT core optimization
    pub rt_core_optimization: bool,
    /// Vendor-specific flags
    pub vendor_specific_flags: HashMap<String, String>,
}

impl Default for GpuVendorOptimizations {
    fn default() -> Self {
        Self {
            vendor: GpuVendor::Unknown,
            optimization_flags: Vec::new(),
            recommended_settings: std::collections::HashMap::new(),
            recommended_cuda_version: String::new(),
            optimal_block_sizes: Vec::new(),
            memory_coalescing_hints: Vec::new(),
            tensor_core_optimization: false,
            rt_core_optimization: false,
            vendor_specific_flags: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CpuVendorDetector tests ---

    #[test]
    fn test_cpu_vendor_detector_default() {
        let detector = CpuVendorDetector::default();
        assert_eq!(detector.vendor, "Unknown");
        assert_eq!(detector.model, "Unknown");
        assert!(detector.features.is_empty());
    }

    #[test]
    fn test_cpu_vendor_detector_new() {
        let detector = CpuVendorDetector::new();
        assert_eq!(detector.vendor, "Unknown");
    }

    #[test]
    fn test_cpu_vendor_detector_detect_capabilities() {
        let detector = CpuVendorDetector {
            vendor: "Intel".to_string(),
            model: "i9-13900K".to_string(),
            features: vec!["avx2".to_string(), "sse4.2".to_string()],
        };
        let caps = detector.detect_cpu_capabilities();
        assert!(caps.is_ok());
        let features = caps.expect("should succeed");
        assert_eq!(features.len(), 2);
        assert_eq!(features[0], "avx2");
    }

    // --- CpuBenchmarkSuite tests ---

    #[test]
    fn test_cpu_benchmark_suite_default() {
        let suite = CpuBenchmarkSuite::default();
        assert!((suite.single_core_score - 0.0).abs() < 1e-9);
        assert!((suite.multi_core_score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cpu_benchmark_suite_new() {
        let suite = CpuBenchmarkSuite::new();
        assert!((suite.integer_score - 0.0).abs() < 1e-9);
    }

    /// Regression: `execute_comprehensive_benchmarks` used to write the
    /// literals 1000/8000/1200/900 into the four score fields and report
    /// success without running a benchmark.
    #[test]
    fn test_cpu_benchmark_suite_execute() {
        let mut suite = CpuBenchmarkSuite::new();
        let err = suite
            .execute_comprehensive_benchmarks()
            .expect_err("no CPU benchmark harness exists, so no scores can be reported");
        assert!(err.to_string().contains("CPU benchmark scores"), "{err}");
        // The scores must stay at their defaults: nothing measured them.
        assert!((suite.single_core_score - 0.0).abs() < 1e-9);
        assert!((suite.multi_core_score - 0.0).abs() < 1e-9);
    }

    // --- CpuProfilingState tests ---

    #[test]
    fn test_cpu_profiling_state_default() {
        let state = CpuProfilingState::default();
        assert!(!state.active);
        assert_eq!(state.samples_collected, 0);
    }

    // --- MemoryHierarchyAnalyzer tests ---

    #[test]
    fn test_memory_hierarchy_analyzer_default() {
        let analyzer = MemoryHierarchyAnalyzer::default();
        assert_eq!(analyzer.l1_size, 32768);
        assert_eq!(analyzer.l2_size, 262144);
        assert_eq!(analyzer.l3_size, 8388608);
    }

    #[test]
    fn test_memory_hierarchy_analyzer_new() {
        let analyzer = MemoryHierarchyAnalyzer::new();
        assert!(analyzer.main_memory_size > 0);
    }

    #[test]
    fn test_memory_hierarchy_analyze() {
        let analyzer = MemoryHierarchyAnalyzer::new();
        let result = analyzer.analyze_memory_hierarchy();
        assert!(result.is_ok());
        let (l1, l2, l3, main) = result.expect("should succeed");
        assert!(l1 < l2);
        assert!(l2 < l3);
        assert!(l3 < main);
    }

    // --- MemoryBandwidthTester tests ---

    #[test]
    fn test_memory_bandwidth_tester_default() {
        let tester = MemoryBandwidthTester::default();
        assert!((tester.read_bandwidth - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_memory_bandwidth_tester_new() {
        let tester = MemoryBandwidthTester::new();
        assert!((tester.write_bandwidth - 0.0).abs() < 1e-9);
    }

    /// Regression: this used to store 25/20/22 GB/s and return them as a
    /// bandwidth measurement.
    #[test]
    fn test_memory_bandwidth_test() {
        let mut tester = MemoryBandwidthTester::new();
        let err = tester
            .test_comprehensive_bandwidth()
            .expect_err("no bandwidth benchmark exists");
        assert!(err.to_string().contains("memory bandwidth"), "{err}");
        assert!((tester.read_bandwidth - 0.0).abs() < 1e-9);
        assert!((tester.write_bandwidth - 0.0).abs() < 1e-9);
        assert!((tester.copy_bandwidth - 0.0).abs() < 1e-9);
    }

    // --- MemoryLatencyTester tests ---

    #[test]
    fn test_memory_latency_tester_default() {
        let tester = MemoryLatencyTester::default();
        assert!((tester.l1_latency - 1.0).abs() < 1e-9);
        assert!(tester.l1_latency < tester.l2_latency);
        assert!(tester.l2_latency < tester.l3_latency);
        assert!(tester.l3_latency < tester.main_latency);
    }

    #[test]
    fn test_memory_latency_tester_new() {
        let tester = MemoryLatencyTester::new();
        assert!(tester.main_latency > 0.0);
    }

    /// Regression: this used to echo the struct's `Default` latencies back as
    /// a measurement.
    #[test]
    fn test_memory_latency_measure() {
        let mut tester = MemoryLatencyTester::new();
        let err = tester
            .measure_comprehensive_latency()
            .expect_err("no pointer-chase harness exists");
        assert!(err.to_string().contains("memory latency"), "{err}");
    }

    // --- NumaTopologyAnalyzer tests ---

    #[test]
    fn test_numa_topology_default() {
        let analyzer = NumaTopologyAnalyzer::default();
        assert_eq!(analyzer.node_count, 1);
        assert!(analyzer.cpus_per_node.is_empty());
    }

    #[test]
    fn test_numa_topology_analyze() {
        let analyzer = NumaTopologyAnalyzer {
            node_count: 2,
            cpus_per_node: vec![8, 8],
            memory_per_node: vec![16384, 16384],
        };
        let result = analyzer.analyze_numa_performance();
        assert!(result.is_ok());
        let (count, cpus, mem) = result.expect("should succeed");
        assert_eq!(count, 2);
        assert_eq!(cpus.len(), 2);
        assert_eq!(mem.len(), 2);
    }

    // --- StorageDeviceAnalyzer tests ---

    #[test]
    fn test_storage_device_analyzer_default() {
        let analyzer = StorageDeviceAnalyzer::default();
        assert_eq!(analyzer.device_type, "Unknown");
        assert_eq!(analyzer.iops, 0);
    }

    #[test]
    /// Regression: every machine used to be reported as an `"NVMe"` doing
    /// 3500/3000 MB/s at 500,000 IOPS.
    fn test_storage_device_analyzer_analyze() {
        let mut analyzer = StorageDeviceAnalyzer::new();
        let err = analyzer
            .analyze_storage_devices()
            .expect_err("device throughput cannot be measured without running I/O");
        assert!(err.to_string().contains("storage device"), "{err}");
        assert_eq!(analyzer.device_type, "Unknown");
        assert_eq!(analyzer.iops, 0);
    }

    // --- IoPatternAnalyzer tests ---

    #[test]
    fn test_io_pattern_analyzer_default() {
        let analyzer = IoPatternAnalyzer::default();
        assert!((analyzer.sequential_ratio - 0.5).abs() < 1e-9);
        assert!((analyzer.random_ratio - 0.5).abs() < 1e-9);
        assert_eq!(analyzer.avg_request_size, 4096);
    }

    /// Regression: this used to return the struct's own 50/50 default split
    /// as an observed access pattern.
    #[test]
    fn test_io_pattern_analyzer_analyze() {
        let mut analyzer = IoPatternAnalyzer::new();
        let err = analyzer
            .analyze_io_patterns()
            .expect_err("classifying access patterns needs a block-layer trace");
        assert!(err.to_string().contains("I/O access patterns"), "{err}");
    }

    // --- QueueDepthOptimizer tests ---

    #[test]
    fn test_queue_depth_optimizer_default() {
        let optimizer = QueueDepthOptimizer::default();
        assert_eq!(optimizer.optimal_depth, 32);
        assert_eq!(optimizer.current_depth, 1);
    }

    /// Regression: an "optimal" depth of 64 at 5000 throughput used to be
    /// asserted for every device without sweeping a single depth.
    #[test]
    fn test_queue_depth_optimize() {
        let mut optimizer = QueueDepthOptimizer::new();
        let err = optimizer
            .optimize_queue_depths()
            .expect_err("an optimum needs measurements at several depths");
        assert!(err.to_string().contains("queue depth"), "{err}");
        assert_eq!(optimizer.optimal_depth, 32);
    }

    // --- IoLatencyAnalyzer tests ---

    #[test]
    fn test_io_latency_analyzer_default() {
        let analyzer = IoLatencyAnalyzer::default();
        assert!((analyzer.avg_read_latency - 0.0).abs() < 1e-9);
    }

    /// Regression: 0.5/0.7/2.5 ms used to be reported as measured latencies,
    /// including a p99 with no sample distribution behind it.
    #[test]
    fn test_io_latency_analyzer_analyze() {
        let mut analyzer = IoLatencyAnalyzer::new();
        let err = analyzer
            .analyze_comprehensive_latency()
            .expect_err("a percentile needs samples");
        assert!(err.to_string().contains("I/O latency"), "{err}");
        assert!((analyzer.p99_latency - 0.0).abs() < 1e-9);
    }

    // --- NetworkInterfaceAnalyzer tests ---

    #[test]
    fn test_network_interface_analyzer_default() {
        let analyzer = NetworkInterfaceAnalyzer::default();
        assert_eq!(analyzer.interface_name, "eth0");
        assert_eq!(analyzer.link_speed, 1000);
        assert_eq!(analyzer.mtu, 1500);
    }

    /// Regression: this used to report an interface literally named `eth0` at
    /// 1000 Mbps, whatever the machine actually had.
    #[test]
    fn test_network_interface_analyze() {
        let mut analyzer = NetworkInterfaceAnalyzer::new();
        let err = analyzer
            .analyze_network_interfaces()
            .expect_err("no interface-query API is used on this build");
        assert!(err.to_string().contains("network interface"), "{err}");
    }

    // --- NetworkBandwidthTester tests ---

    #[test]
    fn test_network_bandwidth_tester_default() {
        let tester = NetworkBandwidthTester::default();
        assert!((tester.upload_bandwidth - 0.0).abs() < 1e-9);
    }

    /// Regression: 950/980 Mbps used to be reported without contacting a peer.
    #[test]
    fn test_network_bandwidth_test() {
        let mut tester = NetworkBandwidthTester::new();
        let err = tester
            .test_comprehensive_bandwidth()
            .expect_err("bandwidth needs a transfer against a peer");
        assert!(err.to_string().contains("network bandwidth"), "{err}");
        assert!((tester.upload_bandwidth - 0.0).abs() < 1e-9);
    }

    // --- NetworkLatencyTester tests ---

    #[test]
    fn test_network_latency_tester_default() {
        let tester = NetworkLatencyTester::default();
        assert!((tester.avg_latency - 0.0).abs() < 1e-9);
    }

    /// Regression: 1.5 ms latency, 0.3 ms jitter and 0.1% loss used to be
    /// reported for a path over which no packet had been sent.
    #[test]
    fn test_network_latency_analyze() {
        let mut tester = NetworkLatencyTester::new();
        let err = tester.analyze_comprehensive_latency().expect_err("latency needs a probe");
        assert!(err.to_string().contains("network latency"), "{err}");
        assert!((tester.avg_latency - 0.0).abs() < 1e-9);
    }

    // --- MtuOptimizer tests ---

    #[test]
    fn test_mtu_optimizer_default() {
        let optimizer = MtuOptimizer::default();
        assert_eq!(optimizer.optimal_mtu, 1500);
        assert_eq!(optimizer.current_mtu, 1500);
    }

    // --- GpuVendorDetector tests ---

    #[test]
    fn test_gpu_vendor_detector_default() {
        let detector = GpuVendorDetector::default();
        assert!(matches!(detector.vendor, GpuVendor::Unknown));
        assert_eq!(detector.model, "Unknown");
    }

    /// Regression: this used to return the `Unknown` default as a detection
    /// result, so "no GPU" and "never looked" were indistinguishable.
    #[test]
    fn test_gpu_vendor_detector_detect() {
        let mut detector = GpuVendorDetector::new();
        let err = detector
            .detect_gpu_capabilities()
            .expect_err("no GPU driver query is linked in");
        assert!(err.to_string().contains("GPU vendor"), "{err}");
    }

    // --- GpuVendor tests ---

    #[test]
    fn test_gpu_vendor_default() {
        let vendor = GpuVendor::default();
        assert!(matches!(vendor, GpuVendor::Unknown));
    }

    // --- GpuComputeBenchmarks tests ---

    #[test]
    fn test_gpu_compute_benchmarks_default() {
        let bench = GpuComputeBenchmarks::default();
        assert!((bench.compute_score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_gpu_compute_benchmarks_new() {
        let bench = GpuComputeBenchmarks::new();
        assert!((bench.memory_bandwidth_score - 0.0).abs() < 1e-9);
    }

    // --- GpuMemoryTester tests ---

    #[test]
    fn test_gpu_memory_tester_default() {
        let tester = GpuMemoryTester::default();
        assert_eq!(tester.total_memory, 0);
        assert_eq!(tester.available_memory, 0);
    }

    /// Regression: a 12 GiB device with 10 GiB free at 600 GB/s used to be
    /// reported on every host, GPU or not.
    #[test]
    fn test_gpu_memory_tester_test() {
        let mut tester = GpuMemoryTester::new();
        let err = tester
            .test_comprehensive_memory_performance()
            .expect_err("no GPU driver query is linked in");
        assert!(err.to_string().contains("GPU memory"), "{err}");
        assert_eq!(tester.total_memory, 0);
    }

    // --- GpuKernelAnalyzer tests ---

    #[test]
    fn test_gpu_kernel_analyzer_default() {
        let analyzer = GpuKernelAnalyzer::default();
        assert!((analyzer.execution_time - 0.0).abs() < 1e-9);
        assert!((analyzer.occupancy - 0.0).abs() < 1e-9);
    }

    /// Regression: an 85%-occupancy analysis of a kernel named
    /// `"default_kernel"` used to be returned without any kernel being run.
    #[test]
    fn test_gpu_kernel_analyzer_analyze() {
        let mut analyzer = GpuKernelAnalyzer::new();
        let err = analyzer
            .analyze_kernel_performance()
            .expect_err("no GPU kernel is launched on this build");
        assert!(err.to_string().contains("GPU kernel"), "{err}");
        assert!((analyzer.occupancy - 0.0).abs() < 1e-9);
    }

    // --- GpuVendorOptimizations tests ---

    #[test]
    fn test_gpu_vendor_optimizations_default() {
        let opts = GpuVendorOptimizations::default();
        assert!(matches!(opts.vendor, GpuVendor::Unknown));
        assert!(opts.optimization_flags.is_empty());
        assert!(!opts.tensor_core_optimization);
        assert!(!opts.rt_core_optimization);
    }

    // --- Profiling state tests ---

    #[test]
    fn test_memory_profiling_state_default() {
        let state = MemoryProfilingState::default();
        assert!(!state.active);
        assert_eq!(state.allocations_tracked, 0);
    }

    #[test]
    fn test_io_profiling_state_default() {
        let state = IoProfilingState::default();
        assert!(!state.active);
        assert_eq!(state.operations_tracked, 0);
    }

    #[test]
    fn test_network_profiling_state_default() {
        let state = NetworkProfilingState::default();
        assert!(!state.active);
        assert_eq!(state.packets_analyzed, 0);
    }

    #[test]
    fn test_gpu_profiling_state_default() {
        let state = GpuProfilingState::default();
        assert!(!state.active);
        assert_eq!(state.kernels_profiled, 0);
    }
}
