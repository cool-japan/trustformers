//! Hardware resource modeling for performance optimization.
//!
//! This module provides comprehensive system resource modeling capabilities
//! including CPU, Memory, IO, Network, and GPU modeling for performance
//! optimization and intelligent parallelism control.

use chrono::{DateTime, Utc};
use std::{collections::HashMap, time::Duration};

use super::types::{MemoryType, NetworkInterfaceStatus, NetworkInterfaceType, StorageDeviceType};

/// System resource model aggregating all hardware components
#[derive(Debug, Clone)]
pub struct SystemResourceModel {
    /// CPU model
    pub cpu_model: CpuModel,
    /// Memory model
    pub memory_model: MemoryModel,
    /// I/O model
    pub io_model: IoModel,
    /// Network model
    pub network_model: NetworkModel,
    /// GPU model
    pub gpu_model: Option<GpuModel>,
    /// Model last updated
    pub last_updated: DateTime<Utc>,
}

impl Default for SystemResourceModel {
    fn default() -> Self {
        Self {
            cpu_model: CpuModel::default(),
            memory_model: MemoryModel::default(),
            io_model: IoModel::default(),
            network_model: NetworkModel::default(),
            gpu_model: None,
            last_updated: Utc::now(),
        }
    }
}

/// CPU model characteristics
#[derive(Debug, Clone)]
pub struct CpuModel {
    /// Number of cores
    pub core_count: usize,
    /// Number of threads
    pub thread_count: usize,
    /// Base frequency (MHz)
    pub base_frequency_mhz: u32,
    /// Max frequency (MHz)
    pub max_frequency_mhz: u32,
    /// Cache hierarchy
    pub cache_hierarchy: CacheHierarchy,
    /// Performance characteristics
    pub performance_characteristics: CpuPerformanceCharacteristics,
}

impl Default for CpuModel {
    fn default() -> Self {
        Self {
            core_count: 8,
            thread_count: 16,
            base_frequency_mhz: 3000,
            max_frequency_mhz: 4000,
            cache_hierarchy: CacheHierarchy::default(),
            performance_characteristics: CpuPerformanceCharacteristics::default(),
        }
    }
}

/// Cache hierarchy
#[derive(Debug, Clone)]
pub struct CacheHierarchy {
    /// L1 cache size (KB)
    pub l1_cache_kb: u32,
    /// L2 cache size (KB)
    pub l2_cache_kb: u32,
    /// L3 cache size (KB)
    pub l3_cache_kb: Option<u32>,
    /// Cache line size (bytes)
    pub cache_line_size: u32,
}

impl Default for CacheHierarchy {
    fn default() -> Self {
        Self {
            l1_cache_kb: 32,
            l2_cache_kb: 256,
            l3_cache_kb: Some(16384),
            cache_line_size: 64,
        }
    }
}

/// CPU performance characteristics
#[derive(Debug, Clone)]
pub struct CpuPerformanceCharacteristics {
    /// Instructions per clock
    pub instructions_per_clock: f32,
    /// Context switch overhead
    pub context_switch_overhead: Duration,
    /// Thread creation overhead
    pub thread_creation_overhead: Duration,
    /// NUMA topology
    pub numa_topology: Option<NumaTopology>,
}

impl Default for CpuPerformanceCharacteristics {
    fn default() -> Self {
        Self {
            instructions_per_clock: 2.5,
            context_switch_overhead: Duration::from_micros(10),
            thread_creation_overhead: Duration::from_micros(100),
            numa_topology: None,
        }
    }
}

/// NUMA topology
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub node_count: usize,
    /// Cores per NUMA node
    pub cores_per_node: Vec<usize>,
    /// Inter-node latency
    pub inter_node_latency: Duration,
    /// Intra-node latency
    pub intra_node_latency: Duration,
}

/// Memory model characteristics
#[derive(Debug, Clone)]
pub struct MemoryModel {
    /// Total memory (MB)
    pub total_memory_mb: u64,
    /// Memory type
    pub memory_type: MemoryType,
    /// Memory speed (MHz)
    pub memory_speed_mhz: u32,
    /// Memory bandwidth (GB/s)
    pub bandwidth_gbps: f32,
    /// Memory latency
    pub latency: Duration,
    /// Page size (KB)
    pub page_size_kb: u32,
}

impl Default for MemoryModel {
    fn default() -> Self {
        Self {
            total_memory_mb: 16384,
            memory_type: MemoryType::Ddr4,
            memory_speed_mhz: 3200,
            bandwidth_gbps: 51.2,
            latency: Duration::from_nanos(100),
            page_size_kb: 4,
        }
    }
}

/// I/O model characteristics
#[derive(Debug, Clone)]
pub struct IoModel {
    /// Storage devices
    pub storage_devices: Vec<StorageDevice>,
    /// Total I/O bandwidth (MB/s)
    pub total_bandwidth_mbps: f32,
    /// Average I/O latency
    pub average_latency: Duration,
    /// I/O queue depth
    pub queue_depth: usize,
}

impl Default for IoModel {
    fn default() -> Self {
        Self {
            storage_devices: vec![StorageDevice::default()],
            total_bandwidth_mbps: 500.0,
            average_latency: Duration::from_millis(1),
            queue_depth: 32,
        }
    }
}

/// Storage device characteristics
#[derive(Debug, Clone)]
pub struct StorageDevice {
    /// Device type
    pub device_type: StorageDeviceType,
    /// Capacity (GB)
    pub capacity_gb: u64,
    /// Read bandwidth (MB/s)
    pub read_bandwidth_mbps: f32,
    /// Write bandwidth (MB/s)
    pub write_bandwidth_mbps: f32,
    /// Random read IOPS
    pub random_read_iops: u32,
    /// Random write IOPS
    pub random_write_iops: u32,
    /// Access latency
    pub access_latency: Duration,
}

impl Default for StorageDevice {
    fn default() -> Self {
        Self {
            device_type: StorageDeviceType::Ssd,
            capacity_gb: 512,
            read_bandwidth_mbps: 500.0,
            write_bandwidth_mbps: 450.0,
            random_read_iops: 100000,
            random_write_iops: 80000,
            access_latency: Duration::from_millis(1),
        }
    }
}

/// Network model characteristics
#[derive(Debug, Clone)]
pub struct NetworkModel {
    /// Network interfaces
    pub interfaces: Vec<NetworkInterface>,
    /// Total bandwidth (Mbps)
    pub total_bandwidth_mbps: f32,
    /// Network latency
    pub latency: Duration,
    /// Packet loss rate
    pub packet_loss_rate: f32,
}

impl Default for NetworkModel {
    fn default() -> Self {
        Self {
            interfaces: vec![NetworkInterface::default()],
            total_bandwidth_mbps: 1000.0,
            latency: Duration::from_millis(1),
            packet_loss_rate: 0.001,
        }
    }
}

/// Network interface characteristics
#[derive(Debug, Clone)]
pub struct NetworkInterface {
    /// Interface type
    pub interface_type: NetworkInterfaceType,
    /// Bandwidth (Mbps)
    pub bandwidth_mbps: f32,
    /// MTU size
    pub mtu_size: u32,
    /// Interface status
    pub status: NetworkInterfaceStatus,
}

impl Default for NetworkInterface {
    fn default() -> Self {
        Self {
            interface_type: NetworkInterfaceType::Ethernet,
            bandwidth_mbps: 1000.0,
            mtu_size: 1500,
            status: NetworkInterfaceStatus::Up,
        }
    }
}

/// GPU model characteristics
#[derive(Debug, Clone)]
pub struct GpuModel {
    /// GPU devices
    pub devices: Vec<GpuDeviceModel>,
    /// Total GPU memory (MB)
    pub total_memory_mb: u64,
    /// GPU compute capability
    pub compute_capability: f32,
    /// GPU utilization characteristics
    pub utilization_characteristics: GpuUtilizationCharacteristics,
}

impl Default for GpuModel {
    fn default() -> Self {
        Self {
            devices: vec![GpuDeviceModel::default()],
            total_memory_mb: 8192,
            compute_capability: 8.0,
            utilization_characteristics: GpuUtilizationCharacteristics::default(),
        }
    }
}

/// GPU device model
#[derive(Debug, Clone)]
pub struct GpuDeviceModel {
    /// Device ID
    pub device_id: usize,
    /// Device name
    pub device_name: String,
    /// Memory (MB)
    pub memory_mb: u64,
    /// Compute units
    pub compute_units: u32,
    /// Base clock (MHz)
    pub base_clock_mhz: u32,
    /// Boost clock (MHz)
    pub boost_clock_mhz: u32,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth_gbps: f32,
}

impl Default for GpuDeviceModel {
    fn default() -> Self {
        Self {
            device_id: 0,
            device_name: "Generic GPU".to_string(),
            memory_mb: 8192,
            compute_units: 64,
            base_clock_mhz: 1500,
            boost_clock_mhz: 1800,
            memory_bandwidth_gbps: 448.0,
        }
    }
}

/// GPU utilization characteristics
#[derive(Debug, Clone)]
pub struct GpuUtilizationCharacteristics {
    /// Context switch overhead
    pub context_switch_overhead: Duration,
    /// Memory transfer overhead
    pub memory_transfer_overhead: Duration,
    /// Kernel launch overhead
    pub kernel_launch_overhead: Duration,
    /// Maximum concurrent kernels
    pub max_concurrent_kernels: usize,
}

impl Default for GpuUtilizationCharacteristics {
    fn default() -> Self {
        Self {
            context_switch_overhead: Duration::from_micros(50),
            memory_transfer_overhead: Duration::from_micros(100),
            kernel_launch_overhead: Duration::from_micros(20),
            max_concurrent_kernels: 32,
        }
    }
}

impl SystemResourceModel {
    /// Create a new system resource model with detected hardware
    pub fn new() -> Self {
        Self::default()
    }

    /// Update CPU model with new information
    pub fn update_cpu_model(&mut self, cpu_model: CpuModel) {
        self.cpu_model = cpu_model;
        self.last_updated = Utc::now();
    }

    /// Update memory model with new information
    pub fn update_memory_model(&mut self, memory_model: MemoryModel) {
        self.memory_model = memory_model;
        self.last_updated = Utc::now();
    }

    /// Update I/O model with new information
    pub fn update_io_model(&mut self, io_model: IoModel) {
        self.io_model = io_model;
        self.last_updated = Utc::now();
    }

    /// Update network model with new information
    pub fn update_network_model(&mut self, network_model: NetworkModel) {
        self.network_model = network_model;
        self.last_updated = Utc::now();
    }

    /// Update GPU model with new information
    pub fn update_gpu_model(&mut self, gpu_model: Option<GpuModel>) {
        self.gpu_model = gpu_model;
        self.last_updated = Utc::now();
    }

    /// Get total system computational capacity score
    pub fn get_computational_capacity(&self) -> f32 {
        let cpu_score = self.cpu_model.core_count as f32
            * (self.cpu_model.max_frequency_mhz as f32 / 1000.0)
            * self.cpu_model.performance_characteristics.instructions_per_clock;

        let memory_score =
            (self.memory_model.total_memory_mb as f32 / 1024.0) * self.memory_model.bandwidth_gbps;

        let io_score = self.io_model.total_bandwidth_mbps / 10.0; // Scale down

        let gpu_score = if let Some(ref gpu) = self.gpu_model {
            gpu.compute_capability * (gpu.total_memory_mb as f32 / 1024.0)
        } else {
            0.0
        };

        cpu_score + memory_score + io_score + gpu_score
    }

    /// Estimate optimal parallelism based on hardware characteristics
    pub fn estimate_optimal_parallelism(&self, workload_intensity: f32) -> usize {
        let base_parallelism = self.cpu_model.thread_count;

        // Factor in memory bandwidth constraints
        let memory_factor = (self.memory_model.bandwidth_gbps / 50.0).min(2.0);

        // Factor in I/O characteristics
        let io_factor = (self.io_model.total_bandwidth_mbps / 500.0).min(1.5);

        // Apply workload intensity scaling
        let intensity_factor = workload_intensity.clamp(0.1, 2.0);

        let estimated =
            (base_parallelism as f32 * memory_factor * io_factor * intensity_factor) as usize;

        // Ensure reasonable bounds
        estimated.clamp(1, base_parallelism * 2)
    }

    /// Get memory pressure estimate for given parallelism level
    pub fn estimate_memory_pressure(&self, parallelism: usize, memory_per_task_mb: u64) -> f32 {
        let total_memory_needed = parallelism as u64 * memory_per_task_mb;
        let available_memory = self.memory_model.total_memory_mb * 8 / 10; // 80% usable

        if total_memory_needed <= available_memory {
            total_memory_needed as f32 / available_memory as f32
        } else {
            1.0 + ((total_memory_needed - available_memory) as f32 / available_memory as f32)
        }
    }

    /// Check if GPU acceleration is available and beneficial
    pub fn is_gpu_acceleration_beneficial(
        &self,
        task_characteristics: &HashMap<String, f32>,
    ) -> bool {
        if self.gpu_model.is_none() {
            return false;
        }

        let gpu_intensity = task_characteristics.get("gpu_intensity").copied().unwrap_or(0.0);
        let parallel_workload =
            task_characteristics.get("parallel_workload").copied().unwrap_or(0.0);

        gpu_intensity > 0.3 && parallel_workload > 0.5
    }

    /// Get system bottleneck analysis
    pub fn analyze_bottlenecks(
        &self,
        current_load: &HashMap<String, f32>,
    ) -> Vec<SystemBottleneck> {
        let mut bottlenecks = Vec::new();

        let cpu_utilization = current_load.get("cpu").copied().unwrap_or(0.0);
        let memory_utilization = current_load.get("memory").copied().unwrap_or(0.0);
        let io_utilization = current_load.get("io").copied().unwrap_or(0.0);
        let _network_utilization = current_load.get("network").copied().unwrap_or(0.0);

        if cpu_utilization > 0.8 {
            bottlenecks.push(SystemBottleneck {
                resource_type: "CPU".to_string(),
                severity: cpu_utilization,
                impact: "High CPU usage may limit parallelism effectiveness".to_string(),
                recommendation:
                    "Consider reducing parallelism or optimizing CPU-intensive operations"
                        .to_string(),
            });
        }

        if memory_utilization > 0.85 {
            bottlenecks.push(SystemBottleneck {
                resource_type: "Memory".to_string(),
                severity: memory_utilization,
                impact: "High memory usage may cause swapping and performance degradation"
                    .to_string(),
                recommendation: "Reduce memory usage per task or decrease parallelism".to_string(),
            });
        }

        if io_utilization > 0.9 {
            bottlenecks.push(SystemBottleneck {
                resource_type: "I/O".to_string(),
                severity: io_utilization,
                impact: "I/O saturation causing performance bottleneck".to_string(),
                recommendation: "Optimize I/O patterns or reduce I/O intensive operations"
                    .to_string(),
            });
        }

        bottlenecks
    }
}

/// System bottleneck analysis result
#[derive(Debug, Clone)]
pub struct SystemBottleneck {
    /// Resource type causing bottleneck
    pub resource_type: String,
    /// Severity of bottleneck (0.0 to 1.0+)
    pub severity: f32,
    /// Impact description
    pub impact: String,
    /// Recommendation for mitigation
    pub recommendation: String,
}

impl CpuModel {
    /// Get effective parallelism considering NUMA topology
    pub fn get_effective_parallelism(&self, target_parallelism: usize) -> usize {
        if let Some(ref numa) = self.performance_characteristics.numa_topology {
            // Prefer keeping tasks within NUMA nodes when possible
            let cores_per_node = numa.cores_per_node.iter().sum::<usize>();
            if target_parallelism <= cores_per_node {
                target_parallelism
            } else {
                // Distribute across NUMA nodes
                let nodes_needed = target_parallelism.div_ceil(cores_per_node);
                nodes_needed.min(numa.node_count) * cores_per_node
            }
        } else {
            target_parallelism.min(self.thread_count)
        }
    }

    /// Calculate context switch overhead for parallelism level
    pub fn calculate_context_switch_overhead(&self, parallelism: usize) -> Duration {
        if parallelism <= self.thread_count {
            // No overhead if within thread count
            Duration::ZERO
        } else {
            // Additional overhead for oversubscription
            let oversubscription_factor = parallelism as f32 / self.thread_count as f32;
            Duration::from_nanos(
                (self.performance_characteristics.context_switch_overhead.as_nanos() as f32
                    * oversubscription_factor) as u64,
            )
        }
    }
}

impl MemoryModel {
    /// Calculate memory bandwidth utilization for parallelism level
    pub fn calculate_bandwidth_utilization(
        &self,
        parallelism: usize,
        memory_per_task_mb: u64,
    ) -> f32 {
        let total_memory_used = parallelism as u64 * memory_per_task_mb;
        let memory_access_rate = total_memory_used as f32 / 1024.0; // Convert to GB

        // Estimate bandwidth utilization based on access patterns
        let estimated_bandwidth_usage = memory_access_rate * 0.1; // Assume 10% of data accessed per second
        estimated_bandwidth_usage / self.bandwidth_gbps
    }

    /// Check if memory configuration supports parallelism level
    pub fn supports_parallelism(&self, parallelism: usize, memory_per_task_mb: u64) -> bool {
        let total_memory_needed = parallelism as u64 * memory_per_task_mb;
        let available_memory = self.total_memory_mb * 85 / 100; // Leave 15% for system

        total_memory_needed <= available_memory
    }
}

impl IoModel {
    /// Calculate I/O contention for parallelism level
    pub fn calculate_io_contention(&self, parallelism: usize, io_per_task_mbps: f32) -> f32 {
        let total_io_demand = parallelism as f32 * io_per_task_mbps;
        if total_io_demand <= self.total_bandwidth_mbps {
            total_io_demand / self.total_bandwidth_mbps
        } else {
            1.0 + ((total_io_demand - self.total_bandwidth_mbps) / self.total_bandwidth_mbps)
        }
    }

    /// Get recommended I/O optimization for workload
    pub fn get_io_optimization_recommendation(&self, workload_pattern: &str) -> String {
        match workload_pattern {
            "sequential" => "Consider larger I/O buffer sizes and prefetching".to_string(),
            "random" => "Optimize for low latency and consider SSD storage".to_string(),
            "mixed" => "Balance between sequential and random optimizations".to_string(),
            _ => "Analyze I/O patterns for specific optimizations".to_string(),
        }
    }
}

impl NetworkModel {
    /// Calculate network utilization for distributed workload
    pub fn calculate_network_utilization(&self, parallelism: usize, data_per_task_mb: f32) -> f32 {
        let total_network_usage = parallelism as f32 * data_per_task_mb * 8.0; // Convert to Mbps
        total_network_usage / self.total_bandwidth_mbps
    }

    /// Check if network can handle distributed parallelism
    pub fn supports_distributed_parallelism(
        &self,
        parallelism: usize,
        data_per_task_mb: f32,
    ) -> bool {
        let utilization = self.calculate_network_utilization(parallelism, data_per_task_mb);
        utilization < 0.8 // Keep under 80% to avoid congestion
    }
}

impl GpuModel {
    /// Calculate optimal GPU parallelism
    pub fn calculate_optimal_gpu_parallelism(
        &self,
        task_characteristics: &HashMap<String, f32>,
    ) -> Option<usize> {
        let gpu_intensity = task_characteristics.get("gpu_intensity").copied().unwrap_or(0.0);

        if gpu_intensity < 0.3 {
            return None; // Not worth GPU acceleration
        }

        // Calculate based on compute units and memory
        let compute_parallelism = self.devices.iter()
            .map(|device| device.compute_units as usize * 32) // Assume 32 threads per compute unit
            .sum::<usize>();

        let memory_parallelism = (self.total_memory_mb / 256) as usize; // Assume 256MB per task

        Some(compute_parallelism.min(memory_parallelism))
    }

    /// Estimate GPU memory pressure
    pub fn estimate_gpu_memory_pressure(&self, parallelism: usize, memory_per_task_mb: u64) -> f32 {
        let total_memory_needed = parallelism as u64 * memory_per_task_mb;
        let available_memory = self.total_memory_mb * 9 / 10; // Leave 10% buffer

        if total_memory_needed <= available_memory {
            total_memory_needed as f32 / available_memory as f32
        } else {
            1.0 + ((total_memory_needed - available_memory) as f32 / available_memory as f32)
        }
    }
}
