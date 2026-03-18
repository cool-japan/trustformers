//! Topology types for resource modeling
//!
//! Types for system topology analysis including NUMA, cache, memory,
//! PCI, storage, and network topology.

use crate::performance_optimizer::types::NumaTopology;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// Complete topology analysis results
///
/// Comprehensive topology analysis covering NUMA, cache, memory,
/// and I/O topology with detailed characterization and optimization hints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyAnalysisResults {
    /// NUMA topology
    pub numa_topology: Option<NumaTopology>,

    /// Cache analysis
    pub cache_analysis: CacheAnalysis,

    /// Memory topology
    pub memory_topology: MemoryTopology,

    /// I/O topology
    pub io_topology: IoTopology,

    /// Analysis timestamp
    pub analysis_timestamp: DateTime<Utc>,
}

/// Cache topology analysis and optimization
///
/// Detailed cache hierarchy analysis including level information,
/// coherency protocol, and total cache capacity for optimization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheAnalysis {
    /// Cache levels information
    pub cache_levels: Vec<CacheLevelInfo>,

    /// Cache coherency protocol
    pub cache_coherency_protocol: String,

    /// Total cache size
    pub total_cache_size_kb: u32,
}

/// Memory topology characterization
///
/// Memory subsystem topology including channel configuration,
/// DIMM layout, interleaving, and ECC capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryTopology {
    /// Number of memory channels
    pub memory_channels: usize,

    /// DIMM configuration
    pub dimm_configuration: Vec<DimmInfo>,

    /// Memory interleaving enabled
    pub interleaving_enabled: bool,

    /// ECC memory enabled
    pub ecc_enabled: bool,
}

/// I/O subsystem topology analysis
///
/// I/O topology including PCI layout, storage controllers,
/// and network infrastructure for I/O optimization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IoTopology {
    /// PCI topology
    pub pci_topology: PciTopology,

    /// Storage topology
    pub storage_topology: StorageTopology,

    /// Network topology
    pub network_topology: NetworkTopologyInfo,
}

/// Cache level detailed information
///
/// Comprehensive information about a specific cache level including
/// size, associativity, line size, and sharing characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLevelInfo {
    /// Cache level (1, 2, 3, etc.)
    pub level: u32,

    /// Cache size (KB)
    pub size_kb: u32,

    /// Cache associativity
    pub associativity: u32,

    /// Cache line size
    pub line_size: u32,

    /// Number of cores sharing this cache
    pub shared_by_cores: usize,
}

/// DIMM configuration information
///
/// Individual DIMM characteristics including slot position,
/// capacity, speed, and memory technology type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimmInfo {
    /// DIMM slot number
    pub slot: usize,

    /// DIMM size (MB)
    pub size_mb: u64,

    /// DIMM speed (MHz)
    pub speed_mhz: u32,

    /// Memory type
    pub memory_type: String,
}

/// PCI topology and device layout
///
/// PCI subsystem topology including root complexes, available lanes,
/// and connected devices for I/O optimization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PciTopology {
    /// Number of root complexes
    pub root_complex_count: usize,

    /// Total PCIe lanes
    pub pcie_lanes: usize,

    /// PCI devices
    pub devices: Vec<PciDeviceInfo>,
}

/// Storage subsystem topology
///
/// Storage topology including controllers, devices, and
/// interconnect layout for storage optimization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageTopology {
    /// Storage controllers
    pub controllers: Vec<StorageControllerInfo>,

    /// Storage devices
    pub devices: Vec<StorageDeviceInfo>,
}

/// Network topology information
///
/// Network infrastructure topology including interfaces,
/// routing configuration, and connectivity layout.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkTopologyInfo {
    /// Network interfaces
    pub interfaces: Vec<NetworkInterfaceInfo>,

    /// Routing table
    pub routing_table: Vec<RouteInfo>,
}

/// PCI device detailed information
///
/// Individual PCI device characteristics including identification,
/// class, and slot information for device-specific optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PciDeviceInfo {
    /// Device ID
    pub device_id: String,

    /// Vendor ID
    pub vendor_id: String,

    /// Device class
    pub device_class: String,

    /// PCIe slot
    pub pcie_slot: String,
}

/// Storage controller characteristics
///
/// Storage controller information including type, model,
/// and port configuration for storage optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageControllerInfo {
    /// Controller type
    pub controller_type: String,

    /// Controller model
    pub model: String,

    /// Number of ports
    pub port_count: usize,
}

/// Storage device detailed information
///
/// Individual storage device characteristics including path,
/// type, model, and capacity for storage optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageDeviceInfo {
    /// Device path
    pub device_path: String,

    /// Device type
    pub device_type: String,

    /// Device model
    pub model: String,

    /// Capacity
    pub capacity_gb: u64,
}

/// Network interface detailed information
///
/// Network interface characteristics including name, type,
/// hardware address, and IP configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterfaceInfo {
    /// Interface name
    pub name: String,

    /// Interface type
    pub interface_type: String,

    /// MAC address
    pub mac_address: String,

    /// IP addresses
    pub ip_addresses: Vec<String>,
}

/// Network routing information
///
/// Network route characteristics including destination,
/// gateway, interface, and routing metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteInfo {
    /// Destination network
    pub destination: String,

    /// Gateway
    pub gateway: String,

    /// Interface
    pub interface: String,

    /// Metric
    pub metric: u32,
}

/// Optimization hints and recommendations
///
/// System-generated optimization hints based on analysis
/// including priority, impact assessment, and implementation guidance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHints {
    /// NUMA optimization hints
    pub numa_hints: Vec<String>,

    /// Cache optimization hints
    pub cache_hints: Vec<String>,

    /// Memory optimization hints
    pub memory_hints: Vec<String>,

    /// I/O optimization hints
    pub io_hints: Vec<String>,

    /// Network optimization hints
    pub network_hints: Vec<String>,

    /// Priority scores
    pub priority_scores: HashMap<String, f32>,
}

/// Hardware capability detection results
///
/// Results of hardware capability detection including supported
/// features, performance characteristics, and optimization opportunities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    /// CPU capabilities
    pub cpu_capabilities: Vec<String>,

    /// Memory capabilities
    pub memory_capabilities: Vec<String>,

    /// I/O capabilities
    pub io_capabilities: Vec<String>,

    /// Network capabilities
    pub network_capabilities: Vec<String>,

    /// GPU capabilities
    pub gpu_capabilities: Vec<String>,

    /// Security capabilities
    pub security_capabilities: Vec<String>,
}

/// Performance baseline establishment
///
/// Baseline performance measurements for comparative analysis
/// and optimization effectiveness assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Baseline CPU performance
    pub cpu_baseline: f64,

    /// Baseline memory performance
    pub memory_baseline: f32,

    /// Baseline I/O performance
    pub io_baseline: f32,

    /// Baseline network performance
    pub network_baseline: f32,

    /// Baseline GPU performance
    pub gpu_baseline: Option<f64>,

    /// Baseline timestamp
    pub timestamp: DateTime<Utc>,

    /// Baseline validity duration
    pub validity_duration: Duration,
}

/// Resource allocation optimization
///
/// Optimized resource allocation recommendations based on
/// system topology, workload characteristics, and performance targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// CPU allocation strategy
    pub cpu_allocation: HashMap<String, usize>,

    /// Memory allocation strategy
    pub memory_allocation: HashMap<String, u64>,

    /// I/O allocation strategy
    pub io_allocation: HashMap<String, f32>,

    /// Network allocation strategy
    pub network_allocation: HashMap<String, f32>,

    /// Allocation effectiveness score
    pub effectiveness_score: f32,
}
