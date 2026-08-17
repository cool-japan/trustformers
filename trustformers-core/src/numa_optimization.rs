//! NUMA (Non-Uniform Memory Access) optimization for TrustformeRS.
//!
//! Topology is read from the kernel on Linux (`/sys/devices/system/node`).
//! On platforms without a NUMA interface this module reports
//! [`ErrorKind::UnsupportedOperation`](crate::errors::ErrorKind::UnsupportedOperation)
//! rather than synthesising a topology: macOS in particular has no NUMA API and
//! a fabricated multi-node topology would produce misleading placement advice.
//!
//! Allocations made through [`NumaAllocator`] are real heap allocations with
//! real addresses. Binding a page to a specific node needs `mbind`/libnuma,
//! which this crate does not link, so [`NumaAllocation::bound_to_node`] records
//! whether the requested node was actually enforced (currently never on any
//! platform) instead of implying that it was.

use crate::errors::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::alloc::Layout;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

/// NUMA node information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumaNode {
    /// Kernel node index.
    pub node_id: u32,
    /// CPU cores belonging to this node.
    pub cpu_cores: Vec<u32>,
    /// Total memory attached to this node, in GiB.
    pub memory_size_gb: f64,
    /// Free memory on this node, in GiB.
    pub available_memory_gb: f64,
    /// Memory bandwidth in GB/s, when the platform reports it.
    ///
    /// `None` on Linux: `/sys` exposes no bandwidth figure, and inventing one
    /// would silently drive node selection.
    pub memory_bandwidth_gbps: Option<f64>,
    /// Relative access cost to each other node, from the ACPI SLIT table
    /// (`/sys/devices/system/node/nodeN/distance`). These are *relative
    /// distances* (10 = local), not nanoseconds.
    pub relative_distance: HashMap<u32, u32>,
    /// Whether the kernel currently reports the node as online.
    pub is_available: bool,
}

/// NUMA topology information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaTopology {
    pub nodes: HashMap<u32, NumaNode>,
    pub total_nodes: u32,
    pub total_cores: u32,
    pub total_memory_gb: f64,
    pub node_distances: HashMap<(u32, u32), u32>, // (from, to) -> distance
}

/// NUMA allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NumaStrategy {
    /// Allocate memory on the same node as the current thread
    LocalNode,
    /// Spread allocations across all available nodes
    Interleaved,
    /// Prefer specific nodes in order
    PreferredNodes(Vec<u32>),
    /// Custom strategy based on workload characteristics
    WorkloadAware,
    /// Bind to specific nodes
    Bind(Vec<u32>),
}

/// NUMA memory allocation policy
#[derive(Debug, Clone)]
pub struct NumaPolicy {
    pub strategy: NumaStrategy,
    pub strict: bool, // Fail if preferred nodes are not available
    pub fallback_strategy: Option<NumaStrategy>,
    pub large_page_support: bool,
    pub memory_prefetch: bool,
}

impl Default for NumaPolicy {
    fn default() -> Self {
        Self {
            strategy: NumaStrategy::LocalNode,
            strict: false,
            fallback_strategy: Some(NumaStrategy::Interleaved),
            large_page_support: true,
            memory_prefetch: false,
        }
    }
}

/// NUMA allocation tracking.
///
/// `address` is the address of a real heap allocation owned by the
/// [`NumaAllocator`]; it stays valid until [`NumaAllocator::deallocate`] is
/// called with the matching `allocation_id`.
#[derive(Debug, Clone)]
pub struct NumaAllocation {
    /// Identifier used to free this allocation.
    pub allocation_id: String,
    /// The node the allocation was *requested* on. See `bound_to_node`.
    pub node_id: u32,
    /// Size of the allocation in bytes.
    pub size_bytes: usize,
    /// Real address of the allocated block.
    pub address: usize,
    /// Alignment the block was allocated with.
    pub alignment: usize,
    /// Whether the pages were actually bound to `node_id`.
    ///
    /// Binding requires `mbind(2)`/libnuma, which this crate does not link, so
    /// this is currently always `false`: the allocation follows the kernel's
    /// default (first-touch) policy.
    pub bound_to_node: bool,
    /// When the allocation was made.
    pub allocation_time: std::time::SystemTime,
    /// Declared access pattern, used for placement scoring.
    pub access_pattern: AccessPattern,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AccessPattern {
    Sequential,
    Random,
    Strided(usize),
    HotCold { hot_ratio: f64 },
    ReadOnly,
    WriteOnly,
    ReadWrite,
    Interleaved,
}

/// Thread affinity configuration
#[derive(Debug, Clone)]
pub struct ThreadAffinity {
    pub thread_id: thread::ThreadId,
    pub preferred_nodes: Vec<u32>,
    pub cpu_cores: Vec<u32>,
    pub priority: ThreadPriority,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ThreadPriority {
    Low,
    Normal,
    High,
    RealTime,
}

/// NUMA-aware memory allocator
pub struct NumaAllocator {
    topology: Arc<RwLock<NumaTopology>>,
    allocations: Arc<Mutex<HashMap<String, NumaAllocation>>>,
    policies: Arc<RwLock<HashMap<String, NumaPolicy>>>,
    allocation_counter: Arc<Mutex<u64>>,
    performance_monitor: Arc<Mutex<NumaPerformanceMonitor>>,
}

/// Performance monitoring for NUMA operations
#[derive(Debug, Clone, Default)]
pub struct NumaPerformanceMonitor {
    pub allocation_stats: HashMap<u32, AllocationStats>,
    pub memory_bandwidth_usage: HashMap<u32, f64>,
    pub cross_node_traffic: HashMap<(u32, u32), u64>,
    pub cache_miss_rates: HashMap<u32, f64>,
    pub memory_latencies: HashMap<u32, Vec<u64>>,
}

#[derive(Debug, Default, Clone)]
pub struct AllocationStats {
    pub total_allocations: u64,
    pub total_bytes: u64,
    pub average_allocation_size: f64,
    pub peak_memory_usage: u64,
    pub current_memory_usage: u64,
    pub allocation_failures: u64,
}

impl NumaAllocator {
    pub fn new() -> Result<Self> {
        let topology = Self::detect_numa_topology()?;

        Ok(Self {
            topology: Arc::new(RwLock::new(topology)),
            allocations: Arc::new(Mutex::new(HashMap::new())),
            policies: Arc::new(RwLock::new(HashMap::new())),
            allocation_counter: Arc::new(Mutex::new(0)),
            performance_monitor: Arc::new(Mutex::new(NumaPerformanceMonitor::default())),
        })
    }

    /// Detect the real NUMA topology of the current system.
    ///
    /// Linux only: every field is read from `/sys/devices/system/node`. On any
    /// other platform this returns an `UnsupportedOperation` error, because
    /// there is no NUMA interface to read and a synthesised topology would
    /// drive placement decisions from fiction.
    pub fn detect_numa_topology() -> Result<NumaTopology> {
        #[cfg(target_os = "linux")]
        {
            Self::detect_numa_topology_linux()
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(crate::errors::unsupported_operation(
                "NUMA topology detection",
                format!(
                    "{} (only Linux exposes a NUMA topology interface; \
                     no topology is reported on this platform)",
                    std::env::consts::OS
                ),
            ))
        }
    }

    /// Read the topology from `/sys/devices/system/node`.
    #[cfg(target_os = "linux")]
    fn detect_numa_topology_linux() -> Result<NumaTopology> {
        use std::fs;

        let node_root = std::path::Path::new("/sys/devices/system/node");
        let entries = fs::read_dir(node_root).map_err(|error| {
            crate::errors::unsupported_operation(
                "NUMA topology detection",
                format!("cannot read {}: {}", node_root.display(), error),
            )
        })?;

        let mut node_ids: Vec<u32> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(index) = name.strip_prefix("node") {
                if let Ok(node_id) = index.parse::<u32>() {
                    node_ids.push(node_id);
                }
            }
        }
        node_ids.sort_unstable();

        if node_ids.is_empty() {
            return Err(crate::errors::unsupported_operation(
                "NUMA topology detection",
                format!("{} lists no NUMA nodes", node_root.display()),
            ));
        }

        let mut nodes = HashMap::new();
        let mut node_distances = HashMap::new();
        let mut total_cores = 0u32;
        let mut total_memory_gb = 0.0f64;

        for &node_id in &node_ids {
            let node_dir = node_root.join(format!("node{}", node_id));

            let cpu_cores = fs::read_to_string(node_dir.join("cpulist"))
                .ok()
                .map(|list| parse_cpu_list(list.trim()))
                .unwrap_or_default();
            total_cores += cpu_cores.len() as u32;

            let (memory_size_gb, available_memory_gb) =
                fs::read_to_string(node_dir.join("meminfo"))
                    .ok()
                    .map(|meminfo| parse_node_meminfo(&meminfo))
                    .unwrap_or((0.0, 0.0));
            total_memory_gb += memory_size_gb;

            let mut relative_distance = HashMap::new();
            if let Ok(distances) = fs::read_to_string(node_dir.join("distance")) {
                for (index, value) in distances.split_whitespace().enumerate() {
                    if let (Some(&other), Ok(distance)) =
                        (node_ids.get(index), value.parse::<u32>())
                    {
                        relative_distance.insert(other, distance);
                        node_distances.insert((node_id, other), distance);
                    }
                }
            }

            let is_available = fs::read_to_string(node_root.join("online"))
                .map(|online| parse_cpu_list(online.trim()).contains(&node_id))
                .unwrap_or(true);

            nodes.insert(
                node_id,
                NumaNode {
                    node_id,
                    cpu_cores,
                    memory_size_gb,
                    available_memory_gb,
                    memory_bandwidth_gbps: None,
                    relative_distance,
                    is_available,
                },
            );
        }

        Ok(NumaTopology {
            total_nodes: nodes.len() as u32,
            nodes,
            total_cores,
            total_memory_gb,
            node_distances,
        })
    }

    /// Allocate memory with NUMA awareness
    pub fn allocate_numa_aware(
        &self,
        size: usize,
        alignment: usize,
        policy_name: Option<&str>,
        access_pattern: AccessPattern,
    ) -> Result<NumaAllocation> {
        let policy = if let Some(name) = policy_name {
            let policies = self.policies.read().unwrap_or_else(|poisoned| poisoned.into_inner());
            policies.get(name).cloned().unwrap_or_default()
        } else {
            NumaPolicy::default()
        };

        let node_id = self.select_optimal_node(&policy, size, &access_pattern)?;

        // Real heap allocation; `bound_to_node` records that node binding was
        // not enforced (see the module docs).
        let (address, alignment) = self.allocate_on_node(node_id, size, alignment)?;

        let allocation_id = self.generate_allocation_id();
        let allocation = NumaAllocation {
            allocation_id: allocation_id.clone(),
            node_id,
            size_bytes: size,
            address,
            alignment,
            bound_to_node: false,
            allocation_time: std::time::SystemTime::now(),
            access_pattern,
        };

        // Track allocation
        {
            let mut allocations =
                self.allocations.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            allocations.insert(allocation_id, allocation.clone());
        }

        // Update performance statistics
        self.update_allocation_stats(node_id, size);

        Ok(allocation)
    }

    /// Select optimal NUMA node based on policy and workload characteristics
    fn select_optimal_node(
        &self,
        policy: &NumaPolicy,
        size: usize,
        access_pattern: &AccessPattern,
    ) -> Result<u32> {
        let topology = self.topology.read().unwrap_or_else(|poisoned| poisoned.into_inner());

        match &policy.strategy {
            NumaStrategy::LocalNode => self.get_current_node(),
            NumaStrategy::Interleaved => self.select_least_loaded_node(&topology),
            NumaStrategy::PreferredNodes(nodes) => {
                self.select_from_preferred_nodes(&topology, nodes, policy.strict)
            },
            NumaStrategy::WorkloadAware => {
                self.select_workload_aware_node(&topology, size, access_pattern)
            },
            NumaStrategy::Bind(nodes) => {
                if nodes.is_empty() {
                    Err(TrustformersError::other(
                        "No nodes specified for bind strategy".to_string(),
                    ))
                } else {
                    Ok(nodes[0]) // Use first node in bind list
                }
            },
        }
    }

    fn get_current_node(&self) -> Result<u32> {
        // In a real implementation, this would detect which NUMA node the current thread is running on
        // For now, we'll use a simple heuristic based on thread ID
        let thread_id = thread::current().id();
        let topology = self.topology.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        let node_count = topology.total_nodes;

        // Simple hash-based selection
        let hash = format!("{:?}", thread_id).len();
        Ok((hash as u32) % node_count)
    }

    fn select_least_loaded_node(&self, topology: &NumaTopology) -> Result<u32> {
        let monitor =
            self.performance_monitor.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        let least_loaded = topology
            .nodes
            .keys()
            .min_by_key(|&&node_id| {
                monitor
                    .allocation_stats
                    .get(&node_id)
                    .map(|stats| stats.current_memory_usage)
                    .unwrap_or(0)
            })
            .copied();

        least_loaded.ok_or_else(|| TrustformersError::other("No available NUMA nodes".to_string()))
    }

    fn select_from_preferred_nodes(
        &self,
        topology: &NumaTopology,
        preferred_nodes: &[u32],
        strict: bool,
    ) -> Result<u32> {
        for &node_id in preferred_nodes {
            if topology.nodes.contains_key(&node_id) {
                let node = &topology.nodes[&node_id];
                if node.is_available && node.available_memory_gb > 0.1 {
                    return Ok(node_id);
                }
            }
        }

        if strict {
            Err(TrustformersError::other(
                "No preferred NUMA nodes available".to_string(),
            ))
        } else {
            self.select_least_loaded_node(topology)
        }
    }

    fn select_workload_aware_node(
        &self,
        topology: &NumaTopology,
        size: usize,
        access_pattern: &AccessPattern,
    ) -> Result<u32> {
        // Nodes that cannot hold the request are not candidates at all.
        let required_gb = size as f64 / 1024.0 / 1024.0 / 1024.0;
        let mut scores = HashMap::new();
        let monitor =
            self.performance_monitor.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        for (&node_id, node) in &topology.nodes {
            if !node.is_available || node.available_memory_gb < required_gb {
                continue;
            }

            let mut score = 0.0;

            // Memory availability score
            let memory_score = node.available_memory_gb / node.memory_size_gb;
            score += memory_score * 0.3;

            // Bandwidth utilization score (prefer less utilized nodes).
            // Only contributes when the platform actually reports a bandwidth
            // figure for the node; otherwise the term is skipped rather than
            // scored against an invented denominator.
            if let Some(bandwidth_gbps) = node.memory_bandwidth_gbps.filter(|value| *value > 0.0) {
                let bandwidth_util =
                    monitor.memory_bandwidth_usage.get(&node_id).copied().unwrap_or(0.0);
                let bandwidth_score = 1.0 - (bandwidth_util / bandwidth_gbps);
                score += bandwidth_score * 0.2;
            }

            // Access pattern compatibility score
            let pattern_score = match access_pattern {
                AccessPattern::Sequential => {
                    // Prefer nodes with lower cross-node traffic
                    let cross_traffic: u64 = monitor
                        .cross_node_traffic
                        .iter()
                        .filter(|((from, _to), _)| *from == node_id)
                        .map(|(_, traffic)| *traffic)
                        .sum();
                    1.0 / (1.0 + cross_traffic as f64 / 1000000.0) // Normalize
                },
                AccessPattern::Random => {
                    // Prefer nodes with better cache performance
                    let cache_miss_rate =
                        monitor.cache_miss_rates.get(&node_id).copied().unwrap_or(0.1);
                    1.0 - cache_miss_rate
                },
                _ => 0.5, // Neutral score for other patterns
            };
            score += pattern_score * 0.3;

            // Current load score
            let current_load = monitor
                .allocation_stats
                .get(&node_id)
                .map(|stats| {
                    stats.current_memory_usage as f64
                        / (node.memory_size_gb * 1024.0 * 1024.0 * 1024.0)
                })
                .unwrap_or(0.0);
            let load_score = 1.0 - current_load;
            score += load_score * 0.2;

            scores.insert(node_id, score);
        }

        scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(node_id, _)| node_id)
            .ok_or_else(|| TrustformersError::other("No suitable NUMA node found".to_string()))
    }

    /// Allocate a real block of memory for `node_id`.
    ///
    /// Returns `(address, alignment)`. The block is a genuine heap allocation:
    /// the returned address is dereferenceable for `size` bytes until
    /// [`Self::deallocate`] frees it. Node *binding* is not performed — see
    /// [`NumaAllocation::bound_to_node`].
    fn allocate_on_node(
        &self,
        node_id: u32,
        size: usize,
        alignment: usize,
    ) -> Result<(usize, usize)> {
        {
            let topology = self.topology.read().unwrap_or_else(|poisoned| poisoned.into_inner());
            if !topology.nodes.contains_key(&node_id) {
                return Err(TrustformersError::other(format!(
                    "Invalid NUMA node: {}",
                    node_id
                )));
            }
        }

        if size == 0 {
            return Err(TrustformersError::invalid_input(
                "NUMA allocation size must be greater than zero".to_string(),
            ));
        }

        // `Layout` requires a power-of-two alignment of at least 1.
        let alignment = alignment.max(1).next_power_of_two();
        let layout = Layout::from_size_align(size, alignment).map_err(|error| {
            TrustformersError::invalid_input(format!(
                "invalid NUMA allocation layout (size {}, alignment {}): {}",
                size, alignment, error
            ))
        })?;

        // SAFETY: `layout` has a non-zero size and a valid power-of-two
        // alignment. The pointer is stored in `self.allocations` together with
        // its layout and is freed exactly once in `deallocate`.
        let pointer = unsafe { std::alloc::alloc(layout) };
        if pointer.is_null() {
            let mut monitor =
                self.performance_monitor.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            monitor.allocation_stats.entry(node_id).or_default().allocation_failures += 1;
            return Err(TrustformersError::resource_exhausted(format!(
                "failed to allocate {} bytes for NUMA node {}",
                size, node_id
            )));
        }

        Ok((pointer as usize, alignment))
    }

    fn generate_allocation_id(&self) -> String {
        let mut counter =
            self.allocation_counter.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        *counter += 1;
        format!("numa_alloc_{}", *counter)
    }

    fn update_allocation_stats(&self, node_id: u32, size: usize) {
        let mut monitor =
            self.performance_monitor.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let stats = monitor.allocation_stats.entry(node_id).or_default();

        stats.total_allocations += 1;
        stats.total_bytes += size as u64;
        stats.current_memory_usage += size as u64;
        stats.average_allocation_size = stats.total_bytes as f64 / stats.total_allocations as f64;

        if stats.current_memory_usage > stats.peak_memory_usage {
            stats.peak_memory_usage = stats.current_memory_usage;
        }
    }

    /// Pin the calling thread to the CPUs of the given NUMA nodes.
    ///
    /// Setting CPU affinity requires `sched_setaffinity` (Linux) or
    /// `SetThreadAffinityMask` (Windows). `trustformers-core` links no libc
    /// bindings, so this reports [`TrustformersError::not_implemented`] rather
    /// than logging a message and returning `Ok(())` while the thread stays
    /// unpinned.
    pub fn set_thread_affinity(&self, affinity: ThreadAffinity) -> Result<()> {
        Err(TrustformersError::not_implemented(format!(
            "thread affinity (requested nodes {:?}, cores {:?}) needs sched_setaffinity / \
             SetThreadAffinityMask bindings, which are not linked into trustformers-core",
            affinity.preferred_nodes, affinity.cpu_cores
        )))
    }

    /// CPU cores that belong to the given NUMA nodes, per the detected topology.
    ///
    /// This is the information a caller needs to pin threads itself; it does
    /// not change any thread's affinity.
    pub fn cores_for_nodes(&self, node_ids: &[u32]) -> Vec<u32> {
        let topology = self.topology.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut cores: Vec<u32> = node_ids
            .iter()
            .filter_map(|node_id| topology.nodes.get(node_id))
            .flat_map(|node| node.cpu_cores.iter().copied())
            .collect();
        cores.sort_unstable();
        cores.dedup();
        cores
    }

    /// Free NUMA-aware allocated memory
    pub fn deallocate(&self, allocation_id: &str) -> Result<()> {
        let allocation = {
            let mut allocations =
                self.allocations.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            allocations.remove(allocation_id).ok_or_else(|| {
                TrustformersError::other(format!("Allocation not found: {}", allocation_id))
            })?
        };

        // Update statistics
        {
            let mut monitor =
                self.performance_monitor.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(stats) = monitor.allocation_stats.get_mut(&allocation.node_id) {
                stats.current_memory_usage =
                    stats.current_memory_usage.saturating_sub(allocation.size_bytes as u64);
            }
        }

        // Really free the block that `allocate_on_node` allocated.
        let layout = Layout::from_size_align(allocation.size_bytes, allocation.alignment).map_err(
            |error| {
                TrustformersError::invalid_state(format!(
                    "tracked NUMA allocation {} has an invalid layout: {}",
                    allocation_id, error
                ))
            },
        )?;

        // SAFETY: `allocation.address` came from `std::alloc::alloc` with
        // exactly this layout, was removed from the tracking map above (so it
        // cannot be freed twice), and no `NumaAllocation` clone can free it.
        unsafe {
            std::alloc::dealloc(allocation.address as *mut u8, layout);
        }

        tracing::debug!(
            "Deallocated {} bytes from NUMA node {} (allocation: {})",
            allocation.size_bytes,
            allocation.node_id,
            allocation_id
        );

        Ok(())
    }

    /// Register a custom NUMA policy
    pub fn register_policy(&self, name: String, policy: NumaPolicy) {
        let mut policies = self.policies.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        policies.insert(name, policy);
    }

    /// Get NUMA topology information
    pub fn get_topology(&self) -> NumaTopology {
        let topology = self.topology.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        (*topology).clone()
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> NumaPerformanceMonitor {
        let monitor =
            self.performance_monitor.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        (*monitor).clone()
    }

    /// Optimize memory layout for a specific access pattern
    pub fn optimize_memory_layout(
        &self,
        allocations: &[String],
        access_pattern: AccessPattern,
    ) -> Result<Vec<String>> {
        let mut optimized_allocations = Vec::new();
        let allocations_map =
            self.allocations.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        match access_pattern {
            AccessPattern::Sequential => {
                // For sequential access, try to place allocations on the same node
                if let Some(first_alloc) =
                    allocations.first().and_then(|id| allocations_map.get(id))
                {
                    let preferred_node = first_alloc.node_id;

                    for alloc_id in allocations {
                        if let Some(allocation) = allocations_map.get(alloc_id) {
                            if allocation.node_id != preferred_node {
                                // Suggest migration
                                let new_id = format!("{}_migrated", alloc_id);
                                optimized_allocations.push(new_id);
                            } else {
                                optimized_allocations.push(alloc_id.clone());
                            }
                        }
                    }
                }
            },
            AccessPattern::Interleaved => {
                // For interleaved access, spread allocations across nodes
                let topology =
                    self.topology.read().unwrap_or_else(|poisoned| poisoned.into_inner());
                let available_nodes: Vec<u32> = topology.nodes.keys().copied().collect();

                for (node_index, alloc_id) in allocations.iter().enumerate() {
                    let target_node = available_nodes[node_index % available_nodes.len()];

                    if let Some(allocation) = allocations_map.get(alloc_id) {
                        if allocation.node_id != target_node {
                            let new_id = format!("{}_migrated_to_node_{}", alloc_id, target_node);
                            optimized_allocations.push(new_id);
                        } else {
                            optimized_allocations.push(alloc_id.clone());
                        }
                    }
                }
            },
            _ => {
                // For other patterns, keep current layout
                optimized_allocations.extend_from_slice(allocations);
            },
        }

        Ok(optimized_allocations)
    }

    /// Monitor cross-NUMA traffic and suggest optimizations
    pub fn analyze_numa_traffic(&self) -> NumaTrafficAnalysis {
        let monitor =
            self.performance_monitor.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        let mut analysis = NumaTrafficAnalysis {
            total_cross_node_traffic: 0,
            hotspots: Vec::new(),
            optimization_suggestions: Vec::new(),
        };

        // Calculate total cross-node traffic
        for ((from, to), traffic) in &monitor.cross_node_traffic {
            if from != to {
                analysis.total_cross_node_traffic += traffic;
            }
        }

        // Identify traffic hotspots
        let mut traffic_by_node: HashMap<u32, u64> = HashMap::new();
        for ((from, _to), traffic) in &monitor.cross_node_traffic {
            *traffic_by_node.entry(*from).or_insert(0) += traffic;
        }

        let mut sorted_traffic: Vec<_> = traffic_by_node.into_iter().collect();
        sorted_traffic.sort_by_key(|item| std::cmp::Reverse(item.1));

        for (node_id, traffic) in sorted_traffic.into_iter().take(3) {
            analysis.hotspots.push(TrafficHotspot {
                node_id,
                traffic_volume: traffic,
                severity: if traffic > 1000000 {
                    HotspotSeverity::High
                } else if traffic > 100000 {
                    HotspotSeverity::Medium
                } else {
                    HotspotSeverity::Low
                },
            });
        }

        // Generate optimization suggestions
        if analysis.total_cross_node_traffic > 10000000 {
            analysis.optimization_suggestions.push(
                "Consider using NUMA-local allocations to reduce cross-node traffic".to_string(),
            );
        }

        for hotspot in &analysis.hotspots {
            if hotspot.severity == HotspotSeverity::High {
                analysis.optimization_suggestions.push(format!(
                    "Node {} is experiencing high traffic - consider redistributing workload",
                    hotspot.node_id
                ));
            }
        }

        analysis
    }
}

#[derive(Debug, Clone)]
pub struct NumaTrafficAnalysis {
    pub total_cross_node_traffic: u64,
    pub hotspots: Vec<TrafficHotspot>,
    pub optimization_suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TrafficHotspot {
    pub node_id: u32,
    pub traffic_volume: u64,
    pub severity: HotspotSeverity,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HotspotSeverity {
    Low,
    Medium,
    High,
}

/// Parse a Linux CPU/node list such as `0-3,8,10-11` into individual indices.
///
/// Only reachable from the Linux topology reader; the unit tests exercise it on
/// every platform.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_cpu_list(list: &str) -> Vec<u32> {
    let mut indices = Vec::new();
    for part in list.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((start, end)) => {
                if let (Ok(start), Ok(end)) =
                    (start.trim().parse::<u32>(), end.trim().parse::<u32>())
                {
                    for index in start..=end {
                        indices.push(index);
                    }
                }
            },
            None => {
                if let Ok(index) = part.parse::<u32>() {
                    indices.push(index);
                }
            },
        }
    }
    indices
}

/// Parse `/sys/devices/system/node/nodeN/meminfo` into `(total_gb, free_gb)`.
///
/// Only reachable from the Linux topology reader; the unit tests exercise it on
/// every platform.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_node_meminfo(meminfo: &str) -> (f64, f64) {
    let mut total_gb = 0.0;
    let mut free_gb = 0.0;
    for line in meminfo.lines() {
        // Format: "Node 0 MemTotal:       16384000 kB"
        let fields: Vec<&str> = line.split_whitespace().collect();
        let Some(position) = fields.iter().position(|field| field.ends_with(':')) else {
            continue;
        };
        let Some(value) = fields.get(position + 1).and_then(|value| value.parse::<f64>().ok())
        else {
            continue;
        };
        let gib = value / 1024.0 / 1024.0;
        match fields[position] {
            "MemTotal:" => total_gb = gib,
            "MemFree:" => free_gb = gib,
            _ => {},
        }
    }
    (total_gb, free_gb)
}

/// Global NUMA allocator instance
static NUMA_ALLOCATOR: std::sync::OnceLock<Arc<NumaAllocator>> = std::sync::OnceLock::new();

/// Initialize global NUMA allocator
pub fn init_numa_allocator() -> Result<()> {
    let allocator = Arc::new(NumaAllocator::new()?);
    NUMA_ALLOCATOR
        .set(allocator)
        .map_err(|_| TrustformersError::other("NUMA allocator already initialized".to_string()))?;
    Ok(())
}

/// Get global NUMA allocator
pub fn get_numa_allocator() -> Result<Arc<NumaAllocator>> {
    NUMA_ALLOCATOR
        .get()
        .cloned()
        .ok_or_else(|| TrustformersError::other("NUMA allocator not initialized".to_string()))
}

/// Convenience function for NUMA-aware allocation
pub fn numa_alloc(
    size: usize,
    alignment: usize,
    policy: Option<&str>,
    pattern: AccessPattern,
) -> Result<NumaAllocation> {
    get_numa_allocator()?.allocate_numa_aware(size, alignment, policy, pattern)
}

/// Convenience function for NUMA deallocation
pub fn numa_free(allocation_id: &str) -> Result<()> {
    get_numa_allocator()?.deallocate(allocation_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: on platforms with no NUMA interface the allocator used
    /// to fabricate a multi-node topology (invented core ranges, `memory / n`
    /// per node, invented interconnect latencies). It must now say so.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_numa_is_unsupported_off_linux() {
        let error =
            NumaAllocator::detect_numa_topology().expect_err("no NUMA interface on this platform");
        let message = error.to_string();
        assert!(
            message.contains("NUMA topology detection"),
            "unexpected error: {message}"
        );
        assert!(
            NumaAllocator::new().is_err(),
            "constructing an allocator must fail when no topology can be read"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_numa_allocator_creation() {
        let allocator = NumaAllocator::new().expect("operation failed in test");
        let topology = allocator.get_topology();
        assert!(topology.total_nodes > 0);
        assert!(topology.total_cores > 0);
    }

    /// Regression test: `allocate_on_node` used to return the synthetic value
    /// `0x1000000 + node * 0x10000000 + size` and `deallocate` only logged.
    /// The address must now be a real, writable, freeable allocation.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_numa_allocation_returns_a_real_pointer() {
        let allocator = NumaAllocator::new().expect("operation failed in test");

        let allocation = allocator
            .allocate_numa_aware(1024, 64, None, AccessPattern::Sequential)
            .expect("operation failed in test");

        assert_eq!(allocation.size_bytes, 1024);
        assert!(!allocation.allocation_id.is_empty());
        assert_ne!(allocation.address, 0);
        assert_eq!(
            allocation.address % allocation.alignment,
            0,
            "the returned address must honour the requested alignment"
        );
        assert_ne!(
            allocation.address,
            0x1000000 + (allocation.node_id as usize * 0x10000000) + 1024,
            "the old synthetic address formula must not reappear"
        );
        assert!(
            !allocation.bound_to_node,
            "node binding is not performed and must not be claimed"
        );

        // SAFETY: the allocator owns `size_bytes` writable bytes at this
        // address until `deallocate` is called.
        unsafe {
            let pointer = allocation.address as *mut u8;
            std::ptr::write_bytes(pointer, 0xAB, allocation.size_bytes);
            assert_eq!(std::ptr::read(pointer.add(allocation.size_bytes - 1)), 0xAB);
        }

        allocator
            .deallocate(&allocation.allocation_id)
            .expect("operation failed in test");
        assert!(
            allocator.deallocate(&allocation.allocation_id).is_err(),
            "a freed allocation must not be freeable twice"
        );
    }

    /// Regression test: thread affinity used to log and return `Ok(())` without
    /// pinning anything.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_thread_affinity_is_reported_as_unimplemented() {
        let allocator = NumaAllocator::new().expect("operation failed in test");
        let result = allocator.set_thread_affinity(ThreadAffinity {
            thread_id: std::thread::current().id(),
            preferred_nodes: vec![0],
            cpu_cores: vec![0],
            priority: ThreadPriority::Normal,
        });
        assert!(result.is_err(), "affinity is not actually applied");
    }

    #[test]
    fn test_parse_cpu_list() {
        assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("0,2,4"), vec![0, 2, 4]);
        assert_eq!(parse_cpu_list("0-1,4,6-7"), vec![0, 1, 4, 6, 7]);
        assert!(parse_cpu_list("").is_empty());
    }

    #[test]
    fn test_parse_node_meminfo() {
        let meminfo = "Node 0 MemTotal:       16777216 kB\n\
                       Node 0 MemFree:         8388608 kB\n\
                       Node 0 MemUsed:         8388608 kB\n";
        let (total, free) = parse_node_meminfo(meminfo);
        assert!((total - 16.0).abs() < 1e-6, "got {total}");
        assert!((free - 8.0).abs() < 1e-6, "got {free}");
    }

    #[test]
    fn test_numa_policy() {
        let policy = NumaPolicy {
            strategy: NumaStrategy::PreferredNodes(vec![0, 1]),
            strict: true,
            ..Default::default()
        };

        assert_eq!(policy.strategy, NumaStrategy::PreferredNodes(vec![0, 1]));
        assert!(policy.strict);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_topology_detection() {
        let topology = NumaAllocator::detect_numa_topology().expect("operation failed in test");
        assert!(topology.total_nodes >= 1);
        assert!(!topology.nodes.is_empty());

        for (node_id, node) in &topology.nodes {
            assert_eq!(*node_id, node.node_id);
            assert!(node.memory_size_gb > 0.0);
            assert!(!node.cpu_cores.is_empty());
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_workload_aware_selection() {
        let allocator = NumaAllocator::new().expect("operation failed in test");
        let topology = allocator.get_topology();

        let node_id = allocator
            .select_workload_aware_node(&topology, 1024 * 1024, &AccessPattern::Sequential)
            .expect("operation failed in test");

        assert!(topology.nodes.contains_key(&node_id));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_performance_monitoring() {
        let allocator = NumaAllocator::new().expect("operation failed in test");

        // Make some allocations
        let _alloc1 = allocator
            .allocate_numa_aware(1024, 64, None, AccessPattern::Sequential)
            .expect("operation failed in test");

        let _alloc2 = allocator
            .allocate_numa_aware(2048, 64, None, AccessPattern::Random)
            .expect("operation failed in test");

        let stats = allocator.get_performance_stats();
        let total_allocations: u64 =
            stats.allocation_stats.values().map(|s| s.total_allocations).sum();

        assert!(total_allocations >= 2);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_memory_layout_optimization() {
        let allocator = NumaAllocator::new().expect("operation failed in test");

        let alloc1 = allocator
            .allocate_numa_aware(1024, 64, None, AccessPattern::Sequential)
            .expect("operation failed in test");

        let alloc2 = allocator
            .allocate_numa_aware(1024, 64, None, AccessPattern::Sequential)
            .expect("operation failed in test");

        let allocation_ids = vec![alloc1.allocation_id.clone(), alloc2.allocation_id.clone()];

        let optimized = allocator
            .optimize_memory_layout(&allocation_ids, AccessPattern::Sequential)
            .expect("operation failed in test");

        assert_eq!(optimized.len(), allocation_ids.len());
    }
}
