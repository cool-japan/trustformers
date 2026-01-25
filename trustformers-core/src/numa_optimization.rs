#![allow(unused_variables)] // NUMA optimization with platform-specific code

use crate::errors::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

/// NUMA (Non-Uniform Memory Access) optimization system for TrustformeRS
/// Provides intelligent memory allocation and thread scheduling for optimal performance on multi-socket systems
///
/// NUMA node information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumaNode {
    pub node_id: u32,
    pub cpu_cores: Vec<u32>,
    pub memory_size_gb: f64,
    pub available_memory_gb: f64,
    pub memory_bandwidth_gbps: f64,
    pub interconnect_latency_ns: HashMap<u32, u32>, // node_id -> latency
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

/// NUMA allocation tracking
#[derive(Debug, Clone)]
pub struct NumaAllocation {
    pub allocation_id: String,
    pub node_id: u32,
    pub size_bytes: usize,
    pub address: usize,
    pub allocation_time: std::time::SystemTime,
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

    /// Detect NUMA topology on the current system
    fn detect_numa_topology() -> Result<NumaTopology> {
        // In a real implementation, this would use system calls to detect actual NUMA topology
        // For now, we'll create a mock topology for demonstration

        let num_nodes = Self::get_numa_node_count()?;
        let mut nodes = HashMap::new();
        let mut node_distances = HashMap::new();

        let cores_per_node = num_cpus::get() / num_nodes as usize;
        let memory_per_node = Self::get_total_memory()? / num_nodes as f64;

        for node_id in 0..num_nodes {
            let cpu_cores: Vec<u32> = ((node_id * cores_per_node as u32)
                ..((node_id + 1) * cores_per_node as u32))
                .collect();

            let mut interconnect_latency = HashMap::new();
            for other_node in 0..num_nodes {
                let latency = if node_id == other_node {
                    10 // Local access latency (ns)
                } else {
                    50 + (node_id.abs_diff(other_node) * 10) // Remote access latency
                };
                interconnect_latency.insert(other_node, latency);
            }

            let node = NumaNode {
                node_id,
                cpu_cores,
                memory_size_gb: memory_per_node,
                available_memory_gb: memory_per_node * 0.8, // 80% available
                memory_bandwidth_gbps: 100.0,               // GB/s
                interconnect_latency_ns: interconnect_latency,
                is_available: true,
            };

            nodes.insert(node_id, node);

            // Calculate node distances
            for other_node in 0..num_nodes {
                let distance = if node_id == other_node {
                    10 // Local distance
                } else {
                    20 + (node_id.abs_diff(other_node) * 10) // Remote distance
                };
                node_distances.insert((node_id, other_node), distance);
            }
        }

        Ok(NumaTopology {
            nodes,
            total_nodes: num_nodes,
            total_cores: num_cpus::get() as u32,
            total_memory_gb: Self::get_total_memory()?,
            node_distances,
        })
    }

    fn get_numa_node_count() -> Result<u32> {
        // Try to detect actual NUMA nodes, fallback to 1 if not available
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            match fs::read_dir("/sys/devices/system/node") {
                Ok(entries) => {
                    let count = entries
                        .filter_map(|entry| entry.ok())
                        .filter(|entry| entry.file_name().to_string_lossy().starts_with("node"))
                        .count() as u32;
                    Ok(if count > 0 { count } else { 1 })
                },
                Err(_) => Ok(1),
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            // For non-Linux systems, assume single NUMA node for now
            // In a full implementation, this would use platform-specific APIs
            Ok(std::cmp::max(1, (num_cpus::get() / 8) as u32))
        }
    }

    fn get_total_memory() -> Result<f64> {
        // Get total system memory in GB
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return Ok(kb as f64 / 1024.0 / 1024.0); // Convert KB to GB
                            }
                        }
                    }
                }
            }
        }

        // Fallback estimation
        Ok(8.0) // Default to 8GB
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
            let policies = self.policies.read().expect("lock should not be poisoned");
            policies.get(name).cloned().unwrap_or_default()
        } else {
            NumaPolicy::default()
        };

        let node_id = self.select_optimal_node(&policy, size, &access_pattern)?;

        // Simulate memory allocation (in a real implementation, this would use NUMA-specific allocation)
        let address = self.allocate_on_node(node_id, size, alignment)?;

        let allocation_id = self.generate_allocation_id();
        let allocation = NumaAllocation {
            allocation_id: allocation_id.clone(),
            node_id,
            size_bytes: size,
            address,
            allocation_time: std::time::SystemTime::now(),
            access_pattern,
        };

        // Track allocation
        {
            let mut allocations = self.allocations.lock().expect("lock should not be poisoned");
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
        let topology = self.topology.read().expect("lock should not be poisoned");

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
        let topology = self.topology.read().expect("lock should not be poisoned");
        let node_count = topology.total_nodes;

        // Simple hash-based selection
        let hash = format!("{:?}", thread_id).len();
        Ok((hash as u32) % node_count)
    }

    fn select_least_loaded_node(&self, topology: &NumaTopology) -> Result<u32> {
        let monitor = self.performance_monitor.lock().expect("lock should not be poisoned");

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
        let mut scores = HashMap::new();
        let monitor = self.performance_monitor.lock().expect("lock should not be poisoned");

        for (&node_id, node) in &topology.nodes {
            if !node.is_available {
                continue;
            }

            let mut score = 0.0;

            // Memory availability score
            let memory_score = node.available_memory_gb / node.memory_size_gb;
            score += memory_score * 0.3;

            // Bandwidth utilization score (prefer less utilized nodes)
            let bandwidth_util =
                monitor.memory_bandwidth_usage.get(&node_id).copied().unwrap_or(0.0);
            let bandwidth_score = 1.0 - (bandwidth_util / node.memory_bandwidth_gbps);
            score += bandwidth_score * 0.2;

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

    fn allocate_on_node(&self, node_id: u32, size: usize, _alignment: usize) -> Result<usize> {
        // In a real implementation, this would use NUMA-specific allocation APIs
        // For now, we'll simulate allocation

        let topology = self.topology.read().expect("lock should not be poisoned");
        if !topology.nodes.contains_key(&node_id) {
            return Err(TrustformersError::other(format!(
                "Invalid NUMA node: {}",
                node_id
            )));
        }

        // Simulate memory allocation by returning a mock address
        // In reality, this would call numa_alloc_onnode() or similar
        let mock_address = 0x1000000 + (node_id as usize * 0x10000000) + size;
        Ok(mock_address)
    }

    fn generate_allocation_id(&self) -> String {
        let mut counter = self.allocation_counter.lock().expect("lock should not be poisoned");
        *counter += 1;
        format!("numa_alloc_{}", *counter)
    }

    fn update_allocation_stats(&self, node_id: u32, size: usize) {
        let mut monitor = self.performance_monitor.lock().expect("lock should not be poisoned");
        let stats = monitor.allocation_stats.entry(node_id).or_default();

        stats.total_allocations += 1;
        stats.total_bytes += size as u64;
        stats.current_memory_usage += size as u64;
        stats.average_allocation_size = stats.total_bytes as f64 / stats.total_allocations as f64;

        if stats.current_memory_usage > stats.peak_memory_usage {
            stats.peak_memory_usage = stats.current_memory_usage;
        }
    }

    /// Set thread affinity to specific NUMA nodes
    pub fn set_thread_affinity(&self, affinity: ThreadAffinity) -> Result<()> {
        // In a real implementation, this would set CPU affinity using platform-specific APIs
        // For Linux: sched_setaffinity()
        // For Windows: SetThreadAffinityMask()

        tracing::info!(
            "Setting thread affinity for {:?} to nodes {:?}",
            affinity.thread_id,
            affinity.preferred_nodes
        );

        // Mock implementation - in reality would call OS-specific APIs
        self.bind_thread_to_nodes(&affinity.preferred_nodes)?;

        Ok(())
    }

    fn bind_thread_to_nodes(&self, node_ids: &[u32]) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            // On Linux, we would use libnuma or direct syscalls
            // This is a simplified mock implementation
            tracing::debug!("Binding thread to NUMA nodes: {:?}", node_ids);
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, we would use SetThreadAffinityMask
            tracing::debug!("Binding thread to NUMA nodes: {:?}", node_ids);
        }

        Ok(())
    }

    /// Free NUMA-aware allocated memory
    pub fn deallocate(&self, allocation_id: &str) -> Result<()> {
        let allocation = {
            let mut allocations = self.allocations.lock().expect("lock should not be poisoned");
            allocations.remove(allocation_id).ok_or_else(|| {
                TrustformersError::other(format!("Allocation not found: {}", allocation_id))
            })?
        };

        // Update statistics
        {
            let mut monitor = self.performance_monitor.lock().expect("lock should not be poisoned");
            if let Some(stats) = monitor.allocation_stats.get_mut(&allocation.node_id) {
                stats.current_memory_usage =
                    stats.current_memory_usage.saturating_sub(allocation.size_bytes as u64);
            }
        }

        // In a real implementation, this would call numa_free() or similar
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
        let mut policies = self.policies.write().expect("lock should not be poisoned");
        policies.insert(name, policy);
    }

    /// Get NUMA topology information
    pub fn get_topology(&self) -> NumaTopology {
        let topology = self.topology.read().expect("lock should not be poisoned");
        (*topology).clone()
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> NumaPerformanceMonitor {
        let monitor = self.performance_monitor.lock().expect("lock should not be poisoned");
        (*monitor).clone()
    }

    /// Optimize memory layout for a specific access pattern
    pub fn optimize_memory_layout(
        &self,
        allocations: &[String],
        access_pattern: AccessPattern,
    ) -> Result<Vec<String>> {
        let mut optimized_allocations = Vec::new();
        let allocations_map = self.allocations.lock().expect("Lock poisoned");

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
                let topology = self.topology.read().expect("lock should not be poisoned");
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
        let monitor = self.performance_monitor.lock().expect("lock should not be poisoned");
        let topology = self.topology.read().expect("lock should not be poisoned");

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
        sorted_traffic.sort_by(|a, b| b.1.cmp(&a.1));

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

    #[test]
    fn test_numa_allocator_creation() {
        let allocator = NumaAllocator::new().unwrap();
        let topology = allocator.get_topology();
        assert!(topology.total_nodes > 0);
        assert!(topology.total_cores > 0);
    }

    #[test]
    fn test_numa_allocation() {
        let allocator = NumaAllocator::new().unwrap();

        let allocation = allocator
            .allocate_numa_aware(1024, 64, None, AccessPattern::Sequential)
            .unwrap();

        assert_eq!(allocation.size_bytes, 1024);
        assert!(!allocation.allocation_id.is_empty());

        allocator.deallocate(&allocation.allocation_id).unwrap();
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

    #[test]
    fn test_topology_detection() {
        let topology = NumaAllocator::detect_numa_topology().unwrap();
        assert!(topology.total_nodes >= 1);
        assert!(!topology.nodes.is_empty());

        for (node_id, node) in &topology.nodes {
            assert_eq!(*node_id, node.node_id);
            assert!(node.memory_size_gb > 0.0);
            assert!(!node.cpu_cores.is_empty());
        }
    }

    #[test]
    fn test_workload_aware_selection() {
        let allocator = NumaAllocator::new().unwrap();
        let topology = allocator.get_topology();

        let node_id = allocator
            .select_workload_aware_node(&topology, 1024 * 1024, &AccessPattern::Sequential)
            .unwrap();

        assert!(topology.nodes.contains_key(&node_id));
    }

    #[test]
    fn test_performance_monitoring() {
        let allocator = NumaAllocator::new().unwrap();

        // Make some allocations
        let _alloc1 = allocator
            .allocate_numa_aware(1024, 64, None, AccessPattern::Sequential)
            .unwrap();

        let _alloc2 = allocator.allocate_numa_aware(2048, 64, None, AccessPattern::Random).unwrap();

        let stats = allocator.get_performance_stats();
        let total_allocations: u64 =
            stats.allocation_stats.values().map(|s| s.total_allocations).sum();

        assert!(total_allocations >= 2);
    }

    #[test]
    fn test_memory_layout_optimization() {
        let allocator = NumaAllocator::new().unwrap();

        let alloc1 = allocator
            .allocate_numa_aware(1024, 64, None, AccessPattern::Sequential)
            .unwrap();

        let alloc2 = allocator
            .allocate_numa_aware(1024, 64, None, AccessPattern::Sequential)
            .unwrap();

        let allocation_ids = vec![alloc1.allocation_id.clone(), alloc2.allocation_id.clone()];

        let optimized = allocator
            .optimize_memory_layout(&allocation_ids, AccessPattern::Sequential)
            .unwrap();

        assert_eq!(optimized.len(), allocation_ids.len());
    }
}
