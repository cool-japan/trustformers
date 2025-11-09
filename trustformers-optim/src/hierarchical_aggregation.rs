use anyhow::Result;
use std::collections::HashMap;
use trustformers_core::parallel::CommunicationBackend;
use trustformers_core::tensor::Tensor;

/// Hierarchical aggregation strategies for distributed training
///
/// This module provides advanced hierarchical aggregation algorithms that optimize
/// communication patterns for different network topologies and cluster configurations.
/// It supports tree-based, ring-based, and butterfly aggregation patterns.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregationStrategy {
    /// Binary tree aggregation (optimal for small clusters)
    BinaryTree,
    /// Ring-based aggregation (bandwidth-optimal)
    Ring,
    /// Butterfly aggregation (latency-optimal)
    Butterfly,
    /// Adaptive strategy that selects best algorithm based on cluster topology
    Adaptive,
}

#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Number of nodes in the cluster
    pub num_nodes: usize,
    /// Number of devices per node
    pub devices_per_node: usize,
    /// Node rank (0-based)
    pub node_rank: usize,
    /// Local rank within node
    pub local_rank: usize,
    /// Global rank across all nodes
    pub global_rank: usize,
    /// Aggregation strategy
    pub strategy: AggregationStrategy,
    /// Communication backend
    pub comm_backend: CommunicationBackend,
    /// Enable compression during aggregation
    pub enable_compression: bool,
    /// Compression threshold (only compress if savings > threshold)
    pub compression_threshold: f32,
    /// Enable fault tolerance
    pub enable_fault_tolerance: bool,
    /// Timeout for communication operations (ms)
    pub comm_timeout_ms: u64,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            num_nodes: 1,
            devices_per_node: 1,
            node_rank: 0,
            local_rank: 0,
            global_rank: 0,
            strategy: AggregationStrategy::Adaptive,
            comm_backend: CommunicationBackend::Mpi,
            enable_compression: true,
            compression_threshold: 0.1,
            enable_fault_tolerance: true,
            comm_timeout_ms: 30000,
        }
    }
}

impl HierarchicalConfig {
    pub fn new(
        num_nodes: usize,
        devices_per_node: usize,
        node_rank: usize,
        local_rank: usize,
    ) -> Self {
        let global_rank = node_rank * devices_per_node + local_rank;
        Self {
            num_nodes,
            devices_per_node,
            node_rank,
            local_rank,
            global_rank,
            ..Default::default()
        }
    }

    pub fn world_size(&self) -> usize {
        self.num_nodes * self.devices_per_node
    }

    pub fn is_master(&self) -> bool {
        self.global_rank == 0
    }

    pub fn is_node_master(&self) -> bool {
        self.local_rank == 0
    }
}

/// Hierarchical aggregation coordinator
pub struct HierarchicalAggregator {
    config: HierarchicalConfig,
    #[allow(dead_code)]
    node_topology: NodeTopology,
    communication_groups: CommunicationGroups,
    aggregation_stats: AggregationStats,
    #[allow(dead_code)]
    fault_detector: Option<FaultDetector>,
}

/// Network topology representation
#[derive(Debug, Clone)]
pub struct NodeTopology {
    /// Adjacency matrix for inter-node connectivity
    pub node_adjacency: Vec<Vec<bool>>,
    /// Bandwidth matrix between nodes (MB/s)
    pub node_bandwidth: Vec<Vec<f32>>,
    /// Latency matrix between nodes (ms)
    pub node_latency: Vec<Vec<f32>>,
    /// Intra-node connectivity (assumed full connectivity)
    pub intra_node_bandwidth: f32,
    /// Intra-node latency
    pub intra_node_latency: f32,
}

/// Communication groups for hierarchical operations
#[derive(Debug, Clone)]
pub struct CommunicationGroups {
    /// Ranks within the same node
    pub node_local_group: Vec<usize>,
    /// Node master ranks for cross-node communication
    pub cross_node_group: Vec<usize>,
    /// Binary tree structure for tree-based aggregation
    pub tree_structure: TreeStructure,
    /// Ring structure for ring-based aggregation
    pub ring_structure: RingStructure,
    /// Butterfly structure for butterfly aggregation
    pub butterfly_structure: ButterflyStructure,
}

#[derive(Debug, Clone)]
pub struct TreeStructure {
    /// Parent rank in the tree (-1 if root)
    pub parent: Option<usize>,
    /// Children ranks in the tree
    pub children: Vec<usize>,
    /// Tree depth
    pub depth: usize,
    /// Tree height
    pub height: usize,
}

#[derive(Debug, Clone)]
pub struct RingStructure {
    /// Next rank in the ring
    pub next_rank: usize,
    /// Previous rank in the ring
    pub prev_rank: usize,
    /// Ring size
    pub ring_size: usize,
}

#[derive(Debug, Clone)]
pub struct ButterflyStructure {
    /// Butterfly connections for each stage
    pub connections: Vec<Vec<usize>>,
    /// Number of stages
    pub num_stages: usize,
}

/// Aggregation operation statistics
#[derive(Debug, Clone)]
pub struct AggregationStats {
    /// Total number of aggregation operations
    pub total_operations: usize,
    /// Average aggregation time (ms)
    pub avg_aggregation_time: f32,
    /// Total bytes transferred
    pub total_bytes_transferred: usize,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Number of failed operations
    pub failed_operations: usize,
    /// Strategy selection history
    pub strategy_history: HashMap<AggregationStrategy, usize>,
}

/// Fault detection and recovery
#[derive(Debug)]
pub struct FaultDetector {
    /// Failed nodes
    pub failed_nodes: Vec<usize>,
    /// Timeout threshold for detecting failures
    pub timeout_threshold: u64,
    /// Recovery strategy
    pub recovery_strategy: RecoveryStrategy,
}

#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Skip failed nodes and continue
    Skip,
    /// Retry with backup nodes
    Retry,
    /// Abort aggregation
    Abort,
}

impl Default for AggregationStats {
    fn default() -> Self {
        Self {
            total_operations: 0,
            avg_aggregation_time: 0.0,
            total_bytes_transferred: 0,
            compression_ratio: 1.0,
            failed_operations: 0,
            strategy_history: HashMap::new(),
        }
    }
}

impl HierarchicalAggregator {
    pub fn new(config: HierarchicalConfig) -> Result<Self> {
        let node_topology = Self::detect_network_topology(&config)?;
        let communication_groups = Self::build_communication_groups(&config, &node_topology)?;
        let aggregation_stats = AggregationStats::default();

        let fault_detector = if config.enable_fault_tolerance {
            Some(FaultDetector {
                failed_nodes: Vec::new(),
                timeout_threshold: config.comm_timeout_ms,
                recovery_strategy: RecoveryStrategy::Skip,
            })
        } else {
            None
        };

        Ok(Self {
            config,
            node_topology,
            communication_groups,
            aggregation_stats,
            fault_detector,
        })
    }

    /// Detect network topology and measure bandwidth/latency
    fn detect_network_topology(config: &HierarchicalConfig) -> Result<NodeTopology> {
        let num_nodes = config.num_nodes;

        // Initialize topology matrices
        let mut node_adjacency = vec![vec![false; num_nodes]; num_nodes];
        let mut node_bandwidth = vec![vec![0.0; num_nodes]; num_nodes];
        let mut node_latency = vec![vec![0.0; num_nodes]; num_nodes];

        // For this implementation, assume full connectivity with estimated values
        // In practice, these would be measured through benchmarking
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i != j {
                    node_adjacency[i][j] = true;
                    // Estimate bandwidth based on network topology
                    node_bandwidth[i][j] = if (i as i32 - j as i32).abs() == 1 {
                        10000.0 // Adjacent nodes: 10 GB/s
                    } else {
                        1000.0 // Non-adjacent nodes: 1 GB/s
                    };
                    // Estimate latency
                    node_latency[i][j] = if (i as i32 - j as i32).abs() == 1 {
                        0.1 // Adjacent nodes: 0.1ms
                    } else {
                        1.0 // Non-adjacent nodes: 1ms
                    };
                } else {
                    node_adjacency[i][j] = false;
                    node_bandwidth[i][j] = f32::INFINITY;
                    node_latency[i][j] = 0.0;
                }
            }
        }

        Ok(NodeTopology {
            node_adjacency,
            node_bandwidth,
            node_latency,
            intra_node_bandwidth: 80000.0, // 80 GB/s intra-node
            intra_node_latency: 0.01,      // 0.01ms intra-node
        })
    }

    /// Build communication groups for different aggregation strategies
    fn build_communication_groups(
        config: &HierarchicalConfig,
        topology: &NodeTopology,
    ) -> Result<CommunicationGroups> {
        // Node-local group
        let node_local_group: Vec<usize> = (0..config.devices_per_node)
            .map(|i| config.node_rank * config.devices_per_node + i)
            .collect();

        // Cross-node group (node masters)
        let cross_node_group: Vec<usize> =
            (0..config.num_nodes).map(|i| i * config.devices_per_node).collect();

        // Build tree structure
        let tree_structure = Self::build_tree_structure(config, topology)?;

        // Build ring structure
        let ring_structure = Self::build_ring_structure(config)?;

        // Build butterfly structure
        let butterfly_structure = Self::build_butterfly_structure(config)?;

        Ok(CommunicationGroups {
            node_local_group,
            cross_node_group,
            tree_structure,
            ring_structure,
            butterfly_structure,
        })
    }

    /// Build binary tree structure for tree-based aggregation
    fn build_tree_structure(
        config: &HierarchicalConfig,
        _topology: &NodeTopology,
    ) -> Result<TreeStructure> {
        let world_size = config.world_size();
        let rank = config.global_rank;

        // Build binary tree
        let parent = if rank == 0 { None } else { Some((rank - 1) / 2) };

        let mut children = Vec::new();
        let left_child = 2 * rank + 1;
        let right_child = 2 * rank + 2;

        if left_child < world_size {
            children.push(left_child);
        }
        if right_child < world_size {
            children.push(right_child);
        }

        // Calculate depth and height
        let depth = (rank as f32).log2().floor() as usize;
        let height = (world_size as f32).log2().ceil() as usize;

        Ok(TreeStructure {
            parent,
            children,
            depth,
            height,
        })
    }

    /// Build ring structure for ring-based aggregation
    fn build_ring_structure(config: &HierarchicalConfig) -> Result<RingStructure> {
        let world_size = config.world_size();
        let rank = config.global_rank;

        let next_rank = (rank + 1) % world_size;
        let prev_rank = (rank + world_size - 1) % world_size;

        Ok(RingStructure {
            next_rank,
            prev_rank,
            ring_size: world_size,
        })
    }

    /// Build butterfly structure for butterfly aggregation
    fn build_butterfly_structure(config: &HierarchicalConfig) -> Result<ButterflyStructure> {
        let world_size = config.world_size();
        let rank = config.global_rank;
        let num_stages = (world_size as f32).log2().ceil() as usize;

        let mut connections = Vec::new();

        for stage in 0..num_stages {
            let mut stage_connections = Vec::new();
            let distance = 1 << stage;

            // XOR-based butterfly connections
            let partner = rank ^ distance;
            if partner < world_size {
                stage_connections.push(partner);
            }

            connections.push(stage_connections);
        }

        Ok(ButterflyStructure {
            connections,
            num_stages,
        })
    }

    /// Perform hierarchical all-reduce operation
    pub fn hierarchical_all_reduce(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Select optimal strategy based on configuration and topology
        let strategy = self.select_optimal_strategy(gradients)?;

        // Perform aggregation based on selected strategy
        match strategy {
            AggregationStrategy::BinaryTree => {
                self.tree_based_all_reduce(gradients)?;
            },
            AggregationStrategy::Ring => {
                self.ring_based_all_reduce(gradients)?;
            },
            AggregationStrategy::Butterfly => {
                self.butterfly_based_all_reduce(gradients)?;
            },
            AggregationStrategy::Adaptive => {
                // Adaptive strategy selects the best algorithm dynamically
                let optimal_strategy = self.adaptive_strategy_selection(gradients)?;
                match optimal_strategy {
                    AggregationStrategy::BinaryTree => self.tree_based_all_reduce(gradients)?,
                    AggregationStrategy::Ring => self.ring_based_all_reduce(gradients)?,
                    AggregationStrategy::Butterfly => self.butterfly_based_all_reduce(gradients)?,
                    AggregationStrategy::Adaptive => {
                        return Err(anyhow::anyhow!(
                            "Invalid adaptive strategy selection: recursive Adaptive strategy returned"
                        ));
                    },
                }
            },
        }

        // Update statistics
        let elapsed = start_time.elapsed().as_millis() as f32;
        self.update_aggregation_stats(strategy, elapsed, gradients)?;

        Ok(())
    }

    /// Select optimal aggregation strategy
    fn select_optimal_strategy(
        &self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<AggregationStrategy> {
        match self.config.strategy {
            AggregationStrategy::Adaptive => self.adaptive_strategy_selection(gradients),
            strategy => Ok(strategy),
        }
    }

    /// Adaptive strategy selection based on cluster topology and data characteristics
    fn adaptive_strategy_selection(
        &self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<AggregationStrategy> {
        let world_size = self.config.world_size();
        let num_nodes = self.config.num_nodes;

        // Calculate total data size
        let total_data_size: usize = gradients.values().map(|tensor| tensor.memory_usage()).sum();

        // Strategy selection heuristics
        if world_size <= 8 {
            // Small clusters: tree is optimal
            Ok(AggregationStrategy::BinaryTree)
        } else if total_data_size > 100 * 1024 * 1024 {
            // Large data: ring is bandwidth-optimal
            Ok(AggregationStrategy::Ring)
        } else if num_nodes > 16 {
            // Large clusters with small data: butterfly is latency-optimal
            Ok(AggregationStrategy::Butterfly)
        } else {
            // Default to tree for medium-sized clusters
            Ok(AggregationStrategy::BinaryTree)
        }
    }

    /// Tree-based all-reduce (divide-and-conquer)
    fn tree_based_all_reduce(&mut self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        let tree = self.communication_groups.tree_structure.clone();

        // Phase 1: Reduce up the tree
        self.tree_reduce_up(gradients, &tree)?;

        // Phase 2: Broadcast down the tree
        self.tree_broadcast_down(gradients, &tree)?;

        Ok(())
    }

    /// Ring-based all-reduce (bandwidth-optimal)
    fn ring_based_all_reduce(&mut self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        let ring = self.communication_groups.ring_structure.clone();

        // Phase 1: Reduce-scatter
        self.ring_reduce_scatter(gradients, &ring)?;

        // Phase 2: All-gather
        self.ring_all_gather(gradients, &ring)?;

        Ok(())
    }

    /// Butterfly-based all-reduce (latency-optimal)
    fn butterfly_based_all_reduce(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        let butterfly = self.communication_groups.butterfly_structure.clone();

        // Butterfly all-reduce in multiple stages
        for stage in 0..butterfly.num_stages {
            self.butterfly_stage_operation(gradients, &butterfly, stage)?;
        }

        Ok(())
    }

    /// Tree reduce-up phase
    fn tree_reduce_up(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
        tree: &TreeStructure,
    ) -> Result<()> {
        // Collect gradients from children
        for &child_rank in &tree.children {
            for (name, gradient) in gradients.iter_mut() {
                // Simulate receiving gradient from child
                let child_gradient = self.simulate_receive_gradient(child_rank, name)?;
                *gradient = gradient.add(&child_gradient)?;
            }
        }

        // Send reduced gradients to parent
        if let Some(parent_rank) = tree.parent {
            for (name, gradient) in gradients.iter() {
                self.simulate_send_gradient(parent_rank, name, gradient)?;
            }
        }

        Ok(())
    }

    /// Tree broadcast-down phase
    fn tree_broadcast_down(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
        tree: &TreeStructure,
    ) -> Result<()> {
        // Receive final gradients from parent
        if let Some(parent_rank) = tree.parent {
            for (name, gradient) in gradients.iter_mut() {
                *gradient = self.simulate_receive_gradient(parent_rank, name)?;
            }
        }

        // Broadcast to children
        for &child_rank in &tree.children {
            for (name, gradient) in gradients.iter() {
                self.simulate_send_gradient(child_rank, name, gradient)?;
            }
        }

        Ok(())
    }

    /// Ring reduce-scatter phase
    fn ring_reduce_scatter(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
        ring: &RingStructure,
    ) -> Result<()> {
        let num_chunks = ring.ring_size;
        let rank = self.config.global_rank;

        for step in 0..num_chunks - 1 {
            let _send_chunk = (rank + ring.ring_size - step) % ring.ring_size;
            let _recv_chunk = (rank + ring.ring_size - step - 1) % ring.ring_size;

            // Send to next rank and receive from previous rank
            for (name, gradient) in gradients.iter_mut() {
                let chunk_gradient = self.simulate_receive_gradient(ring.prev_rank, name)?;
                *gradient = gradient.add(&chunk_gradient)?;
                self.simulate_send_gradient(ring.next_rank, name, gradient)?;
            }
        }

        Ok(())
    }

    /// Ring all-gather phase
    fn ring_all_gather(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
        ring: &RingStructure,
    ) -> Result<()> {
        let num_chunks = ring.ring_size;

        for _step in 0..num_chunks - 1 {
            // Send to next rank and receive from previous rank
            for (name, gradient) in gradients.iter_mut() {
                let chunk_gradient = self.simulate_receive_gradient(ring.prev_rank, name)?;
                *gradient = gradient.add(&chunk_gradient)?;
                self.simulate_send_gradient(ring.next_rank, name, gradient)?;
            }
        }

        Ok(())
    }

    /// Butterfly stage operation
    fn butterfly_stage_operation(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
        butterfly: &ButterflyStructure,
        stage: usize,
    ) -> Result<()> {
        if stage < butterfly.connections.len() {
            for &partner_rank in &butterfly.connections[stage] {
                for (name, gradient) in gradients.iter_mut() {
                    // Exchange gradients with partner
                    let partner_gradient = self.simulate_receive_gradient(partner_rank, name)?;
                    *gradient = gradient.add(&partner_gradient)?;
                    self.simulate_send_gradient(partner_rank, name, gradient)?;
                }
            }
        }

        Ok(())
    }

    /// Simulate receiving gradient from another rank
    fn simulate_receive_gradient(&self, _from_rank: usize, _name: &str) -> Result<Tensor> {
        // In a real implementation, this would use MPI or other communication backend
        // For this implementation, we'll create a dummy tensor
        Ok(Tensor::zeros(&[1])?)
    }

    /// Simulate sending gradient to another rank
    fn simulate_send_gradient(
        &self,
        _to_rank: usize,
        _name: &str,
        _gradient: &Tensor,
    ) -> Result<()> {
        // In a real implementation, this would use MPI or other communication backend
        Ok(())
    }

    /// Update aggregation statistics
    fn update_aggregation_stats(
        &mut self,
        strategy: AggregationStrategy,
        elapsed_ms: f32,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let stats = &mut self.aggregation_stats;

        stats.total_operations += 1;
        stats.avg_aggregation_time =
            (stats.avg_aggregation_time * (stats.total_operations - 1) as f32 + elapsed_ms)
                / stats.total_operations as f32;

        let bytes_transferred: usize = gradients.values().map(|tensor| tensor.memory_usage()).sum();
        stats.total_bytes_transferred += bytes_transferred;

        *stats.strategy_history.entry(strategy).or_insert(0) += 1;

        Ok(())
    }

    /// Get current aggregation statistics
    pub fn get_stats(&self) -> &AggregationStats {
        &self.aggregation_stats
    }

    /// Reset aggregation statistics
    pub fn reset_stats(&mut self) {
        self.aggregation_stats = AggregationStats::default();
    }

    /// Get recommended strategy for current configuration
    pub fn get_recommended_strategy(&self) -> AggregationStrategy {
        let world_size = self.config.world_size();
        let num_nodes = self.config.num_nodes;

        if world_size <= 8 {
            AggregationStrategy::BinaryTree
        } else if num_nodes > 16 {
            AggregationStrategy::Butterfly
        } else {
            AggregationStrategy::Ring
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_config() {
        let config = HierarchicalConfig::new(4, 8, 2, 3);
        assert_eq!(config.num_nodes, 4);
        assert_eq!(config.devices_per_node, 8);
        assert_eq!(config.node_rank, 2);
        assert_eq!(config.local_rank, 3);
        assert_eq!(config.global_rank, 19);
        assert_eq!(config.world_size(), 32);
        assert!(!config.is_master());
        assert!(!config.is_node_master());
    }

    #[test]
    fn test_tree_structure_building() {
        let config = HierarchicalConfig::new(2, 4, 0, 0);
        let topology = HierarchicalAggregator::detect_network_topology(&config).unwrap();
        let tree = HierarchicalAggregator::build_tree_structure(&config, &topology).unwrap();

        assert_eq!(tree.parent, None); // Root node
        assert_eq!(tree.children, vec![1, 2]);
        assert_eq!(tree.depth, 0);
    }

    #[test]
    fn test_ring_structure_building() {
        let config = HierarchicalConfig::new(2, 4, 0, 1);
        let ring = HierarchicalAggregator::build_ring_structure(&config).unwrap();

        assert_eq!(ring.next_rank, 2);
        assert_eq!(ring.prev_rank, 0);
        assert_eq!(ring.ring_size, 8);
    }

    #[test]
    fn test_adaptive_strategy_selection() {
        let config = HierarchicalConfig::new(4, 4, 0, 0);
        let aggregator = HierarchicalAggregator::new(config).unwrap();

        let mut gradients = HashMap::new();
        // Create a large tensor that exceeds 100MB threshold: 8000x8000x4bytes = 256MB
        gradients.insert("param1".to_string(), Tensor::zeros(&[8000, 8000]).unwrap());

        let strategy = aggregator.adaptive_strategy_selection(&gradients).unwrap();
        // Should select ring for large data
        assert!(matches!(strategy, AggregationStrategy::Ring));
    }

    #[test]
    fn test_aggregation_stats_update() {
        let config = HierarchicalConfig::new(2, 2, 0, 0);
        let mut aggregator = HierarchicalAggregator::new(config).unwrap();

        let mut gradients = HashMap::new();
        gradients.insert("param1".to_string(), Tensor::zeros(&[10, 10]).unwrap());

        aggregator
            .update_aggregation_stats(AggregationStrategy::BinaryTree, 100.0, &gradients)
            .unwrap();

        let stats = aggregator.get_stats();
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.avg_aggregation_time, 100.0);
        assert_eq!(
            stats.strategy_history.get(&AggregationStrategy::BinaryTree),
            Some(&1)
        );
    }

    #[test]
    fn test_recommended_strategy() {
        let small_config = HierarchicalConfig::new(2, 2, 0, 0);
        let small_aggregator = HierarchicalAggregator::new(small_config).unwrap();
        assert!(matches!(
            small_aggregator.get_recommended_strategy(),
            AggregationStrategy::BinaryTree
        ));

        let large_config = HierarchicalConfig::new(20, 1, 0, 0);
        let large_aggregator = HierarchicalAggregator::new(large_config).unwrap();
        assert!(matches!(
            large_aggregator.get_recommended_strategy(),
            AggregationStrategy::Butterfly
        ));
    }

    #[test]
    fn test_butterfly_structure() {
        let config = HierarchicalConfig::new(1, 8, 0, 0);
        let butterfly = HierarchicalAggregator::build_butterfly_structure(&config).unwrap();

        assert_eq!(butterfly.num_stages, 3); // log2(8) = 3
        assert_eq!(butterfly.connections.len(), 3);
    }

    #[test]
    fn test_network_topology_detection() {
        let config = HierarchicalConfig::new(3, 2, 0, 0);
        let topology = HierarchicalAggregator::detect_network_topology(&config).unwrap();

        assert_eq!(topology.node_adjacency.len(), 3);
        assert_eq!(topology.node_bandwidth.len(), 3);
        assert_eq!(topology.node_latency.len(), 3);
        assert!(topology.intra_node_bandwidth > 0.0);
        assert!(topology.intra_node_latency > 0.0);
    }
}
