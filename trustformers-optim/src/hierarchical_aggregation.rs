// reason: research-stage module — reserved API/scaffolding fields and methods
// retained intentionally for in-progress features; not yet on active call paths.
#![allow(dead_code)]

pub mod collective;
pub mod transport;

use anyhow::Result;
use collective::{Collective, ReduceOp};
use std::collections::HashMap;
use std::sync::Arc;
use transport::Transport;
use trustformers_core::parallel::CommunicationBackend;
use trustformers_core::tensor::Tensor;

/// Hierarchical aggregation strategies for distributed training
///
/// This module provides advanced hierarchical aggregation algorithms that optimize
/// communication patterns for different network topologies and cluster configurations.
/// It supports tree-based, ring-based, and butterfly aggregation patterns.
///
/// # Communication
///
/// Aggregation is performed over a real [`transport::Transport`] (in-process
/// shared memory or TCP) through the algorithms in [`collective`]. A
/// [`HierarchicalAggregator`] built without a transport can only service a
/// world size of one; any larger configuration returns
/// [`AggregationError::NoCommunicator`] instead of aggregating against
/// fabricated data.

/// Errors raised by hierarchical aggregation.
#[derive(Debug, thiserror::Error)]
pub enum AggregationError {
    /// No transport was attached, so no peer data can be exchanged.
    #[error(
        "hierarchical aggregation over world size {world_size} requires a transport; \
         build the aggregator with HierarchicalAggregator::with_transport"
    )]
    NoCommunicator {
        /// Configured world size.
        world_size: usize,
    },

    /// The attached transport disagrees with the configuration.
    #[error(
        "transport world size {transport_world_size} does not match the configured world size \
         {config_world_size}"
    )]
    WorldSizeMismatch {
        /// World size reported by the transport.
        transport_world_size: usize,
        /// World size derived from the configuration.
        config_world_size: usize,
        /// Rank reported by the transport.
        transport_rank: usize,
    },

    /// Butterfly (recursive doubling) requires a power-of-two world size.
    #[error(
        "butterfly aggregation requires a power-of-two world size, got {world_size}; \
         use AggregationStrategy::Ring or AggregationStrategy::BinaryTree"
    )]
    ButterflyRequiresPowerOfTwo {
        /// Configured world size.
        world_size: usize,
    },
}

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
    node_topology: NodeTopology,
    communication_groups: CommunicationGroups,
    aggregation_stats: AggregationStats,
    fault_detector: Option<FaultDetector>,
    communicator: Option<Collective<Arc<dyn Transport>>>,
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
            communicator: None,
        })
    }

    /// Build an aggregator bound to a real transport.
    ///
    /// The transport's world size must match `num_nodes * devices_per_node`
    /// and its rank must match `global_rank`.
    pub fn with_transport(
        config: HierarchicalConfig,
        transport: Arc<dyn Transport>,
    ) -> Result<Self> {
        let mut aggregator = Self::new(config)?;
        aggregator.attach_transport(transport)?;
        Ok(aggregator)
    }

    /// Attach (or replace) the transport used for aggregation.
    pub fn attach_transport(&mut self, transport: Arc<dyn Transport>) -> Result<()> {
        let config_world_size = self.config.world_size();
        if transport.world_size() != config_world_size {
            return Err(AggregationError::WorldSizeMismatch {
                transport_world_size: transport.world_size(),
                config_world_size,
                transport_rank: transport.rank(),
            }
            .into());
        }
        self.communicator = Some(Collective::new(transport)?);
        Ok(())
    }

    /// Whether a real communicator is attached.
    pub fn has_communicator(&self) -> bool {
        self.communicator.is_some()
    }

    /// Borrow the communicator, or explain why aggregation cannot proceed.
    fn require_communicator(&self) -> Result<&Collective<Arc<dyn Transport>>> {
        self.communicator.as_ref().ok_or_else(|| {
            AggregationError::NoCommunicator {
                world_size: self.config.world_size(),
            }
            .into()
        })
    }

    /// Deterministic, rank-independent parameter ordering.
    fn ordered_names(gradients: &HashMap<String, Tensor>) -> Vec<String> {
        let mut names: Vec<String> = gradients.keys().cloned().collect();
        names.sort();
        names
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
        } else if num_nodes > 16 && world_size.is_power_of_two() {
            // Large clusters with small data: butterfly (recursive doubling) is
            // latency-optimal, but it is only defined for power-of-two sizes.
            Ok(AggregationStrategy::Butterfly)
        } else {
            // Default to tree for medium-sized clusters
            Ok(AggregationStrategy::BinaryTree)
        }
    }

    /// Tree-based all-reduce: a binomial-tree reduce to the root followed by a
    /// binomial-tree broadcast back out.
    ///
    /// Latency scales as `2 * log2(world_size)` messages, which is why it is
    /// preferred for small clusters and small payloads.
    fn tree_based_all_reduce(&mut self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        let communicator = self.require_communicator()?;
        // The binomial tree built by `build_communication_groups` is rooted at
        // global rank 0 (rank 0 is the only node with `parent == None`).
        const TREE_ROOT: usize = 0;
        let root = TREE_ROOT;

        for name in Self::ordered_names(gradients) {
            let Some(gradient) = gradients.get(&name) else {
                continue;
            };
            let shape = gradient.shape();
            let mut values = gradient.to_vec_f32()?;

            communicator.reduce(&mut values, root, ReduceOp::Sum)?;
            communicator.broadcast(&mut values, root)?;

            if let Some(slot) = gradients.get_mut(&name) {
                *slot = Tensor::from_slice(&values, &shape)?;
            }
        }

        Ok(())
    }

    /// Ring all-reduce: reduce-scatter around the ring followed by all-gather.
    ///
    /// Each rank transmits `2 * (world_size - 1) / world_size` of the payload,
    /// which is bandwidth-optimal and independent of the world size.
    fn ring_based_all_reduce(&mut self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        let communicator = self.require_communicator()?;

        for name in Self::ordered_names(gradients) {
            let Some(gradient) = gradients.get(&name) else {
                continue;
            };
            let shape = gradient.shape();
            let mut values = gradient.to_vec_f32()?;

            communicator.all_reduce(&mut values, ReduceOp::Sum)?;

            if let Some(slot) = gradients.get_mut(&name) {
                *slot = Tensor::from_slice(&values, &shape)?;
            }
        }

        Ok(())
    }

    /// Butterfly (recursive-doubling) all-reduce.
    ///
    /// At stage `s` every rank exchanges its full buffer with the partner whose
    /// rank differs in bit `s` and combines the two, so after `log2(n)` stages
    /// every rank holds the complete reduction. Requires a power-of-two world
    /// size; other sizes return
    /// [`AggregationError::ButterflyRequiresPowerOfTwo`].
    fn butterfly_based_all_reduce(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        let communicator = self.require_communicator()?;
        let world_size = communicator.world_size();

        if !world_size.is_power_of_two() {
            return Err(AggregationError::ButterflyRequiresPowerOfTwo { world_size }.into());
        }
        if world_size == 1 {
            return Ok(());
        }

        let rank = communicator.rank();
        let butterfly = self.communication_groups.butterfly_structure.clone();

        for name in Self::ordered_names(gradients) {
            let Some(gradient) = gradients.get(&name) else {
                continue;
            };
            let shape = gradient.shape();
            let mut values = gradient.to_vec_f32()?;

            for stage in 0..butterfly.num_stages {
                let partner = rank ^ (1usize << stage);
                if partner >= world_size {
                    continue;
                }
                let incoming = communicator.exchange(partner, &values)?;
                for (slot, value) in values.iter_mut().zip(incoming) {
                    *slot += value;
                }
            }

            if let Some(slot) = gradients.get_mut(&name) {
                *slot = Tensor::from_slice(&values, &shape)?;
            }
        }

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
    use super::transport::InProcessSession;
    use super::*;

    /// Run `body` on `world_size` ranks in parallel, each with its own
    /// aggregator bound to a shared in-process transport.
    fn spmd_aggregators<R, F>(
        num_nodes: usize,
        devices_per_node: usize,
        strategy: AggregationStrategy,
        body: F,
    ) -> Vec<R>
    where
        R: Send + 'static,
        F: Fn(usize, &mut HierarchicalAggregator) -> R + Send + Sync + 'static,
    {
        let world_size = num_nodes * devices_per_node;
        let session = InProcessSession::new(world_size).expect("session must be created in test");
        let body = std::sync::Arc::new(body);

        let handles: Vec<_> = (0..world_size)
            .map(|global_rank| {
                let transport: Arc<dyn Transport> = Arc::new(
                    session.transport(global_rank).expect("rank must be claimable in test"),
                );
                let mut config = HierarchicalConfig::new(
                    num_nodes,
                    devices_per_node,
                    global_rank / devices_per_node,
                    global_rank % devices_per_node,
                );
                config.strategy = strategy;
                let body = std::sync::Arc::clone(&body);
                std::thread::spawn(move || {
                    let mut aggregator = HierarchicalAggregator::with_transport(config, transport)
                        .expect("aggregator must build in test");
                    body(global_rank, &mut aggregator)
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|handle| handle.join().expect("rank thread must not panic in test"))
            .collect()
    }

    fn gradient_map(rank: usize) -> HashMap<String, Tensor> {
        let mut gradients = HashMap::new();
        gradients.insert(
            "layer.0.weight".to_string(),
            Tensor::from_slice(&[rank as f32, rank as f32 + 1.0, rank as f32 + 2.0], &[3])
                .expect("tensor must build in test"),
        );
        gradients.insert(
            "layer.0.bias".to_string(),
            Tensor::from_slice(&[rank as f32 * 0.5], &[1]).expect("tensor must build in test"),
        );
        gradients
    }

    /// Reference: elementwise sum over all ranks of `gradient_map`.
    fn expected_sums(world_size: usize) -> HashMap<String, Vec<f32>> {
        let mut expected = HashMap::new();
        expected.insert(
            "layer.0.weight".to_string(),
            (0..3)
                .map(|i| (0..world_size).map(|r| r as f32 + i as f32).sum::<f32>())
                .collect::<Vec<f32>>(),
        );
        expected.insert(
            "layer.0.bias".to_string(),
            vec![(0..world_size).map(|r| r as f32 * 0.5).sum::<f32>()],
        );
        expected
    }

    fn assert_matches_reference(
        results: &[HashMap<String, Tensor>],
        world_size: usize,
        label: &str,
    ) {
        let expected = expected_sums(world_size);
        for (rank, gradients) in results.iter().enumerate() {
            for (name, want) in &expected {
                let got = gradients
                    .get(name)
                    .unwrap_or_else(|| panic!("{label}: rank {rank} lost `{name}`"))
                    .to_vec_f32()
                    .expect("tensor read must succeed in test");
                assert_eq!(
                    got.len(),
                    want.len(),
                    "{label}: rank {rank} `{name}` length"
                );
                for (got_value, want_value) in got.iter().zip(want) {
                    approx::assert_relative_eq!(got_value, want_value, epsilon = 1e-4);
                }
            }
        }
    }

    #[test]
    fn ring_all_reduce_matches_elementwise_sum() {
        let world_size = 4;
        let results = spmd_aggregators(2, 2, AggregationStrategy::Ring, |rank, aggregator| {
            let mut gradients = gradient_map(rank);
            aggregator
                .hierarchical_all_reduce(&mut gradients)
                .expect("all-reduce must succeed in test");
            gradients
        });
        assert_matches_reference(&results, world_size, "ring");
    }

    #[test]
    fn tree_all_reduce_matches_elementwise_sum() {
        let world_size = 4;
        let results =
            spmd_aggregators(1, 4, AggregationStrategy::BinaryTree, |rank, aggregator| {
                let mut gradients = gradient_map(rank);
                aggregator
                    .hierarchical_all_reduce(&mut gradients)
                    .expect("all-reduce must succeed in test");
                gradients
            });
        assert_matches_reference(&results, world_size, "tree");
    }

    #[test]
    fn butterfly_all_reduce_matches_elementwise_sum() {
        let world_size = 4;
        let results = spmd_aggregators(2, 2, AggregationStrategy::Butterfly, |rank, aggregator| {
            let mut gradients = gradient_map(rank);
            aggregator
                .hierarchical_all_reduce(&mut gradients)
                .expect("all-reduce must succeed in test");
            gradients
        });
        assert_matches_reference(&results, world_size, "butterfly");
    }

    #[test]
    fn all_reduce_result_depends_on_peer_data() {
        // The previous implementation replaced every non-root gradient with a
        // shape-[1] zero tensor, so this asserts both the shape and the fact
        // that peers actually contribute.
        let results = spmd_aggregators(1, 2, AggregationStrategy::Ring, |rank, aggregator| {
            let mut gradients = HashMap::new();
            gradients.insert(
                "w".to_string(),
                Tensor::from_slice(&[if rank == 0 { 1.0 } else { 10.0 }; 4], &[4])
                    .expect("tensor must build in test"),
            );
            aggregator
                .hierarchical_all_reduce(&mut gradients)
                .expect("all-reduce must succeed in test");
            gradients
                .get("w")
                .expect("gradient must survive")
                .to_vec_f32()
                .expect("tensor read must succeed in test")
        });

        for values in &results {
            assert_eq!(values.len(), 4, "shape must be preserved");
            assert_eq!(values, &vec![11.0f32; 4]);
        }
    }

    #[test]
    fn aggregation_without_transport_errors_instead_of_faking() {
        let config = HierarchicalConfig::new(2, 2, 0, 0);
        let mut aggregator =
            HierarchicalAggregator::new(config).expect("aggregator must build in test");
        assert!(!aggregator.has_communicator());

        let mut gradients = gradient_map(0);
        let err = aggregator
            .hierarchical_all_reduce(&mut gradients)
            .expect_err("aggregation without a transport must fail");
        assert!(matches!(
            err.downcast_ref::<AggregationError>(),
            Some(AggregationError::NoCommunicator { .. })
        ));
    }

    #[test]
    fn transport_world_size_must_match_configuration() {
        let session = InProcessSession::new(3).expect("session must be created in test");
        let transport: Arc<dyn Transport> =
            Arc::new(session.transport(0).expect("rank must be claimable in test"));
        let config = HierarchicalConfig::new(2, 2, 0, 0); // world size 4, not 3
        let err = match HierarchicalAggregator::with_transport(config, transport) {
            Ok(_) => panic!("mismatched world size must be rejected"),
            Err(err) => err,
        };
        assert!(matches!(
            err.downcast_ref::<AggregationError>(),
            Some(AggregationError::WorldSizeMismatch { .. })
        ));
    }

    #[test]
    fn butterfly_rejects_non_power_of_two_world_size() {
        let results = spmd_aggregators(1, 3, AggregationStrategy::Butterfly, |rank, aggregator| {
            let mut gradients = gradient_map(rank);
            aggregator.hierarchical_all_reduce(&mut gradients).is_err()
        });
        assert!(results.iter().all(|failed| *failed));
    }

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
        let topology = HierarchicalAggregator::detect_network_topology(&config)
            .expect("Operation failed in test");
        let tree = HierarchicalAggregator::build_tree_structure(&config, &topology)
            .expect("Operation failed in test");

        assert_eq!(tree.parent, None); // Root node
        assert_eq!(tree.children, vec![1, 2]);
        assert_eq!(tree.depth, 0);
    }

    #[test]
    fn test_ring_structure_building() {
        let config = HierarchicalConfig::new(2, 4, 0, 1);
        let ring = HierarchicalAggregator::build_ring_structure(&config)
            .expect("Operation failed in test");

        assert_eq!(ring.next_rank, 2);
        assert_eq!(ring.prev_rank, 0);
        assert_eq!(ring.ring_size, 8);
    }

    #[test]
    fn test_adaptive_strategy_selection() {
        let config = HierarchicalConfig::new(4, 4, 0, 0);
        let aggregator = HierarchicalAggregator::new(config).expect("Construction failed");

        let mut gradients = HashMap::new();
        // Create a large tensor that exceeds 100MB threshold: 8000x8000x4bytes = 256MB
        gradients.insert(
            "param1".to_string(),
            Tensor::zeros(&[8000, 8000]).expect("Failed to create tensor"),
        );

        let strategy = aggregator
            .adaptive_strategy_selection(&gradients)
            .expect("Operation failed in test");
        // Should select ring for large data
        assert!(matches!(strategy, AggregationStrategy::Ring));
    }

    #[test]
    fn test_aggregation_stats_update() {
        let config = HierarchicalConfig::new(2, 2, 0, 0);
        let mut aggregator = HierarchicalAggregator::new(config).expect("Construction failed");

        let mut gradients = HashMap::new();
        gradients.insert(
            "param1".to_string(),
            Tensor::zeros(&[10, 10]).expect("Failed to create tensor"),
        );

        aggregator
            .update_aggregation_stats(AggregationStrategy::BinaryTree, 100.0, &gradients)
            .expect("Operation failed in test");

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
        let small_aggregator =
            HierarchicalAggregator::new(small_config).expect("Construction failed");
        assert!(matches!(
            small_aggregator.get_recommended_strategy(),
            AggregationStrategy::BinaryTree
        ));

        let large_config = HierarchicalConfig::new(20, 1, 0, 0);
        let large_aggregator =
            HierarchicalAggregator::new(large_config).expect("Construction failed");
        assert!(matches!(
            large_aggregator.get_recommended_strategy(),
            AggregationStrategy::Butterfly
        ));
    }

    #[test]
    fn test_butterfly_structure() {
        let config = HierarchicalConfig::new(1, 8, 0, 0);
        let butterfly = HierarchicalAggregator::build_butterfly_structure(&config)
            .expect("Operation failed in test");

        assert_eq!(butterfly.num_stages, 3); // log2(8) = 3
        assert_eq!(butterfly.connections.len(), 3);
    }

    #[test]
    fn test_network_topology_detection() {
        let config = HierarchicalConfig::new(3, 2, 0, 0);
        let topology = HierarchicalAggregator::detect_network_topology(&config)
            .expect("Operation failed in test");

        assert_eq!(topology.node_adjacency.len(), 3);
        assert_eq!(topology.node_bandwidth.len(), 3);
        assert_eq!(topology.node_latency.len(), 3);
        assert!(topology.intra_node_bandwidth > 0.0);
        assert!(topology.intra_node_latency > 0.0);
    }
}
