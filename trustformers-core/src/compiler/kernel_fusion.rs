/*!
# Kernel Fusion Module

This module provides automatic kernel fusion capabilities for optimizing computation graphs by:

- **Operation Fusion**: Automatically identify and fuse compatible operations
- **Memory Bandwidth Optimization**: Reduce memory traffic through strategic fusion
- **Custom Fusion Patterns**: Support for domain-specific fusion patterns
- **Cost-Benefit Analysis**: Intelligent fusion decisions based on performance modeling
- **Multi-Level Fusion**: Support for basic, intermediate, and advanced fusion strategies
*/

#![allow(unused_variables)] // Compiler kernel fusion

use crate::compiler::{CompilerConfig, ComputationGraph, GraphNode};
use crate::errors::invalid_input;
use crate::errors::TrustformersError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// How many times the graph's average per-node memory cost a fusion group's
/// *own* nodes may reasonably cost, used by
/// `KernelFusion::check_memory_constraints`.
///
/// `GraphNode::memory_cost` has no fixed real-world unit — it is whatever
/// scale the graph's producer used — so comparing it against a fixed
/// absolute number (the old `1000.0`) would be a comparison against an
/// undefined scale. Comparing it against a multiple of the *same graph's*
/// average per-node cost, scaled by the group's own node count, keeps the
/// comparison meaningful regardless of that scale: a group of ordinarily-
/// priced nodes always passes, however small the surrounding graph is,
/// while a group whose combined cost is disproportionate to what that many
/// average-priced nodes would cost is flagged. (A flat fraction of the
/// graph's *total* was tried first and rejected: every fusion pattern in
/// `initialize_patterns` below is 2-6 nodes, so on any graph under roughly
/// a dozen nodes -- the common case, including every test graph in this
/// module -- a real pattern match already accounts for a large share of
/// the graph, and a "% of total" rule silently disabled fusion for it.)
/// This is still an unvalidated modelling assumption, not a measurement:
/// no in-tree profiling backs the exact multiple.
const FUSION_GROUP_MEMORY_HEADROOM: f64 = 4.0;

/// Assumed memory-cost retention when combining a fusion group's nodes into
/// one fused node, used by `KernelFusion::create_fused_node`.
///
/// An unvalidated modelling assumption, not a measurement: nothing in this
/// crate profiles fused kernels against their unfused originals, so there is
/// no real per-pattern memory-savings data to differentiate this by (unlike
/// `FusionPattern::expected_speedup` below, which *is* a real per-pattern
/// table). It intentionally stays a single, clearly-labelled constant rather
/// than a per-pattern table dressed up with invented numbers.
const ASSUMED_FUSION_MEMORY_RETENTION: f64 = 0.8; // i.e. an assumed 20% memory saving

/// Kernel fusion engine for automatic operation fusion
pub struct KernelFusion {
    config: CompilerConfig,
    fusion_patterns: Vec<FusionPattern>,
    fusion_cache: HashMap<String, FusionGroup>,
    fusion_stats: FusionStatistics,
}

impl KernelFusion {
    /// Create a new kernel fusion engine
    pub fn new(config: &CompilerConfig) -> Result<Self, TrustformersError> {
        let mut fusion = Self {
            config: config.clone(),
            fusion_patterns: Vec::new(),
            fusion_cache: HashMap::new(),
            fusion_stats: FusionStatistics::new(),
        };

        fusion.initialize_patterns()?;
        Ok(fusion)
    }

    /// Update the configuration
    pub fn update_config(&mut self, config: &CompilerConfig) -> Result<(), TrustformersError> {
        self.config = config.clone();
        self.initialize_patterns()?;
        Ok(())
    }

    /// Initialize fusion patterns based on configuration
    fn initialize_patterns(&mut self) -> Result<(), TrustformersError> {
        self.fusion_patterns.clear();

        // Basic element-wise fusion patterns
        self.fusion_patterns.push(FusionPattern::new(
            "ElementWiseChain",
            vec!["Add", "Mul", "Sub", "Div"],
            FusionType::ElementWise,
            1.5, // Expected speedup
        ));

        // Activation fusion patterns
        self.fusion_patterns.push(FusionPattern::new(
            "ActivationFusion",
            vec!["Linear", "ReLU"],
            FusionType::Activation,
            1.3,
        ));

        self.fusion_patterns.push(FusionPattern::new(
            "ActivationFusion",
            vec!["Linear", "GELU"],
            FusionType::Activation,
            1.3,
        ));

        self.fusion_patterns.push(FusionPattern::new(
            "ActivationFusion",
            vec!["MatMul", "Add", "ReLU"],
            FusionType::Activation,
            1.4,
        ));

        // Normalization fusion patterns
        self.fusion_patterns.push(FusionPattern::new(
            "LayerNormFusion",
            vec!["Sub", "Pow", "Mean", "Add", "Sqrt", "Div"],
            FusionType::Normalization,
            2.0,
        ));

        // Attention fusion patterns
        self.fusion_patterns.push(FusionPattern::new(
            "AttentionFusion",
            vec!["MatMul", "Scale", "Add", "Softmax", "MatMul"],
            FusionType::Attention,
            1.8,
        ));

        // Convolution fusion patterns
        self.fusion_patterns.push(FusionPattern::new(
            "ConvActivationFusion",
            vec!["Conv2D", "BatchNorm", "ReLU"],
            FusionType::Convolution,
            1.6,
        ));

        Ok(())
    }

    /// Apply kernel fusion to a computation graph
    pub fn apply_fusion(
        &mut self,
        graph: &mut ComputationGraph,
    ) -> Result<FusionResult, TrustformersError> {
        let start_time = std::time::Instant::now();

        // Find fusion opportunities
        let fusion_groups = self.find_fusion_opportunities(graph)?;

        let mut total_fused = 0;
        let mut total_benefit = 0.0;
        let mut applied_patterns = Vec::new();

        // Apply fusion for each group
        for group in fusion_groups {
            if self.should_apply_fusion(&group, graph)? {
                let benefit = self.apply_fusion_group(graph, &group)?;
                total_fused += group.nodes.len();
                total_benefit += benefit;
                applied_patterns.push(group.pattern.clone());
            }
        }

        let fusion_time = start_time.elapsed();

        // Update statistics
        self.fusion_stats.total_fusions += total_fused;
        self.fusion_stats.total_fusion_time += fusion_time;

        Ok(FusionResult {
            fused_operations: total_fused,
            estimated_speedup: total_benefit,
            fusion_time_ms: fusion_time.as_millis() as u64,
            applied_patterns,
        })
    }

    /// Find fusion opportunities in the graph
    fn find_fusion_opportunities(
        &self,
        graph: &ComputationGraph,
    ) -> Result<Vec<FusionGroup>, TrustformersError> {
        let mut fusion_groups = Vec::new();

        for pattern in &self.fusion_patterns {
            let groups = self.find_pattern_matches(graph, pattern)?;
            fusion_groups.extend(groups);
        }

        // Remove overlapping groups (prefer higher benefit)
        fusion_groups.sort_by(|a, b| {
            b.estimated_benefit
                .partial_cmp(&a.estimated_benefit)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut used_nodes = HashSet::new();
        let mut non_overlapping_groups = Vec::new();

        for group in fusion_groups {
            let has_overlap = group.nodes.iter().any(|&node| used_nodes.contains(&node));
            if !has_overlap {
                for &node in &group.nodes {
                    used_nodes.insert(node);
                }
                non_overlapping_groups.push(group);
            }
        }

        Ok(non_overlapping_groups)
    }

    /// Find matches for a specific fusion pattern
    fn find_pattern_matches(
        &self,
        graph: &ComputationGraph,
        pattern: &FusionPattern,
    ) -> Result<Vec<FusionGroup>, TrustformersError> {
        let mut matches = Vec::new();

        // Simple pattern matching - look for sequences of operations
        if pattern.operations.len() == 1 {
            // Single operation pattern
            for (i, node) in graph.nodes.iter().enumerate() {
                if node.op_type == pattern.operations[0] {
                    matches.push(FusionGroup {
                        nodes: vec![i],
                        pattern: pattern.name.clone(),
                        fusion_type: pattern.fusion_type,
                        estimated_benefit: pattern.expected_speedup,
                    });
                }
            }
        } else {
            // Multi-operation pattern
            matches.extend(self.find_sequential_matches(graph, pattern)?);
        }

        Ok(matches)
    }

    /// Find sequential matches for multi-operation patterns
    fn find_sequential_matches(
        &self,
        graph: &ComputationGraph,
        pattern: &FusionPattern,
    ) -> Result<Vec<FusionGroup>, TrustformersError> {
        let mut matches = Vec::new();

        // Build adjacency list for forward traversal
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for edge in &graph.edges {
            adj.entry(edge.from).or_default().push(edge.to);
        }

        // Try to match pattern starting from each node
        for (start_idx, start_node) in graph.nodes.iter().enumerate() {
            if start_node.op_type == pattern.operations[0] {
                if let Some(sequence) =
                    self.match_sequence_from_node(graph, &adj, start_idx, pattern)
                {
                    matches.push(FusionGroup {
                        nodes: sequence,
                        pattern: pattern.name.clone(),
                        fusion_type: pattern.fusion_type,
                        estimated_benefit: pattern.expected_speedup,
                    });
                }
            }
        }

        Ok(matches)
    }

    /// Find neighbors that match a target operation type
    fn find_matching_neighbors(
        &self,
        graph: &ComputationGraph,
        adj: &HashMap<usize, Vec<usize>>,
        current_nodes: &[usize],
        target_op: &str,
    ) -> Vec<usize> {
        let mut next_nodes = Vec::new();
        for &current in current_nodes {
            let Some(neighbors) = adj.get(&current) else {
                continue;
            };
            for &neighbor in neighbors {
                if neighbor < graph.nodes.len() && graph.nodes[neighbor].op_type == *target_op {
                    next_nodes.push(neighbor);
                }
            }
        }
        next_nodes
    }

    /// Match a sequence of operations starting from a given node
    fn match_sequence_from_node(
        &self,
        graph: &ComputationGraph,
        adj: &HashMap<usize, Vec<usize>>,
        start_node: usize,
        pattern: &FusionPattern,
    ) -> Option<Vec<usize>> {
        let mut sequence = vec![start_node];
        let mut current_nodes = vec![start_node];

        for op_idx in 1..pattern.operations.len() {
            let target_op = &pattern.operations[op_idx];
            let next_nodes = self.find_matching_neighbors(graph, adj, &current_nodes, target_op);

            if next_nodes.is_empty() {
                return None; // Pattern doesn't match
            }

            // For simplicity, take the first matching node
            // In practice, we might want to consider all possibilities
            sequence.push(next_nodes[0]);
            current_nodes = vec![next_nodes[0]];
        }

        Some(sequence)
    }

    /// Determine if a fusion should be applied
    fn should_apply_fusion(
        &self,
        group: &FusionGroup,
        graph: &ComputationGraph,
    ) -> Result<bool, TrustformersError> {
        // Check if fusion is beneficial
        if group.estimated_benefit < 1.1 {
            return Ok(false);
        }

        // Check hardware compatibility
        if !self.is_hardware_compatible(group)? {
            return Ok(false);
        }

        // Check memory constraints
        if !self.check_memory_constraints(group, graph)? {
            return Ok(false);
        }

        Ok(true)
    }

    /// Check if fusion is compatible with target hardware
    fn is_hardware_compatible(&self, group: &FusionGroup) -> Result<bool, TrustformersError> {
        match self.config.target_hardware.device_type {
            crate::compiler::DeviceType::GPU => {
                // GPU supports most fusion types
                Ok(true)
            },
            crate::compiler::DeviceType::CPU => {
                // CPU supports element-wise and some activation fusions
                match group.fusion_type {
                    FusionType::ElementWise | FusionType::Activation => Ok(true),
                    _ => Ok(false),
                }
            },
            _ => Ok(false),
        }
    }

    /// Check memory constraints for fusion
    fn check_memory_constraints(
        &self,
        group: &FusionGroup,
        graph: &ComputationGraph,
    ) -> Result<bool, TrustformersError> {
        let total_memory: f64 = group
            .nodes
            .iter()
            .filter_map(|&idx| graph.nodes.get(idx))
            .map(|node| node.memory_cost)
            .sum();

        // Relative to the graph's own average per-node memory cost, not a
        // fixed absolute number -- see `FUSION_GROUP_MEMORY_HEADROOM`.
        let graph_total_memory: f64 = graph.nodes.iter().map(|node| node.memory_cost).sum();
        if graph_total_memory <= 0.0 {
            // No cost data to judge against (e.g. a front-end that never
            // populated `memory_cost`): the constraint has no evidence to
            // reject on, so it abstains rather than failing closed.
            return Ok(true);
        }
        let average_node_memory = graph_total_memory / graph.nodes.len().max(1) as f64;
        let memory_threshold =
            average_node_memory * group.nodes.len() as f64 * FUSION_GROUP_MEMORY_HEADROOM;
        Ok(total_memory < memory_threshold)
    }

    /// Apply fusion for a specific group
    fn apply_fusion_group(
        &mut self,
        graph: &mut ComputationGraph,
        group: &FusionGroup,
    ) -> Result<f64, TrustformersError> {
        // Create a fused node
        let fused_node = self.create_fused_node(graph, group)?;

        // Replace the original nodes with the fused node
        self.replace_nodes_with_fused(graph, group, fused_node)?;

        Ok(group.estimated_benefit)
    }

    /// Create a fused node from a fusion group
    fn create_fused_node(
        &self,
        graph: &ComputationGraph,
        group: &FusionGroup,
    ) -> Result<GraphNode, TrustformersError> {
        if group.nodes.is_empty() {
            return Err(invalid_input("Empty fusion group"));
        }

        let first_node = &graph.nodes[group.nodes[0]];
        let last_node = &graph.nodes[*group.nodes.last().ok_or_else(|| {
            crate::errors::runtime_error(format!("{} in create_fused_node", "group is not empty"))
        })?];

        // Create fused operation name
        let op_types: Vec<String> =
            group.nodes.iter().map(|&idx| graph.nodes[idx].op_type.clone()).collect();
        let fused_op_type = format!("Fused[{}]", op_types.join("+"));

        // Combine attributes
        let mut attributes = HashMap::new();
        attributes.insert("fusion_pattern".to_string(), group.pattern.clone());
        attributes.insert(
            "fusion_type".to_string(),
            format!("{:?}", group.fusion_type),
        );
        attributes.insert("original_ops".to_string(), op_types.join(","));

        // Calculate combined costs
        let total_compute_cost: f64 =
            group.nodes.iter().map(|&idx| graph.nodes[idx].compute_cost).sum();
        let total_memory_cost: f64 =
            group.nodes.iter().map(|&idx| graph.nodes[idx].memory_cost).sum();

        // Apply fusion benefit
        let optimized_compute_cost = total_compute_cost / group.estimated_benefit;
        let optimized_memory_cost = total_memory_cost * ASSUMED_FUSION_MEMORY_RETENTION;

        Ok(GraphNode {
            id: first_node.id, // Will be updated when inserted
            op_type: fused_op_type,
            attributes,
            input_shapes: first_node.input_shapes.clone(),
            output_shapes: last_node.output_shapes.clone(),
            compute_cost: optimized_compute_cost,
            memory_cost: optimized_memory_cost,
        })
    }

    /// Replace nodes with a fused node in the graph
    fn replace_nodes_with_fused(
        &self,
        graph: &mut ComputationGraph,
        group: &FusionGroup,
        mut fused_node: GraphNode,
    ) -> Result<(), TrustformersError> {
        if group.nodes.is_empty() {
            return Ok(());
        }

        // Sort nodes in descending order for safe removal
        let mut sorted_nodes = group.nodes.clone();
        sorted_nodes.sort_by(|a, b| b.cmp(a));

        // Find edges that need to be updated
        let first_node = group.nodes[0];
        let last_node = *group.nodes.last().ok_or_else(|| {
            crate::errors::runtime_error(format!(
                "{} in replace_nodes_with_fused",
                "group is not empty"
            ))
        })?;

        // Update fused node ID
        fused_node.id = first_node;

        // Remove nodes from last to first to preserve indices
        for &node_idx in &sorted_nodes[1..] {
            // Keep the first node
            if node_idx >= graph.nodes.len() {
                continue;
            }

            graph.nodes.remove(node_idx);

            // Update node indices in edges
            for edge in &mut graph.edges {
                if edge.from > node_idx {
                    edge.from -= 1;
                }
                if edge.to > node_idx {
                    edge.to -= 1;
                }
            }
        }

        // Update the first node with fused node
        if first_node < graph.nodes.len() {
            graph.nodes[first_node] = fused_node;
        }

        // Remove internal edges within the fusion group
        let internal_nodes: HashSet<_> = group.nodes.iter().collect();
        graph.edges.retain(|edge| {
            !(internal_nodes.contains(&edge.from) && internal_nodes.contains(&edge.to))
        });

        Ok(())
    }

    /// Get fusion statistics
    pub fn get_stats(&self) -> &FusionStatistics {
        &self.fusion_stats
    }

    /// Reset fusion statistics
    pub fn reset_stats(&mut self) {
        self.fusion_stats = FusionStatistics::new();
    }

    /// Clear fusion cache
    pub fn clear_cache(&mut self) {
        self.fusion_cache.clear();
    }
}

/// Fusion pattern definition
#[derive(Debug, Clone)]
pub struct FusionPattern {
    pub name: String,
    pub operations: Vec<String>,
    pub fusion_type: FusionType,
    pub expected_speedup: f64,
}

impl FusionPattern {
    pub fn new(
        name: &str,
        operations: Vec<&str>,
        fusion_type: FusionType,
        expected_speedup: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            operations: operations.into_iter().map(|s| s.to_string()).collect(),
            fusion_type,
            expected_speedup,
        }
    }
}

/// Types of fusion patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionType {
    /// Element-wise operations that can be fused
    ElementWise,
    /// Matrix multiplication with activation
    Activation,
    /// Normalization operations
    Normalization,
    /// Attention mechanism fusion
    Attention,
    /// Convolution operations
    Convolution,
    /// Custom fusion pattern
    Custom,
}

/// Group of nodes to be fused together
#[derive(Debug, Clone)]
pub struct FusionGroup {
    pub nodes: Vec<usize>,
    pub pattern: String,
    pub fusion_type: FusionType,
    pub estimated_benefit: f64,
}

/// Result of applying kernel fusion
#[derive(Debug)]
pub struct FusionResult {
    pub fused_operations: usize,
    pub estimated_speedup: f64,
    pub fusion_time_ms: u64,
    pub applied_patterns: Vec<String>,
}

/// Fusion statistics
#[derive(Debug, Default, Clone)]
pub struct FusionStatistics {
    pub total_fusions: usize,
    pub total_fusion_time: std::time::Duration,
    pub pattern_usage: HashMap<String, usize>,
    pub average_speedup: f64,
}

impl FusionStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn average_fusion_time(&self) -> std::time::Duration {
        if self.total_fusions == 0 {
            std::time::Duration::ZERO
        } else {
            self.total_fusion_time / self.total_fusions as u32
        }
    }
}

/// Fusion configuration for fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Enable element-wise fusion
    pub enable_elementwise: bool,
    /// Enable activation fusion
    pub enable_activation: bool,
    /// Enable normalization fusion
    pub enable_normalization: bool,
    /// Enable attention fusion
    pub enable_attention: bool,
    /// Enable convolution fusion
    pub enable_convolution: bool,
    /// Minimum benefit threshold for fusion
    pub min_benefit_threshold: f64,
    /// Maximum fusion group size
    pub max_group_size: usize,
    /// Memory threshold for fusion (in bytes)
    pub memory_threshold: u64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            enable_elementwise: true,
            enable_activation: true,
            enable_normalization: true,
            enable_attention: true,
            enable_convolution: true,
            min_benefit_threshold: 1.1,
            max_group_size: 10,
            memory_threshold: 1024 * 1024, // 1MB
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{CompilerConfig, ComputationGraph, GraphNode};

    #[test]
    fn test_kernel_fusion_creation() {
        let config = CompilerConfig::default();
        let result = KernelFusion::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fusion_pattern() {
        let pattern = FusionPattern::new(
            "TestPattern",
            vec!["Add", "ReLU"],
            FusionType::Activation,
            1.5,
        );

        assert_eq!(pattern.name, "TestPattern");
        assert_eq!(pattern.operations.len(), 2);
        assert_eq!(pattern.fusion_type, FusionType::Activation);
        assert_eq!(pattern.expected_speedup, 1.5);
    }

    #[test]
    fn test_fusion_types() {
        assert_ne!(FusionType::ElementWise, FusionType::Activation);
        assert_eq!(FusionType::Attention, FusionType::Attention);
    }

    #[test]
    fn test_fusion_config() {
        let config = FusionConfig::default();
        assert!(config.enable_elementwise);
        assert!(config.enable_activation);
        assert!(config.min_benefit_threshold > 1.0);
    }

    #[test]
    fn test_fusion_statistics() {
        let mut stats = FusionStatistics::new();
        assert_eq!(stats.total_fusions, 0);
        assert_eq!(stats.average_fusion_time(), std::time::Duration::ZERO);

        stats.total_fusions = 1;
        stats.total_fusion_time = std::time::Duration::from_millis(100);
        assert_eq!(
            stats.average_fusion_time(),
            std::time::Duration::from_millis(100)
        );
    }

    #[test]
    fn test_fusion_group() {
        let group = FusionGroup {
            nodes: vec![0, 1, 2],
            pattern: "TestPattern".to_string(),
            fusion_type: FusionType::ElementWise,
            estimated_benefit: 1.5,
        };

        assert_eq!(group.nodes.len(), 3);
        assert_eq!(group.pattern, "TestPattern");
        assert_eq!(group.estimated_benefit, 1.5);
    }

    #[test]
    fn test_create_fused_node() {
        let config = CompilerConfig::default();
        let fusion = KernelFusion::new(&config).expect("operation failed in test");

        let mut graph = ComputationGraph::new();

        let node1 = GraphNode {
            id: 0,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            input_shapes: vec![vec![10, 10]],
            output_shapes: vec![vec![10, 10]],
            compute_cost: 50.0,
            memory_cost: 25.0,
        };

        let node2 = GraphNode {
            id: 1,
            op_type: "ReLU".to_string(),
            attributes: HashMap::new(),
            input_shapes: vec![vec![10, 10]],
            output_shapes: vec![vec![10, 10]],
            compute_cost: 30.0,
            memory_cost: 15.0,
        };

        graph.add_node(node1);
        graph.add_node(node2);

        let group = FusionGroup {
            nodes: vec![0, 1],
            pattern: "AddReLU".to_string(),
            fusion_type: FusionType::Activation,
            estimated_benefit: 1.5,
        };

        let result = fusion.create_fused_node(&graph, &group);
        assert!(result.is_ok());

        let fused_node = result.expect("operation failed in test");
        assert!(fused_node.op_type.contains("Fused"));
        assert!(fused_node.compute_cost < 80.0); // Should be less than sum due to fusion benefit
    }

    fn add_node(graph: &mut ComputationGraph, id: usize, op_type: &str, memory_cost: f64) {
        graph.add_node(GraphNode {
            id,
            op_type: op_type.to_string(),
            attributes: HashMap::new(),
            input_shapes: vec![],
            output_shapes: vec![vec![4]],
            compute_cost: 1.0,
            memory_cost,
        });
    }

    /// Regression test: `check_memory_constraints` compared a fusion group's
    /// memory cost against a fixed absolute `1000.0`, meaningless against
    /// `memory_cost`'s undefined unit. A first fix (percentage of the
    /// graph's total) traded that for a check that rejected almost every
    /// realistic small fusion, because every pattern here is 2-6 nodes and
    /// small graphs are the common case (see
    /// `test_apply_fusion_still_fuses_a_real_pattern_in_a_small_graph`
    /// below, which is what caught it). It now scales with the group's own
    /// node count against the graph's average per-node cost, so an
    /// ordinarily-priced group is allowed regardless of graph size, while a
    /// group whose combined cost is disproportionate to what that many
    /// average-priced nodes should cost is rejected.
    #[test]
    fn test_check_memory_constraints_is_graph_relative_not_a_fixed_number() {
        let config = CompilerConfig::default();
        let fusion = KernelFusion::new(&config).expect("construction failed");

        let group = FusionGroup {
            nodes: vec![0, 1],
            pattern: "test".to_string(),
            fusion_type: FusionType::ElementWise,
            estimated_benefit: 1.5,
        };

        // A small graph where every node, including the group's, costs the
        // same: an ordinarily-priced group must be allowed no matter how
        // small the graph around it is.
        let mut uniform_graph = ComputationGraph::new();
        add_node(&mut uniform_graph, 0, "Add", 10.0);
        add_node(&mut uniform_graph, 1, "Add", 10.0);
        let allowed_in_uniform_graph =
            fusion.check_memory_constraints(&group, &uniform_graph).expect("check failed");
        assert!(
            allowed_in_uniform_graph,
            "a 2-node group costing the same as every other node in the graph must be allowed"
        );

        // A graph of many cheap nodes plus one pair (the group) that costs
        // wildly more than the rest: the group is disproportionate to the
        // graph's typical node cost and should be rejected.
        let mut skewed_graph = ComputationGraph::new();
        add_node(&mut skewed_graph, 0, "Add", 500.0);
        add_node(&mut skewed_graph, 1, "Add", 500.0);
        for id in 2..12 {
            add_node(&mut skewed_graph, id, "Add", 1.0);
        }
        let allowed_in_skewed_graph =
            fusion.check_memory_constraints(&group, &skewed_graph).expect("check failed");
        assert!(
            !allowed_in_skewed_graph,
            "a group costing 1000 against 10 other nodes costing 1.0 each is disproportionate \
             and should be rejected"
        );

        assert_ne!(
            allowed_in_uniform_graph, allowed_in_skewed_graph,
            "the verdict must vary with graph content, not stay constant"
        );

        // No cost data at all (e.g. a front-end that never populated
        // `memory_cost`): the constraint has no evidence to reject on, so it
        // must abstain (allow) rather than fail closed on a 0.0 < 0.0 threshold.
        let mut zero_cost_graph = ComputationGraph::new();
        add_node(&mut zero_cost_graph, 0, "Add", 0.0);
        add_node(&mut zero_cost_graph, 1, "Add", 0.0);
        let allowed_with_zero_cost =
            fusion.check_memory_constraints(&group, &zero_cost_graph).expect("check failed");
        assert!(
            allowed_with_zero_cost,
            "a graph with no memory cost data must not be rejected on evidence it doesn't have"
        );
    }

    /// Regression test for the memory-constraint fix above: prove, end to
    /// end through `apply_fusion` (not `check_memory_constraints` in
    /// isolation), that a real fusion pattern in a small, realistically
    /// small graph still actually fuses. An early version of the
    /// graph-relative fix compared a group's cost against a flat percentage
    /// of the graph's total; since every pattern below is 2-6 nodes, that
    /// silently disabled fusion on any graph under roughly a dozen nodes,
    /// which is the common case, including every other graph in this test
    /// module.
    #[test]
    fn test_apply_fusion_still_fuses_a_real_pattern_in_a_small_graph() {
        let config = CompilerConfig::default();
        let mut fusion = KernelFusion::new(&config).expect("construction failed");

        // Matches the "ActivationFusion" pattern (Linear -> ReLU) from
        // `initialize_patterns`, with the same node-cost scale
        // `test_create_fused_node` above uses (tens, not thousands) -- a
        // graph containing nothing but the pattern itself, the minimal case.
        let mut graph = ComputationGraph::new();
        add_node(&mut graph, 0, "Linear", 25.0);
        add_node(&mut graph, 1, "ReLU", 15.0);
        graph.add_edge(crate::compiler::GraphEdge {
            from: 0,
            to: 1,
            output_idx: 0,
            input_idx: 0,
            shape: vec![4],
            dtype: "f32".to_string(),
        });

        let result = fusion.apply_fusion(&mut graph).expect("fusion must succeed");
        assert!(
            result.fused_operations > 0,
            "a real, hardware-compatible, sufficiently-beneficial pattern in a small graph must \
             still be fused -- got 0 fused operations"
        );
        assert!(result.applied_patterns.contains(&"ActivationFusion".to_string()));
    }
}
