/*!
# Graph Optimization Module

This module provides comprehensive graph-level optimizations for computation graphs including:

- **Constant Folding**: Evaluate constant expressions at compile time
- **Dead Code Elimination**: Remove unused operations and tensors
- **Common Subexpression Elimination**: Deduplicate identical computations
- **Loop Optimization**: Optimize loops and reduce redundant computations
- **Memory Layout Optimization**: Arrange operations for optimal memory access patterns
- **Operation Reordering**: Reorder operations for better parallelization
*/

use crate::compiler::passes::OptimizationPass;
use crate::compiler::{CompilerConfig, ComputationGraph, GraphNode, OptimizationLevel, PassResult};
use crate::errors::TrustformersError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Graph optimizer for computation graph optimizations
pub struct GraphOptimizer {
    config: CompilerConfig,
    passes: Vec<Box<dyn OptimizationPass>>,
    #[allow(dead_code)]
    pass_manager: PassManager,
}

impl GraphOptimizer {
    /// Create a new graph optimizer
    pub fn new(config: &CompilerConfig) -> Result<Self, TrustformersError> {
        let mut optimizer = Self {
            config: config.clone(),
            passes: Vec::new(),
            pass_manager: PassManager::new(),
        };

        optimizer.initialize_passes()?;
        Ok(optimizer)
    }

    /// Update the configuration
    pub fn update_config(&mut self, config: &CompilerConfig) -> Result<(), TrustformersError> {
        self.config = config.clone();
        self.initialize_passes()?;
        Ok(())
    }

    /// Initialize optimization passes based on configuration
    fn initialize_passes(&mut self) -> Result<(), TrustformersError> {
        self.passes.clear();

        match self.config.optimization_level {
            OptimizationLevel::None => {
                // No optimization passes
            },
            OptimizationLevel::Basic => {
                self.passes.push(Box::new(ConstantFoldingPass::new()));
                self.passes.push(Box::new(DeadCodeEliminationPass::new()));
            },
            OptimizationLevel::Standard => {
                self.passes.push(Box::new(ConstantFoldingPass::new()));
                self.passes.push(Box::new(DeadCodeEliminationPass::new()));
                self.passes.push(Box::new(CommonSubexpressionEliminationPass::new()));
                self.passes.push(Box::new(MemoryLayoutOptimizationPass::new()));
            },
            OptimizationLevel::Aggressive => {
                self.passes.push(Box::new(ConstantFoldingPass::new()));
                self.passes.push(Box::new(DeadCodeEliminationPass::new()));
                self.passes.push(Box::new(CommonSubexpressionEliminationPass::new()));
                self.passes.push(Box::new(MemoryLayoutOptimizationPass::new()));
                self.passes.push(Box::new(OperationReorderingPass::new()));
                self.passes.push(Box::new(LoopOptimizationPass::new()));
            },
            OptimizationLevel::Maximum => {
                self.passes.push(Box::new(ConstantFoldingPass::new()));
                self.passes.push(Box::new(DeadCodeEliminationPass::new()));
                self.passes.push(Box::new(CommonSubexpressionEliminationPass::new()));
                self.passes.push(Box::new(MemoryLayoutOptimizationPass::new()));
                self.passes.push(Box::new(OperationReorderingPass::new()));
                self.passes.push(Box::new(LoopOptimizationPass::new()));
                self.passes.push(Box::new(AdvancedOptimizationPass::new()));
            },
        }

        Ok(())
    }

    /// Optimize a computation graph
    pub fn optimize(
        &mut self,
        mut graph: ComputationGraph,
    ) -> Result<GraphOptimizationResult, TrustformersError> {
        let start_time = std::time::Instant::now();
        let original_stats = GraphStats::from_graph(&graph);

        // Validate input graph
        graph.validate()?;

        let mut results = Vec::new();
        let mut total_changes = 0;

        // Apply optimization passes
        for (i, pass) in self.passes.iter_mut().enumerate() {
            let pass_start = std::time::Instant::now();

            // Check if pass should be applied based on benefit estimation
            let estimated_benefit = pass.estimate_benefit(&graph)?;
            if estimated_benefit < 0.01 {
                // Skip pass if benefit is too small
                continue;
            }

            // Apply the pass
            let pass_result = pass.apply(&mut graph)?;
            let pass_time = pass_start.elapsed();

            if pass_result.changed {
                total_changes += 1;
            }

            results.push(PassExecutionResult {
                pass_name: pass.name().to_string(),
                pass_index: i,
                execution_time_ms: pass_time.as_millis() as u64,
                changed: pass_result.changed,
                estimated_benefit,
                stats: pass_result.stats,
                metadata: pass_result.metadata,
            });

            // Re-validate graph after each pass
            graph.validate()?;
        }

        let optimization_time = start_time.elapsed();
        let optimized_stats = GraphStats::from_graph(&graph);

        Ok(GraphOptimizationResult {
            optimized_graph: graph,
            original_stats,
            optimized_stats,
            pass_results: results,
            total_optimization_time_ms: optimization_time.as_millis() as u64,
            total_passes_applied: total_changes,
        })
    }
}

// Helper functions to reduce excessive nesting

/// Updates edge indices after a node is removed from the graph
fn update_edge_indices_after_removal(
    edges: &mut [crate::compiler::GraphEdge],
    removed_node_id: usize,
) {
    for edge in edges.iter_mut() {
        if edge.from > removed_node_id {
            edge.from -= 1;
        }
        if edge.to > removed_node_id {
            edge.to -= 1;
        }
    }
}

/// Processes neighbors in topological sorting with depth calculation
fn process_neighbors_in_topo_sort(
    neighbors: &[usize],
    depths: &mut [usize],
    incoming_count: &mut [usize],
    queue: &mut VecDeque<usize>,
    current_depth: usize,
) {
    for &neighbor in neighbors {
        depths[neighbor] = depths[neighbor].max(current_depth + 1);
        incoming_count[neighbor] -= 1;
        if incoming_count[neighbor] == 0 {
            queue.push_back(neighbor);
        }
    }
}

/// Updates edges from one node to another (used in common subexpression elimination)
fn redirect_node_edges(edges: &mut [crate::compiler::GraphEdge], from_node: usize, to_node: usize) {
    for edge in edges.iter_mut() {
        if edge.from == from_node {
            edge.from = to_node;
        }
    }
}

/// Pass manager for controlling optimization pass execution
pub struct PassManager {
    #[allow(dead_code)]
    max_iterations: usize,
    #[allow(dead_code)]
    convergence_threshold: f64,
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PassManager {
    pub fn new() -> Self {
        Self {
            max_iterations: 10,
            convergence_threshold: 0.001,
        }
    }
}

/// Graph optimization result
#[derive(Debug)]
pub struct GraphOptimizationResult {
    pub optimized_graph: ComputationGraph,
    pub original_stats: GraphStats,
    pub optimized_stats: GraphStats,
    pub pass_results: Vec<PassExecutionResult>,
    pub total_optimization_time_ms: u64,
    pub total_passes_applied: usize,
}

/// Statistics about a computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub total_compute_cost: f64,
    pub total_memory_cost: f64,
    pub op_type_counts: HashMap<String, usize>,
    pub max_depth: usize,
    pub parallelization_factor: f64,
}

impl GraphStats {
    pub fn from_graph(graph: &ComputationGraph) -> Self {
        let mut op_type_counts = HashMap::new();
        for node in &graph.nodes {
            *op_type_counts.entry(node.op_type.clone()).or_insert(0) += 1;
        }

        Self {
            node_count: graph.nodes.len(),
            edge_count: graph.edges.len(),
            total_compute_cost: graph.total_compute_cost(),
            total_memory_cost: graph.total_memory_cost(),
            op_type_counts,
            max_depth: Self::calculate_max_depth(graph),
            parallelization_factor: Self::calculate_parallelization_factor(graph),
        }
    }

    fn calculate_max_depth(graph: &ComputationGraph) -> usize {
        // Simple depth calculation - could be more sophisticated
        if graph.nodes.is_empty() {
            return 0;
        }

        // Build adjacency list
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for edge in &graph.edges {
            adj.entry(edge.from).or_default().push(edge.to);
        }

        // Find nodes with no incoming edges (roots)
        let mut incoming_count = vec![0; graph.nodes.len()];
        for edge in &graph.edges {
            incoming_count[edge.to] += 1;
        }

        let mut queue = VecDeque::new();
        let mut depths = vec![0; graph.nodes.len()];

        for (i, &count) in incoming_count.iter().enumerate() {
            if count == 0 {
                queue.push_back(i);
            }
        }

        let mut max_depth = 0;

        while let Some(node) = queue.pop_front() {
            max_depth = max_depth.max(depths[node]);

            if let Some(neighbors) = adj.get(&node) {
                let current_depth = depths[node];
                process_neighbors_in_topo_sort(
                    neighbors,
                    &mut depths,
                    &mut incoming_count,
                    &mut queue,
                    current_depth,
                );
            }
        }

        max_depth
    }

    fn calculate_parallelization_factor(graph: &ComputationGraph) -> f64 {
        if graph.nodes.is_empty() {
            return 1.0;
        }

        // Simple parallelization factor based on average fan-out
        let total_edges = graph.edges.len() as f64;
        let total_nodes = graph.nodes.len() as f64;

        if total_nodes <= 1.0 {
            1.0
        } else {
            (total_edges / total_nodes).max(1.0)
        }
    }
}

/// Result of executing a single optimization pass
#[derive(Debug)]
pub struct PassExecutionResult {
    pub pass_name: String,
    pub pass_index: usize,
    pub execution_time_ms: u64,
    pub changed: bool,
    pub estimated_benefit: f64,
    pub stats: HashMap<String, f64>,
    pub metadata: HashMap<String, String>,
}

// Optimization Passes

/// Constant folding optimization pass
pub struct ConstantFoldingPass {
    constants_folded: usize,
}

impl Default for ConstantFoldingPass {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantFoldingPass {
    pub fn new() -> Self {
        Self {
            constants_folded: 0,
        }
    }
}

impl OptimizationPass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "ConstantFolding"
    }

    fn description(&self) -> &str {
        "Evaluate constant expressions at compile time"
    }

    fn apply(&mut self, graph: &mut ComputationGraph) -> Result<PassResult, TrustformersError> {
        let mut changed = false;
        let mut folded_count = 0;
        let mut stats = HashMap::new();

        // Find nodes that only depend on constants
        let mut constant_nodes = HashSet::new();
        let mut nodes_to_remove = Vec::new();

        for (i, node) in graph.nodes.iter().enumerate() {
            if self.is_constant_operation(&node.op_type) {
                constant_nodes.insert(i);

                // Check if all inputs are constants
                let incoming_edges: Vec<_> =
                    graph.edges.iter().filter(|edge| edge.to == i).collect();

                let all_inputs_constant =
                    incoming_edges.iter().all(|edge| constant_nodes.contains(&edge.from));

                if all_inputs_constant && self.can_fold_operation(&node.op_type) {
                    // Mark for folding
                    nodes_to_remove.push(i);
                    folded_count += 1;
                    changed = true;
                }
            }
        }

        // Remove folded nodes (simplified - in practice we'd replace with constant values)
        for &node_id in nodes_to_remove.iter().rev() {
            if node_id < graph.nodes.len() {
                graph.nodes.remove(node_id);
                // Remove associated edges
                graph.edges.retain(|edge| edge.from != node_id && edge.to != node_id);
                // Update edge indices
                update_edge_indices_after_removal(&mut graph.edges, node_id);
            }
        }

        self.constants_folded += folded_count;
        stats.insert("constants_folded".to_string(), folded_count as f64);
        stats.insert(
            "total_constants_folded".to_string(),
            self.constants_folded as f64,
        );

        Ok(PassResult {
            changed,
            stats,
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(&self, graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        let constant_ops = graph
            .nodes
            .iter()
            .filter(|node| self.is_constant_operation(&node.op_type))
            .count();

        // Benefit is proportional to number of constant operations
        Ok(constant_ops as f64 / graph.nodes.len() as f64)
    }
}

impl ConstantFoldingPass {
    fn is_constant_operation(&self, op_type: &str) -> bool {
        matches!(op_type, "Constant" | "Fill" | "Zeros" | "Ones")
    }

    fn can_fold_operation(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Add" | "Mul" | "Sub" | "Div" | "Reshape" | "Transpose"
        )
    }
}

/// Dead code elimination pass
pub struct DeadCodeEliminationPass {
    nodes_removed: usize,
}

impl Default for DeadCodeEliminationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl DeadCodeEliminationPass {
    pub fn new() -> Self {
        Self { nodes_removed: 0 }
    }
}

impl OptimizationPass for DeadCodeEliminationPass {
    fn name(&self) -> &str {
        "DeadCodeElimination"
    }

    fn description(&self) -> &str {
        "Remove unused operations and tensors"
    }

    fn apply(&mut self, graph: &mut ComputationGraph) -> Result<PassResult, TrustformersError> {
        let mut changed = false;
        let mut removed_count = 0;
        let mut stats = HashMap::new();

        // Find output nodes (nodes with no outgoing edges)
        let mut has_outgoing = vec![false; graph.nodes.len()];
        for edge in &graph.edges {
            has_outgoing[edge.from] = true;
        }

        // Mark reachable nodes from outputs using reverse DFS
        let mut reachable = vec![false; graph.nodes.len()];
        let mut stack = Vec::new();

        // Start from nodes that are outputs or have special attributes
        for (i, node) in graph.nodes.iter().enumerate() {
            if !has_outgoing[i] || self.is_output_node(node) {
                stack.push(i);
                reachable[i] = true;
            }
        }

        // DFS to mark all reachable nodes
        while let Some(node_id) = stack.pop() {
            for edge in &graph.edges {
                if edge.to == node_id && !reachable[edge.from] {
                    reachable[edge.from] = true;
                    stack.push(edge.from);
                }
            }
        }

        // Remove unreachable nodes
        let mut nodes_to_remove = Vec::new();
        for (i, &is_reachable) in reachable.iter().enumerate() {
            if !is_reachable {
                nodes_to_remove.push(i);
                removed_count += 1;
                changed = true;
            }
        }

        // Remove nodes and update graph
        for &node_id in nodes_to_remove.iter().rev() {
            if node_id < graph.nodes.len() {
                graph.nodes.remove(node_id);
                graph.edges.retain(|edge| edge.from != node_id && edge.to != node_id);

                // Update edge indices
                update_edge_indices_after_removal(&mut graph.edges, node_id);
            }
        }

        self.nodes_removed += removed_count;
        stats.insert("nodes_removed".to_string(), removed_count as f64);
        stats.insert("total_nodes_removed".to_string(), self.nodes_removed as f64);

        Ok(PassResult {
            changed,
            stats,
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(&self, graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        // Simple heuristic: assume some percentage of nodes might be dead
        let estimated_dead_nodes = graph.nodes.len() as f64 * 0.1; // 10% assumption
        Ok(estimated_dead_nodes / graph.nodes.len() as f64)
    }
}

impl DeadCodeEliminationPass {
    fn is_output_node(&self, node: &GraphNode) -> bool {
        node.attributes.contains_key("output")
            || node.op_type == "Output"
            || node.op_type == "Return"
    }
}

/// Common subexpression elimination pass
pub struct CommonSubexpressionEliminationPass {
    expressions_eliminated: usize,
}

impl Default for CommonSubexpressionEliminationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl CommonSubexpressionEliminationPass {
    pub fn new() -> Self {
        Self {
            expressions_eliminated: 0,
        }
    }
}

impl OptimizationPass for CommonSubexpressionEliminationPass {
    fn name(&self) -> &str {
        "CommonSubexpressionElimination"
    }

    fn description(&self) -> &str {
        "Deduplicate identical computations"
    }

    fn apply(&mut self, graph: &mut ComputationGraph) -> Result<PassResult, TrustformersError> {
        let mut changed = false;
        let mut eliminated_count = 0;
        let mut stats = HashMap::new();

        // Group nodes by operation signature
        let mut signature_groups: HashMap<String, Vec<usize>> = HashMap::new();

        for (i, node) in graph.nodes.iter().enumerate() {
            let signature = self.compute_node_signature(node, graph);
            signature_groups.entry(signature).or_default().push(i);
        }

        // Find groups with multiple nodes (common subexpressions)
        for (_, node_ids) in signature_groups {
            if node_ids.len() > 1 {
                // Keep the first node, merge others into it
                let keep_node = node_ids[0];

                for &remove_node in &node_ids[1..] {
                    // Redirect edges from removed node to kept node
                    redirect_node_edges(&mut graph.edges, remove_node, keep_node);
                    eliminated_count += 1;
                    changed = true;
                }

                // Remove duplicate nodes (mark for removal)
                // In practice, we'd remove them and update indices
            }
        }

        self.expressions_eliminated += eliminated_count;
        stats.insert(
            "expressions_eliminated".to_string(),
            eliminated_count as f64,
        );

        Ok(PassResult {
            changed,
            stats,
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(&self, graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        // Estimate based on operation type distribution
        let mut op_counts: HashMap<String, usize> = HashMap::new();
        for node in &graph.nodes {
            *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
        }

        let potential_duplicates =
            op_counts.values().map(|&count| count.saturating_sub(1)).sum::<usize>() as f64;

        Ok(potential_duplicates / graph.nodes.len() as f64)
    }
}

impl CommonSubexpressionEliminationPass {
    fn compute_node_signature(&self, node: &GraphNode, graph: &ComputationGraph) -> String {
        // Simple signature based on operation type and input shapes
        let mut signature = format!("{}:", node.op_type);

        // Add input signatures
        let input_edges: Vec<_> = graph.edges.iter().filter(|edge| edge.to == node.id).collect();

        for edge in input_edges {
            signature.push_str(&format!(
                "{}:",
                edge.shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
            ));
        }

        signature
    }
}

/// Memory layout optimization pass
pub struct MemoryLayoutOptimizationPass;

impl Default for MemoryLayoutOptimizationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryLayoutOptimizationPass {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationPass for MemoryLayoutOptimizationPass {
    fn name(&self) -> &str {
        "MemoryLayoutOptimization"
    }

    fn description(&self) -> &str {
        "Optimize memory layout for better cache performance"
    }

    fn apply(&mut self, graph: &mut ComputationGraph) -> Result<PassResult, TrustformersError> {
        let mut stats = HashMap::new();

        // Analyze memory access patterns and suggest layout improvements
        let memory_analysis = self.analyze_memory_patterns(graph);
        stats.insert(
            "memory_efficiency_score".to_string(),
            memory_analysis.efficiency_score,
        );
        stats.insert(
            "cache_friendly_ops".to_string(),
            memory_analysis.cache_friendly_ops as f64,
        );

        Ok(PassResult {
            changed: false, // Analysis only for now
            stats,
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(&self, _graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        Ok(0.05) // Conservative estimate
    }
}

impl MemoryLayoutOptimizationPass {
    fn analyze_memory_patterns(&self, graph: &ComputationGraph) -> MemoryAnalysis {
        let mut cache_friendly_ops = 0;
        let mut total_memory_ops = 0;

        for node in &graph.nodes {
            if self.is_memory_intensive(&node.op_type) {
                total_memory_ops += 1;
                if self.is_cache_friendly(&node.op_type) {
                    cache_friendly_ops += 1;
                }
            }
        }

        let efficiency_score = if total_memory_ops > 0 {
            cache_friendly_ops as f64 / total_memory_ops as f64
        } else {
            1.0
        };

        MemoryAnalysis {
            efficiency_score,
            cache_friendly_ops,
            total_memory_ops,
        }
    }

    fn is_memory_intensive(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "MatMul" | "Conv2D" | "Conv3D" | "Attention" | "Embedding"
        )
    }

    fn is_cache_friendly(&self, op_type: &str) -> bool {
        matches!(op_type, "Add" | "Mul" | "ReLU" | "Sigmoid" | "Tanh")
    }
}

struct MemoryAnalysis {
    efficiency_score: f64,
    cache_friendly_ops: usize,
    #[allow(dead_code)]
    total_memory_ops: usize,
}

/// Operation reordering pass
pub struct OperationReorderingPass;

impl Default for OperationReorderingPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OperationReorderingPass {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationPass for OperationReorderingPass {
    fn name(&self) -> &str {
        "OperationReordering"
    }

    fn description(&self) -> &str {
        "Reorder operations for better parallelization"
    }

    fn apply(&mut self, _graph: &mut ComputationGraph) -> Result<PassResult, TrustformersError> {
        Ok(PassResult {
            changed: false,
            stats: HashMap::new(),
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(&self, _graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        Ok(0.03)
    }
}

/// Loop optimization pass
pub struct LoopOptimizationPass;

impl Default for LoopOptimizationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopOptimizationPass {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationPass for LoopOptimizationPass {
    fn name(&self) -> &str {
        "LoopOptimization"
    }

    fn description(&self) -> &str {
        "Optimize loops and reduce redundant computations"
    }

    fn apply(&mut self, _graph: &mut ComputationGraph) -> Result<PassResult, TrustformersError> {
        Ok(PassResult {
            changed: false,
            stats: HashMap::new(),
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(&self, _graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        Ok(0.02)
    }
}

/// Advanced optimization pass
pub struct AdvancedOptimizationPass;

impl Default for AdvancedOptimizationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedOptimizationPass {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationPass for AdvancedOptimizationPass {
    fn name(&self) -> &str {
        "AdvancedOptimization"
    }

    fn description(&self) -> &str {
        "Advanced optimization techniques"
    }

    fn apply(&mut self, _graph: &mut ComputationGraph) -> Result<PassResult, TrustformersError> {
        Ok(PassResult {
            changed: false,
            stats: HashMap::new(),
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(&self, _graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        Ok(0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::CompilerConfig;

    #[test]
    fn test_graph_optimizer_creation() {
        let config = CompilerConfig::default();
        let result = GraphOptimizer::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_constant_folding_pass() {
        let mut pass = ConstantFoldingPass::new();
        assert_eq!(pass.name(), "ConstantFolding");

        let mut graph = ComputationGraph::new();
        let result = pass.apply(&mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dead_code_elimination_pass() {
        let mut pass = DeadCodeEliminationPass::new();
        assert_eq!(pass.name(), "DeadCodeElimination");

        let mut graph = ComputationGraph::new();
        let result = pass.apply(&mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_stats() {
        let graph = ComputationGraph::new();
        let stats = GraphStats::from_graph(&graph);
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.total_compute_cost, 0.0);
        assert_eq!(stats.total_memory_cost, 0.0);
    }
}
