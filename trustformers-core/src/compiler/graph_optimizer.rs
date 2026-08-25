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
        let mut stats = HashMap::new();

        let foldable = self.find_foldable_nodes(graph);
        let folded_count = foldable.len();
        let changed = folded_count > 0;

        // Tag newly-foldable nodes rather than delete them. This pass has no
        // expression evaluator (it only tracks *which* nodes are provably
        // constant, never their actual value), so it cannot synthesize a
        // literal replacement node; deleting a folded node here without one
        // would silently drop the input edge of whatever consumes it,
        // corrupting the graph. Tagging (matching `passes::ConstantFoldingPass`'s
        // convention) leaves physical removal to a pass that can safely do
        // it, e.g. dead-code elimination once nothing reads the tag.
        for &node_id in &foldable {
            if let Some(node) = graph.get_node_mut(node_id) {
                node.attributes.insert("constant_folded".to_string(), "true".to_string());
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

    /// Real estimate: runs the same fixed-point constant-propagation `apply`
    /// uses to decide what to tag, read-only, and reports the foldable count
    /// as a fraction of the graph. Empty graphs report 0.0 (not NaN).
    fn estimate_benefit(&self, graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        let foldable = self.find_foldable_nodes(graph);
        Ok(foldable.len() as f64 / graph.nodes.len().max(1) as f64)
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

    /// Nodes whose value is knowable at compile time: every `can_fold_operation`
    /// node (arithmetic/layout op) whose inputs all trace back, transitively,
    /// to `is_constant_operation` producers. Found by propagating to a fixed
    /// point, bounded by the node count so a pathological graph cannot loop
    /// unboundedly. Excludes the seed constant-producer nodes themselves —
    /// they have nothing to fold, they *are* the constants. Shared by `apply`
    /// (which tags the result) and `estimate_benefit` (which only counts it),
    /// so the two can never disagree.
    fn find_foldable_nodes(&self, graph: &ComputationGraph) -> HashSet<usize> {
        let mut constant_nodes: HashSet<usize> = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| self.is_constant_operation(&node.op_type))
            .map(|(i, _)| i)
            .collect();

        let mut foldable = HashSet::new();

        for _ in 0..graph.nodes.len().min(64) {
            let mut newly_constant = Vec::new();
            for (i, node) in graph.nodes.iter().enumerate() {
                if constant_nodes.contains(&i) || !self.can_fold_operation(&node.op_type) {
                    continue;
                }

                let mut incoming = graph.edges.iter().filter(|edge| edge.to == i).peekable();
                // A node with no inputs at all has nothing constant feeding
                // it; `all()` on an empty iterator is vacuously true, so this
                // must be checked explicitly rather than folded in below.
                if incoming.peek().is_none() {
                    continue;
                }
                if incoming.all(|edge| constant_nodes.contains(&edge.from)) {
                    newly_constant.push(i);
                }
            }

            if newly_constant.is_empty() {
                break;
            }

            for i in newly_constant {
                constant_nodes.insert(i);
                foldable.insert(i);
            }
        }

        foldable
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
        let mut stats = HashMap::new();

        let nodes_to_remove = self.find_dead_nodes(graph);
        let removed_count = nodes_to_remove.len();
        let changed = removed_count > 0;

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

    /// Real benefit estimate: runs the exact same reverse-reachability walk
    /// `apply` uses to decide what to delete, but read-only, and reports the
    /// dead-node count as a fraction of the graph. Because `find_dead_nodes`
    /// is the single source of truth for both methods, this can never predict
    /// a different outcome than `apply` actually produces. Empty graphs
    /// report 0.0 (not NaN).
    fn estimate_benefit(&self, graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        let dead_count = self.find_dead_nodes(graph).len();
        Ok(dead_count as f64 / graph.nodes.len().max(1) as f64)
    }
}

impl DeadCodeEliminationPass {
    fn is_output_node(&self, node: &GraphNode) -> bool {
        node.attributes.contains_key("output")
            || node.op_type == "Output"
            || node.op_type == "Return"
    }

    /// Indices of nodes unreachable by reverse DFS from the graph's outputs
    /// (nodes with no outgoing edge, or explicitly marked as an output /
    /// return). Shared by `apply` (which deletes them) and `estimate_benefit`
    /// (which only counts them) so the two can never disagree.
    fn find_dead_nodes(&self, graph: &ComputationGraph) -> Vec<usize> {
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

        reachable
            .iter()
            .enumerate()
            .filter_map(|(i, &is_reachable)| (!is_reachable).then_some(i))
            .collect()
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

                // The duplicate nodes themselves are left in `graph.nodes`,
                // untouched and untagged: redirecting their outgoing edges
                // above is what makes them dead (nothing downstream still
                // reads their output). Splicing them out here would need the
                // same index-shifting care `DeadCodeEliminationPass::apply`
                // takes; a following DCE pass is expected to do that once it
                // can tell a same-signature duplicate apart from a real
                // output (see that pass's reachability rule).
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

        // `.max(1)` keeps an empty graph at a defined 0.0 instead of NaN.
        Ok(potential_duplicates / graph.nodes.len().max(1) as f64)
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

    /// Real estimate: reuses the same `analyze_memory_patterns` scan `apply`
    /// already runs (and publishes as `memory_efficiency_score` /
    /// `cache_friendly_ops`), expressed as the fraction of the graph's nodes
    /// that are memory-intensive but not yet laid out cache-friendly. Varies
    /// with the graph; 0.0 when nothing memory-intensive is present.
    fn estimate_benefit(&self, graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        let memory_analysis = self.analyze_memory_patterns(graph);
        let improvable_ops = memory_analysis
            .total_memory_ops
            .saturating_sub(memory_analysis.cache_friendly_ops);
        Ok(improvable_ops as f64 / graph.nodes.len().max(1) as f64)
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
    total_memory_ops: usize,
}

/// Operation reordering pass.
///
/// NOT YET IMPLEMENTED: `apply` performs no reordering and unconditionally
/// reports `changed: false` for every graph. `estimate_benefit` reports the
/// true benefit of running this pass today, which is exactly 0.0 — not a
/// placeholder guess. That keeps the pipeline's `estimated_benefit < 0.01`
/// gate (see `GraphOptimizer::optimize`) skipping it honestly instead of
/// running a no-op and publishing a fabricated benefit number for it.
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
        "Not yet implemented: intended to reorder operations for better parallelization"
    }

    fn apply(&mut self, _graph: &mut ComputationGraph) -> Result<PassResult, TrustformersError> {
        // Not yet implemented: no reordering is performed.
        Ok(PassResult {
            changed: false,
            stats: HashMap::new(),
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(&self, _graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        // `apply` above is a true no-op for every input, so the honest
        // benefit of running it is 0.0, not an invented nonzero guess.
        Ok(0.0)
    }
}

/// Loop optimization pass.
///
/// NOT YET IMPLEMENTED: this compiler's `ComputationGraph` is a DAG of tensor
/// operations with no loop constructs (see `analysis::GraphAnalyzer::analyze_loops`,
/// which refuses for the same reason), so `apply` has nothing to optimize and
/// unconditionally reports `changed: false`. `estimate_benefit` reports the
/// true current benefit, 0.0, so the pipeline's benefit gate skips it rather
/// than running a no-op under a fabricated nonzero score.
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
        "Not yet implemented: the graph IR has no loop constructs to optimize"
    }

    fn apply(&mut self, _graph: &mut ComputationGraph) -> Result<PassResult, TrustformersError> {
        // Not yet implemented: no loop constructs exist in this IR to act on.
        Ok(PassResult {
            changed: false,
            stats: HashMap::new(),
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(&self, _graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        // `apply` above is a true no-op for every input, so the honest
        // benefit of running it is 0.0, not an invented nonzero guess.
        Ok(0.0)
    }
}

/// Advanced optimization pass.
///
/// NOT YET IMPLEMENTED: placeholder for future techniques; `apply` performs
/// no transformation and unconditionally reports `changed: false`.
/// `estimate_benefit` reports the true current benefit, 0.0, so the
/// pipeline's benefit gate skips it rather than running a no-op under a
/// fabricated nonzero score.
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
        "Not yet implemented: reserved for future advanced optimization techniques"
    }

    fn apply(&mut self, _graph: &mut ComputationGraph) -> Result<PassResult, TrustformersError> {
        // Not yet implemented: no transformation is performed.
        Ok(PassResult {
            changed: false,
            stats: HashMap::new(),
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(&self, _graph: &ComputationGraph) -> Result<f64, TrustformersError> {
        // `apply` above is a true no-op for every input, so the honest
        // benefit of running it is 0.0, not an invented nonzero guess.
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{CompilerConfig, GraphEdge};

    fn node(id: usize, op_type: &str) -> GraphNode {
        GraphNode {
            id,
            op_type: op_type.to_string(),
            attributes: HashMap::new(),
            input_shapes: Vec::new(),
            output_shapes: vec![vec![4, 4]],
            compute_cost: 10.0,
            memory_cost: 10.0,
        }
    }

    fn edge(from: usize, to: usize) -> GraphEdge {
        GraphEdge {
            from,
            to,
            output_idx: 0,
            input_idx: 0,
            shape: vec![4, 4],
            dtype: "f32".to_string(),
        }
    }

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

    // -- De-fabrication regression tests -----------------------------------
    //
    // Each of these previously returned a constant regardless of graph
    // content (see the module's git history / Wave 6d report). They now
    // compute the estimate from the graph, so these tests prove: (a) two
    // different graphs produce different estimates, and (b) an empty graph
    // reports a defined 0.0 rather than NaN.

    #[test]
    fn test_dead_code_elimination_estimate_benefit_empty_graph_is_zero_not_nan() {
        let pass = DeadCodeEliminationPass::new();
        let graph = ComputationGraph::new();
        let benefit = pass.estimate_benefit(&graph).expect("estimate must succeed");
        assert_eq!(
            benefit, 0.0,
            "empty graph must report a defined 0.0, not NaN"
        );
    }

    #[test]
    fn test_dead_code_elimination_estimate_benefit_matches_apply_and_is_input_dependent() {
        let pass = DeadCodeEliminationPass::new();

        // A single node with no edges at all is (by this pass's own reverse-
        // reachability rule) auto-classified as a graph output, so nothing
        // is dead.
        let mut all_live = ComputationGraph::new();
        all_live.add_node(node(0, "Add"));
        let live_benefit = pass.estimate_benefit(&all_live).expect("estimate must succeed");
        assert_eq!(live_benefit, 0.0);

        // Nodes 0 and 1 reference only each other (0 -> 1 -> 0): neither is
        // a sink (each has an outgoing edge) and neither is reachable from
        // node 2, the graph's one true output (no edges at all), so both are
        // genuinely dead -- a real, if unusual, DCE scenario (a causally
        // isolated component), not a contrived input.
        let mut with_dead_component = ComputationGraph::new();
        with_dead_component.add_node(node(0, "Add"));
        with_dead_component.add_node(node(1, "Mul"));
        with_dead_component.add_node(node(2, "Sub"));
        with_dead_component.add_edge(edge(0, 1));
        with_dead_component.add_edge(edge(1, 0));

        let dead_benefit =
            pass.estimate_benefit(&with_dead_component).expect("estimate must succeed");
        assert_ne!(
            live_benefit, dead_benefit,
            "estimate must vary with graph content, not stay constant"
        );
        assert!(
            (dead_benefit - 2.0 / 3.0).abs() < 1e-9,
            "expected 2 of 3 nodes dead, got {dead_benefit}"
        );

        // The estimate must correspond exactly to what `apply` actually
        // removes -- the whole point of sharing `find_dead_nodes` between
        // the two methods instead of guessing independently.
        let mut pass = DeadCodeEliminationPass::new();
        let mut graph_to_mutate = with_dead_component.clone();
        let result = pass.apply(&mut graph_to_mutate).expect("apply must succeed");
        assert_eq!(result.stats["nodes_removed"], 2.0);
        assert_eq!(graph_to_mutate.nodes.len(), 1);
    }

    #[test]
    fn test_constant_folding_estimate_benefit_is_input_dependent_and_honest_on_no_op() {
        let pass = ConstantFoldingPass::new();

        // No constants anywhere: nothing is foldable.
        let mut no_constants = ComputationGraph::new();
        no_constants.add_node(node(0, "MatMul"));
        no_constants.add_node(node(1, "Add"));
        no_constants.add_edge(edge(0, 1));
        let no_fold_benefit = pass.estimate_benefit(&no_constants).expect("estimate must succeed");
        assert_eq!(no_fold_benefit, 0.0);

        // Constant -> Add: the Add's only input traces back to a literal
        // constant, so it is genuinely foldable.
        let mut with_constant = ComputationGraph::new();
        with_constant.add_node(node(0, "Constant"));
        with_constant.add_node(node(1, "Add"));
        with_constant.add_edge(edge(0, 1));
        let fold_benefit = pass.estimate_benefit(&with_constant).expect("estimate must succeed");

        assert_ne!(
            no_fold_benefit, fold_benefit,
            "estimate must vary with graph content, not stay constant"
        );
        assert!(
            fold_benefit > 0.0,
            "graph with a real foldable node must report nonzero benefit"
        );

        // `apply` on the same graph must actually tag exactly the node the
        // estimate credited, proving the two do not disagree.
        let mut pass = ConstantFoldingPass::new();
        let mut graph_to_mutate = with_constant.clone();
        let result = pass.apply(&mut graph_to_mutate).expect("apply must succeed");
        assert_eq!(result.stats["constants_folded"], 1.0);
        assert_eq!(
            graph_to_mutate.nodes[1].attributes.get("constant_folded").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn test_memory_layout_estimate_benefit_is_input_dependent() {
        let pass = MemoryLayoutOptimizationPass::new();

        // Only cache-friendly, non-memory-intensive ops: nothing to improve.
        let mut cache_friendly = ComputationGraph::new();
        cache_friendly.add_node(node(0, "Add"));
        cache_friendly.add_node(node(1, "ReLU"));
        let friendly_benefit =
            pass.estimate_benefit(&cache_friendly).expect("estimate must succeed");
        assert_eq!(friendly_benefit, 0.0);

        // A MatMul is memory-intensive and not cache-friendly: real headroom.
        let mut with_matmul = ComputationGraph::new();
        with_matmul.add_node(node(0, "MatMul"));
        let matmul_benefit = pass.estimate_benefit(&with_matmul).expect("estimate must succeed");

        assert_ne!(
            friendly_benefit, matmul_benefit,
            "estimate must vary with graph content, not stay constant"
        );
        assert!(matmul_benefit > 0.0);
    }

    #[test]
    fn test_maximum_level_skips_unimplemented_passes_but_still_runs_real_ones() {
        let config = CompilerConfig {
            optimization_level: OptimizationLevel::Maximum,
            ..CompilerConfig::default()
        };
        let mut optimizer = GraphOptimizer::new(&config).expect("optimizer construction failed");

        // Constant -> MatMul: gives ConstantFolding and MemoryLayoutOptimization
        // real, nonzero benefit so the test is not vacuous, while every
        // acyclic graph gives DeadCodeElimination and CommonSubexpression
        // nothing to do -- none of that affects the assertion below, which
        // is specifically about the three not-yet-implemented passes.
        let mut graph = ComputationGraph::new();
        graph.add_node(node(0, "Constant"));
        graph.add_node(node(1, "MatMul"));
        graph.add_edge(edge(0, 1));

        let result = optimizer.optimize(graph).expect("optimize must succeed");
        let names: Vec<&str> = result.pass_results.iter().map(|r| r.pass_name.as_str()).collect();

        assert!(
            !names.is_empty(),
            "expected at least one real pass to clear the benefit gate, got none: {names:?}"
        );

        for unimplemented_pass in [
            "OperationReordering",
            "LoopOptimization",
            "AdvancedOptimization",
        ] {
            assert!(
                !names.contains(&unimplemented_pass),
                "{unimplemented_pass} is not yet implemented (apply is a true no-op); it must \
                 report 0.0 benefit and be skipped by the pipeline's benefit gate rather than \
                 appear in pass_results: {names:?}"
            );
        }
    }

    #[test]
    fn test_unimplemented_passes_report_honest_zero_benefit() {
        let mut graph = ComputationGraph::new();
        graph.add_node(node(0, "MatMul"));
        graph.add_node(node(1, "Add"));
        graph.add_edge(edge(0, 1));

        assert_eq!(
            OperationReorderingPass::new().estimate_benefit(&graph).expect("ok"),
            0.0
        );
        assert_eq!(
            LoopOptimizationPass::new().estimate_benefit(&graph).expect("ok"),
            0.0
        );
        assert_eq!(
            AdvancedOptimizationPass::new().estimate_benefit(&graph).expect("ok"),
            0.0
        );
    }
}
