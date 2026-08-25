//! Optimization Passes Module
//!
//! This module provides a comprehensive set of optimization passes for computation graphs including:
//!
//! - **Graph-Level Passes**: Constant folding, dead code elimination, common subexpression elimination
//! - **Memory Passes**: Memory layout optimization, buffer reuse, allocation coalescing
//! - **Compute Passes**: Operation fusion, loop optimizations, vectorization
//! - **Hardware-Specific Passes**: Target-specific optimizations for CPU, GPU, TPU
//!
//! Each pass implements the OptimizationPass trait and can be composed into optimization pipelines.

#![allow(clippy::excessive_nesting)] // Complex compiler optimization algorithms require deep nesting
#![allow(unused_variables)] // Compiler passes

use crate::compiler::{ComputationGraph, GraphNode, PassResult};
use std::collections::{HashMap, HashSet};

/// Trait for optimization passes
pub trait OptimizationPass {
    /// Name of the optimization pass
    fn name(&self) -> &str;

    /// Description of what the pass does
    fn description(&self) -> &str;

    /// Whether this pass requires specific hardware features
    fn hardware_requirements(&self) -> Vec<String> {
        Vec::new()
    }

    /// Apply the optimization pass
    fn apply(
        &mut self,
        graph: &mut ComputationGraph,
    ) -> Result<PassResult, crate::errors::TrustformersError>;

    /// Estimate the benefit of applying this pass
    fn estimate_benefit(
        &self,
        graph: &ComputationGraph,
    ) -> Result<f64, crate::errors::TrustformersError>;
}

/// Constant folding optimization pass
pub struct ConstantFoldingPass {
    #[allow(dead_code)]
    constant_values: HashMap<usize, Vec<f32>>,
}

impl ConstantFoldingPass {
    pub fn new() -> Self {
        Self {
            constant_values: HashMap::new(),
        }
    }

    /// True for arithmetic ops this pass knows how to fold once their inputs
    /// are all known to be constant.
    fn is_foldable_op(op_type: &str) -> bool {
        matches!(op_type, "Add" | "Mul" | "Sub" | "Div")
    }

    /// Nodes whose value is knowable at compile time: literal constant
    /// producers (op_type `"Constant"`, or an explicit `constant = "true"`
    /// attribute), plus any foldable arithmetic node whose inputs all trace
    /// back, transitively, to one -- found by propagating to a fixed point
    /// (capped at 10 iterations, matching `apply`'s original loop bound, so
    /// a pathological graph cannot loop unboundedly). Shared by `apply`
    /// (which tags the result) and `estimate_benefit` (which only reads it),
    /// so the two can never disagree.
    fn find_constant_nodes(&self, graph: &ComputationGraph) -> HashSet<usize> {
        let mut constants = HashSet::new();
        for (i, node) in graph.nodes.iter().enumerate() {
            if node.op_type == "Constant"
                || node.attributes.get("constant").is_some_and(|v| v == "true")
            {
                constants.insert(i);
            }
        }

        let mut iterations = 0;
        while iterations < 10 {
            // Limit iterations to prevent infinite loops
            let mut new_constants = HashSet::new();

            for (i, node) in graph.nodes.iter().enumerate() {
                if constants.contains(&i) || !Self::is_foldable_op(&node.op_type) {
                    continue;
                }

                let mut input_edges = graph.edges.iter().filter(|edge| edge.to == i).peekable();
                // A node with no inputs at all has nothing constant feeding
                // it; `all()` on an empty iterator is vacuously true, so this
                // must be checked explicitly rather than folded into it.
                if input_edges.peek().is_none() {
                    continue;
                }
                if input_edges.all(|edge| constants.contains(&edge.from)) {
                    new_constants.insert(i);
                }
            }

            if new_constants.is_empty() {
                break;
            }

            constants.extend(new_constants);
            iterations += 1;
        }

        constants
    }
}

impl OptimizationPass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "constant-folding"
    }

    fn description(&self) -> &str {
        "Evaluate constant expressions at compile time"
    }

    fn apply(
        &mut self,
        graph: &mut ComputationGraph,
    ) -> Result<PassResult, crate::errors::TrustformersError> {
        let constants = self.find_constant_nodes(graph);
        let folded_ops = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, node)| constants.contains(i) && Self::is_foldable_op(&node.op_type))
            .count();
        let changed = folded_ops > 0;

        // Tag constant-valued nodes rather than remove them: this pass has
        // no expression evaluator (it tracks *which* nodes are provably
        // constant, never their actual value), so it cannot synthesize a
        // literal replacement node and safely delete the originals without
        // corrupting whatever consumes them. A later pass that can compute
        // and splice in real values is expected to act on the "folded" tag.
        for &constant_id in &constants {
            if let Some(node) = graph.get_node_mut(constant_id) {
                node.attributes.insert("folded".to_string(), "true".to_string());
            }
        }

        let mut stats = HashMap::new();
        stats.insert("folded_operations".to_string(), folded_ops as f64);
        stats.insert("constant_nodes_total".to_string(), constants.len() as f64);

        Ok(PassResult {
            changed,
            stats,
            metadata: HashMap::new(),
        })
    }

    /// Real estimate: runs the same fixed-point constant-propagation `apply`
    /// uses, read-only, and weights each newly-foldable node by its declared
    /// `compute_cost` -- the fraction of the graph's total compute cost that
    /// folding could eliminate. Empty graphs, and graphs with zero total
    /// compute cost, report 0.0 (not NaN).
    fn estimate_benefit(
        &self,
        graph: &ComputationGraph,
    ) -> Result<f64, crate::errors::TrustformersError> {
        let total_compute_cost: f64 = graph.nodes.iter().map(|node| node.compute_cost).sum();
        if total_compute_cost <= 0.0 {
            return Ok(0.0);
        }

        let constants = self.find_constant_nodes(graph);
        let foldable_compute_cost: f64 = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, node)| constants.contains(i) && Self::is_foldable_op(&node.op_type))
            .map(|(_, node)| node.compute_cost)
            .sum();

        Ok(foldable_compute_cost / total_compute_cost)
    }
}

impl Default for ConstantFoldingPass {
    fn default() -> Self {
        Self::new()
    }
}

/// Dead code elimination pass
pub struct DeadCodeEliminationPass {
    output_nodes: HashSet<usize>,
}

impl DeadCodeEliminationPass {
    pub fn new() -> Self {
        Self {
            output_nodes: HashSet::new(),
        }
    }

    /// Mark specific nodes as outputs that must be preserved
    pub fn mark_output(&mut self, node_id: usize) {
        self.output_nodes.insert(node_id);
    }
}

impl OptimizationPass for DeadCodeEliminationPass {
    fn name(&self) -> &str {
        "dead-code-elimination"
    }

    fn description(&self) -> &str {
        "Remove operations that don't contribute to outputs"
    }

    fn apply(
        &mut self,
        graph: &mut ComputationGraph,
    ) -> Result<PassResult, crate::errors::TrustformersError> {
        // Mark live nodes starting from outputs
        let mut live_nodes = HashSet::new();

        // If no specific outputs marked, consider nodes with no outgoing edges as outputs
        if self.output_nodes.is_empty() {
            for (i, _) in graph.nodes.iter().enumerate() {
                let has_outgoing = graph.edges.iter().any(|edge| edge.from == i);
                if !has_outgoing {
                    self.output_nodes.insert(i);
                }
            }
        }

        // Start from output nodes and work backwards
        let mut worklist: Vec<usize> = self.output_nodes.iter().copied().collect();

        while let Some(node_id) = worklist.pop() {
            if live_nodes.contains(&node_id) {
                continue;
            }

            live_nodes.insert(node_id);

            // Add all input nodes to worklist
            for edge in &graph.edges {
                if edge.to == node_id && !live_nodes.contains(&edge.from) {
                    worklist.push(edge.from);
                }
            }
        }

        let original_count = graph.nodes.len();
        let dead_count = original_count - live_nodes.len();

        // Mark dead nodes for removal
        for (i, node) in graph.nodes.iter_mut().enumerate() {
            if !live_nodes.contains(&i) {
                node.attributes.insert("dead".to_string(), "true".to_string());
            }
        }

        // Remove dead edges
        graph
            .edges
            .retain(|edge| live_nodes.contains(&edge.from) && live_nodes.contains(&edge.to));

        let changed = dead_count > 0;

        let mut stats = HashMap::new();
        stats.insert("original_nodes".to_string(), original_count as f64);
        stats.insert("live_nodes".to_string(), live_nodes.len() as f64);
        stats.insert("dead_nodes".to_string(), dead_count as f64);
        stats.insert(
            "elimination_ratio".to_string(),
            if original_count > 0 { dead_count as f64 / original_count as f64 } else { 0.0 },
        );

        Ok(PassResult {
            changed,
            stats,
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(
        &self,
        graph: &ComputationGraph,
    ) -> Result<f64, crate::errors::TrustformersError> {
        // Simple heuristic: estimate based on nodes with no outputs
        let mut potentially_dead = 0;

        for (i, _) in graph.nodes.iter().enumerate() {
            let has_outgoing = graph.edges.iter().any(|edge| edge.from == i);
            if !has_outgoing && !self.output_nodes.contains(&i) {
                potentially_dead += 1;
            }
        }

        Ok(potentially_dead as f64 / graph.nodes.len().max(1) as f64)
    }
}

impl Default for DeadCodeEliminationPass {
    fn default() -> Self {
        Self::new()
    }
}

/// Common subexpression elimination pass
pub struct CommonSubexpressionEliminationPass {
    expression_map: HashMap<String, usize>,
}

impl CommonSubexpressionEliminationPass {
    pub fn new() -> Self {
        Self {
            expression_map: HashMap::new(),
        }
    }

    /// Generate expression signature for a node
    fn expression_signature(&self, node: &GraphNode, graph: &ComputationGraph) -> String {
        let mut signature = node.op_type.clone();

        // Add input signatures in deterministic order
        let mut inputs: Vec<String> = graph
            .edges
            .iter()
            .filter(|edge| edge.to == node.id)
            .map(|edge| format!("{}:{}", edge.from, edge.output_idx))
            .collect();
        inputs.sort();

        signature.push_str(&format!("({})", inputs.join(",")));

        // Add relevant attributes
        for (key, value) in &node.attributes {
            if !key.starts_with("_") {
                // Skip internal attributes
                signature.push_str(&format!("|{}={}", key, value));
            }
        }

        signature
    }
}

impl OptimizationPass for CommonSubexpressionEliminationPass {
    fn name(&self) -> &str {
        "common-subexpression-elimination"
    }

    fn description(&self) -> &str {
        "Eliminate redundant computations by reusing common subexpressions"
    }

    fn apply(
        &mut self,
        graph: &mut ComputationGraph,
    ) -> Result<PassResult, crate::errors::TrustformersError> {
        self.expression_map.clear();
        let mut changed = false;
        let mut eliminated_expressions = 0;

        // Build map of expressions to their first occurrence
        let mut redundant_nodes = Vec::new();
        for (i, node) in graph.nodes.iter().enumerate() {
            let signature = self.expression_signature(node, graph);

            if let Some(&first_occurrence) = self.expression_map.get(&signature) {
                redundant_nodes.push((i, first_occurrence));
                eliminated_expressions += 1;
                changed = true;
            } else {
                self.expression_map.insert(signature, i);
            }
        }

        // Now mark redundant nodes for elimination
        for (node_id, canonical_id) in redundant_nodes {
            if let Some(node) = graph.get_node_mut(node_id) {
                node.attributes.insert("cse_redundant".to_string(), "true".to_string());
                node.attributes.insert("cse_canonical".to_string(), canonical_id.to_string());
            }
        }

        // Update edges to point to canonical expressions
        let mut edge_updates = Vec::new();
        for (edge_idx, edge) in graph.edges.iter().enumerate() {
            if let Some(from_node) = graph.get_node(edge.from) {
                if let Some(canonical_str) = from_node.attributes.get("cse_canonical") {
                    if let Ok(canonical_id) = canonical_str.parse::<usize>() {
                        edge_updates.push((edge_idx, canonical_id));
                    }
                }
            }
        }

        // Apply edge updates
        for (edge_idx, canonical_id) in edge_updates {
            graph.edges[edge_idx].from = canonical_id;
        }

        let mut stats = HashMap::new();
        stats.insert(
            "eliminated_expressions".to_string(),
            eliminated_expressions as f64,
        );
        stats.insert(
            "unique_expressions".to_string(),
            self.expression_map.len() as f64,
        );

        Ok(PassResult {
            changed,
            stats,
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(
        &self,
        graph: &ComputationGraph,
    ) -> Result<f64, crate::errors::TrustformersError> {
        let mut signatures = HashMap::new();
        let mut duplicates = 0;

        for node in &graph.nodes {
            let signature = self.expression_signature(node, graph);
            if let Some(count) = signatures.get_mut(&signature) {
                *count += 1;
                duplicates += 1;
            } else {
                signatures.insert(signature, 1);
            }
        }

        Ok(duplicates as f64 / graph.nodes.len().max(1) as f64)
    }
}

impl Default for CommonSubexpressionEliminationPass {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory layout optimization pass
pub struct MemoryLayoutOptimizationPass {
    #[allow(dead_code)]
    layout_preferences: HashMap<String, String>,
}

impl MemoryLayoutOptimizationPass {
    pub fn new() -> Self {
        Self {
            layout_preferences: HashMap::new(),
        }
    }
}

impl OptimizationPass for MemoryLayoutOptimizationPass {
    fn name(&self) -> &str {
        "memory-layout-optimization"
    }

    fn description(&self) -> &str {
        "Optimize tensor memory layouts for better cache locality"
    }

    fn apply(
        &mut self,
        graph: &mut ComputationGraph,
    ) -> Result<PassResult, crate::errors::TrustformersError> {
        let mut changed = false;
        let mut optimized_layouts = 0;

        for node in &mut graph.nodes {
            match node.op_type.as_str() {
                "MatMul"
                    // Prefer row-major layout for first input, column-major for second
                    if !node.attributes.contains_key("layout_optimized") => {
                        node.attributes
                            .insert("input0_layout".to_string(), "row_major".to_string());
                        node.attributes
                            .insert("input1_layout".to_string(), "col_major".to_string());
                        node.attributes.insert("layout_optimized".to_string(), "true".to_string());
                        optimized_layouts += 1;
                        changed = true;
                    },
                "Conv2D"
                    // Prefer NCHW layout for convolutions on GPU, NHWC for CPU
                    if !node.attributes.contains_key("layout_optimized") => {
                        node.attributes.insert("data_layout".to_string(), "NCHW".to_string());
                        node.attributes.insert("weight_layout".to_string(), "OIHW".to_string());
                        node.attributes.insert("layout_optimized".to_string(), "true".to_string());
                        optimized_layouts += 1;
                        changed = true;
                    },
                _ => {},
            }
        }

        let mut stats = HashMap::new();
        stats.insert("optimized_layouts".to_string(), optimized_layouts as f64);

        Ok(PassResult {
            changed,
            stats,
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(
        &self,
        graph: &ComputationGraph,
    ) -> Result<f64, crate::errors::TrustformersError> {
        let mut optimizable_ops = 0;

        for node in &graph.nodes {
            match node.op_type.as_str() {
                "MatMul" | "Conv2D" | "Conv3D"
                    if !node.attributes.contains_key("layout_optimized") =>
                {
                    optimizable_ops += 1;
                },
                _ => {},
            }
        }

        // Estimate 10-30% performance improvement for layout optimization
        Ok((optimizable_ops as f64 / graph.nodes.len().max(1) as f64) * 0.2)
    }
}

impl Default for MemoryLayoutOptimizationPass {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation fusion pass (basic)
pub struct OperationFusionPass {
    fusion_patterns: Vec<FusionPattern>,
}

impl OperationFusionPass {
    pub fn new() -> Self {
        Self {
            fusion_patterns: Self::default_fusion_patterns(),
        }
    }

    fn default_fusion_patterns() -> Vec<FusionPattern> {
        vec![
            FusionPattern {
                name: "matmul_bias_add".to_string(),
                pattern: vec!["MatMul".to_string(), "Add".to_string()],
                fused_op: "FusedMatMulBiasAdd".to_string(),
                benefit: 1.2, // 20% improvement
            },
            FusionPattern {
                name: "conv_relu".to_string(),
                pattern: vec!["Conv2D".to_string(), "ReLU".to_string()],
                fused_op: "FusedConvReLU".to_string(),
                benefit: 1.15, // 15% improvement
            },
            FusionPattern {
                name: "linear_relu".to_string(),
                pattern: vec!["Linear".to_string(), "ReLU".to_string()],
                fused_op: "FusedLinearReLU".to_string(),
                benefit: 1.1, // 10% improvement
            },
        ]
    }
}

#[derive(Debug, Clone)]
struct FusionPattern {
    name: String,
    pattern: Vec<String>,
    fused_op: String,
    benefit: f64,
}

impl OptimizationPass for OperationFusionPass {
    fn name(&self) -> &str {
        "operation-fusion"
    }

    fn description(&self) -> &str {
        "Fuse compatible operations to reduce memory bandwidth and improve performance"
    }

    fn apply(
        &mut self,
        graph: &mut ComputationGraph,
    ) -> Result<PassResult, crate::errors::TrustformersError> {
        let mut changed = false;
        let mut fused_operations = 0;

        // Look for fusion opportunities
        let mut fusion_candidates = Vec::new();
        for pattern in &self.fusion_patterns {
            if pattern.pattern.len() != 2 {
                continue; // Only handle binary patterns for now
            }

            for edge in &graph.edges {
                let from_node = graph.get_node(edge.from);
                let to_node = graph.get_node(edge.to);

                if let (Some(from), Some(to)) = (from_node, to_node) {
                    if from.op_type == pattern.pattern[0] && to.op_type == pattern.pattern[1] {
                        fusion_candidates.push((edge.from, edge.to, pattern.clone()));
                        fused_operations += 1;
                        changed = true;
                    }
                }
            }
        }

        // Apply fusion to collected candidates
        for (from_id, to_id, pattern) in fusion_candidates {
            if let Some(to_node_mut) = graph.get_node_mut(to_id) {
                to_node_mut.op_type = pattern.fused_op.clone();
                to_node_mut.attributes.insert("fused_pattern".to_string(), pattern.name.clone());
                to_node_mut.attributes.insert("fused_from".to_string(), from_id.to_string());
            }

            if let Some(from_node_mut) = graph.get_node_mut(from_id) {
                from_node_mut.attributes.insert("fused_into".to_string(), to_id.to_string());
            }
        }

        let mut stats = HashMap::new();
        stats.insert("fused_operations".to_string(), fused_operations as f64);

        Ok(PassResult {
            changed,
            stats,
            metadata: HashMap::new(),
        })
    }

    fn estimate_benefit(
        &self,
        graph: &ComputationGraph,
    ) -> Result<f64, crate::errors::TrustformersError> {
        let mut _potential_fusions = 0;
        let mut total_benefit = 0.0;

        for pattern in &self.fusion_patterns {
            if pattern.pattern.len() != 2 {
                continue;
            }

            for edge in &graph.edges {
                if let (Some(from), Some(to)) = (graph.get_node(edge.from), graph.get_node(edge.to))
                {
                    if from.op_type == pattern.pattern[0] && to.op_type == pattern.pattern[1] {
                        _potential_fusions += 1;
                        total_benefit += pattern.benefit - 1.0; // Convert to improvement ratio
                    }
                }
            }
        }

        Ok(total_benefit / graph.nodes.len().max(1) as f64)
    }
}

impl Default for OperationFusionPass {
    fn default() -> Self {
        Self::new()
    }
}

/// Pass manager for orchestrating multiple optimization passes
pub struct PassManager {
    passes: Vec<Box<dyn OptimizationPass>>,
    max_iterations: usize,
}

impl PassManager {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            max_iterations: 10,
        }
    }

    /// Add a pass to the manager
    pub fn add_pass<P: OptimizationPass + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }

    /// Set maximum number of iterations
    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations;
    }

    /// Run all passes until convergence or max iterations
    pub fn run(
        &mut self,
        graph: &mut ComputationGraph,
    ) -> Result<Vec<PassResult>, crate::errors::TrustformersError> {
        let mut all_results = Vec::new();

        for iteration in 0..self.max_iterations {
            let mut changed = false;
            let mut iteration_results = Vec::new();

            for pass in &mut self.passes {
                let result = pass.apply(graph)?;
                changed |= result.changed;
                iteration_results.push(result);
            }

            all_results.extend(iteration_results);

            if !changed {
                break; // Converged
            }
        }

        Ok(all_results)
    }

    /// Get a default pass pipeline for standard optimization
    pub fn default_pipeline() -> Self {
        let mut manager = Self::new();
        manager.add_pass(DeadCodeEliminationPass::new());
        manager.add_pass(ConstantFoldingPass::new());
        manager.add_pass(CommonSubexpressionEliminationPass::new());
        manager.add_pass(OperationFusionPass::new());
        manager.add_pass(MemoryLayoutOptimizationPass::new());
        manager
    }

    /// Get an aggressive pass pipeline for maximum optimization
    pub fn aggressive_pipeline() -> Self {
        let mut manager = Self::default_pipeline();
        manager.set_max_iterations(20);
        manager
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{ComputationGraph, GraphEdge, GraphNode};

    fn create_test_graph() -> ComputationGraph {
        let mut graph = ComputationGraph::new();

        let node1 = GraphNode {
            id: 0,
            op_type: "MatMul".to_string(),
            attributes: HashMap::new(),
            input_shapes: vec![vec![128, 256], vec![256, 512]],
            output_shapes: vec![vec![128, 512]],
            compute_cost: 100.0,
            memory_cost: 50.0,
        };

        let node2 = GraphNode {
            id: 1,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            input_shapes: vec![vec![128, 512], vec![128, 512]],
            output_shapes: vec![vec![128, 512]],
            compute_cost: 10.0,
            memory_cost: 5.0,
        };

        graph.add_node(node1);
        graph.add_node(node2);

        let edge = GraphEdge {
            from: 0,
            to: 1,
            output_idx: 0,
            input_idx: 0,
            shape: vec![128, 512],
            dtype: "f32".to_string(),
        };

        graph.add_edge(edge);
        graph
    }

    #[test]
    fn test_constant_folding_pass() {
        let mut graph = create_test_graph();
        let mut pass = ConstantFoldingPass::new();

        let result = pass.apply(&mut graph);
        assert!(result.is_ok());

        let benefit = pass.estimate_benefit(&graph);
        assert!(benefit.is_ok());
    }

    /// Regression test: `estimate_benefit` counted every `Add`/`Mul`/`Sub`/
    /// `Div` node regardless of whether its inputs were actually constant,
    /// then multiplied by an unexplained `0.1`, disagreeing with `apply`'s
    /// real constant-propagation logic right above it. It now shares that
    /// propagation (`find_constant_nodes`) and weights by real
    /// `compute_cost`, matching its own "percentage of compute cost that
    /// could be eliminated" doc comment instead of contradicting it. Also
    /// covers the `removed_nodes` stat, which was hardcoded to `0` (`apply`
    /// never removes a node) and has been dropped rather than kept
    /// permanently zero.
    #[test]
    fn test_constant_folding_estimate_benefit_is_input_dependent_and_cost_weighted() {
        let pass = ConstantFoldingPass::new();

        // MatMul -> Add, no constants anywhere: nothing is foldable.
        let no_constants = create_test_graph();
        let no_fold_benefit = pass.estimate_benefit(&no_constants).expect("estimate failed");
        assert_eq!(no_fold_benefit, 0.0);

        // Constant(cost 1) -> Add(cost 9) -> MatMul(cost 90): the Add is
        // genuinely foldable (its only input traces to a literal constant);
        // the MatMul is not (wrong op type for this pass), even though its
        // own input is now known constant.
        let mut with_constant = ComputationGraph::new();
        with_constant.add_node(GraphNode {
            id: 0,
            op_type: "Constant".to_string(),
            attributes: HashMap::new(),
            input_shapes: vec![],
            output_shapes: vec![vec![1]],
            compute_cost: 1.0,
            memory_cost: 1.0,
        });
        with_constant.add_node(GraphNode {
            id: 1,
            op_type: "Add".to_string(),
            attributes: HashMap::new(),
            input_shapes: vec![vec![1]],
            output_shapes: vec![vec![1]],
            compute_cost: 9.0,
            memory_cost: 1.0,
        });
        with_constant.add_node(GraphNode {
            id: 2,
            op_type: "MatMul".to_string(),
            attributes: HashMap::new(),
            input_shapes: vec![vec![1]],
            output_shapes: vec![vec![1]],
            compute_cost: 90.0,
            memory_cost: 1.0,
        });
        with_constant.add_edge(GraphEdge {
            from: 0,
            to: 1,
            output_idx: 0,
            input_idx: 0,
            shape: vec![1],
            dtype: "f32".to_string(),
        });
        with_constant.add_edge(GraphEdge {
            from: 1,
            to: 2,
            output_idx: 0,
            input_idx: 0,
            shape: vec![1],
            dtype: "f32".to_string(),
        });

        let fold_benefit = pass.estimate_benefit(&with_constant).expect("estimate failed");
        assert_ne!(
            no_fold_benefit, fold_benefit,
            "estimate must vary with graph content"
        );
        assert!(
            (fold_benefit - 0.09).abs() < 1e-9,
            "expected the foldable Add's 9.0 compute cost / 100.0 total compute cost, got {fold_benefit}"
        );

        // `apply` must report exactly the node(s) the estimate credited, and
        // must not publish a fabricated always-zero `removed_nodes`.
        let mut pass = ConstantFoldingPass::new();
        let mut graph_to_mutate = with_constant.clone();
        let result = pass.apply(&mut graph_to_mutate).expect("apply failed");
        assert_eq!(result.stats["folded_operations"], 1.0);
        assert_eq!(result.stats["constant_nodes_total"], 2.0);
        assert!(
            !result.stats.contains_key("removed_nodes"),
            "removed_nodes always reported 0 regardless of what happened, because apply never \
             removes a node; publishing it was a fabricated claim, so it must be gone rather \
             than kept permanently zero"
        );
    }

    #[test]
    fn test_dead_code_elimination_pass() {
        let mut graph = create_test_graph();
        let mut pass = DeadCodeEliminationPass::new();

        let result = pass.apply(&mut graph);
        assert!(result.is_ok());

        let benefit = pass.estimate_benefit(&graph);
        assert!(benefit.is_ok());
    }

    #[test]
    fn test_operation_fusion_pass() {
        let mut graph = create_test_graph();
        let mut pass = OperationFusionPass::new();

        let result = pass.apply(&mut graph);
        assert!(result.is_ok());

        let benefit = pass.estimate_benefit(&graph);
        assert!(benefit.is_ok());
    }

    #[test]
    fn test_pass_manager() {
        let mut graph = create_test_graph();
        let mut manager = PassManager::default_pipeline();

        let results = manager.run(&mut graph);
        assert!(results.is_ok());
    }

    #[test]
    fn test_memory_layout_optimization() {
        let mut graph = create_test_graph();
        let mut pass = MemoryLayoutOptimizationPass::new();

        let result = pass.apply(&mut graph);
        assert!(result.is_ok());

        let benefit = pass.estimate_benefit(&graph);
        assert!(benefit.is_ok());
    }
}
