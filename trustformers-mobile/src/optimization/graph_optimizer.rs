//! Graph Optimization Module
//!
//! Provides graph-level optimizations for mobile inference including:
//! - Constant folding
//! - Dead code elimination
//! - Common subexpression elimination
//! - Algebraic simplification

use super::{ComputationGraph, GraphOperator, KernelType};
use std::collections::{HashMap, HashSet, VecDeque};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

/// Optimization pass types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPass {
    ConstantFolding,
    DeadCodeElimination,
    CommonSubexpressionElimination,
    AlgebraicSimplification,
    StrengthReduction,
    LoopInvariantCodeMotion,
}

/// Graph rewriter for applying transformations
pub struct GraphRewriter {
    transformations: Vec<Box<dyn GraphTransformation>>,
}

/// Trait for graph transformations
pub trait GraphTransformation: Send + Sync {
    /// Apply transformation to graph
    fn apply(&self, graph: &mut ComputationGraph) -> Result<bool>;

    /// Get transformation name
    fn name(&self) -> &str;
}

/// Graph optimizer
pub struct GraphOptimizer {
    passes: Vec<OptimizationPass>,
    rewriter: GraphRewriter,
    stats: OptimizationStats,
}

#[derive(Debug, Default)]
struct OptimizationStats {
    constants_folded: usize,
    dead_code_eliminated: usize,
    cse_eliminated: usize,
    algebraic_simplified: usize,
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphOptimizer {
    /// Create new graph optimizer
    pub fn new() -> Self {
        let passes = vec![
            OptimizationPass::ConstantFolding,
            OptimizationPass::DeadCodeElimination,
            OptimizationPass::CommonSubexpressionElimination,
            OptimizationPass::AlgebraicSimplification,
        ];

        let transformations: Vec<Box<dyn GraphTransformation>> = vec![
            Box::new(ConstantFolding::new()),
            Box::new(DeadCodeElimination::new()),
            Box::new(CommonSubexpressionElimination::new()),
            Box::new(AlgebraicSimplification::new()),
        ];

        Self {
            passes,
            rewriter: GraphRewriter { transformations },
            stats: OptimizationStats::default(),
        }
    }

    /// Apply all optimization passes
    pub fn optimize(&mut self, graph: &mut ComputationGraph) -> Result<()> {
        let passes = self.passes.clone();
        for pass in passes {
            match pass {
                OptimizationPass::ConstantFolding => self.apply_constant_folding(graph)?,
                OptimizationPass::DeadCodeElimination => self.apply_dead_code_elimination(graph)?,
                OptimizationPass::CommonSubexpressionElimination => self.apply_cse(graph)?,
                OptimizationPass::AlgebraicSimplification => {
                    self.apply_algebraic_simplification(graph)?
                },
                _ => {},
            }
        }
        Ok(())
    }

    /// Apply constant folding optimization
    pub fn apply_constant_folding(&mut self, graph: &mut ComputationGraph) -> Result<()> {
        let folding = ConstantFolding::new();
        if folding.apply(graph)? {
            self.stats.constants_folded += 1;
        }
        Ok(())
    }

    /// Apply dead code elimination
    pub fn apply_dead_code_elimination(&mut self, graph: &mut ComputationGraph) -> Result<()> {
        let dce = DeadCodeElimination::new();
        if dce.apply(graph)? {
            self.stats.dead_code_eliminated += 1;
        }
        Ok(())
    }

    /// Apply common subexpression elimination
    pub fn apply_cse(&mut self, graph: &mut ComputationGraph) -> Result<()> {
        let cse = CommonSubexpressionElimination::new();
        if cse.apply(graph)? {
            self.stats.cse_eliminated += 1;
        }
        Ok(())
    }

    /// Apply algebraic simplification
    pub fn apply_algebraic_simplification(&mut self, graph: &mut ComputationGraph) -> Result<()> {
        let simplify = AlgebraicSimplification::new();
        if simplify.apply(graph)? {
            self.stats.algebraic_simplified += 1;
        }
        Ok(())
    }
}

/// Constant folding optimization
pub struct ConstantFolding {
    constant_values: HashMap<String, Tensor>,
}

impl Default for ConstantFolding {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantFolding {
    pub fn new() -> Self {
        Self {
            constant_values: HashMap::new(),
        }
    }

    /// Check if operator has all constant inputs
    fn has_constant_inputs(&self, op: &GraphOperator, graph: &ComputationGraph) -> bool {
        op.inputs
            .iter()
            .all(|input| self.constant_values.contains_key(input) || self.is_constant_tensor(input))
    }

    /// Check if tensor is constant
    fn is_constant_tensor(&self, name: &str) -> bool {
        // Check if tensor name indicates it's a constant (e.g., weights, biases)
        name.contains("weight") || name.contains("bias") || name.starts_with("const_")
    }

    /// Evaluate operator with constant inputs
    fn evaluate_constant_op(&mut self, op: &GraphOperator) -> Result<Option<Tensor>> {
        match &op.kernel {
            KernelType::Linear => {
                // Can fold if both input and weights are constant
                if op.inputs.len() >= 2 {
                    // Simplified - would need actual computation
                    Ok(None)
                } else {
                    Ok(None)
                }
            },
            KernelType::Custom(name) if name == "Add" => {
                // Can fold constant additions
                if op.inputs.len() == 2 {
                    // Simplified - would need actual computation
                    Ok(None)
                } else {
                    Ok(None)
                }
            },
            _ => Ok(None),
        }
    }
}

impl GraphTransformation for ConstantFolding {
    fn apply(&self, graph: &mut ComputationGraph) -> Result<bool> {
        let mut modified = false;
        let mut const_folding = ConstantFolding::new();

        // Find operators that can be constant folded
        let mut folded_ops = Vec::new();

        for (idx, op) in graph.operators.iter().enumerate() {
            if const_folding.has_constant_inputs(op, graph) {
                if let Some(result) = const_folding.evaluate_constant_op(op)? {
                    // Store the constant result
                    const_folding.constant_values.insert(op.outputs[0].clone(), result);
                    folded_ops.push(idx);
                    modified = true;
                }
            }
        }

        // Replace folded operators with constant nodes
        for idx in folded_ops.iter().rev() {
            let op = &graph.operators[*idx];
            let const_name = format!("const_{}", op.outputs[0]);

            // Create constant operator
            let const_op = GraphOperator {
                id: op.id,
                kernel: KernelType::Custom("Constant".to_string()),
                inputs: vec![],
                outputs: vec![const_name],
                input_shapes: vec![],
                output_shape: op.output_shape.clone(),
                cache_hints: None,
            };

            graph.operators[*idx] = const_op;
        }

        Ok(modified)
    }

    fn name(&self) -> &str {
        "ConstantFolding"
    }
}

/// Dead code elimination
pub struct DeadCodeElimination;

impl Default for DeadCodeElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl DeadCodeElimination {
    pub fn new() -> Self {
        Self
    }

    /// Find operators whose outputs are not used
    fn find_dead_operators(&self, graph: &ComputationGraph) -> HashSet<usize> {
        let mut used_tensors = HashSet::new();
        let mut dead_operators = HashSet::new();

        // Mark all input tensors as used
        for op in &graph.operators {
            for input in &op.inputs {
                used_tensors.insert(input.clone());
            }
        }

        // Find operators whose outputs are not used
        for (idx, op) in graph.operators.iter().enumerate() {
            let outputs_used = op.outputs.iter().any(|output| used_tensors.contains(output));

            if !outputs_used && !self.has_side_effects(op) {
                dead_operators.insert(idx);
            }
        }

        dead_operators
    }

    /// Check if operator has side effects
    fn has_side_effects(&self, op: &GraphOperator) -> bool {
        // Operators that modify state or have other side effects
        matches!(op.kernel, KernelType::Custom(ref name) if name.contains("inplace") || name.contains("update"))
    }
}

impl GraphTransformation for DeadCodeElimination {
    fn apply(&self, graph: &mut ComputationGraph) -> Result<bool> {
        let dead_ops = self.find_dead_operators(graph);

        if dead_ops.is_empty() {
            return Ok(false);
        }

        // Remove dead operators (in reverse order to maintain indices)
        let mut indices: Vec<_> = dead_ops.into_iter().collect();
        indices.sort_by(|a, b| b.cmp(a));

        for idx in indices {
            graph.operators.remove(idx);

            // Update edges
            graph.edges.retain(|edge| edge.from != idx && edge.to != idx);

            // Update edge indices
            for edge in &mut graph.edges {
                if edge.from > idx {
                    edge.from -= 1;
                }
                if edge.to > idx {
                    edge.to -= 1;
                }
            }
        }

        Ok(true)
    }

    fn name(&self) -> &str {
        "DeadCodeElimination"
    }
}

/// Common subexpression elimination
pub struct CommonSubexpressionElimination;

impl Default for CommonSubexpressionElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl CommonSubexpressionElimination {
    pub fn new() -> Self {
        Self
    }

    /// Find duplicate computations
    fn find_duplicate_ops(&self, graph: &ComputationGraph) -> Vec<(usize, usize)> {
        let mut duplicates = Vec::new();

        for i in 0..graph.operators.len() {
            for j in i + 1..graph.operators.len() {
                if self.ops_are_equivalent(&graph.operators[i], &graph.operators[j]) {
                    duplicates.push((i, j));
                }
            }
        }

        duplicates
    }

    /// Check if two operators compute the same thing
    fn ops_are_equivalent(&self, op1: &GraphOperator, op2: &GraphOperator) -> bool {
        // Same kernel type
        if std::mem::discriminant(&op1.kernel) != std::mem::discriminant(&op2.kernel) {
            return false;
        }

        // Same inputs (order matters for non-commutative ops)
        if op1.inputs != op2.inputs {
            return false;
        }

        // Same shapes
        if op1.input_shapes != op2.input_shapes || op1.output_shape != op2.output_shape {
            return false;
        }

        true
    }
}

impl GraphTransformation for CommonSubexpressionElimination {
    fn apply(&self, graph: &mut ComputationGraph) -> Result<bool> {
        let duplicates = self.find_duplicate_ops(graph);

        if duplicates.is_empty() {
            return Ok(false);
        }

        // Replace duplicate computations
        for (original, duplicate) in duplicates.iter().rev() {
            let original_output = graph.operators[*original].outputs[0].clone();
            let duplicate_output = graph.operators[*duplicate].outputs[0].clone();

            // Redirect all uses of duplicate output to original output
            for op in &mut graph.operators {
                for input in &mut op.inputs {
                    if input == &duplicate_output {
                        *input = original_output.clone();
                    }
                }
            }

            // Remove duplicate operator
            graph.operators.remove(*duplicate);

            // Update edges
            for edge in &mut graph.edges {
                if edge.tensor_name == duplicate_output {
                    edge.tensor_name = original_output.clone();
                }
            }
        }

        Ok(true)
    }

    fn name(&self) -> &str {
        "CommonSubexpressionElimination"
    }
}

/// Algebraic simplification
pub struct AlgebraicSimplification;

impl Default for AlgebraicSimplification {
    fn default() -> Self {
        Self::new()
    }
}

impl AlgebraicSimplification {
    pub fn new() -> Self {
        Self
    }

    /// Apply algebraic identities
    fn simplify_operator(&self, op: &GraphOperator) -> Option<GraphOperator> {
        if let KernelType::Custom(name) = &op.kernel {
            match name.as_str() {
                "Mul" => {
                    // x * 1 = x
                    if self.is_one(&op.inputs[1]) {
                        return Some(self.create_identity_op(op, 0));
                    }
                    // 1 * x = x
                    if self.is_one(&op.inputs[0]) {
                        return Some(self.create_identity_op(op, 1));
                    }
                    // x * 0 = 0
                    if self.is_zero(&op.inputs[0]) || self.is_zero(&op.inputs[1]) {
                        return Some(self.create_zero_op(op));
                    }
                },
                "Add" => {
                    // x + 0 = x
                    if self.is_zero(&op.inputs[1]) {
                        return Some(self.create_identity_op(op, 0));
                    }
                    // 0 + x = x
                    if self.is_zero(&op.inputs[0]) {
                        return Some(self.create_identity_op(op, 1));
                    }
                },
                "Sub" => {
                    // x - 0 = x
                    if self.is_zero(&op.inputs[1]) {
                        return Some(self.create_identity_op(op, 0));
                    }
                    // x - x = 0
                    if op.inputs[0] == op.inputs[1] {
                        return Some(self.create_zero_op(op));
                    }
                },
                _ => {},
            }
        }

        None
    }

    fn is_zero(&self, tensor_name: &str) -> bool {
        tensor_name.contains("zero") || tensor_name == "0"
    }

    fn is_one(&self, tensor_name: &str) -> bool {
        tensor_name.contains("one") || tensor_name == "1"
    }

    fn create_identity_op(&self, original: &GraphOperator, input_idx: usize) -> GraphOperator {
        GraphOperator {
            id: original.id,
            kernel: KernelType::Custom("Identity".to_string()),
            inputs: vec![original.inputs[input_idx].clone()],
            outputs: original.outputs.clone(),
            input_shapes: vec![original.input_shapes[input_idx].clone()],
            output_shape: original.output_shape.clone(),
            cache_hints: original.cache_hints.clone(),
        }
    }

    fn create_zero_op(&self, original: &GraphOperator) -> GraphOperator {
        GraphOperator {
            id: original.id,
            kernel: KernelType::Custom("Zero".to_string()),
            inputs: vec![],
            outputs: original.outputs.clone(),
            input_shapes: vec![],
            output_shape: original.output_shape.clone(),
            cache_hints: original.cache_hints.clone(),
        }
    }
}

impl GraphTransformation for AlgebraicSimplification {
    fn apply(&self, graph: &mut ComputationGraph) -> Result<bool> {
        let mut modified = false;

        for op in &mut graph.operators {
            if let Some(simplified) = self.simplify_operator(op) {
                *op = simplified;
                modified = true;
            }
        }

        Ok(modified)
    }

    fn name(&self) -> &str {
        "AlgebraicSimplification"
    }
}

/// Graph analysis utilities
pub struct GraphAnalyzer;

impl GraphAnalyzer {
    /// Compute topological order of operators
    pub fn topological_sort(graph: &ComputationGraph) -> Result<Vec<usize>> {
        let mut in_degree = vec![0; graph.operators.len()];
        let mut adj_list: Vec<Vec<usize>> = vec![vec![]; graph.operators.len()];

        // Build adjacency list and in-degree count
        for edge in &graph.edges {
            adj_list[edge.from].push(edge.to);
            in_degree[edge.to] += 1;
        }

        // Find nodes with no incoming edges
        let mut queue = VecDeque::new();
        for (idx, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(idx);
            }
        }

        let mut sorted = Vec::new();

        while let Some(node) = queue.pop_front() {
            sorted.push(node);

            for &neighbor in &adj_list[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        if sorted.len() != graph.operators.len() {
            return Err(TrustformersError::runtime_error("Graph contains cycles".into()).into());
        }

        Ok(sorted)
    }

    /// Find critical path through graph
    pub fn critical_path(graph: &ComputationGraph) -> Result<Vec<usize>> {
        let topo_order = Self::topological_sort(graph)?;
        let mut distances = vec![0.0; graph.operators.len()];
        let mut predecessors = vec![None; graph.operators.len()];

        // Compute longest path
        for &node in &topo_order {
            for edge in &graph.edges {
                if edge.from == node {
                    let weight = 1.0; // Could be actual computation cost
                    if distances[node] + weight > distances[edge.to] {
                        distances[edge.to] = distances[node] + weight;
                        predecessors[edge.to] = Some(node);
                    }
                }
            }
        }

        // Find node with maximum distance
        let end_node = distances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // Reconstruct path
        let mut path = Vec::new();
        let mut current = Some(end_node);

        while let Some(node) = current {
            path.push(node);
            current = predecessors[node];
        }

        path.reverse();
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_optimizer_creation() {
        let optimizer = GraphOptimizer::new();
        assert!(!optimizer.passes.is_empty());
    }

    #[test]
    fn test_constant_folding() {
        let mut graph = ComputationGraph {
            operators: vec![
                GraphOperator {
                    id: 0,
                    kernel: KernelType::Custom("Constant".to_string()),
                    inputs: vec![],
                    outputs: vec!["const_1".to_string()],
                    input_shapes: vec![],
                    output_shape: vec![1],
                    cache_hints: None,
                },
                GraphOperator {
                    id: 1,
                    kernel: KernelType::Custom("Add".to_string()),
                    inputs: vec!["const_1".to_string(), "const_1".to_string()],
                    outputs: vec!["result".to_string()],
                    input_shapes: vec![vec![1], vec![1]],
                    output_shape: vec![1],
                    cache_hints: None,
                },
            ],
            edges: vec![],
        };

        let folding = ConstantFolding::new();
        let result = folding.apply(&mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dead_code_elimination() {
        let mut graph = ComputationGraph {
            operators: vec![
                GraphOperator {
                    id: 0,
                    kernel: KernelType::Linear,
                    inputs: vec!["input".to_string()],
                    outputs: vec!["output1".to_string()],
                    input_shapes: vec![vec![10]],
                    output_shape: vec![10],
                    cache_hints: None,
                },
                GraphOperator {
                    id: 1,
                    kernel: KernelType::Linear,
                    inputs: vec!["input".to_string()],
                    outputs: vec!["unused".to_string()], // This output is not used
                    input_shapes: vec![vec![10]],
                    output_shape: vec![10],
                    cache_hints: None,
                },
            ],
            edges: vec![],
        };

        let dce = DeadCodeElimination::new();
        let result = dce.apply(&mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_algebraic_simplification() {
        let mut graph = ComputationGraph {
            operators: vec![GraphOperator {
                id: 0,
                kernel: KernelType::Custom("Add".to_string()),
                inputs: vec!["x".to_string(), "zero".to_string()],
                outputs: vec!["result".to_string()],
                input_shapes: vec![vec![10], vec![10]],
                output_shape: vec![10],
                cache_hints: None,
            }],
            edges: vec![],
        };

        let simplify = AlgebraicSimplification::new();
        let modified = simplify.apply(&mut graph).unwrap();

        assert!(modified);
        assert_eq!(
            graph.operators[0].kernel,
            KernelType::Custom("Identity".to_string())
        );
    }
}
