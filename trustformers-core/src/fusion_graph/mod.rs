//! Layer fusion graph — identify and apply fusion opportunities.
//!
//! This module provides a lightweight computation graph that records transformer
//! layer operations and detects adjacent operation patterns that can be replaced
//! by fused kernels (see [`crate::fused`]).
//!
//! # Typical workflow
//!
//! 1. Build the graph by calling [`FusionGraph::add_node`] for every layer.
//! 2. Call [`FusionGraph::analyze_fusions`] with one of the [`LayerFusionPattern`]s
//!    returned by [`LayerFusionPattern::standard_patterns`].
//! 3. Optionally commit the detected groups via [`FusionGraph::apply_fusions`].
//! 4. Inspect the result with [`FusionGraph::flop_analysis`] or
//!    [`FusionGraph::to_dot`].

use std::fmt;

// ── Node type ─────────────────────────────────────────────────────────────────

/// The type of operation represented by a graph node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeType {
    LayerNorm,
    RmsNorm,
    Linear,
    Attention,
    SwiGLU,
    GeGLU,
    ReLU,
    SiLU,
    GELU,
    Residual,
    Embedding,
    Dropout,
    Softmax,
    Custom(String),
}

impl fmt::Display for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeType::LayerNorm => write!(f, "LayerNorm"),
            NodeType::RmsNorm => write!(f, "RmsNorm"),
            NodeType::Linear => write!(f, "Linear"),
            NodeType::Attention => write!(f, "Attention"),
            NodeType::SwiGLU => write!(f, "SwiGLU"),
            NodeType::GeGLU => write!(f, "GeGLU"),
            NodeType::ReLU => write!(f, "ReLU"),
            NodeType::SiLU => write!(f, "SiLU"),
            NodeType::GELU => write!(f, "GELU"),
            NodeType::Residual => write!(f, "Residual"),
            NodeType::Embedding => write!(f, "Embedding"),
            NodeType::Dropout => write!(f, "Dropout"),
            NodeType::Softmax => write!(f, "Softmax"),
            NodeType::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

// ── Fused op type ─────────────────────────────────────────────────────────────

/// Identifies which fused kernel a [`FusedGroup`] maps to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusedOpType {
    LayerNormLinear,
    RmsNormLinear,
    AttentionScores,
    SwiGLUFused,
    GeGLUFused,
    ResidualAddNorm,
    MultiOpFusion(Vec<String>),
}

impl fmt::Display for FusedOpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusedOpType::LayerNormLinear => write!(f, "LayerNormLinear"),
            FusedOpType::RmsNormLinear => write!(f, "RmsNormLinear"),
            FusedOpType::AttentionScores => write!(f, "AttentionScores"),
            FusedOpType::SwiGLUFused => write!(f, "SwiGLUFused"),
            FusedOpType::GeGLUFused => write!(f, "GeGLUFused"),
            FusedOpType::ResidualAddNorm => write!(f, "ResidualAddNorm"),
            FusedOpType::MultiOpFusion(ops) => write!(f, "MultiOpFusion({})", ops.join("+")),
        }
    }
}

// ── Graph node ────────────────────────────────────────────────────────────────

/// A single node in the fusion computation graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique, monotonically-increasing identifier.
    pub id: usize,
    /// Human-readable name (e.g. `"self_attn_q_proj"`).
    pub name: String,
    /// Operation category.
    pub node_type: NodeType,
    /// IDs of upstream (input) nodes.
    pub inputs: Vec<usize>,
    /// Output shape hint (may be empty if unknown).
    pub output_shape: Vec<usize>,
    /// Whether this node has been absorbed into a fused group.
    pub fused: bool,
    /// Estimated floating-point operations for this node.
    pub flops: u64,
}

impl GraphNode {
    /// Create a new node with default values for optional fields.
    pub fn new(id: usize, name: &str, node_type: NodeType, inputs: Vec<usize>) -> Self {
        Self {
            id,
            name: name.to_string(),
            node_type,
            inputs,
            output_shape: Vec::new(),
            fused: false,
            flops: 0,
        }
    }

    /// Builder-style setter for estimated FLOPs.
    pub fn with_flops(mut self, flops: u64) -> Self {
        self.flops = flops;
        self
    }

    /// Builder-style setter for output shape hint.
    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shape = shape;
        self
    }
}

// ── Fused group ───────────────────────────────────────────────────────────────

/// A detected group of fusable nodes.
#[derive(Debug, Clone)]
pub struct FusedGroup {
    /// Unique group identifier.
    pub id: usize,
    /// Pattern name that matched (e.g. `"LayerNorm+Linear"`).
    pub name: String,
    /// IDs of all nodes in this group (in dependency order).
    pub node_ids: Vec<usize>,
    /// The fused kernel that replaces this group.
    pub fused_op: FusedOpType,
    /// Estimated FLOPs saved relative to executing each node separately.
    pub flops_saved: u64,
}

// ── Fusion pattern ────────────────────────────────────────────────────────────

/// A rule that describes a fusable sequence of node types.
pub struct LayerFusionPattern {
    /// Sequence of node types that must form a chain.
    pub pattern: Vec<NodeType>,
    /// The fused kernel to emit when this pattern is matched.
    pub fused_op: FusedOpType,
    /// Descriptive name shown in reports.
    pub name: String,
    /// Rough minimum expected speedup (informational only).
    pub min_speedup: f32,
}

impl LayerFusionPattern {
    /// Returns the built-in fusion patterns for standard transformer blocks.
    pub fn standard_patterns() -> Vec<LayerFusionPattern> {
        vec![
            LayerFusionPattern {
                pattern: vec![NodeType::LayerNorm, NodeType::Linear],
                fused_op: FusedOpType::LayerNormLinear,
                name: "LayerNorm+Linear".to_string(),
                min_speedup: 1.3,
            },
            LayerFusionPattern {
                pattern: vec![NodeType::RmsNorm, NodeType::Linear],
                fused_op: FusedOpType::RmsNormLinear,
                name: "RMSNorm+Linear".to_string(),
                min_speedup: 1.3,
            },
            LayerFusionPattern {
                pattern: vec![NodeType::Residual, NodeType::RmsNorm],
                fused_op: FusedOpType::ResidualAddNorm,
                name: "Residual+RMSNorm".to_string(),
                min_speedup: 1.2,
            },
            LayerFusionPattern {
                pattern: vec![NodeType::SwiGLU, NodeType::Linear],
                fused_op: FusedOpType::SwiGLUFused,
                name: "SwiGLU+Down".to_string(),
                min_speedup: 1.15,
            },
        ]
    }
}

// ── FLOPs analysis ────────────────────────────────────────────────────────────

/// Summary of estimated compute costs for the graph.
#[derive(Debug, Clone)]
pub struct FlopAnalysis {
    /// Total FLOPs summed over all non-fused nodes.
    pub total_flops: u64,
    /// FLOPs that would be saved by the applied fusions.
    pub fused_flops_saved: u64,
    /// Number of fused groups that have been applied.
    pub num_fused_groups: usize,
    /// Rough speedup estimate: `total / (total - saved)`.
    pub speedup_estimate: f32,
}

// ── Fusion graph ──────────────────────────────────────────────────────────────

/// Computation graph with layer fusion analysis.
///
/// # Example
///
/// ```rust
/// use trustformers_core::fusion_graph::{FusionGraph, NodeType, LayerFusionPattern};
///
/// let mut g = FusionGraph::new();
/// let ln = g.add_node("ln", NodeType::RmsNorm, vec![]);
/// let proj = g.add_node("q_proj", NodeType::Linear, vec![ln]);
///
/// let patterns = LayerFusionPattern::standard_patterns();
/// let groups = g.analyze_fusions(&patterns);
/// assert!(!groups.is_empty());
/// ```
pub struct FusionGraph {
    nodes: Vec<GraphNode>,
    fused_groups: Vec<FusedGroup>,
    next_id: usize,
}

impl Default for FusionGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionGraph {
    /// Create an empty fusion graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            fused_groups: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a node to the graph and return its assigned ID.
    ///
    /// IDs are assigned monotonically from 0; the first node gets ID 0.
    pub fn add_node(
        &mut self,
        name: &str,
        node_type: NodeType,
        inputs: Vec<usize>,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(GraphNode::new(id, name, node_type, inputs));
        id
    }

    /// Add a node that already has FLOPs/shape metadata set.
    pub fn add_node_with_metadata(&mut self, node: GraphNode) -> usize {
        let id = node.id;
        self.nodes.push(node);
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        id
    }

    /// Return an immutable reference to all nodes.
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    // ── Pattern matching ──────────────────────────────────────────────────

    /// Check whether the nodes starting at `start_idx` (by sequential position
    /// in the `nodes` vector) match `pattern` and form a strict input chain.
    fn matches_pattern_at(&self, start_idx: usize, pattern: &[NodeType]) -> bool {
        if start_idx + pattern.len() > self.nodes.len() {
            return false;
        }
        for (i, expected_type) in pattern.iter().enumerate() {
            let node = &self.nodes[start_idx + i];
            if &node.node_type != expected_type {
                return false;
            }
            if i > 0 {
                // The previous node in the pattern must be a direct input to this node
                let prev_id = self.nodes[start_idx + i - 1].id;
                if !node.inputs.contains(&prev_id) {
                    return false;
                }
            }
        }
        true
    }

    // ── Fusion analysis ───────────────────────────────────────────────────

    /// Scan the graph for all occurrences of each pattern and return the
    /// detected [`FusedGroup`]s (without modifying the graph).
    pub fn analyze_fusions(&self, patterns: &[LayerFusionPattern]) -> Vec<FusedGroup> {
        let mut groups: Vec<FusedGroup> = Vec::new();
        let mut next_group_id: usize = 0;

        for (i, _node) in self.nodes.iter().enumerate() {
            for pattern in patterns {
                if self.matches_pattern_at(i, &pattern.pattern) {
                    let node_ids: Vec<usize> =
                        (i..i + pattern.pattern.len()).map(|idx| self.nodes[idx].id).collect();

                    let flops_saved = node_ids
                        .iter()
                        .filter_map(|&nid| self.nodes.get(nid))
                        .map(|n| n.flops)
                        .sum::<u64>()
                        / 10; // conservative 10 % savings estimate

                    groups.push(FusedGroup {
                        id: next_group_id,
                        name: pattern.name.clone(),
                        node_ids,
                        fused_op: pattern.fused_op.clone(),
                        flops_saved,
                    });
                    next_group_id += 1;
                }
            }
        }
        groups
    }

    // ── Apply fusions ─────────────────────────────────────────────────────

    /// Mark the interior nodes of each fused group as `fused = true` and
    /// record the groups in the graph.
    ///
    /// The *last* node of each group is kept unfused so that downstream
    /// consumers can still reference it as an output.
    pub fn apply_fusions(&mut self, groups: &[FusedGroup]) {
        for group in groups {
            // All nodes except the last are absorbed
            let fused_count = group.node_ids.len().saturating_sub(1);
            for &node_id in &group.node_ids[..fused_count] {
                if let Some(node) = self.nodes.iter_mut().find(|n| n.id == node_id) {
                    node.fused = true;
                }
            }
        }
        self.fused_groups.extend(groups.iter().cloned());
    }

    // ── Queries ───────────────────────────────────────────────────────────

    /// Count nodes that have **not** been absorbed into a fused group.
    pub fn unfused_count(&self) -> usize {
        self.nodes.iter().filter(|n| !n.fused).count()
    }

    /// Total number of nodes in the graph.
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Slice of all applied fused groups.
    pub fn fused_groups(&self) -> &[FusedGroup] {
        &self.fused_groups
    }

    /// Compute FLOPs statistics for the graph.
    pub fn flop_analysis(&self) -> FlopAnalysis {
        let total_flops: u64 = self.nodes.iter().map(|n| n.flops).sum();
        let fused_flops_saved: u64 = self.fused_groups.iter().map(|g| g.flops_saved).sum();
        let num_fused_groups = self.fused_groups.len();
        let effective_flops = total_flops.saturating_sub(fused_flops_saved);
        let speedup_estimate = if effective_flops == 0 {
            1.0
        } else {
            total_flops as f32 / effective_flops as f32
        };
        FlopAnalysis {
            total_flops,
            fused_flops_saved,
            num_fused_groups,
            speedup_estimate,
        }
    }

    // ── Visualisation ─────────────────────────────────────────────────────

    /// Emit a Graphviz DOT representation of the graph.
    ///
    /// Fused nodes are coloured grey; active nodes are light blue.
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph fusion_graph {\n");
        for node in &self.nodes {
            let color = if node.fused { "gray" } else { "lightblue" };
            out.push_str(&format!(
                "  {} [label=\"{}\\n{}\", style=filled, fillcolor={}];\n",
                node.id, node.name, node.node_type, color
            ));
            for &input_id in &node.inputs {
                out.push_str(&format!("  {} -> {};\n", input_id, node.id));
            }
        }
        out.push_str("}\n");
        out
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── node construction ────────────────────────────────────────────────────

    #[test]
    fn test_graph_node_new() {
        let node = GraphNode::new(3, "my_layer", NodeType::Linear, vec![1, 2]);
        assert_eq!(node.id, 3);
        assert_eq!(node.name, "my_layer");
        assert_eq!(node.node_type, NodeType::Linear);
        assert_eq!(node.inputs, vec![1, 2]);
        assert!(!node.fused);
        assert_eq!(node.flops, 0);
        assert!(node.output_shape.is_empty());
    }

    #[test]
    fn test_fusion_graph_add_node() {
        let mut g = FusionGraph::new();
        let id0 = g.add_node("embedding", NodeType::Embedding, vec![]);
        let id1 = g.add_node("norm", NodeType::RmsNorm, vec![id0]);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(g.total_nodes(), 2);
    }

    #[test]
    fn test_fusion_graph_chain() {
        let mut g = FusionGraph::new();
        let a = g.add_node("a", NodeType::RmsNorm, vec![]);
        let b = g.add_node("b", NodeType::Linear, vec![a]);
        let c = g.add_node("c", NodeType::Attention, vec![b]);
        assert_eq!(g.nodes()[1].inputs, vec![a]);
        assert_eq!(g.nodes()[2].inputs, vec![b]);
        let _ = c; // used
    }

    // ── standard patterns ────────────────────────────────────────────────────

    #[test]
    fn test_standard_patterns_count() {
        let patterns = LayerFusionPattern::standard_patterns();
        assert!(
            patterns.len() >= 4,
            "Expected at least 4 standard patterns, got {}",
            patterns.len()
        );
    }

    #[test]
    fn test_pattern_match_layer_norm_linear() {
        let mut g = FusionGraph::new();
        let ln = g.add_node("ln", NodeType::LayerNorm, vec![]);
        let _lin = g.add_node("linear", NodeType::Linear, vec![ln]);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        let found = groups.iter().any(|grp| grp.fused_op == FusedOpType::LayerNormLinear);
        assert!(found, "LayerNorm+Linear pattern should be detected");
    }

    #[test]
    fn test_pattern_match_rms_norm_linear() {
        let mut g = FusionGraph::new();
        let rn = g.add_node("rms", NodeType::RmsNorm, vec![]);
        let _lin = g.add_node("proj", NodeType::Linear, vec![rn]);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        let found = groups.iter().any(|grp| grp.fused_op == FusedOpType::RmsNormLinear);
        assert!(found, "RMSNorm+Linear pattern should be detected");
    }

    #[test]
    fn test_pattern_no_match_non_chain() {
        // Linear does NOT consume LayerNorm (different input) → no match
        let mut g = FusionGraph::new();
        let _ln = g.add_node("ln", NodeType::LayerNorm, vec![]);
        let emb = g.add_node("emb", NodeType::Embedding, vec![]);
        let _lin = g.add_node("linear", NodeType::Linear, vec![emb]); // input is emb, not ln
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        let ln_lin = groups.iter().any(|grp| grp.fused_op == FusedOpType::LayerNormLinear);
        assert!(!ln_lin, "Should NOT match when nodes are not chained");
    }

    #[test]
    fn test_pattern_no_match_wrong_type() {
        let mut g = FusionGraph::new();
        let rn = g.add_node("relu", NodeType::ReLU, vec![]);
        let _lin = g.add_node("linear", NodeType::Linear, vec![rn]);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        let found = groups
            .iter()
            .any(|grp| matches!(grp.fused_op, FusedOpType::LayerNormLinear | FusedOpType::RmsNormLinear));
        assert!(!found, "ReLU+Linear should not match norm+linear patterns");
    }

    // ── apply fusions ────────────────────────────────────────────────────────

    #[test]
    fn test_apply_fusions_marks_fused() {
        let mut g = FusionGraph::new();
        let rn = g.add_node("rms", NodeType::RmsNorm, vec![]);
        let _lin = g.add_node("proj", NodeType::Linear, vec![rn]);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        g.apply_fusions(&groups);
        // The RmsNorm node (first in the chain) should be fused
        let rms_fused = g.nodes().iter().any(|n| n.node_type == NodeType::RmsNorm && n.fused);
        assert!(rms_fused, "RmsNorm should be marked as fused");
        // The Linear node (last in the chain) should remain unfused as the output
        let lin_unfused = g.nodes().iter().any(|n| n.node_type == NodeType::Linear && !n.fused);
        assert!(lin_unfused, "Linear (last in group) should remain unfused");
    }

    // ── unfused count ────────────────────────────────────────────────────────

    #[test]
    fn test_unfused_count() {
        let mut g = FusionGraph::new();
        let rn = g.add_node("rms", NodeType::RmsNorm, vec![]);
        let lin = g.add_node("linear", NodeType::Linear, vec![rn]);
        let _attn = g.add_node("attn", NodeType::Attention, vec![lin]);
        assert_eq!(g.unfused_count(), 3);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        g.apply_fusions(&groups);
        // RmsNorm is now fused → unfused = 2
        assert_eq!(g.unfused_count(), 2);
    }

    // ── flop analysis ────────────────────────────────────────────────────────

    #[test]
    fn test_flop_analysis() {
        let mut g = FusionGraph::new();
        let n0 = GraphNode::new(0, "rms", NodeType::RmsNorm, vec![]).with_flops(1_000);
        let n1_id = {
            // We need n1's id to reference n0
            let n1 = GraphNode::new(1, "linear", NodeType::Linear, vec![0]).with_flops(5_000);
            let id = n1.id;
            g.add_node_with_metadata(n1);
            id
        };
        g.add_node_with_metadata(n0.clone()); // re-add n0 (order doesn't matter for flops sum)
        // Total = 6_000 (both nodes)
        let fa_before = g.flop_analysis();
        assert_eq!(fa_before.total_flops, 6_000);
        assert_eq!(fa_before.num_fused_groups, 0);
        assert!((fa_before.speedup_estimate - 1.0).abs() < 1e-6);
        let _ = n1_id; // used
    }

    // ── DOT output ───────────────────────────────────────────────────────────

    #[test]
    fn test_fusion_graph_to_dot() {
        let mut g = FusionGraph::new();
        let rn = g.add_node("rms_norm", NodeType::RmsNorm, vec![]);
        let _lin = g.add_node("q_proj", NodeType::Linear, vec![rn]);
        let dot = g.to_dot();
        assert!(dot.contains("digraph fusion_graph"));
        assert!(dot.contains("rms_norm"));
        assert!(dot.contains("q_proj"));
        assert!(dot.contains("->"));
    }

    // ── fused group creation ─────────────────────────────────────────────────

    #[test]
    fn test_fused_group_creation() {
        let group = FusedGroup {
            id: 0,
            name: "test_group".to_string(),
            node_ids: vec![0, 1],
            fused_op: FusedOpType::RmsNormLinear,
            flops_saved: 500,
        };
        assert_eq!(group.id, 0);
        assert_eq!(group.node_ids.len(), 2);
        assert_eq!(group.fused_op, FusedOpType::RmsNormLinear);
        assert_eq!(group.flops_saved, 500);
    }

    // ── display impls ─────────────────────────────────────────────────────────

    #[test]
    fn test_fused_op_type_display() {
        assert_eq!(FusedOpType::LayerNormLinear.to_string(), "LayerNormLinear");
        assert_eq!(FusedOpType::RmsNormLinear.to_string(), "RmsNormLinear");
        assert_eq!(FusedOpType::AttentionScores.to_string(), "AttentionScores");
        assert_eq!(FusedOpType::SwiGLUFused.to_string(), "SwiGLUFused");
        assert_eq!(FusedOpType::GeGLUFused.to_string(), "GeGLUFused");
        assert_eq!(FusedOpType::ResidualAddNorm.to_string(), "ResidualAddNorm");
        let multi = FusedOpType::MultiOpFusion(vec!["A".to_string(), "B".to_string()]);
        assert!(multi.to_string().contains("A+B"));
    }

    #[test]
    fn test_node_type_display() {
        assert_eq!(NodeType::LayerNorm.to_string(), "LayerNorm");
        assert_eq!(NodeType::RmsNorm.to_string(), "RmsNorm");
        assert_eq!(NodeType::Linear.to_string(), "Linear");
        assert_eq!(NodeType::Attention.to_string(), "Attention");
        assert_eq!(NodeType::SwiGLU.to_string(), "SwiGLU");
        assert_eq!(NodeType::GeGLU.to_string(), "GeGLU");
        assert_eq!(NodeType::Residual.to_string(), "Residual");
        let custom = NodeType::Custom("MyOp".to_string());
        assert!(custom.to_string().contains("MyOp"));
    }

    // ── full transformer block simulation ────────────────────────────────────

    #[test]
    fn test_fusion_analysis_transformer_block() {
        // Simulate a typical LLaMA-style transformer block:
        //   Residual → RmsNorm → Linear (Q) × 3 + Attention → Residual → RmsNorm → SwiGLU → Linear
        let mut g = FusionGraph::new();
        let emb = g.add_node("embedding", NodeType::Embedding, vec![]);
        let res1 = g.add_node("residual1", NodeType::Residual, vec![emb]);
        let rn1 = g.add_node("pre_attn_norm", NodeType::RmsNorm, vec![res1]);
        let q_proj = g.add_node("q_proj", NodeType::Linear, vec![rn1]);
        let k_proj = g.add_node("k_proj", NodeType::Linear, vec![rn1]);
        let v_proj = g.add_node("v_proj", NodeType::Linear, vec![rn1]);
        let attn = g.add_node("attention", NodeType::Attention, vec![q_proj, k_proj, v_proj]);
        let res2 = g.add_node("residual2", NodeType::Residual, vec![attn]);
        let rn2 = g.add_node("pre_ffn_norm", NodeType::RmsNorm, vec![res2]);
        let ffn = g.add_node("ffn_swiglu", NodeType::SwiGLU, vec![rn2]);
        let out_proj = g.add_node("out_proj", NodeType::Linear, vec![ffn]);

        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);

        // At minimum we expect ≥ 2 fused groups
        assert!(
            groups.len() >= 2,
            "Expected at least 2 fused groups in transformer block, got {}",
            groups.len()
        );

        let has_rms_lin = groups.iter().any(|g| g.fused_op == FusedOpType::RmsNormLinear);
        assert!(has_rms_lin, "Should detect RmsNorm+Linear pattern");

        let has_swiglu = groups.iter().any(|g| g.fused_op == FusedOpType::SwiGLUFused);
        assert!(has_swiglu, "Should detect SwiGLU+Linear pattern");

        let _ = (out_proj, k_proj, v_proj); // mark as used
    }

    // ── Additional tests ──────────────────────────────────────────────────────

    /// 3-node chain: RmsNorm → Linear → Attention; only first 2 match RmsNorm+Linear.
    #[test]
    fn test_chain_detection_three_nodes() {
        let mut g = FusionGraph::new();
        let a = g.add_node("rms", NodeType::RmsNorm, vec![]);
        let b = g.add_node("linear", NodeType::Linear, vec![a]);
        let _c = g.add_node("attn", NodeType::Attention, vec![b]);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        let rms_lin = groups.iter().filter(|g| g.fused_op == FusedOpType::RmsNormLinear).count();
        assert_eq!(rms_lin, 1, "exactly one RmsNorm+Linear chain detected");
    }

    /// Single-node graph never matches any 2-node pattern.
    #[test]
    fn test_single_node_no_pattern_match() {
        let mut g = FusionGraph::new();
        let _id = g.add_node("only_rms", NodeType::RmsNorm, vec![]);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        assert!(groups.is_empty(), "single node should not match any pattern");
    }

    /// Fork pattern: two Linear nodes share same RmsNorm parent (fan-out).
    /// Both can individually match RmsNorm+Linear if they appear sequentially.
    #[test]
    fn test_fork_pattern_fan_out() {
        let mut g = FusionGraph::new();
        let rn = g.add_node("rms", NodeType::RmsNorm, vec![]);
        let lin1 = g.add_node("q_proj", NodeType::Linear, vec![rn]);
        // lin2 also takes rn as input (fork)
        let _lin2 = g.add_node("k_proj", NodeType::Linear, vec![rn]);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        // rms → lin1 appears in sequential order → should match
        let has_rms_lin = groups.iter().any(|g| g.fused_op == FusedOpType::RmsNormLinear);
        assert!(has_rms_lin, "fork fan-out should still detect RmsNorm+Linear");
        let _ = lin1;
    }

    /// DOT export format: contains node IDs, names, and edge arrows.
    #[test]
    fn test_dot_export_format_correctness() {
        let mut g = FusionGraph::new();
        let a = g.add_node("layer_a", NodeType::RmsNorm, vec![]);
        let b = g.add_node("layer_b", NodeType::Linear, vec![a]);
        let dot = g.to_dot();
        assert!(dot.starts_with("digraph fusion_graph"), "must open with digraph");
        assert!(dot.contains("layer_a"), "must contain node name");
        assert!(dot.contains("layer_b"), "must contain node name");
        assert!(dot.contains("->"), "must contain edge arrows");
        assert!(dot.ends_with("}\n"), "must close with closing brace");
        let _ = b;
    }

    /// DOT export uses different colours for fused vs active nodes.
    #[test]
    fn test_dot_export_fused_vs_active_colors() {
        let mut g = FusionGraph::new();
        let rn = g.add_node("rms", NodeType::RmsNorm, vec![]);
        let _lin = g.add_node("proj", NodeType::Linear, vec![rn]);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        g.apply_fusions(&groups);
        let dot = g.to_dot();
        // Fused nodes are grey; active nodes are lightblue.
        assert!(dot.contains("gray"), "fused node should appear gray");
        assert!(dot.contains("lightblue"), "active node should appear lightblue");
    }

    /// Node activation tracking: fused flag set after apply_fusions.
    #[test]
    fn test_node_activation_tracking_after_apply() {
        let mut g = FusionGraph::new();
        let rn = g.add_node("rms", NodeType::RmsNorm, vec![]);
        let _lin = g.add_node("proj", NodeType::Linear, vec![rn]);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        // Before apply: no fused nodes
        assert!(g.nodes().iter().all(|n| !n.fused), "no fused nodes before apply");
        g.apply_fusions(&groups);
        // After apply: the first node (RmsNorm) in the chain is fused
        let has_fused = g.nodes().iter().any(|n| n.fused);
        assert!(has_fused, "at least one node should be fused after apply_fusions");
    }

    /// Graph traversal: nodes() preserves insertion order (BFS-friendly).
    #[test]
    fn test_graph_traversal_insertion_order() {
        let mut g = FusionGraph::new();
        let a = g.add_node("first", NodeType::Embedding, vec![]);
        let b = g.add_node("second", NodeType::RmsNorm, vec![a]);
        let c = g.add_node("third", NodeType::Linear, vec![b]);
        let ids: Vec<usize> = g.nodes().iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![a, b, c], "nodes must be in insertion order");
    }

    /// Multiple pattern matches in one graph.
    #[test]
    fn test_multiple_pattern_matches_in_one_graph() {
        let mut g = FusionGraph::new();
        // Block 1: RmsNorm + Linear
        let rn1 = g.add_node("rms1", NodeType::RmsNorm, vec![]);
        let lin1 = g.add_node("lin1", NodeType::Linear, vec![rn1]);
        // Block 2: another RmsNorm + Linear
        let rn2 = g.add_node("rms2", NodeType::RmsNorm, vec![lin1]);
        let _lin2 = g.add_node("lin2", NodeType::Linear, vec![rn2]);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        let rms_lin_count = groups.iter().filter(|g| g.fused_op == FusedOpType::RmsNormLinear).count();
        assert!(rms_lin_count >= 2, "both RmsNorm+Linear chains must be detected, got {rms_lin_count}");
    }

    /// Graph with no matching patterns gives empty groups.
    #[test]
    fn test_no_matching_patterns_gives_empty_groups() {
        let mut g = FusionGraph::new();
        // Attention followed by Dropout — no standard pattern matches this.
        let a = g.add_node("attn", NodeType::Attention, vec![]);
        let _d = g.add_node("drop", NodeType::Dropout, vec![a]);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        assert!(groups.is_empty(), "Attention+Dropout should not match any standard pattern");
    }

    /// FlopAnalysis speedup_estimate > 1.0 after fusion when nodes have flops.
    #[test]
    fn test_flop_analysis_speedup_after_fusion() {
        let mut g = FusionGraph::new();
        let n0 = GraphNode::new(0, "rms", NodeType::RmsNorm, vec![]).with_flops(10_000);
        let n1 = GraphNode::new(1, "proj", NodeType::Linear, vec![0]).with_flops(50_000);
        g.add_node_with_metadata(n0);
        g.add_node_with_metadata(n1);

        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        g.apply_fusions(&groups);

        let fa = g.flop_analysis();
        assert!(fa.num_fused_groups > 0, "should have at least one fused group");
        assert!(fa.speedup_estimate >= 1.0, "speedup must be >= 1.0 after fusion");
    }

    /// FusedGroup flops_saved reflects sum of node flops (divided by 10).
    #[test]
    fn test_fused_group_flops_saved() {
        let mut g = FusionGraph::new();
        let n0 = GraphNode::new(0, "rms", NodeType::RmsNorm, vec![]).with_flops(1_000);
        let n1 = GraphNode::new(1, "proj", NodeType::Linear, vec![0]).with_flops(9_000);
        g.add_node_with_metadata(n0);
        g.add_node_with_metadata(n1);

        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        // flops_saved = (1000 + 9000) / 10 = 1000
        assert!(!groups.is_empty(), "should match RmsNorm+Linear");
        assert_eq!(groups[0].flops_saved, 1_000, "flops_saved = total_flops / 10");
    }

    /// Join pattern (fan-in): Attention with multiple inputs, followed by a
    /// node that joins them — the Attention node itself should be addable.
    #[test]
    fn test_join_pattern_fan_in() {
        let mut g = FusionGraph::new();
        let q = g.add_node("q", NodeType::Linear, vec![]);
        let k = g.add_node("k", NodeType::Linear, vec![]);
        let v = g.add_node("v", NodeType::Linear, vec![]);
        let _attn = g.add_node("attn", NodeType::Attention, vec![q, k, v]);
        assert_eq!(g.total_nodes(), 4);
        // No standard pattern includes an Attention node as the first of a pair
        // so groups should be empty (or possibly match Linear+something).
        // Mainly verify the graph accepts fan-in topology without panic.
        let patterns = LayerFusionPattern::standard_patterns();
        let _groups = g.analyze_fusions(&patterns);
    }

    /// fused_groups() slice is accessible and reflects applied groups.
    #[test]
    fn test_fused_groups_slice_accessible() {
        let mut g = FusionGraph::new();
        let rn = g.add_node("rms", NodeType::RmsNorm, vec![]);
        let _lin = g.add_node("lin", NodeType::Linear, vec![rn]);
        let patterns = LayerFusionPattern::standard_patterns();
        let groups = g.analyze_fusions(&patterns);
        g.apply_fusions(&groups);
        let stored = g.fused_groups();
        assert!(!stored.is_empty(), "fused_groups slice should be non-empty after apply");
        assert_eq!(stored[0].fused_op, FusedOpType::RmsNormLinear);
    }

    /// Node with_flops and with_shape builder setters work correctly.
    #[test]
    fn test_graph_node_builders() {
        let node = GraphNode::new(7, "my_node", NodeType::SiLU, vec![3, 5])
            .with_flops(99_000)
            .with_shape(vec![1, 512, 768]);
        assert_eq!(node.flops, 99_000);
        assert_eq!(node.output_shape, vec![1, 512, 768]);
        assert_eq!(node.inputs, vec![3, 5]);
    }

    /// Custom node type display contains the custom string.
    #[test]
    fn test_custom_node_type_display() {
        let nt = NodeType::Custom("SpecialOp".to_string());
        let s = nt.to_string();
        assert!(s.contains("SpecialOp"), "custom node display: {s}");
    }
}
