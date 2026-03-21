//! Computation graph debugger for automatic differentiation
//!
//! This module provides debugging and visualization tools for computation graphs,
//! helping developers understand gradient flow, detect issues, and optimize
//! automatic differentiation computations.

#![allow(unused_variables)] // Autodiff debugger

use super::graph::{ComputationGraph, GraphNode, NodeId, OperationType};
use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write;

/// Computation graph debugger
pub struct GraphDebugger {
    /// Configuration for debugging
    config: DebuggerConfig,
    /// Analysis results cache
    analysis_cache: HashMap<String, AnalysisResult>,
    /// Breakpoints for debugging
    breakpoints: HashSet<NodeId>,
}

/// Configuration for the graph debugger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebuggerConfig {
    /// Maximum number of nodes to display in summaries
    pub max_display_nodes: usize,
    /// Whether to show gradient information
    pub show_gradients: bool,
    /// Whether to show tensor shapes
    pub show_shapes: bool,
    /// Whether to show tensor values (can be verbose)
    pub show_values: bool,
    /// Output format for graph visualization
    pub output_format: GraphOutputFormat,
    /// Threshold for gradient magnitude warnings
    pub gradient_magnitude_threshold: f32,
    /// Threshold for detecting vanishing gradients
    pub vanishing_gradient_threshold: f32,
    /// Threshold for detecting exploding gradients
    pub exploding_gradient_threshold: f32,
}

/// Output formats for graph visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphOutputFormat {
    /// DOT format for Graphviz
    Dot,
    /// ASCII art representation
    ASCII,
    /// JSON format for programmatic use
    JSON,
    /// HTML with interactive features
    HTML,
}

/// Analysis result for computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of leaf nodes
    pub leaf_nodes: usize,
    /// Number of root nodes
    pub root_nodes: usize,
    /// Maximum depth of the graph
    pub max_depth: usize,
    /// Number of operations by type
    pub operation_counts: HashMap<String, usize>,
    /// Gradient flow statistics
    pub gradient_stats: GradientFlowStats,
    /// Memory usage estimates
    pub memory_stats: MemoryStats,
    /// Potential issues detected
    pub issues: Vec<GraphIssue>,
}

/// Statistics about gradient flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlowStats {
    /// Nodes with gradients
    pub nodes_with_gradients: usize,
    /// Nodes requiring gradients
    pub nodes_requiring_gradients: usize,
    /// Average gradient magnitude
    pub average_gradient_magnitude: f32,
    /// Maximum gradient magnitude
    pub max_gradient_magnitude: f32,
    /// Minimum gradient magnitude
    pub min_gradient_magnitude: f32,
    /// Nodes with vanishing gradients
    pub vanishing_gradient_nodes: Vec<NodeId>,
    /// Nodes with exploding gradients
    pub exploding_gradient_nodes: Vec<NodeId>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total memory used by tensors (bytes)
    pub total_tensor_memory: usize,
    /// Total memory used by gradients (bytes)
    pub total_gradient_memory: usize,
    /// Peak memory usage estimate
    pub peak_memory_estimate: usize,
    /// Memory usage by node
    pub memory_per_node: HashMap<NodeId, usize>,
}

/// Issues detected in the computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphIssue {
    /// Type of issue
    pub issue_type: IssueType,
    /// Nodes involved in the issue
    pub nodes: Vec<NodeId>,
    /// Description of the issue
    pub description: String,
    /// Severity level
    pub severity: IssueSeverity,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Types of issues that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueType {
    /// Vanishing gradients
    VanishingGradients,
    /// Exploding gradients
    ExplodingGradients,
    /// Disconnected subgraphs
    DisconnectedSubgraph,
    /// Cycles in the graph
    CyclicDependency,
    /// Inefficient operations
    IneffientOperation,
    /// Shape mismatches
    ShapeMismatch,
    /// Memory issues
    MemoryIssue,
    /// Numerical instability
    NumericalInstability,
}

/// Severity levels for issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Critical issues that will cause failures
    Critical,
    /// Warning issues that may cause problems
    Warning,
    /// Info issues for optimization
    Info,
}

/// Node information for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDebugInfo {
    pub id: NodeId,
    pub name: Option<String>,
    pub operation: Option<OperationType>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub has_gradient: bool,
    pub gradient_magnitude: Option<f32>,
    pub tensor_magnitude: f32,
    pub memory_usage: usize,
    pub parents: Vec<NodeId>,
    pub children: Vec<NodeId>,
    pub depth_from_root: usize,
}

/// Graph traversal information
#[derive(Debug, Clone)]
pub struct TraversalInfo {
    pub visited_nodes: HashSet<NodeId>,
    pub node_depths: HashMap<NodeId, usize>,
    pub execution_order: Vec<NodeId>,
}

impl Default for DebuggerConfig {
    fn default() -> Self {
        Self {
            max_display_nodes: 50,
            show_gradients: true,
            show_shapes: true,
            show_values: false,
            output_format: GraphOutputFormat::Dot,
            gradient_magnitude_threshold: 1e-6,
            vanishing_gradient_threshold: 1e-7,
            exploding_gradient_threshold: 1e3,
        }
    }
}

impl Default for GraphDebugger {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphDebugger {
    /// Create a new graph debugger
    pub fn new() -> Self {
        Self {
            config: DebuggerConfig::default(),
            analysis_cache: HashMap::new(),
            breakpoints: HashSet::new(),
        }
    }

    /// Create a new graph debugger with custom configuration
    pub fn with_config(config: DebuggerConfig) -> Self {
        Self {
            config,
            analysis_cache: HashMap::new(),
            breakpoints: HashSet::new(),
        }
    }

    /// Analyze a computation graph
    pub fn analyze(&mut self, graph: &ComputationGraph) -> Result<AnalysisResult> {
        let graph_hash = self.compute_graph_hash(graph);

        if let Some(cached_result) = self.analysis_cache.get(&graph_hash) {
            return Ok(cached_result.clone());
        }

        let nodes = self.get_all_nodes(graph)?;
        let total_nodes = nodes.len();

        // Count different types of nodes
        let leaf_nodes = nodes.iter().filter(|n| n.is_leaf).count();
        let root_nodes = nodes.iter().filter(|n| n.parents.is_empty()).count();

        // Compute graph depth
        let max_depth = self.compute_max_depth(graph, &nodes)?;

        // Count operations by type
        let operation_counts = self.count_operations(&nodes);

        // Analyze gradient flow
        let gradient_stats = self.analyze_gradient_flow(&nodes)?;

        // Compute memory statistics
        let memory_stats = self.compute_memory_stats(&nodes)?;

        // Detect issues
        let issues = self.detect_issues(graph, &nodes, &gradient_stats)?;

        let result = AnalysisResult {
            total_nodes,
            leaf_nodes,
            root_nodes,
            max_depth,
            operation_counts,
            gradient_stats,
            memory_stats,
            issues,
        };

        self.analysis_cache.insert(graph_hash, result.clone());
        Ok(result)
    }

    /// Generate a visual representation of the computation graph
    pub fn visualize(&self, graph: &ComputationGraph) -> Result<String> {
        match self.config.output_format {
            GraphOutputFormat::Dot => self.generate_dot_graph(graph),
            GraphOutputFormat::ASCII => self.generate_ascii_graph(graph),
            GraphOutputFormat::JSON => self.generate_json_graph(graph),
            GraphOutputFormat::HTML => self.generate_html_graph(graph),
        }
    }

    /// Trace gradient flow from a specific node
    pub fn trace_gradients(
        &self,
        graph: &ComputationGraph,
        start_node: NodeId,
    ) -> Result<Vec<NodeDebugInfo>> {
        let mut trace = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start_node);

        while let Some(node_id) = queue.pop_front() {
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id);

            let node = self.get_node(graph, node_id)?;
            let debug_info = self.create_node_debug_info(&node);
            trace.push(debug_info);

            // Add parent nodes to trace backward through gradients
            for &parent_id in &node.parents {
                if !visited.contains(&parent_id) {
                    queue.push_back(parent_id);
                }
            }
        }

        Ok(trace)
    }

    /// Set a breakpoint at a specific node
    pub fn set_breakpoint(&mut self, node_id: NodeId) {
        self.breakpoints.insert(node_id);
    }

    /// Remove a breakpoint
    pub fn remove_breakpoint(&mut self, node_id: NodeId) {
        self.breakpoints.remove(&node_id);
    }

    /// Check if execution should break at a node
    pub fn should_break(&self, node_id: NodeId) -> bool {
        self.breakpoints.contains(&node_id)
    }

    /// Get debug information for a specific node
    pub fn get_node_info(
        &self,
        graph: &ComputationGraph,
        node_id: NodeId,
    ) -> Result<NodeDebugInfo> {
        let node = self.get_node(graph, node_id)?;
        Ok(self.create_node_debug_info(&node))
    }

    /// Find nodes by name pattern
    pub fn find_nodes_by_name(
        &self,
        graph: &ComputationGraph,
        pattern: &str,
    ) -> Result<Vec<NodeId>> {
        let nodes = self.get_all_nodes(graph)?;
        let matching_nodes = nodes
            .iter()
            .filter(|node| node.name.as_ref().map(|name| name.contains(pattern)).unwrap_or(false))
            .map(|node| node.id)
            .collect();

        Ok(matching_nodes)
    }

    /// Generate a summary report of the computation graph
    pub fn generate_summary(&mut self, graph: &ComputationGraph) -> Result<String> {
        let analysis = self.analyze(graph)?;
        let mut report = String::new();

        writeln!(report, "Computation Graph Summary")?;
        writeln!(report, "=========================")?;
        writeln!(report)?;

        writeln!(report, "Graph Structure:")?;
        writeln!(report, "  Total nodes: {}", analysis.total_nodes)?;
        writeln!(report, "  Leaf nodes: {}", analysis.leaf_nodes)?;
        writeln!(report, "  Root nodes: {}", analysis.root_nodes)?;
        writeln!(report, "  Maximum depth: {}", analysis.max_depth)?;
        writeln!(report)?;

        writeln!(report, "Operations:")?;
        for (op_type, count) in &analysis.operation_counts {
            writeln!(report, "  {}: {}", op_type, count)?;
        }
        writeln!(report)?;

        writeln!(report, "Gradient Flow:")?;
        writeln!(
            report,
            "  Nodes with gradients: {}",
            analysis.gradient_stats.nodes_with_gradients
        )?;
        writeln!(
            report,
            "  Nodes requiring gradients: {}",
            analysis.gradient_stats.nodes_requiring_gradients
        )?;
        writeln!(
            report,
            "  Average gradient magnitude: {:.6}",
            analysis.gradient_stats.average_gradient_magnitude
        )?;
        writeln!(
            report,
            "  Max gradient magnitude: {:.6}",
            analysis.gradient_stats.max_gradient_magnitude
        )?;
        writeln!(
            report,
            "  Min gradient magnitude: {:.6}",
            analysis.gradient_stats.min_gradient_magnitude
        )?;
        writeln!(report)?;

        writeln!(report, "Memory Usage:")?;
        writeln!(
            report,
            "  Total tensor memory: {} bytes",
            analysis.memory_stats.total_tensor_memory
        )?;
        writeln!(
            report,
            "  Total gradient memory: {} bytes",
            analysis.memory_stats.total_gradient_memory
        )?;
        writeln!(
            report,
            "  Peak memory estimate: {} bytes",
            analysis.memory_stats.peak_memory_estimate
        )?;
        writeln!(report)?;

        if !analysis.issues.is_empty() {
            writeln!(report, "Issues Detected:")?;
            for issue in &analysis.issues {
                writeln!(
                    report,
                    "  [{:?}] {:?}: {}",
                    issue.severity, issue.issue_type, issue.description
                )?;
                if let Some(suggestion) = &issue.suggestion {
                    writeln!(report, "    Suggestion: {}", suggestion)?;
                }
            }
        } else {
            writeln!(report, "No issues detected.")?;
        }

        Ok(report)
    }

    /// Save debug information to file
    pub fn save_debug_info(&mut self, graph: &ComputationGraph, path: &str) -> Result<()> {
        let analysis = self.analyze(graph)?;
        let json_data = serde_json::to_string_pretty(&analysis)?;
        std::fs::write(path, json_data)?;
        Ok(())
    }

    // Helper methods

    fn get_all_nodes(&self, graph: &ComputationGraph) -> Result<Vec<GraphNode>> {
        // Access all nodes from the computation graph using the public export method
        Ok(graph.export_graph().nodes)
    }

    fn get_node(&self, graph: &ComputationGraph, node_id: NodeId) -> Result<GraphNode> {
        // Get a specific node from the computation graph
        let export = graph.export_graph();
        export.nodes.into_iter().find(|node| node.id == node_id).ok_or_else(|| {
            TrustformersError::new(crate::errors::ErrorKind::TensorOpError {
                operation: "get_node".to_string(),
                reason: format!("Node {} not found in computation graph", node_id),
            })
        })
    }

    fn compute_graph_hash(&self, graph: &ComputationGraph) -> String {
        // Compute a hash based on graph structure and operations
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash number of nodes
        graph.num_nodes().hash(&mut hasher);

        // Hash topological order
        graph.get_topological_order().hash(&mut hasher);

        // Hash each node's structure (operations and connections)
        let export = graph.export_graph();
        let mut nodes = export.nodes;
        nodes.sort_by_key(|node| node.id);

        for node in nodes {
            node.id.hash(&mut hasher);

            // Hash operation type
            if let Some(ref op) = node.operation {
                std::mem::discriminant(op).hash(&mut hasher);
            }

            // Hash parent connections
            let mut parents = node.parents.clone();
            parents.sort();
            parents.hash(&mut hasher);

            // Hash whether requires grad
            node.requires_grad.hash(&mut hasher);
            node.is_leaf.hash(&mut hasher);
        }

        format!("graph_{:x}", hasher.finish())
    }

    fn compute_max_depth(&self, graph: &ComputationGraph, nodes: &[GraphNode]) -> Result<usize> {
        let mut max_depth = 0;
        let mut visited = HashSet::new();

        for node in nodes {
            if node.is_leaf {
                let depth = self.compute_node_depth(graph, node.id, &mut visited)?;
                max_depth = max_depth.max(depth);
            }
        }

        Ok(max_depth)
    }

    fn compute_node_depth(
        &self,
        graph: &ComputationGraph,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
    ) -> Result<usize> {
        if visited.contains(&node_id) {
            return Ok(0); // Avoid infinite recursion
        }
        visited.insert(node_id);

        let node = self.get_node(graph, node_id)?;
        if node.children.is_empty() {
            return Ok(0);
        }

        let mut max_child_depth = 0;
        for &child_id in &node.children {
            let child_depth = self.compute_node_depth(graph, child_id, visited)?;
            max_child_depth = max_child_depth.max(child_depth);
        }

        Ok(max_child_depth + 1)
    }

    fn count_operations(&self, nodes: &[GraphNode]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        for node in nodes {
            if let Some(ref op) = node.operation {
                let op_name = format!("{:?}", op);
                *counts.entry(op_name).or_insert(0) += 1;
            }
        }

        counts
    }

    fn analyze_gradient_flow(&self, nodes: &[GraphNode]) -> Result<GradientFlowStats> {
        let nodes_with_gradients = nodes.iter().filter(|n| n.gradient.is_some()).count();
        let nodes_requiring_gradients = nodes.iter().filter(|n| n.requires_grad).count();

        let gradient_magnitudes: Vec<f32> = nodes
            .iter()
            .filter_map(|node| {
                node.gradient.as_ref().and_then(|grad| self.compute_tensor_magnitude(grad).ok())
            })
            .collect();

        let (average_gradient_magnitude, max_gradient_magnitude, min_gradient_magnitude) =
            if gradient_magnitudes.is_empty() {
                (0.0, 0.0, 0.0)
            } else {
                let sum: f32 = gradient_magnitudes.iter().sum();
                let avg = sum / gradient_magnitudes.len() as f32;
                let max = gradient_magnitudes.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let min = gradient_magnitudes.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                (avg, max, min)
            };

        let vanishing_gradient_nodes: Vec<NodeId> = nodes
            .iter()
            .filter(|node| {
                node.gradient
                    .as_ref()
                    .and_then(|grad| self.compute_tensor_magnitude(grad).ok())
                    .map(|mag| mag < self.config.vanishing_gradient_threshold)
                    .unwrap_or(false)
            })
            .map(|node| node.id)
            .collect();

        let exploding_gradient_nodes: Vec<NodeId> = nodes
            .iter()
            .filter(|node| {
                node.gradient
                    .as_ref()
                    .and_then(|grad| self.compute_tensor_magnitude(grad).ok())
                    .map(|mag| mag > self.config.exploding_gradient_threshold)
                    .unwrap_or(false)
            })
            .map(|node| node.id)
            .collect();

        Ok(GradientFlowStats {
            nodes_with_gradients,
            nodes_requiring_gradients,
            average_gradient_magnitude,
            max_gradient_magnitude,
            min_gradient_magnitude,
            vanishing_gradient_nodes,
            exploding_gradient_nodes,
        })
    }

    fn compute_memory_stats(&self, nodes: &[GraphNode]) -> Result<MemoryStats> {
        let mut total_tensor_memory = 0;
        let mut total_gradient_memory = 0;
        let mut memory_per_node = HashMap::new();

        for node in nodes {
            let tensor_memory = node.value.memory_usage();
            let gradient_memory = node.gradient.as_ref().map(|g| g.memory_usage()).unwrap_or(0);

            total_tensor_memory += tensor_memory;
            total_gradient_memory += gradient_memory;
            memory_per_node.insert(node.id, tensor_memory + gradient_memory);
        }

        let peak_memory_estimate = total_tensor_memory + total_gradient_memory;

        Ok(MemoryStats {
            total_tensor_memory,
            total_gradient_memory,
            peak_memory_estimate,
            memory_per_node,
        })
    }

    fn detect_issues(
        &self,
        graph: &ComputationGraph,
        nodes: &[GraphNode],
        gradient_stats: &GradientFlowStats,
    ) -> Result<Vec<GraphIssue>> {
        let mut issues = Vec::new();

        // Check for vanishing gradients
        if !gradient_stats.vanishing_gradient_nodes.is_empty() {
            issues.push(GraphIssue {
                issue_type: IssueType::VanishingGradients,
                nodes: gradient_stats.vanishing_gradient_nodes.clone(),
                description: format!(
                    "Detected {} nodes with vanishing gradients",
                    gradient_stats.vanishing_gradient_nodes.len()
                ),
                severity: IssueSeverity::Warning,
                suggestion: Some(
                    "Consider using gradient clipping or adjusting learning rates".to_string(),
                ),
            });
        }

        // Check for exploding gradients
        if !gradient_stats.exploding_gradient_nodes.is_empty() {
            issues.push(GraphIssue {
                issue_type: IssueType::ExplodingGradients,
                nodes: gradient_stats.exploding_gradient_nodes.clone(),
                description: format!(
                    "Detected {} nodes with exploding gradients",
                    gradient_stats.exploding_gradient_nodes.len()
                ),
                severity: IssueSeverity::Critical,
                suggestion: Some("Apply gradient clipping to prevent instability".to_string()),
            });
        }

        // Check for disconnected subgraphs
        let disconnected_nodes = self.find_disconnected_nodes(graph, nodes)?;
        if !disconnected_nodes.is_empty() {
            issues.push(GraphIssue {
                issue_type: IssueType::DisconnectedSubgraph,
                nodes: disconnected_nodes,
                description: "Found disconnected nodes in the computation graph".to_string(),
                severity: IssueSeverity::Warning,
                suggestion: Some("Check that all variables are properly connected".to_string()),
            });
        }

        Ok(issues)
    }

    fn find_disconnected_nodes(
        &self,
        graph: &ComputationGraph,
        nodes: &[GraphNode],
    ) -> Result<Vec<NodeId>> {
        // Placeholder implementation
        Ok(Vec::new())
    }

    fn compute_tensor_magnitude(&self, tensor: &Tensor) -> Result<f32> {
        match tensor {
            Tensor::F32(arr) => {
                let magnitude = arr.iter().map(|&x| x * x).sum::<f32>().sqrt();
                Ok(magnitude)
            },
            _ => Err(TrustformersError::new(
                crate::errors::ErrorKind::TensorOpError {
                    operation: "compute_magnitude".to_string(),
                    reason: "Magnitude computation not supported for this tensor type".to_string(),
                },
            )),
        }
    }

    fn create_node_debug_info(&self, node: &GraphNode) -> NodeDebugInfo {
        let gradient_magnitude =
            node.gradient.as_ref().and_then(|grad| self.compute_tensor_magnitude(grad).ok());

        let tensor_magnitude = self.compute_tensor_magnitude(&node.value).unwrap_or(0.0);

        NodeDebugInfo {
            id: node.id,
            name: node.name.clone(),
            operation: node.operation.clone(),
            shape: node.shape.clone(),
            requires_grad: node.requires_grad,
            is_leaf: node.is_leaf,
            has_gradient: node.gradient.is_some(),
            gradient_magnitude,
            tensor_magnitude,
            memory_usage: node.value.memory_usage(),
            parents: node.parents.clone(),
            children: node.children.clone(),
            depth_from_root: 0, // Would be computed in real implementation
        }
    }

    fn generate_dot_graph(&self, graph: &ComputationGraph) -> Result<String> {
        let mut dot = String::new();
        writeln!(dot, "digraph ComputationGraph {{")?;
        writeln!(dot, "  rankdir=TB;")?;
        writeln!(dot, "  node [shape=box, style=filled, fontname=Arial];")?;

        let nodes = self.get_all_nodes(graph)?;

        for node in &nodes {
            let color = if node.is_leaf {
                "lightblue"
            } else if node.gradient.is_some() {
                "lightgreen"
            } else {
                "lightgray"
            };

            let label = if let Some(ref name) = node.name {
                format!(
                    "{}\\n{:?}",
                    name,
                    node.operation.as_ref().unwrap_or(&OperationType::Add)
                )
            } else {
                format!(
                    "Node {}\\n{:?}",
                    node.id,
                    node.operation.as_ref().unwrap_or(&OperationType::Add)
                )
            };

            writeln!(
                dot,
                "  {} [label=\"{}\", fillcolor={}];",
                node.id, label, color
            )?;
        }

        for node in &nodes {
            for &child_id in &node.children {
                writeln!(dot, "  {} -> {};", node.id, child_id)?;
            }
        }

        writeln!(dot, "}}")?;
        Ok(dot)
    }

    fn generate_ascii_graph(&self, graph: &ComputationGraph) -> Result<String> {
        let mut output = String::new();
        writeln!(output, "Computation Graph (ASCII)")?;
        writeln!(output, "=========================")?;

        let nodes = self.get_all_nodes(graph)?;

        for node in &nodes {
            let status = if node.is_leaf { "[LEAF]" } else { "[OP]" };
            let grad_status = if node.gradient.is_some() { "[GRAD]" } else { "" };

            writeln!(
                output,
                "Node {}: {} {} {:?}",
                node.id,
                status,
                grad_status,
                node.operation.as_ref().unwrap_or(&OperationType::Add)
            )?;

            if !node.children.is_empty() {
                writeln!(output, "  └─ Children: {:?}", node.children)?;
            }
        }

        Ok(output)
    }

    fn generate_json_graph(&self, graph: &ComputationGraph) -> Result<String> {
        let nodes = self.get_all_nodes(graph)?;
        let debug_nodes: Vec<NodeDebugInfo> =
            nodes.iter().map(|node| self.create_node_debug_info(node)).collect();

        let json_data = serde_json::json!({
            "nodes": debug_nodes,
            "total_nodes": nodes.len(),
        });

        Ok(serde_json::to_string_pretty(&json_data)?)
    }

    fn generate_html_graph(&self, graph: &ComputationGraph) -> Result<String> {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>Computation Graph Debug View</title>\n");
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html.push_str(
            ".node { border: 1px solid #ccc; margin: 10px; padding: 10px; border-radius: 5px; }\n",
        );
        html.push_str(".leaf { background-color: #e3f2fd; }\n");
        html.push_str(".op { background-color: #f3e5f5; }\n");
        html.push_str(".grad { border-left: 4px solid #4caf50; }\n");
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");

        html.push_str("<h1>Computation Graph Debug View</h1>\n");

        let nodes = self.get_all_nodes(graph)?;

        for node in &nodes {
            let node_class = if node.is_leaf { "node leaf" } else { "node op" };
            let grad_class = if node.gradient.is_some() { " grad" } else { "" };

            html.push_str(&format!("<div class=\"{}{}\">\n", node_class, grad_class));
            html.push_str(&format!("<h3>Node {}</h3>\n", node.id));

            if let Some(ref name) = node.name {
                html.push_str(&format!("<p><strong>Name:</strong> {}</p>\n", name));
            }

            if let Some(ref op) = node.operation {
                html.push_str(&format!("<p><strong>Operation:</strong> {:?}</p>\n", op));
            }

            html.push_str(&format!(
                "<p><strong>Shape:</strong> {:?}</p>\n",
                node.shape
            ));
            html.push_str(&format!(
                "<p><strong>Requires Grad:</strong> {}</p>\n",
                node.requires_grad
            ));
            html.push_str(&format!(
                "<p><strong>Is Leaf:</strong> {}</p>\n",
                node.is_leaf
            ));
            html.push_str(&format!(
                "<p><strong>Has Gradient:</strong> {}</p>\n",
                node.gradient.is_some()
            ));
            html.push_str(&format!(
                "<p><strong>Memory:</strong> {} bytes</p>\n",
                node.value.memory_usage()
            ));

            html.push_str("</div>\n");
        }

        html.push_str("</body>\n</html>\n");
        Ok(html)
    }
}

// From<std::fmt::Error> implementation is already provided in error.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugger_creation() {
        let debugger = GraphDebugger::new();
        assert_eq!(debugger.config.max_display_nodes, 50);
        assert_eq!(debugger.config.output_format, GraphOutputFormat::Dot);
    }

    #[test]
    fn test_config_serialization() {
        let config = DebuggerConfig::default();
        let serialized = serde_json::to_string(&config).expect("JSON serialization failed");
        let deserialized: DebuggerConfig =
            serde_json::from_str(&serialized).expect("JSON deserialization failed");

        assert_eq!(config.max_display_nodes, deserialized.max_display_nodes);
        assert_eq!(config.show_gradients, deserialized.show_gradients);
    }

    #[test]
    fn test_breakpoint_management() {
        let mut debugger = GraphDebugger::new();

        debugger.set_breakpoint(1);
        debugger.set_breakpoint(2);

        assert!(debugger.should_break(1));
        assert!(debugger.should_break(2));
        assert!(!debugger.should_break(3));

        debugger.remove_breakpoint(1);
        assert!(!debugger.should_break(1));
        assert!(debugger.should_break(2));
    }

    #[test]
    fn test_issue_severity() {
        assert!(matches!(IssueSeverity::Critical, IssueSeverity::Critical));
        assert!(matches!(IssueSeverity::Warning, IssueSeverity::Warning));
        assert!(matches!(IssueSeverity::Info, IssueSeverity::Info));
    }

    #[test]
    fn test_issue_types() {
        let issue = GraphIssue {
            issue_type: IssueType::VanishingGradients,
            nodes: vec![1, 2, 3],
            description: "Test issue".to_string(),
            severity: IssueSeverity::Warning,
            suggestion: Some("Test suggestion".to_string()),
        };

        assert_eq!(issue.issue_type, IssueType::VanishingGradients);
        assert_eq!(issue.nodes.len(), 3);
        assert!(issue.suggestion.is_some());
    }

    #[test]
    fn test_output_formats() {
        assert_eq!(GraphOutputFormat::Dot, GraphOutputFormat::Dot);
        assert_ne!(GraphOutputFormat::Dot, GraphOutputFormat::ASCII);
    }
}
