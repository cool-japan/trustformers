//! Computation graph visualization tools
//!
//! This module provides tools to visualize the computation graph of neural networks,
//! including layer connections, tensor shapes, and operation flows.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Graph visualizer for computation graphs
#[derive(Debug)]
pub struct GraphVisualizer {
    /// Graph definition
    graph: ComputationGraph,
    /// Visualization config
    config: GraphVisualizerConfig,
}

/// Configuration for graph visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphVisualizerConfig {
    /// Include tensor shapes in visualization
    pub show_shapes: bool,
    /// Include data types in visualization
    pub show_dtypes: bool,
    /// Include operation attributes
    pub show_attributes: bool,
    /// Layout direction (TB=top-to-bottom, LR=left-to-right)
    pub layout_direction: LayoutDirection,
    /// Maximum depth to visualize (-1 for unlimited)
    pub max_depth: i32,
    /// Color scheme
    pub color_scheme: GraphColorScheme,
}

/// Layout direction for graph visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayoutDirection {
    /// Top to bottom
    TopToBottom,
    /// Left to right
    LeftToRight,
    /// Bottom to top
    BottomToTop,
    /// Right to left
    RightToLeft,
}

/// Color scheme for graph nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphColorScheme {
    /// Default colors
    Default,
    /// By layer type
    ByLayerType,
    /// By computational cost
    ByCost,
    /// By data flow
    ByDataFlow,
}

impl Default for GraphVisualizerConfig {
    fn default() -> Self {
        Self {
            show_shapes: true,
            show_dtypes: true,
            show_attributes: false,
            layout_direction: LayoutDirection::TopToBottom,
            max_depth: -1,
            color_scheme: GraphColorScheme::ByLayerType,
        }
    }
}

/// Computation graph definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationGraph {
    /// Graph name
    pub name: String,
    /// Graph nodes
    pub nodes: Vec<GraphNode>,
    /// Graph edges
    pub edges: Vec<GraphEdge>,
    /// Input nodes
    pub inputs: Vec<String>,
    /// Output nodes
    pub outputs: Vec<String>,
}

/// Graph node representing an operation or tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Node ID
    pub id: String,
    /// Node label (display name)
    pub label: String,
    /// Operation type
    pub op_type: String,
    /// Tensor shape (if applicable)
    pub shape: Option<Vec<i64>>,
    /// Data type
    pub dtype: Option<String>,
    /// Node attributes
    pub attributes: HashMap<String, String>,
    /// Node depth in the graph
    pub depth: usize,
}

/// Graph edge representing data flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Edge label (optional)
    pub label: Option<String>,
    /// Tensor shape along this edge
    pub shape: Option<Vec<i64>>,
}

impl GraphVisualizer {
    /// Create a new graph visualizer
    ///
    /// # Arguments
    ///
    /// * `graph_name` - Name of the computation graph
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::GraphVisualizer;
    ///
    /// let visualizer = GraphVisualizer::new("my_model");
    /// ```
    pub fn new(graph_name: &str) -> Self {
        let graph = ComputationGraph {
            name: graph_name.to_string(),
            nodes: Vec::new(),
            edges: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };

        Self {
            graph,
            config: GraphVisualizerConfig::default(),
        }
    }

    /// Create a graph visualizer with custom configuration
    pub fn with_config(graph_name: &str, config: GraphVisualizerConfig) -> Self {
        let graph = ComputationGraph {
            name: graph_name.to_string(),
            nodes: Vec::new(),
            edges: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };

        Self { graph, config }
    }

    /// Add a node to the graph
    ///
    /// # Example
    ///
    /// ```
    /// # use trustformers_debug::GraphVisualizer;
    /// # use std::collections::HashMap;
    /// # let mut visualizer = GraphVisualizer::new("model");
    /// visualizer.add_node(
    ///     "layer1",
    ///     "Linear Layer 1",
    ///     "Linear",
    ///     Some(vec![10, 20]),
    ///     Some("float32".to_string()),
    ///     HashMap::new(),
    /// );
    /// ```
    pub fn add_node(
        &mut self,
        id: &str,
        label: &str,
        op_type: &str,
        shape: Option<Vec<i64>>,
        dtype: Option<String>,
        attributes: HashMap<String, String>,
    ) {
        let node = GraphNode {
            id: id.to_string(),
            label: label.to_string(),
            op_type: op_type.to_string(),
            shape,
            dtype,
            attributes,
            depth: 0, // Will be computed later
        };

        self.graph.nodes.push(node);
    }

    /// Add an edge to the graph
    pub fn add_edge(
        &mut self,
        from: &str,
        to: &str,
        label: Option<String>,
        shape: Option<Vec<i64>>,
    ) {
        let edge = GraphEdge {
            from: from.to_string(),
            to: to.to_string(),
            label,
            shape,
        };

        self.graph.edges.push(edge);
    }

    /// Mark a node as an input
    pub fn mark_input(&mut self, node_id: &str) {
        if !self.graph.inputs.contains(&node_id.to_string()) {
            self.graph.inputs.push(node_id.to_string());
        }
    }

    /// Mark a node as an output
    pub fn mark_output(&mut self, node_id: &str) {
        if !self.graph.outputs.contains(&node_id.to_string()) {
            self.graph.outputs.push(node_id.to_string());
        }
    }

    /// Compute node depths in the graph
    fn compute_depths(&mut self) {
        // Build adjacency list
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        for edge in &self.graph.edges {
            adjacency
                .entry(edge.from.clone())
                .or_default()
                .push(edge.to.clone());
        }

        // BFS from input nodes
        let mut depths: HashMap<String, usize> = HashMap::new();
        let mut queue: Vec<(String, usize)> = Vec::new();

        for input in &self.graph.inputs {
            queue.push((input.clone(), 0));
            depths.insert(input.clone(), 0);
        }

        while let Some((node_id, depth)) = queue.pop() {
            if let Some(neighbors) = adjacency.get(&node_id) {
                for neighbor in neighbors {
                    let new_depth = depth + 1;
                    if !depths.contains_key(neighbor) || depths[neighbor] < new_depth {
                        depths.insert(neighbor.clone(), new_depth);
                        queue.push((neighbor.clone(), new_depth));
                    }
                }
            }
        }

        // Update node depths
        for node in &mut self.graph.nodes {
            node.depth = *depths.get(&node.id).unwrap_or(&0);
        }
    }

    /// Export graph to GraphViz DOT format
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use trustformers_debug::GraphVisualizer;
    /// # let mut visualizer = GraphVisualizer::new("model");
    /// visualizer.export_to_dot("model.dot").unwrap();
    /// ```
    pub fn export_to_dot<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.compute_depths();

        let mut dot = String::from("digraph {\n");

        // Graph attributes
        let direction = match self.config.layout_direction {
            LayoutDirection::TopToBottom => "TB",
            LayoutDirection::LeftToRight => "LR",
            LayoutDirection::BottomToTop => "BT",
            LayoutDirection::RightToLeft => "RL",
        };
        dot.push_str(&format!("  rankdir={};\n", direction));
        dot.push_str("  node [shape=box, style=rounded];\n\n");

        // Add nodes
        for node in &self.graph.nodes {
            if self.config.max_depth >= 0 && node.depth > self.config.max_depth as usize {
                continue;
            }

            let color = self.get_node_color(node);
            let mut label = node.label.to_string();

            if self.config.show_shapes {
                if let Some(ref shape) = node.shape {
                    label.push_str(&format!("\\nshape: {:?}", shape));
                }
            }

            if self.config.show_dtypes {
                if let Some(ref dtype) = node.dtype {
                    label.push_str(&format!("\\ndtype: {}", dtype));
                }
            }

            dot.push_str(&format!(
                "  \"{}\" [label=\"{}\", fillcolor=\"{}\", style=\"filled,rounded\"];\n",
                node.id, label, color
            ));
        }

        dot.push('\n');

        // Add edges
        for edge in &self.graph.edges {
            let mut edge_label = String::new();

            if let Some(ref label) = edge.label {
                edge_label = label.clone();
            } else if self.config.show_shapes {
                if let Some(ref shape) = edge.shape {
                    edge_label = format!("{:?}", shape);
                }
            }

            if !edge_label.is_empty() {
                dot.push_str(&format!(
                    "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
                    edge.from, edge.to, edge_label
                ));
            } else {
                dot.push_str(&format!("  \"{}\" -> \"{}\";\n", edge.from, edge.to));
            }
        }

        dot.push_str("}\n");

        fs::write(path, dot)?;
        Ok(())
    }

    /// Get color for a node based on color scheme
    fn get_node_color(&self, node: &GraphNode) -> &'static str {
        match self.config.color_scheme {
            GraphColorScheme::Default => "lightblue",
            GraphColorScheme::ByLayerType => match node.op_type.as_str() {
                "Linear" | "Dense" => "lightblue",
                "Conv2d" | "Conv1d" => "lightgreen",
                "BatchNorm" | "LayerNorm" => "lightyellow",
                "ReLU" | "GELU" | "Softmax" => "lightcoral",
                "Dropout" => "lightgray",
                "Attention" | "MultiHeadAttention" => "plum",
                _ => "white",
            },
            GraphColorScheme::ByCost => {
                // Simplified: use depth as proxy for computational cost
                if node.depth > 10 {
                    "darkred"
                } else if node.depth > 5 {
                    "orange"
                } else {
                    "lightgreen"
                }
            },
            GraphColorScheme::ByDataFlow => {
                if self.graph.inputs.contains(&node.id) {
                    "lightgreen"
                } else if self.graph.outputs.contains(&node.id) {
                    "lightcoral"
                } else {
                    "lightblue"
                }
            },
        }
    }

    /// Export graph to JSON format
    pub fn export_to_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.graph)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Get statistics about the graph
    pub fn statistics(&self) -> GraphStatistics {
        let num_nodes = self.graph.nodes.len();
        let num_edges = self.graph.edges.len();

        let op_type_counts: HashMap<String, usize> =
            self.graph.nodes.iter().fold(HashMap::new(), |mut acc, node| {
                *acc.entry(node.op_type.clone()).or_insert(0) += 1;
                acc
            });

        let max_depth = self.graph.nodes.iter().map(|n| n.depth).max().unwrap_or(0);

        GraphStatistics {
            num_nodes,
            num_edges,
            num_inputs: self.graph.inputs.len(),
            num_outputs: self.graph.outputs.len(),
            max_depth,
            op_type_counts,
        }
    }

    /// Print a summary of the graph
    pub fn summary(&self) -> String {
        let stats = self.statistics();

        let mut output = String::new();
        output.push_str(&format!("Computation Graph: {}\n", self.graph.name));
        output.push_str(&"=".repeat(60));
        output.push('\n');
        output.push_str(&format!("Nodes: {}\n", stats.num_nodes));
        output.push_str(&format!("Edges: {}\n", stats.num_edges));
        output.push_str(&format!("Inputs: {}\n", stats.num_inputs));
        output.push_str(&format!("Outputs: {}\n", stats.num_outputs));
        output.push_str(&format!("Max Depth: {}\n", stats.max_depth));

        output.push_str("\nOperation Types:\n");
        for (op_type, count) in &stats.op_type_counts {
            output.push_str(&format!("  {}: {}\n", op_type, count));
        }

        output
    }
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Number of input nodes
    pub num_inputs: usize,
    /// Number of output nodes
    pub num_outputs: usize,
    /// Maximum depth
    pub max_depth: usize,
    /// Operation type counts
    pub op_type_counts: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_graph_visualizer_creation() {
        let visualizer = GraphVisualizer::new("test_graph");
        assert_eq!(visualizer.graph.name, "test_graph");
        assert_eq!(visualizer.graph.nodes.len(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut visualizer = GraphVisualizer::new("test");

        visualizer.add_node(
            "node1",
            "Layer 1",
            "Linear",
            Some(vec![10, 20]),
            Some("float32".to_string()),
            HashMap::new(),
        );

        assert_eq!(visualizer.graph.nodes.len(), 1);
        assert_eq!(visualizer.graph.nodes[0].id, "node1");
    }

    #[test]
    fn test_add_edge() {
        let mut visualizer = GraphVisualizer::new("test");

        visualizer.add_node("node1", "N1", "Linear", None, None, HashMap::new());
        visualizer.add_node("node2", "N2", "ReLU", None, None, HashMap::new());
        visualizer.add_edge("node1", "node2", None, Some(vec![10, 20]));

        assert_eq!(visualizer.graph.edges.len(), 1);
        assert_eq!(visualizer.graph.edges[0].from, "node1");
        assert_eq!(visualizer.graph.edges[0].to, "node2");
    }

    #[test]
    fn test_mark_input_output() {
        let mut visualizer = GraphVisualizer::new("test");

        visualizer.add_node("input", "Input", "Input", None, None, HashMap::new());
        visualizer.add_node("output", "Output", "Output", None, None, HashMap::new());

        visualizer.mark_input("input");
        visualizer.mark_output("output");

        assert_eq!(visualizer.graph.inputs.len(), 1);
        assert_eq!(visualizer.graph.outputs.len(), 1);
    }

    #[test]
    fn test_export_to_dot() {
        let temp_dir = env::temp_dir();
        let output_path = temp_dir.join("test_graph.dot");

        let mut visualizer = GraphVisualizer::new("test");

        visualizer.add_node("input", "Input", "Input", None, None, HashMap::new());
        visualizer.add_node(
            "layer1",
            "Linear",
            "Linear",
            Some(vec![10, 20]),
            None,
            HashMap::new(),
        );
        visualizer.add_edge("input", "layer1", None, None);

        visualizer.mark_input("input");

        visualizer.export_to_dot(&output_path).unwrap();
        assert!(output_path.exists());

        // Clean up
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn test_export_to_json() {
        let temp_dir = env::temp_dir();
        let output_path = temp_dir.join("test_graph.json");

        let mut visualizer = GraphVisualizer::new("test");
        visualizer.add_node("node1", "N1", "Linear", None, None, HashMap::new());

        visualizer.export_to_json(&output_path).unwrap();
        assert!(output_path.exists());

        // Clean up
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn test_statistics() {
        let mut visualizer = GraphVisualizer::new("test");

        visualizer.add_node("n1", "N1", "Linear", None, None, HashMap::new());
        visualizer.add_node("n2", "N2", "Linear", None, None, HashMap::new());
        visualizer.add_node("n3", "N3", "ReLU", None, None, HashMap::new());

        visualizer.add_edge("n1", "n2", None, None);
        visualizer.add_edge("n2", "n3", None, None);

        visualizer.mark_input("n1");
        visualizer.mark_output("n3");

        let stats = visualizer.statistics();

        assert_eq!(stats.num_nodes, 3);
        assert_eq!(stats.num_edges, 2);
        assert_eq!(stats.num_inputs, 1);
        assert_eq!(stats.num_outputs, 1);
    }

    #[test]
    fn test_summary() {
        let mut visualizer = GraphVisualizer::new("test_model");

        visualizer.add_node("input", "Input", "Input", None, None, HashMap::new());
        visualizer.add_node("layer1", "Linear", "Linear", None, None, HashMap::new());

        let summary = visualizer.summary();
        assert!(summary.contains("test_model"));
        assert!(summary.contains("Nodes: 2"));
    }

    #[test]
    fn test_compute_depths() {
        let mut visualizer = GraphVisualizer::new("test");

        visualizer.add_node("input", "Input", "Input", None, None, HashMap::new());
        visualizer.add_node("layer1", "L1", "Linear", None, None, HashMap::new());
        visualizer.add_node("layer2", "L2", "ReLU", None, None, HashMap::new());

        visualizer.add_edge("input", "layer1", None, None);
        visualizer.add_edge("layer1", "layer2", None, None);

        visualizer.mark_input("input");

        visualizer.compute_depths();

        assert_eq!(visualizer.graph.nodes[0].depth, 0);
        assert_eq!(visualizer.graph.nodes[1].depth, 1);
        assert_eq!(visualizer.graph.nodes[2].depth, 2);
    }
}
