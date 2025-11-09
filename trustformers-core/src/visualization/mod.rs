//! Model architecture visualization and graph export
//!
//! This module provides tools to visualize transformer model architectures
//! and export them in various graph formats.

mod auto_visualizer;
mod tensor_viz;

pub use auto_visualizer::{AnalysisReport, AutoVisualizer, GraphAnalyzer};
pub use tensor_viz::*;

use crate::errors::Result;
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Supported export formats for model graphs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GraphFormat {
    /// DOT format for Graphviz
    Dot,
    /// Mermaid diagram format
    Mermaid,
    /// JSON representation
    Json,
    /// PlantUML format
    PlantUML,
    /// SVG (requires Graphviz)
    Svg,
}

/// Node in the model graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique identifier
    pub id: String,
    /// Node label
    pub label: String,
    /// Node type (e.g., "Linear", "Attention", "LayerNorm")
    pub node_type: String,
    /// Additional properties
    pub properties: HashMap<String, String>,
    /// Shape information
    pub input_shape: Option<Vec<usize>>,
    pub output_shape: Option<Vec<usize>>,
}

/// Edge in the model graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Edge label (optional)
    pub label: Option<String>,
    /// Tensor shape flowing through edge
    pub tensor_shape: Option<Vec<usize>>,
}

/// Model architecture graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGraph {
    /// Graph name
    pub name: String,
    /// Graph nodes
    pub nodes: Vec<GraphNode>,
    /// Graph edges
    pub edges: Vec<GraphEdge>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ModelGraph {
    /// Create a new model graph
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.push(node);
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.push(edge);
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Export graph to specified format
    pub fn export(&self, format: GraphFormat, output_path: &Path) -> Result<()> {
        let content = match format {
            GraphFormat::Dot => self.to_dot()?,
            GraphFormat::Mermaid => self.to_mermaid()?,
            GraphFormat::Json => self.to_json()?,
            GraphFormat::PlantUML => self.to_plantuml()?,
            GraphFormat::Svg => return self.to_svg(output_path),
        };

        let mut file = File::create(output_path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    /// Convert to DOT format
    pub fn to_dot(&self) -> Result<String> {
        let mut dot = String::new();
        writeln!(dot, "digraph {} {{", self.name)?;
        writeln!(dot, "  rankdir=TB;")?;
        writeln!(dot, "  node [shape=box, style=rounded];")?;

        // Add nodes
        for node in &self.nodes {
            let shape_info =
                if let (Some(input), Some(output)) = (&node.input_shape, &node.output_shape) {
                    format!("\\n{:?} → {:?}", input, output)
                } else {
                    String::new()
                };

            let color = match node.node_type.as_str() {
                "Attention" => "lightblue",
                "Linear" => "lightgreen",
                "LayerNorm" => "lightyellow",
                "Embedding" => "lightcoral",
                "Activation" => "lightgray",
                _ => "white",
            };

            writeln!(
                dot,
                "  {} [label=\"{}\\n{}{}\" fillcolor={} style=filled];",
                node.id, node.label, node.node_type, shape_info, color
            )?;
        }

        // Add edges
        for edge in &self.edges {
            let label = edge.label.as_deref().unwrap_or("");
            let shape_label = if let Some(shape) = &edge.tensor_shape {
                format!(" [{:?}]", shape)
            } else {
                String::new()
            };
            writeln!(
                dot,
                "  {} -> {} [label=\"{}{}\"];",
                edge.from, edge.to, label, shape_label
            )?;
        }

        writeln!(dot, "}}")?;
        Ok(dot)
    }

    /// Convert to Mermaid format
    pub fn to_mermaid(&self) -> Result<String> {
        let mut mermaid = String::new();
        writeln!(mermaid, "graph TB")?;

        // Add nodes
        for node in &self.nodes {
            let shape_syntax = match node.node_type.as_str() {
                "Attention" => ("[[", "]]"),
                "Linear" => ("[", "]"),
                "LayerNorm" => ("([", "])]"),
                "Embedding" => ("[/", "/]"),
                _ => ("[", "]"),
            };

            writeln!(
                mermaid,
                "    {}{}\"{}<br/>{}\"{}",
                node.id, shape_syntax.0, node.label, node.node_type, shape_syntax.1
            )?;
        }

        // Add edges
        for edge in &self.edges {
            let arrow = "-->";
            let label = edge.label.as_deref().unwrap_or("");
            writeln!(
                mermaid,
                "    {} {}|{}| {}",
                edge.from, arrow, label, edge.to
            )?;
        }

        // Add styling
        writeln!(mermaid, "\n    classDef attention fill:#add8e6")?;
        writeln!(mermaid, "    classDef linear fill:#90ee90")?;
        writeln!(mermaid, "    classDef norm fill:#ffffe0")?;

        // Apply styles
        for node in &self.nodes {
            match node.node_type.as_str() {
                "Attention" => writeln!(mermaid, "    class {} attention", node.id)?,
                "Linear" => writeln!(mermaid, "    class {} linear", node.id)?,
                "LayerNorm" => writeln!(mermaid, "    class {} norm", node.id)?,
                _ => {},
            }
        }

        Ok(mermaid)
    }

    /// Convert to JSON format
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Convert to PlantUML format
    pub fn to_plantuml(&self) -> Result<String> {
        let mut plantuml = String::new();
        writeln!(plantuml, "@startuml {}", self.name)?;
        writeln!(plantuml, "!theme plain")?;
        writeln!(plantuml, "skinparam componentStyle rectangle")?;

        // Add nodes
        for node in &self.nodes {
            let stereotype = match node.node_type.as_str() {
                "Attention" => "<<attention>>",
                "Linear" => "<<linear>>",
                "LayerNorm" => "<<norm>>",
                "Embedding" => "<<embed>>",
                _ => "",
            };

            writeln!(
                plantuml,
                "component \"{}\" as {} {}",
                node.label, node.id, stereotype
            )?;
        }

        // Add edges
        for edge in &self.edges {
            let label = edge.label.as_deref().unwrap_or("");
            writeln!(plantuml, "{} --> {} : {}", edge.from, edge.to, label)?;
        }

        writeln!(plantuml, "@enduml")?;
        Ok(plantuml)
    }

    /// Export to SVG using Graphviz
    pub fn to_svg(&self, output_path: &Path) -> Result<()> {
        let dot_content = self.to_dot()?;

        // Create temporary DOT file
        let dot_path = output_path.with_extension("dot");
        let mut dot_file = File::create(&dot_path)?;
        dot_file.write_all(dot_content.as_bytes())?;
        drop(dot_file);

        // Run Graphviz
        let output = std::process::Command::new("dot")
            .args(["-Tsvg", "-o"])
            .arg(output_path)
            .arg(&dot_path)
            .output()
            .map_err(|e| {
                anyhow!(
                    "Failed to run Graphviz: {}. Make sure 'dot' is installed.",
                    e
                )
            })?;

        // Clean up temporary file
        std::fs::remove_file(dot_path).ok();

        if !output.status.success() {
            return Err(anyhow!(
                "Graphviz failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        Ok(())
    }
}

/// Model architecture visualizer
pub struct ModelVisualizer {
    /// Include parameter counts
    pub include_params: bool,
    /// Include activation shapes
    pub include_shapes: bool,
    /// Include computation FLOPs
    pub include_flops: bool,
    /// Maximum depth to visualize
    pub max_depth: Option<usize>,
}

impl Default for ModelVisualizer {
    fn default() -> Self {
        Self {
            include_params: true,
            include_shapes: true,
            include_flops: false,
            max_depth: None,
        }
    }
}

impl ModelVisualizer {
    /// Create a new visualizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set parameter inclusion
    pub fn with_params(mut self, include: bool) -> Self {
        self.include_params = include;
        self
    }

    /// Set shape inclusion
    pub fn with_shapes(mut self, include: bool) -> Self {
        self.include_shapes = include;
        self
    }

    /// Set FLOPs inclusion
    pub fn with_flops(mut self, include: bool) -> Self {
        self.include_flops = include;
        self
    }

    /// Set maximum visualization depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Visualize a transformer model
    pub fn visualize_transformer(
        &self,
        model_name: &str,
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        vocab_size: usize,
    ) -> Result<ModelGraph> {
        let mut graph = ModelGraph::new(model_name);

        // Add metadata
        graph.add_metadata("model_type", "transformer");
        graph.add_metadata("num_layers", num_layers.to_string());
        graph.add_metadata("hidden_size", hidden_size.to_string());
        graph.add_metadata("num_heads", num_heads.to_string());

        // Input node
        graph.add_node(GraphNode {
            id: "input".to_string(),
            label: "Input Tokens".to_string(),
            node_type: "Input".to_string(),
            properties: HashMap::new(),
            input_shape: Some(vec![1, 512]), // batch_size, seq_len
            output_shape: Some(vec![1, 512]),
        });

        // Embedding
        let embed_params = vocab_size * hidden_size;
        graph.add_node(GraphNode {
            id: "embedding".to_string(),
            label: format!("Token Embedding\n{:.1}M params", embed_params as f32 / 1e6),
            node_type: "Embedding".to_string(),
            properties: HashMap::from([
                ("vocab_size".to_string(), vocab_size.to_string()),
                ("hidden_size".to_string(), hidden_size.to_string()),
            ]),
            input_shape: Some(vec![1, 512]),
            output_shape: Some(vec![1, 512, hidden_size]),
        });

        graph.add_edge(GraphEdge {
            from: "input".to_string(),
            to: "embedding".to_string(),
            label: None,
            tensor_shape: Some(vec![1, 512]),
        });

        // Add transformer layers
        let mut prev_node = "embedding".to_string();

        for i in 0..num_layers {
            let layer_prefix = format!("layer_{}", i);

            // Multi-head attention
            let attn_id = format!("{}_attention", layer_prefix);
            let attn_params = 4 * hidden_size * hidden_size; // Q, K, V, O projections
            graph.add_node(GraphNode {
                id: attn_id.clone(),
                label: format!(
                    "Multi-Head Attention\n{} heads\n{:.1}M params",
                    num_heads,
                    attn_params as f32 / 1e6
                ),
                node_type: "Attention".to_string(),
                properties: HashMap::from([
                    ("num_heads".to_string(), num_heads.to_string()),
                    (
                        "head_dim".to_string(),
                        (hidden_size / num_heads).to_string(),
                    ),
                ]),
                input_shape: Some(vec![1, 512, hidden_size]),
                output_shape: Some(vec![1, 512, hidden_size]),
            });

            // Add + Norm after attention
            let norm1_id = format!("{}_norm1", layer_prefix);
            graph.add_node(GraphNode {
                id: norm1_id.clone(),
                label: "Add & LayerNorm".to_string(),
                node_type: "LayerNorm".to_string(),
                properties: HashMap::new(),
                input_shape: Some(vec![1, 512, hidden_size]),
                output_shape: Some(vec![1, 512, hidden_size]),
            });

            // FFN
            let ffn_id = format!("{}_ffn", layer_prefix);
            let ffn_params = 2 * hidden_size * 4 * hidden_size; // Up and down projections
            graph.add_node(GraphNode {
                id: ffn_id.clone(),
                label: format!("Feed Forward\n{:.1}M params", ffn_params as f32 / 1e6),
                node_type: "Linear".to_string(),
                properties: HashMap::from([(
                    "intermediate_size".to_string(),
                    (4 * hidden_size).to_string(),
                )]),
                input_shape: Some(vec![1, 512, hidden_size]),
                output_shape: Some(vec![1, 512, hidden_size]),
            });

            // Add + Norm after FFN
            let norm2_id = format!("{}_norm2", layer_prefix);
            graph.add_node(GraphNode {
                id: norm2_id.clone(),
                label: "Add & LayerNorm".to_string(),
                node_type: "LayerNorm".to_string(),
                properties: HashMap::new(),
                input_shape: Some(vec![1, 512, hidden_size]),
                output_shape: Some(vec![1, 512, hidden_size]),
            });

            // Connect nodes
            graph.add_edge(GraphEdge {
                from: prev_node.clone(),
                to: attn_id.clone(),
                label: None,
                tensor_shape: Some(vec![1, 512, hidden_size]),
            });

            graph.add_edge(GraphEdge {
                from: attn_id,
                to: norm1_id.clone(),
                label: Some("residual".to_string()),
                tensor_shape: Some(vec![1, 512, hidden_size]),
            });

            graph.add_edge(GraphEdge {
                from: norm1_id.clone(),
                to: ffn_id.clone(),
                label: None,
                tensor_shape: Some(vec![1, 512, hidden_size]),
            });

            graph.add_edge(GraphEdge {
                from: ffn_id,
                to: norm2_id.clone(),
                label: Some("residual".to_string()),
                tensor_shape: Some(vec![1, 512, hidden_size]),
            });

            prev_node = norm2_id;
        }

        // Output layer
        graph.add_node(GraphNode {
            id: "output".to_string(),
            label: format!(
                "Output Projection\n{:.1}M params",
                (hidden_size * vocab_size) as f32 / 1e6
            ),
            node_type: "Linear".to_string(),
            properties: HashMap::new(),
            input_shape: Some(vec![1, 512, hidden_size]),
            output_shape: Some(vec![1, 512, vocab_size]),
        });

        graph.add_edge(GraphEdge {
            from: prev_node,
            to: "output".to_string(),
            label: None,
            tensor_shape: Some(vec![1, 512, hidden_size]),
        });

        Ok(graph)
    }

    /// Create a summary table of model architecture
    pub fn create_summary(&self, graph: &ModelGraph) -> Result<String> {
        let mut summary = String::new();

        writeln!(summary, "Model Architecture Summary")?;
        writeln!(summary, "==========================")?;
        writeln!(summary, "Model: {}", graph.name)?;

        // Count node types
        let mut node_counts: HashMap<String, usize> = HashMap::new();
        for node in &graph.nodes {
            *node_counts.entry(node.node_type.clone()).or_insert(0) += 1;
        }

        writeln!(summary, "\nLayer Types:")?;
        for (node_type, count) in node_counts {
            writeln!(summary, "  {node_type}: {count}")?;
        }

        // Calculate total parameters
        if self.include_params {
            let total_params = self.calculate_total_params(graph);
            writeln!(
                summary,
                "\nTotal Parameters: {:.2}M",
                total_params as f32 / 1e6
            )?;
        }

        // Show metadata
        if !graph.metadata.is_empty() {
            writeln!(summary, "\nMetadata:")?;
            for (key, value) in &graph.metadata {
                writeln!(summary, "  {key}: {value}")?;
            }
        }

        Ok(summary)
    }

    /// Calculate total parameters from graph
    fn calculate_total_params(&self, graph: &ModelGraph) -> usize {
        let mut total = 0;

        for node in &graph.nodes {
            if let Some(params_str) = node.properties.get("params") {
                if let Ok(params) = params_str.parse::<usize>() {
                    total += params;
                }
            }
        }

        total
    }
}

/// Interactive graph explorer (for terminal output)
pub struct GraphExplorer {
    graph: ModelGraph,
}

impl GraphExplorer {
    /// Create a new graph explorer
    pub fn new(graph: ModelGraph) -> Self {
        Self { graph }
    }

    /// Print graph as ASCII art
    pub fn print_ascii(&self) -> Result<()> {
        println!("Model: {}", self.graph.name);
        println!("{}", "=".repeat(50));

        // Create a simple top-to-bottom layout
        let mut visited = HashSet::new();
        let mut queue = vec!["input".to_string()];

        while let Some(node_id) = queue.pop() {
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id.clone());

            // Find node
            if let Some(node) = self.graph.nodes.iter().find(|n| n.id == node_id) {
                println!("\n┌─ {} ─┐", "─".repeat(node.label.len()));
                println!("│ {} │", node.label);
                println!("└─ {} ─┘", "─".repeat(node.label.len()));

                if let (Some(input), Some(output)) = (&node.input_shape, &node.output_shape) {
                    println!("  Shape: {input:?} → {output:?}");
                }
            }

            // Find outgoing edges
            for edge in &self.graph.edges {
                if edge.from == node_id {
                    println!("    ↓");
                    if let Some(label) = &edge.label {
                        println!("    {label}");
                    }
                    queue.push(edge.to.clone());
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let mut graph = ModelGraph::new("test_model");

        graph.add_node(GraphNode {
            id: "input".to_string(),
            label: "Input".to_string(),
            node_type: "Input".to_string(),
            properties: HashMap::new(),
            input_shape: Some(vec![1, 512]),
            output_shape: Some(vec![1, 512, 768]),
        });

        graph.add_node(GraphNode {
            id: "attention".to_string(),
            label: "Attention".to_string(),
            node_type: "Attention".to_string(),
            properties: HashMap::new(),
            input_shape: Some(vec![1, 512, 768]),
            output_shape: Some(vec![1, 512, 768]),
        });

        graph.add_edge(GraphEdge {
            from: "input".to_string(),
            to: "attention".to_string(),
            label: None,
            tensor_shape: Some(vec![1, 512, 768]),
        });

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn test_dot_export() {
        let graph = create_test_graph();
        let dot = graph.to_dot().unwrap();

        assert!(dot.contains("digraph test_model"));
        assert!(dot.contains("input -> attention"));
    }

    #[test]
    fn test_mermaid_export() {
        let graph = create_test_graph();
        let mermaid = graph.to_mermaid().unwrap();

        assert!(mermaid.contains("graph TB"));
        assert!(mermaid.contains("input -->"));
    }

    #[test]
    fn test_visualizer() {
        let visualizer = ModelVisualizer::new();
        let graph = visualizer.visualize_transformer("bert-base", 12, 768, 12, 30522).unwrap();

        // Check that all layers are created
        assert!(graph.nodes.len() > 12 * 4); // At least 4 nodes per layer
        assert!(graph.metadata.contains_key("num_layers"));
    }

    fn create_test_graph() -> ModelGraph {
        let mut graph = ModelGraph::new("test_model");

        graph.add_node(GraphNode {
            id: "input".to_string(),
            label: "Input".to_string(),
            node_type: "Input".to_string(),
            properties: HashMap::new(),
            input_shape: None,
            output_shape: None,
        });

        graph.add_node(GraphNode {
            id: "attention".to_string(),
            label: "Attention".to_string(),
            node_type: "Attention".to_string(),
            properties: HashMap::new(),
            input_shape: None,
            output_shape: None,
        });

        graph.add_edge(GraphEdge {
            from: "input".to_string(),
            to: "attention".to_string(),
            label: None,
            tensor_shape: None,
        });

        graph
    }
}
