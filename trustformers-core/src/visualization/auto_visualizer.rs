//! Automatic model visualization from existing models

use super::{GraphEdge, GraphNode, ModelGraph, ModelVisualizer};
use crate::errors::Result;
use crate::layers::{Embedding, FeedForward, LayerNorm, Linear, MultiHeadAttention};
use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Automatic model graph builder
pub struct AutoVisualizer {
    #[allow(dead_code)] // Reserved for future graph visualization features
    visualizer: ModelVisualizer,
    node_counter: usize,
    #[allow(dead_code)] // Reserved for future tensor flow tracking
    /// Track tensor flow between layers
    tensor_flow: HashMap<String, Vec<usize>>,
}

impl Default for AutoVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoVisualizer {
    /// Create a new auto visualizer
    pub fn new() -> Self {
        Self {
            visualizer: ModelVisualizer::default(),
            node_counter: 0,
            tensor_flow: HashMap::new(),
        }
    }

    /// Generate unique node ID
    fn next_node_id(&mut self, prefix: &str) -> String {
        let id = format!("{}_{}", prefix, self.node_counter);
        self.node_counter += 1;
        id
    }

    /// Detect layer type from trait object
    #[allow(dead_code)] // Reserved for future layer analysis features
    fn detect_layer_type(layer: &dyn Any) -> &'static str {
        let type_id = layer.type_id();

        if type_id == TypeId::of::<Linear>() {
            "Linear"
        } else if type_id == TypeId::of::<MultiHeadAttention>() {
            "Attention"
        } else if type_id == TypeId::of::<LayerNorm>() {
            "LayerNorm"
        } else if type_id == TypeId::of::<Embedding>() {
            "Embedding"
        } else if type_id == TypeId::of::<FeedForward>() {
            "FeedForward"
        } else {
            "Layer"
        }
    }

    /// Extract layer parameters count
    #[allow(dead_code)] // Reserved for future parameter counting features
    fn count_parameters<L>(layer: &L) -> usize
    where
        L: crate::traits::Layer,
    {
        // Simplified placeholder - in practice would use layer's actual parameter count
        // Cannot downcast generic layer, so return default estimate
        let _ = layer; // Avoid unused parameter warning
        768 * 768 // Default estimate
    }

    /// Visualize a BERT-like model
    pub fn visualize_bert_model(&mut self, num_layers: usize) -> Result<ModelGraph> {
        let mut graph = ModelGraph::new("BERT Model");

        // Input node
        let input_id = self.next_node_id("input");
        graph.add_node(GraphNode {
            id: input_id.clone(),
            label: "Input IDs".to_string(),
            node_type: "Input".to_string(),
            properties: HashMap::new(),
            input_shape: Some(vec![1, 512]),
            output_shape: Some(vec![1, 512]),
        });

        // Token embeddings
        let token_embed_id = self.next_node_id("token_embed");
        graph.add_node(GraphNode {
            id: token_embed_id.clone(),
            label: "Token Embeddings\n23.4M params".to_string(),
            node_type: "Embedding".to_string(),
            properties: HashMap::from([
                ("vocab_size".to_string(), "30522".to_string()),
                ("hidden_size".to_string(), "768".to_string()),
            ]),
            input_shape: Some(vec![1, 512]),
            output_shape: Some(vec![1, 512, 768]),
        });

        // Position embeddings
        let pos_embed_id = self.next_node_id("pos_embed");
        graph.add_node(GraphNode {
            id: pos_embed_id.clone(),
            label: "Position Embeddings\n0.4M params".to_string(),
            node_type: "Embedding".to_string(),
            properties: HashMap::from([
                ("max_positions".to_string(), "512".to_string()),
                ("hidden_size".to_string(), "768".to_string()),
            ]),
            input_shape: Some(vec![1, 512]),
            output_shape: Some(vec![1, 512, 768]),
        });

        // Embedding sum
        let embed_sum_id = self.next_node_id("embed_sum");
        graph.add_node(GraphNode {
            id: embed_sum_id.clone(),
            label: "Embedding Sum + LayerNorm".to_string(),
            node_type: "LayerNorm".to_string(),
            properties: HashMap::new(),
            input_shape: Some(vec![1, 512, 768]),
            output_shape: Some(vec![1, 512, 768]),
        });

        // Connect input to embeddings
        graph.add_edge(GraphEdge {
            from: input_id.clone(),
            to: token_embed_id.clone(),
            label: None,
            tensor_shape: Some(vec![1, 512]),
        });

        graph.add_edge(GraphEdge {
            from: input_id,
            to: pos_embed_id.clone(),
            label: None,
            tensor_shape: Some(vec![1, 512]),
        });

        graph.add_edge(GraphEdge {
            from: token_embed_id,
            to: embed_sum_id.clone(),
            label: Some("add".to_string()),
            tensor_shape: Some(vec![1, 512, 768]),
        });

        graph.add_edge(GraphEdge {
            from: pos_embed_id,
            to: embed_sum_id.clone(),
            label: Some("add".to_string()),
            tensor_shape: Some(vec![1, 512, 768]),
        });

        // Add transformer layers
        let mut prev_id = embed_sum_id;

        for i in 0..num_layers {
            // Self-attention
            let attn_id = self.next_node_id(&format!("layer{}_attn", i));
            graph.add_node(GraphNode {
                id: attn_id.clone(),
                label: format!("Layer {} Self-Attention\n12 heads, 2.4M params", i),
                node_type: "Attention".to_string(),
                properties: HashMap::from([
                    ("num_heads".to_string(), "12".to_string()),
                    ("hidden_size".to_string(), "768".to_string()),
                ]),
                input_shape: Some(vec![1, 512, 768]),
                output_shape: Some(vec![1, 512, 768]),
            });

            let attn_norm_id = self.next_node_id(&format!("layer{}_attn_norm", i));
            graph.add_node(GraphNode {
                id: attn_norm_id.clone(),
                label: "Residual + LayerNorm".to_string(),
                node_type: "LayerNorm".to_string(),
                properties: HashMap::new(),
                input_shape: Some(vec![1, 512, 768]),
                output_shape: Some(vec![1, 512, 768]),
            });

            // FFN
            let ffn_id = self.next_node_id(&format!("layer{}_ffn", i));
            graph.add_node(GraphNode {
                id: ffn_id.clone(),
                label: format!("Layer {} FFN\n4.7M params", i),
                node_type: "FeedForward".to_string(),
                properties: HashMap::from([("intermediate_size".to_string(), "3072".to_string())]),
                input_shape: Some(vec![1, 512, 768]),
                output_shape: Some(vec![1, 512, 768]),
            });

            let ffn_norm_id = self.next_node_id(&format!("layer{}_ffn_norm", i));
            graph.add_node(GraphNode {
                id: ffn_norm_id.clone(),
                label: "Residual + LayerNorm".to_string(),
                node_type: "LayerNorm".to_string(),
                properties: HashMap::new(),
                input_shape: Some(vec![1, 512, 768]),
                output_shape: Some(vec![1, 512, 768]),
            });

            // Connect layer components
            graph.add_edge(GraphEdge {
                from: prev_id.clone(),
                to: attn_id.clone(),
                label: None,
                tensor_shape: Some(vec![1, 512, 768]),
            });

            graph.add_edge(GraphEdge {
                from: attn_id.clone(),
                to: attn_norm_id.clone(),
                label: None,
                tensor_shape: Some(vec![1, 512, 768]),
            });

            graph.add_edge(GraphEdge {
                from: prev_id,
                to: attn_norm_id.clone(),
                label: Some("residual".to_string()),
                tensor_shape: Some(vec![1, 512, 768]),
            });

            graph.add_edge(GraphEdge {
                from: attn_norm_id.clone(),
                to: ffn_id.clone(),
                label: None,
                tensor_shape: Some(vec![1, 512, 768]),
            });

            graph.add_edge(GraphEdge {
                from: ffn_id.clone(),
                to: ffn_norm_id.clone(),
                label: None,
                tensor_shape: Some(vec![1, 512, 768]),
            });

            graph.add_edge(GraphEdge {
                from: attn_norm_id,
                to: ffn_norm_id.clone(),
                label: Some("residual".to_string()),
                tensor_shape: Some(vec![1, 512, 768]),
            });

            prev_id = ffn_norm_id;
        }

        // Pooler
        let pooler_id = self.next_node_id("pooler");
        graph.add_node(GraphNode {
            id: pooler_id.clone(),
            label: "Pooler\n0.6M params".to_string(),
            node_type: "Linear".to_string(),
            properties: HashMap::from([("output_size".to_string(), "768".to_string())]),
            input_shape: Some(vec![1, 768]),
            output_shape: Some(vec![1, 768]),
        });

        graph.add_edge(GraphEdge {
            from: prev_id,
            to: pooler_id,
            label: Some("[CLS] token".to_string()),
            tensor_shape: Some(vec![1, 768]),
        });

        // Add metadata
        graph.add_metadata("total_params", "110M");
        graph.add_metadata("architecture", "BERT");
        graph.add_metadata("num_layers", num_layers.to_string());

        Ok(graph)
    }

    /// Visualize a GPT-like model
    pub fn visualize_gpt_model(&mut self, num_layers: usize) -> Result<ModelGraph> {
        let mut graph = ModelGraph::new("GPT Model");

        // Input
        let input_id = self.next_node_id("input");
        graph.add_node(GraphNode {
            id: input_id.clone(),
            label: "Input IDs".to_string(),
            node_type: "Input".to_string(),
            properties: HashMap::new(),
            input_shape: Some(vec![1, 1024]),
            output_shape: Some(vec![1, 1024]),
        });

        // Token + Position embeddings
        let embed_id = self.next_node_id("embeddings");
        graph.add_node(GraphNode {
            id: embed_id.clone(),
            label: "Token + Position Embeddings\n38.6M params".to_string(),
            node_type: "Embedding".to_string(),
            properties: HashMap::from([
                ("vocab_size".to_string(), "50257".to_string()),
                ("hidden_size".to_string(), "768".to_string()),
                ("max_positions".to_string(), "1024".to_string()),
            ]),
            input_shape: Some(vec![1, 1024]),
            output_shape: Some(vec![1, 1024, 768]),
        });

        graph.add_edge(GraphEdge {
            from: input_id,
            to: embed_id.clone(),
            label: None,
            tensor_shape: Some(vec![1, 1024]),
        });

        let mut prev_id = embed_id;

        // Transformer layers
        for i in 0..num_layers {
            // Layer norm before attention
            let ln1_id = self.next_node_id(&format!("layer{}_ln1", i));
            graph.add_node(GraphNode {
                id: ln1_id.clone(),
                label: "LayerNorm".to_string(),
                node_type: "LayerNorm".to_string(),
                properties: HashMap::new(),
                input_shape: Some(vec![1, 1024, 768]),
                output_shape: Some(vec![1, 1024, 768]),
            });

            // Causal self-attention
            let attn_id = self.next_node_id(&format!("layer{}_attn", i));
            graph.add_node(GraphNode {
                id: attn_id.clone(),
                label: format!("Layer {} Causal Attention\n12 heads, 2.4M params", i),
                node_type: "Attention".to_string(),
                properties: HashMap::from([
                    ("num_heads".to_string(), "12".to_string()),
                    ("causal".to_string(), "true".to_string()),
                ]),
                input_shape: Some(vec![1, 1024, 768]),
                output_shape: Some(vec![1, 1024, 768]),
            });

            // Residual after attention
            let res1_id = self.next_node_id(&format!("layer{}_res1", i));
            graph.add_node(GraphNode {
                id: res1_id.clone(),
                label: "Residual Add".to_string(),
                node_type: "Add".to_string(),
                properties: HashMap::new(),
                input_shape: Some(vec![1, 1024, 768]),
                output_shape: Some(vec![1, 1024, 768]),
            });

            // Layer norm before FFN
            let ln2_id = self.next_node_id(&format!("layer{}_ln2", i));
            graph.add_node(GraphNode {
                id: ln2_id.clone(),
                label: "LayerNorm".to_string(),
                node_type: "LayerNorm".to_string(),
                properties: HashMap::new(),
                input_shape: Some(vec![1, 1024, 768]),
                output_shape: Some(vec![1, 1024, 768]),
            });

            // FFN
            let ffn_id = self.next_node_id(&format!("layer{}_ffn", i));
            graph.add_node(GraphNode {
                id: ffn_id.clone(),
                label: format!("Layer {} FFN\n4.7M params", i),
                node_type: "FeedForward".to_string(),
                properties: HashMap::from([
                    ("intermediate_size".to_string(), "3072".to_string()),
                    ("activation".to_string(), "gelu".to_string()),
                ]),
                input_shape: Some(vec![1, 1024, 768]),
                output_shape: Some(vec![1, 1024, 768]),
            });

            // Residual after FFN
            let res2_id = self.next_node_id(&format!("layer{}_res2", i));
            graph.add_node(GraphNode {
                id: res2_id.clone(),
                label: "Residual Add".to_string(),
                node_type: "Add".to_string(),
                properties: HashMap::new(),
                input_shape: Some(vec![1, 1024, 768]),
                output_shape: Some(vec![1, 1024, 768]),
            });

            // Connect nodes
            graph.add_edge(GraphEdge {
                from: prev_id.clone(),
                to: ln1_id.clone(),
                label: None,
                tensor_shape: Some(vec![1, 1024, 768]),
            });

            graph.add_edge(GraphEdge {
                from: ln1_id,
                to: attn_id.clone(),
                label: None,
                tensor_shape: Some(vec![1, 1024, 768]),
            });

            graph.add_edge(GraphEdge {
                from: attn_id,
                to: res1_id.clone(),
                label: None,
                tensor_shape: Some(vec![1, 1024, 768]),
            });

            graph.add_edge(GraphEdge {
                from: prev_id,
                to: res1_id.clone(),
                label: Some("residual".to_string()),
                tensor_shape: Some(vec![1, 1024, 768]),
            });

            graph.add_edge(GraphEdge {
                from: res1_id.clone(),
                to: ln2_id.clone(),
                label: None,
                tensor_shape: Some(vec![1, 1024, 768]),
            });

            graph.add_edge(GraphEdge {
                from: ln2_id,
                to: ffn_id.clone(),
                label: None,
                tensor_shape: Some(vec![1, 1024, 768]),
            });

            graph.add_edge(GraphEdge {
                from: ffn_id,
                to: res2_id.clone(),
                label: None,
                tensor_shape: Some(vec![1, 1024, 768]),
            });

            graph.add_edge(GraphEdge {
                from: res1_id,
                to: res2_id.clone(),
                label: Some("residual".to_string()),
                tensor_shape: Some(vec![1, 1024, 768]),
            });

            prev_id = res2_id;
        }

        // Final layer norm
        let final_ln_id = self.next_node_id("final_ln");
        graph.add_node(GraphNode {
            id: final_ln_id.clone(),
            label: "Final LayerNorm".to_string(),
            node_type: "LayerNorm".to_string(),
            properties: HashMap::new(),
            input_shape: Some(vec![1, 1024, 768]),
            output_shape: Some(vec![1, 1024, 768]),
        });

        // Language modeling head
        let lm_head_id = self.next_node_id("lm_head");
        graph.add_node(GraphNode {
            id: lm_head_id.clone(),
            label: "LM Head\n38.6M params".to_string(),
            node_type: "Linear".to_string(),
            properties: HashMap::from([("vocab_size".to_string(), "50257".to_string())]),
            input_shape: Some(vec![1, 1024, 768]),
            output_shape: Some(vec![1, 1024, 50257]),
        });

        graph.add_edge(GraphEdge {
            from: prev_id,
            to: final_ln_id.clone(),
            label: None,
            tensor_shape: Some(vec![1, 1024, 768]),
        });

        graph.add_edge(GraphEdge {
            from: final_ln_id,
            to: lm_head_id,
            label: None,
            tensor_shape: Some(vec![1, 1024, 768]),
        });

        // Metadata
        graph.add_metadata("total_params", "124M");
        graph.add_metadata("architecture", "GPT");
        graph.add_metadata("num_layers", num_layers.to_string());

        Ok(graph)
    }
}

/// Model graph analyzer
pub struct GraphAnalyzer;

impl GraphAnalyzer {
    /// Analyze graph for optimization opportunities
    pub fn analyze(graph: &ModelGraph) -> AnalysisReport {
        // Count parameters
        let total_parameters = Self::count_total_parameters(graph);

        // Analyze layer distribution
        let mut layer_counts = HashMap::new();
        for node in &graph.nodes {
            *layer_counts.entry(node.node_type.clone()).or_insert(0) += 1;
        }

        // Find bottlenecks
        let bottlenecks = Self::find_bottlenecks(graph);

        // Suggest optimizations
        let optimization_suggestions = Self::suggest_optimizations(graph);

        AnalysisReport {
            total_parameters,
            layer_distribution: layer_counts,
            bottlenecks,
            optimization_suggestions,
        }
    }

    fn count_total_parameters(graph: &ModelGraph) -> usize {
        let mut total = 0;
        // Static regex pattern - always compiles successfully
        let params_regex = regex::Regex::new(r"(\d+\.?\d*)M params")
            .expect("Static regex pattern is always valid");
        for node in &graph.nodes {
            if let Some(params_str) = node.properties.get("params") {
                if let Ok(params) = params_str.parse::<usize>() {
                    total += params;
                }
            } else {
                // Estimate from label if available
                if let Some(captures) = params_regex.captures(&node.label) {
                    if let Ok(millions) = captures[1].parse::<f32>() {
                        total += (millions * 1_000_000.0) as usize;
                    }
                }
            }
        }
        total
    }

    fn find_bottlenecks(graph: &ModelGraph) -> Vec<String> {
        let mut bottlenecks = Vec::new();

        // Find nodes with shape mismatches
        for edge in &graph.edges {
            if let (Some(from_node), Some(to_node)) = (
                graph.nodes.iter().find(|n| n.id == edge.from),
                graph.nodes.iter().find(|n| n.id == edge.to),
            ) {
                if let (Some(out_shape), Some(in_shape)) =
                    (&from_node.output_shape, &to_node.input_shape)
                {
                    if out_shape != in_shape {
                        bottlenecks.push(format!(
                            "Shape mismatch: {} -> {} ({:?} != {:?})",
                            from_node.label, to_node.label, out_shape, in_shape
                        ));
                    }
                }
            }
        }

        // Find layers with excessive parameters
        for node in &graph.nodes {
            if let Some(params_str) = node.properties.get("params") {
                if let Ok(params) = params_str.parse::<usize>() {
                    if params > 10_000_000 {
                        bottlenecks.push(format!(
                            "{} has {:.1}M parameters (consider factorization)",
                            node.label,
                            params as f32 / 1_000_000.0
                        ));
                    }
                }
            }
        }

        bottlenecks
    }

    fn suggest_optimizations(graph: &ModelGraph) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Check for attention layers
        let attention_count = graph.nodes.iter().filter(|n| n.node_type == "Attention").count();

        if attention_count > 8 {
            suggestions
                .push("Consider using sparse attention patterns for long sequences".to_string());
        }

        // Check for large embeddings
        for node in &graph.nodes {
            if node.node_type == "Embedding" {
                if let Some(vocab_str) = node.properties.get("vocab_size") {
                    if let Ok(vocab_size) = vocab_str.parse::<usize>() {
                        if vocab_size > 50000 {
                            suggestions.push(format!(
                                "Large vocabulary ({} tokens) - consider vocabulary pruning or hash embeddings",
                                vocab_size
                            ));
                        }
                    }
                }
            }
        }

        // Check for repeated patterns
        let layer_types: Vec<_> = graph.nodes.iter().map(|n| n.node_type.as_str()).collect();

        if layer_types.windows(4).any(|w| w[0] == w[2] && w[1] == w[3]) {
            suggestions
                .push("Repeated layer patterns detected - consider weight sharing".to_string());
        }

        suggestions
    }
}

/// Analysis report from graph analyzer
#[derive(Debug, Default)]
pub struct AnalysisReport {
    pub total_parameters: usize,
    pub layer_distribution: HashMap<String, usize>,
    pub bottlenecks: Vec<String>,
    pub optimization_suggestions: Vec<String>,
}
