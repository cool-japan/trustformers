//! Demonstration of model architecture visualization
#![allow(unused_variables)]

use anyhow::Result;
use std::path::Path;
use trustformers_core::visualization::{
    AutoVisualizer, GraphAnalyzer, GraphExplorer, GraphFormat, ModelGraph, ModelVisualizer,
};

fn main() -> Result<()> {
    println!("TrustformeRS Model Architecture Visualizer Demo");
    println!("==============================================\n");

    // Example 1: Manual graph creation
    println!("Example 1: Manual Model Graph Creation");
    demo_manual_graph()?;

    // Example 2: Automatic transformer visualization
    println!("\nExample 2: Automatic Transformer Visualization");
    demo_transformer_visualization()?;

    // Example 3: Auto-visualize BERT
    println!("\nExample 3: Auto-Visualize BERT Architecture");
    demo_bert_visualization()?;

    // Example 4: Auto-visualize GPT
    println!("\nExample 4: Auto-Visualize GPT Architecture");
    demo_gpt_visualization()?;

    // Example 5: Graph analysis
    println!("\nExample 5: Model Graph Analysis");
    demo_graph_analysis()?;

    // Example 6: Export formats
    println!("\nExample 6: Export to Different Formats");
    demo_export_formats()?;

    Ok(())
}

fn demo_manual_graph() -> Result<()> {
    use trustformers_core::visualization::{GraphEdge, GraphNode};

    let mut graph = ModelGraph::new("Simple Transformer Block");

    // Add nodes
    graph.add_node(GraphNode {
        id: "input".to_string(),
        label: "Input".to_string(),
        node_type: "Input".to_string(),
        properties: std::collections::HashMap::new(),
        input_shape: Some(vec![1, 512, 768]),
        output_shape: Some(vec![1, 512, 768]),
    });

    graph.add_node(GraphNode {
        id: "attention".to_string(),
        label: "Multi-Head Attention\n12 heads".to_string(),
        node_type: "Attention".to_string(),
        properties: std::collections::HashMap::from([
            ("num_heads".to_string(), "12".to_string()),
            ("hidden_size".to_string(), "768".to_string()),
        ]),
        input_shape: Some(vec![1, 512, 768]),
        output_shape: Some(vec![1, 512, 768]),
    });

    graph.add_node(GraphNode {
        id: "ffn".to_string(),
        label: "Feed Forward Network".to_string(),
        node_type: "FeedForward".to_string(),
        properties: std::collections::HashMap::from([(
            "intermediate_size".to_string(),
            "3072".to_string(),
        )]),
        input_shape: Some(vec![1, 512, 768]),
        output_shape: Some(vec![1, 512, 768]),
    });

    // Add edges
    graph.add_edge(GraphEdge {
        from: "input".to_string(),
        to: "attention".to_string(),
        label: None,
        tensor_shape: Some(vec![1, 512, 768]),
    });

    graph.add_edge(GraphEdge {
        from: "attention".to_string(),
        to: "ffn".to_string(),
        label: Some("+ residual".to_string()),
        tensor_shape: Some(vec![1, 512, 768]),
    });

    // Print as ASCII
    let explorer = GraphExplorer::new(graph.clone());
    explorer.print_ascii()?;

    println!("✓ Manual graph created successfully");
    Ok(())
}

fn demo_transformer_visualization() -> Result<()> {
    let visualizer = ModelVisualizer::new().with_params(true).with_shapes(true);

    // Visualize a BERT-base like model
    let graph = visualizer.visualize_transformer(
        "BERT-Base",
        12,    // layers
        768,   // hidden_size
        12,    // num_heads
        30522, // vocab_size
    )?;

    // Create summary
    let summary = visualizer.create_summary(&graph)?;
    println!("{}", summary);

    // Export to temporary file
    let temp_path = Path::new("/tmp/bert_base_architecture.dot");
    graph.export(GraphFormat::Dot, temp_path)?;
    println!("✓ Exported to: {:?}", temp_path);

    Ok(())
}

fn demo_bert_visualization() -> Result<()> {
    let mut visualizer = AutoVisualizer::new();
    let graph = visualizer.visualize_bert_model(12)?;

    println!("BERT Model Architecture:");
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Edges: {}", graph.edges.len());

    // Show some key components
    for node in graph.nodes.iter().take(5) {
        println!("  - {}: {} ({})", node.id, node.label, node.node_type);
    }
    println!("  ... and {} more nodes", graph.nodes.len() - 5);

    // Export as Mermaid diagram
    let mermaid = graph.to_mermaid()?;
    println!("\nMermaid diagram preview:");
    println!("{}", &mermaid[..200.min(mermaid.len())]);
    println!("...");

    Ok(())
}

fn demo_gpt_visualization() -> Result<()> {
    let mut visualizer = AutoVisualizer::new();
    let graph = visualizer.visualize_gpt_model(12)?;

    println!("GPT Model Architecture:");
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Edges: {}", graph.edges.len());

    // Show metadata
    println!("\nMetadata:");
    for (key, value) in &graph.metadata {
        println!("  {}: {}", key, value);
    }

    // Export as PlantUML
    let plantuml = graph.to_plantuml()?;
    let plantuml_path = Path::new("/tmp/gpt_architecture.puml");
    graph.export(GraphFormat::PlantUML, plantuml_path)?;
    println!("\n✓ Exported PlantUML to: {:?}", plantuml_path);

    Ok(())
}

fn demo_graph_analysis() -> Result<()> {
    // Create a model graph
    let mut visualizer = AutoVisualizer::new();
    let graph = visualizer.visualize_bert_model(24)?; // BERT-Large

    // Analyze the graph
    let report = GraphAnalyzer::analyze(&graph);

    println!("Model Analysis Report:");
    println!("=====================");
    println!(
        "Total Parameters: {:.1}M",
        report.total_parameters as f32 / 1_000_000.0
    );

    println!("\nLayer Distribution:");
    for (layer_type, count) in &report.layer_distribution {
        println!("  {}: {}", layer_type, count);
    }

    if !report.bottlenecks.is_empty() {
        println!("\nPotential Bottlenecks:");
        for bottleneck in &report.bottlenecks {
            println!("  ⚠️  {}", bottleneck);
        }
    }

    if !report.optimization_suggestions.is_empty() {
        println!("\nOptimization Suggestions:");
        for suggestion in &report.optimization_suggestions {
            println!("  💡 {}", suggestion);
        }
    }

    Ok(())
}

fn demo_export_formats() -> Result<()> {
    let visualizer = ModelVisualizer::new();
    let graph = visualizer.visualize_transformer(
        "MiniTransformer",
        2,    // layers
        256,  // hidden_size
        4,    // num_heads
        1000, // vocab_size
    )?;

    // Export to different formats
    let formats = vec![
        (GraphFormat::Dot, "/tmp/model.dot", "Graphviz DOT"),
        (GraphFormat::Mermaid, "/tmp/model.mmd", "Mermaid"),
        (GraphFormat::Json, "/tmp/model.json", "JSON"),
        (GraphFormat::PlantUML, "/tmp/model.puml", "PlantUML"),
    ];

    println!("Exporting to multiple formats:");
    for (format, path, name) in formats {
        let path = Path::new(path);
        match graph.export(format, path) {
            Ok(_) => println!("  ✓ {} -> {:?}", name, path),
            Err(e) => println!("  ✗ {} failed: {}", name, e),
        }
    }

    // Try SVG export (requires Graphviz)
    println!("\nAttempting SVG export (requires Graphviz):");
    let svg_path = Path::new("/tmp/model.svg");
    match graph.export(GraphFormat::Svg, svg_path) {
        Ok(_) => println!("  ✓ SVG exported successfully to {:?}", svg_path),
        Err(e) => println!("  ℹ️  SVG export failed: {}", e),
    }

    // Show JSON content
    let json = graph.to_json()?;
    println!("\nJSON representation (first 200 chars):");
    println!("{}", &json[..200.min(json.len())]);
    println!("...");

    Ok(())
}

// Example: Custom model visualization
#[allow(dead_code)]
fn visualize_custom_model() -> Result<()> {
    use trustformers_core::visualization::GraphNode;

    println!("\n=== Bonus: Custom Model Visualization ===");

    let mut graph = ModelGraph::new("Vision Transformer (ViT)");

    // Patch embedding
    graph.add_node(GraphNode {
        id: "patch_embed".to_string(),
        label: "Patch Embedding\n16x16 patches".to_string(),
        node_type: "PatchEmbed".to_string(),
        properties: std::collections::HashMap::from([
            ("patch_size".to_string(), "16".to_string()),
            ("embed_dim".to_string(), "768".to_string()),
        ]),
        input_shape: Some(vec![1, 3, 224, 224]),
        output_shape: Some(vec![1, 196, 768]),
    });

    // Position embedding
    graph.add_node(GraphNode {
        id: "pos_embed".to_string(),
        label: "Position Embedding".to_string(),
        node_type: "Embedding".to_string(),
        properties: std::collections::HashMap::new(),
        input_shape: Some(vec![1, 196]),
        output_shape: Some(vec![1, 196, 768]),
    });

    // Class token
    graph.add_node(GraphNode {
        id: "cls_token".to_string(),
        label: "[CLS] Token".to_string(),
        node_type: "Parameter".to_string(),
        properties: std::collections::HashMap::new(),
        input_shape: None,
        output_shape: Some(vec![1, 1, 768]),
    });

    // Transformer encoder
    for i in 0..12 {
        let encoder_id = format!("encoder_{}", i);
        graph.add_node(GraphNode {
            id: encoder_id.clone(),
            label: format!("Transformer Block {}", i),
            node_type: "TransformerBlock".to_string(),
            properties: std::collections::HashMap::from([
                ("num_heads".to_string(), "12".to_string()),
                ("mlp_ratio".to_string(), "4".to_string()),
            ]),
            input_shape: Some(vec![1, 197, 768]),
            output_shape: Some(vec![1, 197, 768]),
        });
    }

    // Classification head
    graph.add_node(GraphNode {
        id: "classifier".to_string(),
        label: "Classification Head\n1000 classes".to_string(),
        node_type: "Linear".to_string(),
        properties: std::collections::HashMap::from([(
            "num_classes".to_string(),
            "1000".to_string(),
        )]),
        input_shape: Some(vec![1, 768]),
        output_shape: Some(vec![1, 1000]),
    });

    println!("✓ Vision Transformer graph created");
    println!("  Total nodes: {}", graph.nodes.len());

    Ok(())
}
