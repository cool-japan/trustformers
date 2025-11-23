//! Comprehensive demonstration of new visualization features
//!
//! This example demonstrates:
//! - TensorBoard integration
//! - Netron/ONNX export
//! - Activation visualization
//! - Attention pattern visualization
//! - Numerical stability checking
//! - Computation graph visualization

use anyhow::Result;
use std::collections::HashMap;
use std::env;

use trustformers_debug::{
    ActivationVisualizer, AttentionType, AttentionVisualizer, ExportFormat, GraphVisualizer,
    NetronExporter, StabilityChecker, TensorBoardWriter,
};

fn main() -> Result<()> {
    println!("=== TrustformeRS Debug Visualization Demo ===\n");

    let temp_dir = env::temp_dir().join("trustformers_debug_demo");
    std::fs::create_dir_all(&temp_dir)?;

    // =========================================================================
    // 1. TensorBoard Integration Demo
    // =========================================================================
    println!("1. TensorBoard Integration");
    println!("--------------------------");

    let tensorboard_dir = temp_dir.join("tensorboard_logs");
    let mut writer = TensorBoardWriter::new(&tensorboard_dir)?;

    // Log scalars (e.g., training metrics)
    for step in 0..100 {
        let loss = 1.0 - (step as f64 / 100.0);
        let accuracy = step as f64 / 100.0;

        writer.add_scalar("loss/train", loss, step)?;
        writer.add_scalar("accuracy/train", accuracy, step)?;
    }

    // Log histograms (e.g., weight distributions)
    let weights: Vec<f64> = (0..1000).map(|i| (i as f64 / 1000.0) * 2.0 - 1.0).collect();
    writer.add_histogram("layer.0.weight", &weights, 50)?;

    // Log text
    writer.add_text(
        "experiment/description",
        "Testing new visualization features",
        0,
    )?;

    // Flush to disk
    writer.flush()?;
    println!("✓ TensorBoard logs written to: {:?}", tensorboard_dir);
    println!(
        "  Run: tensorboard --logdir={}\n",
        tensorboard_dir.display()
    );

    // =========================================================================
    // 2. Netron/ONNX Export Demo
    // =========================================================================
    println!("2. Netron/ONNX Export");
    println!("---------------------");

    let mut exporter = NetronExporter::new("bert-tiny", "Tiny BERT model for demonstration")
        .with_format(ExportFormat::Json);

    exporter.set_author("TrustformeRS");
    exporter.set_version("0.1.0");
    exporter.add_property("framework", "TrustformeRS");

    // Add inputs
    exporter.add_input("input_ids", "int64", vec![1, 128]);
    exporter.add_input("attention_mask", "int64", vec![1, 128]);

    // Add outputs
    exporter.add_output("logits", "float32", vec![1, 128, 30522]);

    // Add layers
    let mut attrs = HashMap::new();
    attrs.insert(
        "vocab_size".to_string(),
        trustformers_debug::AttributeValue::Int(30522),
    );
    attrs.insert(
        "hidden_size".to_string(),
        trustformers_debug::AttributeValue::Int(768),
    );

    exporter.add_node(
        "embeddings",
        "Embedding",
        vec!["input_ids".to_string()],
        vec!["hidden_states".to_string()],
        attrs.clone(),
    );

    // Add attention layer
    attrs.clear();
    attrs.insert(
        "num_heads".to_string(),
        trustformers_debug::AttributeValue::Int(12),
    );
    attrs.insert(
        "head_dim".to_string(),
        trustformers_debug::AttributeValue::Int(64),
    );

    exporter.add_node(
        "layer.0.attention",
        "MultiHeadAttention",
        vec!["hidden_states".to_string()],
        vec!["attention_output".to_string()],
        attrs,
    );

    let netron_path = temp_dir.join("model.json");
    exporter.export(&netron_path)?;
    println!("✓ Model exported to: {:?}", netron_path);
    println!("  View at: https://netron.app/\n");

    // =========================================================================
    // 3. Activation Visualization Demo
    // =========================================================================
    println!("3. Activation Visualization");
    println!("---------------------------");

    let mut act_viz = ActivationVisualizer::new();

    // Simulate activations from different layers
    let layer1_activations: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) * 2.0 - 1.0).collect();
    act_viz.register("layer1.relu", layer1_activations, vec![1, 1000])?;

    let layer2_activations: Vec<f32> = (0..1000)
        .map(|i| {
            let x = i as f32 / 1000.0;
            (x * 6.28).sin()
        })
        .collect();
    act_viz.register("layer2.gelu", layer2_activations, vec![1, 1000])?;

    // Print summary
    println!("{}", act_viz.print_summary()?);

    // Create ASCII histogram
    let histogram = act_viz.plot_distribution_ascii("layer1.relu")?;
    println!("{}", histogram);

    // Export statistics
    let stats_path = temp_dir.join("activation_stats.json");
    act_viz.export_statistics("layer1.relu", &stats_path)?;
    println!("✓ Activation statistics exported to: {:?}\n", stats_path);

    // =========================================================================
    // 4. Attention Visualization Demo
    // =========================================================================
    println!("4. Attention Visualization");
    println!("--------------------------");

    let mut att_viz = AttentionVisualizer::new();

    // Simulate attention weights for a simple sentence
    let tokens = ["The", "cat", "sat", "on", "the", "mat"];
    let num_tokens = tokens.len();

    // Create attention weights (2 heads)
    let mut weights = Vec::new();

    // Head 0: Strong diagonal (self-attention)
    let mut head0 = Vec::new();
    for i in 0..num_tokens {
        let mut row = vec![0.1; num_tokens];
        row[i] = 0.6; // Strong self-attention
        head0.push(row);
    }
    weights.push(head0);

    // Head 1: Attending to first and last tokens
    let mut head1 = Vec::new();
    for _i in 0..num_tokens {
        let mut row = vec![0.15; num_tokens];
        row[0] = 0.3; // Attend to first token
        row[num_tokens - 1] = 0.3; // Attend to last token
        head1.push(row);
    }
    weights.push(head1);

    let token_strings: Vec<String> = tokens.iter().map(|&s| s.to_string()).collect();

    att_viz.register(
        "layer.0.attention",
        weights,
        token_strings.clone(),
        token_strings,
        AttentionType::SelfAttention,
    )?;

    // Print summary
    println!("{}", att_viz.summary());

    // Analyze patterns
    let analysis = att_viz.analyze("layer.0.attention")?;
    println!("Entropy per head: {:?}", analysis.entropy_per_head);
    println!("Sparsity per head: {:?}", analysis.sparsity_per_head);

    // Plot heatmap (ASCII)
    let heatmap = att_viz.plot_heatmap_ascii("layer.0.attention", 0)?;
    println!("\n{}", heatmap);

    // Export to BertViz format
    let bertviz_path = temp_dir.join("attention.html");
    att_viz.export_to_bertviz("layer.0.attention", &bertviz_path)?;
    println!("✓ BertViz HTML exported to: {:?}\n", bertviz_path);

    // =========================================================================
    // 5. Numerical Stability Checking Demo
    // =========================================================================
    println!("5. Numerical Stability Checking");
    println!("--------------------------------");

    let mut checker = StabilityChecker::new();

    // Check various tensor values
    let normal_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    checker.check_tensor("layer1.output", &normal_values)?;
    println!("✓ Normal values: No issues detected");

    // Check problematic values
    let problematic_values = vec![
        1.0,
        f64::NAN,
        2.0,
        f64::INFINITY,
        1e-20,
        1e20,
        f64::NEG_INFINITY,
    ];
    checker.check_tensor("layer2.output", &problematic_values)?;

    // Print report
    println!("\n{}", checker.report());

    // Export issues
    let issues_path = temp_dir.join("stability_issues.json");
    checker.export_to_json(&issues_path)?;
    println!("✓ Stability issues exported to: {:?}\n", issues_path);

    // =========================================================================
    // 6. Computation Graph Visualization Demo
    // =========================================================================
    println!("6. Computation Graph Visualization");
    println!("-----------------------------------");

    let mut graph_viz = GraphVisualizer::new("simple_transformer");

    // Add nodes
    graph_viz.add_node(
        "input",
        "Input",
        "Input",
        Some(vec![1, 128]),
        Some("int64".to_string()),
        HashMap::new(),
    );

    graph_viz.add_node(
        "embeddings",
        "Embeddings",
        "Embedding",
        Some(vec![1, 128, 768]),
        Some("float32".to_string()),
        HashMap::new(),
    );

    let mut attn_attrs = HashMap::new();
    attn_attrs.insert("num_heads".to_string(), "12".to_string());

    graph_viz.add_node(
        "attention",
        "Self-Attention",
        "Attention",
        Some(vec![1, 128, 768]),
        Some("float32".to_string()),
        attn_attrs,
    );

    graph_viz.add_node(
        "ffn",
        "Feed Forward",
        "Linear",
        Some(vec![1, 128, 768]),
        Some("float32".to_string()),
        HashMap::new(),
    );

    graph_viz.add_node(
        "output",
        "Output",
        "Output",
        Some(vec![1, 128, 30522]),
        Some("float32".to_string()),
        HashMap::new(),
    );

    // Add edges
    graph_viz.add_edge("input", "embeddings", None, Some(vec![1, 128]));
    graph_viz.add_edge("embeddings", "attention", None, Some(vec![1, 128, 768]));
    graph_viz.add_edge("attention", "ffn", None, Some(vec![1, 128, 768]));
    graph_viz.add_edge("ffn", "output", None, Some(vec![1, 128, 768]));

    // Mark inputs and outputs
    graph_viz.mark_input("input");
    graph_viz.mark_output("output");

    // Print summary
    println!("{}", graph_viz.summary());

    // Export to GraphViz DOT
    let dot_path = temp_dir.join("graph.dot");
    graph_viz.export_to_dot(&dot_path)?;
    println!("✓ Graph exported to: {:?}", dot_path);
    println!(
        "  Convert to PNG: dot -Tpng {} -o graph.png",
        dot_path.display()
    );

    // Export to JSON
    let graph_json_path = temp_dir.join("graph.json");
    graph_viz.export_to_json(&graph_json_path)?;
    println!("✓ Graph JSON exported to: {:?}\n", graph_json_path);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=== Demo Complete ===");
    println!("\nAll artifacts saved to: {:?}", temp_dir);
    println!("\nNext steps:");
    println!(
        "1. View TensorBoard: tensorboard --logdir={}",
        tensorboard_dir.display()
    );
    println!(
        "2. View Netron export: Open {} at https://netron.app/",
        netron_path.display()
    );
    println!(
        "3. View attention patterns: Open {} in a browser",
        bertviz_path.display()
    );
    println!(
        "4. Convert graph to image: dot -Tpng {} -o graph.png",
        dot_path.display()
    );

    Ok(())
}
