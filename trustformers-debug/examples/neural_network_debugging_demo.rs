//! Neural Network Debugging Demo
#![allow(unused_variables)]
//!
//! This example demonstrates the advanced neural network debugging capabilities
//! for transformer models and attention mechanisms in TrustformeRS Debug.

use anyhow::Result;
use scirs2_core::ndarray::{Array2, ArrayD};
use trustformers_debug::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 TrustformeRS Neural Network Debugging Demo");
    println!("==============================================");

    // Demo 1: Basic attention debugging with convenience macro
    demo_attention_debugging().await?;

    // Demo 2: Full transformer debugging
    demo_transformer_debugging().await?;

    // Demo 3: Integration with main debugging session
    demo_integrated_debugging().await?;

    println!("✅ Neural network debugging demo completed successfully!");
    Ok(())
}

/// Demonstrate attention mechanism debugging
async fn demo_attention_debugging() -> Result<()> {
    println!("\n📊 Demo 1: Attention Mechanism Debugging");
    println!("----------------------------------------");

    // Create sample attention weights for a single layer with 8 heads
    let mut attention_weights = Vec::new();
    for head in 0..8 {
        // Create a 64x64 attention matrix with different patterns for each head
        let mut weights = Array2::<f32>::zeros((64, 64));

        match head {
            0 | 1 => {
                // Local attention pattern (diagonal)
                for i in 0..64 {
                    for j in 0..64 {
                        let distance = (i as i32 - j as i32).abs();
                        if distance <= 3 {
                            weights[[i, j]] = 1.0 / (1.0 + distance as f32);
                        }
                    }
                }
            },
            2 | 3 => {
                // Long-range attention pattern
                for i in 0..64 {
                    for j in 0..64 {
                        let distance = (i as i32 - j as i32).abs();
                        if distance > 16 {
                            weights[[i, j]] = 0.8 / (1.0 + (distance as f32 / 10.0));
                        }
                    }
                }
            },
            4 | 5 => {
                // Block-structured attention
                let block_size = 16;
                for i in 0..64 {
                    for j in 0..64 {
                        if (i / block_size) == (j / block_size) {
                            weights[[i, j]] = 0.9;
                        }
                    }
                }
            },
            _ => {
                // Sparse attention pattern
                for i in 0..64 {
                    for j in 0..64 {
                        if (i + j) % 8 == 0 {
                            weights[[i, j]] = 0.7;
                        }
                    }
                }
            },
        }

        // Normalize attention weights (softmax-like)
        let row_sums: Vec<f32> = weights.rows().into_iter().map(|row| row.sum()).collect();

        for i in 0..64 {
            let sum = row_sums[i];
            if sum > 0.0 {
                for j in 0..64 {
                    weights[[i, j]] /= sum;
                }
            }
        }

        // Convert to ArrayD for the debugger
        let weights_d = weights.into_dyn();
        attention_weights.push(weights_d);
    }

    // Create attention debugger
    let mut debugger = AttentionDebugger::new(AttentionDebugConfig::default());

    // Analyze the attention layer
    let layer_analysis = debugger.analyze_attention_layer(0, &attention_weights)?;

    println!("Layer Analysis Results:");
    println!("  • Layer Index: {}", layer_analysis.layer_index);
    println!("  • Number of Heads: {}", layer_analysis.num_heads);
    println!(
        "  • Layer Diversity Score: {:.3}",
        layer_analysis.layer_diversity_score
    );
    println!(
        "  • Overall Redundancy Score: {:.3}",
        layer_analysis.redundancy_analysis.overall_redundancy_score
    );

    println!("\nHead Specializations:");
    for (i, head_analysis) in layer_analysis.head_analyses.iter().enumerate() {
        println!(
            "  • Head {}: {:?} (Importance: {:.3})",
            i, head_analysis.specialization_type, head_analysis.importance_score
        );
    }

    println!("\nAttention Patterns Detected:");
    for (i, attention_map) in layer_analysis.attention_maps.iter().enumerate() {
        println!(
            "  • Head {}: {:?} (Entropy: {:.3}, Sparsity: {:.3})",
            i,
            attention_map.attention_pattern,
            attention_map.attention_entropy,
            attention_map.sparsity_ratio
        );
    }

    // Using the convenience macro
    println!("\nUsing debug_attention! macro:");
    match debug_attention!(&attention_weights) {
        Ok(analysis) => {
            println!(
                "  • Quick analysis completed for {} heads",
                analysis.num_heads
            );
        },
        Err(e) => println!("  • Error in quick analysis: {}", e),
    }

    Ok(())
}

/// Demonstrate full transformer debugging
async fn demo_transformer_debugging() -> Result<()> {
    println!("\n🔍 Demo 2: Full Transformer Debugging");
    println!("-------------------------------------");

    // Create sample attention weights for multiple layers (simulating a 6-layer transformer)
    let mut model_attention_weights = Vec::new();

    for layer in 0..6 {
        let mut layer_weights = Vec::new();

        // Each layer has 12 attention heads
        for head in 0..12 {
            let mut weights = Array2::<f32>::zeros((32, 32)); // Smaller for demo

            // Create different attention patterns across layers
            match layer {
                0 | 1 => {
                    // Early layers: more local attention
                    for i in 0..32 {
                        for j in 0..32 {
                            let distance = (i as i32 - j as i32).abs();
                            if distance <= 2 {
                                weights[[i, j]] = 1.0 / (1.0 + distance as f32);
                            }
                        }
                    }
                },
                2 | 3 => {
                    // Middle layers: mixed patterns
                    for i in 0..32 {
                        for j in 0..32 {
                            let distance = (i as i32 - j as i32).abs();
                            if head < 6 {
                                // Half heads do local attention
                                if distance <= 3 {
                                    weights[[i, j]] = 0.8 / (1.0 + distance as f32);
                                }
                            } else {
                                // Half heads do long-range attention
                                if distance > 8 {
                                    weights[[i, j]] = 0.6 / (1.0 + (distance as f32 / 5.0));
                                }
                            }
                        }
                    }
                },
                _ => {
                    // Later layers: more global attention
                    for i in 0..32 {
                        for j in 0..32 {
                            weights[[i, j]] = 0.5 + 0.3 * ((i * j) as f32 / 1024.0).sin();
                        }
                    }
                },
            }

            // Normalize
            for i in 0..32 {
                let sum: f32 = (0..32).map(|j| weights[[i, j]]).sum();
                if sum > 0.0 {
                    for j in 0..32 {
                        weights[[i, j]] /= sum;
                    }
                }
            }

            layer_weights.push(weights.into_dyn());
        }

        model_attention_weights.push(layer_weights);
    }

    // Create transformer debugger
    let mut transformer_debugger = TransformerDebugger::new(TransformerDebugConfig::default());

    // Analyze the entire transformer
    let transformer_analysis =
        transformer_debugger.analyze_transformer_attention(&model_attention_weights)?;

    println!("Transformer Analysis Results:");
    println!("  • Total Layers: {}", transformer_analysis.num_layers);
    println!(
        "  • Model Attention Health: {:?}",
        transformer_analysis.model_attention_summary.model_attention_health
    );
    println!(
        "  • Total Heads: {}",
        transformer_analysis.model_attention_summary.total_heads
    );
    println!(
        "  • Average Diversity Score: {:.3}",
        transformer_analysis.model_attention_summary.average_diversity_score
    );
    println!(
        "  • Average Redundancy Score: {:.3}",
        transformer_analysis.model_attention_summary.average_redundancy_score
    );

    if let Some(cross_layer) = &transformer_analysis.cross_layer_analysis {
        println!("\nCross-Layer Analysis:");
        println!(
            "  • Attention Evolution: {:?}",
            cross_layer.attention_evolution.evolution_type
        );
        println!(
            "  • Head Consistency Score: {:.3}",
            cross_layer.head_consistency.consistency_score
        );

        println!("  • Entropy Trend Across Layers:");
        for (layer, entropy) in cross_layer.attention_evolution.entropy_trend.iter().enumerate() {
            println!("    - Layer {}: {:.3}", layer, entropy);
        }

        println!("  • Dominant Pattern Sequence:");
        for (layer, pattern) in
            cross_layer.pattern_progression.dominant_pattern_sequence.iter().enumerate()
        {
            println!("    - Layer {}: {:?}", layer, pattern);
        }
    }

    // Using the convenience macro
    println!("\nUsing debug_transformer! macro:");
    match debug_transformer!(&model_attention_weights) {
        Ok(analysis) => {
            println!(
                "  • Quick transformer analysis completed for {} layers",
                analysis.num_layers
            );
        },
        Err(e) => println!("  • Error in quick transformer analysis: {}", e),
    }

    Ok(())
}

/// Demonstrate integration with main debugging session
async fn demo_integrated_debugging() -> Result<()> {
    println!("\n🔧 Demo 3: Integrated Debugging Session");
    println!("--------------------------------------");

    // Create a debug session with transformer debugging enabled
    let mut session = debug_session_with_transformer();

    println!("Created debug session with transformer debugging enabled");

    // Start the session
    session.start().await?;
    println!("Debug session started");

    // Check if transformer debugger is available
    if let Some(transformer_debugger) = session.transformer_debugger() {
        println!("✅ Transformer debugger is available and configured");

        // You can now access the transformer debugger through the session
        println!(
            "  • Max layers to analyze: {}",
            transformer_debugger.config.max_layers_to_analyze
        );
        println!(
            "  • Max heads to analyze: {}",
            transformer_debugger.config.attention_config.max_heads_to_analyze
        );
        println!(
            "  • Cross-layer analysis enabled: {}",
            transformer_debugger.config.enable_cross_layer_analysis
        );
    } else {
        println!("❌ Transformer debugger not available");
    }

    // For demonstration, let's create some sample data and analyze it
    let sample_layer_weights = vec![
        Array2::<f32>::eye(16).into_dyn(),
        Array2::<f32>::zeros((16, 16)).into_dyn(),
    ];

    if let Some(transformer_debugger) = session.transformer_debugger_mut() {
        let sample_model_weights = vec![sample_layer_weights];

        match transformer_debugger.analyze_transformer_attention(&sample_model_weights) {
            Ok(analysis) => {
                println!("✅ Successfully analyzed sample transformer model");
                println!(
                    "  • Health status: {:?}",
                    analysis.model_attention_summary.model_attention_health
                );
            },
            Err(e) => {
                println!("❌ Error analyzing sample model: {}", e);
            },
        }
    }

    // Stop the session and generate report
    let report = session.stop().await?;
    println!("Debug session completed");

    // The report now includes all debugging components
    let summary = report.summary();
    println!("Session Summary:");
    println!("  • Session ID: {}", summary.session_id);
    println!("  • Total Issues: {}", summary.total_issues);
    println!("  • Critical Issues: {}", summary.critical_issues);

    Ok(())
}

/// Helper function to create sample attention weights with specific patterns
#[allow(dead_code)]
fn create_sample_attention_weights(seq_len: usize, pattern: &str) -> ArrayD<f32> {
    let mut weights = Array2::<f32>::zeros((seq_len, seq_len));

    match pattern {
        "diagonal" => {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let distance = (i as i32 - j as i32).abs();
                    if distance <= 2 {
                        weights[[i, j]] = 1.0 / (1.0 + distance as f32);
                    }
                }
            }
        },
        "uniform" => {
            weights.fill(1.0 / seq_len as f32);
        },
        "sparse" => {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if (i + j) % 4 == 0 {
                        weights[[i, j]] = 1.0;
                    }
                }
            }
        },
        _ => {
            // Random pattern
            for i in 0..seq_len {
                for j in 0..seq_len {
                    weights[[i, j]] = ((i * j) as f32).sin().abs();
                }
            }
        },
    }

    // Normalize rows
    for mut row in weights.rows_mut() {
        let sum: f32 = row.sum();
        if sum > 0.0 {
            row /= sum;
        }
    }

    weights.into_dyn()
}
