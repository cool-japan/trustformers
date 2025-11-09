//! Demonstration of the performance optimization advisor
#![allow(unused_variables)]

use anyhow::Result;
use std::collections::HashMap;
use trustformers_core::performance::{
    AnalysisContext, HardwareInfo, LatencyMetrics, MemoryMetrics, OptimizationAdvisor,
    OptimizationCategory, ProfileResult, ThroughputMetrics,
};
use trustformers_core::visualization::AutoVisualizer;
use trustformers_core::TrustformersError;

fn main() -> Result<()> {
    println!("TrustformeRS Performance Optimization Advisor Demo");
    println!("================================================\n");

    // Example 1: Basic optimization analysis
    println!("Example 1: Basic Optimization Analysis");
    demo_basic_analysis()?;

    // Example 2: Analysis with model graph
    println!("\nExample 2: Analysis with Model Graph");
    demo_model_graph_analysis()?;

    // Example 3: Analysis with profiling data
    println!("\nExample 3: Analysis with Profiling Data");
    demo_profiling_analysis()?;

    // Example 4: Memory-focused optimization
    println!("\nExample 4: Memory-Focused Optimization");
    demo_memory_optimization()?;

    // Example 5: Hardware-specific optimization
    println!("\nExample 5: Hardware-Specific Optimization");
    demo_hardware_optimization()?;

    // Example 6: Generate full report
    println!("\nExample 6: Full Optimization Report");
    demo_full_report()?;

    Ok(())
}

fn demo_basic_analysis() -> Result<()> {
    let advisor = OptimizationAdvisor::new();

    // Create basic context
    let context = AnalysisContext {
        model_graph: None,
        profile_results: None,
        latency_metrics: None,
        memory_metrics: None,
        throughput_metrics: None,
        hardware_info: HardwareInfo::default(),
        current_config: HashMap::new(),
    };

    let report = advisor.analyze(&context)?;

    println!(
        "Found {} optimization suggestions",
        report.suggestions.len()
    );
    for suggestion in report.suggestions.iter().take(3) {
        println!(
            "- {} ({}, {}): {}",
            suggestion.title, suggestion.impact, suggestion.difficulty, suggestion.description
        );
    }

    Ok(())
}

fn demo_model_graph_analysis() -> Result<()> {
    let advisor = OptimizationAdvisor::new();

    // Create a model graph using auto-visualizer
    let mut visualizer = AutoVisualizer::new();
    let graph = visualizer.visualize_bert_model(24)?; // BERT-Large

    // Create context with model graph
    let context = AnalysisContext {
        model_graph: Some(graph),
        profile_results: None,
        latency_metrics: None,
        memory_metrics: None,
        throughput_metrics: None,
        hardware_info: HardwareInfo {
            cpu_cores: 16,
            gpu_model: Some("NVIDIA A100".to_string()),
            gpu_memory_mb: Some(40960),
            system_memory_mb: 64000,
            ..Default::default()
        },
        current_config: HashMap::from([
            ("mode".to_string(), "inference".to_string()),
            ("batch_size".to_string(), "1".to_string()),
        ]),
    };

    let report = advisor.analyze(&context)?;

    println!("\nModel-based optimizations:");
    for suggestion in report.suggestions_by_category(OptimizationCategory::Architecture) {
        println!("- {}: {}", suggestion.title, suggestion.description);
        if let Some(improvement) = suggestion.expected_improvement.latency_reduction {
            println!("  Expected latency reduction: {:.1}%", improvement);
        }
    }

    Ok(())
}

fn demo_profiling_analysis() -> Result<()> {
    let advisor = OptimizationAdvisor::new();

    // Create mock profiling data
    let mut profile = ProfileResult::new("inference".to_string());

    // Simulate many small operations
    for i in 0..20 {
        let op_name = format!("small_op_{}", i);
        profile.record_operation_start(&op_name);
        std::thread::sleep(std::time::Duration::from_micros(50));
        profile.record_operation_end(&op_name);
    }

    // Simulate attention operations
    for i in 0..12 {
        let op_name = format!("attention_layer_{}", i);
        profile.record_operation_start(&op_name);
        std::thread::sleep(std::time::Duration::from_millis(5));
        profile.record_operation_end(&op_name);
    }

    let context = AnalysisContext {
        model_graph: None,
        profile_results: Some(profile),
        latency_metrics: Some(LatencyMetrics {
            count: 1000,
            mean_ms: 50.0,
            median_ms: 45.0,
            std_dev_ms: 15.0,
            min_ms: 30.0,
            max_ms: 120.0,
            p50_ms: 45.0,
            p90_ms: 65.0,
            p95_ms: 70.0,
            p99_ms: 90.0,
            p999_ms: 110.0,
            window_duration: std::time::Duration::from_secs(60),
        }),
        memory_metrics: None,
        throughput_metrics: None,
        hardware_info: HardwareInfo::default(),
        current_config: HashMap::new(),
    };

    let report = advisor.analyze(&context)?;

    println!("\nProfiling-based optimizations:");
    for suggestion in &report.suggestions {
        if suggestion.category == OptimizationCategory::Compute {
            println!("- {}: {}", suggestion.title, suggestion.description);
        }
    }

    Ok(())
}

fn demo_memory_optimization() -> Result<()> {
    let advisor = OptimizationAdvisor::new();

    let context = AnalysisContext {
        model_graph: None,
        profile_results: None,
        latency_metrics: None,
        memory_metrics: Some(MemoryMetrics {
            current_bytes: 2000 * 1024 * 1024,    // 2GB
            peak_bytes: 8000 * 1024 * 1024,       // 8GB
            allocated_bytes: 15000 * 1024 * 1024, // 15GB
            reserved_bytes: 16000 * 1024 * 1024,  // 16GB
            num_allocations: 10000,
            num_deallocations: 9500,
            fragmentation_percent: 25.0, // High fragmentation
        }),
        throughput_metrics: None,
        hardware_info: HardwareInfo {
            system_memory_mb: 16000,
            ..Default::default()
        },
        current_config: HashMap::new(),
    };

    let report = advisor.analyze(&context)?;

    println!("\nMemory optimization suggestions:");
    for suggestion in report.suggestions_by_category(OptimizationCategory::Memory) {
        println!("- {}: {}", suggestion.title, suggestion.description);
        if let Some(reduction) = suggestion.expected_improvement.memory_reduction {
            println!("  Expected memory reduction: {:.1}%", reduction);
        }
        println!("  Implementation steps:");
        for (i, step) in suggestion.implementation_steps.iter().enumerate() {
            println!("    {}. {}", i + 1, step);
        }
    }

    Ok(())
}

fn demo_hardware_optimization() -> Result<()> {
    let advisor = OptimizationAdvisor::new();

    // High-end GPU setup
    let context = AnalysisContext {
        model_graph: None,
        profile_results: None,
        latency_metrics: None,
        memory_metrics: None,
        throughput_metrics: None,
        hardware_info: HardwareInfo {
            cpu_model: Some("AMD EPYC 7763".to_string()),
            cpu_cores: 64,
            gpu_model: Some("NVIDIA H100".to_string()),
            gpu_memory_mb: Some(80000),
            system_memory_mb: 512000,
            simd_capabilities: vec!["AVX512".to_string(), "AMX".to_string()],
        },
        current_config: HashMap::from([
            ("parallel_enabled".to_string(), "false".to_string()),
            ("mixed_precision".to_string(), "false".to_string()),
            ("batch_size".to_string(), "1".to_string()),
        ]),
    };

    let report = advisor.analyze(&context)?;

    println!("\nHardware-specific optimizations:");
    let high_impact = report.high_impact_suggestions();
    for suggestion in high_impact {
        println!("🔥 {} ({})", suggestion.title, suggestion.category);
        println!("   {}", suggestion.description);

        if let Some(examples) = &suggestion.code_examples {
            for example in examples {
                println!("\n   Code example ({}):", example.description);
                println!("   ```rust");
                for line in example.code.lines() {
                    println!("   {}", line);
                }
                println!("   ```");
            }
        }
    }

    Ok(())
}

fn demo_full_report() -> Result<()> {
    let advisor = OptimizationAdvisor::new();

    // Create comprehensive context
    let mut visualizer = AutoVisualizer::new();
    let graph = visualizer.visualize_gpt_model(12)?;

    let context = AnalysisContext {
        model_graph: Some(graph),
        profile_results: None,
        latency_metrics: Some(LatencyMetrics {
            count: 5000,
            mean_ms: 100.0,
            median_ms: 95.0,
            std_dev_ms: 25.0,
            min_ms: 80.0,
            max_ms: 300.0,
            p50_ms: 95.0,
            p90_ms: 140.0,
            p95_ms: 150.0,
            p99_ms: 200.0,
            p999_ms: 250.0,
            window_duration: std::time::Duration::from_secs(300),
        }),
        memory_metrics: Some(MemoryMetrics {
            current_bytes: 4000 * 1024 * 1024,    // 4GB
            peak_bytes: 6000 * 1024 * 1024,       // 6GB
            allocated_bytes: 20000 * 1024 * 1024, // 20GB
            reserved_bytes: 21000 * 1024 * 1024,  // 21GB
            num_allocations: 5000,
            num_deallocations: 4800,
            fragmentation_percent: 15.0,
        }),
        throughput_metrics: Some(ThroughputMetrics {
            tokens_per_second: 500.0,
            batches_per_second: 10.0,
            samples_per_second: 10.0,
            avg_batch_size: 1.0,
            avg_sequence_length: 50.0,
            total_tokens: 250000,
            total_batches: 5000,
            total_duration: std::time::Duration::from_secs(300),
        }),
        hardware_info: HardwareInfo {
            cpu_model: Some("Intel Xeon Gold 6330".to_string()),
            cpu_cores: 28,
            gpu_model: Some("NVIDIA A100 40GB".to_string()),
            gpu_memory_mb: Some(40960),
            system_memory_mb: 256000,
            simd_capabilities: vec!["AVX512".to_string()],
        },
        current_config: HashMap::from([
            ("mode".to_string(), "inference".to_string()),
            ("batch_size".to_string(), "1".to_string()),
            ("quantization".to_string(), "false".to_string()),
            ("kv_cache_enabled".to_string(), "false".to_string()),
        ]),
    };

    let report = advisor.analyze(&context)?;

    // Save report to file
    let markdown = report.to_markdown();
    std::fs::write("/tmp/optimization_report.md", &markdown)?;
    println!("\n✅ Full optimization report saved to: /tmp/optimization_report.md");

    // Print summary
    println!("\nOptimization Summary:");
    println!("====================");
    println!("Total suggestions: {}", report.summary.total_suggestions);
    println!(
        "Potential improvements: Latency -{:.1}%, Memory -{:.1}%, Throughput +{:.1}%",
        report.summary.potential_latency_reduction,
        report.summary.potential_memory_reduction,
        report.summary.potential_throughput_increase
    );

    println!("\nSuggestions by category:");
    for (category, count) in &report.summary.suggestions_by_category {
        println!("  {:?}: {}", category, count);
    }

    println!("\nTop 3 Easy Wins:");
    for suggestion in report.easy_suggestions().iter().take(3) {
        println!("  ✨ {}: {}", suggestion.title, suggestion.description);
    }

    Ok(())
}

// Example: Custom optimization rule
#[allow(dead_code)]
struct CustomModelRule;

impl trustformers_core::performance::OptimizationRule for CustomModelRule {
    fn analyze(
        &self,
        context: &AnalysisContext,
    ) -> Result<Option<trustformers_core::performance::OptimizationSuggestion>, TrustformersError>
    {
        // Check for specific model patterns
        if let Some(graph) = &context.model_graph {
            if graph.nodes.len() > 100 {
                return Ok(Some(
                    trustformers_core::performance::OptimizationSuggestion {
                        id: "custom_optimization".to_string(),
                        category: OptimizationCategory::Architecture,
                        impact: trustformers_core::performance::ImpactLevel::Medium,
                        difficulty: trustformers_core::performance::Difficulty::Hard,
                        title: "Consider Model Distillation".to_string(),
                        description: format!(
                            "Model has {} layers. Consider distilling to a smaller student model.",
                            graph.nodes.len()
                        ),
                        expected_improvement:
                            trustformers_core::performance::PerformanceImprovement {
                                latency_reduction: Some(60.0),
                                throughput_increase: Some(150.0),
                                memory_reduction: Some(70.0),
                                other_metrics: HashMap::new(),
                            },
                        implementation_steps: vec![
                            "Train a smaller student model".to_string(),
                            "Use knowledge distillation loss".to_string(),
                            "Validate accuracy preservation".to_string(),
                        ],
                        code_examples: None,
                        warnings: vec!["Requires retraining".to_string()],
                        related_suggestions: vec![],
                    },
                ));
            }
        }
        Ok(None)
    }
}

#[allow(dead_code)]
fn demo_custom_rule() -> Result<()> {
    println!("\n=== Bonus: Custom Optimization Rule ===");

    let mut advisor = OptimizationAdvisor::new();
    advisor.add_rule(Box::new(CustomModelRule));

    // Run analysis with custom rule
    let mut visualizer = AutoVisualizer::new();
    let graph = visualizer.visualize_bert_model(24)?; // Large model

    let context = AnalysisContext {
        model_graph: Some(graph),
        profile_results: None,
        latency_metrics: None,
        memory_metrics: None,
        throughput_metrics: None,
        hardware_info: HardwareInfo::default(),
        current_config: HashMap::new(),
    };

    let report = advisor.analyze(&context)?;

    // Find our custom suggestion
    if let Some(custom) = report.suggestions.iter().find(|s| s.id == "custom_optimization") {
        println!("✓ Custom rule triggered: {}", custom.title);
        println!("  {}", custom.description);
    }

    Ok(())
}
