//! Performance Tuning Demo
//!
//! This example demonstrates the automated performance tuning system that
//! analyzes model performance and provides actionable optimization recommendations.

use anyhow::Result;
use std::collections::HashMap;
use trustformers_debug::{HardwareType, PerformanceSnapshot, PerformanceTuner, TunerConfig};

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== Performance Tuning Demo ===\n");

    // Demo 1: Analyze GPU underutilization
    demo_gpu_underutilization()?;

    // Demo 2: Analyze memory fragmentation
    demo_memory_fragmentation()?;

    // Demo 3: Analyze batch size optimization
    demo_batch_size_optimization()?;

    // Demo 4: Comprehensive analysis
    demo_comprehensive_analysis()?;

    println!("\n=== Demo Complete ===");

    Ok(())
}

fn demo_gpu_underutilization() -> Result<()> {
    println!("--- Demo 1: GPU Underutilization Analysis ---\n");

    let config = TunerConfig {
        target_hardware: HardwareType::NvidiaGpu,
        enable_memory_tuning: false,
        enable_batch_tuning: false,
        ..Default::default()
    };

    let mut tuner = PerformanceTuner::new(config);

    // Simulate low GPU utilization scenario
    for i in 0..10 {
        let snapshot = PerformanceSnapshot {
            timestamp: i,
            total_time_ms: 150.0,
            memory_usage_mb: 2000.0,
            peak_memory_mb: 2100.0,
            gpu_utilization: 35.0, // Low GPU utilization!
            throughput: 25.0,
            batch_size: 16,
            layer_timings: HashMap::new(),
            layer_memory: HashMap::new(),
        };

        tuner.record_snapshot(snapshot);
    }

    println!("Recorded 10 performance snapshots with low GPU utilization (35%)");

    let report = tuner.analyze()?;

    println!("\n📊 Analysis Results:");
    println!("  Current Performance:");
    println!(
        "    - GPU Utilization: {:.1}%",
        report.current_performance.gpu_utilization
    );
    println!(
        "    - Throughput: {:.1} samples/sec",
        report.current_performance.avg_throughput
    );
    println!(
        "    - Efficiency Score: {:.1}/100",
        report.current_performance.efficiency_score
    );

    println!("\n  📈 Estimated Performance (after optimizations):");
    println!(
        "    - GPU Utilization: {:.1}%",
        report.estimated_performance.gpu_utilization
    );
    println!(
        "    - Throughput: {:.1} samples/sec",
        report.estimated_performance.avg_throughput
    );
    println!(
        "    - Efficiency Score: {:.1}/100",
        report.estimated_performance.efficiency_score
    );

    println!("\n  💡 Recommendations ({}):", report.recommendations.len());
    for (i, rec) in report.recommendations.iter().enumerate() {
        println!(
            "\n  {}. {} (Priority: {:?}, Confidence: {:.0}%)",
            i + 1,
            rec.title,
            rec.priority,
            rec.confidence * 100.0
        );
        println!("     Description: {}", rec.description);
        println!("     Expected Speedup: {:.1}x", rec.expected_impact.speedup);
        println!("     Actions:");
        for action in &rec.actions {
            println!("       - {}", action);
        }

        if let Some(example) = &rec.code_example {
            println!("     Code Example:");
            for line in example.lines() {
                println!("       {}", line);
            }
        }
    }

    println!();

    Ok(())
}

fn demo_memory_fragmentation() -> Result<()> {
    println!("--- Demo 2: Memory Fragmentation Analysis ---\n");

    let config = TunerConfig {
        enable_compute_tuning: false,
        enable_batch_tuning: false,
        ..Default::default()
    };

    let mut tuner = PerformanceTuner::new(config);

    // Simulate memory fragmentation (high peak vs average)
    for i in 0..10 {
        let snapshot = PerformanceSnapshot {
            timestamp: i,
            total_time_ms: 120.0,
            memory_usage_mb: 1500.0, // Average memory
            peak_memory_mb: 3000.0,  // High peak (2x average!)
            gpu_utilization: 80.0,
            throughput: 40.0,
            batch_size: 16,
            layer_timings: HashMap::new(),
            layer_memory: HashMap::new(),
        };

        tuner.record_snapshot(snapshot);
    }

    println!("Recorded snapshots with memory fragmentation:");
    println!("  Average Memory: 1500 MB");
    println!("  Peak Memory: 3000 MB (2x average!)");

    let report = tuner.analyze()?;

    println!("\n💡 Recommendations:");
    for rec in &report.recommendations {
        println!(
            "\n  • {} (Confidence: {:.0}%)",
            rec.title,
            rec.confidence * 100.0
        );
        println!("    {}", rec.description);
        println!(
            "    Expected memory reduction: {:.0} MB",
            rec.expected_impact.memory_reduction_mb
        );
        println!("    Actions:");
        for action in &rec.actions {
            println!("      - {}", action);
        }
    }

    println!();

    Ok(())
}

fn demo_batch_size_optimization() -> Result<()> {
    println!("--- Demo 3: Batch Size Optimization ---\n");

    let config = TunerConfig {
        enable_memory_tuning: false,
        enable_compute_tuning: false,
        target_hardware: HardwareType::NvidiaGpu,
        ..Default::default()
    };

    let mut tuner = PerformanceTuner::new(config);

    // Simulate small batch size
    let snapshot = PerformanceSnapshot {
        timestamp: 0,
        total_time_ms: 100.0,
        memory_usage_mb: 1000.0,
        peak_memory_mb: 1200.0,
        gpu_utilization: 60.0,
        throughput: 30.0,
        batch_size: 4, // Very small batch size!
        layer_timings: HashMap::new(),
        layer_memory: HashMap::new(),
    };

    tuner.record_snapshot(snapshot);

    println!("Current batch size: 4 (small for GPU)");

    let report = tuner.analyze()?;

    println!("\n💡 Recommendations:");
    for rec in &report.recommendations {
        println!("\n  • {}", rec.title);
        println!("    {}", rec.description);
        println!(
            "    Expected throughput improvement: +{:.0}%",
            rec.expected_impact.throughput_improvement
        );

        if let Some(example) = &rec.code_example {
            println!("\n    Example:");
            for line in example.lines() {
                println!("      {}", line);
            }
        }
    }

    println!();

    Ok(())
}

fn demo_comprehensive_analysis() -> Result<()> {
    println!("--- Demo 4: Comprehensive Performance Analysis ---\n");

    let config = TunerConfig::default();
    let mut tuner = PerformanceTuner::new(config);

    // Simulate realistic training scenario with multiple issues
    for i in 0..20 {
        let mut layer_timings = HashMap::new();
        layer_timings.insert("embedding".to_string(), 5.0);
        layer_timings.insert("attention".to_string(), 65.0); // Bottleneck!
        layer_timings.insert("ffn".to_string(), 20.0);
        layer_timings.insert("output".to_string(), 10.0);

        let snapshot = PerformanceSnapshot {
            timestamp: i,
            total_time_ms: 100.0,
            memory_usage_mb: 2500.0,
            peak_memory_mb: 4000.0, // Memory fragmentation
            gpu_utilization: 45.0,  // Low GPU usage
            throughput: 22.0,
            batch_size: 8, // Small batch
            layer_timings,
            layer_memory: HashMap::new(),
        };

        tuner.record_snapshot(snapshot);
    }

    println!("Recorded 20 snapshots with multiple performance issues:");
    println!("  ✗ Low GPU utilization (45%)");
    println!("  ✗ Memory fragmentation (4000 MB peak vs 2500 MB avg)");
    println!("  ✗ Small batch size (8)");
    println!("  ✗ Attention layer bottleneck (65% of time)");

    let report = tuner.analyze()?;

    println!("\n📊 Performance Summary:");
    println!("\n  Current:");
    println!(
        "    - Execution Time: {:.1} ms",
        report.current_performance.avg_time_ms
    );
    println!(
        "    - Memory Usage: {:.0} MB",
        report.current_performance.avg_memory_mb
    );
    println!(
        "    - Throughput: {:.1} samples/sec",
        report.current_performance.avg_throughput
    );
    println!(
        "    - GPU Utilization: {:.1}%",
        report.current_performance.gpu_utilization
    );
    println!(
        "    - Efficiency Score: {:.1}/100",
        report.current_performance.efficiency_score
    );

    println!("\n  Estimated (after optimizations):");
    println!(
        "    - Execution Time: {:.1} ms (-{:.1}%)",
        report.estimated_performance.avg_time_ms,
        (1.0 - report.estimated_performance.avg_time_ms / report.current_performance.avg_time_ms)
            * 100.0
    );
    println!(
        "    - Memory Usage: {:.0} MB (-{:.0} MB)",
        report.estimated_performance.avg_memory_mb,
        report.current_performance.avg_memory_mb - report.estimated_performance.avg_memory_mb
    );
    println!(
        "    - Throughput: {:.1} samples/sec (+{:.1}%)",
        report.estimated_performance.avg_throughput,
        (report.estimated_performance.avg_throughput / report.current_performance.avg_throughput
            - 1.0)
            * 100.0
    );
    println!(
        "    - Efficiency Score: {:.1}/100",
        report.estimated_performance.efficiency_score
    );

    println!("\n💡 Top {} Recommendations:", report.recommendations.len());

    for (i, rec) in report.recommendations.iter().enumerate().take(5) {
        println!("\n{}. {} ({:?} Priority)", i + 1, rec.title, rec.priority);
        println!("   Category: {:?}", rec.category);
        println!("   Confidence: {:.0}%", rec.confidence * 100.0);
        println!(
            "   Impact: {:.1}x speedup, +{:.0}% throughput",
            rec.expected_impact.speedup, rec.expected_impact.throughput_improvement
        );
        println!("   Difficulty: {:?}", rec.difficulty);

        println!("   Actions:");
        for action in &rec.actions {
            println!("     • {}", action);
        }
    }

    println!();

    Ok(())
}
