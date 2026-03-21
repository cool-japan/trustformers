#![allow(clippy::result_large_err)]
//! # Optimizer Selection Demo
//!
//! This example demonstrates how to use the OptimizerSelector to choose
//! the best optimizer for your specific training requirements.

use trustformers_optim::monitoring::OptimizerSelector;

fn main() {
    println!("🚀 TrustformeRS Optimizer Selection Demo");
    println!("========================================\n");

    // Example 1: Production training with time constraints
    println!("📊 Example 1: Production Training (time-sensitive, large model)");
    println!("---------------------------------------------------------------");

    let production_selector = OptimizerSelector::new(10_000_000) // 10M parameters
        .time_sensitive(true)
        .fast_convergence(true);

    let report = production_selector.generate_report();
    println!("{}", report);

    // Example 2: Research experiment with robustness priority
    println!("\n📊 Example 2: Research Experiment (robustness priority)");
    println!("-------------------------------------------------------");

    let research_selector = OptimizerSelector::new(100_000) // 100K parameters
        .robustness_priority(true)
        .advanced_features(true);

    let research_report = research_selector.generate_report();
    println!("{}", research_report);

    // Example 3: Memory-constrained edge deployment
    println!("\n📊 Example 3: Edge Deployment (memory-constrained)");
    println!("--------------------------------------------------");

    let edge_selector = OptimizerSelector::new(50_000) // 50K parameters
        .memory_constrained(true)
        .time_sensitive(true);

    let edge_report = edge_selector.generate_report();
    println!("{}", edge_report);

    // Example 4: Quick recommendation lookup
    println!("\n📊 Example 4: Quick Recommendations");
    println!("-----------------------------------");

    let quick_selector = OptimizerSelector::new(1_000_000);
    let recommendations = quick_selector.get_recommendations();

    println!("🏆 Top 3 Optimizers for General Use:");
    for (i, rec) in recommendations.iter().take(3).enumerate() {
        let emoji = match i {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "📊",
        };
        println!(
            "   {} {}: {} ({:.1}x overhead)",
            emoji, rec.name, rec.description, rec.estimated_overhead
        );
    }

    println!(
        "\n✨ Demo completed! Use OptimizerSelector to choose the best optimizer for your needs."
    );
}
