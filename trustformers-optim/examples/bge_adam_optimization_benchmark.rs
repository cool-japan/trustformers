//! # BGE-Adam Optimization Benchmark
//!
//! Comprehensive benchmark comparing the original BGE-Adam implementation
//! with the optimized version to demonstrate performance improvements.

use std::time::{Duration, Instant};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;
use trustformers_optim::{bge_adam_optimized::OptimizedBGEAdam, BGEAdam, StatefulOptimizer};

fn create_test_data(size: usize) -> (Vec<Tensor>, Vec<Tensor>) {
    let mut parameters = Vec::new();
    let mut gradients = Vec::new();

    for _ in 0..10 {
        // 10 parameter tensors
        let param_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01) % 1.0).collect();
        let grad_data: Vec<f32> =
            (0..size).map(|i| ((i as f32 * 0.02 + 0.5) % 2.0) - 1.0).collect();

        parameters.push(Tensor::new(param_data).unwrap());
        gradients.push(Tensor::new(grad_data).unwrap());
    }

    (parameters, gradients)
}

fn benchmark_optimizer<O: Optimizer>(
    optimizer: &mut O,
    parameters: &mut [Tensor],
    gradients: &[Tensor],
    iterations: usize,
    name: &str,
) -> Duration {
    let start = Instant::now();

    for _ in 0..iterations {
        for (param, grad) in parameters.iter_mut().zip(gradients.iter()) {
            optimizer.update(param, grad).unwrap();
        }
        optimizer.step();
    }

    let duration = start.elapsed();

    println!(
        "📊 {}: {} iterations in {:.2?} ({:.1}µs/iter)",
        name,
        iterations,
        duration,
        duration.as_micros() as f64 / iterations as f64
    );

    duration
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 BGE-Adam Optimization Benchmark");
    println!("=====================================");
    println!("🔬 Comparing original vs optimized BGE-Adam implementations");
    println!();

    let test_sizes = vec![1000, 10000, 50000];
    let iterations = 100;

    for &size in &test_sizes {
        println!("🎯 Testing with {} parameters per tensor", size);
        println!("────────────────────────────────────────");

        // Create test data
        let (mut params_original, gradients) = create_test_data(size);
        let (mut params_optimized, _) = create_test_data(size);

        // Create optimizers
        let mut original_optimizer = BGEAdam::new(
            1e-3,         // learning rate
            (0.9, 0.999), // (β1, β2)
            1e-8,         // epsilon
            0.01,         // weight decay
            0.1,          // entropy scaling factor
            0.05,         // β1 adaptation factor
            0.05,         // β2 adaptation factor
        );

        let mut optimized_optimizer = OptimizedBGEAdam::for_high_performance();

        // Benchmark original implementation
        let original_time = benchmark_optimizer(
            &mut original_optimizer,
            &mut params_original,
            &gradients,
            iterations,
            "Original BGE-Adam",
        );

        // Benchmark optimized implementation
        let optimized_time = benchmark_optimizer(
            &mut optimized_optimizer,
            &mut params_optimized,
            &gradients,
            iterations,
            "Optimized BGE-Adam",
        );

        // Calculate speedup
        let speedup = original_time.as_micros() as f64 / optimized_time.as_micros() as f64;
        let memory_reduction = original_optimizer.memory_usage().total_bytes as f64
            / optimized_optimizer.memory_usage().total_bytes as f64;

        println!("📈 Performance Summary for {} parameters:", size);
        println!("   🚀 Speedup: {:.1}x faster", speedup);
        println!(
            "   💾 Memory efficiency: {:.1}x less memory",
            memory_reduction
        );

        // Show entropy statistics
        let (min_ent, max_ent, avg_ent) = optimized_optimizer.get_entropy_stats();
        println!(
            "   📊 Entropy stats (min/max/avg): {:.4}/{:.4}/{:.4}",
            min_ent, max_ent, avg_ent
        );

        // Show performance details
        println!(
            "   🔧 {}",
            optimized_optimizer.performance_stats().lines().next().unwrap_or("")
        );

        println!();
    }

    println!("🎉 Benchmark completed!");
    println!("✨ Key improvements in optimized BGE-Adam:");
    println!("   • Single-pass processing eliminates redundant calculations");
    println!("   • Reduced tensor conversions (3-5x fewer .data() calls)");
    println!("   • Vectorized operations for better cache utilization");
    println!("   • Pre-allocated buffers reduce memory allocations");
    println!("   • Efficient entropy history management");
    println!("   • SIMD-friendly data processing patterns");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_functions() {
        let (mut params, grads) = create_test_data(100);
        assert_eq!(params.len(), 10);
        assert_eq!(grads.len(), 10);

        let mut optimizer = OptimizedBGEAdam::new();
        let duration =
            benchmark_optimizer(&mut optimizer, &mut params, &grads, 5, "Test Optimizer");

        assert!(duration > Duration::from_nanos(1)); // Should take some time
    }

    #[test]
    fn test_create_test_data() {
        let (params, grads) = create_test_data(50);

        assert_eq!(params.len(), 10);
        assert_eq!(grads.len(), 10);

        // Check that each tensor has correct size
        for param in &params {
            assert_eq!(param.data().unwrap().len(), 50);
        }

        for grad in &grads {
            assert_eq!(grad.data().unwrap().len(), 50);
        }
    }
}
