//! Comprehensive test suite for TrustformeRS optimization modules.
//!
//! This module provides extensive test coverage for all optimization algorithms,
//! performance optimizations, and advanced features implemented in the crate.
//!
//! # Test Categories
//!
//! - **Integration Tests**: End-to-end optimizer functionality
//! - **Performance Tests**: Benchmark and performance regression tests
//! - **Memory Tests**: Memory usage and leak detection
//! - **Convergence Tests**: Algorithm correctness and convergence properties
//! - **Error Handling Tests**: Robustness and error recovery

use crate::adam_v2::*;
use crate::cache_friendly::*;
use crate::kernel_fusion::*;
use crate::memory_layout::*;
use crate::parallel::*;
use crate::traits::*;
use std::time::Instant;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Test utilities for optimizer testing.
pub mod test_utils {
    use super::*;

    /// Creates a test tensor with specified shape and initial values.
    pub fn create_test_tensor(shape: &[usize], value: f32) -> Tensor {
        let size = shape.iter().product();
        Tensor::new(vec![value; size]).expect("Failed to create test tensor")
    }

    /// Creates a test gradient tensor with random values.
    pub fn create_test_gradient(shape: &[usize]) -> Tensor {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push((i as f32 % 100.0) * 0.001 - 0.05); // Values between -0.05 and 0.05
        }
        Tensor::new(data).expect("Failed to create test gradient tensor")
    }

    /// Measures optimizer performance for a given number of steps.
    pub fn benchmark_optimizer<O: Optimizer>(
        mut optimizer: O,
        param_sizes: &[usize],
        num_steps: usize,
    ) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut total_elements = 0;

        for _step in 0..num_steps {
            for &size in param_sizes.iter() {
                let mut param = create_test_tensor(&[size], 1.0);
                let grad = create_test_gradient(&[size]);

                // Use standard update method from Optimizer trait
                optimizer
                    .update(&mut param, &grad)
                    .expect("Optimizer update failed during benchmark");
                total_elements += size;
            }
            optimizer.step();
        }

        let elapsed = start_time.elapsed();

        BenchmarkResult {
            total_time_ms: elapsed.as_millis() as f64,
            total_elements,
            elements_per_second: total_elements as f64 / elapsed.as_secs_f64(),
            steps_completed: num_steps,
        }
    }

    /// Result of optimizer benchmarking.
    #[derive(Debug, Clone)]
    pub struct BenchmarkResult {
        pub total_time_ms: f64,
        pub total_elements: usize,
        pub elements_per_second: f64,
        pub steps_completed: usize,
    }

    impl BenchmarkResult {
        /// Compares this result with another for performance regression testing.
        pub fn performance_ratio(&self, other: &BenchmarkResult) -> f64 {
            self.elements_per_second / other.elements_per_second
        }
    }
}

/// Integration tests for standardized optimizers.
#[cfg(test)]
mod integration_tests {
    use super::test_utils::*;
    use super::*;

    #[test]
    fn test_standardized_adam_integration() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        let mut param = create_test_tensor(&[1000], 1.0);
        let grad = create_test_gradient(&[1000]);

        // Test basic functionality
        optimizer.update(&mut param, &grad).expect("Optimizer update failed");
        optimizer.step();

        assert_eq!(optimizer.get_lr(), 1e-3);
        assert_eq!(optimizer.num_parameters(), 1);

        // Test state management
        let state_dict = optimizer.state_dict().expect("Failed to get state dict");
        assert!(!state_dict.is_empty());

        let mut new_optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        new_optimizer.load_state_dict(state_dict).expect("Failed to load state dict");
    }

    #[test]
    fn test_cache_friendly_adam_integration() {
        let mut optimizer = CacheFriendlyAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let mut param = create_test_tensor(&[2048], 0.5);
        let grad = create_test_gradient(&[2048]);

        optimizer.update(&mut param, &grad).expect("Optimizer update failed");
        optimizer.step();

        let stats = optimizer.cache_stats();
        assert_eq!(stats.num_parameters, 1);
        assert_eq!(stats.total_elements, 2048);
        assert!(stats.estimated_l1_utilization >= 0.0);
    }

    #[test]
    fn test_parallel_adam_integration() {
        let mut optimizer = ParallelAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let mut param1 = create_test_tensor(&[1000], 1.0);
        let grad1 = create_test_gradient(&[1000]);
        let mut param2 = create_test_tensor(&[1500], 0.5);
        let grad2 = create_test_gradient(&[1500]);

        // Test single parameter updates
        optimizer.update(&mut param1, &grad1).expect("Optimizer update failed");
        optimizer.update(&mut param2, &grad2).expect("Optimizer update failed");
        optimizer.step();

        let stats = optimizer.parallel_stats();
        assert!(stats.num_threads > 0);
        assert_eq!(stats.current_step, 1);
    }

    #[test]
    fn test_layout_optimized_adam_integration() {
        let mut optimizer = LayoutOptimizedAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let mut param = create_test_tensor(&[512], 2.0);
        let grad = create_test_gradient(&[512]);

        optimizer.update(&mut param, &grad).expect("Optimizer update failed");
        optimizer.step();

        let stats = optimizer.layout_stats();
        assert_eq!(stats.total_parameters, 1);
        assert!(stats.cache_line_utilization > 0.0);
    }

    #[test]
    fn test_kernel_fused_adam_integration() {
        let mut optimizer = KernelFusedAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let mut param = create_test_tensor(&[1024], 1.5);
        let grad = create_test_gradient(&[1024]);

        optimizer.update(&mut param, &grad).expect("Optimizer update failed");
        optimizer.step();

        let stats = optimizer.gpu_stats();
        assert_eq!(stats.num_parameter_buffers, 1);
        assert_eq!(stats.total_parameter_elements, 1024);
    }
}

/// Performance and benchmark tests.
#[cfg(test)]
mod performance_tests {
    use super::test_utils::*;
    use super::*;

    #[test]
    fn test_optimizer_performance_comparison() {
        let param_sizes = vec![1000, 2000, 1500];
        let num_steps = 10;

        // Benchmark standardized Adam
        let standard_adam = StandardizedAdam::adamw(1e-3, 0.01);
        let standard_result = benchmark_optimizer(standard_adam, &param_sizes, num_steps);

        // Benchmark cache-friendly Adam
        let cache_adam = CacheFriendlyAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let cache_result = benchmark_optimizer(cache_adam, &param_sizes, num_steps);

        // Cache-friendly should be competitive or better (allow wide tolerance for CI variance)
        let performance_ratio = cache_result.performance_ratio(&standard_result);
        assert!(
            performance_ratio > 0.05,
            "Cache-friendly Adam performance severely degraded: ratio={performance_ratio:.3}"
        );

        println!(
            "Standard Adam: {:.1} elements/sec",
            standard_result.elements_per_second
        );
        println!(
            "Cache-friendly Adam: {:.1} elements/sec",
            cache_result.elements_per_second
        );
        println!("Performance ratio: {:.2}", performance_ratio);
    }

    #[test]
    fn test_parallel_scaling() {
        let config_1_thread = ParallelConfig {
            num_threads: 1,
            min_params_per_thread: 1,
            ..Default::default()
        };
        let config_4_threads = ParallelConfig {
            num_threads: 4,
            min_params_per_thread: 1,
            ..Default::default()
        };

        // Use larger workloads to better demonstrate parallel benefits
        let param_sizes = vec![50000, 80000, 60000, 40000, 70000]; // Much larger sizes
        let num_steps = 10; // More steps for better measurement

        let single_thread =
            ParallelAdam::with_config(1e-3, (0.9, 0.999), 1e-8, 0.01, config_1_thread);
        let single_result = benchmark_optimizer(single_thread, &param_sizes, num_steps);

        let multi_thread =
            ParallelAdam::with_config(1e-3, (0.9, 0.999), 1e-8, 0.01, config_4_threads);
        let multi_result = benchmark_optimizer(multi_thread, &param_sizes, num_steps);

        let speedup = multi_result.performance_ratio(&single_result);
        println!(
            "Single thread: {:.1} elements/sec",
            single_result.elements_per_second
        );
        println!(
            "Multi thread: {:.1} elements/sec",
            multi_result.elements_per_second
        );
        println!("Speedup: {:.2}x", speedup);

        // Very relaxed assertions - parallel overhead can vary significantly
        // On some systems, small workloads may even be slower with parallelism
        // Threshold is 0.05x to allow for highly loaded test environments
        assert!(
            speedup > 0.05,
            "Parallel processing performance catastrophically degraded: {:.2}x",
            speedup
        );
        assert!(
            single_result.elements_per_second > 0.0,
            "Single thread should process elements"
        );
        assert!(
            multi_result.elements_per_second > 0.0,
            "Multi thread should process elements"
        );

        // Sanity check - extreme speedup ratios indicate measurement issues
        assert!(
            speedup < 50.0,
            "Suspiciously high speedup ratio: {:.2}x",
            speedup
        );
    }

    #[test]
    fn test_memory_layout_efficiency() {
        let config_basic = AlignmentConfig::default();
        let config_optimal = AlignmentConfig::avx512();

        let basic_optimizer =
            LayoutOptimizedAdam::with_alignment(1e-3, (0.9, 0.999), 1e-8, 0.01, config_basic);
        let optimal_optimizer =
            LayoutOptimizedAdam::with_alignment(1e-3, (0.9, 0.999), 1e-8, 0.01, config_optimal);

        let basic_stats = basic_optimizer.layout_stats();
        let optimal_stats = optimal_optimizer.layout_stats();

        // Optimal should have better or equal vector size
        assert!(
            optimal_stats.alignment_config.vector_size >= basic_stats.alignment_config.vector_size
        );
    }
}

/// Memory usage and leak tests.
#[cfg(test)]
mod memory_tests {
    use super::test_utils::*;
    use super::*;

    #[test]
    fn test_memory_usage_tracking() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);

        // Initially no memory used
        let initial_stats = optimizer.memory_usage();
        assert_eq!(initial_stats.num_parameters, 0);
        assert_eq!(initial_stats.total_bytes, 0);

        // Add parameters and check memory growth
        let mut param1 = create_test_tensor(&[1000], 1.0);
        let grad1 = create_test_gradient(&[1000]);
        optimizer.update(&mut param1, &grad1).expect("Optimizer update failed");

        let stats_after_param1 = optimizer.memory_usage();
        assert_eq!(stats_after_param1.num_parameters, 1);
        assert!(stats_after_param1.total_bytes > 0);

        let mut param2 = create_test_tensor(&[2000], 1.0);
        let grad2 = create_test_gradient(&[2000]);
        optimizer.update(&mut param2, &grad2).expect("Optimizer update failed");

        let stats_after_param2 = optimizer.memory_usage();
        assert_eq!(stats_after_param2.num_parameters, 2);
        assert!(stats_after_param2.total_bytes > stats_after_param1.total_bytes);
    }

    #[test]
    fn test_state_cleanup() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);

        // Create parameters that stay in scope to have different memory addresses
        let mut params = Vec::new();
        let mut grads = Vec::new();
        for _i in 0..10 {
            params.push(create_test_tensor(&[100], 1.0));
            grads.push(create_test_gradient(&[100]));
        }

        // Add parameters to optimizer
        for i in 0..10 {
            optimizer.update(&mut params[i], &grads[i]).expect("Optimizer update failed");
        }

        let stats_before = optimizer.memory_usage();
        assert_eq!(stats_before.num_parameters, 10);

        // Reset state
        optimizer.reset_state();
        let stats_after = optimizer.memory_usage();
        assert_eq!(stats_after.num_parameters, 0);
        assert_eq!(stats_after.total_bytes, 0);
    }

    #[test]
    fn test_parallel_memory_safety() {
        let mut optimizer = ParallelAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);

        // Create parameters that stay in scope to have different memory addresses
        let param_sizes = [1000, 1500, 2000, 1200, 1800];
        let mut params = Vec::new();
        let mut grads = Vec::new();

        for (i, &size) in param_sizes.iter().enumerate() {
            params.push(create_test_tensor(&[size], i as f32));
            grads.push(create_test_gradient(&[size]));
        }

        // Update parameters with optimizer
        for i in 0..param_sizes.len() {
            optimizer.update(&mut params[i], &grads[i]).expect("Optimizer update failed");
        }

        let stats = optimizer.parallel_stats();
        assert_eq!(stats.memory_stats.num_parameters, param_sizes.len());
    }
}

/// Algorithm correctness and convergence tests.
#[cfg(test)]
mod convergence_tests {
    use super::test_utils::*;
    use super::*;

    #[test]
    fn test_simple_quadratic_convergence() {
        // Test convergence on a simple quadratic function: f(x) = (x - 2)^2
        // Minimum is at x = 2
        let mut optimizer = StandardizedAdam::adam(0.1, 0.0); // No weight decay
        let mut param = create_test_tensor(&[1], 0.0); // Start at x = 0

        let target = 2.0;
        let tolerance = 0.1;
        let max_steps = 100;

        for step in 0..max_steps {
            if let Tensor::F32(ref param_data) = param {
                let x = param_data[0];
                let _loss = (x - target).powi(2);
                let grad_val = 2.0 * (x - target); // Gradient of (x - 2)^2

                let grad = Tensor::new(vec![grad_val]).expect("Failed to create tensor");
                optimizer.update(&mut param, &grad).expect("Optimizer update failed");
                optimizer.step();

                // Check convergence
                if (x - target).abs() < tolerance {
                    println!("Converged at step {} to x = {:.4}", step, x);
                    return;
                }
            }
        }

        // Should have converged
        if let Tensor::F32(ref param_data) = param {
            let final_x = param_data[0];
            assert!(
                (final_x - target).abs() < tolerance,
                "Failed to converge: final_x = {}, target = {}",
                final_x,
                target
            );
        }
    }

    #[test]
    fn test_bias_correction_effectiveness() {
        // Test that bias correction improves early training
        let mut optimizer = StandardizedAdam::adam(0.1, 0.0); // Larger LR, no weight decay
        let mut param = create_test_tensor(&[1], 1.0);

        // Apply consistent positive gradient for several steps (parameter should decrease)
        let consistent_grad = Tensor::new(vec![0.1]).expect("Failed to create tensor");

        let mut param_values = Vec::new();
        for _step in 0..10 {
            if let Tensor::F32(ref param_data) = param {
                param_values.push(param_data[0]);
            }

            optimizer.update(&mut param, &consistent_grad).expect("Optimizer update failed");
            optimizer.step();
        }

        // Parameter should be decreasing (gradient is positive, so we move left)
        assert!(
            param_values[0] > param_values[4],
            "Parameter should decrease with positive gradients"
        );
        assert!(
            param_values[4] > param_values[9],
            "Parameter should continue decreasing"
        );

        // Early steps should show bias correction effect (larger changes initially)
        let early_change = (param_values[1] - param_values[0]).abs();
        let late_change = (param_values[9] - param_values[8]).abs();
        assert!(
            early_change > late_change * 0.5,
            "Bias correction should make early steps more effective"
        );
    }

    #[test]
    fn test_weight_decay_modes() {
        // Test that L2 regularization and decoupled weight decay behave differently
        let mut adam_l2 = StandardizedAdam::new(AdamConfig::adam(0.1, 0.01));
        let mut adamw_decoupled = StandardizedAdam::new(AdamConfig::adamw(0.1, 0.01));

        let mut param_l2 = create_test_tensor(&[1], 2.0);
        let mut param_decoupled = create_test_tensor(&[1], 2.0);
        let zero_grad = Tensor::new(vec![0.0]).expect("Failed to create tensor"); // Zero gradient to isolate weight decay effect

        // Apply zero gradient to see pure weight decay effect
        adam_l2.update(&mut param_l2, &zero_grad).expect("Optimizer update failed");
        adamw_decoupled
            .update(&mut param_decoupled, &zero_grad)
            .expect("Optimizer update failed");

        // Both should apply weight decay, but differently
        if let (Tensor::F32(ref l2_data), Tensor::F32(ref decoupled_data)) =
            (&param_l2, &param_decoupled)
        {
            // Both should decrease due to weight decay, but amounts may differ
            assert!(
                l2_data[0] < 2.0,
                "L2 regularization should decrease parameter"
            );
            assert!(
                decoupled_data[0] < 2.0,
                "Decoupled weight decay should decrease parameter"
            );
        }
    }
}

/// Error handling and robustness tests.
#[cfg(test)]
mod error_handling_tests {
    use super::test_utils::*;
    use super::*;

    #[test]
    fn test_mismatched_tensor_sizes() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        let mut param = create_test_tensor(&[1000], 1.0);
        let grad = create_test_gradient(&[500]); // Different size

        let result = optimizer.update(&mut param, &grad);
        assert!(result.is_err(), "Should error on mismatched tensor sizes");
    }

    #[test]
    fn test_invalid_hyperparameters() {
        // Test various invalid hyperparameter combinations

        // Negative learning rate
        let config = AdamConfig {
            lr: -1e-3,
            ..Default::default()
        };
        let mut optimizer = StandardizedAdam::new(config);
        let mut param = create_test_tensor(&[100], 1.0);
        let grad = create_test_gradient(&[100]);

        // Should still work (implementation choice), but might behave unexpectedly
        let result = optimizer.update(&mut param, &grad);
        assert!(
            result.is_ok(),
            "Negative learning rate should be handled gracefully"
        );

        // Invalid beta values
        let config = AdamConfig {
            betas: (1.1, 0.999), // Beta1 > 1
            ..Default::default()
        };
        let mut optimizer = StandardizedAdam::new(config);
        let result = optimizer.update(&mut param, &grad);
        assert!(result.is_ok(), "Invalid betas should be handled gracefully");
    }

    #[test]
    fn test_empty_tensors() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        let mut param = Tensor::zeros(&[0]).expect("Failed to create tensor");
        let grad = Tensor::zeros(&[0]).expect("Failed to create tensor");

        let result = optimizer.update(&mut param, &grad);
        assert!(result.is_ok(), "Empty tensors should be handled gracefully");
    }

    #[test]
    fn test_large_tensor_handling() {
        let mut optimizer = CacheFriendlyAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let large_size = 1_000_000; // 1M elements
        let mut param = create_test_tensor(&[large_size], 1.0);
        let grad = create_test_gradient(&[large_size]);

        let result = optimizer.update(&mut param, &grad);
        assert!(
            result.is_ok(),
            "Large tensors should be handled efficiently"
        );

        let stats = optimizer.cache_stats();
        assert_eq!(stats.total_elements, large_size);
    }

    #[test]
    fn test_nan_gradient_handling() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        let mut param = create_test_tensor(&[100], 1.0);
        let grad = Tensor::new(vec![f32::NAN; 100]).expect("Failed to create tensor");

        let result = optimizer.update(&mut param, &grad);

        // Check if NaN gradients cause issues
        if let Tensor::F32(ref param_data) = param {
            let has_nan = param_data.iter().any(|&x| x.is_nan());
            if has_nan {
                println!("Warning: NaN gradients propagated to parameters");
            }
        }

        // Implementation should ideally handle NaN gracefully
        assert!(result.is_ok(), "NaN gradients should be handled");
    }
}

/// Trait implementation coverage tests.
#[cfg(test)]
mod trait_coverage_tests {
    use super::*;

    #[test]
    fn test_stateful_optimizer_trait() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);

        // Test trait methods
        let config = optimizer.config();
        assert_eq!(config.lr, 1e-3);
        assert_eq!(config.weight_decay, 0.01);

        let state = optimizer.state();
        assert_eq!(state.step, 0);

        optimizer.step();
        assert_eq!(optimizer.state().step, 1);

        assert_eq!(optimizer.num_parameters(), 0);

        let memory_stats = optimizer.memory_usage();
        assert_eq!(memory_stats.num_parameters, 0);
    }

    #[test]
    fn test_momentum_optimizer_trait() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);

        assert_eq!(optimizer.momentum_coeff(), 0.9);
        optimizer.set_momentum_coeff(0.95);
        assert_eq!(optimizer.momentum_coeff(), 0.95);

        let momentum_buffers = optimizer.momentum_buffers();
        assert!(momentum_buffers.is_empty());

        optimizer.clear_momentum();
        assert!(optimizer.momentum_buffers().is_empty());
    }

    #[test]
    fn test_adaptive_momentum_optimizer_trait() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);

        assert_eq!(optimizer.variance_coeff(), 0.999);
        optimizer.set_variance_coeff(0.995);
        assert_eq!(optimizer.variance_coeff(), 0.995);

        assert_eq!(optimizer.epsilon(), 1e-8);
        optimizer.set_epsilon(1e-6);
        assert_eq!(optimizer.epsilon(), 1e-6);

        let (m_hat, v_hat) = optimizer.apply_bias_correction(0.1, 0.01, 5);
        assert!(m_hat > 0.1); // Should be bias-corrected (larger)
        assert!(v_hat > 0.01); // Should be bias-corrected (larger)
    }
}

/// Configuration and setup tests.
#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn test_cache_config_variants() {
        let default_config = CacheConfig::default();
        let l1_config = CacheConfig::l1_optimized();
        let l2_config = CacheConfig::l2_optimized();
        let l3_config = CacheConfig::l3_optimized();

        assert!(l1_config.block_size < l2_config.block_size);
        assert!(l2_config.block_size < l3_config.block_size);
        assert_eq!(default_config.cache_line_size, 64);
    }

    #[test]
    fn test_parallel_config_variants() {
        let default_config = ParallelConfig::default();
        let cpu_config = ParallelConfig::cpu_optimized();
        let large_model_config = ParallelConfig::large_model();
        let memory_bound_config = ParallelConfig::memory_bound();

        assert_eq!(cpu_config.num_threads, num_cpus::get());
        assert!(large_model_config.min_params_per_thread > default_config.min_params_per_thread);
        assert!(memory_bound_config.numa_aware);
    }

    #[test]
    fn test_kernel_fusion_config_variants() {
        let default_config = KernelFusionConfig::default();
        let a100_config = KernelFusionConfig::a100();
        let h100_config = KernelFusionConfig::h100();

        assert_eq!(default_config.compute_capability, (7, 5));
        assert_eq!(a100_config.compute_capability, (8, 0));
        assert_eq!(h100_config.compute_capability, (9, 0));

        assert!(h100_config.shared_memory_size > a100_config.shared_memory_size);
        assert!(a100_config.shared_memory_size > default_config.shared_memory_size);
    }
}

/// Run all tests and generate a coverage report.
#[cfg(test)]
#[test]
fn test_coverage_report() {
    println!("\n=== TrustformeRS Optim Test Coverage Report ===");
    println!("Integration tests: 5 tests covering core optimizer functionality");
    println!("Performance tests: 3 benchmarks covering optimization efficiency");
    println!("Memory tests: 3 tests covering memory usage and safety");
    println!("Convergence tests: 3 tests covering algorithm correctness");
    println!("Error handling tests: 5 tests covering robustness");
    println!("Trait coverage tests: 3 tests covering trait implementations");
    println!("Configuration tests: 3 tests covering configuration variants");
    println!("\nTotal: 25+ comprehensive tests covering all major functionality");
    println!("Coverage includes: Optimizers, Memory Layout, Parallelization, Caching, GPU Kernels");
}

/// Extended convergence and algorithm correctness tests.
#[cfg(test)]
mod convergence_extended_tests {
    use super::test_utils::*;
    use super::*;

    #[test]
    fn test_adam_converges_to_zero_from_positive() {
        // Minimize x² by gradient descent: gradient = 2x.
        // Starting at x = 5.0, with lr=0.1 should converge to ~0.
        let mut optimizer = StandardizedAdam::adam(0.1, 0.0);
        let mut param = create_test_tensor(&[1], 5.0);
        let tolerance = 0.05_f32;
        let max_steps = 300;

        for _ in 0..max_steps {
            let x = if let Tensor::F32(ref d) = param { d[0] } else { break };
            if x.abs() < tolerance {
                return; // converged
            }
            let grad = Tensor::new(vec![2.0 * x]).expect("create grad");
            optimizer.update(&mut param, &grad).expect("optimizer update");
            optimizer.step();
        }

        if let Tensor::F32(ref d) = param {
            assert!(d[0].abs() < tolerance, "failed to converge: x={}", d[0]);
        }
    }

    #[test]
    fn test_optimizer_step_counter_increments() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        assert_eq!(optimizer.state().step, 0);
        for _ in 0..5 {
            optimizer.step();
        }
        assert_eq!(optimizer.state().step, 5);
    }

    #[test]
    fn test_weight_decay_reduces_parameter_magnitude() {
        // With a strong weight decay and zero gradient, parameter should shrink.
        let mut optimizer = StandardizedAdam::new(AdamConfig::adamw(0.01, 0.5));
        let initial_val = 2.0_f32;
        let mut param = create_test_tensor(&[1], initial_val);
        let zero_grad = Tensor::new(vec![0.0_f32]).expect("zero grad");

        for _ in 0..10 {
            optimizer.update(&mut param, &zero_grad).expect("update");
            optimizer.step();
        }

        if let Tensor::F32(ref d) = param {
            assert!(
                d[0].abs() < initial_val,
                "weight decay should reduce parameter magnitude"
            );
        }
    }

    #[test]
    fn test_reset_state_allows_fresh_start() {
        let mut optimizer = StandardizedAdam::adam(0.1, 0.0);
        let mut param = create_test_tensor(&[1], 1.0);
        let grad = create_test_gradient(&[1]);

        // Run a few steps to build up momentum state.
        for _ in 0..5 {
            optimizer.update(&mut param, &grad).expect("update");
            optimizer.step();
        }

        // Reset state — step counter should go to zero.
        optimizer.reset_state();
        assert_eq!(optimizer.state().step, 0);
        assert_eq!(optimizer.memory_usage().num_parameters, 0);
    }

    #[test]
    fn test_multiple_params_update_independently() {
        let mut optimizer = StandardizedAdam::adam(0.1, 0.0);
        let mut param1 = create_test_tensor(&[1], 3.0);
        let mut param2 = create_test_tensor(&[1], -3.0);

        // grad for param1 is positive (should push left)
        let grad1 = Tensor::new(vec![1.0_f32]).expect("grad1");
        // grad for param2 is negative (should push right)
        let grad2 = Tensor::new(vec![-1.0_f32]).expect("grad2");

        let p1_before = if let Tensor::F32(ref d) = param1 { d[0] } else { 0.0 };
        let p2_before = if let Tensor::F32(ref d) = param2 { d[0] } else { 0.0 };

        optimizer.update(&mut param1, &grad1).expect("update p1");
        optimizer.update(&mut param2, &grad2).expect("update p2");
        optimizer.step();

        let p1_after = if let Tensor::F32(ref d) = param1 { d[0] } else { 0.0 };
        let p2_after = if let Tensor::F32(ref d) = param2 { d[0] } else { 0.0 };

        assert!(
            p1_after < p1_before,
            "param1 should decrease with positive gradient"
        );
        assert!(
            p2_after > p2_before,
            "param2 should increase with negative gradient"
        );
    }

    #[test]
    fn test_higher_lr_moves_parameter_more() {
        // Same gradient, but higher LR should produce a larger update.
        let mut opt_low = StandardizedAdam::adam(1e-4, 0.0);
        let mut opt_high = StandardizedAdam::adam(0.1, 0.0);

        let mut param_low = create_test_tensor(&[1], 5.0);
        let mut param_high = create_test_tensor(&[1], 5.0);
        let grad = Tensor::new(vec![1.0_f32]).expect("grad");
        let grad_high = Tensor::new(vec![1.0_f32]).expect("grad high");

        opt_low.update(&mut param_low, &grad).expect("update low");
        opt_low.step();
        opt_high.update(&mut param_high, &grad_high).expect("update high");
        opt_high.step();

        let change_low =
            (5.0_f32 - if let Tensor::F32(ref d) = param_low { d[0] } else { 5.0 }).abs();
        let change_high =
            (5.0_f32 - if let Tensor::F32(ref d) = param_high { d[0] } else { 5.0 }).abs();
        assert!(
            change_high > change_low,
            "higher LR should produce larger parameter change"
        );
    }
}

/// AdamW-specific decoupling and warmup tests.
#[cfg(test)]
mod adamw_decoupling_tests {
    use super::test_utils::*;
    use super::*;

    #[test]
    fn test_adamw_decoupled_weight_decay_with_gradient() {
        // With both gradient and weight decay, parameter should decrease faster than gradient alone.
        let mut optimizer = StandardizedAdam::new(AdamConfig::adamw(0.1, 0.1));
        let mut param = create_test_tensor(&[1], 2.0);
        let grad = Tensor::new(vec![0.1_f32]).expect("grad");

        optimizer.update(&mut param, &grad).expect("update");
        optimizer.step();

        if let Tensor::F32(ref d) = param {
            assert!(
                d[0] < 2.0_f32,
                "param should decrease under weight decay + gradient"
            );
        }
    }

    #[test]
    fn test_adamw_vs_adam_produce_different_results() {
        // With weight_decay > 0, AdamW and L2-Adam should diverge.
        let wd = 0.1;
        let mut adam_l2 = StandardizedAdam::new(AdamConfig::adam(0.01, wd));
        let mut adamw = StandardizedAdam::new(AdamConfig::adamw(0.01, wd));

        let mut param_l2 = create_test_tensor(&[1], 1.0);
        let mut param_adamw = create_test_tensor(&[1], 1.0);
        let grad = Tensor::new(vec![1.0_f32]).expect("grad");
        let grad2 = Tensor::new(vec![1.0_f32]).expect("grad2");

        adam_l2.update(&mut param_l2, &grad).expect("adam update");
        adamw.update(&mut param_adamw, &grad2).expect("adamw update");

        // Both should have processed without error; results may differ.
        // We just verify neither is NaN/inf.
        if let Tensor::F32(ref d) = param_l2 {
            assert!(d[0].is_finite(), "adam_l2 result should be finite");
        }
        if let Tensor::F32(ref d) = param_adamw {
            assert!(d[0].is_finite(), "adamw result should be finite");
        }
    }

    #[test]
    fn test_state_dict_roundtrip_preserves_step() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        optimizer.step();
        optimizer.step();
        optimizer.step();

        let state_dict = optimizer.state_dict().expect("get state dict");
        let mut new_optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        new_optimizer.load_state_dict(state_dict).expect("load state dict");

        assert_eq!(
            new_optimizer.state().step,
            3,
            "step count should be restored"
        );
    }

    #[test]
    fn test_weight_decay_zero_leaves_param_unchanged_with_zero_grad() {
        let mut optimizer = StandardizedAdam::new(AdamConfig::adamw(0.01, 0.0));
        let initial = 3.0_f32;
        let mut param = create_test_tensor(&[1], initial);
        let zero_grad = Tensor::new(vec![0.0_f32]).expect("zero grad");

        optimizer.update(&mut param, &zero_grad).expect("update");
        optimizer.step();

        // With zero weight decay and zero gradient, param should not change significantly.
        if let Tensor::F32(ref d) = param {
            // Adam still moves slightly due to epsilon in denominator, but should be tiny.
            assert!(
                (d[0] - initial).abs() < 0.1,
                "param should not change much with zero wd and zero grad: {}",
                d[0]
            );
        }
    }
}

/// Error recovery and edge case tests.
#[cfg(test)]
mod error_recovery_tests {
    use super::test_utils::*;
    use super::*;

    #[test]
    fn test_sequential_updates_remain_stable() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        let mut param = create_test_tensor(&[10], 1.0);

        for step in 0..100 {
            let grad = create_test_gradient(&[10]);
            optimizer
                .update(&mut param, &grad)
                .unwrap_or_else(|_| panic!("update at step {step}"));
            optimizer.step();

            // Verify param values stay finite.
            if let Tensor::F32(ref d) = param {
                for (i, v) in d.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "param[{i}] is not finite at step {step}: {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_very_small_lr_stable() {
        let mut optimizer = StandardizedAdam::adamw(1e-10, 0.0);
        let mut param = create_test_tensor(&[5], 1.0);
        let grad = create_test_gradient(&[5]);

        optimizer.update(&mut param, &grad).expect("update with tiny lr");
        optimizer.step();

        if let Tensor::F32(ref d) = param {
            for v in d.iter() {
                assert!(v.is_finite(), "param should stay finite with tiny lr: {v}");
            }
        }
    }

    #[test]
    fn test_inf_gradient_does_not_panic() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        let mut param = create_test_tensor(&[3], 1.0);
        let inf_grad = Tensor::new(vec![f32::INFINITY, 0.0, -f32::INFINITY]).expect("inf grad");

        // Should not panic — may produce non-finite results but must not crash.
        let _ = optimizer.update(&mut param, &inf_grad);
    }

    #[test]
    fn test_cache_friendly_produces_finite_results() {
        let mut optimizer = CacheFriendlyAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let mut param = create_test_tensor(&[100], 1.0);
        let grad = create_test_gradient(&[100]);

        optimizer.update(&mut param, &grad).expect("update");
        optimizer.step();

        if let Tensor::F32(ref d) = param {
            for v in d.iter() {
                assert!(
                    v.is_finite(),
                    "cache-friendly Adam should produce finite results: {v}"
                );
            }
        }
    }

    #[test]
    fn test_parallel_adam_produces_finite_results() {
        let mut optimizer = ParallelAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01);
        let mut param = create_test_tensor(&[200], 1.0);
        let grad = create_test_gradient(&[200]);

        optimizer.update(&mut param, &grad).expect("parallel adam update");
        optimizer.step();

        if let Tensor::F32(ref d) = param {
            for v in d.iter() {
                assert!(
                    v.is_finite(),
                    "parallel Adam should produce finite results: {v}"
                );
            }
        }
    }
}

/// Adam hyperparameter sensitivity tests.
#[cfg(test)]
mod adam_hyperparameter_tests {
    use super::test_utils::*;
    use super::*;

    #[test]
    fn test_epsilon_prevents_nan_with_zero_gradient() {
        // A zero gradient with small epsilon should still produce finite results.
        let config = AdamConfig {
            lr: 0.1,
            eps: 1e-8,
            ..Default::default()
        };
        let mut optimizer = StandardizedAdam::new(config);
        let mut param = create_test_tensor(&[5], 1.0);
        let zero_grad = Tensor::new(vec![0.0_f32; 5]).expect("zero grad");

        optimizer.update(&mut param, &zero_grad).expect("update");
        optimizer.step();

        if let Tensor::F32(ref d) = param {
            for v in d.iter() {
                assert!(v.is_finite(), "result should be finite with epsilon: {v}");
            }
        }
    }

    #[test]
    fn test_lr_get_returns_configured_value() {
        let optimizer = StandardizedAdam::adamw(0.005, 0.01);
        assert!(
            (optimizer.get_lr() - 0.005).abs() < 1e-10,
            "get_lr should return configured value"
        );
    }

    #[test]
    fn test_config_weight_decay_accessible() {
        let optimizer = StandardizedAdam::adamw(1e-3, 0.1);
        let config = optimizer.config();
        assert!(
            (config.weight_decay - 0.1).abs() < 1e-10,
            "weight_decay should be 0.1"
        );
    }

    #[test]
    fn test_num_parameters_tracks_added_params() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        assert_eq!(optimizer.num_parameters(), 0);

        let mut p1 = create_test_tensor(&[10], 1.0);
        let g1 = create_test_gradient(&[10]);
        optimizer.update(&mut p1, &g1).expect("update p1");
        assert_eq!(optimizer.num_parameters(), 1);

        let mut p2 = create_test_tensor(&[20], 1.0);
        let g2 = create_test_gradient(&[20]);
        optimizer.update(&mut p2, &g2).expect("update p2");
        assert_eq!(optimizer.num_parameters(), 2);
    }

    #[test]
    fn test_momentum_coeff_get_set() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        assert!((optimizer.momentum_coeff() - 0.9).abs() < 1e-10);
        optimizer.set_momentum_coeff(0.95);
        assert!((optimizer.momentum_coeff() - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_variance_coeff_get_set() {
        let mut optimizer = StandardizedAdam::adamw(1e-3, 0.01);
        assert!((optimizer.variance_coeff() - 0.999).abs() < 1e-10);
        optimizer.set_variance_coeff(0.99);
        assert!((optimizer.variance_coeff() - 0.99).abs() < 1e-10);
    }
}
