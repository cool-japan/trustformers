//! Additional tests for core performance optimizer functions
//!
//! These tests complement the inline tests in functions.rs by exercising edge cases,
//! boundary conditions, and additional code paths.

#[cfg(test)]
mod tests {
    use crate::performance_optimizer::core::functions::functions::*;
    use crate::performance_optimizer::core::types::*;
    use crate::test_parallelization::PerformanceOptimizationConfig;
    use chrono::Utc;
    use std::collections::HashMap;
    use std::time::Duration;

    // --- AdaptiveParallelismController tests ---

    #[tokio::test]
    async fn test_controller_parallelism_clamped_to_min() {
        let config = AdaptiveParallelismConfig::default();
        let controller = AdaptiveParallelismController::new(config.clone())
            .await
            .expect("Controller must initialize");
        // Try setting below minimum: should be clamped
        controller
            .set_parallelism_level(0)
            .await
            .expect("set_parallelism_level must succeed");
        let level = controller.get_current_parallelism();
        assert!(level >= config.min_parallelism, "level must be >= min_parallelism");
    }

    #[tokio::test]
    async fn test_controller_parallelism_clamped_to_max() {
        let config = AdaptiveParallelismConfig::default();
        let controller = AdaptiveParallelismController::new(config.clone())
            .await
            .expect("Controller must initialize");
        let huge = usize::MAX;
        controller
            .set_parallelism_level(huge)
            .await
            .expect("set_parallelism_level must succeed");
        let level = controller.get_current_parallelism();
        assert!(level <= config.max_parallelism, "level must be <= max_parallelism");
    }

    #[tokio::test]
    async fn test_controller_initial_parallelism_at_min() {
        let config = AdaptiveParallelismConfig::default();
        let controller = AdaptiveParallelismController::new(config.clone())
            .await
            .expect("Controller must initialize");
        let initial = controller.get_current_parallelism();
        assert!(initial >= config.min_parallelism);
        assert!(initial <= config.max_parallelism);
    }

    #[tokio::test]
    async fn test_controller_recommend_parallelism_confidence_in_range() {
        let config = AdaptiveParallelismConfig::default();
        let controller = AdaptiveParallelismController::new(config)
            .await
            .expect("Controller must initialize");
        let characteristics = TestCharacteristics::default();
        let estimate = controller
            .recommend_parallelism(&characteristics)
            .await
            .expect("recommend_parallelism must succeed");
        assert!(estimate.confidence >= 0.0 && estimate.confidence <= 1.0,
            "confidence must be in [0, 1]");
        assert!(estimate.optimal_parallelism >= 1, "optimal must be >= 1");
    }

    // --- OptimalParallelismEstimator tests ---

    #[tokio::test]
    async fn test_estimator_high_cpu_intensity_reduces_parallelism() {
        let estimator = OptimalParallelismEstimator::new()
            .await
            .expect("Estimator must initialize");
        let high_cpu = TestCharacteristics {
            resource_intensity: ResourceIntensity {
                cpu_intensity: 0.95,
                memory_intensity: 0.3,
                io_intensity: 0.1,
                ..ResourceIntensity::default()
            },
            ..TestCharacteristics::default()
        };
        let low_cpu = TestCharacteristics {
            resource_intensity: ResourceIntensity {
                cpu_intensity: 0.2,
                memory_intensity: 0.3,
                io_intensity: 0.1,
                ..ResourceIntensity::default()
            },
            ..TestCharacteristics::default()
        };
        let high_cpu_estimate = estimator
            .estimate_optimal_parallelism_for_characteristics(&high_cpu)
            .await
            .expect("estimation must succeed");
        let low_cpu_estimate = estimator
            .estimate_optimal_parallelism_for_characteristics(&low_cpu)
            .await
            .expect("estimation must succeed");
        // High CPU intensity should result in lower or equal parallelism than low CPU
        assert!(
            high_cpu_estimate.optimal_parallelism <= low_cpu_estimate.optimal_parallelism,
            "high CPU intensity should not exceed low CPU in parallelism"
        );
    }

    #[tokio::test]
    async fn test_estimator_returns_positive_parallelism() {
        let estimator = OptimalParallelismEstimator::new()
            .await
            .expect("Estimator must initialize");
        let chars = TestCharacteristics::default();
        let estimate = estimator
            .estimate_optimal_parallelism_for_characteristics(&chars)
            .await
            .expect("estimation must succeed");
        assert!(estimate.optimal_parallelism >= 1, "parallelism must be >= 1");
        assert!(!estimate.method.is_empty(), "method name must not be empty");
    }

    // --- PerformanceOptimizer resource recommendation logic ---

    #[tokio::test]
    async fn test_resource_recommendations_low_cpu_suggests_increase() {
        let optimizer = PerformanceOptimizer::new(PerformanceOptimizationConfig::default())
            .await
            .expect("Optimizer must initialize");
        // CPU utilization below 0.6 should suggest increase_parallelism
        let metrics = PerformanceMeasurement {
            throughput: 80.0,
            average_latency: Duration::from_millis(50),
            cpu_utilization: 0.3, // below 0.6 threshold
            memory_utilization: 0.4,
            resource_efficiency: 0.7,
            timestamp: Utc::now(),
            measurement_duration: Duration::from_secs(30),
            cpu_usage: 0.3,
            memory_usage: 0.4,
            latency: Duration::from_millis(50),
        };
        let chars = TestCharacteristics::default();
        let recs = optimizer
            .generate_resource_optimization_recommendations(&metrics, &chars)
            .await
            .expect("Recommendations must succeed");
        assert!(
            recs.iter().any(|r| r.action == "increase_parallelism"),
            "low CPU utilization must suggest increase_parallelism"
        );
    }

    #[tokio::test]
    async fn test_resource_recommendations_high_cpu_suggests_decrease() {
        let optimizer = PerformanceOptimizer::new(PerformanceOptimizationConfig::default())
            .await
            .expect("Optimizer must initialize");
        let metrics = PerformanceMeasurement {
            throughput: 200.0,
            average_latency: Duration::from_millis(30),
            cpu_utilization: 0.95, // above 0.9 threshold
            memory_utilization: 0.5,
            resource_efficiency: 0.8,
            timestamp: Utc::now(),
            measurement_duration: Duration::from_secs(30),
            cpu_usage: 0.95,
            memory_usage: 0.5,
            latency: Duration::from_millis(30),
        };
        let chars = TestCharacteristics::default();
        let recs = optimizer
            .generate_resource_optimization_recommendations(&metrics, &chars)
            .await
            .expect("Recommendations must succeed");
        assert!(
            recs.iter().any(|r| r.action == "decrease_parallelism"),
            "high CPU utilization must suggest decrease_parallelism"
        );
    }

    #[tokio::test]
    async fn test_resource_recommendations_high_memory_adds_memory_opt() {
        let optimizer = PerformanceOptimizer::new(PerformanceOptimizationConfig::default())
            .await
            .expect("Optimizer must initialize");
        let metrics = PerformanceMeasurement {
            throughput: 100.0,
            average_latency: Duration::from_millis(40),
            cpu_utilization: 0.7,  // moderate CPU
            memory_utilization: 0.9, // above 0.85 threshold
            resource_efficiency: 0.75,
            timestamp: Utc::now(),
            measurement_duration: Duration::from_secs(30),
            cpu_usage: 0.7,
            memory_usage: 0.9,
            latency: Duration::from_millis(40),
        };
        let chars = TestCharacteristics::default();
        let recs = optimizer
            .generate_resource_optimization_recommendations(&metrics, &chars)
            .await
            .expect("Recommendations must succeed");
        assert!(
            recs.iter().any(|r| r.resource_type == "memory"),
            "high memory utilization must add memory optimization"
        );
    }

    // --- Batching recommendation logic ---

    #[tokio::test]
    async fn test_batching_short_tests_have_large_batch() {
        let optimizer = PerformanceOptimizer::new(PerformanceOptimizationConfig::default())
            .await
            .expect("Optimizer must initialize");
        let chars = TestCharacteristics {
            average_duration: Duration::from_millis(500), // < 2 seconds
            ..TestCharacteristics::default()
        };
        let rec = optimizer
            .generate_batching_recommendations(&chars)
            .await
            .expect("Batching recommendations must succeed");
        assert!(rec.batch_size >= 1);
        assert!(!rec.strategy.is_empty());
    }

    #[tokio::test]
    async fn test_batching_long_tests_have_smaller_batch() {
        let optimizer = PerformanceOptimizer::new(PerformanceOptimizationConfig::default())
            .await
            .expect("Optimizer must initialize");
        let short_chars = TestCharacteristics {
            average_duration: Duration::from_millis(500), // < 2 seconds
            ..TestCharacteristics::default()
        };
        let long_chars = TestCharacteristics {
            average_duration: Duration::from_secs(30), // > 10 seconds
            ..TestCharacteristics::default()
        };
        let short_rec = optimizer
            .generate_batching_recommendations(&short_chars)
            .await
            .expect("must succeed");
        let long_rec = optimizer
            .generate_batching_recommendations(&long_chars)
            .await
            .expect("must succeed");
        assert!(
            short_rec.batch_size >= long_rec.batch_size,
            "short tests should have larger or equal batch size vs long tests"
        );
    }

    // --- Priority calculation ---

    #[tokio::test]
    async fn test_priority_bounded_to_one() {
        let optimizer = PerformanceOptimizer::new(PerformanceOptimizationConfig::default())
            .await
            .expect("Optimizer must initialize");
        // Worst-case metrics that would push all penalty factors
        let worst_metrics = PerformanceMeasurement {
            throughput: 10.0,
            average_latency: Duration::from_millis(500),
            cpu_utilization: 0.1,
            memory_utilization: 0.9,
            resource_efficiency: 0.1, // very low -> large priority penalty
            timestamp: Utc::now(),
            measurement_duration: Duration::from_secs(30),
            cpu_usage: 0.1,
            memory_usage: 0.9,
            latency: Duration::from_millis(500),
        };
        let chars = TestCharacteristics {
            dependency_complexity: 0.9, // > 0.6 adds more priority
            ..TestCharacteristics::default()
        };
        let priority = optimizer
            .calculate_optimization_priority(&worst_metrics, &chars)
            .await
            .expect("priority calculation must succeed");
        assert!(priority >= 0.0, "priority must be >= 0");
        assert!(priority <= 1.0, "priority must be clamped to 1.0");
    }

    // --- Improvement potential ---

    #[tokio::test]
    async fn test_improvement_potential_bounded_to_half() {
        let optimizer = PerformanceOptimizer::new(PerformanceOptimizationConfig::default())
            .await
            .expect("Optimizer must initialize");
        let metrics = PerformanceMeasurement {
            throughput: 50.0,
            average_latency: Duration::from_millis(100),
            cpu_utilization: 0.5,
            memory_utilization: 0.5,
            resource_efficiency: 0.1, // very low efficiency
            timestamp: Utc::now(),
            measurement_duration: Duration::from_secs(30),
            cpu_usage: 0.5,
            memory_usage: 0.5,
            latency: Duration::from_millis(100),
        };
        let chars = TestCharacteristics {
            average_duration: Duration::from_millis(500), // < 5s -> adds 0.1
            ..TestCharacteristics::default()
        };
        let potential = optimizer
            .estimate_improvement_potential(&metrics, &chars)
            .await
            .expect("improvement estimation must succeed");
        assert!(potential >= 0.0, "potential must be >= 0");
        assert!(potential <= 0.5, "potential must be clamped to 0.5");
    }
}
