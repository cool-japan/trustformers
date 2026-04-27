//! Auto-generated module structure

pub mod bottlenecktype_traits;
pub mod complexitylevel_traits;
pub mod conflictresolutionconfig_traits;
pub mod cpuscalingconfig_traits;
pub mod databasepoolconfig_traits;
pub mod dependencytype_traits;
pub mod gpupoolconfig_traits;
pub mod groupformationcriteria_traits;
pub mod independenceanalysisconfig_traits;
pub mod isolationrequirements_traits;
pub mod loadbalancingconfig_traits;
pub mod memoryoptimizationconfig_traits;
pub mod optimizationtype_traits;
pub mod parallelizationhints_traits;
pub mod parallelperformancemonitoringconfig_traits;
pub mod performanceoptimizationconfig_traits;
pub mod portpoolconfig_traits;
pub mod priorityfactors_traits;
pub mod priorityschedulingconfig_traits;
pub mod recommendationpriority_traits;
pub mod resourcealertconfig_traits;
pub mod resourcecleanupconfig_traits;
pub mod resourcelimits_traits;
pub mod resourcemanagementconfig_traits;
pub mod resourcemonitoringconfig_traits;
pub mod resourcepoolconfig_traits;
pub mod resourcesharingcapabilities_traits;
pub mod resourceusagethresholds_traits;
pub mod schedulingconfig_traits;
pub mod schedulingstrategy_traits;
pub mod suitedefinitionconfig_traits;
pub mod tempdirpoolconfig_traits;
pub mod testbatchingconfig_traits;
pub mod testgroupingconfig_traits;
pub mod testparallelizationconfig_traits;
pub mod testsuiteorganizationconfig_traits;
pub mod types;
pub mod warmupoptimizationconfig_traits;

// Re-export all types
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ── DependencyType ────────────────────────────────────────────────────

    #[test]
    fn test_dependency_type_hard_debug() {
        let d = DependencyType::Hard;
        assert!(format!("{:?}", d).contains("Hard"));
    }

    #[test]
    fn test_dependency_type_soft_debug() {
        let d = DependencyType::Soft;
        assert!(format!("{:?}", d).contains("Soft"));
    }

    #[test]
    fn test_dependency_type_equality() {
        assert_eq!(DependencyType::Hard, DependencyType::Hard);
        assert_ne!(DependencyType::Hard, DependencyType::Soft);
        assert_ne!(DependencyType::Conflict, DependencyType::Ordering);
    }

    #[test]
    fn test_dependency_type_all_variants_distinct() {
        let variants = [
            DependencyType::Hard,
            DependencyType::Soft,
            DependencyType::Conflict,
            DependencyType::Ordering,
            DependencyType::Setup,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // ── CleanupStrategy ───────────────────────────────────────────────────

    #[test]
    fn test_cleanup_strategy_immediate_debug() {
        let s = CleanupStrategy::Immediate;
        assert!(format!("{:?}", s).contains("Immediate"));
    }

    #[test]
    fn test_cleanup_strategy_deferred_carries_duration() {
        let delay = Duration::from_secs(5);
        let s = CleanupStrategy::Deferred(delay);
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("Deferred"));
    }

    #[test]
    fn test_cleanup_strategy_lazy_debug() {
        let s = CleanupStrategy::Lazy;
        assert!(format!("{:?}", s).contains("Lazy"));
    }

    #[test]
    fn test_cleanup_strategy_none_debug() {
        let s = CleanupStrategy::None;
        assert!(format!("{:?}", s).contains("None"));
    }

    #[test]
    fn test_cleanup_strategy_custom_carries_name() {
        let s = CleanupStrategy::Custom("my_cleanup".to_string());
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("Custom"));
    }

    // ── ComplexityLevel ───────────────────────────────────────────────────

    #[test]
    fn test_complexity_level_variants_debug() {
        assert!(format!("{:?}", ComplexityLevel::Low).contains("Low"));
        assert!(format!("{:?}", ComplexityLevel::Medium).contains("Medium"));
        assert!(format!("{:?}", ComplexityLevel::High).contains("High"));
        assert!(format!("{:?}", ComplexityLevel::VeryHigh).contains("VeryHigh"));
    }

    // ── MemoryOptimizationConfig ──────────────────────────────────────────

    #[test]
    fn test_memory_optimization_config_fields() {
        let cfg = MemoryOptimizationConfig {
            memory_aware_scheduling: true,
            memory_warning_threshold: 0.7_f32,
            memory_throttling_threshold: 0.9_f32,
            gc_hints: false,
            cleanup_between_tests: true,
        };
        assert!(cfg.memory_aware_scheduling);
        assert!((cfg.memory_warning_threshold - 0.7).abs() < 1e-6);
        assert!((cfg.memory_throttling_threshold - 0.9).abs() < 1e-6);
        assert!(!cfg.gc_hints);
        assert!(cfg.cleanup_between_tests);
    }

    // ── SequentialComparison ──────────────────────────────────────────────

    #[test]
    fn test_sequential_comparison_speedup_positive() {
        let seq_time = Duration::from_secs(10);
        let par_time = Duration::from_secs(4);
        let cmp = SequentialComparison {
            estimated_sequential_time: seq_time,
            actual_parallel_time: par_time,
            time_savings: Duration::from_secs(6),
            speedup_factor: 2.5_f32,
            efficiency_percentage: 83.3_f32,
        };
        assert!(cmp.speedup_factor > 1.0);
        assert!(cmp.time_savings > Duration::ZERO);
    }

    // ── PerformanceTrend ──────────────────────────────────────────────────

    #[test]
    fn test_performance_trend_improving_carries_value() {
        let t = PerformanceTrend::Improving(0.15_f32);
        let dbg = format!("{:?}", t);
        assert!(dbg.contains("Improving"));
    }

    #[test]
    fn test_performance_trend_stable_debug() {
        let t = PerformanceTrend::Stable;
        assert!(format!("{:?}", t).contains("Stable"));
    }

    #[test]
    fn test_performance_trend_degrading_carries_value() {
        let t = PerformanceTrend::Degrading(0.05_f32);
        let dbg = format!("{:?}", t);
        assert!(dbg.contains("Degrading"));
    }

    #[test]
    fn test_performance_trend_insufficient_data_debug() {
        let t = PerformanceTrend::InsufficientData;
        assert!(format!("{:?}", t).contains("InsufficientData"));
    }

    // ── EarlyTerminationStrategy ──────────────────────────────────────────

    #[test]
    fn test_early_termination_none_debug() {
        let e = EarlyTerminationStrategy::None;
        assert!(format!("{:?}", e).contains("None"));
    }

    #[test]
    fn test_early_termination_error_rate_threshold() {
        let e = EarlyTerminationStrategy::ErrorRateThreshold(0.1_f32);
        assert!(format!("{:?}", e).contains("ErrorRateThreshold"));
    }

    #[test]
    fn test_early_termination_time_budget() {
        let e = EarlyTerminationStrategy::TimeBudget(Duration::from_secs(60));
        assert!(format!("{:?}", e).contains("TimeBudget"));
    }

    #[test]
    fn test_early_termination_resource_exhaustion() {
        let e = EarlyTerminationStrategy::ResourceExhaustion;
        assert!(format!("{:?}", e).contains("ResourceExhaustion"));
    }

    // ── FailureHandlingStrategy ───────────────────────────────────────────

    #[test]
    fn test_failure_handling_continue_all_debug() {
        let f = FailureHandlingStrategy::ContinueAll;
        assert!(format!("{:?}", f).contains("ContinueAll"));
    }

    #[test]
    fn test_failure_handling_stop_dependent_debug() {
        let f = FailureHandlingStrategy::StopDependent;
        assert!(format!("{:?}", f).contains("StopDependent"));
    }

    // ── TestGroupingStrategy ──────────────────────────────────────────────

    #[test]
    fn test_grouping_strategy_by_category_debug() {
        let g = TestGroupingStrategy::ByCategory;
        assert!(format!("{:?}", g).contains("ByCategory"));
    }

    #[test]
    fn test_grouping_strategy_by_resource_debug() {
        let g = TestGroupingStrategy::ByResource;
        assert!(format!("{:?}", g).contains("ByResource"));
    }

    #[test]
    fn test_grouping_strategy_custom_carries_name() {
        let g = TestGroupingStrategy::Custom("my_grouping".to_string());
        assert!(format!("{:?}", g).contains("Custom"));
    }

    // ── WarmupOptimizationConfig ──────────────────────────────────────────

    #[test]
    fn test_warmup_optimization_config_fields() {
        let cfg = WarmupOptimizationConfig {
            enabled: true,
            warmup_iterations: 3,
            warmup_timeout: Duration::from_secs(30),
            cache_warmup: true,
            parallel_warmup: false,
        };
        assert!(cfg.enabled);
        assert_eq!(cfg.warmup_iterations, 3);
        assert!(!cfg.parallel_warmup);
    }

    // ── ResourceSharingCapabilities ───────────────────────────────────────

    #[test]
    fn test_resource_sharing_capabilities_all_enabled() {
        let caps = ResourceSharingCapabilities {
            cpu_sharing: true,
            memory_sharing: true,
            gpu_sharing: false,
            network_sharing: true,
            filesystem_sharing: false,
        };
        assert!(caps.cpu_sharing);
        assert!(caps.memory_sharing);
        assert!(!caps.gpu_sharing);
        assert!(!caps.filesystem_sharing);
    }

    // ── ScalabilityMetrics ────────────────────────────────────────────────

    #[test]
    fn test_scalability_metrics_fields() {
        let sm = ScalabilityMetrics {
            optimal_concurrency: 8,
            efficiency_curve: vec![(1, 1.0), (2, 0.95), (4, 0.85)],
            bottleneck_resources: vec!["cpu".to_string()],
            scalability_score: 0.85_f32,
        };
        assert_eq!(sm.optimal_concurrency, 8);
        assert_eq!(sm.efficiency_curve.len(), 3);
        assert!(!sm.bottleneck_resources.is_empty());
    }

    // ── CpuScalingConfig ──────────────────────────────────────────────────

    #[test]
    fn test_cpu_scaling_config_fields() {
        let cfg = CpuScalingConfig {
            enabled: true,
            min_cpu_utilization: 0.2_f32,
            max_cpu_utilization: 0.95_f32,
            scaling_factor: 1.5_f32,
            adjustment_interval: Duration::from_secs(10),
        };
        assert!(cfg.enabled);
        assert!((cfg.min_cpu_utilization - 0.2).abs() < 1e-6);
        assert!((cfg.max_cpu_utilization - 0.95).abs() < 1e-6);
    }

    // ── ConflictResolutionConfig ──────────────────────────────────────────

    #[test]
    fn test_conflict_resolution_config_fields() {
        let cfg = ConflictResolutionConfig {
            detection_strategy: ConflictDetectionStrategy::Hybrid,
            resolution_strategy: ConflictResolutionStrategy::Queue,
            conflict_timeout: Duration::from_secs(5),
            max_resolution_attempts: 3,
        };
        assert_eq!(cfg.max_resolution_attempts, 3);
        assert_eq!(cfg.conflict_timeout, Duration::from_secs(5));
    }

    // ── DatabaseIsolationStrategy ─────────────────────────────────────────

    #[test]
    fn test_database_isolation_strategy_per_test_debug() {
        let s = DatabaseIsolationStrategy::PerTest;
        assert!(format!("{:?}", s).contains("PerTest"));
    }

    #[test]
    fn test_database_isolation_strategy_shared_transaction_debug() {
        let s = DatabaseIsolationStrategy::SharedTransaction;
        assert!(format!("{:?}", s).contains("SharedTransaction"));
    }

    // ── ConflictDetectionStrategy ─────────────────────────────────────────

    #[test]
    fn test_conflict_detection_static_debug() {
        let s = ConflictDetectionStrategy::Static;
        assert!(format!("{:?}", s).contains("Static"));
    }

    #[test]
    fn test_conflict_detection_runtime_debug() {
        let s = ConflictDetectionStrategy::Runtime;
        assert!(format!("{:?}", s).contains("Runtime"));
    }

    #[test]
    fn test_conflict_detection_hybrid_debug() {
        let s = ConflictDetectionStrategy::Hybrid;
        assert!(format!("{:?}", s).contains("Hybrid"));
    }

    // ── TestBatchingConfig ────────────────────────────────────────────────

    #[test]
    fn test_batching_config_fields() {
        let cfg = TestBatchingConfig {
            enabled: true,
            optimal_batch_size: 10,
            max_batch_size: 50,
            batching_strategy: BatchingStrategy::ByCategory,
            batch_timeout: Duration::from_secs(60),
        };
        assert!(cfg.enabled);
        assert_eq!(cfg.optimal_batch_size, 10);
        assert_eq!(cfg.max_batch_size, 50);
    }

    // ── ResourceCleanupConfig ─────────────────────────────────────────────

    #[test]
    fn test_resource_cleanup_config_force_shutdown() {
        let cfg = ResourceCleanupConfig {
            enabled: true,
            cleanup_interval: Duration::from_secs(30),
            cleanup_strategies: std::collections::HashMap::new(),
            force_cleanup_on_shutdown: true,
        };
        assert!(cfg.enabled);
        assert!(cfg.force_cleanup_on_shutdown);
    }

    // ── ParallelizationMetrics ────────────────────────────────────────────

    #[test]
    fn test_parallelization_metrics_efficiency_bounds() {
        let sm = ScalabilityMetrics {
            optimal_concurrency: 4,
            efficiency_curve: vec![],
            bottleneck_resources: vec![],
            scalability_score: 0.9_f32,
        };
        let pm = ParallelizationMetrics {
            concurrent_tests: 4,
            parallel_efficiency: 0.82_f32,
            resource_contention: false,
            load_balancing_effectiveness: 0.9_f32,
            scheduling_overhead: Duration::from_millis(10),
            total_overhead: Duration::from_millis(25),
            speedup_factor: 3.2_f32,
            scalability_metrics: sm,
        };
        assert!(pm.parallel_efficiency >= 0.0 && pm.parallel_efficiency <= 1.0);
        assert!(pm.speedup_factor > 0.0);
    }
}
