//! Tests for kernel_optimizer module

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::kernel_optimizer::{
        AccessPatternType, BalancingStrategyType, BlockSizeConstraint, ConflictResolutionType,
        ConflictSeverity, ComputeIntensityCategory, ConfigurationRecommendationType,
        DependencyType, FusionType, KernelOptimizationAnalyzer, KernelOptimizationConfig,
        KernelProfileData, MemoryOptimizationRecommendationType, OccupancyLimitingFactor,
        OptimizationDirection, OptimizationPotential, PerformanceTrend, ResourceOptimizationRecommendationType,
        ResourcePressure, StrideImpact, StrideOptimizationType, SynchronizationComplexity,
        TestType,
    };

    // -------------------------------------------------------------------------
    // Helper: make a KernelProfileData with sensible values
    // -------------------------------------------------------------------------

    fn make_profile_data(occupancy: f64, compute_util: f64) -> KernelProfileData {
        KernelProfileData {
            execution_time: Duration::from_micros(100),
            grid_size: (128, 1, 1),
            block_size: (256, 1, 1),
            shared_memory_bytes: 4096,
            registers_per_thread: 32,
            occupancy,
            compute_utilization: compute_util,
            memory_bandwidth_utilization: 0.6,
            warp_efficiency: 0.9,
            memory_efficiency: 0.8,
        }
    }

    // -------------------------------------------------------------------------
    // KernelOptimizationConfig
    // -------------------------------------------------------------------------

    #[test]
    fn test_kernel_optimization_config_default() {
        let config = KernelOptimizationConfig::default();
        assert!(config.enable_launch_config_optimization);
        assert!(config.enable_memory_access_optimization);
        assert!(config.enable_kernel_fusion);
        assert!(config.enable_regression_detection);
        assert!(config.max_optimization_suggestions > 0);
        assert!(config.min_improvement_threshold > 0.0);
    }

    #[test]
    fn test_kernel_optimization_config_custom() {
        let config = KernelOptimizationConfig {
            enable_launch_config_optimization: false,
            enable_memory_access_optimization: true,
            enable_kernel_fusion: false,
            enable_regression_detection: false,
            max_optimization_suggestions: 5,
            min_improvement_threshold: 1.0,
        };
        assert!(!config.enable_launch_config_optimization);
        assert_eq!(config.max_optimization_suggestions, 5);
    }

    // -------------------------------------------------------------------------
    // KernelOptimizationAnalyzer construction
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyzer_new() {
        let analyzer = KernelOptimizationAnalyzer::new();
        assert!(analyzer.is_ok(), "KernelOptimizationAnalyzer::new() should succeed");
    }

    #[test]
    fn test_analyzer_new_stub() {
        let analyzer = KernelOptimizationAnalyzer::new_stub();
        let dbg = format!("{:?}", analyzer);
        assert!(!dbg.is_empty());
    }

    // -------------------------------------------------------------------------
    // analyze_kernel
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyze_kernel_returns_optimizations() {
        let analyzer = KernelOptimizationAnalyzer::new();
        let mut analyzer = analyzer.expect("new ok");

        let profile = make_profile_data(0.4, 0.5);
        let result = analyzer.analyze_kernel("test_kernel", profile);
        assert!(result.is_ok(), "analyze_kernel should succeed");
    }

    #[test]
    fn test_analyze_kernel_low_occupancy() {
        let mut analyzer = KernelOptimizationAnalyzer::new().expect("ok");
        let profile = make_profile_data(0.2, 0.3); // Low occupancy and compute
        let result = analyzer.analyze_kernel("low_occ_kernel", profile);
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_kernel_multiple_calls() {
        let mut analyzer = KernelOptimizationAnalyzer::new().expect("ok");
        // First call
        let profile1 = make_profile_data(0.5, 0.6);
        analyzer.analyze_kernel("kernel_a", profile1).expect("first analysis ok");
        // Second call same kernel
        let profile2 = make_profile_data(0.55, 0.65);
        let result = analyzer.analyze_kernel("kernel_a", profile2);
        assert!(result.is_ok(), "repeated analysis of same kernel should succeed");
    }

    // -------------------------------------------------------------------------
    // get_optimization_report – error for unknown kernel
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_report_unknown_kernel_fails() {
        let analyzer = KernelOptimizationAnalyzer::new().expect("ok");
        let result = analyzer.get_optimization_report("nonexistent_kernel");
        assert!(result.is_err(), "Should fail for unknown kernel");
    }

    #[test]
    fn test_get_report_after_analyze() {
        let mut analyzer = KernelOptimizationAnalyzer::new().expect("ok");
        let profile = make_profile_data(0.6, 0.7);
        analyzer.analyze_kernel("my_kernel", profile).expect("analyze ok");
        let result = analyzer.get_optimization_report("my_kernel");
        assert!(result.is_ok(), "Report should succeed after analyze");
        let report = result.expect("report ok");
        assert_eq!(report.kernel_name, "my_kernel");
    }

    // -------------------------------------------------------------------------
    // analyze_fusion_opportunities
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyze_fusion_opportunities_empty() {
        let mut analyzer = KernelOptimizationAnalyzer::new().expect("ok");
        let result = analyzer.analyze_fusion_opportunities(&[]);
        assert!(result.is_ok(), "Fusion analysis with empty sequence should succeed");
    }

    #[test]
    fn test_analyze_fusion_opportunities_single_kernel() {
        let mut analyzer = KernelOptimizationAnalyzer::new().expect("ok");
        let result = analyzer.analyze_fusion_opportunities(&["kernel_a".to_string()]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_fusion_opportunities_multiple_kernels() {
        let mut analyzer = KernelOptimizationAnalyzer::new().expect("ok");
        let kernels =
            vec!["kernel_a".to_string(), "kernel_b".to_string(), "kernel_c".to_string()];
        let result = analyzer.analyze_fusion_opportunities(&kernels);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // KernelProfileData
    // -------------------------------------------------------------------------

    #[test]
    fn test_kernel_profile_data_fields() {
        let profile = KernelProfileData {
            execution_time: Duration::from_millis(5),
            grid_size: (64, 64, 1),
            block_size: (32, 32, 1),
            shared_memory_bytes: 8192,
            registers_per_thread: 64,
            occupancy: 0.75,
            compute_utilization: 0.85,
            memory_bandwidth_utilization: 0.9,
            warp_efficiency: 0.95,
            memory_efficiency: 0.88,
        };
        assert_eq!(profile.grid_size, (64, 64, 1));
        assert!(profile.occupancy > 0.0 && profile.occupancy <= 1.0);
    }

    // -------------------------------------------------------------------------
    // OptimizationPotential
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimization_potential_fields() {
        let potential = OptimizationPotential {
            max_performance_gain: 25.0,
            total_memory_savings: 15.0,
            avg_implementation_difficulty: 2.5,
            optimization_count: 5,
            priority_score: 4.0,
        };
        assert!(potential.max_performance_gain > 0.0);
        assert_eq!(potential.optimization_count, 5);
    }

    // -------------------------------------------------------------------------
    // Enum variants: BlockSizeConstraint
    // -------------------------------------------------------------------------

    #[test]
    fn test_block_size_constraint_variants() {
        let constraints = [
            BlockSizeConstraint::MultipleOf(32),
            BlockSizeConstraint::PowerOfTwo,
            BlockSizeConstraint::MaxThreadsPerBlock(1024),
            BlockSizeConstraint::SharedMemoryLimit(49152),
            BlockSizeConstraint::RegisterLimit(65536),
        ];
        for c in &constraints {
            let dbg = format!("{:?}", c);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // AccessPatternType
    // -------------------------------------------------------------------------

    #[test]
    fn test_access_pattern_type_variants() {
        let types = [
            AccessPatternType::Sequential,
            AccessPatternType::Strided,
            AccessPatternType::Random,
            AccessPatternType::Blocked,
            AccessPatternType::Sparse,
            AccessPatternType::Irregular,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // StrideImpact
    // -------------------------------------------------------------------------

    #[test]
    fn test_stride_impact_variants() {
        let impacts = [
            StrideImpact::Optimal,
            StrideImpact::Good,
            StrideImpact::Moderate,
            StrideImpact::Poor,
            StrideImpact::Critical,
        ];
        for i in &impacts {
            let dbg = format!("{:?}", i);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // FusionType
    // -------------------------------------------------------------------------

    #[test]
    fn test_fusion_type_variants() {
        let types = [
            FusionType::ElementwiseFusion,
            FusionType::ProducerConsumerFusion,
            FusionType::LoopFusion,
            FusionType::ReductionFusion,
            FusionType::ConvolutionFusion,
            FusionType::AttentionFusion,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // DependencyType
    // -------------------------------------------------------------------------

    #[test]
    fn test_dependency_type_variants() {
        let types = [
            DependencyType::ReadAfterWrite,
            DependencyType::WriteAfterRead,
            DependencyType::WriteAfterWrite,
            DependencyType::Reduction,
            DependencyType::Broadcast,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // ConflictSeverity
    // -------------------------------------------------------------------------

    #[test]
    fn test_conflict_severity_variants() {
        let severities = [
            ConflictSeverity::None,
            ConflictSeverity::Low,
            ConflictSeverity::Medium,
            ConflictSeverity::High,
            ConflictSeverity::Severe,
        ];
        for s in &severities {
            let dbg = format!("{:?}", s);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // ComputeIntensityCategory
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_intensity_category_variants() {
        let categories = [
            ComputeIntensityCategory::MemoryBound,
            ComputeIntensityCategory::Balanced,
            ComputeIntensityCategory::ComputeBound,
        ];
        for c in &categories {
            let dbg = format!("{:?}", c);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // OptimizationDirection
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimization_direction_variants() {
        let directions = [
            OptimizationDirection::IncreaseComputeIntensity,
            OptimizationDirection::ImproveMemoryEfficiency,
            OptimizationDirection::BalanceComputeMemory,
            OptimizationDirection::OptimizeForLatency,
        ];
        for d in &directions {
            let dbg = format!("{:?}", d);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // PerformanceTrend
    // -------------------------------------------------------------------------

    #[test]
    fn test_performance_trend_variants() {
        let trends = [
            PerformanceTrend::Improving,
            PerformanceTrend::Stable,
            PerformanceTrend::Degrading,
            PerformanceTrend::Volatile,
        ];
        for t in &trends {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // ResourcePressure
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_pressure_variants() {
        let pressures = [
            ResourcePressure::Low,
            ResourcePressure::Medium,
            ResourcePressure::High,
            ResourcePressure::Critical,
        ];
        for p in &pressures {
            let dbg = format!("{:?}", p);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // OccupancyLimitingFactor
    // -------------------------------------------------------------------------

    #[test]
    fn test_occupancy_limiting_factor_variants() {
        let factors = [
            OccupancyLimitingFactor::RegisterCount,
            OccupancyLimitingFactor::SharedMemoryUsage,
            OccupancyLimitingFactor::BlockSize,
            OccupancyLimitingFactor::WarpCount,
            OccupancyLimitingFactor::None,
        ];
        for f in &factors {
            let dbg = format!("{:?}", f);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // BalancingStrategyType
    // -------------------------------------------------------------------------

    #[test]
    fn test_balancing_strategy_type_variants() {
        let types = [
            BalancingStrategyType::RegisterOptimization,
            BalancingStrategyType::SharedMemoryOptimization,
            BalancingStrategyType::BlockSizeAdjustment,
            BalancingStrategyType::ResourcePartitioning,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // ConfigurationRecommendationType
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_recommendation_type_variants() {
        let types = [
            ConfigurationRecommendationType::BlockSizeOptimization,
            ConfigurationRecommendationType::GridSizeOptimization,
            ConfigurationRecommendationType::SharedMemoryOptimization,
            ConfigurationRecommendationType::OccupancyImprovement,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // MemoryOptimizationRecommendationType
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_optimization_recommendation_type_variants() {
        let types = [
            MemoryOptimizationRecommendationType::CoalescingImprovement,
            MemoryOptimizationRecommendationType::CacheOptimization,
            MemoryOptimizationRecommendationType::StrideOptimization,
            MemoryOptimizationRecommendationType::BankConflictResolution,
            MemoryOptimizationRecommendationType::PrefetchingStrategy,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // ResourceOptimizationRecommendationType
    // -------------------------------------------------------------------------

    #[test]
    fn test_resource_optimization_recommendation_type_variants() {
        let types = [
            ResourceOptimizationRecommendationType::RegisterOptimization,
            ResourceOptimizationRecommendationType::SharedMemoryOptimization,
            ResourceOptimizationRecommendationType::OccupancyImprovement,
            ResourceOptimizationRecommendationType::ComputeIntensityBalance,
            ResourceOptimizationRecommendationType::ResourceLoadBalancing,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // SynchronizationComplexity
    // -------------------------------------------------------------------------

    #[test]
    fn test_synchronization_complexity_variants() {
        let complexities = [
            SynchronizationComplexity::None,
            SynchronizationComplexity::Moderate,
            SynchronizationComplexity::Complex,
            SynchronizationComplexity::Prohibitive,
        ];
        for c in &complexities {
            let dbg = format!("{:?}", c);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // ConflictResolutionType
    // -------------------------------------------------------------------------

    #[test]
    fn test_conflict_resolution_type_variants() {
        let types = [
            ConflictResolutionType::ArrayPadding,
            ConflictResolutionType::AccessReordering,
            ConflictResolutionType::DataStructureReorganization,
            ConflictResolutionType::BroadcastOptimization,
            ConflictResolutionType::MemoryLayoutChange,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // TestType
    // -------------------------------------------------------------------------

    #[test]
    fn test_test_type_variants() {
        let types = [
            TestType::TTest,
            TestType::MannWhitneyU,
            TestType::KolmogorovSmirnov,
            TestType::ChangePointDetection,
            TestType::AnomalyDetection,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // StrideOptimizationType
    // -------------------------------------------------------------------------

    #[test]
    fn test_stride_optimization_type_variants() {
        let types = [
            StrideOptimizationType::DataLayoutReorganization,
            StrideOptimizationType::AccessReordering,
            StrideOptimizationType::TilingStrategy,
            StrideOptimizationType::PrefetchingStrategy,
            StrideOptimizationType::VectorizedAccess,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // Profile execution time tracking
    // -------------------------------------------------------------------------

    #[test]
    fn test_report_kernel_name_matches() {
        let mut analyzer = KernelOptimizationAnalyzer::new().expect("ok");
        let profile = make_profile_data(0.7, 0.8);
        analyzer.analyze_kernel("my_specific_kernel", profile).expect("ok");
        let report = analyzer.get_optimization_report("my_specific_kernel").expect("report ok");
        assert_eq!(report.kernel_name, "my_specific_kernel");
    }
}
