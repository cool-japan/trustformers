//! Performance Optimization for Test Parallelization
//!
//! This module has been refactored into a modular architecture for better maintainability.
//! All functionality has been organized into focused sub-modules while maintaining
//! backward compatibility through comprehensive re-exports.
//!
//! # Architecture Overview
//!
//! The performance optimization system is organized into the following modules:
//! - `types`: Core types, enums, and data structures for performance optimization
//! - `adaptive_parallelism`: Adaptive parallelism control and estimation
//! - `system_models`: Hardware resource modeling (CPU, Memory, IO, Network, GPU)
//! - `resource_modeling`: Advanced system resource modeling and performance profiling
//! - `performance_modeling`: Machine learning models for performance prediction and optimization
//! - `feedback_systems`: Performance feedback and machine learning models
//! - `optimization_history`: Performance tracking, trending, and history management
//! - `recommendations`: Optimization recommendation generation and scoring
//! - `manager`: Main PerformanceOptimizer orchestrating all components
//! - `real_time_metrics`: Real-time metrics collection and monitoring
//! - `test_characterization`: Test characteristics analysis and profiling
//!
//! # Usage
//!
//! All previous functionality remains available at the same import paths:
//!
//! ```rust
//! use trustformers_serve::performance_optimizer_backup::{
//!     PerformanceOptimizer, PerformanceOptimizationConfig,
//!     AdaptiveParallelismController, OptimalParallelismEstimator
//! };
//! ```

// Import the modular structure
pub mod performance_optimizer;

// Re-export everything to maintain backward compatibility
pub use performance_optimizer::*;

// Legacy re-exports for backward compatibility
pub use performance_optimizer::{
    // Core types and configuration
    PerformanceOptimizationConfig, OptimizationLevel, AdaptiveLearningConfig,
    ResourceOptimizationConfig, LoadBalancingConfig, CacheOptimizationConfig,
    BatchingConfig, WarmupConfig, MonitoringConfig, AlertConfig,
    ThresholdConfig, SamplingConfig, ProfilerIntegrationConfig,

    // Main service types
    PerformanceOptimizer, OptimizationError, OptimizationResult,

    // Adaptive parallelism types
    AdaptiveParallelismController, OptimalParallelismEstimator,
    ParallelismEstimate, PerformanceModel, AdaptiveLearningModel,
    TrainingDataset, PerformanceFeedback,

    // System resource modeling types
    SystemResourceModel, CpuModel, MemoryModel, IoModel,
    NetworkModel, GpuModel, ResourceModelingManager,
    PerformanceProfiler, TemperatureMonitor, TopologyAnalyzer,
    ResourceUtilizationTracker, HardwareDetector,

    // Performance modeling types
    PerformancePrediction, ModelAccuracyMetrics, PerformanceModelFactory,
    FeatureEngineering, CrossValidationStrategy, HoldoutValidationStrategy,
    BootstrapValidationStrategy, AdaptiveGradientDescent,

    // Feedback system types
    PerformanceFeedbackSystem, FeedbackProcessor, QualityAssessment,
    AggregationStrategy, RecommendationEngine, EventSystem,
    ValidationFramework,

    // Optimization history types
    PerformanceSnapshot, OptimizationHistory, OptimizationEvent,
    PerformanceTrend, OptimizationStatistics, OptimizationTracker,
    PatternRecognitionEngine, PredictiveAnalyticsEngine,

    // Real-time metrics types
    RealTimeMetricsCollector, LiveOptimizationEngine, ThresholdMonitor,
    MetricsAggregator, PerformanceAlerting, StreamingAnalytics,

    // Test characterization types
    TestCharacterizationProfiler, TestMetricsCollector, ResourceIntensityAnalyzer,
    PerformancePatternDetector, TestOptimizationRecommender, ExecutionResourceManager,
    DefaultResourceMonitor, ResourceUsageSnapshot,

    // Utility types
    PerformanceMetrics, ResourceUsage, SystemInfo, OptimizationRecommendations,
    ResourceOptimizationRecommendation, BatchingRecommendation,
};

// Legacy compatibility types (maintain exact same interface as before)
pub type PerformanceOptimizerLegacy = PerformanceOptimizer;

// Legacy initialization functions for backward compatibility
pub use performance_optimizer::{
    init_performance_optimizer, init_performance_optimizer_with_config,
    validate_optimization_config, get_optimization_capabilities
};

// Additional convenience functions

/// Create a default performance optimizer
pub fn create_default_performance_optimizer() -> Result<PerformanceOptimizer, OptimizationError> {
    let config = PerformanceOptimizationConfig::default();
    PerformanceOptimizer::new(config)
}

/// Create a performance optimizer optimized for high-throughput scenarios
pub fn create_high_throughput_optimizer() -> Result<PerformanceOptimizer, OptimizationError> {
    let mut config = PerformanceOptimizationConfig::default();
    config.optimization_level = OptimizationLevel::Aggressive;
    config.adaptive_learning.enable_adaptive_learning = true;
    config.resource_optimization.enable_cpu_scaling = true;
    config.resource_optimization.enable_memory_optimization = true;
    config.load_balancing.enable_dynamic_load_balancing = true;
    PerformanceOptimizer::new(config)
}

/// Create a performance optimizer optimized for resource-constrained environments
pub fn create_resource_constrained_optimizer() -> Result<PerformanceOptimizer, OptimizationError> {
    let mut config = PerformanceOptimizationConfig::default();
    config.optimization_level = OptimizationLevel::Conservative;
    config.resource_optimization.enable_memory_optimization = true;
    config.resource_optimization.memory_threshold = 0.7; // Conservative memory usage
    config.monitoring.enable_resource_monitoring = true;
    PerformanceOptimizer::new(config)
}

/// Create a performance optimizer with machine learning enabled
pub fn create_ml_enhanced_optimizer() -> Result<PerformanceOptimizer, OptimizationError> {
    let mut config = PerformanceOptimizationConfig::default();
    config.adaptive_learning.enable_adaptive_learning = true;
    config.adaptive_learning.learning_rate = 0.01;
    config.adaptive_learning.enable_pattern_recognition = true;
    config.adaptive_learning.enable_predictive_scaling = true;
    PerformanceOptimizer::new(config)
}

/// Create a performance optimizer with custom parallelism settings
pub fn create_custom_parallelism_optimizer(
    target_parallelism: usize,
    max_parallelism: usize,
) -> Result<PerformanceOptimizer, OptimizationError> {
    let mut config = PerformanceOptimizationConfig::default();
    config.adaptive_learning.target_parallelism = target_parallelism;
    config.adaptive_learning.max_parallelism = max_parallelism;
    config.adaptive_learning.enable_adaptive_learning = true;
    PerformanceOptimizer::new(config)
}

/// Get performance optimizer capabilities
pub fn get_optimizer_capabilities() -> OptimizerCapabilities {
    get_performance_optimizer_capabilities()
}

/// Validate that the performance optimizer is properly configured and functional
pub fn validate_performance_optimizer_system() -> Result<OptimizerValidationReport, OptimizationError> {
    let config = PerformanceOptimizationConfig::default();
    validate_optimization_config(&config)?;

    let optimizer = PerformanceOptimizer::new(config)?;

    // Test basic optimizer operations
    let health = optimizer.health_check();
    let capabilities = optimizer.get_capabilities();

    let validation_passed = matches!(health.overall_health, HealthStatus::Healthy) &&
                           capabilities.adaptive_parallelism;

    Ok(OptimizerValidationReport {
        validation_passed,
        optimizer_health: health.overall_health,
        capabilities,
        validation_errors: vec![],
        recommendations: health.recommendations,
    })
}

/// Performance optimizer validation report
#[derive(Debug, Clone)]
pub struct OptimizerValidationReport {
    pub validation_passed: bool,
    pub optimizer_health: HealthStatus,
    pub capabilities: SystemCapabilities,
    pub validation_errors: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Performance optimizer capabilities
#[derive(Debug, Clone)]
pub struct OptimizerCapabilities {
    pub supported_optimization_levels: Vec<String>,
    pub supported_algorithms: Vec<String>,
    pub adaptive_parallelism: bool,
    pub machine_learning_support: bool,
    pub real_time_monitoring: bool,
    pub resource_optimization: bool,
    pub load_balancing: bool,
    pub predictive_scaling: bool,
    pub pattern_recognition: bool,
    pub performance_modeling: bool,
    pub optimization_recommendations: bool,
    pub export_formats: Vec<String>,
}

/// Get performance optimizer capabilities
pub fn get_performance_optimizer_capabilities() -> OptimizerCapabilities {
    OptimizerCapabilities {
        supported_optimization_levels: vec![
            "Conservative".to_string(),
            "Moderate".to_string(),
            "Aggressive".to_string(),
            "Extreme".to_string(),
        ],
        supported_algorithms: vec![
            "AdaptiveParallelism".to_string(),
            "ResourceOptimization".to_string(),
            "LoadBalancing".to_string(),
            "PredictiveScaling".to_string(),
            "PatternRecognition".to_string(),
            "PerformanceModeling".to_string(),
        ],
        adaptive_parallelism: true,
        machine_learning_support: true,
        real_time_monitoring: true,
        resource_optimization: true,
        load_balancing: true,
        predictive_scaling: true,
        pattern_recognition: true,
        performance_modeling: true,
        optimization_recommendations: true,
        export_formats: vec![
            "JSON".to_string(),
            "CSV".to_string(),
            "Binary".to_string(),
            "Metrics".to_string(),
        ],
    }
}

/// Utility functions for common optimization patterns

/// Quick performance assessment for immediate decision making
pub fn quick_performance_assessment() -> Result<SystemHealth, OptimizationError> {
    let optimizer = create_default_performance_optimizer()?;
    let snapshot = optimizer.take_snapshot()?;
    Ok(optimizer.assess_system_health(&snapshot))
}

/// Start optimization with smart defaults based on system characteristics
pub fn start_smart_optimization() -> Result<PerformanceOptimizer, OptimizationError> {
    let system_info = crate::system_utils::SystemUtils::detect_system_capabilities()?;

    let optimizer = match system_info.performance_tier {
        crate::system_utils::PerformanceTier::High => create_high_throughput_optimizer()?,
        crate::system_utils::PerformanceTier::Medium => create_default_performance_optimizer()?,
        crate::system_utils::PerformanceTier::Low => create_resource_constrained_optimizer()?,
    };

    Ok(optimizer)
}

/// Get optimization recommendations based on current system state
pub fn get_optimization_recommendations() -> Result<Vec<OptimizationRecommendation>, OptimizationError> {
    let optimizer = create_default_performance_optimizer()?;
    let snapshot = optimizer.take_snapshot()?;
    optimizer.analyze_performance(&snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_optimizer_creation() {
        let optimizer = create_default_performance_optimizer();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_high_throughput_optimizer() {
        let optimizer = create_high_throughput_optimizer();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_resource_constrained_optimizer() {
        let optimizer = create_resource_constrained_optimizer();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_ml_enhanced_optimizer() {
        let optimizer = create_ml_enhanced_optimizer();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_custom_parallelism_optimizer() {
        let optimizer = create_custom_parallelism_optimizer(4, 16);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_optimizer_capabilities() {
        let capabilities = get_optimizer_capabilities();
        assert!(!capabilities.supported_optimization_levels.is_empty());
        assert!(capabilities.adaptive_parallelism);
        assert!(capabilities.machine_learning_support);
        assert!(capabilities.real_time_monitoring);
        assert!(capabilities.resource_optimization);
    }

    #[test]
    fn test_validation_system() {
        let report = validate_performance_optimizer_system();
        assert!(report.is_ok());

        if let Ok(validation) = report {
            assert!(validation.validation_passed);
        }
    }

    #[test]
    fn test_quick_performance_assessment() {
        let assessment = quick_performance_assessment();
        assert!(assessment.is_ok());
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that old code patterns still work
        let config = PerformanceOptimizationConfig::default();
        let optimizer = PerformanceOptimizer::new(config);
        assert!(optimizer.is_ok());

        // Test legacy type alias
        if let Ok(optimizer) = optimizer {
            let _legacy_optimizer: PerformanceOptimizerLegacy = optimizer;
        }
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together seamlessly
        let optimizer = create_default_performance_optimizer().unwrap();

        let snapshot = optimizer.take_snapshot();
        assert!(snapshot.is_ok());

        let health = optimizer.health_check();
        assert!(matches!(health.overall_health, HealthStatus::Healthy));
    }

    #[test]
    fn test_optimizer_capabilities_completeness() {
        let capabilities = get_performance_optimizer_capabilities();

        // Verify all expected optimization levels are supported
        let expected_levels = vec!["Conservative", "Moderate", "Aggressive", "Extreme"];
        for level in expected_levels {
            assert!(capabilities.supported_optimization_levels.contains(&level.to_string()));
        }

        // Verify all expected algorithms are supported
        let expected_algorithms = vec![
            "AdaptiveParallelism", "ResourceOptimization", "LoadBalancing",
            "PredictiveScaling", "PatternRecognition", "PerformanceModeling"
        ];
        for algorithm in expected_algorithms {
            assert!(capabilities.supported_algorithms.contains(&algorithm.to_string()));
        }

        // Verify all expected export formats are supported
        let expected_formats = vec!["JSON", "CSV", "Binary", "Metrics"];
        for format in expected_formats {
            assert!(capabilities.export_formats.contains(&format.to_string()));
        }
    }

    #[test]
    fn test_smart_optimization_initialization() {
        let result = start_smart_optimization();
        assert!(result.is_ok());
    }

    #[test]
    fn test_optimization_recommendations() {
        let recommendations = get_optimization_recommendations();
        assert!(recommendations.is_ok());
    }

    #[test]
    fn test_validation_report_structure() {
        let config = PerformanceOptimizationConfig::default();
        let optimizer = PerformanceOptimizer::new(config).unwrap();
        let health = optimizer.health_check();
        let capabilities = optimizer.get_capabilities();

        let report = OptimizerValidationReport {
            validation_passed: true,
            optimizer_health: health.overall_health,
            capabilities,
            validation_errors: vec![],
            recommendations: health.recommendations,
        };

        assert!(report.validation_passed);
        assert!(matches!(report.optimizer_health, HealthStatus::Healthy));
    }
}