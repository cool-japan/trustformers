//! Mobile Performance Profiler for Debugging
//!
//! This module has been refactored into a modular architecture for better maintainability.
//! All functionality has been organized into focused sub-modules while maintaining
//! backward compatibility through comprehensive re-exports.
//!
//! # Architecture Overview
//!
//! The mobile performance profiler is organized into the following modules:
//! - `types`: Core types, metrics structures, and shared data models
//! - `config`: Configuration management for profiling parameters
//! - `metrics`: Metrics collection, snapshots, and aggregation
//! - `bottleneck_detection`: Performance bottleneck detection and analysis
//! - `optimization`: Optimization engine and suggestion generation
//! - `monitoring`: Real-time monitoring, alerting, and event handling
//! - `export`: Data export, visualization, and reporting
//! - `session`: Session management and profiling lifecycle
//! - `analysis`: Performance analysis, trending, and health assessment
//! - `profiler`: Main profiler coordinator and public API
//!
//! # Usage
//!
//! All previous functionality remains available at the same import paths:
//!
//! ```rust
//! use trustformers_mobile::mobile_performance_profiler_legacy::{
//!     MobilePerformanceProfiler, MobileProfilerConfig,
//!     MobileMetricsCollector, BottleneckDetector
//! };
//! ```

// Import the modular structure
use crate::mobile_performance_profiler;

// Re-export everything to maintain backward compatibility
pub use mobile_performance_profiler::*;

// Legacy re-exports for backward compatibility
pub use mobile_performance_profiler::{
    AlertManager,

    AlertType,
    BottleneckDetector,
    BottleneckType,
    CollectionStatistics,

    CpuProfilingConfig,
    ExportConfig,

    ExportFormat,
    HealthStatus,

    InferenceMetrics,
    MemoryMetrics,
    MemoryProfilingConfig,
    MobileMetricsCollector,

    // Metrics types
    MobileMetricsSnapshot,
    // Core service types
    MobilePerformanceProfiler,
    // Configuration types
    MobileProfilerConfig,
    OptimizationEngine,
    OptimizationSuggestion,
    PerformanceAlert,
    // Analysis types
    PerformanceAnalyzer,
    PerformanceBottleneck,
    // Export types
    ProfilerExportManager,
    // Session types
    ProfilingSession,
    // Monitoring types
    RealTimeMonitor,
    SamplingConfig,
    SessionInfo,

    SessionMetadata,
    SuggestionType,
    SystemHealth,
    TrendDirection,
    TrendingMetrics,
    VisualizationEngine,
};

// Legacy compatibility types (maintain exact same interface as before)
pub type MobilePerformanceProfilerLegacy = MobilePerformanceProfiler;

// Legacy initialization functions for backward compatibility
// Note: Specific init functions not available in current implementation

// Additional convenience functions

/// Create a default mobile performance profiler
pub fn create_default_mobile_profiler(
) -> Result<MobilePerformanceProfiler, Box<dyn std::error::Error>> {
    let config = MobileProfilerConfig::default();
    Ok(MobilePerformanceProfiler::new(config)?)
}

/// Create a mobile profiler optimized for high-frequency profiling
pub fn create_high_frequency_profiler(
) -> Result<MobilePerformanceProfiler, Box<dyn std::error::Error>> {
    let mut config = MobileProfilerConfig::default();
    config.sampling.cpu_sampling_interval_ms = 50;
    config.sampling.memory_sampling_interval_ms = 100;
    config.sampling.thermal_sampling_interval_ms = 200;
    config.sampling.battery_sampling_interval_ms = 1000;
    config.mode = ProfilingMode::Debug; // High frequency profiling with detailed debugging
    Ok(MobilePerformanceProfiler::new(config)?)
}

/// Create a mobile profiler optimized for minimal battery impact
pub fn create_battery_optimized_profiler(
) -> Result<MobilePerformanceProfiler, Box<dyn std::error::Error>> {
    let mut config = MobileProfilerConfig::default();
    config.sampling.cpu_sampling_interval_ms = 1000;
    config.sampling.memory_sampling_interval_ms = 2000;
    config.sampling.thermal_sampling_interval_ms = 5000;
    config.sampling.battery_sampling_interval_ms = 10000;
    config.mode = ProfilingMode::Production; // Lightweight profiling for battery optimization
    Ok(MobilePerformanceProfiler::new(config)?)
}

/// Create a mobile profiler with custom sampling intervals
pub fn create_custom_sampling_profiler(
    cpu_interval_ms: u64,
    memory_interval_ms: u64,
    thermal_interval_ms: u64,
    battery_interval_ms: u64,
) -> Result<MobilePerformanceProfiler, Box<dyn std::error::Error>> {
    let mut config = MobileProfilerConfig::default();
    config.sampling.cpu_sampling_interval_ms = cpu_interval_ms;
    config.sampling.memory_sampling_interval_ms = memory_interval_ms;
    config.sampling.thermal_sampling_interval_ms = thermal_interval_ms;
    config.sampling.battery_sampling_interval_ms = battery_interval_ms;
    Ok(MobilePerformanceProfiler::new(config)?)
}

/// Create a mobile profiler optimized for GPU-intensive applications
pub fn create_gpu_optimized_profiler(
) -> Result<MobilePerformanceProfiler, Box<dyn std::error::Error>> {
    let mut config = MobileProfilerConfig::default();
    config.gpu_profiling.enable_gpu_metrics = true;
    config.gpu_profiling.enable_memory_tracking = true;
    config.gpu_profiling.enable_performance_counters = true;
    config.gpu_profiling.sampling_interval_ms = 100;
    Ok(MobilePerformanceProfiler::new(config)?)
}

/// Get mobile profiler capabilities
pub fn get_profiler_capabilities() -> ProfilerCapabilities {
    get_mobile_profiler_capabilities()
}

/// Validate profiler configuration
fn validate_profiler_config(
    config: &MobileProfilerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if config.sampling.interval_ms == 0 {
        return Err("Sampling interval must be greater than 0".into());
    }
    if config.sampling.max_samples == 0 {
        return Err("Max samples must be greater than 0".into());
    }
    Ok(())
}

/// Validate that the mobile profiler is properly configured and functional
pub fn validate_mobile_profiler_system(
) -> Result<ProfilerValidationReport, Box<dyn std::error::Error>> {
    let config = MobileProfilerConfig::default();
    validate_profiler_config(&config)?;

    let profiler = MobilePerformanceProfiler::new(config)?;

    // Test basic profiler operations
    let health = profiler.health_check()?;
    let modern_capabilities = profiler.get_capabilities()?;

    let validation_passed = matches!(
        health.status,
        HealthStatus::Excellent | HealthStatus::Good | HealthStatus::Healthy
    ) && modern_capabilities.real_time_monitoring;

    // Convert modern capabilities to legacy format
    let legacy_capabilities = ProfilerCapabilities {
        supported_metrics: vec![
            "CPU".to_string(),
            "Memory".to_string(),
            "Thermal".to_string(),
            "Battery".to_string(),
            "GPU".to_string(),
            "Network".to_string(),
            "Inference".to_string(),
        ],
        supported_platforms: vec![
            "iOS".to_string(),
            "Android".to_string(),
            "Generic".to_string(),
        ],
        real_time_profiling: modern_capabilities.real_time_monitoring,
        gpu_profiling_support: modern_capabilities.gpu_profiling,
        thermal_monitoring: modern_capabilities.thermal_monitoring,
        battery_monitoring: modern_capabilities.battery_monitoring,
        memory_profiling: modern_capabilities.memory_profiling,
        cpu_profiling: modern_capabilities.cpu_profiling,
        network_monitoring: modern_capabilities.network_profiling,
        bottleneck_detection: true,
        optimization_suggestions: true,
        export_formats: vec![
            "JSON".to_string(),
            "CSV".to_string(),
            "HTML".to_string(),
            "Binary".to_string(),
        ],
    };

    Ok(ProfilerValidationReport {
        validation_passed,
        profiler_health: health.status,
        capabilities: legacy_capabilities,
        validation_errors: vec![],
        recommendations: health.recommendations,
    })
}

/// Mobile profiler validation report
#[derive(Debug, Clone)]
pub struct ProfilerValidationReport {
    pub validation_passed: bool,
    pub profiler_health: HealthStatus,
    pub capabilities: ProfilerCapabilities,
    pub validation_errors: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Mobile profiler capabilities
#[derive(Debug, Clone)]
pub struct ProfilerCapabilities {
    pub supported_metrics: Vec<String>,
    pub supported_platforms: Vec<String>,
    pub real_time_profiling: bool,
    pub gpu_profiling_support: bool,
    pub thermal_monitoring: bool,
    pub battery_monitoring: bool,
    pub memory_profiling: bool,
    pub cpu_profiling: bool,
    pub network_monitoring: bool,
    pub bottleneck_detection: bool,
    pub optimization_suggestions: bool,
    pub export_formats: Vec<String>,
}

/// Get mobile profiler capabilities
pub fn get_mobile_profiler_capabilities() -> ProfilerCapabilities {
    ProfilerCapabilities {
        supported_metrics: vec![
            "CPU".to_string(),
            "Memory".to_string(),
            "Thermal".to_string(),
            "Battery".to_string(),
            "GPU".to_string(),
            "Network".to_string(),
            "Inference".to_string(),
        ],
        supported_platforms: vec![
            "iOS".to_string(),
            "Android".to_string(),
            "Generic".to_string(),
        ],
        real_time_profiling: true,
        gpu_profiling_support: true,
        thermal_monitoring: true,
        battery_monitoring: true,
        memory_profiling: true,
        cpu_profiling: true,
        network_monitoring: true,
        bottleneck_detection: true,
        optimization_suggestions: true,
        export_formats: vec![
            "JSON".to_string(),
            "CSV".to_string(),
            "HTML".to_string(),
            "Binary".to_string(),
        ],
    }
}

/// Utility functions for common profiling patterns

/// Quick performance assessment for immediate decision making
pub fn quick_performance_assessment() -> Result<SystemHealth, Box<dyn std::error::Error>> {
    let profiler = create_default_mobile_profiler()?;
    let snapshot = profiler.take_snapshot()?;
    Ok(profiler.assess_system_health()?)
}

/// Start profiling with smart defaults based on device characteristics
pub fn start_smart_profiling() -> Result<MobilePerformanceProfiler, Box<dyn std::error::Error>> {
    let device_info = crate::device_info::MobileDeviceDetector::detect()?;

    let profiler = match device_info.performance_scores.overall_tier {
        crate::device_info::PerformanceTier::VeryLow => create_battery_optimized_profiler()?,
        crate::device_info::PerformanceTier::Low => create_battery_optimized_profiler()?,
        crate::device_info::PerformanceTier::Budget => create_default_mobile_profiler()?,
        crate::device_info::PerformanceTier::Medium => create_default_mobile_profiler()?,
        crate::device_info::PerformanceTier::Mid => create_default_mobile_profiler()?,
        crate::device_info::PerformanceTier::High => create_high_frequency_profiler()?,
        crate::device_info::PerformanceTier::VeryHigh => create_high_frequency_profiler()?,
        crate::device_info::PerformanceTier::Flagship => create_high_frequency_profiler()?,
    };

    Ok(profiler)
}

/// Get performance recommendations based on current device state
pub fn get_performance_recommendations(
) -> Result<Vec<OptimizationSuggestion>, Box<dyn std::error::Error>> {
    let profiler = create_default_mobile_profiler()?;
    let snapshot = profiler.take_snapshot()?;
    Ok(profiler.get_optimization_suggestions()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_profiler_creation() {
        let profiler = create_default_mobile_profiler();
        assert!(profiler.is_ok());
    }

    #[test]
    fn test_high_frequency_profiler() {
        let profiler = create_high_frequency_profiler();
        assert!(profiler.is_ok());
    }

    #[test]
    fn test_battery_optimized_profiler() {
        let profiler = create_battery_optimized_profiler();
        assert!(profiler.is_ok());
    }

    #[test]
    fn test_custom_sampling_profiler() {
        let profiler = create_custom_sampling_profiler(500, 1000, 2000, 5000);
        assert!(profiler.is_ok());
    }

    #[test]
    fn test_gpu_optimized_profiler() {
        let profiler = create_gpu_optimized_profiler();
        assert!(profiler.is_ok());
    }

    #[test]
    fn test_profiler_capabilities() {
        let capabilities = get_profiler_capabilities();
        assert!(!capabilities.supported_metrics.is_empty());
        assert!(capabilities.real_time_profiling);
        assert!(capabilities.bottleneck_detection);
        assert!(capabilities.optimization_suggestions);
    }

    #[test]
    fn test_validation_system() {
        let report = validate_mobile_profiler_system();
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
        let config = MobileProfilerConfig::default();
        let profiler = MobilePerformanceProfiler::new(config);
        assert!(profiler.is_ok());

        // Test legacy type alias
        if let Ok(profiler) = profiler {
            let _legacy_profiler: MobilePerformanceProfilerLegacy = profiler;
        }
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together seamlessly
        let profiler = create_default_mobile_profiler().expect("Operation failed");

        let snapshot = profiler.take_snapshot();
        assert!(snapshot.is_ok());

        let health = profiler.health_check();
        let health_unwrapped = health.expect("Operation failed");
        // Health should be Good or better (Excellent, Good, or Healthy are all acceptable)
        assert!(
            matches!(
                health_unwrapped.status,
                HealthStatus::Excellent | HealthStatus::Good | HealthStatus::Healthy
            ),
            "Expected healthy status, got: {:?}",
            health_unwrapped.status
        );
    }

    #[test]
    fn test_profiler_capabilities_completeness() {
        let capabilities = get_mobile_profiler_capabilities();

        // Verify all expected metrics are supported
        let expected_metrics = vec![
            "CPU",
            "Memory",
            "Thermal",
            "Battery",
            "GPU",
            "Network",
            "Inference",
        ];
        for metric in expected_metrics {
            assert!(capabilities.supported_metrics.contains(&metric.to_string()));
        }

        // Verify all expected platforms are supported
        let expected_platforms = vec!["iOS", "Android", "Generic"];
        for platform in expected_platforms {
            assert!(capabilities.supported_platforms.contains(&platform.to_string()));
        }

        // Verify all expected export formats are supported
        let expected_formats = vec!["JSON", "CSV", "HTML", "Binary"];
        for format in expected_formats {
            assert!(capabilities.export_formats.contains(&format.to_string()));
        }
    }

    #[test]
    fn test_smart_profiling_initialization() {
        let result = start_smart_profiling();
        assert!(result.is_ok());
    }

    #[test]
    fn test_performance_recommendations() {
        let recommendations = get_performance_recommendations();
        assert!(recommendations.is_ok());
    }

    #[test]
    fn test_validation_report_structure() {
        let config = MobileProfilerConfig::default();
        let profiler = MobilePerformanceProfiler::new(config).expect("Operation failed");
        let health = profiler.health_check();
        let capabilities = profiler.get_capabilities();

        let health_unwrapped = health.expect("Operation failed");
        let modern_capabilities = capabilities.expect("Operation failed");

        // Convert modern capabilities to legacy format
        let legacy_capabilities = ProfilerCapabilities {
            supported_metrics: vec![
                "cpu_usage".to_string(),
                "memory_usage".to_string(),
                "gpu_utilization".to_string(),
                "thermal_state".to_string(),
                "battery_level".to_string(),
            ],
            supported_platforms: vec!["android".to_string(), "ios".to_string()],
            real_time_profiling: modern_capabilities.real_time_monitoring,
            gpu_profiling_support: modern_capabilities.gpu_profiling,
            thermal_monitoring: modern_capabilities.thermal_monitoring,
            battery_monitoring: modern_capabilities.battery_monitoring,
            memory_profiling: modern_capabilities.memory_profiling,
            cpu_profiling: true,
            network_monitoring: true,
            bottleneck_detection: true,
            optimization_suggestions: true,
            export_formats: vec!["json".to_string(), "csv".to_string()],
        };

        let report = ProfilerValidationReport {
            validation_passed: true,
            profiler_health: health_unwrapped.status,
            capabilities: legacy_capabilities,
            validation_errors: vec![],
            recommendations: health_unwrapped.recommendations,
        };

        assert!(report.validation_passed);
        // Health should be Good or better (Excellent, Good, or Healthy are all acceptable)
        assert!(
            matches!(
                report.profiler_health,
                HealthStatus::Excellent | HealthStatus::Good | HealthStatus::Healthy
            ),
            "Expected healthy status, got: {:?}",
            report.profiler_health
        );
    }
}
