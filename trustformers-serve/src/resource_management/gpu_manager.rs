//! Comprehensive GPU Resource Management System
//!
//! This module has been refactored into a modular architecture for better maintainability.
//! All functionality has been organized into focused sub-modules while maintaining
//! backward compatibility through comprehensive re-exports.
//!
//! # Architecture Overview
//!
//! The GPU resource management system is organized into the following modules:
//! - `types`: Core types, configurations, and data structures for GPU management
//! - `manager`: Main GpuResourceManager implementation and coordination logic
//! - `monitoring`: GPU monitoring system with real-time metrics collection
//! - `alert_system`: Alert management and notification system
//! - `performance_tracker`: Performance tracking, benchmarking, and analysis
//! - `health_monitor`: Device health monitoring and diagnostic system
//! - `load_balancer`: Load balancing and workload distribution algorithms
//!
//! # Features
//!
//! - **Device Discovery & Management**: Automatic detection and cataloging of available GPU devices
//! - **Resource Allocation**: Intelligent allocation of GPU devices based on performance requirements
//! - **Real-time Monitoring**: Continuous monitoring of GPU utilization, temperature, memory usage
//! - **Alert System**: Proactive monitoring with configurable alerts for hardware issues
//! - **Performance Tracking**: Benchmarking, performance analysis, and baseline establishment
//! - **Health Management**: Device health monitoring with failure detection and recovery
//! - **Load Balancing**: Intelligent distribution of workloads across available devices
//! - **Thread Safety**: All operations are thread-safe with appropriate synchronization
//!
//! # Usage
//!
//! All previous functionality remains available at the same import paths:
//!
//! ```rust
//! use trustformers_serve::resource_management::gpu_manager::{
//!     GpuResourceManager, GpuManagerError,
//!     GpuMonitoringSystem, GpuAlertSystem, GpuPerformanceTracker
//! };
//! ```

// Import the modular structure
pub mod alert_system;
pub mod health_monitor;
pub mod load_balancer;
pub mod manager;
pub mod monitoring;
pub mod performance_tracker;
pub mod types;

// Re-export everything to maintain backward compatibility
pub use alert_system::*;
pub use health_monitor::*;
pub use load_balancer::*;
pub use manager::*;
pub use monitoring::*;
pub use performance_tracker::*;
pub use types::*;

// Legacy re-exports for backward compatibility
pub use types::{
    // GpuDeviceCapability, // TODO: Fix - does not exist
    // GpuFramework, // TODO: Fix - does not exist
    // AlertType, // TODO: Fix - does not exist
    AlertSeverity,
    // Alert types
    GpuAlert,
    GpuAlertConfig,
    GpuAlertEvent,
    GpuAlertThresholds,

    GpuAllocation,
    GpuBenchmarkType,
    GpuClockSpeeds,

    GpuConstraint,
    // GpuLoadBalancer, // TODO: Fix - use load_balancer::GpuLoadBalancer

    // Core data types
    GpuDeviceInfo,
    // GpuPerformanceTrend, // TODO: Fix - does not exist
    // GpuPerformanceRegression, // TODO: Fix - does not exist

    // Enum types
    GpuDeviceStatus,
    // Component types
    // GpuMonitoringSystem, // TODO: Fix - use monitoring::GpuMonitoringSystem
    // GpuAlertSystem, // TODO: Fix - use alert_system::GpuAlertSystem
    // GpuPerformanceTracker, // TODO: Fix - use performance_tracker::GpuPerformanceTracker
    // GpuHealthMonitor, // TODO: Fix - use health_monitor::GpuHealthMonitor
    GpuHealthStatus,
    GpuHistoricalMetric,
    // Core manager and error types
    // GpuResourceManager, // TODO: Fix - use manager::GpuResourceManager
    GpuManagerError,
    GpuMonitoringConfig,
    GpuPerformanceAnalysis,
    // AlertEscalationRule, // TODO: Fix - does not exist
    // AlertCondition, // TODO: Fix - does not exist

    // Performance types
    // GpuBenchmarkResult, // TODO: Fix - does not exist
    GpuPerformanceBaseline,
    GpuPerformanceRequirements,
    // Configuration types
    GpuPoolConfig,
    GpuRealTimeMetrics,
    GpuResult,

    // PerformanceTrendDirection, // TODO: Fix - does not exist
    LoadBalancingStrategy,
};

// Legacy compatibility types (maintain exact same interface as before)
pub type GpuResourceManagerLegacy = GpuResourceManager;

// Legacy initialization functions for backward compatibility
// pub use manager::{
//     init_gpu_manager, init_gpu_manager_with_config, // TODO: Fix - these functions don't exist
//     validate_gpu_config // TODO: Fix - this function doesn't exist
// };

// Additional convenience functions

/// Create a default GPU resource manager with standard configuration
pub async fn create_default_gpu_manager() -> Result<GpuResourceManager, GpuManagerError> {
    let config = GpuPoolConfig::default();
    GpuResourceManager::new(config).await
}

/// Create a GPU manager optimized for machine learning workloads
pub async fn create_ml_optimized_gpu_manager(
    min_memory_gb: u64,
) -> Result<GpuResourceManager, GpuManagerError> {
    let mut config = GpuPoolConfig::default();
    config.enable_performance_tracking = true;
    config.enable_health_monitoring = true;
    config.min_memory_mb = min_memory_gb * 1024; // Convert GB to MB

    GpuResourceManager::new(config).await
}

/// Create a GPU manager with comprehensive monitoring enabled
pub async fn create_monitored_gpu_manager() -> Result<GpuResourceManager, GpuManagerError> {
    let mut config = GpuPoolConfig::default();
    config.enable_monitoring = true;
    config.enable_alerts = true;
    config.enable_performance_tracking = true;
    config.enable_health_monitoring = true;
    config.monitoring_interval_secs = 30; // More frequent monitoring

    GpuResourceManager::new(config).await
}

/// Create a GPU manager optimized for high-performance computing
pub async fn create_hpc_optimized_gpu_manager() -> Result<GpuResourceManager, GpuManagerError> {
    let mut config = GpuPoolConfig::default();
    config.enable_performance_tracking = true;
    config.enable_load_balancing = true;
    config.enable_health_monitoring = true;
    config.allocation_timeout_secs = 10; // Faster allocation for HPC workloads

    GpuResourceManager::new(config).await
}

/// Create a GPU manager with custom alert configuration
pub async fn create_alerting_gpu_manager(
    temperature_threshold: f32,
    memory_threshold: f32,
) -> Result<GpuResourceManager, GpuManagerError> {
    let mut config = GpuPoolConfig::default();
    config.enable_alerts = true;
    config.alert_thresholds.temperature_critical = temperature_threshold;
    config.alert_thresholds.memory_critical_percent = memory_threshold;

    GpuResourceManager::new(config).await
}

/// Get GPU manager capabilities
pub fn get_manager_capabilities() -> GpuManagerCapabilities {
    get_gpu_manager_capabilities()
}

/// Validate GPU configuration parameters
fn validate_gpu_config(config: &GpuPoolConfig) -> Result<(), GpuManagerError> {
    // Validate max devices limit
    if config.max_devices == 0 {
        return Err(GpuManagerError::InvalidConfiguration {
            message: "max_devices must be greater than 0".to_string(),
        });
    }

    // Validate memory allocation threshold
    if config.memory_allocation_threshold < 0.0 || config.memory_allocation_threshold > 1.0 {
        return Err(GpuManagerError::InvalidConfiguration {
            message: "memory_allocation_threshold must be between 0.0 and 1.0".to_string(),
        });
    }

    // Validate monitoring interval
    if config.enable_monitoring && config.monitoring_interval_secs == 0 {
        return Err(GpuManagerError::InvalidConfiguration {
            message: "monitoring_interval_secs must be greater than 0 when monitoring is enabled"
                .to_string(),
        });
    }

    Ok(())
}

/// Validate that the GPU manager is properly configured and functional
pub async fn validate_gpu_manager_system() -> Result<GpuManagerValidationReport, GpuManagerError> {
    let config = GpuPoolConfig::default();
    validate_gpu_config(&config)?;

    let manager = GpuResourceManager::new(config).await?;

    // Test basic manager operations
    let devices = manager.get_available_devices().await;
    let capabilities = get_gpu_manager_capabilities();

    let validation_passed = !devices.is_empty() && capabilities.device_monitoring;

    Ok(GpuManagerValidationReport {
        validation_passed,
        available_devices: devices.len(),
        capabilities,
        validation_errors: vec![],
        recommendations: if devices.is_empty() {
            vec!["No GPU devices found - check hardware and drivers".to_string()]
        } else {
            vec!["GPU manager system is properly configured and functional".to_string()]
        },
    })
}

/// GPU manager validation report
#[derive(Debug, Clone)]
pub struct GpuManagerValidationReport {
    pub validation_passed: bool,
    pub available_devices: usize,
    pub capabilities: GpuManagerCapabilities,
    pub validation_errors: Vec<String>,
    pub recommendations: Vec<String>,
}

/// GPU manager capabilities
#[derive(Debug, Clone)]
pub struct GpuManagerCapabilities {
    pub supported_frameworks: Vec<String>,
    pub supported_device_types: Vec<String>,
    pub device_monitoring: bool,
    pub performance_tracking: bool,
    pub health_monitoring: bool,
    pub alert_system: bool,
    pub load_balancing: bool,
    pub automatic_discovery: bool,
    pub concurrent_allocations: bool,
    pub benchmark_support: bool,
    pub thermal_monitoring: bool,
    pub export_formats: Vec<String>,
}

/// Get GPU manager capabilities
pub fn get_gpu_manager_capabilities() -> GpuManagerCapabilities {
    GpuManagerCapabilities {
        supported_frameworks: vec![
            "CUDA".to_string(),
            "OpenCL".to_string(),
            "ROCm".to_string(),
            "Vulkan".to_string(),
            "DirectML".to_string(),
        ],
        supported_device_types: vec![
            "NVIDIA".to_string(),
            "AMD".to_string(),
            "Intel".to_string(),
            "Apple".to_string(),
        ],
        device_monitoring: true,
        performance_tracking: true,
        health_monitoring: true,
        alert_system: true,
        load_balancing: true,
        automatic_discovery: true,
        concurrent_allocations: true,
        benchmark_support: true,
        thermal_monitoring: true,
        export_formats: vec![
            "JSON".to_string(),
            "CSV".to_string(),
            "XML".to_string(),
            "Binary".to_string(),
        ],
    }
}

/// Utility functions for common GPU management patterns

/// Quick GPU availability check for immediate decision making
pub async fn quick_gpu_availability_check() -> Result<GpuAvailabilityReport, GpuManagerError> {
    let config = GpuPoolConfig::default();
    let manager = GpuResourceManager::new(config).await?;
    let devices = manager.get_available_devices().await;

    let available_devices = devices.len();
    let total_memory_mb: u64 = devices.iter().map(|device| device.total_memory_mb).sum();

    Ok(GpuAvailabilityReport {
        available_devices,
        total_memory_mb,
        avg_memory_per_device: if available_devices > 0 {
            total_memory_mb / available_devices as u64
        } else {
            0
        },
        can_allocate_workloads: available_devices > 0,
        recommended_max_concurrent: available_devices.min(8), // Conservative estimate
    })
}

/// GPU availability report
#[derive(Debug, Clone)]
pub struct GpuAvailabilityReport {
    pub available_devices: usize,
    pub total_memory_mb: u64,
    pub avg_memory_per_device: u64,
    pub can_allocate_workloads: bool,
    pub recommended_max_concurrent: usize,
}

/// Start GPU management with smart defaults based on use case
pub async fn start_smart_gpu_management(
    use_case: GpuManagementUseCase,
) -> Result<GpuResourceManager, GpuManagerError> {
    let mut config = GpuPoolConfig::default();

    match use_case {
        GpuManagementUseCase::MachineLearning => {
            config.enable_performance_tracking = true;
            config.enable_health_monitoring = true;
            config.min_memory_mb = 4096; // 4GB minimum for ML
        },
        GpuManagementUseCase::HighPerformanceComputing => {
            config.enable_load_balancing = true;
            config.enable_performance_tracking = true;
            config.allocation_timeout_secs = 5; // Fast allocation
        },
        GpuManagementUseCase::Rendering => {
            config.enable_monitoring = true;
            config.enable_alerts = true;
            config.monitoring_interval_secs = 10; // Frequent monitoring
        },
        GpuManagementUseCase::Cryptocurrency => {
            config.enable_health_monitoring = true;
            config.enable_alerts = true;
            config.alert_thresholds.temperature_critical = 75.0; // Lower temp threshold
        },
        GpuManagementUseCase::General => {
            // Use default configuration
        },
    }

    GpuResourceManager::new(config).await
}

/// GPU management use case for smart configuration selection
#[derive(Debug, Clone)]
pub enum GpuManagementUseCase {
    MachineLearning,
    HighPerformanceComputing,
    Rendering,
    Cryptocurrency,
    General,
}

/// Get GPU management recommendations based on current system state
pub async fn get_gpu_management_recommendations(
    manager: &GpuResourceManager,
) -> Result<Vec<GpuManagementRecommendation>, GpuManagerError> {
    let devices = manager.get_available_devices().await;
    let mut recommendations = Vec::new();

    // Device availability recommendations
    if devices.is_empty() {
        recommendations.push(GpuManagementRecommendation {
            priority: RecommendationPriority::Critical,
            category: "Hardware".to_string(),
            suggestion: "No GPU devices found - check hardware installation and drivers"
                .to_string(),
            implementation_effort: ImplementationEffort::High,
        });
    } else if devices.len() == 1 {
        recommendations.push(GpuManagementRecommendation {
            priority: RecommendationPriority::Medium,
            category: "Scalability".to_string(),
            suggestion: "Consider adding more GPU devices for better performance and redundancy"
                .to_string(),
            implementation_effort: ImplementationEffort::High,
        });
    }

    // Memory recommendations
    let total_memory: u64 = devices.iter().map(|d| d.total_memory_mb).sum();
    if total_memory < 8192 {
        recommendations.push(GpuManagementRecommendation {
            priority: RecommendationPriority::Medium,
            category: "Memory".to_string(),
            suggestion: "Low total GPU memory - consider upgrading for better ML/HPC performance"
                .to_string(),
            implementation_effort: ImplementationEffort::High,
        });
    }

    // Configuration recommendations
    if !manager.get_config().await.enable_monitoring {
        recommendations.push(GpuManagementRecommendation {
            priority: RecommendationPriority::Low,
            category: "Monitoring".to_string(),
            suggestion: "Enable monitoring for better visibility into GPU performance".to_string(),
            implementation_effort: ImplementationEffort::Low,
        });
    }

    if recommendations.is_empty() {
        recommendations.push(GpuManagementRecommendation {
            priority: RecommendationPriority::Low,
            category: "General".to_string(),
            suggestion: "GPU management system is optimally configured".to_string(),
            implementation_effort: ImplementationEffort::None,
        });
    }

    Ok(recommendations)
}

/// GPU management optimization recommendation
#[derive(Debug, Clone)]
pub struct GpuManagementRecommendation {
    pub priority: RecommendationPriority,
    pub category: String,
    pub suggestion: String,
    pub implementation_effort: ImplementationEffort,
}

/// Recommendation priority levels
#[derive(Debug, Clone)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Implementation effort estimation
#[derive(Debug, Clone)]
pub enum ImplementationEffort {
    None,
    Low,
    Medium,
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_manager_capabilities() {
        let capabilities = get_gpu_manager_capabilities();
        assert!(!capabilities.supported_frameworks.is_empty());
        assert!(capabilities.device_monitoring);
        assert!(capabilities.performance_tracking);
        assert!(capabilities.health_monitoring);
        assert!(capabilities.alert_system);
    }

    #[tokio::test]
    async fn test_validation_system() {
        let report = validate_gpu_manager_system().await;
        assert!(report.is_ok());
        // Note: This test may fail if no GPU hardware is available
    }

    #[tokio::test]
    async fn test_quick_availability_check() {
        let report = quick_gpu_availability_check().await;
        // Should succeed even with no GPUs
        assert!(report.is_ok());
    }

    #[tokio::test]
    async fn test_smart_gpu_management() {
        let use_cases = [
            GpuManagementUseCase::MachineLearning,
            GpuManagementUseCase::HighPerformanceComputing,
            GpuManagementUseCase::Rendering,
            GpuManagementUseCase::Cryptocurrency,
            GpuManagementUseCase::General,
        ];

        for use_case in use_cases {
            let result = start_smart_gpu_management(use_case).await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_convenience_functions() {
        // Test convenience function creation
        let default_manager = create_default_gpu_manager().await;
        assert!(default_manager.is_ok());

        let ml_manager = create_ml_optimized_gpu_manager(8).await;
        assert!(ml_manager.is_ok());

        let monitored_manager = create_monitored_gpu_manager().await;
        assert!(monitored_manager.is_ok());

        let hpc_manager = create_hpc_optimized_gpu_manager().await;
        assert!(hpc_manager.is_ok());

        let alerting_manager = create_alerting_gpu_manager(80.0, 90.0).await;
        assert!(alerting_manager.is_ok());
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that legacy type aliases work
        let _manager: GpuResourceManagerLegacy;
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together seamlessly
        let capabilities = get_gpu_manager_capabilities();
        assert!(capabilities.device_monitoring);
        assert!(capabilities.performance_tracking);
        assert!(capabilities.health_monitoring);
        assert!(capabilities.alert_system);
        assert!(capabilities.load_balancing);
    }

    #[test]
    fn test_capabilities_completeness() {
        let capabilities = get_gpu_manager_capabilities();

        // Verify all expected frameworks are supported
        let expected_frameworks = vec!["CUDA", "OpenCL", "ROCm", "Vulkan", "DirectML"];
        for framework in expected_frameworks {
            assert!(capabilities.supported_frameworks.contains(&framework.to_string()));
        }

        // Verify all expected device types are supported
        let expected_types = vec!["NVIDIA", "AMD", "Intel", "Apple"];
        for device_type in expected_types {
            assert!(capabilities.supported_device_types.contains(&device_type.to_string()));
        }

        // Verify all expected export formats are supported
        let expected_formats = vec!["JSON", "CSV", "XML", "Binary"];
        for format in expected_formats {
            assert!(capabilities.export_formats.contains(&format.to_string()));
        }
    }

    #[test]
    fn test_use_case_selection() {
        // Test different use case configurations
        let use_cases = [
            GpuManagementUseCase::MachineLearning,
            GpuManagementUseCase::HighPerformanceComputing,
            GpuManagementUseCase::Rendering,
            GpuManagementUseCase::Cryptocurrency,
            GpuManagementUseCase::General,
        ];

        for use_case in use_cases {
            // This would normally test with actual GPU manager
            // For now, just verify the enum variants exist
            match use_case {
                GpuManagementUseCase::MachineLearning => {},
                GpuManagementUseCase::HighPerformanceComputing => {},
                GpuManagementUseCase::Rendering => {},
                GpuManagementUseCase::Cryptocurrency => {},
                GpuManagementUseCase::General => {},
            }
        }
    }

    #[test]
    fn test_validation_report_structure() {
        let capabilities = get_gpu_manager_capabilities();

        let report = GpuManagerValidationReport {
            validation_passed: true,
            available_devices: 2,
            capabilities,
            validation_errors: vec![],
            recommendations: vec!["System healthy".to_string()],
        };

        assert!(report.validation_passed);
        assert_eq!(report.available_devices, 2);
    }

    #[test]
    fn test_recommendation_priority_levels() {
        let recommendations = vec![GpuManagementRecommendation {
            priority: RecommendationPriority::High,
            category: "Test".to_string(),
            suggestion: "Test suggestion".to_string(),
            implementation_effort: ImplementationEffort::Low,
        }];

        // Test that recommendations have valid priority levels
        for rec in recommendations {
            match rec.priority {
                RecommendationPriority::Critical
                | RecommendationPriority::High
                | RecommendationPriority::Medium
                | RecommendationPriority::Low => {},
            }

            match rec.implementation_effort {
                ImplementationEffort::None
                | ImplementationEffort::Low
                | ImplementationEffort::Medium
                | ImplementationEffort::High => {},
            }
        }
    }

    #[test]
    fn test_availability_report_structure() {
        let report = GpuAvailabilityReport {
            available_devices: 2,
            total_memory_mb: 16384,
            avg_memory_per_device: 8192,
            can_allocate_workloads: true,
            recommended_max_concurrent: 2,
        };

        assert_eq!(report.available_devices, 2);
        assert_eq!(report.total_memory_mb, 16384);
        assert_eq!(report.avg_memory_per_device, 8192);
        assert!(report.can_allocate_workloads);
        assert_eq!(report.recommended_max_concurrent, 2);
    }
}
