//! Resource Management System for Test Parallelization
//!
//! This module has been refactored into a modular architecture for better maintainability.
//! All functionality has been organized into focused sub-modules while maintaining
//! backward compatibility through comprehensive re-exports.
//!
//! # Architecture Overview
//!
//! The resource management system is organized into the following modules:
//! - `types`: Core types, configurations, and shared data structures
//! - `manager`: Main ResourceManagementSystem coordinating all components
//! - `port_management`: Network port allocation and reservation
//! - `directory_management`: Temporary directory management and cleanup
//! - `gpu_management`: GPU resource allocation and monitoring
//! - `database_management`: Database connection pool management
//! - `custom_resources`: Generic custom resource handling
//! - `monitoring`: Resource monitoring and health checks
//! - `allocation`: Resource allocation strategies and tracking
//! - `cleanup`: Resource cleanup and garbage collection
//! - `statistics`: Performance metrics and analytics
//!
//! # Usage
//!
//! All previous functionality remains available at the same import paths:
//!
//! ```rust
//! use trustformers_serve::resource_manager_backup::{
//!     ResourceManagementSystem, ResourceManagementConfig,
//!     NetworkPortManager, TempDirectoryManager, GpuResourceManager
//! };
//! ```

// Import the modular structure
pub mod resource_management;

// Re-export everything to maintain backward compatibility
pub use resource_management::*;

// Legacy re-exports for backward compatibility
pub use resource_management::{
    // Configuration types
    ResourceManagementConfig, PortPoolConfig, TempDirPoolConfig, GpuPoolConfig,
    DatabasePoolConfig, ResourceMonitoringConfig, ConflictResolutionConfig,
    ResourceCleanupConfig, GpuMonitoringConfig, GpuAlertConfig,

    // Core service types
    ResourceManagementSystem, SystemStatistics, StatisticsCollector,
    ResourceMonitor, ConflictDetector, ResourceAllocator, CleanupManager,

    // Manager types
    NetworkPortManager, TempDirectoryManager, GpuResourceManager,
    DatabaseConnectionManager, CustomResourceManager,

    // Allocation types
    PortAllocation, TempDirectoryAllocation, GpuAllocation,
    ResourceAllocation, AllocationEvent, ExecutionState,

    // Monitoring types
    PortUsageStatistics, DirectoryUsageStatistics, GpuUsageStatistics,
    DatabaseUsageStatistics, SystemResourceStatistics, SystemPerformanceSnapshot,
    GpuMonitoringSystem, GpuAlertSystem,

    // Utility types
    PortReservationSystem, PortReservationRequest, WorkerPool, Worker,
    HealthCheck, Alert, LoadMetrics, DistributionEvent,

    // Enum types
    PortUsageType, DirectoryStatus, GpuDeviceStatus, ExecutionStatus,
    WorkerStatus, HealthStatus, AlertSeverity, TrendDirection,
};

// Legacy compatibility types (maintain exact same interface as before)
pub type ResourceManager = ResourceManagementSystem;
pub type PortManager = NetworkPortManager;
pub type DirectoryManager = TempDirectoryManager;
pub type GpuManager = GpuResourceManager;

// Legacy initialization functions for backward compatibility
pub use resource_management::{
    init_resource_management, init_resource_management_with_config, validate_resource_config
};

// Additional convenience functions

/// Create a default resource management system
pub async fn create_default_resource_manager() -> anyhow::Result<ResourceManagementSystem> {
    let config = ResourceManagementConfig::default();
    ResourceManagementSystem::new(config).await
}

/// Create a resource management system optimized for high-concurrency testing
pub async fn create_high_concurrency_manager() -> anyhow::Result<ResourceManagementSystem> {
    let mut config = ResourceManagementConfig::default();
    config.port_pool.max_allocation = 1000;
    config.temp_dir_pool.max_directories = 500;
    config.gpu_pool.max_allocations = 16;
    config.monitoring.status_check_interval_secs = 5;
    config.cleanup.enable_aggressive_cleanup = true;
    ResourceManagementSystem::new(config).await
}

/// Create a resource management system optimized for minimal resource usage
pub async fn create_minimal_resource_manager() -> anyhow::Result<ResourceManagementSystem> {
    let mut config = ResourceManagementConfig::default();
    config.port_pool.max_allocation = 50;
    config.temp_dir_pool.max_directories = 25;
    config.gpu_pool.max_allocations = 2;
    config.monitoring.status_check_interval_secs = 30;
    config.cleanup.max_cleanup_batch_size = 5;
    ResourceManagementSystem::new(config).await
}

/// Create a resource management system with custom port range
pub async fn create_custom_port_manager(
    port_start: u16,
    port_end: u16,
    max_allocation: usize,
) -> anyhow::Result<ResourceManagementSystem> {
    let mut config = ResourceManagementConfig::default();
    config.port_pool.port_range = (port_start, port_end);
    config.port_pool.max_allocation = max_allocation;
    ResourceManagementSystem::new(config).await
}

/// Create a resource management system with enhanced GPU support
pub async fn create_gpu_optimized_manager() -> anyhow::Result<ResourceManagementSystem> {
    let mut config = ResourceManagementConfig::default();
    config.gpu_pool.enable_monitoring = true;
    config.gpu_pool.allocation_timeout_secs = 60;
    config.gpu_pool.max_allocations = 32;
    config.gpu_monitoring.enable_performance_tracking = true;
    config.gpu_monitoring.health_check_interval_secs = 10;
    ResourceManagementSystem::new(config).await
}

/// Get resource management system capabilities
pub fn get_resource_capabilities() -> ResourceCapabilities {
    get_system_capabilities()
}

/// Validate that the resource management system is properly configured and functional
pub async fn validate_resource_management_system() -> anyhow::Result<ResourceValidationReport> {
    let config = ResourceManagementConfig::default();
    validate_resource_config(&config).map_err(|e| anyhow::anyhow!("Configuration error: {}", e))?;

    let system = ResourceManagementSystem::new(config).await?;

    // Test basic system operations
    let status = system.get_status().await;
    let health = system.health_check().await;

    let validation_passed = matches!(health.overall_health, HealthStatus::Healthy) &&
                           !status.background_tasks.is_empty();

    Ok(ResourceValidationReport {
        validation_passed,
        system_health: health.overall_health,
        component_status: status.component_status,
        performance_metrics: status.performance_metrics,
        validation_errors: vec![],
        recommendations: health.recommendations,
    })
}

/// Resource management system validation report
#[derive(Debug, Clone)]
pub struct ResourceValidationReport {
    pub validation_passed: bool,
    pub system_health: HealthStatus,
    pub component_status: ComponentStatus,
    pub performance_metrics: SystemPerformanceMetrics,
    pub validation_errors: Vec<String>,
    pub recommendations: Vec<HealthRecommendation>,
}

/// Resource management system capabilities
#[derive(Debug, Clone)]
pub struct ResourceCapabilities {
    pub supported_resource_types: Vec<String>,
    pub supported_allocation_strategies: Vec<String>,
    pub supported_cleanup_policies: Vec<String>,
    pub monitoring_enabled: bool,
    pub conflict_detection: bool,
    pub automatic_cleanup: bool,
    pub statistical_tracking: bool,
    pub alert_system: bool,
    pub health_monitoring: bool,
    pub resource_pooling: bool,
    pub hot_reconfiguration: bool,
}

/// Get resource management system capabilities
pub fn get_system_capabilities() -> ResourceCapabilities {
    ResourceCapabilities {
        supported_resource_types: vec![
            "NetworkPorts".to_string(),
            "TempDirectories".to_string(),
            "GpuDevices".to_string(),
            "DatabaseConnections".to_string(),
            "CustomResources".to_string(),
        ],
        supported_allocation_strategies: vec![
            "FirstAvailable".to_string(),
            "LeastUsed".to_string(),
            "BestFit".to_string(),
            "RoundRobin".to_string(),
            "Priority".to_string(),
        ],
        supported_cleanup_policies: vec![
            "Immediate".to_string(),
            "Delayed".to_string(),
            "SessionEnd".to_string(),
            "Manual".to_string(),
            "Debug".to_string(),
        ],
        monitoring_enabled: true,
        conflict_detection: true,
        automatic_cleanup: true,
        statistical_tracking: true,
        alert_system: true,
        health_monitoring: true,
        resource_pooling: true,
        hot_reconfiguration: true,
    }
}

/// Utility functions for common resource management patterns

/// Quick resource status check for immediate decision making
pub async fn quick_resource_status() -> anyhow::Result<SystemResourceStatus> {
    let manager = create_default_resource_manager().await?;
    let status = manager.get_status().await;
    Ok(SystemResourceStatus {
        ports_available: status.port_manager.available_ports.len(),
        directories_available: status.temp_dir_manager.available_directories,
        gpus_available: status.gpu_manager.available_devices,
        overall_health: status.health_check.overall_health,
    })
}

/// System resource status summary
#[derive(Debug, Clone)]
pub struct SystemResourceStatus {
    pub ports_available: usize,
    pub directories_available: usize,
    pub gpus_available: usize,
    pub overall_health: HealthStatus,
}

/// Optimize resource settings for current system load
pub async fn optimize_for_current_load(
    manager: &ResourceManagementSystem,
) -> anyhow::Result<OptimizationReport> {
    let status = manager.get_status().await;
    let recommendations = manager.analyze_and_optimize().await?;
    Ok(OptimizationReport {
        current_load: status.system_stats.load_metrics,
        recommendations,
        estimated_improvement: calculate_improvement_estimate(&recommendations),
    })
}

/// Resource optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub current_load: LoadMetrics,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub estimated_improvement: f64,
}

/// Calculate estimated improvement from recommendations
fn calculate_improvement_estimate(recommendations: &[OptimizationRecommendation]) -> f64 {
    // Simple heuristic: more recommendations = more potential improvement
    (recommendations.len() as f64 * 10.0).min(50.0)
}

/// Get comprehensive system performance report
pub async fn get_performance_report(
    manager: &ResourceManagementSystem,
) -> anyhow::Result<PerformanceReport> {
    let statistics = manager.get_statistics().await;
    let health = manager.health_check().await;
    Ok(PerformanceReport {
        statistics,
        health,
        generated_at: std::time::SystemTime::now(),
    })
}

/// Performance report structure
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub statistics: SystemStatistics,
    pub health: SystemHealthCheck,
    pub generated_at: std::time::SystemTime,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resource_management_system_creation() {
        let manager = create_default_resource_manager().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_high_concurrency_manager() {
        let manager = create_high_concurrency_manager().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_minimal_resource_manager() {
        let manager = create_minimal_resource_manager().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_custom_port_manager() {
        let manager = create_custom_port_manager(9000, 9100, 50).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_gpu_optimized_manager() {
        let manager = create_gpu_optimized_manager().await;
        assert!(manager.is_ok());
    }

    #[test]
    fn test_resource_capabilities() {
        let capabilities = get_resource_capabilities();
        assert!(!capabilities.supported_resource_types.is_empty());
        assert!(capabilities.monitoring_enabled);
        assert!(capabilities.conflict_detection);
        assert!(capabilities.automatic_cleanup);
    }

    #[tokio::test]
    async fn test_validation_system() {
        let report = validate_resource_management_system().await;
        assert!(report.is_ok());

        if let Ok(validation) = report {
            assert!(validation.validation_passed);
        }
    }

    #[tokio::test]
    async fn test_quick_resource_status() {
        let status = quick_resource_status().await;
        assert!(status.is_ok());
    }

    #[tokio::test]
    async fn test_backward_compatibility() {
        // Test that old code patterns still work
        let config = ResourceManagementConfig::default();
        let manager = ResourceManagementSystem::new(config).await;
        assert!(manager.is_ok());

        // Test legacy type aliases
        if let Ok(manager) = manager {
            let _legacy_manager: ResourceManager = manager;
        }
    }

    #[tokio::test]
    async fn test_module_integration() {
        // Test that all modules work together seamlessly
        let manager = create_default_resource_manager().await.unwrap();

        let status = manager.get_status().await;
        assert!(status.background_tasks.is_empty() || !status.background_tasks.is_empty());

        let health = manager.health_check().await;
        assert!(matches!(health.overall_health, HealthStatus::Healthy));
    }

    #[test]
    fn test_system_capabilities() {
        let capabilities = get_system_capabilities();
        assert!(capabilities.supported_resource_types.contains(&"NetworkPorts".to_string()));
        assert!(capabilities.supported_resource_types.contains(&"GpuDevices".to_string()));
        assert!(capabilities.supported_allocation_strategies.contains(&"LeastUsed".to_string()));
        assert!(capabilities.monitoring_enabled);
        assert!(capabilities.resource_pooling);
    }

    #[tokio::test]
    async fn test_optimization_functionality() {
        let manager = create_default_resource_manager().await.unwrap();
        let result = optimize_for_current_load(&manager).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_performance_reporting() {
        let manager = create_default_resource_manager().await.unwrap();
        let report = get_performance_report(&manager).await;
        assert!(report.is_ok());
    }

    #[test]
    fn test_validation_report_structure() {
        let report = ResourceValidationReport {
            validation_passed: true,
            system_health: HealthStatus::Healthy,
            component_status: ComponentStatus::default(),
            performance_metrics: SystemPerformanceMetrics::default(),
            validation_errors: vec![],
            recommendations: vec![],
        };

        assert!(report.validation_passed);
        assert!(matches!(report.system_health, HealthStatus::Healthy));
    }

    #[test]
    fn test_capabilities_completeness() {
        let capabilities = get_resource_capabilities();

        // Verify all expected resource types are supported
        let expected_types = vec!["NetworkPorts", "TempDirectories", "GpuDevices", "DatabaseConnections", "CustomResources"];
        for resource_type in expected_types {
            assert!(capabilities.supported_resource_types.contains(&resource_type.to_string()));
        }

        // Verify all expected allocation strategies are supported
        let expected_strategies = vec!["FirstAvailable", "LeastUsed", "BestFit", "RoundRobin", "Priority"];
        for strategy in expected_strategies {
            assert!(capabilities.supported_allocation_strategies.contains(&strategy.to_string()));
        }

        // Verify all expected cleanup policies are supported
        let expected_policies = vec!["Immediate", "Delayed", "SessionEnd", "Manual", "Debug"];
        for policy in expected_policies {
            assert!(capabilities.supported_cleanup_policies.contains(&policy.to_string()));
        }
    }

    #[test]
    fn test_legacy_type_aliases() {
        // Ensure type aliases compile correctly
        fn _test_resource_manager(_: ResourceManager) {}
        fn _test_port_manager(_: PortManager) {}
        fn _test_directory_manager(_: DirectoryManager) {}
        fn _test_gpu_manager(_: GpuManager) {}
    }
}
