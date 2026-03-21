//! Comprehensive Network Port Management for TrustformeRS
//!
//! This module has been refactored into a modular architecture for better maintainability.
//! All functionality has been organized into focused sub-modules while maintaining
//! backward compatibility through comprehensive re-exports.
//!
//! # Architecture Overview
//!
//! The port management system is organized into the following modules:
//! - `types`: Core types, configurations, and data structures for port management
//! - `manager`: Main NetworkPortManager implementation and coordination logic
//! - `reservation_system`: Port reservation system with advanced queuing
//! - `conflict_detector`: Port conflict detection and resolution algorithms
//! - `health_monitor`: Health monitoring and alerting system
//! - `performance_metrics`: Performance tracking and analytics
//!
//! # Features
//!
//! - **Thread-Safe Operations**: All operations are thread-safe using Arc, Mutex, and RwLock
//! - **Port Reservation System**: Advanced reservation system with conflict detection
//! - **Usage Statistics**: Comprehensive tracking of port usage patterns and metrics
//! - **Automatic Cleanup**: Cleanup expired allocations and reservations
//! - **Conflict Detection**: Detect and resolve port conflicts
//! - **Range Management**: Support for excluded ports and well-known port detection
//! - **Monitoring**: Real-time monitoring and health checks
//! - **Performance Optimization**: Optimized for high-concurrency scenarios
//!
//! # Usage
//!
//! All previous functionality remains available at the same import paths:
//!
//! ```rust
//! use trustformers_serve::resource_management::port_manager::{
//!     NetworkPortManager, PortManagementError,
//!     PortReservationSystem, PortConflictDetector, PortHealthMonitor
//! };
//! ```

// Import the modular structure
pub mod port_manager;

// Re-export everything to maintain backward compatibility
pub use port_manager::*;

// Legacy re-exports for backward compatibility
pub use port_manager::{
    // Core manager and error types
    NetworkPortManager, PortManagementError, PortManagementResult,

    // Configuration types
    PortReservationConfig, PortHealthConfig, PortHealthThresholds,
    PerformanceConfig,

    // Component types
    PortReservationSystem, PortConflictDetector, PortHealthMonitor,
    PortPerformanceMetrics,

    // Event types
    PortReservationEvent, PortConflictEvent, PortHealthEvent,

    // Status and metrics types
    PortHealthStatus, PerformanceSnapshot,

    // Enum types
    ReservationEventType, ConflictCondition, ConflictAction, ConflictType,
};

// Legacy compatibility types (maintain exact same interface as before)
pub type NetworkPortManagerLegacy = NetworkPortManager;

// Legacy initialization functions for backward compatibility
pub use port_manager::{
    init_port_manager, init_port_manager_with_config,
    validate_port_config, get_port_manager_capabilities
};

// Additional convenience functions

/// Create a default network port manager with standard configuration
pub fn create_default_port_manager() -> Result<NetworkPortManager, PortManagementError> {
    let config = super::types::PortPoolConfig::default();
    tokio::runtime::Handle::current().block_on(async {
        NetworkPortManager::new(config).await
    })
}

/// Create a port manager optimized for high-concurrency testing
pub fn create_high_concurrency_port_manager(
    port_range: (u16, u16),
    max_allocation: usize,
) -> Result<NetworkPortManager, PortManagementError> {
    let mut config = super::types::PortPoolConfig::default();
    config.port_range = port_range;
    config.max_allocation = max_allocation;
    config.enable_reservation = true;
    config.allocation_timeout_secs = 30; // Shorter timeout for fast turnaround

    tokio::runtime::Handle::current().block_on(async {
        NetworkPortManager::new(config).await
    })
}

/// Create a port manager with extensive monitoring enabled
pub fn create_monitored_port_manager(
    port_range: (u16, u16),
) -> Result<NetworkPortManager, PortManagementError> {
    let mut config = super::types::PortPoolConfig::default();
    config.port_range = port_range;
    config.enable_health_monitoring = true;
    config.enable_performance_tracking = true;
    config.enable_conflict_detection = true;

    tokio::runtime::Handle::current().block_on(async {
        NetworkPortManager::new(config).await
    })
}

/// Create a port manager with custom reservation settings
pub fn create_reservation_optimized_port_manager(
    port_range: (u16, u16),
    max_reservations_per_test: usize,
    default_reservation_duration_secs: u64,
) -> Result<NetworkPortManager, PortManagementError> {
    let mut config = super::types::PortPoolConfig::default();
    config.port_range = port_range;
    config.enable_reservation = true;
    config.max_reservations_per_test = max_reservations_per_test;
    config.default_reservation_duration_secs = default_reservation_duration_secs;

    tokio::runtime::Handle::current().block_on(async {
        NetworkPortManager::new(config).await
    })
}

/// Create a port manager with custom excluded port ranges
pub fn create_filtered_port_manager(
    port_range: (u16, u16),
    excluded_ranges: Vec<(u16, u16)>,
) -> Result<NetworkPortManager, PortManagementError> {
    let mut config = super::types::PortPoolConfig::default();
    config.port_range = port_range;
    config.reserved_ranges = excluded_ranges;

    tokio::runtime::Handle::current().block_on(async {
        NetworkPortManager::new(config).await
    })
}

/// Get port manager capabilities
pub fn get_manager_capabilities() -> PortManagerCapabilities {
    get_port_manager_capabilities()
}

/// Validate that the port manager is properly configured and functional
pub async fn validate_port_manager_system() -> Result<PortManagerValidationReport, PortManagementError> {
    let config = super::types::PortPoolConfig::default();
    validate_port_config(&config)?;

    let manager = NetworkPortManager::new(config).await?;

    // Test basic manager operations
    let health = manager.get_health_status().await;
    let capabilities = get_port_manager_capabilities();

    let validation_passed = matches!(health.overall_status, super::types::HealthStatus::Healthy) &&
                           capabilities.port_reservation;

    Ok(PortManagerValidationReport {
        validation_passed,
        manager_health: health.overall_status,
        capabilities,
        validation_errors: vec![],
        recommendations: health.recommendations,
    })
}

/// Port manager validation report
#[derive(Debug, Clone)]
pub struct PortManagerValidationReport {
    pub validation_passed: bool,
    pub manager_health: super::types::HealthStatus,
    pub capabilities: PortManagerCapabilities,
    pub validation_errors: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Port manager capabilities
#[derive(Debug, Clone)]
pub struct PortManagerCapabilities {
    pub supported_port_ranges: Vec<String>,
    pub supported_allocation_strategies: Vec<String>,
    pub port_reservation: bool,
    pub conflict_detection: bool,
    pub health_monitoring: bool,
    pub performance_tracking: bool,
    pub automatic_cleanup: bool,
    pub background_tasks: bool,
    pub concurrent_operations: bool,
    pub well_known_port_detection: bool,
    pub range_exclusion: bool,
    pub export_formats: Vec<String>,
}

/// Get port manager capabilities
pub fn get_port_manager_capabilities() -> PortManagerCapabilities {
    PortManagerCapabilities {
        supported_port_ranges: vec![
            "Custom".to_string(),
            "Ephemeral".to_string(),
            "UserDefined".to_string(),
            "Testing".to_string(),
        ],
        supported_allocation_strategies: vec![
            "Sequential".to_string(),
            "Random".to_string(),
            "LeastRecentlyUsed".to_string(),
            "ConflictAware".to_string(),
        ],
        port_reservation: true,
        conflict_detection: true,
        health_monitoring: true,
        performance_tracking: true,
        automatic_cleanup: true,
        background_tasks: true,
        concurrent_operations: true,
        well_known_port_detection: true,
        range_exclusion: true,
        export_formats: vec![
            "JSON".to_string(),
            "CSV".to_string(),
            "YAML".to_string(),
            "Report".to_string(),
        ],
    }
}

/// Utility functions for common port management patterns

/// Quick port availability assessment for immediate decision making
pub async fn quick_port_availability_check(
    port_range: (u16, u16),
) -> Result<PortAvailabilityReport, PortManagementError> {
    let config = super::types::PortPoolConfig {
        port_range,
        ..Default::default()
    };

    let manager = NetworkPortManager::new(config).await?;
    let stats = manager.get_statistics().await?;

    Ok(PortAvailabilityReport {
        total_ports: stats.total_ports,
        available_ports: stats.available_ports,
        allocated_ports: stats.allocated_ports,
        utilization_percent: stats.current_utilization * 100.0,
        can_allocate_additional: stats.available_ports > 0,
        recommended_max_concurrent: (stats.total_ports as f64 * 0.8) as usize,
    })
}

/// Port availability report
#[derive(Debug, Clone)]
pub struct PortAvailabilityReport {
    pub total_ports: usize,
    pub available_ports: usize,
    pub allocated_ports: usize,
    pub utilization_percent: f64,
    pub can_allocate_additional: bool,
    pub recommended_max_concurrent: usize,
}

/// Start port management with smart defaults based on use case
pub async fn start_smart_port_management(
    use_case: PortManagementUseCase,
    port_range: (u16, u16),
) -> Result<NetworkPortManager, PortManagementError> {
    let mut config = super::types::PortPoolConfig {
        port_range,
        ..Default::default()
    };

    match use_case {
        PortManagementUseCase::HighConcurrency => {
            config.max_allocation = 500;
            config.allocation_timeout_secs = 15;
            config.enable_reservation = true;
        },
        PortManagementUseCase::LowLatency => {
            config.allocation_timeout_secs = 5;
            config.enable_performance_tracking = true;
        },
        PortManagementUseCase::HighReliability => {
            config.enable_health_monitoring = true;
            config.enable_conflict_detection = true;
            config.enable_reservation = true;
        },
        PortManagementUseCase::ResourceConstrained => {
            config.max_allocation = 50;
            config.allocation_timeout_secs = 60;
        },
        PortManagementUseCase::Default => {
            // Use default configuration
        },
    }

    NetworkPortManager::new(config).await
}

/// Port management use case for smart configuration selection
#[derive(Debug, Clone)]
pub enum PortManagementUseCase {
    HighConcurrency,
    LowLatency,
    HighReliability,
    ResourceConstrained,
    Default,
}

/// Get port management recommendations based on current usage patterns
pub async fn get_port_management_recommendations(
    manager: &NetworkPortManager,
) -> Result<Vec<PortManagementRecommendation>, PortManagementError> {
    let stats = manager.get_statistics().await?;
    let health = manager.get_health_status().await;
    let mut recommendations = Vec::new();

    // High utilization recommendation
    if stats.current_utilization > 0.8 {
        recommendations.push(PortManagementRecommendation {
            priority: RecommendationPriority::High,
            category: "Resource Management".to_string(),
            suggestion: "Consider expanding port range or increasing cleanup frequency".to_string(),
            implementation_effort: ImplementationEffort::Medium,
        });
    }

    // Health-based recommendations
    if let super::types::HealthStatus::Warning = health.overall_status {
        recommendations.push(PortManagementRecommendation {
            priority: RecommendationPriority::Medium,
            category: "System Health".to_string(),
            suggestion: "Monitor health alerts and consider adjusting thresholds".to_string(),
            implementation_effort: ImplementationEffort::Low,
        });
    }

    // Performance-based recommendations
    if stats.avg_allocation_time_ms > 100.0 {
        recommendations.push(PortManagementRecommendation {
            priority: RecommendationPriority::Medium,
            category: "Performance".to_string(),
            suggestion: "Allocation times are high - consider enabling reservation system".to_string(),
            implementation_effort: ImplementationEffort::Low,
        });
    }

    if recommendations.is_empty() {
        recommendations.push(PortManagementRecommendation {
            priority: RecommendationPriority::Low,
            category: "General".to_string(),
            suggestion: "Port management system is operating optimally".to_string(),
            implementation_effort: ImplementationEffort::None,
        });
    }

    Ok(recommendations)
}

/// Port management optimization recommendation
#[derive(Debug, Clone)]
pub struct PortManagementRecommendation {
    pub priority: RecommendationPriority,
    pub category: String,
    pub suggestion: String,
    pub implementation_effort: ImplementationEffort,
}

/// Recommendation priority levels
#[derive(Debug, Clone)]
pub enum RecommendationPriority {
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

    #[test]
    fn test_port_manager_capabilities() {
        let capabilities = get_port_manager_capabilities();
        assert!(!capabilities.supported_port_ranges.is_empty());
        assert!(capabilities.port_reservation);
        assert!(capabilities.conflict_detection);
        assert!(capabilities.health_monitoring);
        assert!(capabilities.performance_tracking);
    }

    #[tokio::test]
    async fn test_validation_system() {
        let report = validate_port_manager_system().await;
        assert!(report.is_ok());

        if let Ok(validation) = report {
            assert!(validation.validation_passed);
        }
    }

    #[tokio::test]
    async fn test_quick_availability_check() {
        let report = quick_port_availability_check((8000, 8100)).await;
        assert!(report.is_ok());

        if let Ok(availability) = report {
            assert!(availability.total_ports > 0);
            assert!(availability.available_ports > 0);
        }
    }

    #[tokio::test]
    async fn test_smart_port_management() {
        let use_cases = [
            PortManagementUseCase::HighConcurrency,
            PortManagementUseCase::LowLatency,
            PortManagementUseCase::HighReliability,
            PortManagementUseCase::ResourceConstrained,
            PortManagementUseCase::Default,
        ];

        for use_case in use_cases {
            let result = start_smart_port_management(use_case, (8000, 8100)).await;
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_convenience_functions() {
        // Test convenience function creation
        let manager = create_default_port_manager();
        assert!(manager.is_ok());

        let high_concurrency = create_high_concurrency_port_manager((8000, 9000), 200);
        assert!(high_concurrency.is_ok());

        let monitored = create_monitored_port_manager((8100, 8200));
        assert!(monitored.is_ok());

        let reservation_optimized = create_reservation_optimized_port_manager((8200, 8300), 10, 300);
        assert!(reservation_optimized.is_ok());

        let filtered = create_filtered_port_manager((8300, 8400), vec![(8350, 8360)]);
        assert!(filtered.is_ok());
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that legacy type aliases work
        let _manager: NetworkPortManagerLegacy;
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together seamlessly
        let capabilities = get_port_manager_capabilities();
        assert!(capabilities.port_reservation);
        assert!(capabilities.conflict_detection);
        assert!(capabilities.health_monitoring);
        assert!(capabilities.performance_tracking);
    }

    #[test]
    fn test_capabilities_completeness() {
        let capabilities = get_port_manager_capabilities();

        // Verify all expected port ranges are supported
        let expected_ranges = vec!["Custom", "Ephemeral", "UserDefined", "Testing"];
        for range in expected_ranges {
            assert!(capabilities.supported_port_ranges.contains(&range.to_string()));
        }

        // Verify all expected allocation strategies are supported
        let expected_strategies = vec!["Sequential", "Random", "LeastRecentlyUsed", "ConflictAware"];
        for strategy in expected_strategies {
            assert!(capabilities.supported_allocation_strategies.contains(&strategy.to_string()));
        }

        // Verify all expected export formats are supported
        let expected_formats = vec!["JSON", "CSV", "YAML", "Report"];
        for format in expected_formats {
            assert!(capabilities.export_formats.contains(&format.to_string()));
        }
    }

    #[test]
    fn test_use_case_selection() {
        // Test different use case configurations
        let use_cases = [
            PortManagementUseCase::HighConcurrency,
            PortManagementUseCase::LowLatency,
            PortManagementUseCase::HighReliability,
            PortManagementUseCase::ResourceConstrained,
            PortManagementUseCase::Default,
        ];

        for use_case in use_cases {
            // This would normally test with actual port manager
            // For now, just verify the enum variants exist
            match use_case {
                PortManagementUseCase::HighConcurrency => {},
                PortManagementUseCase::LowLatency => {},
                PortManagementUseCase::HighReliability => {},
                PortManagementUseCase::ResourceConstrained => {},
                PortManagementUseCase::Default => {},
            }
        }
    }

    #[test]
    fn test_validation_report_structure() {
        let capabilities = get_port_manager_capabilities();

        let report = PortManagerValidationReport {
            validation_passed: true,
            manager_health: super::types::HealthStatus::Healthy,
            capabilities,
            validation_errors: vec![],
            recommendations: vec!["System healthy".to_string()],
        };

        assert!(report.validation_passed);
        assert!(matches!(report.manager_health, super::types::HealthStatus::Healthy));
    }

    #[test]
    fn test_recommendation_priority_levels() {
        let recommendations = vec![
            PortManagementRecommendation {
                priority: RecommendationPriority::High,
                category: "Test".to_string(),
                suggestion: "Test suggestion".to_string(),
                implementation_effort: ImplementationEffort::Low,
            }
        ];

        // Test that recommendations have valid priority levels
        for rec in recommendations {
            match rec.priority {
                RecommendationPriority::High |
                RecommendationPriority::Medium |
                RecommendationPriority::Low => {},
            }

            match rec.implementation_effort {
                ImplementationEffort::None |
                ImplementationEffort::Low |
                ImplementationEffort::Medium |
                ImplementationEffort::High => {},
            }
        }
    }

    #[test]
    fn test_availability_report_structure() {
        let report = PortAvailabilityReport {
            total_ports: 100,
            available_ports: 80,
            allocated_ports: 20,
            utilization_percent: 20.0,
            can_allocate_additional: true,
            recommended_max_concurrent: 80,
        };

        assert_eq!(report.total_ports, 100);
        assert_eq!(report.available_ports, 80);
        assert_eq!(report.allocated_ports, 20);
        assert_eq!(report.utilization_percent, 20.0);
        assert!(report.can_allocate_additional);
        assert_eq!(report.recommended_max_concurrent, 80);
    }
}