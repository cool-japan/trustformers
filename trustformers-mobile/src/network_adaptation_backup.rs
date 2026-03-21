//! Network Condition Adaptation for Federated Learning
//!
//! This module has been refactored into a modular architecture for better maintainability.
//! All functionality has been organized into focused sub-modules while maintaining
//! backward compatibility through comprehensive re-exports.
//!
//! # Architecture Overview
//!
//! The network adaptation system is organized into the following modules:
//! - `config`: Configuration management for network adaptation parameters
//! - `types`: Core types, enums, and data structures for network adaptation
//! - `monitoring`: Network monitoring and condition assessment
//! - `scheduling`: Federated task scheduling and coordination
//! - `bandwidth`: Bandwidth optimization and traffic management
//! - `synchronization`: Model synchronization coordination
//! - `prediction`: Network performance prediction and modeling
//! - `optimization`: Network optimization algorithms and strategies
//! - `compression`: Compression algorithms for network efficiency
//! - `adaptation`: Adaptive network behavior implementation
//! - `utils`: Utility functions and helper implementations
//!
//! # Usage
//!
//! All previous functionality remains available at the same import paths:
//!
//! ```rust
//! use trustformers_mobile::network_adaptation_backup::{
//!     NetworkAdaptationManager, NetworkAdaptationConfig,
//!     NetworkMonitor, FederatedScheduler
//! };
//! ```

// Import the modular structure
pub mod network_adaptation;

// Re-export everything to maintain backward compatibility
pub use network_adaptation::*;

// Legacy re-exports for backward compatibility
pub use network_adaptation::{
    // Configuration types
    NetworkAdaptationConfig, NetworkQualityThresholds, CommunicationStrategy,
    WiFiStrategy, CellularStrategy, PoorNetworkStrategy, NetworkCompressionConfig,
    RetryConfig, DataUsageLimits, SyncFrequencyConfig, FailureRecoveryConfig,
    NetworkPredictionConfig, MonitoringConfig, DashboardConfig, SubscriptionConfig,
    AnalyticsConfig, HistoricalDataConfig, AlertConfig, ReportConfig,

    // Core service types
    NetworkAdaptationManager, NetworkAdaptationStats, StatisticsCollector,

    // Monitoring types
    NetworkMonitor, NetworkConditions, NetworkQuality, NetworkQualityAnalyzer,
    NetworkTrendAnalyzer, TrendDirection, NetworkMonitoringStats, QualityDistribution,
    NetworkTrendAnalysis,

    // Scheduling types
    FederatedScheduler, FederatedTask, FederatedTaskType, TaskPriority,
    SchedulingDecision, SchedulingStrategy, TaskPrioritizer, ScheduleOptimizer,
    OptimizationStrategy, SchedulingConstraints, PerformancePredictor,

    // Bandwidth and optimization types
    BandwidthOptimizer, TrafficShaper, DataUsageTracker, NetworkOptimizer,
    OptimizationResult, OptimizationMetrics, BandwidthAllocation, TrafficClass,
    QualityOfService, NetworkEfficiencyMetrics,

    // Synchronization types
    ModelSyncCoordinator, SyncStrategy, SyncResult, SyncDecision, SyncMetrics,
    SyncCoordinationStrategy, ConflictResolution, ConsistencyLevel,

    // Prediction types
    NetworkPredictor, PredictionModel, PredictionResult, PredictionAccuracy,
    NetworkForecast, PredictionMetrics, ModelTrainingData, PredictionConfig,

    // Compression types
    NetworkCompressionEngine, GradientCompressor, CompressionStats,
    GradientCompressionAlgorithm, CompressionLevel, CompressionMetrics,
    DecompressionResult,

    // Communication strategy types
    CommunicationStrategyManager, NetworkStrategy, StrategySelection,
    StrategyEffectiveness, AdaptiveStrategy,

    // Utility types
    NetworkUtils, PerformanceMetrics, DiagnosticInfo, SystemHealth,
    ResourceUsage, NetworkDiagnostics, TroubleshootingInfo,
};

// Legacy compatibility types (maintain exact same interface as before)
pub type NetworkAdaptationSystem = NetworkAdaptationManager;

// Legacy initialization functions for backward compatibility
pub use network_adaptation::{
    init_network_adaptation, init_network_adaptation_with_config, validate_network_config
};

// Additional convenience functions

/// Create a default network adaptation manager
pub async fn create_default_network_manager() -> Result<NetworkAdaptationManager, NetworkAdaptationError> {
    let config = NetworkAdaptationConfig::default();
    NetworkAdaptationManager::new(config).await
}

/// Create a network adaptation manager optimized for WiFi networks
pub async fn create_wifi_optimized_manager() -> Result<NetworkAdaptationManager, NetworkAdaptationError> {
    let mut config = NetworkAdaptationConfig::default();
    config.communication_strategy.wifi_strategy.enable_high_frequency_updates = true;
    config.communication_strategy.wifi_strategy.enable_background_sync = true;
    config.communication_strategy.wifi_strategy.max_concurrent_connections = 8;
    config.enable_adaptive_scheduling = true;
    config.enable_bandwidth_optimization = true;
    NetworkAdaptationManager::new(config).await
}

/// Create a network adaptation manager optimized for cellular networks
pub async fn create_cellular_optimized_manager() -> Result<NetworkAdaptationManager, NetworkAdaptationError> {
    let mut config = NetworkAdaptationConfig::default();
    config.communication_strategy.cellular_strategy.data_usage_awareness.adaptive_quality = true;
    config.communication_strategy.cellular_strategy.data_usage_awareness.track_daily_usage = true;
    config.communication_strategy.cellular_strategy.preferred_sync_hours = vec![2, 3, 4]; // Late night
    config.communication_strategy.compression_config.enable_compression = true;
    config.communication_strategy.compression_config.compression_algorithm = GradientCompressionAlgorithm::Adaptive;
    NetworkAdaptationManager::new(config).await
}

/// Create a network adaptation manager for poor network conditions
pub async fn create_poor_network_manager() -> Result<NetworkAdaptationManager, NetworkAdaptationError> {
    let mut config = NetworkAdaptationConfig::default();
    config.communication_strategy.poor_network_strategy.enable_store_and_forward = true;
    config.communication_strategy.poor_network_strategy.aggressive_compression = true;
    config.communication_strategy.poor_network_strategy.minimal_heartbeat_frequency = true;
    config.communication_strategy.compression_config.compression_level = CompressionLevel::Maximum;
    config.communication_strategy.retry_config.max_retries = 5;
    config.quality_thresholds.min_bandwidth_full_sync_mbps = 0.5;
    NetworkAdaptationManager::new(config).await
}

/// Create a network adaptation manager with custom quality thresholds
pub async fn create_custom_threshold_manager(
    min_bandwidth_mbps: f32,
    max_latency_ms: f32,
    max_packet_loss: f32,
) -> Result<NetworkAdaptationManager, NetworkAdaptationError> {
    let mut config = NetworkAdaptationConfig::default();
    config.quality_thresholds.min_bandwidth_full_sync_mbps = min_bandwidth_mbps;
    config.quality_thresholds.min_bandwidth_incremental_sync_mbps = min_bandwidth_mbps * 0.5;
    config.quality_thresholds.max_latency_realtime_ms = max_latency_ms;
    config.quality_thresholds.max_packet_loss_percent = max_packet_loss;
    NetworkAdaptationManager::new(config).await
}

/// Get network adaptation system capabilities
pub fn get_network_capabilities() -> NetworkCapabilities {
    get_adaptation_capabilities()
}

/// Validate that the network adaptation system is properly configured and functional
pub async fn validate_network_adaptation_system() -> Result<NetworkValidationReport, NetworkAdaptationError> {
    let config = NetworkAdaptationConfig::default();
    validate_network_config(&config).map_err(|e| NetworkAdaptationError::ConfigurationError {
        parameter: "config".to_string(),
        reason: e.to_string(),
    })?;

    let manager = NetworkAdaptationManager::new(config).await?;

    // Test basic manager operations
    let status = manager.get_status().await;
    let health = manager.health_check().await;

    let validation_passed = matches!(health.overall_health, HealthStatus::Healthy) &&
                           matches!(status.state, AdaptationState::Idle);

    Ok(NetworkValidationReport {
        validation_passed,
        manager_health: health.overall_health,
        component_status: status.component_status,
        network_metrics: status.network_metrics,
        validation_errors: vec![],
        recommendations: health.recommendations,
    })
}

/// Network adaptation system validation report
#[derive(Debug, Clone)]
pub struct NetworkValidationReport {
    pub validation_passed: bool,
    pub manager_health: HealthStatus,
    pub component_status: ComponentStatus,
    pub network_metrics: NetworkPerformanceMetrics,
    pub validation_errors: Vec<String>,
    pub recommendations: Vec<HealthRecommendation>,
}

/// Network adaptation system capabilities
#[derive(Debug, Clone)]
pub struct NetworkCapabilities {
    pub supported_network_types: Vec<String>,
    pub supported_compression_algorithms: Vec<String>,
    pub supported_optimization_strategies: Vec<String>,
    pub real_time_monitoring: bool,
    pub predictive_adaptation: bool,
    pub multi_network_support: bool,
    pub bandwidth_optimization: bool,
    pub data_usage_awareness: bool,
    pub adaptive_scheduling: bool,
    pub store_and_forward: bool,
    pub compression_support: bool,
    pub quality_of_service: bool,
}

/// Get network adaptation capabilities
pub fn get_adaptation_capabilities() -> NetworkCapabilities {
    NetworkCapabilities {
        supported_network_types: vec![
            "WiFi".to_string(),
            "4G".to_string(),
            "5G".to_string(),
            "3G".to_string(),
            "Ethernet".to_string(),
            "Bluetooth".to_string(),
        ],
        supported_compression_algorithms: vec![
            "TopK".to_string(),
            "RandomSparsification".to_string(),
            "ThresholdBased".to_string(),
            "Quantized".to_string(),
            "Adaptive".to_string(),
        ],
        supported_optimization_strategies: vec![
            "BandwidthOptimization".to_string(),
            "LatencyOptimization".to_string(),
            "PowerOptimization".to_string(),
            "BalancedOptimization".to_string(),
            "ThroughputOptimization".to_string(),
        ],
        real_time_monitoring: true,
        predictive_adaptation: true,
        multi_network_support: true,
        bandwidth_optimization: true,
        data_usage_awareness: true,
        adaptive_scheduling: true,
        store_and_forward: true,
        compression_support: true,
        quality_of_service: true,
    }
}

/// Utility functions for common network adaptation patterns

/// Quick network assessment for immediate decision making
pub async fn quick_network_assessment() -> Result<NetworkQuality, NetworkAdaptationError> {
    let manager = create_default_network_manager().await?;
    let conditions = manager.get_current_network_conditions().await?;
    Ok(conditions.quality_assessment)
}

/// Optimize network settings for current conditions
pub async fn optimize_for_current_network(
    manager: &NetworkAdaptationManager,
) -> Result<OptimizationResult, NetworkAdaptationError> {
    let conditions = manager.get_current_network_conditions().await?;
    manager.optimize_for_conditions(&conditions).await
}

/// Start adaptive monitoring with smart defaults
pub async fn start_smart_monitoring(
    manager: &mut NetworkAdaptationManager,
) -> Result<(), NetworkAdaptationError> {
    manager.start_adaptive_monitoring().await
}

/// Get performance recommendations based on current network
pub async fn get_performance_recommendations(
    manager: &NetworkAdaptationManager,
) -> Result<Vec<PerformanceRecommendation>, NetworkAdaptationError> {
    let conditions = manager.get_current_network_conditions().await?;
    let analytics = manager.get_network_analytics().await?;
    Ok(manager.generate_recommendations(&conditions, &analytics).await?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_network_adaptation_manager_creation() {
        let manager = create_default_network_manager().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_wifi_optimized_manager() {
        let manager = create_wifi_optimized_manager().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_cellular_optimized_manager() {
        let manager = create_cellular_optimized_manager().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_poor_network_manager() {
        let manager = create_poor_network_manager().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_custom_threshold_manager() {
        let manager = create_custom_threshold_manager(5.0, 100.0, 1.0).await;
        assert!(manager.is_ok());
    }

    #[test]
    fn test_network_capabilities() {
        let capabilities = get_network_capabilities();
        assert!(!capabilities.supported_network_types.is_empty());
        assert!(capabilities.real_time_monitoring);
        assert!(capabilities.predictive_adaptation);
        assert!(capabilities.multi_network_support);
    }

    #[tokio::test]
    async fn test_validation_system() {
        let report = validate_network_adaptation_system().await;
        assert!(report.is_ok());

        if let Ok(validation) = report {
            assert!(validation.validation_passed);
        }
    }

    #[tokio::test]
    async fn test_quick_network_assessment() {
        let assessment = quick_network_assessment().await;
        assert!(assessment.is_ok());
    }

    #[tokio::test]
    async fn test_backward_compatibility() {
        // Test that old code patterns still work
        let config = NetworkAdaptationConfig::default();
        let manager = NetworkAdaptationManager::new(config).await;
        assert!(manager.is_ok());

        // Test legacy type alias
        if let Ok(manager) = manager {
            let _legacy_manager: NetworkAdaptationSystem = manager;
        }
    }

    #[tokio::test]
    async fn test_module_integration() {
        // Test that all modules work together seamlessly
        let manager = create_default_network_manager().await.expect("Operation failed");

        let status = manager.get_status().await;
        assert!(matches!(status.state, AdaptationState::Idle));

        let health = manager.health_check().await;
        assert!(matches!(health.overall_health, HealthStatus::Healthy));
    }

    #[test]
    fn test_adaptation_capabilities() {
        let capabilities = get_adaptation_capabilities();
        assert!(capabilities.supported_network_types.contains(&"WiFi".to_string()));
        assert!(capabilities.supported_network_types.contains(&"5G".to_string()));
        assert!(capabilities.supported_compression_algorithms.contains(&"Adaptive".to_string()));
        assert!(capabilities.bandwidth_optimization);
        assert!(capabilities.data_usage_awareness);
    }

    #[tokio::test]
    async fn test_smart_monitoring_start() {
        let mut manager = create_default_network_manager().await.expect("Operation failed");
        let result = start_smart_monitoring(&mut manager).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_performance_optimization() {
        let manager = create_default_network_manager().await.expect("Operation failed");
        let result = optimize_for_current_network(&manager).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_performance_recommendations() {
        let manager = create_default_network_manager().await.expect("Operation failed");
        let recommendations = get_performance_recommendations(&manager).await;
        assert!(recommendations.is_ok());
    }

    #[test]
    fn test_validation_report_structure() {
        let report = NetworkValidationReport {
            validation_passed: true,
            manager_health: HealthStatus::Healthy,
            component_status: ComponentStatus::default(),
            network_metrics: NetworkPerformanceMetrics::default(),
            validation_errors: vec![],
            recommendations: vec![],
        };

        assert!(report.validation_passed);
        assert!(matches!(report.manager_health, HealthStatus::Healthy));
    }

    #[test]
    fn test_network_capabilities_completeness() {
        let capabilities = get_network_capabilities();

        // Verify all expected network types are supported
        let expected_types = vec!["WiFi", "4G", "5G", "3G", "Ethernet", "Bluetooth"];
        for network_type in expected_types {
            assert!(capabilities.supported_network_types.contains(&network_type.to_string()));
        }

        // Verify all expected compression algorithms are supported
        let expected_algorithms = vec!["TopK", "RandomSparsification", "ThresholdBased", "Quantized", "Adaptive"];
        for algorithm in expected_algorithms {
            assert!(capabilities.supported_compression_algorithms.contains(&algorithm.to_string()));
        }

        // Verify all expected optimization strategies are supported
        let expected_strategies = vec![
            "BandwidthOptimization", "LatencyOptimization", "PowerOptimization",
            "BalancedOptimization", "ThroughputOptimization"
        ];
        for strategy in expected_strategies {
            assert!(capabilities.supported_optimization_strategies.contains(&strategy.to_string()));
        }
    }
}
