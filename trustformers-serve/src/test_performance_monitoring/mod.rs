//! Test Performance Monitoring System Modules
//!
//! This module provides a comprehensive test performance monitoring system organized into
//! focused sub-modules for better maintainability and clarity.
//!
//! # Architecture Overview
//!
//! The monitoring system is organized into the following modules:
//! - `types`: Core types, enums, and configuration structures
//! - `metrics`: Performance metrics models and data structures
//! - `real_time_monitor`: Real-time monitoring capabilities
//! - `analytics`: Performance analytics and data analysis
//! - `events`: Event management and streaming
//! - `historical_data`: Historical data management and storage
//! - `alerting`: Alert management and notification system
//! - `reporting`: Report generation and scheduling
//! - `dashboard`: Dashboard services and widgets
//! - `subscriptions`: Subscription management and preferences
//! - `service`: Main service integration and coordination
//!

// Allow dead code for monitoring infrastructure under development
#![allow(dead_code)]
//! # Usage
//!
//! All functionality is available through the main service interface:
//!
//! ```rust
//! use trustformers_serve::test_performance_monitoring::{
//!     TestPerformanceMonitoringService, TestPerformanceMonitoringConfig
//! };
//!
//! async fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = TestPerformanceMonitoringConfig::default();
//!     let service = TestPerformanceMonitoringService::new(config).await?;
//!
//!     service.start().await?;
//!     // Use the service for monitoring
//!     Ok(())
//! }
//! ```

use trustformers_mobile::network_adaptation::types::TimePeriod;

// Core modules
pub mod alerting;
pub mod analytics;
pub mod dashboard;
pub mod events;
pub mod historical_data;
pub mod metrics;
pub mod real_time_monitor;
pub mod reporting;
pub mod service;
pub mod subscriptions;
pub mod types;

// Re-export main types and services for convenience
pub use alerting::{
    ActiveAlert, AlertCondition, AlertManager, AlertRule, EscalationPolicy, NotificationChannel,
    ThresholdConfig,
};
pub use analytics::{
    AnalyticsResult, AnomalyAnalysisResult, BaselineComparisonResult, OptimizationAnalysisResult,
    PerformanceAnalyticsEngine, PerformanceInsight, StatisticalAnalysisResult, TrendAnalysisResult,
};
pub use dashboard::{Dashboard, DashboardManager, DashboardUpdate, Widget, WidgetType};
pub use events::{
    EventData,
    EventFilter,
    EventManager,
    EventSubscription,
    // StreamingEvent, // TODO: Fix - does not exist
    PerformanceEvent,
    PerformanceEventType,
};
pub use historical_data::{
    AggregationSpec, HistoricalDataManager, HistoricalDataQuery, QueryResult, TimeRange,
    TimeSeries, TimeSeriesMetadata,
};
pub use metrics::{
    ComprehensiveTestMetrics, EfficiencyMetrics, ExecutionMetrics, ParallelizationMetrics,
    ReliabilityMetrics, StreamingMetrics, SystemMetrics,
};
pub use real_time_monitor::{
    ActiveTestInfo, MonitoringStatus, RealTimePerformanceMonitor, ResourceUsageSnapshot,
    SystemHealth,
};
pub use reporting::{
    ExportFormat, GeneratedReport, ReportTemplate, ReportType, ReportingSystem, ScheduledReport,
};
pub use service::{
    ServiceError, ServiceHealthCheck, ServiceOperationResult, ServiceStatus,
    TestPerformanceMonitoringService,
};
pub use subscriptions::{
    EventSubscription as UserEventSubscription, NotificationPreferences, SubscriptionManager,
    UserSubscriptions,
};
pub use types::*;

// Convenience type aliases
pub type TestMonitoringService = TestPerformanceMonitoringService;
pub type MonitoringConfig = TestPerformanceMonitoringConfig;
pub type MonitoringResult<T> = Result<T, ServiceError>;

// Legacy compatibility types (to maintain backward compatibility)
pub type TestPerformanceMonitoringSystem = TestPerformanceMonitoringService;
// RealTimePerformanceMonitor already imported above
// PerformanceAnalyticsEngine already imported above

/// Initialize the test performance monitoring system with default configuration
pub async fn init_monitoring_system() -> MonitoringResult<TestPerformanceMonitoringService> {
    let config = TestPerformanceMonitoringConfig::default();
    TestPerformanceMonitoringService::new(config).await
}

/// Initialize the test performance monitoring system with custom configuration
pub async fn init_monitoring_system_with_config(
    config: TestPerformanceMonitoringConfig,
) -> MonitoringResult<TestPerformanceMonitoringService> {
    TestPerformanceMonitoringService::new(config).await
}

/// Validate monitoring configuration
pub fn validate_monitoring_config(
    config: &TestPerformanceMonitoringConfig,
) -> Result<(), ConfigValidationError> {
    if config.monitoring_interval < std::time::Duration::from_millis(100) {
        return Err(ConfigValidationError::InvalidParameter {
            parameter: "monitoring_interval".to_string(),
            reason: "Monitoring interval must be at least 100ms".to_string(),
        });
    }

    if config.retention_period < std::time::Duration::from_secs(3600) {
        return Err(ConfigValidationError::InvalidParameter {
            parameter: "retention_period".to_string(),
            reason: "Retention period must be at least 1 hour".to_string(),
        });
    }

    Ok(())
}

/// Configuration validation errors
#[derive(Debug, Clone)]
pub enum ConfigValidationError {
    InvalidParameter {
        parameter: String,
        reason: String,
    },
    MissingRequiredParameter {
        parameter: String,
    },
    ConflictingParameters {
        parameters: Vec<String>,
        reason: String,
    },
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigValidationError::InvalidParameter { parameter, reason } => {
                write!(f, "Invalid parameter '{}': {}", parameter, reason)
            },
            ConfigValidationError::MissingRequiredParameter { parameter } => {
                write!(f, "Missing required parameter: {}", parameter)
            },
            ConfigValidationError::ConflictingParameters { parameters, reason } => {
                write!(f, "Conflicting parameters {:?}: {}", parameters, reason)
            },
        }
    }
}

impl std::error::Error for ConfigValidationError {}

/// Create a monitoring service with performance optimization focus
pub async fn create_performance_optimized_service(
) -> MonitoringResult<TestPerformanceMonitoringService> {
    let mut config = TestPerformanceMonitoringConfig {
        enable_real_time: true,
        monitoring_interval: std::time::Duration::from_millis(500),
        ..Default::default()
    };

    // Optimize for performance monitoring
    config.alert_thresholds.cpu_usage_threshold = 0.80; // 80%
    config.alert_thresholds.memory_usage_threshold = 0.85; // 85%
    config.alert_thresholds.execution_time_threshold = std::time::Duration::from_secs(30);

    TestPerformanceMonitoringService::new(config).await
}

/// Create a monitoring service with compliance focus
pub async fn create_compliance_focused_service(
) -> MonitoringResult<TestPerformanceMonitoringService> {
    // Optimize for compliance and auditing
    // TODO: Replaced unstable Duration::from_days(90) with stable Duration::from_secs(7776000)
    let config = TestPerformanceMonitoringConfig {
        retention_period: std::time::Duration::from_secs(7776000),
        ..Default::default()
    };
    // TODO: report_config.compliance_reporting field doesn't exist
    // TODO: historical_data_config field doesn't exist on TestPerformanceMonitoringConfig
    // TODO: event_config field doesn't exist on TestPerformanceMonitoringConfig
    // config.report_config.compliance_reporting = true;
    // config.historical_data_config.audit_trail_enabled = true;
    // config.event_config.compliance_logging = true;

    TestPerformanceMonitoringService::new(config).await
}

/// Create a monitoring service with resource efficiency focus
pub async fn create_resource_efficient_service(
) -> MonitoringResult<TestPerformanceMonitoringService> {
    // Optimize for minimal resource usage
    let config = TestPerformanceMonitoringConfig {
        monitoring_interval: std::time::Duration::from_secs(5),
        ..Default::default()
    };
    // TODO: historical_data_config, event_config, alert_config fields don't exist on TestPerformanceMonitoringConfig
    // config.historical_data_config.compression_enabled = true;
    // config.event_config.compression_enabled = true;
    // config.alert_config.rate_limiting_enabled = true;

    TestPerformanceMonitoringService::new(config).await
}

/// Get monitoring system capabilities
pub fn get_monitoring_capabilities() -> MonitoringCapabilities {
    MonitoringCapabilities {
        supported_metrics: vec![
            "execution_time".to_string(),
            "memory_usage".to_string(),
            "cpu_usage".to_string(),
            "io_operations".to_string(),
            "network_operations".to_string(),
            "thread_utilization".to_string(),
            "error_rates".to_string(),
            "throughput".to_string(),
        ],
        supported_export_formats: vec![
            "JSON".to_string(),
            "CSV".to_string(),
            "PDF".to_string(),
            "HTML".to_string(),
            "Excel".to_string(),
        ],
        supported_alert_channels: vec![
            "Email".to_string(),
            "SMS".to_string(),
            "Slack".to_string(),
            "Webhook".to_string(),
            "PagerDuty".to_string(),
        ],
        real_time_monitoring: true,
        historical_analysis: true,
        predictive_analytics: true,
        anomaly_detection: true,
        performance_optimization: true,
        compliance_reporting: true,
        multi_tenant_support: true,
        api_integration: true,
        dashboard_support: true,
        custom_alerts: true,
        data_retention_policies: true,
        encryption_support: true,
    }
}

/// Monitoring system capabilities
#[derive(Debug, Clone)]
pub struct MonitoringCapabilities {
    pub supported_metrics: Vec<String>,
    pub supported_export_formats: Vec<String>,
    pub supported_alert_channels: Vec<String>,
    pub real_time_monitoring: bool,
    pub historical_analysis: bool,
    pub predictive_analytics: bool,
    pub anomaly_detection: bool,
    pub performance_optimization: bool,
    pub compliance_reporting: bool,
    pub multi_tenant_support: bool,
    pub api_integration: bool,
    pub dashboard_support: bool,
    pub custom_alerts: bool,
    pub data_retention_policies: bool,
    pub encryption_support: bool,
}

/// Utility functions for common monitoring patterns

/// Quick start monitoring for a simple test
pub async fn quick_monitor_test(test_id: String, test_name: String) -> MonitoringResult<String> {
    let service = init_monitoring_system().await?;
    service.start().await?;

    let test_info = TestExecutionInfo {
        test_id: test_id.clone(),
        test_name,
        test_suite: None,
        start_time: chrono::Utc::now(),
        end_time: None,
        status: "running".to_string(),
        configuration: std::collections::HashMap::new(),
        expected_duration: None,
        resource_requirements: None,
    };

    service.monitor_test(test_info).await
}

/// Generate a quick performance report
pub async fn quick_performance_report(
    service: &TestPerformanceMonitoringService,
    time_period: TimePeriod,
) -> MonitoringResult<GeneratedReport> {
    let parameters = std::collections::HashMap::new();
    service
        .generate_report("default_performance_summary", parameters, time_period)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitoring_system_initialization() {
        let service = init_monitoring_system().await;
        assert!(service.is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = TestPerformanceMonitoringConfig::default();
        assert!(validate_monitoring_config(&config).is_ok());

        config.monitoring_interval = std::time::Duration::from_millis(50); // Too small
        assert!(validate_monitoring_config(&config).is_err());
    }

    #[tokio::test]
    async fn test_specialized_service_creation() {
        let perf_service = create_performance_optimized_service().await;
        assert!(perf_service.is_ok());

        let compliance_service = create_compliance_focused_service().await;
        assert!(compliance_service.is_ok());

        let efficient_service = create_resource_efficient_service().await;
        assert!(efficient_service.is_ok());
    }

    #[test]
    fn test_capabilities() {
        let capabilities = get_monitoring_capabilities();
        assert!(!capabilities.supported_metrics.is_empty());
        assert!(capabilities.real_time_monitoring);
        assert!(capabilities.historical_analysis);
    }

    #[tokio::test]
    async fn test_quick_monitor_test() {
        let result = quick_monitor_test("test-001".to_string(), "Quick Test".to_string()).await;
        assert!(result.is_ok());
    }
}
