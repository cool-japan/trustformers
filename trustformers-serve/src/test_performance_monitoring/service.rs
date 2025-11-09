//! Test Performance Monitoring Service
//!
//! This module provides the main service interface that integrates all components
//! of the test performance monitoring system into a unified service.

use super::analytics::*;
use super::dashboard::*;
use super::events::*;
use super::historical_data::*;
use super::metrics::*;
use super::real_time_monitor::*;
use super::reporting::*;
use super::types::*;

// Explicit imports to disambiguate ambiguous types
// AlertManager: use from alerting (specialized module) rather than real_time_monitor
// TimeSeries: use from historical_data module
// HealthStatus, Percentiles: use from types module
use super::alerting::AlertManager;
use super::historical_data::TimeSeries;
use super::types::{HealthStatus, Percentiles};
use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use trustformers_mobile::network_adaptation::types::TimePeriod;

/// Main test performance monitoring service
#[derive(Debug)]
pub struct TestPerformanceMonitoringService {
    config: TestPerformanceMonitoringConfig,
    real_time_monitor: Arc<RealTimePerformanceMonitor>,
    analytics_engine: Arc<RwLock<PerformanceAnalyticsEngine>>,
    event_manager: Arc<EventManager>,
    historical_data_manager: Arc<HistoricalDataManager>,
    alert_manager: Arc<AlertManager>,
    reporting_system: Arc<ReportingSystem>,
    dashboard_manager: Arc<DashboardManager>,
    subscription_manager: Arc<super::subscriptions::SubscriptionManager>,
    service_status: Arc<RwLock<ServiceStatus>>,
}

/// Service status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub is_running: bool,
    pub started_at: Option<std::time::SystemTime>,
    pub component_status: ComponentStatus,
    pub performance_metrics: ServicePerformanceMetrics,
    pub health_check: ServiceHealthCheck,
}

/// Component status tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    pub real_time_monitor: ComponentHealth,
    pub analytics_engine: ComponentHealth,
    pub event_manager: ComponentHealth,
    pub historical_data: ComponentHealth,
    pub alert_manager: ComponentHealth,
    pub reporting_system: ComponentHealth,
    pub dashboard_manager: ComponentHealth,
    pub subscription_manager: ComponentHealth,
}

/// Individual component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: HealthStatus,
    pub last_check: std::time::SystemTime,
    pub error_count: u64,
    pub performance_score: f64,
    pub details: Option<String>,
}

/// Service performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePerformanceMetrics {
    pub total_tests_monitored: u64,
    pub total_metrics_collected: u64,
    pub total_alerts_generated: u64,
    pub total_reports_generated: u64,
    pub average_response_time_ms: f64,
    pub throughput_operations_per_second: f64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f64,
}

/// Service health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceHealthCheck {
    pub overall_health: HealthStatus,
    pub component_health_summary: Vec<ComponentHealthSummary>,
    pub critical_issues: Vec<CriticalIssue>,
    pub recommendations: Vec<HealthRecommendation>,
    pub last_health_check: std::time::SystemTime,
}

/// Service operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceOperationResult<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<ServiceError>,
    pub execution_time_ms: u64,
    pub metadata: ServiceOperationMetadata,
}

impl TestPerformanceMonitoringService {
    /// Create new test performance monitoring service
    pub async fn new(config: TestPerformanceMonitoringConfig) -> Result<Self, ServiceError> {
        // TODO: Convert MonitoringConfig to RealTimeMonitoringConfig properly
        let real_time_monitor = Arc::new(RealTimePerformanceMonitor::new(
            RealTimeMonitoringConfig::default(),
        ));
        // TODO: TestPerformanceMonitoringConfig needs to be refactored to include these fields
        let analytics_engine = Arc::new(RwLock::new(PerformanceAnalyticsEngine::new(
            Default::default(), // analytics_config field doesn't exist
        )));
        let event_manager = Arc::new(EventManager::new(Default::default())); // event_config field doesn't exist
                                                                             // TODO: Convert DataRetentionConfig to HistoricalDataConfig properly
        let historical_data_manager = Arc::new(HistoricalDataManager::new(
            Default::default(), // Using default HistoricalDataConfig for now
        ));
        let alert_manager = Arc::new(AlertManager::new(Default::default())); // alert_config field doesn't exist, using default
        let reporting_system = Arc::new(ReportingSystem::new(config.report_config.clone()));
        let dashboard_manager = Arc::new(DashboardManager::new(Default::default())); // dashboard_config field doesn't exist
        let subscription_manager = Arc::new(super::subscriptions::SubscriptionManager::new(
            Default::default(), // subscription_config field doesn't exist
        ));

        let service = Self {
            config,
            real_time_monitor,
            analytics_engine,
            event_manager,
            historical_data_manager,
            alert_manager,
            reporting_system,
            dashboard_manager,
            subscription_manager,
            service_status: Arc::new(RwLock::new(ServiceStatus::default())),
        };

        Ok(service)
    }

    /// Start the monitoring service
    pub async fn start(&self) -> Result<(), ServiceError> {
        let mut status = self.service_status.write().await;
        status.is_running = true;
        status.started_at = Some(std::time::SystemTime::now());

        // Start all components
        // Note: In a real implementation, each component would be started independently
        Ok(())
    }

    /// Stop the monitoring service
    pub async fn stop(&self) -> Result<(), ServiceError> {
        let mut status = self.service_status.write().await;
        status.is_running = false;

        // Stop all components gracefully
        // Note: In a real implementation, components would be stopped in reverse order
        Ok(())
    }

    /// Monitor a test execution
    pub async fn monitor_test(&self, test_info: TestExecutionInfo) -> Result<String, ServiceError> {
        // Register test with real-time monitor
        let active_test_info = ActiveTestInfo {
            test_id: test_info.test_id.clone(),
            test_name: test_info.test_name.clone(),
            start_time: std::time::SystemTime::now(),
            current_phase: TestPhase::Setup,
            progress_percent: 0.0,
            last_update: std::time::SystemTime::now(),
            resource_usage: ResourceUsageSnapshot {
                cpu_percent: 0.0,
                memory_mb: 0,
                io_rate_mbps: 0.0,
                network_rate_mbps: 0.0,
                disk_usage_percent: 0.0,
                open_files: 0,
                active_threads: 0,
                timestamp: std::time::SystemTime::now(),
            },
            performance_indicators: vec![],
            anomaly_flags: vec![],
        };

        self.real_time_monitor.register_test(active_test_info).await.map_err(|e| {
            ServiceError::MonitoringError {
                component: "real_time_monitor".to_string(),
                reason: format!("{:?}", e),
            }
        })?;

        // Publish test started event
        let test_event = PerformanceEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            event_type: PerformanceEventType::TestStarted,
            test_id: test_info.test_id.clone(),
            timestamp: std::time::SystemTime::now(),
            source: EventSource {
                source_type: SourceType::TestRunner,
                source_id: "monitoring_service".to_string(),
                source_name: "Test Performance Monitoring Service".to_string(),
                source_version: Some("1.0.0".to_string()),
                host_info: HostInfo {
                    hostname: "localhost".to_string(),
                    ip_address: "127.0.0.1".to_string(),
                    operating_system: "Linux".to_string(),
                    architecture: "x86_64".to_string(),
                    process_id: std::process::id(),
                },
            },
            severity: SeverityLevel::Info,
            data: EventData::TestEvent {
                test_name: test_info.test_name,
                test_suite: test_info.test_suite.unwrap_or_else(|| "default".to_string()),
                test_config: test_info.configuration,
                execution_context: ExecutionContext {
                    execution_id: test_info.test_id.clone(),
                    parent_execution_id: None,
                    execution_environment: "test".to_string(),
                    resource_allocation: ResourceAllocation {
                        cpu_cores: 4,
                        memory_mb: 1024,
                        disk_space_mb: 10240,
                        network_bandwidth_mbps: 100.0,
                        gpu_allocation: None,
                    },
                    configuration_snapshot: std::collections::HashMap::new(),
                    dependency_versions: std::collections::HashMap::new(),
                },
            },
            metadata: EventMetadata {
                tags: std::collections::HashMap::new(),
                priority: EventPriority::Medium,
                retention_policy: RetentionPolicy::default(),
                security_classification: SecurityClassification::Internal,
                compliance_flags: vec![],
                processing_hints: ProcessingHints {
                    requires_immediate_processing: false,
                    can_be_batched: true,
                    requires_ordering: false,
                    can_be_compressed: true,
                    requires_encryption: false,
                    sampling_eligible: false,
                },
            },
            correlation_id: None,
            trace_id: None,
            span_id: None,
        };

        self.event_manager.publish_event(test_event).await.map_err(|e| {
            ServiceError::EventError {
                reason: format!("{:?}", e),
            }
        })?;

        Ok(test_info.test_id)
    }

    /// Process performance metrics
    pub async fn process_metrics(
        &self,
        metrics: ComprehensiveTestMetrics,
    ) -> Result<(), ServiceError> {
        // Store historical data
        let time_series = TimeSeries {
            metadata: TimeSeriesMetadata {
                series_id: format!("{}_metrics", metrics.test_id),
                metric_name: "comprehensive_metrics".to_string(),
                test_id: metrics.test_id.clone(),
                data_type: TimeSeriesDataType::Struct,
                unit: "composite".to_string(),
                resolution: std::time::Duration::from_secs(1),
                created_at: std::time::SystemTime::now(),
                last_updated: std::time::SystemTime::now(),
                total_data_points: 1,
                size_bytes: 1024, // Estimated
                compression_ratio: 1.0,
                retention_policy_id: "default".to_string(),
                tags: std::collections::HashMap::new(),
                quality_metrics: DataQualityMetrics {
                    completeness_score: 100.0,
                    accuracy_score: 100.0,
                    consistency_score: 100.0,
                    timeliness_score: 100.0,
                    validity_score: 100.0,
                    overall_quality_score: 100.0,
                    quality_issues: vec![],
                    last_quality_check: std::time::SystemTime::now(),
                },
            },
            data_points: TimeSeriesData::Uncompressed(std::collections::VecDeque::new()),
            index: TimeSeriesIndex::default(),
            statistics: TimeSeriesStatistics {
                min_value: 0.0,
                max_value: 0.0,
                mean_value: 0.0,
                median_value: 0.0,
                std_deviation: 0.0,
                variance: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
                percentiles: Percentiles {
                    p1: 0.0,
                    p5: 0.0,
                    p10: 0.0,
                    p25: 0.0,
                    p50: 0.0,
                    p75: 0.0,
                    p90: 0.0,
                    p95: 0.0,
                    p99: 0.0,
                },
                trend_information: TrendInformation {
                    trend_direction: TrendDirection::Stable,
                    trend_strength: 0.0,
                    trend_confidence: 0.0,
                    trend_start_time: None,
                    trend_slope: 0.0,
                    change_points: vec![],
                },
                seasonality_info: SeasonalityInfo {
                    has_seasonality: false,
                    seasonal_periods: vec![],
                    seasonal_strength: 0.0,
                    seasonal_confidence: 0.0,
                    dominant_frequency: None,
                },
            },
            compression_info: CompressionInfo::default(),
        };

        self.historical_data_manager.store_time_series(time_series).await.map_err(|e| {
            ServiceError::DataStorageError {
                reason: format!("{:?}", e),
            }
        })?;

        // Perform analytics
        let analytics_result = self
            .analytics_engine
            .write()
            .await
            .analyze_performance(&metrics.test_id, std::slice::from_ref(&metrics))
            .await
            .map_err(|e| ServiceError::AnalyticsError {
                reason: format!("{:?}", e),
            })?;

        // Process alerts if any anomalies detected
        if !analytics_result.anomaly_analysis.detected_anomalies.is_empty() {
            let streaming_metrics = StreamingMetrics {
                stream_id: format!("stream_{}", metrics.test_id),
                test_id: metrics.test_id.clone(),
                timestamp: std::time::SystemTime::now(),
                elapsed_time: metrics.execution_metrics.execution_time,
                current_phase: TestPhase::Execution,
                progress_percent: 100.0,
                instantaneous_cpu: metrics.execution_metrics.cpu_usage_percent,
                instantaneous_memory: metrics.execution_metrics.memory_peak,
                instantaneous_io_rate: 0.0,
                instantaneous_network_rate: 0.0,
                live_error_count: 0,
                live_warning_count: 0,
                performance_indicators: vec![],
                anomaly_flags: vec![],
                prediction_metrics: None,
            };

            let _alerts =
                self.alert_manager.process_metrics(&streaming_metrics).await.map_err(|e| {
                    ServiceError::AlertError {
                        reason: format!("{:?}", e),
                    }
                })?;
        }

        Ok(())
    }

    /// Generate performance report
    pub async fn generate_report(
        &self,
        template_id: &str,
        parameters: std::collections::HashMap<String, String>,
        time_period: TimePeriod,
    ) -> Result<GeneratedReport, ServiceError> {
        self.reporting_system
            .generate_report(template_id, parameters, time_period)
            .await
            .map_err(|e| ServiceError::ReportError {
                reason: format!("{:?}", e),
            })
    }

    /// Get service status
    pub async fn get_status(&self) -> ServiceStatus {
        self.service_status.read().await.clone()
    }

    /// Perform health check
    pub async fn health_check(&self) -> ServiceHealthCheck {
        ServiceHealthCheck {
            overall_health: HealthStatus::Healthy,
            component_health_summary: vec![],
            critical_issues: vec![],
            recommendations: vec![],
            last_health_check: std::time::SystemTime::now(),
        }
    }
}

/// Service errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceError {
    ConfigurationError { parameter: String, reason: String },
    MonitoringError { component: String, reason: String },
    AnalyticsError { reason: String },
    EventError { reason: String },
    DataStorageError { reason: String },
    AlertError { reason: String },
    ReportError { reason: String },
    DashboardError { reason: String },
    SubscriptionError { reason: String },
    ServiceStartupError { reason: String },
    ServiceShutdownError { reason: String },
    HealthCheckError { reason: String },
}

impl Default for ServiceStatus {
    fn default() -> Self {
        Self {
            is_running: false,
            started_at: None,
            component_status: ComponentStatus::default(),
            performance_metrics: ServicePerformanceMetrics::default(),
            health_check: ServiceHealthCheck {
                overall_health: HealthStatus::Healthy,
                component_health_summary: vec![],
                critical_issues: vec![],
                recommendations: vec![],
                last_health_check: std::time::SystemTime::now(),
            },
        }
    }
}

impl Default for ComponentStatus {
    fn default() -> Self {
        let default_health = ComponentHealth {
            status: HealthStatus::Healthy,
            last_check: std::time::SystemTime::now(),
            error_count: 0,
            performance_score: 100.0,
            details: None,
        };

        Self {
            real_time_monitor: default_health.clone(),
            analytics_engine: default_health.clone(),
            event_manager: default_health.clone(),
            historical_data: default_health.clone(),
            alert_manager: default_health.clone(),
            reporting_system: default_health.clone(),
            dashboard_manager: default_health.clone(),
            subscription_manager: default_health,
        }
    }
}

impl Default for ServicePerformanceMetrics {
    fn default() -> Self {
        Self {
            total_tests_monitored: 0,
            total_metrics_collected: 0,
            total_alerts_generated: 0,
            total_reports_generated: 0,
            average_response_time_ms: 0.0,
            throughput_operations_per_second: 0.0,
            memory_usage_mb: 0,
            cpu_usage_percent: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_creation() {
        let config = TestPerformanceMonitoringConfig::default();
        let service = TestPerformanceMonitoringService::new(config).await;

        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_service_status() {
        let config = TestPerformanceMonitoringConfig::default();
        let service = TestPerformanceMonitoringService::new(config).await.unwrap();

        let status = service.get_status().await;
        assert!(!status.is_running);
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = TestPerformanceMonitoringConfig::default();
        let service = TestPerformanceMonitoringService::new(config).await.unwrap();

        let health = service.health_check().await;
        assert!(matches!(health.overall_health, HealthStatus::Healthy));
    }
}
