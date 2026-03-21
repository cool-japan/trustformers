//! GPU device allocation and monitoring for test parallelization.

// Re-export types for external access
pub use super::types::{
    AlertSeverity, ComparisonOperator, EscalationActionType, EscalationConditionType,
    GpuAlertEventType, GpuAlertType, GpuBenchmarkType, GpuCapability, GpuClockSpeeds,
    GpuConstraint, GpuDeviceStatus, GpuMetricType, GpuPerformanceRequirements, GpuUsageStatistics,
    GpuUsageType, PerformanceRecommendationType, RecommendationDifficulty, RecommendationPriority,
    RegressionSeverity, TrendDirection,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::{debug, info};

use crate::test_parallelization::GpuPoolConfig;

/// GPU resource management
pub struct GpuResourceManager {
    /// Configuration
    config: Arc<RwLock<GpuPoolConfig>>,
    /// Available GPU devices
    available_devices: Arc<Mutex<Vec<GpuDeviceInfo>>>,
    /// Allocated GPU resources
    allocated_resources: Arc<Mutex<HashMap<String, GpuAllocation>>>,
    /// GPU monitoring system
    monitoring_system: Arc<GpuMonitoringSystem>,
    /// GPU performance tracker
    performance_tracker: Arc<GpuPerformanceTracker>,
    /// GPU usage statistics
    usage_stats: Arc<Mutex<GpuUsageStatistics>>,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device ID
    pub device_id: usize,
    /// Device name
    pub device_name: String,
    /// Total memory (MB)
    pub total_memory_mb: u64,
    /// Available memory (MB)
    pub available_memory_mb: u64,
    /// Current utilization percentage
    pub utilization_percent: f32,
    /// Device capabilities
    pub capabilities: Vec<GpuCapability>,
    /// Device status
    pub status: GpuDeviceStatus,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// GPU allocation information
#[derive(Debug, Clone)]
pub struct GpuAllocation {
    /// Allocated device info
    pub device: GpuDeviceInfo,
    /// Test ID that allocated the GPU
    pub test_id: String,
    /// Memory allocated (MB)
    pub memory_allocated_mb: u64,
    /// Allocation timestamp
    pub allocated_at: DateTime<Utc>,
    /// Expected release time
    pub expected_release: Option<DateTime<Utc>>,
    /// GPU usage type
    pub usage_type: GpuUsageType,
    /// Performance requirements
    pub performance_requirements: GpuPerformanceRequirements,
}

/// GPU monitoring system
pub struct GpuMonitoringSystem {
    /// Monitoring configuration
    config: Arc<RwLock<GpuMonitoringConfig>>,
    /// Real-time metrics
    real_time_metrics: Arc<Mutex<HashMap<usize, GpuRealTimeMetrics>>>,
    /// Historical metrics
    historical_metrics: Arc<Mutex<HashMap<usize, Vec<GpuHistoricalMetric>>>>,
    /// Alert system
    alert_system: Arc<GpuAlertSystem>,
}

impl Default for GpuMonitoringSystem {
    fn default() -> Self {
        Self {
            config: Arc::new(RwLock::new(GpuMonitoringConfig::default())),
            real_time_metrics: Arc::new(Mutex::new(HashMap::new())),
            historical_metrics: Arc::new(Mutex::new(HashMap::new())),
            alert_system: Arc::new(GpuAlertSystem::default()),
        }
    }
}

/// GPU monitoring configuration
#[derive(Debug, Clone)]
pub struct GpuMonitoringConfig {
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Metrics retention period
    pub retention_period: Duration,
    /// Enable real-time monitoring
    pub real_time_monitoring: bool,
    /// Alert thresholds
    pub alert_thresholds: GpuAlertThresholds,
    /// Monitored metrics
    pub monitored_metrics: Vec<GpuMetricType>,
}

impl Default for GpuMonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(1),
            retention_period: Duration::from_secs(3600),
            real_time_monitoring: false,
            alert_thresholds: GpuAlertThresholds::default(),
            monitored_metrics: Vec::new(),
        }
    }
}

/// GPU real-time metrics
#[derive(Debug, Clone)]
pub struct GpuRealTimeMetrics {
    /// Device ID
    pub device_id: usize,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Memory usage (MB)
    pub memory_usage_mb: u64,
    /// Utilization percentage
    pub utilization_percent: f32,
    /// Temperature (Celsius)
    pub temperature_celsius: f32,
    /// Power consumption (Watts)
    pub power_consumption_watts: f32,
    /// Clock speeds
    pub clock_speeds: GpuClockSpeeds,
    /// Fan speeds
    pub fan_speeds: Vec<f32>,
}

/// GPU historical metric
#[derive(Debug, Clone)]
pub struct GpuHistoricalMetric {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Metric type
    pub metric_type: GpuMetricType,
    /// Metric value
    pub value: f64,
    /// Associated test ID
    pub test_id: Option<String>,
    /// Metric metadata
    pub metadata: HashMap<String, String>,
}

/// GPU alert system
pub struct GpuAlertSystem {
    /// Alert configuration
    config: Arc<RwLock<GpuAlertConfig>>,
    /// Active alerts
    active_alerts: Arc<Mutex<HashMap<String, GpuAlert>>>,
    /// Alert history
    alert_history: Arc<Mutex<Vec<GpuAlertEvent>>>,
    /// Alert handlers
    alert_handlers: Arc<Mutex<Vec<Box<dyn GpuAlertHandler + Send + Sync>>>>,
}

impl Default for GpuAlertSystem {
    fn default() -> Self {
        Self {
            config: Arc::new(RwLock::new(GpuAlertConfig::default())),
            active_alerts: Arc::new(Mutex::new(HashMap::new())),
            alert_history: Arc::new(Mutex::new(Vec::new())),
            alert_handlers: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

/// GPU alert configuration
#[derive(Debug, Clone)]
pub struct GpuAlertConfig {
    /// Enable alerts
    pub enabled: bool,
    /// Alert cooldown period
    pub cooldown_period: Duration,
    /// Alert thresholds
    pub thresholds: GpuAlertThresholds,
    /// Alert escalation rules
    pub escalation_rules: Vec<GpuAlertEscalationRule>,
}

impl Default for GpuAlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cooldown_period: Duration::from_secs(300), // 5 minutes
            thresholds: GpuAlertThresholds::default(),
            escalation_rules: Vec::new(),
        }
    }
}

/// GPU alert thresholds
#[derive(Debug, Clone)]
pub struct GpuAlertThresholds {
    /// High memory usage threshold
    pub high_memory_usage: f32,
    /// High utilization threshold
    pub high_utilization: f32,
    /// High temperature threshold
    pub high_temperature: f32,
    /// High power consumption threshold
    pub high_power_consumption: f32,
    /// Low performance threshold
    pub low_performance: f32,
    /// Error rate threshold
    pub error_rate: f32,
}

impl Default for GpuAlertThresholds {
    fn default() -> Self {
        Self {
            high_memory_usage: 0.9,
            high_utilization: 0.95,
            high_temperature: 85.0,
            high_power_consumption: 300.0,
            low_performance: 0.5,
            error_rate: 0.05,
        }
    }
}

/// GPU alert
#[derive(Debug, Clone)]
pub struct GpuAlert {
    /// Alert ID
    pub id: String,
    /// Device ID
    pub device_id: usize,
    /// Alert type
    pub alert_type: GpuAlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Triggered timestamp
    pub triggered_at: DateTime<Utc>,
    /// Associated test ID
    pub test_id: Option<String>,
    /// Alert metadata
    pub metadata: HashMap<String, String>,
}

/// GPU alert event
#[derive(Debug, Clone)]
pub struct GpuAlertEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: GpuAlertEventType,
    /// Associated alert
    pub alert: GpuAlert,
    /// Event details
    pub details: HashMap<String, String>,
}

/// GPU alert escalation rule
#[derive(Debug, Clone)]
pub struct GpuAlertEscalationRule {
    /// Rule name
    pub name: String,
    /// Alert types to escalate
    pub alert_types: Vec<GpuAlertType>,
    /// Escalation conditions
    pub conditions: Vec<EscalationCondition>,
    /// Escalation actions
    pub actions: Vec<EscalationAction>,
    /// Escalation delay
    pub delay: Duration,
}

/// Escalation condition
#[derive(Debug, Clone)]
pub struct EscalationCondition {
    /// Condition type
    pub condition_type: EscalationConditionType,
    /// Condition value
    pub value: String,
    /// Condition operator
    pub operator: ComparisonOperator,
}

/// Escalation action
#[derive(Debug, Clone)]
pub struct EscalationAction {
    /// Action type
    pub action_type: EscalationActionType,
    /// Action parameters
    pub parameters: HashMap<String, String>,
    /// Action priority
    pub priority: f32,
}

/// GPU alert handler trait
pub trait GpuAlertHandler {
    /// Handle a GPU alert
    fn handle_alert(&self, alert: &GpuAlert) -> Result<()>;
    /// Get handler name
    fn name(&self) -> &str;
    /// Check if handler can handle alert type
    fn can_handle(&self, alert_type: &GpuAlertType) -> bool;
}

/// GPU performance tracker
#[derive(Debug)]
pub struct GpuPerformanceTracker {
    /// Performance benchmarks
    benchmarks: Arc<Mutex<HashMap<usize, GpuPerformanceBenchmark>>>,
    /// Performance history
    performance_history: Arc<Mutex<HashMap<usize, Vec<GpuPerformanceRecord>>>>,
    /// Performance baselines
    baselines: Arc<Mutex<HashMap<usize, GpuPerformanceBaseline>>>,
    /// Performance analysis
    analysis: Arc<Mutex<GpuPerformanceAnalysis>>,
}

impl Default for GpuPerformanceTracker {
    fn default() -> Self {
        Self {
            benchmarks: Arc::new(Mutex::new(HashMap::new())),
            performance_history: Arc::new(Mutex::new(HashMap::new())),
            baselines: Arc::new(Mutex::new(HashMap::new())),
            analysis: Arc::new(Mutex::new(GpuPerformanceAnalysis::default())),
        }
    }
}

/// GPU performance benchmark
#[derive(Debug, Clone)]
pub struct GpuPerformanceBenchmark {
    /// Device ID
    pub device_id: usize,
    /// Benchmark name
    pub name: String,
    /// Benchmark type
    pub benchmark_type: GpuBenchmarkType,
    /// Benchmark score
    pub score: f64,
    /// Benchmark timestamp
    pub timestamp: DateTime<Utc>,
    /// Benchmark metadata
    pub metadata: HashMap<String, String>,
}

/// GPU performance record
#[derive(Debug, Clone)]
pub struct GpuPerformanceRecord {
    /// Device ID
    pub device_id: usize,
    /// Test ID
    pub test_id: String,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// Record timestamp
    pub timestamp: DateTime<Utc>,
    /// Test duration
    pub duration: Duration,
}

/// GPU performance baseline
#[derive(Debug, Clone)]
pub struct GpuPerformanceBaseline {
    /// Device ID
    pub device_id: usize,
    /// Baseline metrics
    pub baseline_metrics: HashMap<String, f64>,
    /// Baseline timestamp
    pub established_at: DateTime<Utc>,
    /// Baseline confidence
    pub confidence: f32,
    /// Sample count
    pub sample_count: usize,
}

/// GPU performance analysis
#[derive(Debug, Clone)]
pub struct GpuPerformanceAnalysis {
    /// Performance trends
    pub trends: HashMap<usize, PerformanceTrend>,
    /// Performance regressions detected
    pub regressions: Vec<PerformanceRegression>,
    /// Performance recommendations
    pub recommendations: Vec<PerformanceRecommendation>,
    /// Analysis timestamp
    pub analyzed_at: DateTime<Utc>,
}

impl Default for GpuPerformanceAnalysis {
    fn default() -> Self {
        Self {
            trends: HashMap::new(),
            regressions: Vec::new(),
            recommendations: Vec::new(),
            analyzed_at: Utc::now(),
        }
    }
}

/// Performance trend
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f32,
    /// Trend confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Trend period
    pub period: Duration,
}

/// Performance regression
#[derive(Debug, Clone)]
pub struct PerformanceRegression {
    /// Device ID
    pub device_id: usize,
    /// Metric name
    pub metric_name: String,
    /// Baseline value
    pub baseline_value: f64,
    /// Current value
    pub current_value: f64,
    /// Regression percentage
    pub regression_percent: f32,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Regression severity
    pub severity: RegressionSeverity,
}

/// Performance recommendation
#[derive(Debug, Clone)]
pub struct PerformanceRecommendation {
    /// Device ID
    pub device_id: usize,
    /// Recommendation type
    pub recommendation_type: PerformanceRecommendationType,
    /// Recommendation description
    pub description: String,
    /// Expected impact
    pub expected_impact: f32,
    /// Implementation difficulty
    pub difficulty: RecommendationDifficulty,
    /// Recommendation priority
    pub priority: RecommendationPriority,
}

impl GpuResourceManager {
    /// Create new GPU resource manager
    pub async fn new(config: GpuPoolConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            available_devices: Arc::new(Mutex::new(Vec::new())),
            allocated_resources: Arc::new(Mutex::new(HashMap::new())),
            monitoring_system: Arc::new(GpuMonitoringSystem::default()),
            performance_tracker: Arc::new(GpuPerformanceTracker::default()),
            usage_stats: Arc::new(Mutex::new(GpuUsageStatistics::default())),
        })
    }

    /// Allocate GPU devices for a test
    pub async fn allocate_devices(
        &self,
        device_ids: &[usize],
        test_id: &str,
    ) -> Result<Vec<usize>> {
        info!(
            "Allocating GPU devices {:?} for test: {}",
            device_ids, test_id
        );

        // For now, return the requested device IDs
        // In a real implementation, this would:
        // 1. Check device availability and compatibility
        // 2. Reserve the devices
        // 3. Initialize monitoring
        // 4. Track allocation

        let allocated_devices = device_ids.to_vec();

        // Create allocation records
        for &device_id in &allocated_devices {
            let device_info = GpuDeviceInfo {
                device_id,
                device_name: format!("GPU-{}", device_id),
                total_memory_mb: 8192, // 8GB placeholder
                available_memory_mb: 8192,
                utilization_percent: 0.0,
                capabilities: vec![GpuCapability::Cuda("11.0".to_string())],
                status: GpuDeviceStatus::Busy,
                last_updated: Utc::now(),
            };

            let allocation = GpuAllocation {
                device: device_info,
                test_id: test_id.to_string(),
                memory_allocated_mb: 8192,
                allocated_at: Utc::now(),
                expected_release: None,
                usage_type: GpuUsageType::Training,
                performance_requirements: GpuPerformanceRequirements {
                    min_memory_mb: 1024,
                    min_compute_capability: 6.0,
                    required_frameworks: vec!["CUDA".to_string()],
                    constraints: Vec::new(),
                },
            };

            let mut allocated_resources = self.allocated_resources.lock();
            allocated_resources.insert(format!("{}-{}", test_id, device_id), allocation);
        }

        // Update statistics
        let mut stats = self.usage_stats.lock();
        stats.total_allocations += allocated_devices.len() as u64;
        stats.currently_allocated += allocated_devices.len();

        info!(
            "Allocated GPU devices {:?} for test: {}",
            allocated_devices, test_id
        );
        Ok(allocated_devices)
    }

    /// Deallocate a specific GPU device
    pub async fn deallocate_device(&self, device_id: usize) -> Result<()> {
        debug!("Deallocating GPU device: {}", device_id);

        let mut allocated_resources = self.allocated_resources.lock();
        let keys_to_remove: Vec<String> = allocated_resources
            .iter()
            .filter(|(_, allocation)| allocation.device.device_id == device_id)
            .map(|(key, _)| key.clone())
            .collect();

        for key in keys_to_remove {
            allocated_resources.remove(&key);
        }

        // Update statistics
        let mut stats = self.usage_stats.lock();
        stats.currently_allocated = stats.currently_allocated.saturating_sub(1);

        info!("Successfully deallocated GPU device: {}", device_id);
        Ok(())
    }

    /// Deallocate GPU devices for a test
    pub async fn deallocate_devices_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating GPU devices for test: {}", test_id);

        let mut allocated_resources = self.allocated_resources.lock();
        let keys_to_remove: Vec<String> = allocated_resources
            .iter()
            .filter(|(_, allocation)| allocation.test_id == test_id)
            .map(|(key, _)| key.clone())
            .collect();

        let keys_count = keys_to_remove.len();
        for key in keys_to_remove {
            allocated_resources.remove(&key);
        }

        // Update statistics
        let mut stats = self.usage_stats.lock();
        stats.currently_allocated = stats.currently_allocated.saturating_sub(keys_count);

        info!("Released {} GPU devices for test: {}", keys_count, test_id);
        Ok(())
    }

    /// Check if requested GPU devices are available
    pub async fn check_availability(&self, device_ids: &[usize]) -> Result<bool> {
        // In a real implementation, this would check actual device status
        let allocated_resources = self.allocated_resources.lock();

        for &device_id in device_ids {
            let device_allocated = allocated_resources
                .values()
                .any(|allocation| allocation.device.device_id == device_id);

            if device_allocated {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Get GPU usage statistics
    pub async fn get_statistics(&self) -> Result<GpuUsageStatistics> {
        let stats = self.usage_stats.lock();
        // MutexGuard doesn't implement Clone, dereference to clone the inner value
        Ok((*stats).clone())
    }

    /// Get current GPU allocations
    pub async fn get_allocations(&self) -> Result<Vec<GpuAllocation>> {
        let allocated_resources = self.allocated_resources.lock();
        Ok(allocated_resources.values().cloned().collect())
    }

    /// Get allocations for a specific test
    pub async fn get_allocations_for_test(&self, test_id: &str) -> Result<Vec<GpuAllocation>> {
        let allocated_resources = self.allocated_resources.lock();
        Ok(allocated_resources
            .values()
            .filter(|allocation| allocation.test_id == test_id)
            .cloned()
            .collect())
    }

    /// Update GPU monitoring metrics
    pub async fn update_metrics(
        &self,
        device_id: usize,
        metrics: GpuRealTimeMetrics,
    ) -> Result<()> {
        let mut real_time_metrics = self.monitoring_system.real_time_metrics.lock();
        real_time_metrics.insert(device_id, metrics);
        Ok(())
    }

    /// Get performance recommendations for a device
    pub async fn get_performance_recommendations(
        &self,
        device_id: usize,
    ) -> Result<Vec<PerformanceRecommendation>> {
        let analysis = self.performance_tracker.analysis.lock();
        Ok(analysis
            .recommendations
            .iter()
            .filter(|rec| rec.device_id == device_id)
            .cloned()
            .collect())
    }

    /// Force release all GPU allocations
    pub async fn force_release_all(&self) -> Result<usize> {
        let mut allocated_resources = self.allocated_resources.lock();
        let count = allocated_resources.len();
        allocated_resources.clear();

        // Reset statistics
        let mut stats = self.usage_stats.lock();
        stats.currently_allocated = 0;

        info!("Force released all {} GPU allocations", count);
        Ok(count)
    }
}
