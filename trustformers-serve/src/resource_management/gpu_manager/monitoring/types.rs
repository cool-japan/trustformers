//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::{
    GpuAlertConfig as GpuManagerAlertConfig, GpuAlertThresholds as GpuManagerAlertThresholds,
    GpuClockSpeeds as GpuManagerClockSpeeds, GpuRealTimeMetrics as GpuManagerRealTimeMetrics,
};
use crate::resource_management::gpu_manager::alert_system::GpuAlertSystem as GpuAlertSystemImpl;
use crate::resource_management::types::*;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, instrument, warn};

use super::functions::GpuMonitoringResult;

/// GPU metrics summary for statistical analysis
#[derive(Debug, Clone)]
pub struct GpuMetricsSummary {
    pub device_id: usize,
    pub metric_type: GpuMetricType,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub sample_count: usize,
    pub average: f64,
    pub minimum: f64,
    pub maximum: f64,
    pub standard_deviation: f64,
    pub trend: TrendDirection,
}
/// GPU monitoring system for real-time device tracking
///
/// Provides comprehensive monitoring capabilities including:
/// - Real-time metrics collection and storage
/// - Historical data retention and analysis
/// - Performance trend detection
/// - Alert integration for threshold monitoring
/// - Background monitoring task management
/// - Device status tracking
/// - Event generation and notifications
///
/// # Architecture
///
/// The monitoring system is designed as a high-performance, thread-safe component
/// that can handle concurrent monitoring operations across multiple GPU devices.
/// It uses atomic operations for state management and RwLocks for data protection
/// while maintaining high throughput.
///
/// # Performance Considerations
///
/// - Real-time metrics are stored in memory with configurable retention
/// - Historical data is maintained with automatic cleanup to prevent memory growth
/// - Background tasks use efficient async scheduling to minimize overhead
/// - Alert processing is decoupled from metrics collection for performance
///
/// # Thread Safety
///
/// All operations are thread-safe and can be called concurrently from multiple
/// threads. The monitoring system uses lock-free atomic operations where possible
/// and fine-grained locking for data structures.
#[derive(Debug)]
pub struct GpuMonitoringSystem {
    /// Monitoring configuration
    ///
    /// Contains all configuration parameters for the monitoring system including
    /// intervals, thresholds, and feature flags. Configuration can be updated
    /// at runtime through the RwLock protection.
    config: Arc<RwLock<GpuMonitoringConfig>>,
    /// Real-time metrics storage
    ///
    /// Stores the most recent metrics for each GPU device. This provides fast
    /// access to current device status and is updated continuously by the
    /// monitoring system.
    realtime_metrics: Arc<RwLock<HashMap<usize, GpuRealTimeMetrics>>>,
    /// Historical metrics database
    ///
    /// Maintains a rolling buffer of historical metrics data for trend analysis
    /// and performance tracking. Data is automatically pruned based on retention
    /// policies to prevent unbounded memory growth.
    historical_metrics: Arc<RwLock<VecDeque<GpuHistoricalMetric>>>,
    /// Alert system integration
    ///
    /// Provides integration with the GPU alert system for threshold-based
    /// monitoring and notification generation. Alerts are processed asynchronously
    /// to avoid blocking metrics collection.
    alert_system: Arc<GpuAlertSystemImpl>,
    /// Monitoring active flag
    ///
    /// Atomic flag indicating whether monitoring operations are currently active.
    /// Used for coordination between background tasks and system lifecycle management.
    pub(super) monitoring_active: Arc<AtomicBool>,
    /// Background task handles
    ///
    /// Manages handles to all background monitoring tasks for proper cleanup
    /// and shutdown coordination. Tasks include metrics collection, alert processing,
    /// and data maintenance operations.
    background_tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
impl GpuMonitoringSystem {
    /// Create new monitoring system
    ///
    /// Initializes a new GPU monitoring system with the provided configuration.
    /// This sets up all internal data structures and prepares the system for
    /// monitoring operations, but does not start background tasks.
    ///
    /// # Arguments
    ///
    /// * `config` - Monitoring system configuration including intervals, thresholds,
    ///   and feature flags
    ///
    /// # Returns
    ///
    /// A new GpuMonitoringSystem instance ready to begin monitoring operations
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Alert system initialization fails
    /// - Invalid configuration parameters provided
    /// - System resource allocation fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use trustformers_serve::resource_management::gpu_manager::monitoring::GpuMonitoringSystem;
    /// use trustformers_serve::resource_management::types::*;
    /// use std::time::Duration;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let config = GpuMonitoringConfig {
    ///         real_time_monitoring: true,
    ///         monitoring_interval: Duration::from_secs(5),
    ///         enable_performance_tracking: true,
    ///         enable_alerts: true,
    ///         alert_config: GpuAlertConfig::default(),
    ///         ..Default::default()
    ///     };
    ///
    ///     let monitoring_system = GpuMonitoringSystem::new(config).await?;
    ///     Ok(())
    /// }
    /// ```
    #[instrument(skip(config))]
    pub async fn new(config: GpuMonitoringConfig) -> GpuMonitoringResult<Self> {
        info!("Initializing GPU monitoring system");
        Self::validate_config(&config)?;
        let gpu_manager_thresholds = GpuManagerAlertThresholds {
            temperature_warning: config.alert_config.thresholds.high_temperature * 0.9,
            temperature_critical: config.alert_config.thresholds.high_temperature,
            utilization_critical_percent: config.alert_config.thresholds.high_utilization * 100.0,
            memory_critical_percent: config.alert_config.thresholds.high_memory_usage * 100.0,
            power_warning_watts: config.alert_config.thresholds.high_power_consumption * 0.9,
            power_critical_watts: config.alert_config.thresholds.high_power_consumption,
            utilization_warning_percent: config.alert_config.thresholds.high_utilization * 80.0,
            memory_warning_percent: config.alert_config.thresholds.high_memory_usage * 80.0,
        };
        let gpu_manager_alert_config = GpuManagerAlertConfig {
            enable_temperature_alerts: true,
            enable_utilization_alerts: true,
            enable_memory_alerts: true,
            thresholds: gpu_manager_thresholds,
            escalation_rules: Vec::new(),
            cooldown_period: config.alert_config.cooldown_period,
            escalation_enabled: config.alert_config.max_escalation_level > 0,
            escalation_delay_seconds: 600,
        };
        let alert_system = Arc::new(
            GpuAlertSystemImpl::new(gpu_manager_alert_config).await.map_err(|e| {
                GpuMonitoringError::AlertError {
                    message: format!("Failed to initialize alert system: {}", e),
                }
            })?,
        );
        let monitoring_system = Self {
            config: Arc::new(RwLock::new(config)),
            realtime_metrics: Arc::new(RwLock::new(HashMap::new())),
            historical_metrics: Arc::new(RwLock::new(VecDeque::new())),
            alert_system,
            monitoring_active: Arc::new(AtomicBool::new(false)),
            background_tasks: Arc::new(Mutex::new(Vec::new())),
        };
        info!("GPU monitoring system initialized successfully");
        Ok(monitoring_system)
    }
    /// Validate monitoring system configuration
    ///
    /// Performs comprehensive validation of the monitoring configuration to ensure
    /// all parameters are within acceptable ranges and the system can operate
    /// correctly with the provided settings.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration to validate
    ///
    /// # Returns
    ///
    /// Ok(()) if configuration is valid
    ///
    /// # Errors
    ///
    /// Returns ConfigurationError if any validation checks fail
    fn validate_config(config: &GpuMonitoringConfig) -> GpuMonitoringResult<()> {
        if config.monitoring_interval.as_millis() == 0 {
            return Err(GpuMonitoringError::ConfigurationError {
                field: "monitoring_interval".to_string(),
                message: "Monitoring interval must be greater than 0 seconds".to_string(),
            });
        }
        if config.monitoring_interval.as_secs() > 3600 {
            return Err(GpuMonitoringError::ConfigurationError {
                field: "monitoring_interval".to_string(),
                message: "Monitoring interval should not exceed 1 hour".to_string(),
            });
        }
        if config.enable_alerts && config.alert_config.thresholds.high_temperature < 0.0 {
            return Err(GpuMonitoringError::ConfigurationError {
                field: "high_temperature".to_string(),
                message: "Temperature threshold must be non-negative".to_string(),
            });
        }
        info!("GPU monitoring configuration validated successfully");
        Ok(())
    }
    /// Start monitoring operations
    ///
    /// Activates the monitoring system and starts all background tasks including
    /// metrics collection, alert processing, and data maintenance. This method
    /// is idempotent and can be called multiple times safely.
    ///
    /// # Returns
    ///
    /// Ok(()) if monitoring started successfully
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Alert system fails to start
    /// - Background task creation fails
    /// - System is already in an inconsistent state
    ///
    /// # Performance Notes
    ///
    /// Starting monitoring begins background task execution which will consume
    /// system resources. Ensure adequate CPU and memory resources are available
    /// for optimal performance.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use trustformers_serve::resource_management::gpu_manager::monitoring::GpuMonitoringSystem;
    /// # use trustformers_serve::resource_management::types::*;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = GpuMonitoringConfig::default();
    /// let monitoring_system = GpuMonitoringSystem::new(config).await?;
    ///
    /// // Start monitoring operations
    /// monitoring_system.start_monitoring().await?;
    ///
    /// // Monitoring is now active and collecting metrics
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self))]
    pub async fn start_monitoring(&self) -> GpuMonitoringResult<()> {
        if self.monitoring_active.load(Ordering::Acquire) {
            debug!("GPU monitoring system is already running");
            return Ok(());
        }
        info!("Starting GPU monitoring system");
        self.alert_system.start().await.map_err(|e| GpuMonitoringError::AlertError {
            message: format!("Failed to start alert system: {}", e),
        })?;
        self.start_background_tasks().await?;
        self.monitoring_active.store(true, Ordering::Release);
        info!("GPU monitoring system started successfully");
        Ok(())
    }
    /// Stop monitoring operations
    ///
    /// Gracefully shuts down all monitoring operations including background tasks
    /// and alert processing. This method ensures all resources are properly cleaned
    /// up and all tasks complete or are cancelled appropriately.
    ///
    /// # Returns
    ///
    /// Ok(()) if monitoring stopped successfully
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Alert system fails to stop cleanly
    /// - Background tasks fail to shutdown
    /// - Resource cleanup encounters errors
    ///
    /// # Shutdown Process
    ///
    /// 1. Signal all background tasks to stop
    /// 2. Stop the alert system
    /// 3. Wait for task completion or force termination
    /// 4. Clean up resources and mark system as inactive
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use trustformers_serve::resource_management::gpu_manager::monitoring::GpuMonitoringSystem;
    /// # use trustformers_serve::resource_management::types::*;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = GpuMonitoringConfig::default();
    /// # let monitoring_system = GpuMonitoringSystem::new(config).await?;
    /// # monitoring_system.start_monitoring().await?;
    ///
    /// // Stop monitoring operations
    /// monitoring_system.stop_monitoring().await?;
    ///
    /// // Monitoring is now stopped and resources cleaned up
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self))]
    pub async fn stop_monitoring(&self) -> GpuMonitoringResult<()> {
        if !self.monitoring_active.load(Ordering::Acquire) {
            debug!("GPU monitoring system is not running");
            return Ok(());
        }
        info!("Stopping GPU monitoring system");
        self.alert_system.stop().await.map_err(|e| GpuMonitoringError::AlertError {
            message: format!("Failed to stop alert system: {}", e),
        })?;
        self.stop_background_tasks().await?;
        self.monitoring_active.store(false, Ordering::Release);
        info!("GPU monitoring system stopped successfully");
        Ok(())
    }
    /// Start background monitoring tasks
    ///
    /// Initializes and starts all background tasks required for monitoring operations.
    /// This includes metrics collection, data maintenance, and periodic cleanup tasks.
    async fn start_background_tasks(&self) -> GpuMonitoringResult<()> {
        debug!("Starting background monitoring tasks");
        self.start_historical_cleanup_task().await?;
        self.start_metrics_aggregation_task().await?;
        self.start_device_status_task().await?;
        debug!("All background monitoring tasks started successfully");
        Ok(())
    }
    /// Stop all background monitoring tasks
    ///
    /// Gracefully stops all background tasks and waits for their completion.
    /// Tasks that don't complete within a reasonable time are forcibly terminated.
    async fn stop_background_tasks(&self) -> GpuMonitoringResult<()> {
        debug!("Stopping background monitoring tasks");
        let mut tasks = self.background_tasks.lock();
        for task in tasks.drain(..) {
            if !task.is_finished() {
                task.abort();
                debug!("Background task terminated");
            }
        }
        debug!("All background monitoring tasks stopped");
        Ok(())
    }
    /// Start historical data cleanup task
    ///
    /// Starts a background task that periodically cleans up old historical data
    /// to prevent unbounded memory growth. Cleanup frequency and retention
    /// policies are based on configuration settings.
    async fn start_historical_cleanup_task(&self) -> GpuMonitoringResult<()> {
        let historical_metrics = self.historical_metrics.clone();
        let config = self.config.clone();
        let monitoring_active = self.monitoring_active.clone();
        let task = tokio::spawn(async move {
            let mut cleanup_interval = tokio::time::interval(Duration::from_secs(300));
            while monitoring_active.load(Ordering::Acquire) {
                tokio::select! {
                    _ = cleanup_interval.tick() => { let config_read = config.read(); let
                    max_entries = Self::calculate_max_historical_entries(& config_read);
                    let mut historical = historical_metrics.write(); while historical
                    .len() > max_entries { historical.pop_front(); } if historical.len()
                    > max_entries * 8 / 10 {
                    debug!("Historical data cleanup completed, {} entries retained",
                    historical.len()); } }
                }
            }
        });
        self.background_tasks.lock().push(task);
        Ok(())
    }
    /// Calculate maximum number of historical entries to retain
    ///
    /// Determines the optimal number of historical entries to keep based on
    /// monitoring interval and desired retention period.
    pub(crate) fn calculate_max_historical_entries(config: &GpuMonitoringConfig) -> usize {
        let retention_hours = 24;
        let entries_per_hour = 3600 / config.monitoring_interval.as_secs().max(1);
        (entries_per_hour * retention_hours) as usize
    }
    /// Start metrics aggregation task
    ///
    /// Starts a background task that performs periodic aggregation and analysis
    /// of collected metrics data. This includes computing averages, trends,
    /// and statistical summaries.
    async fn start_metrics_aggregation_task(&self) -> GpuMonitoringResult<()> {
        let realtime_metrics = self.realtime_metrics.clone();
        let historical_metrics = self.historical_metrics.clone();
        let monitoring_active = self.monitoring_active.clone();
        let task = tokio::spawn(async move {
            let mut aggregation_interval = tokio::time::interval(Duration::from_secs(60));
            while monitoring_active.load(Ordering::Acquire) {
                tokio::select! {
                    _ = aggregation_interval.tick() => { let realtime = realtime_metrics
                    .read(); let historical = historical_metrics.read(); for (device_id,
                    _current_metrics) in realtime.iter() { let recent_metrics : Vec < _ >
                    = historical.iter().filter(| m | m.device_id == * device_id).filter(|
                    m | { let age = Utc::now().signed_duration_since(m.timestamp); age
                    .num_seconds() < 3600 }).collect(); if ! recent_metrics.is_empty() {
                    debug!("Aggregated {} metrics for device {}", recent_metrics.len(),
                    device_id); } } }
                }
            }
        });
        self.background_tasks.lock().push(task);
        Ok(())
    }
    /// Start device status monitoring task
    ///
    /// Starts a background task that monitors the overall status and health
    /// of devices based on metrics patterns and thresholds.
    async fn start_device_status_task(&self) -> GpuMonitoringResult<()> {
        let realtime_metrics = self.realtime_metrics.clone();
        let monitoring_active = self.monitoring_active.clone();
        let task = tokio::spawn(async move {
            let mut status_interval = tokio::time::interval(Duration::from_secs(30));
            while monitoring_active.load(Ordering::Acquire) {
                tokio::select! {
                    _ = status_interval.tick() => {
                        let metrics = realtime_metrics.read();
                        for (device_id, device_metrics) in metrics.iter() {
                            if device_metrics.temperature_celsius > 85.0 {
                                warn!("Device {} temperature is high: {:.1}°C", device_id,
                                    device_metrics.temperature_celsius);
                            }

                            if device_metrics.utilization_percent > 95.0 {
                                warn!("Device {} utilization is very high: {:.1}%", device_id,
                                    device_metrics.utilization_percent);
                            }

                            let memory_usage_percent =
                                (device_metrics.memory_usage_mb as f32 / 24576.0) * 100.0;
                            if memory_usage_percent > 90.0 {
                                warn!("Device {} memory usage is high: {:.1}%", device_id,
                                    memory_usage_percent);
                            }
                        }
                    }
                }
            }
        });
        self.background_tasks.lock().push(task);
        Ok(())
    }
    /// Update real-time metrics for a device
    ///
    /// Updates the monitoring system with new real-time metrics for a specific GPU device.
    /// This method processes the metrics, stores them in both real-time and historical
    /// storage, and triggers alert checking if configured.
    ///
    /// # Arguments
    ///
    /// * `device_id` - ID of the GPU device
    /// * `metrics` - New real-time metrics data for the device
    ///
    /// # Returns
    ///
    /// Ok(()) if metrics were updated successfully
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Alert system check fails
    /// - Historical data storage fails
    /// - Device ID is invalid
    ///
    /// # Performance Notes
    ///
    /// This method is designed for high-frequency updates and uses efficient
    /// locking strategies to minimize contention. Multiple devices can be
    /// updated concurrently.
    ///
    /// # Data Processing
    ///
    /// The method performs several operations:
    /// 1. Updates real-time metrics storage
    /// 2. Adds data points to historical metrics
    /// 3. Maintains historical data size limits
    /// 4. Triggers alert threshold checking
    /// 5. Logs significant metric changes
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use trustformers_serve::resource_management::gpu_manager::monitoring::GpuMonitoringSystem;
    /// # use trustformers_serve::resource_management::types::*;
    /// # use chrono::Utc;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = GpuMonitoringConfig::default();
    /// # let monitoring_system = GpuMonitoringSystem::new(config).await?;
    ///
    /// let metrics = GpuRealTimeMetrics {
    ///     device_id: 0,
    ///     timestamp: Utc::now(),
    ///     memory_usage_mb: 8192,
    ///     utilization_percent: 75.0,
    ///     temperature_celsius: 65.0,
    ///     power_consumption_watts: 200.0,
    ///     clock_speeds: GpuClockSpeeds {
    ///         core_clock_mhz: 1800,
    ///         memory_clock_mhz: 7000,
    ///         shader_clock_mhz: Some(1900),
    ///     },
    ///     fan_speeds: vec![50.0],
    /// };
    ///
    /// monitoring_system.update_metrics(0, metrics).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self, metrics), fields(device_id = %device_id))]
    pub async fn update_metrics(
        &self,
        device_id: usize,
        metrics: GpuRealTimeMetrics,
    ) -> GpuMonitoringResult<()> {
        if metrics.device_id != device_id {
            return Err(GpuMonitoringError::MetricsCollectionError {
                details: format!(
                    "Device ID mismatch: expected {}, got {}",
                    device_id, metrics.device_id
                ),
            });
        }
        {
            let mut realtime = self.realtime_metrics.write();
            if let Some(previous) = realtime.get(&device_id) {
                let temp_change =
                    (metrics.temperature_celsius - previous.temperature_celsius).abs();
                let util_change =
                    (metrics.utilization_percent - previous.utilization_percent).abs();
                if temp_change > 10.0 {
                    debug!(
                        "Significant temperature change on device {}: {:.1}°C -> {:.1}°C",
                        device_id, previous.temperature_celsius, metrics.temperature_celsius
                    );
                }
                if util_change > 20.0 {
                    debug!(
                        "Significant utilization change on device {}: {:.1}% -> {:.1}%",
                        device_id, previous.utilization_percent, metrics.utilization_percent
                    );
                }
            }
            realtime.insert(device_id, metrics.clone());
        }
        {
            let mut historical = self.historical_metrics.write();
            let metric_entries = vec![
                GpuHistoricalMetric {
                    device_id,
                    metric_type: GpuMetricType::Utilization,
                    value: metrics.utilization_percent as f64,
                    timestamp: metrics.timestamp,
                    test_id: None,
                    metadata: HashMap::new(),
                },
                GpuHistoricalMetric {
                    device_id,
                    metric_type: GpuMetricType::Temperature,
                    value: metrics.temperature_celsius as f64,
                    timestamp: metrics.timestamp,
                    test_id: None,
                    metadata: HashMap::new(),
                },
                GpuHistoricalMetric {
                    device_id,
                    metric_type: GpuMetricType::MemoryUsage,
                    value: metrics.memory_usage_mb as f64,
                    timestamp: metrics.timestamp,
                    test_id: None,
                    metadata: HashMap::new(),
                },
                GpuHistoricalMetric {
                    device_id,
                    metric_type: GpuMetricType::PowerConsumption,
                    value: metrics.power_consumption_watts as f64,
                    timestamp: metrics.timestamp,
                    test_id: None,
                    metadata: HashMap::new(),
                },
            ];
            for entry in metric_entries {
                historical.push_back(entry);
            }
            let config = self.config.read();
            let max_entries = Self::calculate_max_historical_entries(&config);
            while historical.len() > max_entries {
                historical.pop_front();
            }
        }
        let enable_alerts = {
            let config = self.config.read();
            config.enable_alerts
        };
        if enable_alerts {
            let gpu_manager_metrics = GpuManagerRealTimeMetrics {
                device_id: metrics.device_id,
                timestamp: metrics.timestamp,
                memory_usage_mb: metrics.memory_usage_mb,
                utilization_percent: metrics.utilization_percent,
                temperature_celsius: metrics.temperature_celsius,
                power_consumption_watts: metrics.power_consumption_watts,
                clock_speeds: GpuManagerClockSpeeds {
                    core_clock_mhz: metrics.clock_speeds.core_clock_mhz,
                    memory_clock_mhz: metrics.clock_speeds.memory_clock_mhz,
                    shader_clock_mhz: metrics.clock_speeds.shader_clock_mhz,
                },
                fan_speeds: metrics.fan_speeds.clone(),
            };
            if let Err(e) = self
                .alert_system
                .check_metrics_for_alerts(device_id, &gpu_manager_metrics)
                .await
            {
                error!(
                    "Failed to check metrics for alerts on device {}: {}",
                    device_id, e
                );
            }
        }
        debug!(
            "Updated metrics for GPU device {} - Util: {:.1}%, Temp: {:.1}°C, Memory: {}MB",
            device_id,
            metrics.utilization_percent,
            metrics.temperature_celsius,
            metrics.memory_usage_mb
        );
        Ok(())
    }
    /// Get current real-time metrics
    ///
    /// Retrieves the most recent real-time metrics for all monitored GPU devices.
    /// This provides a snapshot of current device performance and status.
    ///
    /// # Returns
    ///
    /// HashMap containing device IDs mapped to their current real-time metrics
    ///
    /// # Performance Notes
    ///
    /// This method uses a read lock and creates a clone of the current metrics
    /// data. For frequently accessed data, consider caching results or using
    /// the data directly rather than repeated calls.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use trustformers_serve::resource_management::gpu_manager::monitoring::GpuMonitoringSystem;
    /// # use trustformers_serve::resource_management::types::*;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = GpuMonitoringConfig::default();
    /// # let monitoring_system = GpuMonitoringSystem::new(config).await?;
    ///
    /// let current_metrics = monitoring_system.get_realtime_metrics().await;
    ///
    /// for (device_id, metrics) in current_metrics {
    ///     println!("Device {}: {:.1}% utilization, {:.1}°C temperature",
    ///              device_id, metrics.utilization_percent, metrics.temperature_celsius);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_realtime_metrics(&self) -> HashMap<usize, GpuRealTimeMetrics> {
        let metrics = self.realtime_metrics.read();
        metrics.clone()
    }
    /// Get historical metrics for analysis
    ///
    /// Retrieves historical metrics data with optional filtering by device ID,
    /// metric type, and time range. This enables detailed performance analysis
    /// and trend identification.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Optional device ID filter (None for all devices)
    /// * `metric_type` - Optional metric type filter (None for all types)
    /// * `since` - Optional time filter for data since specific timestamp (None for all data)
    ///
    /// # Returns
    ///
    /// Vector of historical metric entries matching the filter criteria
    ///
    /// # Performance Considerations
    ///
    /// - Large time ranges may return significant amounts of data
    /// - Consider using time filters for better performance
    /// - Results are sorted chronologically
    ///
    /// # Filtering Logic
    ///
    /// Filters are applied in combination (AND logic):
    /// - If device_id is specified, only metrics for that device are returned
    /// - If metric_type is specified, only metrics of that type are returned
    /// - If since is specified, only metrics after that timestamp are returned
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use trustformers_serve::resource_management::gpu_manager::monitoring::GpuMonitoringSystem;
    /// # use trustformers_serve::resource_management::types::*;
    /// # use chrono::{Utc, Duration};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = GpuMonitoringConfig::default();
    /// # let monitoring_system = GpuMonitoringSystem::new(config).await?;
    ///
    /// // Get all historical data
    /// let all_data = monitoring_system.get_historical_metrics(None, None, None).await;
    ///
    /// // Get temperature data for device 0
    /// let device_0_temp = monitoring_system.get_historical_metrics(
    ///     Some(0),
    ///     Some(GpuMetricType::Temperature),
    ///     None
    /// ).await;
    ///
    /// // Get recent data (last hour)
    /// let since = Utc::now() - Duration::hours(1);
    /// let recent_data = monitoring_system.get_historical_metrics(
    ///     None,
    ///     None,
    ///     Some(since)
    /// ).await;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_historical_metrics(
        &self,
        device_id: Option<usize>,
        metric_type: Option<GpuMetricType>,
        since: Option<DateTime<Utc>>,
    ) -> Vec<GpuHistoricalMetric> {
        let historical = self.historical_metrics.read();
        let filtered_metrics: Vec<GpuHistoricalMetric> = historical
            .iter()
            .filter(|metric| {
                if let Some(id) = device_id {
                    if metric.device_id != id {
                        return false;
                    }
                }
                if let Some(ref mtype) = metric_type {
                    if std::mem::discriminant(&metric.metric_type) != std::mem::discriminant(mtype)
                    {
                        return false;
                    }
                }
                if let Some(since_time) = since {
                    if metric.timestamp < since_time {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();
        debug!(
            "Retrieved {} historical metrics with filters: device_id={:?}, metric_type={:?}, since={:?}",
            filtered_metrics.len(), device_id, metric_type, since
        );
        filtered_metrics
    }
    /// Get metrics summary for a device
    ///
    /// Computes a statistical summary of metrics for a specific device over
    /// a specified time period. This includes averages, minimums, maximums,
    /// and trend information.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Device ID to analyze
    /// * `metric_type` - Type of metric to summarize
    /// * `period` - Time period for analysis
    ///
    /// # Returns
    ///
    /// Statistical summary of the requested metrics
    pub async fn get_metrics_summary(
        &self,
        device_id: usize,
        metric_type: GpuMetricType,
        period: Duration,
    ) -> Option<GpuMetricsSummary> {
        let since = Utc::now() - chrono::Duration::from_std(period).ok()?;
        let metric_type_clone = metric_type.clone();
        let metrics = self
            .get_historical_metrics(Some(device_id), Some(metric_type), Some(since))
            .await;
        if metrics.is_empty() {
            return None;
        }
        let values: Vec<f64> = metrics.iter().map(|m| m.value).collect();
        let count = values.len();
        let sum: f64 = values.iter().sum();
        let average = sum / count as f64;
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let variance = values
            .iter()
            .map(|value| {
                let diff = average - value;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let std_dev = variance.sqrt();
        Some(GpuMetricsSummary {
            device_id,
            metric_type: metric_type_clone,
            period_start: since,
            period_end: Utc::now(),
            sample_count: count,
            average,
            minimum: min,
            maximum: max,
            standard_deviation: std_dev,
            trend: self.calculate_trend(&values),
        })
    }
    /// Calculate trend direction from a series of values
    ///
    /// Analyzes a series of metric values to determine the overall trend
    /// direction using linear regression.
    fn calculate_trend(&self, values: &[f64]) -> TrendDirection {
        if values.len() < 2 {
            return TrendDirection::Stable;
        }
        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x2_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();
        let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum);
        if slope > 0.1 {
            TrendDirection::Improving
        } else if slope < -0.1 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }
    /// Check if monitoring is currently active
    ///
    /// Returns whether the monitoring system is currently running and
    /// collecting metrics.
    pub fn is_monitoring_active(&self) -> bool {
        self.monitoring_active.load(Ordering::Acquire)
    }
    /// Get monitoring system configuration
    ///
    /// Returns a copy of the current monitoring system configuration.
    pub async fn get_config(&self) -> GpuMonitoringConfig {
        let config = self.config.read();
        config.clone()
    }
    /// Update monitoring system configuration
    ///
    /// Updates the monitoring system configuration. Some changes may require
    /// restarting the monitoring system to take effect.
    ///
    /// # Arguments
    ///
    /// * `new_config` - New configuration to apply
    ///
    /// # Returns
    ///
    /// Ok(()) if configuration was updated successfully
    ///
    /// # Errors
    ///
    /// Returns error if configuration validation fails
    pub async fn update_config(&self, new_config: GpuMonitoringConfig) -> GpuMonitoringResult<()> {
        Self::validate_config(&new_config)?;
        {
            let mut config = self.config.write();
            *config = new_config;
        }
        info!("GPU monitoring configuration updated successfully");
        Ok(())
    }
    /// Get monitoring system statistics
    ///
    /// Returns comprehensive statistics about the monitoring system including
    /// data retention, processing performance, and system health.
    pub async fn get_monitoring_statistics(&self) -> GpuMonitoringStatistics {
        let realtime_count = {
            let metrics = self.realtime_metrics.read();
            metrics.len()
        };
        let historical_count = {
            let metrics = self.historical_metrics.read();
            metrics.len()
        };
        let config = self.config.read();
        let background_task_count = {
            let tasks = self.background_tasks.lock();
            tasks.len()
        };
        GpuMonitoringStatistics {
            is_active: self.is_monitoring_active(),
            monitored_devices: realtime_count,
            historical_entries: historical_count,
            background_tasks: background_task_count,
            monitoring_interval: config.monitoring_interval,
            alerts_enabled: config.enable_alerts,
            performance_tracking_enabled: config.enable_performance_tracking,
            uptime_seconds: 0,
        }
    }
}
/// Comprehensive error types for GPU monitoring operations
#[derive(Debug, thiserror::Error)]
pub enum GpuMonitoringError {
    #[error("Monitoring system error: {source}")]
    MonitoringError {
        #[from]
        source: anyhow::Error,
    },
    #[error("Alert system error: {message}")]
    AlertError { message: String },
    #[error("Configuration error: {field} - {message}")]
    ConfigurationError { field: String, message: String },
    #[error("Device {device_id} not found in monitoring system")]
    DeviceNotFound { device_id: usize },
    #[error("Metrics collection failed: {details}")]
    MetricsCollectionError { details: String },
    #[error("Historical data error: {message}")]
    HistoricalDataError { message: String },
    #[error("Background task error: {task_name} - {error}")]
    BackgroundTaskError { task_name: String, error: String },
}
/// GPU monitoring system statistics
#[derive(Debug, Clone)]
pub struct GpuMonitoringStatistics {
    pub is_active: bool,
    pub monitored_devices: usize,
    pub historical_entries: usize,
    pub background_tasks: usize,
    pub monitoring_interval: Duration,
    pub alerts_enabled: bool,
    pub performance_tracking_enabled: bool,
    pub uptime_seconds: u64,
}
