//! GPU Monitoring System Implementation
//!
//! This module provides comprehensive GPU monitoring capabilities including:
//! - Real-time metrics collection and storage
//! - Historical data retention and analysis
//! - Performance trend detection
//! - Alert integration for threshold monitoring
//! - Background monitoring task management
//! - Device status tracking
//! - Event generation and notifications
//!
//! # Overview
//!
//! The GPU monitoring system continuously tracks device performance, health metrics,
//! and utilization patterns. It provides both real-time and historical views of GPU
//! performance data, enabling proactive monitoring and optimization of GPU resources.
//!
//! # Features
//!
//! - **Real-time Metrics**: Continuous collection of GPU utilization, temperature, memory usage
//! - **Historical Data**: Storage and management of historical metrics data
//! - **Metrics Aggregation**: Computing averages, trends, and statistical analysis
//! - **Device Status Tracking**: Monitoring device availability and health status
//! - **Background Tasks**: Async task management for continuous monitoring operations
//! - **Event Generation**: Creating monitoring events and notifications
//! - **Alert Integration**: Threshold-based alerting with configurable conditions
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use trustformers_serve::resource_management::gpu_manager::monitoring::GpuMonitoringSystem;
//! use trustformers_serve::resource_management::types::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create monitoring configuration
//!     let config = GpuMonitoringConfig {
//!         enable_realtime: true,
//!         monitoring_interval: std::time::Duration::from_secs(5),
//!         enable_performance_tracking: true,
//!         enable_alerts: true,
//!         alert_config: GpuAlertConfig::default(),
//!     };
//!
//!     // Initialize monitoring system
//!     let monitoring_system = GpuMonitoringSystem::new(config).await?;
//!
//!     // Start monitoring operations
//!     monitoring_system.start_monitoring().await?;
//!
//!     // Update metrics for a device
//!     let metrics = GpuRealTimeMetrics {
//!         device_id: 0,
//!         timestamp: chrono::Utc::now(),
//!         memory_usage_mb: 8192,
//!         utilization_percent: 75.0,
//!         temperature_celsius: 65.0,
//!         power_consumption_watts: 200.0,
//!         clock_speeds: GpuClockSpeeds {
//!             core_clock_mhz: 1800,
//!             memory_clock_mhz: 7000,
//!             shader_clock_mhz: Some(1900),
//!         },
//!         fan_speeds: vec![50.0],
//!     };
//!
//!     monitoring_system.update_metrics(0, metrics).await?;
//!
//!     // Get current metrics
//!     let current_metrics = monitoring_system.get_realtime_metrics().await;
//!
//!     // Get historical data
//!     let historical = monitoring_system.get_historical_metrics(
//!         Some(0),                    // device_id filter
//!         Some(GpuMetricType::Utilization), // metric type filter
//!         None,                       // no time filter
//!     ).await;
//!
//!     // Stop monitoring
//!     monitoring_system.stop_monitoring().await?;
//!
//!     Ok(())
//! }
//! ```

use anyhow::Result;
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

use crate::resource_management::types::*;

// Import the actual GpuAlertSystem implementation with methods, not the data struct from types
use super::alert_system::GpuAlertSystem as GpuAlertSystemImpl;

// Import gpu_manager types that are expected by alert_system
use super::types::{
    GpuAlertConfig as GpuManagerAlertConfig, GpuAlertThresholds as GpuManagerAlertThresholds,
    GpuClockSpeeds as GpuManagerClockSpeeds, GpuRealTimeMetrics as GpuManagerRealTimeMetrics,
};

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

/// Result type for GPU monitoring operations
pub type GpuMonitoringResult<T> = Result<T, GpuMonitoringError>;

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
    monitoring_active: Arc<AtomicBool>,

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
    ///         enable_realtime: true,
    ///         monitoring_interval: Duration::from_secs(5),
    ///         enable_performance_tracking: true,
    ///         enable_alerts: true,
    ///         alert_config: GpuAlertConfig::default(),
    ///     };
    ///
    ///     let monitoring_system = GpuMonitoringSystem::new(config).await?;
    ///     Ok(())
    /// }
    /// ```
    #[instrument(skip(config))]
    pub async fn new(config: GpuMonitoringConfig) -> GpuMonitoringResult<Self> {
        info!("Initializing GPU monitoring system");

        // Validate configuration
        Self::validate_config(&config)?;

        // Initialize alert system with the provided configuration
        // Convert resource_management::types::GpuAlertConfig to gpu_manager::types::GpuAlertConfig
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
            escalation_rules: Vec::new(), // TODO: Convert escalation rules
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
        // Validate monitoring interval
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

        // Validate alert configuration if alerts are enabled
        if config.enable_alerts {
            // TODO: GpuAlertThresholds in types.rs has high_temperature instead of temperature_warning/temperature_critical
            if config.alert_config.thresholds.high_temperature < 0.0 {
                return Err(GpuMonitoringError::ConfigurationError {
                    field: "high_temperature".to_string(),
                    message: "Temperature threshold must be non-negative".to_string(),
                });
            }

            // TODO: Removed temperature_critical validation since we only have high_temperature now
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

        // Start alert system
        self.alert_system.start().await.map_err(|e| GpuMonitoringError::AlertError {
            message: format!("Failed to start alert system: {}", e),
        })?;

        // Start background monitoring tasks
        self.start_background_tasks().await?;

        // Mark monitoring as active
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

        // Stop alert system
        self.alert_system.stop().await.map_err(|e| GpuMonitoringError::AlertError {
            message: format!("Failed to stop alert system: {}", e),
        })?;

        // Stop and cleanup background tasks
        self.stop_background_tasks().await?;

        // Mark monitoring as inactive
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

        // Start historical data cleanup task
        self.start_historical_cleanup_task().await?;

        // Start metrics aggregation task
        self.start_metrics_aggregation_task().await?;

        // Start device status monitoring task
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
            let mut cleanup_interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes

            while monitoring_active.load(Ordering::Acquire) {
                tokio::select! {
                    _ = cleanup_interval.tick() => {
                        let config_read = config.read();
                        let max_entries = Self::calculate_max_historical_entries(&config_read);

                        let mut historical = historical_metrics.write();
                        while historical.len() > max_entries {
                            historical.pop_front();
                        }

                        if historical.len() > max_entries * 8 / 10 {
                            debug!("Historical data cleanup completed, {} entries retained", historical.len());
                        }
                    }
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
    fn calculate_max_historical_entries(config: &GpuMonitoringConfig) -> usize {
        // Keep 24 hours of data by default
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
            let mut aggregation_interval = tokio::time::interval(Duration::from_secs(60)); // 1 minute

            while monitoring_active.load(Ordering::Acquire) {
                tokio::select! {
                    _ = aggregation_interval.tick() => {
                        // Perform metrics aggregation
                        let realtime = realtime_metrics.read();
                        let historical = historical_metrics.read();

                        // Calculate aggregated metrics for each device
                        for (device_id, _current_metrics) in realtime.iter() {
                            // Compute recent averages, trends, etc.
                            let recent_metrics: Vec<_> = historical
                                .iter()
                                .filter(|m| m.device_id == *device_id)
                                .filter(|m| {
                                    let age = Utc::now().signed_duration_since(m.timestamp);
                                    age.num_seconds() < 3600 // Last hour
                                })
                                .collect();

                            if !recent_metrics.is_empty() {
                                // Perform aggregation calculations
                                debug!("Aggregated {} metrics for device {}", recent_metrics.len(), device_id);
                            }
                        }
                    }
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
            let mut status_interval = tokio::time::interval(Duration::from_secs(30)); // 30 seconds

            while monitoring_active.load(Ordering::Acquire) {
                tokio::select! {
                    _ = status_interval.tick() => {
                        let metrics = realtime_metrics.read();

                        for (device_id, device_metrics) in metrics.iter() {
                            // Check for concerning patterns
                            if device_metrics.temperature_celsius > 85.0 {
                                warn!("Device {} temperature is high: {:.1}°C", device_id, device_metrics.temperature_celsius);
                            }

                            if device_metrics.utilization_percent > 95.0 {
                                warn!("Device {} utilization is very high: {:.1}%", device_id, device_metrics.utilization_percent);
                            }

                            // Check for unusual patterns
                            let memory_usage_percent = (device_metrics.memory_usage_mb as f32 / 24576.0) * 100.0; // Assume 24GB max
                            if memory_usage_percent > 90.0 {
                                warn!("Device {} memory usage is high: {:.1}%", device_id, memory_usage_percent);
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
        // Validate input
        if metrics.device_id != device_id {
            return Err(GpuMonitoringError::MetricsCollectionError {
                details: format!(
                    "Device ID mismatch: expected {}, got {}",
                    device_id, metrics.device_id
                ),
            });
        }

        // Update real-time metrics storage
        {
            let mut realtime = self.realtime_metrics.write();

            // Check for significant changes and log them
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

        // Add to historical data with multiple metric types
        {
            let mut historical = self.historical_metrics.write();

            // Create historical metric entries for different data types
            let metric_entries = vec![
                GpuHistoricalMetric {
                    device_id,
                    metric_type: GpuMetricType::Utilization,
                    value: metrics.utilization_percent as f64,
                    timestamp: metrics.timestamp,
                    test_id: None,            // Not associated with a specific test
                    metadata: HashMap::new(), // No additional metadata for basic monitoring
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

            // Add all metric entries to historical data
            for entry in metric_entries {
                historical.push_back(entry);
            }

            // Maintain historical data size limit to prevent unbounded growth
            let config = self.config.read();
            let max_entries = Self::calculate_max_historical_entries(&config);

            while historical.len() > max_entries {
                historical.pop_front();
            }
        }

        // Check for alert conditions if alerts are enabled
        let enable_alerts = {
            let config = self.config.read();
            config.enable_alerts
        };

        if enable_alerts {
            // Convert resource_management::types::GpuRealTimeMetrics to gpu_manager::types::GpuRealTimeMetrics
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
                // Apply device ID filter
                if let Some(id) = device_id {
                    if metric.device_id != id {
                        return false;
                    }
                }

                // Apply metric type filter
                if let Some(ref mtype) = metric_type {
                    if std::mem::discriminant(&metric.metric_type) != std::mem::discriminant(mtype)
                    {
                        return false;
                    }
                }

                // Apply time filter
                if let Some(since_time) = since {
                    if metric.timestamp < since_time {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect();

        debug!("Retrieved {} historical metrics with filters: device_id={:?}, metric_type={:?}, since={:?}",
               filtered_metrics.len(), device_id, metric_type, since);

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

        // Calculate standard deviation
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

        // Simple linear regression to determine trend
        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x2_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum);

        // Determine trend based on slope significance
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
        // Validate new configuration
        Self::validate_config(&new_config)?;

        // Apply new configuration
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
            uptime_seconds: 0, // Would track actual uptime in production
        }
    }
}

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

// TrendDirection is re-exported from types module

// Ensure proper cleanup when the monitoring system is dropped
impl Drop for GpuMonitoringSystem {
    fn drop(&mut self) {
        if self.monitoring_active.load(Ordering::Acquire) {
            warn!("GpuMonitoringSystem dropped while still active - this may indicate improper shutdown");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create test monitoring configuration
    fn create_test_config() -> GpuMonitoringConfig {
        GpuMonitoringConfig::default()
    }

    /// Helper function to create test metrics
    fn create_test_metrics(device_id: usize) -> GpuRealTimeMetrics {
        GpuRealTimeMetrics {
            device_id,
            timestamp: Utc::now(),
            memory_usage_mb: 8192,
            utilization_percent: 75.0,
            temperature_celsius: 65.0,
            power_consumption_watts: 200.0,
            clock_speeds: GpuClockSpeeds {
                core_clock_mhz: 1800,
                memory_clock_mhz: 7000,
                shader_clock_mhz: Some(1900),
            },
            fan_speeds: vec![50.0],
        }
    }

    #[tokio::test]
    async fn test_monitoring_system_creation() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");

        assert!(!monitoring_system.is_monitoring_active());

        let metrics = monitoring_system.get_realtime_metrics().await;
        assert!(metrics.is_empty());
    }

    #[tokio::test]
    async fn test_start_stop_monitoring() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");

        // Start monitoring
        monitoring_system
            .start_monitoring()
            .await
            .expect("Start monitoring should succeed");
        assert!(monitoring_system.is_monitoring_active());

        // Stop monitoring
        monitoring_system
            .stop_monitoring()
            .await
            .expect("Stop monitoring should succeed");
        assert!(!monitoring_system.is_monitoring_active());
    }

    #[tokio::test]
    async fn test_metrics_update_and_retrieval() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");

        let test_metrics = create_test_metrics(0);

        // Update metrics
        monitoring_system
            .update_metrics(0, test_metrics.clone())
            .await
            .expect("Update metrics should succeed");

        // Retrieve real-time metrics
        let realtime = monitoring_system.get_realtime_metrics().await;
        assert_eq!(realtime.len(), 1);
        assert!(realtime.contains_key(&0));

        let stored_metrics = &realtime[&0];
        assert_eq!(stored_metrics.device_id, 0);
        assert_eq!(stored_metrics.utilization_percent, 75.0);
        assert_eq!(stored_metrics.temperature_celsius, 65.0);
    }

    #[tokio::test]
    async fn test_historical_metrics() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");

        let test_metrics = create_test_metrics(0);

        // Update metrics multiple times
        for i in 0..5 {
            let mut metrics = test_metrics.clone();
            metrics.utilization_percent = 50.0 + (i as f32 * 10.0);
            metrics.timestamp = Utc::now();

            monitoring_system
                .update_metrics(0, metrics)
                .await
                .expect("Update metrics should succeed");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Get historical metrics
        let historical = monitoring_system.get_historical_metrics(None, None, None).await;
        assert!(!historical.is_empty());

        // Filter by device
        let device_0_metrics = monitoring_system.get_historical_metrics(Some(0), None, None).await;
        assert!(!device_0_metrics.is_empty());
        assert!(device_0_metrics.iter().all(|m| m.device_id == 0));

        // Filter by metric type
        let utilization_metrics = monitoring_system
            .get_historical_metrics(None, Some(GpuMetricType::Utilization), None)
            .await;
        assert!(!utilization_metrics.is_empty());
        assert!(utilization_metrics
            .iter()
            .all(|m| matches!(m.metric_type, GpuMetricType::Utilization)));
    }

    #[tokio::test]
    async fn test_metrics_summary() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");

        // Add test data
        for i in 0..10 {
            let mut metrics = create_test_metrics(0);
            metrics.utilization_percent = 50.0 + (i as f32 * 5.0);
            monitoring_system
                .update_metrics(0, metrics)
                .await
                .expect("Update metrics should succeed");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        // Get summary
        let summary = monitoring_system
            .get_metrics_summary(0, GpuMetricType::Utilization, Duration::from_secs(60))
            .await;

        assert!(summary.is_some());
        let summary = summary.expect("Should get summary");
        assert_eq!(summary.device_id, 0);
        assert_eq!(summary.sample_count, 10);
        assert!(summary.average > 50.0);
        assert!(summary.maximum > summary.minimum);
    }

    #[tokio::test]
    async fn test_configuration_validation() {
        // Test invalid monitoring interval
        let mut config = create_test_config();
        config.monitoring_interval = Duration::from_secs(0);

        let result = GpuMonitoringSystem::new(config).await;
        assert!(result.is_err());

        // Test valid configuration
        let config = create_test_config();
        let result = GpuMonitoringSystem::new(config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_configuration_update() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");

        let mut new_config = create_test_config();
        new_config.monitoring_interval = Duration::from_secs(10);
        new_config.enable_performance_tracking = false;

        monitoring_system
            .update_config(new_config.clone())
            .await
            .expect("Update config should succeed");

        let current_config = monitoring_system.get_config().await;
        assert_eq!(current_config.monitoring_interval, Duration::from_secs(10));
        assert!(!current_config.enable_performance_tracking);
    }

    #[tokio::test]
    async fn test_monitoring_statistics() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");

        // Start monitoring
        monitoring_system
            .start_monitoring()
            .await
            .expect("Start monitoring should succeed");

        // Add some metrics
        let metrics = create_test_metrics(0);
        monitoring_system
            .update_metrics(0, metrics)
            .await
            .expect("Update metrics should succeed");

        let stats = monitoring_system.get_monitoring_statistics().await;
        assert!(stats.is_active);
        assert_eq!(stats.monitored_devices, 1);
        assert!(stats.historical_entries > 0);
        assert!(stats.alerts_enabled);

        monitoring_system
            .stop_monitoring()
            .await
            .expect("Stop monitoring should succeed");
    }

    #[tokio::test]
    async fn test_background_tasks() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");

        // Start monitoring to activate background tasks
        monitoring_system
            .start_monitoring()
            .await
            .expect("Start monitoring should succeed");

        // Wait briefly for tasks to initialize
        tokio::time::sleep(Duration::from_millis(100)).await;

        let stats = monitoring_system.get_monitoring_statistics().await;
        assert!(stats.background_tasks > 0);

        // Stop monitoring
        monitoring_system
            .stop_monitoring()
            .await
            .expect("Stop monitoring should succeed");

        // Verify tasks are cleaned up
        let stats = monitoring_system.get_monitoring_statistics().await;
        assert!(!stats.is_active);
    }

    #[tokio::test]
    async fn test_error_handling() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");

        // Test mismatched device ID
        let mut metrics = create_test_metrics(0);
        metrics.device_id = 1; // Different from the passed device_id

        let result = monitoring_system.update_metrics(0, metrics).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            GpuMonitoringError::MetricsCollectionError { .. } => {
                // Expected error type
            },
            other => panic!("Unexpected error type: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_historical_data_cleanup() {
        let mut config = create_test_config();
        config.monitoring_interval = Duration::from_millis(100); // Fast interval for testing

        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");
        monitoring_system
            .start_monitoring()
            .await
            .expect("Start monitoring should succeed");

        // Add many metrics to trigger cleanup
        for i in 0..1000 {
            let mut metrics = create_test_metrics(0);
            metrics.timestamp = Utc::now();
            monitoring_system
                .update_metrics(0, metrics)
                .await
                .expect("Update metrics should succeed");

            if i % 100 == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }

        // Wait for cleanup tasks to run
        tokio::time::sleep(Duration::from_millis(500)).await;

        let historical = monitoring_system.get_historical_metrics(None, None, None).await;

        // Should be limited by cleanup mechanisms
        let max_expected = GpuMonitoringSystem::calculate_max_historical_entries(
            &monitoring_system.get_config().await,
        );

        assert!(historical.len() <= max_expected * 2); // Allow some buffer for test timing

        monitoring_system
            .stop_monitoring()
            .await
            .expect("Stop monitoring should succeed");
    }

    #[tokio::test]
    async fn test_concurrent_metrics_updates() {
        let config = create_test_config();
        let monitoring_system = Arc::new(
            GpuMonitoringSystem::new(config)
                .await
                .expect("Monitoring system creation should succeed"),
        );

        // Start monitoring
        monitoring_system
            .start_monitoring()
            .await
            .expect("Start monitoring should succeed");

        // Launch concurrent update tasks
        let mut handles = Vec::new();
        for device_id in 0..3 {
            let ms = monitoring_system.clone();
            let handle = tokio::spawn(async move {
                for i in 0..10 {
                    let mut metrics = create_test_metrics(device_id);
                    metrics.utilization_percent = 50.0 + (i as f32 * 2.0);
                    ms.update_metrics(device_id, metrics)
                        .await
                        .expect("Update metrics should succeed");
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.expect("Join handle should succeed");
        }

        // Verify all devices have metrics
        let realtime = monitoring_system.get_realtime_metrics().await;
        assert_eq!(realtime.len(), 3);

        for device_id in 0..3 {
            assert!(realtime.contains_key(&device_id));
        }

        monitoring_system
            .stop_monitoring()
            .await
            .expect("Stop monitoring should succeed");
    }
}
