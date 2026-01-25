//! GPU Alert System Implementation
//!
//! This module provides comprehensive GPU alerting capabilities including:
//! - Configurable alert thresholds and conditions for multiple metrics
//! - Multiple alert severity levels with automatic escalation
//! - Alert escalation and notification handling workflows
//! - Alert history and analytics for trend analysis
//! - Integration with external monitoring systems
//! - Concurrent alert operations with thread safety
//! - Comprehensive alert lifecycle management
//!
//! # Features
//!
//! - **Alert Generation**: Creates alerts based on GPU metrics and thresholds
//! - **Alert Escalation**: Handles alert severity levels and escalation rules
//! - **Alert Processing**: Manages alert notification and handling workflows
//! - **Alert History**: Stores and manages alert history and analytics
//! - **Threshold Monitoring**: Checks metrics against configurable thresholds
//! - **Event Handling**: Processes alert events and notifications
//! - **Handler Registration**: Supports custom alert handlers and integrations
//!
//! # Alert Types
//!
//! The system supports various alert types:
//! - Temperature alerts (warning and critical thresholds)
//! - Utilization alerts (high usage detection)
//! - Memory usage alerts (critical memory levels)
//! - Hardware failure alerts (device health issues)
//! - Performance degradation alerts (benchmark regression)
//!
//! # Usage
//!
//! ```rust,no_run
//! use trustformers_serve::resource_management::gpu_manager::alert_system::GpuAlertSystem;
//! use trustformers_serve::resource_management::gpu_manager::types::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize alert system with configuration
//!     let config = GpuAlertConfig::default();
//!     let alert_system = GpuAlertSystem::new(config).await?;
//!
//!     // Start alert processing
//!     alert_system.start().await?;
//!
//!     // Check metrics for alert conditions
//!     let metrics = get_gpu_metrics(0).await;
//!     alert_system.check_metrics_for_alerts(0, &metrics).await?;
//!
//!     // Get active alerts
//!     let active_alerts = alert_system.get_active_alerts().await;
//!
//!     // Acknowledge an alert
//!     if let Some(alert_id) = active_alerts.keys().next() {
//!         alert_system.acknowledge_alert(alert_id).await?;
//!     }
//!
//!     // Stop alert system
//!     alert_system.stop().await?;
//!
//!     Ok(())
//! }
//! ```

use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{task::JoinHandle, time::sleep};
use tracing::{debug, error, info, instrument, warn};

use super::types::*;
use crate::resource_management::types::{GpuAlertEscalation, GpuAlertStatistics};
// FIXME: GpuAlertEscalationStep type not found in test_characterization::types
// use crate::performance_optimizer::test_characterization::types::GpuAlertEscalationStep;

/// GPU alert system error types
#[derive(Debug, thiserror::Error)]
pub enum GpuAlertError {
    #[error("Alert system not running")]
    NotRunning,

    #[error("Alert {alert_id} not found")]
    AlertNotFound { alert_id: String },

    #[error("Invalid alert configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Alert handler error: {handler_name} - {message}")]
    HandlerError {
        handler_name: String,
        message: String,
    },

    #[error("Alert processing error: {message}")]
    ProcessingError { message: String },

    #[error("Alert escalation error: {message}")]
    EscalationError { message: String },
}

/// Result type for alert operations
pub type GpuAlertResult<T> = Result<T, GpuAlertError>;

/// GPU alert system for proactive health monitoring
///
/// Provides comprehensive alerting capabilities including:
/// - Configurable alert thresholds and conditions
/// - Multiple alert severity levels
/// - Alert escalation and notification handling
/// - Alert history and analytics
/// - Integration with external monitoring systems
/// - Thread-safe concurrent alert operations
/// - Comprehensive alert lifecycle management
pub struct GpuAlertSystem {
    /// Alert configuration settings
    config: Arc<RwLock<GpuAlertConfig>>,

    /// Active alerts tracking with unique identifiers
    active_alerts: Arc<RwLock<HashMap<String, GpuAlert>>>,

    /// Alert history for analysis and trend detection
    alert_history: Arc<RwLock<VecDeque<GpuAlertEvent>>>,

    /// Registered alert handlers for different alert types
    alert_handlers: Arc<RwLock<Vec<Box<dyn GpuAlertHandler + Send + Sync>>>>,

    /// Alert processing queue for asynchronous processing
    alert_queue: Arc<Mutex<VecDeque<GpuAlert>>>,

    /// Alert system active flag
    active: Arc<AtomicBool>,

    /// Background task handles for cleanup
    background_tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,

    /// Alert escalation tracking
    escalation_tracking: Arc<RwLock<HashMap<String, GpuAlertEscalation>>>,

    /// Alert statistics for analytics
    alert_statistics: Arc<RwLock<GpuAlertStatistics>>,
}

impl std::fmt::Debug for GpuAlertSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuAlertSystem")
            .field("config", &self.config)
            .field("active_alerts", &self.active_alerts)
            .field("alert_history", &self.alert_history)
            .field(
                "alert_handlers",
                &format!("<{} handlers>", self.alert_handlers.read().len()),
            )
            .field("alert_queue", &self.alert_queue)
            .field("active", &self.active.load(Ordering::SeqCst))
            .field(
                "background_tasks",
                &format!("<{} tasks>", self.background_tasks.lock().len()),
            )
            .field("escalation_tracking", &self.escalation_tracking)
            .field("alert_statistics", &self.alert_statistics)
            .finish()
    }
}

impl GpuAlertSystem {
    /// Create new GPU alert system with configuration
    ///
    /// Initializes the alert system with the specified configuration,
    /// setting up all internal data structures and preparing for operation.
    ///
    /// # Arguments
    ///
    /// * `config` - Alert system configuration parameters
    ///
    /// # Returns
    ///
    /// A configured GPU alert system ready for operation
    ///
    /// # Errors
    ///
    /// Returns error if configuration validation fails
    #[instrument(skip(config))]
    pub async fn new(config: GpuAlertConfig) -> GpuAlertResult<Self> {
        info!("Initializing GPU alert system");

        // Validate configuration
        Self::validate_config(&config)?;

        let alert_system = Self {
            config: Arc::new(RwLock::new(config)),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            alert_handlers: Arc::new(RwLock::new(Vec::new())),
            alert_queue: Arc::new(Mutex::new(VecDeque::new())),
            active: Arc::new(AtomicBool::new(false)),
            background_tasks: Arc::new(Mutex::new(Vec::new())),
            escalation_tracking: Arc::new(RwLock::new(HashMap::new())),
            alert_statistics: Arc::new(RwLock::new(GpuAlertStatistics::default())),
        };

        info!("GPU alert system initialized successfully");
        Ok(alert_system)
    }

    /// Validate alert configuration parameters
    ///
    /// Performs comprehensive validation of alert configuration to ensure
    /// all thresholds and settings are valid and consistent.
    fn validate_config(config: &GpuAlertConfig) -> GpuAlertResult<()> {
        // Validate temperature thresholds
        if config.thresholds.temperature_warning >= config.thresholds.temperature_critical {
            return Err(GpuAlertError::InvalidConfiguration {
                message: "Temperature warning threshold must be less than critical threshold"
                    .to_string(),
            });
        }

        if config.thresholds.temperature_critical > 120.0 {
            return Err(GpuAlertError::InvalidConfiguration {
                message: "Temperature critical threshold cannot exceed 120°C".to_string(),
            });
        }

        // Validate utilization thresholds
        if config.thresholds.utilization_critical_percent < 0.0
            || config.thresholds.utilization_critical_percent > 100.0
        {
            return Err(GpuAlertError::InvalidConfiguration {
                message: "Utilization thresholds must be between 0% and 100%".to_string(),
            });
        }

        // Validate memory thresholds
        if config.thresholds.memory_critical_percent < 0.0
            || config.thresholds.memory_critical_percent > 100.0
        {
            return Err(GpuAlertError::InvalidConfiguration {
                message: "Memory thresholds must be between 0% and 100%".to_string(),
            });
        }

        // Validate escalation settings
        if config.escalation_enabled && config.escalation_delay_seconds == 0 {
            return Err(GpuAlertError::InvalidConfiguration {
                message: "Escalation delay must be greater than 0 when escalation is enabled"
                    .to_string(),
            });
        }

        Ok(())
    }

    /// Start alert system operations
    ///
    /// Initializes and starts all background tasks for alert processing,
    /// escalation handling, and cleanup operations.
    ///
    /// # Returns
    ///
    /// Success if alert system started successfully
    ///
    /// # Errors
    ///
    /// Returns error if alert system is already running or initialization fails
    #[instrument(skip(self))]
    pub async fn start(&self) -> GpuAlertResult<()> {
        if self.active.load(Ordering::Acquire) {
            warn!("GPU alert system is already running");
            return Ok(());
        }

        info!("Starting GPU alert system");

        // Start alert processing task
        self.start_alert_processor().await?;

        // Start alert escalation task
        self.start_escalation_processor().await?;

        // Start cleanup task
        self.start_cleanup_task().await?;

        self.active.store(true, Ordering::Release);
        info!("GPU alert system started successfully");

        Ok(())
    }

    /// Stop alert system operations
    ///
    /// Gracefully stops all background tasks and performs cleanup.
    /// Ensures all active alerts are properly saved to history.
    ///
    /// # Returns
    ///
    /// Success if alert system stopped successfully
    ///
    /// # Errors
    ///
    /// Returns error if alert system is not running
    #[instrument(skip(self))]
    pub async fn stop(&self) -> GpuAlertResult<()> {
        if !self.active.load(Ordering::Acquire) {
            warn!("GPU alert system is not running");
            return Ok(());
        }

        info!("Stopping GPU alert system");

        // Stop background tasks
        let mut tasks = self.background_tasks.lock();
        for task in tasks.drain(..) {
            task.abort();
        }

        // Archive remaining active alerts
        self.archive_active_alerts().await?;

        self.active.store(false, Ordering::Release);
        info!("GPU alert system stopped successfully");

        Ok(())
    }

    /// Start alert processing background task
    ///
    /// Processes alerts from the queue, manages alert lifecycle,
    /// and handles notification to registered handlers.
    async fn start_alert_processor(&self) -> GpuAlertResult<()> {
        let alert_queue = self.alert_queue.clone();
        let alert_handlers = self.alert_handlers.clone();
        let active_alerts = self.active_alerts.clone();
        let alert_history = self.alert_history.clone();
        let alert_statistics = self.alert_statistics.clone();
        let active = self.active.clone();

        let task = tokio::spawn(async move {
            info!("Starting alert processor task");

            while active.load(Ordering::Acquire) {
                // Process alerts from queue
                let alert = {
                    let mut queue = alert_queue.lock();
                    queue.pop_front()
                };

                if let Some(alert) = alert {
                    Self::process_single_alert(
                        alert,
                        &active_alerts,
                        &alert_history,
                        &alert_handlers,
                        &alert_statistics,
                    )
                    .await;
                } else {
                    // No alerts in queue, wait briefly
                    sleep(Duration::from_millis(100)).await;
                }
            }

            debug!("Alert processor task shutting down");
        });

        self.background_tasks.lock().push(task);
        Ok(())
    }

    /// Process a single alert through the complete lifecycle
    async fn process_single_alert(
        alert: GpuAlert,
        active_alerts: &Arc<RwLock<HashMap<String, GpuAlert>>>,
        alert_history: &Arc<RwLock<VecDeque<GpuAlertEvent>>>,
        alert_handlers: &Arc<RwLock<Vec<Box<dyn GpuAlertHandler + Send + Sync>>>>,
        alert_statistics: &Arc<RwLock<GpuAlertStatistics>>,
    ) {
        debug!("Processing alert: {}", alert.alert_id);

        // Check for duplicate alert
        {
            let active = active_alerts.read();
            if active.contains_key(&alert.alert_id) {
                debug!("Duplicate alert ignored: {}", alert.alert_id);
                return;
            }
        }

        // Add to active alerts
        {
            let mut active = active_alerts.write();
            active.insert(alert.alert_id.clone(), alert.clone());
        }

        // Add to history
        {
            let mut history = alert_history.write();
            history.push_back(GpuAlertEvent {
                timestamp: Utc::now(),
                event_type: GpuAlertEventType::Triggered,
                alert: alert.clone(),
                details: HashMap::new(),
            });

            // Maintain history size limit
            if history.len() > 10000 {
                history.pop_front();
            }
        }

        // Update statistics
        {
            let mut stats = alert_statistics.write();
            stats.total_alerts_generated += 1;
            stats
                .alerts_by_type
                .entry(format!("{:?}", alert.alert_type))
                .and_modify(|count| *count += 1)
                .or_insert(1);
            stats
                .alerts_by_severity
                .entry(format!("{:?}", alert.severity))
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }

        // Notify handlers
        {
            let handlers = alert_handlers.read();
            for handler in handlers.iter() {
                if handler.can_handle(&alert.alert_type) {
                    if let Err(e) = handler.handle_alert(&alert) {
                        error!(
                            "Alert handler {} failed for alert {}: {}",
                            handler.name(),
                            alert.alert_id,
                            e
                        );

                        // Update handler failure statistics
                        let mut stats = alert_statistics.write();
                        stats.handler_failures += 1;
                    }
                }
            }
        }

        match alert.severity {
            AlertSeverity::Critical => {
                error!(
                    "CRITICAL GPU Alert [{}]: {} on device {} - {}",
                    alert.alert_id, alert.alert_type, alert.device_id, alert.message
                );
            },
            AlertSeverity::Error => {
                error!(
                    "ERROR GPU Alert [{}]: {} on device {} - {}",
                    alert.alert_id, alert.alert_type, alert.device_id, alert.message
                );
            },
            AlertSeverity::Warning => {
                warn!(
                    "WARNING GPU Alert [{}]: {} on device {} - {}",
                    alert.alert_id, alert.alert_type, alert.device_id, alert.message
                );
            },
            AlertSeverity::Info => {
                info!(
                    "INFO GPU Alert [{}]: {} on device {} - {}",
                    alert.alert_id, alert.alert_type, alert.device_id, alert.message
                );
            },
        }
    }

    /// Start alert escalation processing task
    ///
    /// Monitors alerts for escalation conditions and automatically
    /// escalates alerts based on configured rules and timeouts.
    async fn start_escalation_processor(&self) -> GpuAlertResult<()> {
        let active_alerts = self.active_alerts.clone();
        let escalation_tracking = self.escalation_tracking.clone();
        let config = self.config.clone();
        let active = self.active.clone();

        let task = tokio::spawn(async move {
            info!("Starting alert escalation processor task");

            while active.load(Ordering::Acquire) {
                let escalation_enabled = {
                    let config = config.read();
                    config.escalation_enabled
                };

                if escalation_enabled {
                    Self::process_alert_escalations(&active_alerts, &escalation_tracking, &config)
                        .await;
                }

                // Check for escalations every 30 seconds
                sleep(Duration::from_secs(30)).await;
            }

            debug!("Alert escalation processor task shutting down");
        });

        self.background_tasks.lock().push(task);
        Ok(())
    }

    /// Process alert escalations
    async fn process_alert_escalations(
        active_alerts: &Arc<RwLock<HashMap<String, GpuAlert>>>,
        escalation_tracking: &Arc<RwLock<HashMap<String, GpuAlertEscalation>>>,
        config: &Arc<RwLock<GpuAlertConfig>>,
    ) {
        let config = config.read();
        let escalation_delay = Duration::from_secs(config.escalation_delay_seconds);
        // TODO: GpuAlertConfig no longer has max_escalation_level field
        // Using escalation_rules length as max level, or default to 3
        let max_escalations = config.escalation_rules.len().max(3);

        let active = active_alerts.read();
        let mut escalations = escalation_tracking.write();

        for (alert_id, alert) in active.iter() {
            if alert.acknowledged || alert.severity == AlertSeverity::Critical {
                continue; // Skip acknowledged or already critical alerts
            }

            let escalation = escalations.entry(alert_id.clone()).or_insert_with(|| {
                // Convert gpu_manager::types::AlertSeverity to resource_management::types::AlertSeverity
                let severity_str = format!("{:?}", alert.severity);
                let initial_severity = match severity_str.as_str() {
                    "Info" => crate::resource_management::types::AlertSeverity::Info,
                    "Warning" => crate::resource_management::types::AlertSeverity::Warning,
                    "Error" => crate::resource_management::types::AlertSeverity::Error,
                    "Critical" => crate::resource_management::types::AlertSeverity::Critical,
                    _ => crate::resource_management::types::AlertSeverity::Warning,
                };
                GpuAlertEscalation {
                    escalation_level: 0,
                    notification_channels: Vec::new(),
                    escalation_delay,
                    alert_id: alert_id.clone(),
                    initial_severity,
                    current_level: 0,
                    escalated_at: Utc::now(),
                    escalation_history: Vec::new(),
                }
            });

            let time_since_alert = Utc::now().signed_duration_since(alert.timestamp);

            if time_since_alert.to_std().unwrap_or_default() >= escalation_delay
                && (escalation.current_level as usize) < max_escalations
            {
                escalation.current_level += 1;
                escalation.escalated_at = Utc::now();
                // FIXME: GpuAlertEscalationStep type not defined
                // escalation.escalation_history.push(GpuAlertEscalationStep {
                //     level: escalation.current_level,
                //     timestamp: Utc::now(),
                //     reason: "Timeout escalation".to_string(),
                // });

                warn!(
                    "Alert {} escalated to level {}",
                    alert_id, escalation.current_level
                );
            }
        }
    }

    /// Start cleanup task for managing alert lifecycle
    ///
    /// Periodically cleans up old alerts, manages memory usage,
    /// and performs maintenance operations.
    async fn start_cleanup_task(&self) -> GpuAlertResult<()> {
        let active_alerts = self.active_alerts.clone();
        let alert_history = self.alert_history.clone();
        let escalation_tracking = self.escalation_tracking.clone();
        let config = self.config.clone();
        let active = self.active.clone();

        let task = tokio::spawn(async move {
            info!("Starting alert cleanup task");

            while active.load(Ordering::Acquire) {
                Self::perform_cleanup(
                    &active_alerts,
                    &alert_history,
                    &escalation_tracking,
                    &config,
                )
                .await;

                // Run cleanup every 10 minutes
                sleep(Duration::from_secs(600)).await;
            }

            debug!("Alert cleanup task shutting down");
        });

        self.background_tasks.lock().push(task);
        Ok(())
    }

    /// Perform cleanup operations
    async fn perform_cleanup(
        active_alerts: &Arc<RwLock<HashMap<String, GpuAlert>>>,
        alert_history: &Arc<RwLock<VecDeque<GpuAlertEvent>>>,
        escalation_tracking: &Arc<RwLock<HashMap<String, GpuAlertEscalation>>>,
        _config: &Arc<RwLock<GpuAlertConfig>>,
    ) {
        // TODO: GpuAlertConfig no longer has alert_retention_hours field
        // Using default of 24 hours
        let retention_hours = 24;

        let cutoff_time = Utc::now() - chrono::Duration::hours(retention_hours as i64);

        // Clean up old active alerts
        {
            let mut active = active_alerts.write();
            active.retain(|_, alert| alert.timestamp > cutoff_time || !alert.acknowledged);
        }

        // Clean up old escalation tracking
        {
            let mut escalations = escalation_tracking.write();
            escalations.retain(|alert_id, _| {
                let active = active_alerts.read();
                active.contains_key(alert_id)
            });
        }

        // Trim alert history if too large
        {
            let mut history = alert_history.write();
            while history.len() > 5000 {
                history.pop_front();
            }
        }

        debug!("Alert cleanup completed");
    }

    /// Archive active alerts to history
    async fn archive_active_alerts(&self) -> GpuAlertResult<()> {
        let mut active_alerts = self.active_alerts.write();
        let mut alert_history = self.alert_history.write();

        for (_, alert) in active_alerts.drain() {
            alert_history.push_back(GpuAlertEvent {
                timestamp: Utc::now(),
                event_type: GpuAlertEventType::Archived,
                alert,
                details: HashMap::new(),
            });
        }

        info!("Archived {} active alerts to history", active_alerts.len());
        Ok(())
    }

    /// Check metrics for alert conditions
    ///
    /// Analyzes real-time GPU metrics against configured thresholds
    /// and generates appropriate alerts when conditions are met.
    ///
    /// # Arguments
    ///
    /// * `device_id` - ID of the GPU device being monitored
    /// * `metrics` - Current real-time metrics for the device
    ///
    /// # Returns
    ///
    /// Success if metrics were processed successfully
    ///
    /// # Errors
    ///
    /// Returns error if alert generation or processing fails
    #[instrument(skip(self, metrics))]
    pub async fn check_metrics_for_alerts(
        &self,
        device_id: usize,
        metrics: &GpuRealTimeMetrics,
    ) -> GpuAlertResult<()> {
        if !self.active.load(Ordering::Acquire) {
            return Err(GpuAlertError::NotRunning);
        }

        let config = self.config.read();
        let mut alerts_to_trigger = Vec::new();

        // Check temperature alerts
        if config.enable_temperature_alerts {
            if metrics.temperature_celsius >= config.thresholds.temperature_critical {
                alerts_to_trigger.push(self.create_alert(
                    device_id,
                    GpuAlertType::HighTemperature,
                    AlertSeverity::Critical,
                    format!(
                        "Critical temperature: {:.1}°C (threshold: {:.1}°C)",
                        metrics.temperature_celsius, config.thresholds.temperature_critical
                    ),
                    metrics.temperature_celsius as f64,
                    config.thresholds.temperature_critical as f64,
                ));
            } else if metrics.temperature_celsius >= config.thresholds.temperature_warning {
                alerts_to_trigger.push(self.create_alert(
                    device_id,
                    GpuAlertType::HighTemperature,
                    AlertSeverity::Warning,
                    format!(
                        "High temperature: {:.1}°C (threshold: {:.1}°C)",
                        metrics.temperature_celsius, config.thresholds.temperature_warning
                    ),
                    metrics.temperature_celsius as f64,
                    config.thresholds.temperature_warning as f64,
                ));
            }
        }

        // Check utilization alerts
        if config.enable_utilization_alerts {
            if metrics.utilization_percent >= config.thresholds.utilization_critical_percent {
                alerts_to_trigger.push(self.create_alert(
                    device_id,
                    GpuAlertType::HighUtilization,
                    AlertSeverity::Critical,
                    format!(
                        "Critical utilization: {:.1}% (threshold: {:.1}%)",
                        metrics.utilization_percent, config.thresholds.utilization_critical_percent
                    ),
                    metrics.utilization_percent as f64,
                    config.thresholds.utilization_critical_percent as f64,
                ));
            } else if metrics.utilization_percent >= config.thresholds.utilization_warning_percent {
                alerts_to_trigger.push(self.create_alert(
                    device_id,
                    GpuAlertType::HighUtilization,
                    AlertSeverity::Warning,
                    format!(
                        "High utilization: {:.1}% (threshold: {:.1}%)",
                        metrics.utilization_percent, config.thresholds.utilization_warning_percent
                    ),
                    metrics.utilization_percent as f64,
                    config.thresholds.utilization_warning_percent as f64,
                ));
            }
        }

        // Check memory usage alerts
        if config.enable_memory_alerts {
            // Assume device has known total memory (would be passed from device info)
            let memory_percent = (metrics.memory_usage_mb as f32 / 24576.0) * 100.0; // Default to 24GB

            if memory_percent >= config.thresholds.memory_critical_percent {
                alerts_to_trigger.push(self.create_alert(
                    device_id,
                    GpuAlertType::HighMemoryUsage,
                    AlertSeverity::Critical,
                    format!(
                        "Critical memory usage: {:.1}% ({} MB) (threshold: {:.1}%)",
                        memory_percent,
                        metrics.memory_usage_mb,
                        config.thresholds.memory_critical_percent
                    ),
                    memory_percent as f64,
                    config.thresholds.memory_critical_percent as f64,
                ));
            } else if memory_percent >= config.thresholds.memory_warning_percent {
                alerts_to_trigger.push(self.create_alert(
                    device_id,
                    GpuAlertType::HighMemoryUsage,
                    AlertSeverity::Warning,
                    format!(
                        "High memory usage: {:.1}% ({} MB) (threshold: {:.1}%)",
                        memory_percent,
                        metrics.memory_usage_mb,
                        config.thresholds.memory_warning_percent
                    ),
                    memory_percent as f64,
                    config.thresholds.memory_warning_percent as f64,
                ));
            }
        }

        // Check power consumption alerts
        // TODO: GpuAlertConfig no longer has enable_power_alerts field
        // Checking power consumption directly based on thresholds
        if metrics.power_consumption_watts >= config.thresholds.power_critical_watts {
            alerts_to_trigger.push(self.create_alert(
                device_id,
                GpuAlertType::HighPowerConsumption,
                AlertSeverity::Critical,
                format!(
                    "Critical power consumption: {:.1}W (threshold: {:.1}W)",
                    metrics.power_consumption_watts, config.thresholds.power_critical_watts
                ),
                metrics.power_consumption_watts as f64,
                config.thresholds.power_critical_watts as f64,
            ));
        }

        // Queue alerts for processing
        if !alerts_to_trigger.is_empty() {
            let mut queue = self.alert_queue.lock();
            for alert in alerts_to_trigger {
                debug!("Queuing alert: {} for device {}", alert.alert_id, device_id);
                queue.push_back(alert);
            }
        }

        Ok(())
    }

    /// Create an alert with comprehensive information
    ///
    /// Generates a new alert with all necessary metadata including
    /// unique identification, timestamps, and threshold information.
    fn create_alert(
        &self,
        device_id: usize,
        alert_type: GpuAlertType,
        severity: AlertSeverity,
        message: String,
        current_value: f64,
        threshold_value: f64,
    ) -> GpuAlert {
        let alert_id = format!(
            "alert_{}_{}_{}",
            device_id,
            alert_type.to_string().to_lowercase(),
            Utc::now().timestamp_millis()
        );

        GpuAlert {
            alert_id,
            device_id,
            alert_type,
            severity,
            message,
            current_value,
            threshold_value,
            timestamp: Utc::now(),
            acknowledged: false,
        }
    }

    /// Get active alerts
    ///
    /// Returns a copy of all currently active alerts in the system.
    ///
    /// # Returns
    ///
    /// HashMap of alert IDs to alert objects
    pub async fn get_active_alerts(&self) -> HashMap<String, GpuAlert> {
        let alerts = self.active_alerts.read();
        alerts.clone()
    }

    /// Get active alerts for a specific device
    ///
    /// Returns alerts filtered by device ID.
    ///
    /// # Arguments
    ///
    /// * `device_id` - ID of the device to filter alerts for
    ///
    /// # Returns
    ///
    /// Vector of alerts for the specified device
    pub async fn get_device_alerts(&self, device_id: usize) -> Vec<GpuAlert> {
        let alerts = self.active_alerts.read();
        alerts.values().filter(|alert| alert.device_id == device_id).cloned().collect()
    }

    /// Get alert history
    ///
    /// Returns the complete alert history with optional filtering.
    ///
    /// # Arguments
    ///
    /// * `limit` - Optional limit on number of events to return
    /// * `device_id` - Optional device ID filter
    ///
    /// # Returns
    ///
    /// Vector of alert events
    pub async fn get_alert_history(
        &self,
        limit: Option<usize>,
        device_id: Option<usize>,
    ) -> Vec<GpuAlertEvent> {
        let history = self.alert_history.read();
        let mut events: Vec<_> = history
            .iter()
            .filter(|event| device_id.map_or(true, |id| event.alert.device_id == id))
            .cloned()
            .collect();

        // Reverse to get most recent first
        events.reverse();

        if let Some(limit) = limit {
            events.truncate(limit);
        }

        events
    }

    /// Acknowledge an alert
    ///
    /// Marks an alert as acknowledged, preventing further escalation
    /// and adding an acknowledgment event to the history.
    ///
    /// # Arguments
    ///
    /// * `alert_id` - ID of the alert to acknowledge
    ///
    /// # Returns
    ///
    /// Success if alert was acknowledged
    ///
    /// # Errors
    ///
    /// Returns error if alert is not found
    #[instrument(skip(self))]
    pub async fn acknowledge_alert(&self, alert_id: &str) -> GpuAlertResult<()> {
        let mut active_alerts = self.active_alerts.write();

        if let Some(alert) = active_alerts.get_mut(alert_id) {
            if alert.acknowledged {
                debug!("Alert {} is already acknowledged", alert_id);
                return Ok(());
            }

            alert.acknowledged = true;

            // Add acknowledgment to history
            let mut history = self.alert_history.write();
            history.push_back(GpuAlertEvent {
                timestamp: Utc::now(),
                event_type: GpuAlertEventType::Acknowledged,
                alert: alert.clone(),
                details: HashMap::new(),
            });

            // Update statistics
            let mut stats = self.alert_statistics.write();
            stats.total_alerts_acknowledged += 1;

            info!("Alert {} acknowledged", alert_id);
            Ok(())
        } else {
            Err(GpuAlertError::AlertNotFound {
                alert_id: alert_id.to_string(),
            })
        }
    }

    /// Acknowledge all alerts for a device
    ///
    /// Bulk acknowledges all active alerts for a specific device.
    ///
    /// # Arguments
    ///
    /// * `device_id` - ID of the device whose alerts to acknowledge
    ///
    /// # Returns
    ///
    /// Number of alerts acknowledged
    #[instrument(skip(self))]
    pub async fn acknowledge_device_alerts(&self, device_id: usize) -> GpuAlertResult<usize> {
        let alert_ids: Vec<String> = {
            let alerts = self.active_alerts.read();
            alerts
                .iter()
                .filter(|(_, alert)| alert.device_id == device_id && !alert.acknowledged)
                .map(|(id, _)| id.clone())
                .collect()
        };

        let mut acknowledged_count = 0;
        for alert_id in alert_ids {
            if self.acknowledge_alert(&alert_id).await.is_ok() {
                acknowledged_count += 1;
            }
        }

        info!(
            "Acknowledged {} alerts for device {}",
            acknowledged_count, device_id
        );
        Ok(acknowledged_count)
    }

    /// Register alert handler
    ///
    /// Registers a custom alert handler to receive notifications
    /// for specific types of alerts.
    ///
    /// # Arguments
    ///
    /// * `handler` - Alert handler implementation
    pub async fn register_handler(&self, handler: Box<dyn GpuAlertHandler + Send + Sync>) {
        let mut handlers = self.alert_handlers.write();
        info!("Registering alert handler: {}", handler.name());
        handlers.push(handler);
    }

    /// Unregister alert handler
    ///
    /// Removes a previously registered alert handler.
    ///
    /// # Arguments
    ///
    /// * `handler_name` - Name of the handler to remove
    ///
    /// # Returns
    ///
    /// Success if handler was found and removed
    pub async fn unregister_handler(&self, handler_name: &str) -> GpuAlertResult<()> {
        let mut handlers = self.alert_handlers.write();
        let initial_len = handlers.len();

        handlers.retain(|handler| handler.name() != handler_name);

        if handlers.len() < initial_len {
            info!("Unregistered alert handler: {}", handler_name);
            Ok(())
        } else {
            Err(GpuAlertError::HandlerError {
                handler_name: handler_name.to_string(),
                message: "Handler not found".to_string(),
            })
        }
    }

    /// Update alert configuration
    ///
    /// Updates the alert system configuration at runtime.
    ///
    /// # Arguments
    ///
    /// * `new_config` - New configuration to apply
    ///
    /// # Returns
    ///
    /// Success if configuration was updated
    ///
    /// # Errors
    ///
    /// Returns error if configuration validation fails
    #[instrument(skip(self, new_config))]
    pub async fn update_configuration(&self, new_config: GpuAlertConfig) -> GpuAlertResult<()> {
        // Validate new configuration
        Self::validate_config(&new_config)?;

        // Update configuration
        {
            let mut config = self.config.write();
            *config = new_config;
        }

        info!("Alert system configuration updated");
        Ok(())
    }

    /// Get alert statistics
    ///
    /// Returns comprehensive statistics about alert system operation.
    ///
    /// # Returns
    ///
    /// Alert statistics including counts, types, and performance metrics
    pub async fn get_alert_statistics(&self) -> GpuAlertStatistics {
        let stats = self.alert_statistics.read();
        stats.clone()
    }

    /// Clear alert history
    ///
    /// Removes all entries from alert history. Use with caution.
    ///
    /// # Returns
    ///
    /// Number of history entries cleared
    #[instrument(skip(self))]
    pub async fn clear_alert_history(&self) -> usize {
        let mut history = self.alert_history.write();
        let count = history.len();
        history.clear();

        warn!("Cleared {} alert history entries", count);
        count
    }

    /// Get alert escalation status
    ///
    /// Returns escalation information for all tracked alerts.
    ///
    /// # Returns
    ///
    /// HashMap of alert IDs to escalation information
    pub async fn get_escalation_status(&self) -> HashMap<String, GpuAlertEscalation> {
        let escalations = self.escalation_tracking.read();
        escalations.clone()
    }

    /// Force escalate an alert
    ///
    /// Manually escalates an alert to the next level.
    ///
    /// # Arguments
    ///
    /// * `alert_id` - ID of the alert to escalate
    /// * `reason` - Reason for manual escalation
    ///
    /// # Returns
    ///
    /// Success if alert was escalated
    ///
    /// # Errors
    ///
    /// Returns error if alert is not found or cannot be escalated
    #[instrument(skip(self))]
    pub async fn force_escalate_alert(&self, alert_id: &str, reason: String) -> GpuAlertResult<()> {
        let mut escalations = self.escalation_tracking.write();
        let active_alerts = self.active_alerts.read();

        if !active_alerts.contains_key(alert_id) {
            return Err(GpuAlertError::AlertNotFound {
                alert_id: alert_id.to_string(),
            });
        }

        let escalation_delay = {
            let config = self.config.read();
            Duration::from_secs(config.escalation_delay_seconds)
        };

        let escalation = escalations.entry(alert_id.to_string()).or_insert_with(|| {
            let alert = &active_alerts[alert_id];
            // Convert gpu_manager::types::AlertSeverity to resource_management::types::AlertSeverity
            let severity_str = format!("{:?}", alert.severity);
            let initial_severity = match severity_str.as_str() {
                "Info" => crate::resource_management::types::AlertSeverity::Info,
                "Warning" => crate::resource_management::types::AlertSeverity::Warning,
                "Error" => crate::resource_management::types::AlertSeverity::Error,
                "Critical" => crate::resource_management::types::AlertSeverity::Critical,
                _ => crate::resource_management::types::AlertSeverity::Warning,
            };
            GpuAlertEscalation {
                escalation_level: 0,
                notification_channels: Vec::new(),
                escalation_delay,
                alert_id: alert_id.to_string(),
                initial_severity,
                current_level: 0,
                escalated_at: Utc::now(),
                escalation_history: Vec::new(),
            }
        });

        // TODO: GpuAlertConfig no longer has max_escalation_level field
        // Using escalation_rules length as max level, or default to 3
        let max_level = {
            let config = self.config.read();
            config.escalation_rules.len().max(3)
        };

        if (escalation.current_level as usize) >= max_level {
            return Err(GpuAlertError::EscalationError {
                message: format!("Alert {} is already at maximum escalation level", alert_id),
            });
        }

        escalation.current_level += 1;
        escalation.escalated_at = Utc::now();
        // FIXME: GpuAlertEscalationStep type not defined
        // escalation.escalation_history.push(GpuAlertEscalationStep {
        //     level: escalation.current_level,
        //     timestamp: Utc::now(),
        //     reason,
        // });

        warn!(
            "Alert {} manually escalated to level {}",
            alert_id, escalation.current_level
        );
        Ok(())
    }

    /// Check if system is running
    ///
    /// Returns whether the alert system is currently active.
    ///
    /// # Returns
    ///
    /// True if alert system is running, false otherwise
    pub fn is_running(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    /// Get alert count by severity
    ///
    /// Returns count of active alerts grouped by severity level.
    ///
    /// # Returns
    ///
    /// HashMap of severity levels to alert counts
    pub async fn get_alert_counts_by_severity(&self) -> HashMap<AlertSeverity, usize> {
        let alerts = self.active_alerts.read();
        let mut counts = HashMap::new();

        for alert in alerts.values() {
            *counts.entry(alert.severity).or_insert(0) += 1;
        }

        counts
    }
}

impl Drop for GpuAlertSystem {
    fn drop(&mut self) {
        // Ensure clean shutdown when the alert system is dropped
        if self.active.load(Ordering::Acquire) {
            // Mark as inactive to stop background tasks
            self.active.store(false, Ordering::Release);

            // Abort all background tasks
            let tasks = self.background_tasks.lock();
            for task in tasks.iter() {
                task.abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create test configuration
    fn create_test_alert_config() -> GpuAlertConfig {
        GpuAlertConfig::default()
    }

    /// Helper function to create test metrics
    fn create_test_metrics(
        device_id: usize,
        temp: f32,
        util: f32,
        memory_mb: u64,
    ) -> GpuRealTimeMetrics {
        GpuRealTimeMetrics {
            device_id,
            timestamp: Utc::now(),
            memory_usage_mb: memory_mb,
            utilization_percent: util,
            temperature_celsius: temp,
            power_consumption_watts: 250.0,
            clock_speeds: GpuClockSpeeds {
                core_clock_mhz: 1800,
                memory_clock_mhz: 7000,
                shader_clock_mhz: Some(1900),
            },
            fan_speeds: vec![50.0],
        }
    }

    #[tokio::test]
    async fn test_alert_system_creation() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");

        assert!(!alert_system.is_running());

        let alerts = alert_system.get_active_alerts().await;
        assert!(alerts.is_empty());
    }

    #[tokio::test]
    async fn test_alert_system_start_stop() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");

        // Start system
        alert_system.start().await.expect("Alert system start should succeed");
        assert!(alert_system.is_running());

        // Stop system
        alert_system.stop().await.expect("Alert system stop should succeed");
        assert!(!alert_system.is_running());
    }

    #[tokio::test]
    async fn test_temperature_alert_generation() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");

        // Create metrics that should trigger temperature alert
        let high_temp_metrics = create_test_metrics(0, 90.0, 50.0, 8192);

        // Check metrics for alerts
        alert_system
            .check_metrics_for_alerts(0, &high_temp_metrics)
            .await
            .expect("Check metrics should succeed");

        // Wait for alert processing
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Verify alert was generated
        let alerts = alert_system.get_active_alerts().await;
        assert!(!alerts.is_empty());

        let alert = alerts.values().next().expect("Should have alert");
        assert_eq!(alert.device_id, 0);
        assert_eq!(alert.alert_type, GpuAlertType::HighTemperature);
        assert_eq!(alert.severity, AlertSeverity::Critical);

        alert_system.stop().await.expect("Alert system stop should succeed");
    }

    #[tokio::test]
    async fn test_utilization_alert_generation() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");

        // Create metrics that should trigger utilization alert
        let high_util_metrics = create_test_metrics(0, 60.0, 98.0, 8192);

        alert_system
            .check_metrics_for_alerts(0, &high_util_metrics)
            .await
            .expect("Check metrics should succeed");

        // Wait for processing
        tokio::time::sleep(Duration::from_millis(200)).await;

        let alerts = alert_system.get_active_alerts().await;
        assert!(!alerts.is_empty());

        let alert = alerts.values().next().expect("Should have alert");
        assert_eq!(alert.alert_type, GpuAlertType::HighUtilization);

        alert_system.stop().await.expect("Alert system stop should succeed");
    }

    #[tokio::test]
    async fn test_alert_acknowledgment() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");

        // Generate an alert
        let metrics = create_test_metrics(0, 90.0, 50.0, 8192);
        alert_system
            .check_metrics_for_alerts(0, &metrics)
            .await
            .expect("Check metrics should succeed");

        // Wait for processing
        tokio::time::sleep(Duration::from_millis(200)).await;

        let alerts = alert_system.get_active_alerts().await;
        assert!(!alerts.is_empty());

        let alert_id = alerts.keys().next().expect("Should have alert").clone();
        let alert = &alerts[&alert_id];
        assert!(!alert.acknowledged);

        // Acknowledge the alert
        alert_system
            .acknowledge_alert(&alert_id)
            .await
            .expect("Acknowledge alert should succeed");

        // Verify acknowledgment
        let alerts = alert_system.get_active_alerts().await;
        let alert = &alerts[&alert_id];
        assert!(alert.acknowledged);

        alert_system.stop().await.expect("Alert system stop should succeed");
    }

    #[tokio::test]
    async fn test_alert_history() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");

        // Generate alerts
        let metrics = create_test_metrics(0, 90.0, 50.0, 8192);
        alert_system
            .check_metrics_for_alerts(0, &metrics)
            .await
            .expect("Check metrics should succeed");

        // Wait for processing
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Check history
        let history = alert_system.get_alert_history(None, None).await;
        assert!(!history.is_empty());

        let event = &history[0];
        assert_eq!(event.event_type, GpuAlertEventType::Triggered);
        assert_eq!(event.alert.device_id, 0);

        alert_system.stop().await.expect("Alert system stop should succeed");
    }

    #[tokio::test]
    async fn test_device_specific_alerts() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");

        // Generate alerts for different devices
        let metrics_0 = create_test_metrics(0, 90.0, 50.0, 8192);
        let metrics_1 = create_test_metrics(1, 70.0, 98.0, 8192);

        alert_system
            .check_metrics_for_alerts(0, &metrics_0)
            .await
            .expect("Check metrics should succeed");
        alert_system
            .check_metrics_for_alerts(1, &metrics_1)
            .await
            .expect("Check metrics should succeed");

        // Wait for processing
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Check device-specific alerts
        let device_0_alerts = alert_system.get_device_alerts(0).await;
        let device_1_alerts = alert_system.get_device_alerts(1).await;

        assert!(!device_0_alerts.is_empty());
        assert!(!device_1_alerts.is_empty());

        assert_eq!(device_0_alerts[0].alert_type, GpuAlertType::HighTemperature);
        assert_eq!(device_1_alerts[0].alert_type, GpuAlertType::HighUtilization);

        alert_system.stop().await.expect("Alert system stop should succeed");
    }

    #[tokio::test]
    async fn test_alert_statistics() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");

        // Generate multiple alerts
        let metrics = create_test_metrics(0, 90.0, 98.0, 20000);
        alert_system
            .check_metrics_for_alerts(0, &metrics)
            .await
            .expect("Check metrics should succeed");

        // Wait for processing
        tokio::time::sleep(Duration::from_millis(200)).await;

        let stats = alert_system.get_alert_statistics().await;
        assert!(stats.total_alerts_generated > 0);

        alert_system.stop().await.expect("Alert system stop should succeed");
    }

    #[tokio::test]
    async fn test_configuration_validation() {
        let mut config = create_test_alert_config();

        // Test invalid temperature thresholds
        config.thresholds.temperature_warning = 90.0;
        config.thresholds.temperature_critical = 80.0; // Should be higher than warning

        let result = GpuAlertSystem::new(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_bulk_acknowledgment() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");

        // Generate multiple alerts for same device
        let metrics = create_test_metrics(0, 90.0, 98.0, 20000);
        alert_system
            .check_metrics_for_alerts(0, &metrics)
            .await
            .expect("Check metrics should succeed");

        // Wait for processing
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Acknowledge all alerts for device
        let ack_count = alert_system
            .acknowledge_device_alerts(0)
            .await
            .expect("Acknowledge device alerts should succeed");
        assert!(ack_count > 0);

        // Verify all device alerts are acknowledged
        let device_alerts = alert_system.get_device_alerts(0).await;
        for alert in device_alerts {
            assert!(alert.acknowledged);
        }

        alert_system.stop().await.expect("Alert system stop should succeed");
    }
}
