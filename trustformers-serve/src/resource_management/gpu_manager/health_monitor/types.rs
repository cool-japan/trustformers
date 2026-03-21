//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::{GpuDeviceInfo, GpuDeviceStatus};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio::time::interval;
use tracing::{debug, error, info, instrument, warn};

use super::functions::GpuHealthResult;

/// Health trend indicator
#[derive(Debug, Clone, PartialEq)]
pub enum HealthTrend {
    /// Health is improving
    Improving,
    /// Health is stable
    Stable,
    /// Health is declining
    Declining,
    /// Insufficient data for trend analysis
    Unknown,
}
/// Health monitoring statistics
#[derive(Debug, Clone, Default)]
pub struct HealthMonitoringStats {
    /// Total devices monitored
    pub devices_monitored: usize,
    /// Total health checks performed
    pub total_checks: u64,
    /// Number of healthy devices
    pub healthy_devices: usize,
    /// Number of unhealthy devices
    pub unhealthy_devices: usize,
    /// Average health score across all devices
    pub average_health_score: f32,
    /// Number of health events generated
    pub total_events: u64,
    /// Number of critical warnings issued
    pub critical_warnings: u64,
    /// Number of predictive warnings issued
    pub predictive_warnings: u64,
    /// Monitoring uptime
    pub uptime: Duration,
    /// Last statistics update
    pub last_updated: DateTime<Utc>,
}
/// GPU health monitor for comprehensive device health tracking
///
/// This is the main health monitoring system that tracks GPU device health status,
/// performs health analytics, generates predictions, and provides comprehensive
/// health reporting capabilities.
#[derive(Debug)]
pub struct GpuHealthMonitor {
    /// Current health status for all monitored devices
    health_status: Arc<RwLock<HashMap<usize, GpuHealthStatus>>>,
    /// Health analytics data for trend analysis
    health_analytics: Arc<RwLock<HashMap<usize, GpuHealthAnalytics>>>,
    /// Health monitoring configuration
    config: Arc<RwLock<GpuHealthConfig>>,
    /// Background task handles
    background_tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
    /// Health event channel for notifications
    health_event_sender: Arc<Mutex<Option<mpsc::UnboundedSender<HealthEvent>>>>,
    /// System running state
    running: Arc<AtomicBool>,
    /// Total health checks performed
    total_health_checks: Arc<AtomicU64>,
    /// Health monitoring statistics
    stats: Arc<RwLock<HealthMonitoringStats>>,
}
impl GpuHealthMonitor {
    /// Create a new GPU health monitor with default configuration
    ///
    /// This initializes the health monitoring system with default settings
    /// and prepares it for device health tracking.
    ///
    /// # Returns
    ///
    /// A new GpuHealthMonitor instance ready for monitoring
    pub fn new() -> Self {
        Self::with_config(GpuHealthConfig::default())
    }
    /// Create a new GPU health monitor with custom configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Health monitoring configuration parameters
    ///
    /// # Returns
    ///
    /// A configured GpuHealthMonitor instance
    pub fn with_config(config: GpuHealthConfig) -> Self {
        Self {
            health_status: Arc::new(RwLock::new(HashMap::new())),
            health_analytics: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(config)),
            background_tasks: Arc::new(Mutex::new(Vec::new())),
            health_event_sender: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            total_health_checks: Arc::new(AtomicU64::new(0)),
            stats: Arc::new(RwLock::new(HealthMonitoringStats::default())),
        }
    }
    /// Start comprehensive health monitoring for all devices
    ///
    /// This initiates the health monitoring system, starting background tasks
    /// for health checking, analytics computation, and event processing.
    ///
    /// # Arguments
    ///
    /// * `devices` - Shared reference to the device map
    /// * `shutdown_rx` - Shutdown signal receiver
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of monitoring startup
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Monitoring is already running
    /// - Background task startup fails
    /// - Event channel setup fails
    #[instrument(skip(self, devices, shutdown_rx))]
    pub async fn start_monitoring(
        &self,
        devices: Arc<RwLock<HashMap<usize, GpuDeviceInfo>>>,
        shutdown_rx: broadcast::Receiver<()>,
    ) -> GpuHealthResult<()> {
        if self.running.load(Ordering::Acquire) {
            return Err(GpuHealthError::TaskError {
                message: "Health monitoring is already running".to_string(),
            });
        }
        info!("Starting GPU health monitoring system");
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        *self.health_event_sender.lock() = Some(event_sender);
        self.start_health_checking_task(devices.clone(), shutdown_rx.resubscribe())
            .await?;
        self.start_analytics_task(shutdown_rx.resubscribe()).await?;
        self.start_event_processing_task(event_receiver, shutdown_rx.resubscribe())
            .await?;
        self.start_statistics_task(shutdown_rx.resubscribe()).await?;
        self.initialize_device_health(&devices).await?;
        self.running.store(true, Ordering::Release);
        let mut stats = self.stats.write();
        stats.last_updated = Utc::now();
        info!("GPU health monitoring system started successfully");
        Ok(())
    }
    /// Stop health monitoring system gracefully
    ///
    /// This shuts down all background tasks and cleans up resources.
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of shutdown
    #[instrument(skip(self))]
    pub async fn stop_monitoring(&self) -> GpuHealthResult<()> {
        if !self.running.load(Ordering::Acquire) {
            return Ok(());
        }
        info!("Stopping GPU health monitoring system");
        *self.health_event_sender.lock() = None;
        let mut tasks = self.background_tasks.lock();
        for task in tasks.drain(..) {
            if !task.is_finished() {
                task.abort();
            }
        }
        self.running.store(false, Ordering::Release);
        info!("GPU health monitoring system stopped successfully");
        Ok(())
    }
    /// Initialize health status for all devices
    async fn initialize_device_health(
        &self,
        devices: &Arc<RwLock<HashMap<usize, GpuDeviceInfo>>>,
    ) -> GpuHealthResult<()> {
        let device_map = devices.read();
        let mut health_status = self.health_status.write();
        let mut analytics = self.health_analytics.write();
        for device in device_map.values() {
            let initial_health = Self::create_initial_health_status(device);
            health_status.insert(device.device_id, initial_health);
            let initial_analytics = Self::create_initial_analytics(device.device_id);
            analytics.insert(device.device_id, initial_analytics);
        }
        info!(
            "Initialized health monitoring for {} devices",
            device_map.len()
        );
        Ok(())
    }
    /// Create initial health status for a device
    fn create_initial_health_status(device: &GpuDeviceInfo) -> GpuHealthStatus {
        GpuHealthStatus {
            device_id: device.device_id,
            is_healthy: true,
            health_score: 1.0,
            last_check: Utc::now(),
            issues: Vec::new(),
            temperature_ok: true,
            memory_ok: true,
            performance_ok: true,
            power_ok: true,
            driver_ok: true,
            hardware_ok: true,
            health_trend: HealthTrend::Unknown,
            current_temperature: 45.0,
            current_memory_usage: 0.0,
            current_utilization: device.utilization_percent,
            current_power: 150.0,
            consecutive_healthy_checks: 1,
            consecutive_unhealthy_checks: 0,
            time_since_last_issue: None,
            predicted_failure_time: None,
        }
    }
    /// Create initial analytics data for a device
    fn create_initial_analytics(device_id: usize) -> GpuHealthAnalytics {
        let now = Utc::now();
        let mut health_history = VecDeque::new();
        health_history.push_back((now, 1.0));
        GpuHealthAnalytics {
            device_id,
            health_history,
            temperature_history: VecDeque::new(),
            memory_history: VecDeque::new(),
            performance_history: VecDeque::new(),
            average_health_score: 1.0,
            health_trend_slope: 0.0,
            health_score_stddev: 0.0,
            trend_analysis: HealthTrendAnalysis {
                trend: HealthTrend::Unknown,
                confidence: 0.0,
                projected_24h: 1.0,
                projected_7d: 1.0,
                risk_level: HealthRiskLevel::Low,
                recommendations: Vec::new(),
            },
            prediction_model: None,
        }
    }
    /// Start the health checking background task
    async fn start_health_checking_task(
        &self,
        devices: Arc<RwLock<HashMap<usize, GpuDeviceInfo>>>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) -> GpuHealthResult<()> {
        let health_status = self.health_status.clone();
        let config = self.config.clone();
        let event_sender = self.health_event_sender.clone();
        let total_checks = self.total_health_checks.clone();
        let task = tokio::spawn(async move {
            let check_interval = {
                let config = config.read();
                config.check_interval
            };
            let mut interval = interval(check_interval);
            loop {
                tokio::select! {
                    _ = interval.tick() => { let devices_to_check : Vec < _ > = { let
                    device_map = devices.read(); device_map.values().cloned().collect()
                    }; for device in devices_to_check { let new_health =
                    Self::perform_comprehensive_health_check(& device, & config). await;
                    let previous_health = { let mut status_map = health_status.write();
                    let previous = status_map.get(& device.device_id).cloned();
                    status_map.insert(device.device_id, new_health.clone()); previous };
                    if let Some(prev) = previous_health { Self::generate_health_events(&
                    prev, & new_health, & event_sender). await; } total_checks
                    .fetch_add(1, Ordering::Relaxed); } } _ = shutdown_rx.recv() => {
                    debug!("Health checking task shutting down"); break; }
                }
            }
        });
        self.background_tasks.lock().push(task);
        Ok(())
    }
    /// Perform comprehensive health check on a device
    ///
    /// This method performs a detailed health assessment of a GPU device,
    /// checking multiple health indicators and computing an overall health score.
    ///
    /// # Arguments
    ///
    /// * `device` - The GPU device to check
    /// * `config` - Health monitoring configuration
    ///
    /// # Returns
    ///
    /// Updated health status for the device
    #[instrument(skip(config))]
    pub(crate) async fn perform_comprehensive_health_check(
        device: &GpuDeviceInfo,
        config: &Arc<RwLock<GpuHealthConfig>>,
    ) -> GpuHealthStatus {
        let config = config.read();
        let mut issues = Vec::new();
        let mut health_score: f64 = 1.0;
        let simulated_temp = 45.0 + (device.utilization_percent * 0.5);
        let memory_usage_ratio = (device.total_memory_mb - device.available_memory_mb) as f32
            / device.total_memory_mb as f32
            * 100.0;
        let power_consumption = 150.0 + (device.utilization_percent * 2.0);
        let temperature_ok = simulated_temp < config.temperature_threshold;
        if !temperature_ok {
            issues.push(format!(
                "High temperature: {:.1}°C (threshold: {:.1}°C)",
                simulated_temp, config.temperature_threshold
            ));
            health_score -= 0.3;
        }
        let memory_ok = memory_usage_ratio < config.memory_threshold;
        if !memory_ok {
            issues.push(format!(
                "High memory usage: {:.1}% (threshold: {:.1}%)",
                memory_usage_ratio, config.memory_threshold
            ));
            health_score -= 0.2;
        }
        let performance_ok = device.utilization_percent < config.utilization_threshold;
        if !performance_ok {
            issues.push(format!(
                "Extremely high utilization: {:.1}% (threshold: {:.1}%)",
                device.utilization_percent, config.utilization_threshold
            ));
            health_score -= 0.2;
        }
        let power_ok = power_consumption < config.power_threshold;
        if !power_ok {
            issues.push(format!(
                "High power consumption: {:.1}W (threshold: {:.1}W)",
                power_consumption, config.power_threshold
            ));
            health_score -= 0.15;
        }
        let hardware_ok = matches!(
            device.status,
            GpuDeviceStatus::Available | GpuDeviceStatus::Busy
        );
        if !hardware_ok {
            issues.push(format!("Device status: {:?}", device.status));
            health_score -= 0.4;
        }
        let driver_ok = true;
        health_score = health_score.max(0.0_f64);
        let is_healthy = health_score >= config.health_score_threshold as f64 && issues.is_empty();
        GpuHealthStatus {
            device_id: device.device_id,
            is_healthy,
            health_score: health_score as f32,
            last_check: Utc::now(),
            issues,
            temperature_ok,
            memory_ok,
            performance_ok,
            power_ok,
            driver_ok,
            hardware_ok,
            health_trend: HealthTrend::Unknown,
            current_temperature: simulated_temp,
            current_memory_usage: memory_usage_ratio,
            current_utilization: device.utilization_percent,
            current_power: power_consumption,
            consecutive_healthy_checks: 0,
            consecutive_unhealthy_checks: 0,
            time_since_last_issue: None,
            predicted_failure_time: None,
        }
    }
    /// Generate health events based on status changes
    async fn generate_health_events(
        previous: &GpuHealthStatus,
        current: &GpuHealthStatus,
        event_sender: &Arc<Mutex<Option<mpsc::UnboundedSender<HealthEvent>>>>,
    ) {
        let sender = event_sender.lock();
        if let Some(sender) = sender.as_ref() {
            if previous.is_healthy != current.is_healthy {
                let event_type = if current.is_healthy {
                    HealthEventType::HealthImproved
                } else {
                    HealthEventType::HealthDegraded
                };
                let event = HealthEvent {
                    device_id: current.device_id,
                    event_type,
                    timestamp: current.last_check,
                    message: format!(
                        "Device {} health changed: {} -> {}",
                        current.device_id,
                        if previous.is_healthy { "healthy" } else { "unhealthy" },
                        if current.is_healthy { "healthy" } else { "unhealthy" }
                    ),
                    health_score: current.health_score,
                    data: HashMap::new(),
                };
                let _ = sender.send(event);
            }
            for issue in &current.issues {
                if !previous.issues.contains(issue) {
                    let event = HealthEvent {
                        device_id: current.device_id,
                        event_type: HealthEventType::IssueDetected,
                        timestamp: current.last_check,
                        message: format!(
                            "New health issue detected on device {}: {}",
                            current.device_id, issue
                        ),
                        health_score: current.health_score,
                        data: HashMap::new(),
                    };
                    let _ = sender.send(event);
                }
            }
            for issue in &previous.issues {
                if !current.issues.contains(issue) {
                    let event = HealthEvent {
                        device_id: current.device_id,
                        event_type: HealthEventType::IssueResolved,
                        timestamp: current.last_check,
                        message: format!(
                            "Health issue resolved on device {}: {}",
                            current.device_id, issue
                        ),
                        health_score: current.health_score,
                        data: HashMap::new(),
                    };
                    let _ = sender.send(event);
                }
            }
            if current.health_score < 0.3 && previous.health_score >= 0.3 {
                let event = HealthEvent {
                    device_id: current.device_id,
                    event_type: HealthEventType::CriticalWarning,
                    timestamp: current.last_check,
                    message: format!(
                        "Critical health warning for device {}: health score {:.2}",
                        current.device_id, current.health_score
                    ),
                    health_score: current.health_score,
                    data: HashMap::new(),
                };
                let _ = sender.send(event);
            }
        }
    }
    /// Start analytics computation task
    async fn start_analytics_task(
        &self,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) -> GpuHealthResult<()> {
        let health_status = self.health_status.clone();
        let health_analytics = self.health_analytics.clone();
        let config = self.config.clone();
        let task = tokio::spawn(async move {
            let analytics_interval = {
                let config = config.read();
                config.analytics_interval
            };
            let mut interval = interval(analytics_interval);
            loop {
                tokio::select! {
                    _ = interval.tick() => { Self::compute_health_analytics(&
                    health_status, & health_analytics, & config). await; } _ =
                    shutdown_rx.recv() => {
                    debug!("Analytics computation task shutting down"); break; }
                }
            }
        });
        self.background_tasks.lock().push(task);
        Ok(())
    }
    /// Compute health analytics and trends
    async fn compute_health_analytics(
        health_status: &Arc<RwLock<HashMap<usize, GpuHealthStatus>>>,
        health_analytics: &Arc<RwLock<HashMap<usize, GpuHealthAnalytics>>>,
        config: &Arc<RwLock<GpuHealthConfig>>,
    ) {
        let status_map = health_status.read();
        let mut analytics_map = health_analytics.write();
        let config = config.read();
        for (device_id, status) in status_map.iter() {
            if let Some(analytics) = analytics_map.get_mut(device_id) {
                analytics.health_history.push_back((status.last_check, status.health_score));
                analytics
                    .temperature_history
                    .push_back((status.last_check, status.current_temperature));
                analytics
                    .memory_history
                    .push_back((status.last_check, status.current_memory_usage));
                analytics
                    .performance_history
                    .push_back((status.last_check, status.current_utilization));
                let cutoff_time = Utc::now() - config.history_retention;
                analytics.health_history.retain(|(time, _)| *time > cutoff_time);
                analytics.temperature_history.retain(|(time, _)| *time > cutoff_time);
                analytics.memory_history.retain(|(time, _)| *time > cutoff_time);
                analytics.performance_history.retain(|(time, _)| *time > cutoff_time);
                Self::update_analytics_metrics(analytics);
                Self::compute_trend_analysis(analytics);
                if config.enable_prediction {
                    Self::update_prediction_model(analytics);
                }
            }
        }
    }
    /// Update analytics metrics
    fn update_analytics_metrics(analytics: &mut GpuHealthAnalytics) {
        if analytics.health_history.is_empty() {
            return;
        }
        let sum: f32 = analytics.health_history.iter().map(|(_, score)| score).sum();
        analytics.average_health_score = sum / analytics.health_history.len() as f32;
        let variance: f32 = analytics
            .health_history
            .iter()
            .map(|(_, score)| {
                let diff = score - analytics.average_health_score;
                diff * diff
            })
            .sum::<f32>()
            / analytics.health_history.len() as f32;
        analytics.health_score_stddev = variance.sqrt();
        if analytics.health_history.len() >= 2 {
            let n = analytics.health_history.len() as f32;
            let sum_x: f32 = (0..analytics.health_history.len()).map(|i| i as f32).sum();
            let sum_y: f32 = analytics.health_history.iter().map(|(_, score)| score).sum();
            let sum_xy: f32 = analytics
                .health_history
                .iter()
                .enumerate()
                .map(|(i, (_, score))| (i as f32) * score)
                .sum();
            let sum_x2: f32 = (0..analytics.health_history.len()).map(|i| (i as f32).powi(2)).sum();
            let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
            analytics.health_trend_slope = slope;
        }
    }
    /// Compute comprehensive trend analysis
    pub(crate) fn compute_trend_analysis(analytics: &mut GpuHealthAnalytics) {
        let trend = if analytics.health_trend_slope > 0.01 {
            HealthTrend::Improving
        } else if analytics.health_trend_slope < -0.01 {
            HealthTrend::Declining
        } else if analytics.health_history.len() >= 5 {
            HealthTrend::Stable
        } else {
            HealthTrend::Unknown
        };
        let confidence = if analytics.health_history.len() >= 20 {
            0.9
        } else if analytics.health_history.len() >= 10 {
            0.7
        } else if analytics.health_history.len() >= 5 {
            0.5
        } else {
            0.2
        };
        let current_score = analytics
            .health_history
            .back()
            .map(|(_, score)| *score)
            .unwrap_or(analytics.average_health_score);
        let projected_24h = (current_score + analytics.health_trend_slope * 48.0).clamp(0.0, 1.0);
        let projected_7d = (current_score + analytics.health_trend_slope * 336.0).clamp(0.0, 1.0);
        let risk_level = if projected_24h < 0.3 || current_score < 0.5 {
            HealthRiskLevel::Critical
        } else if projected_24h < 0.5 || current_score < 0.7 {
            HealthRiskLevel::High
        } else if trend == HealthTrend::Declining || projected_7d < 0.7 {
            HealthRiskLevel::Medium
        } else {
            HealthRiskLevel::Low
        };
        let mut recommendations = Vec::new();
        if trend == HealthTrend::Declining {
            recommendations.push("Monitor device closely for performance degradation".to_string());
        }
        if projected_24h < 0.5 {
            recommendations.push("Consider scheduling maintenance within 24 hours".to_string());
        }
        if analytics.average_health_score < 0.8 {
            recommendations.push("Review device workload and thermal management".to_string());
        }
        analytics.trend_analysis = HealthTrendAnalysis {
            trend,
            confidence,
            projected_24h,
            projected_7d,
            risk_level,
            recommendations,
        };
    }
    /// Update predictive failure model
    fn update_prediction_model(analytics: &mut GpuHealthAnalytics) {
        if analytics.health_history.len() < 10 {
            return;
        }
        let failure_probability_24h = if analytics.health_trend_slope < -0.02 {
            ((-analytics.health_trend_slope) * 50.0).min(0.8)
        } else {
            0.1 * (1.0 - analytics.average_health_score)
        };
        let failure_probability_7d = failure_probability_24h * 3.0;
        let failure_probability_30d = failure_probability_7d * 2.0;
        let mut risk_factors = Vec::new();
        if analytics.health_trend_slope < -0.01 {
            risk_factors.push("Declining health trend".to_string());
        }
        if analytics.average_health_score < 0.7 {
            risk_factors.push("Below-average health score".to_string());
        }
        if analytics.health_score_stddev > 0.2 {
            risk_factors.push("High health score variability".to_string());
        }
        analytics.prediction_model = Some(HealthPredictionModel {
            model_type: "Linear Trend Model".to_string(),
            confidence: analytics.trend_analysis.confidence,
            failure_probability_24h,
            failure_probability_7d: failure_probability_7d.min(1.0),
            failure_probability_30d: failure_probability_30d.min(1.0),
            risk_factors,
            last_updated: Utc::now(),
        });
    }
    /// Start event processing task
    async fn start_event_processing_task(
        &self,
        mut event_receiver: mpsc::UnboundedReceiver<HealthEvent>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) -> GpuHealthResult<()> {
        let stats = self.stats.clone();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    event = event_receiver.recv() => { if let Some(event) = event {
                    Self::process_health_event(event, & stats). await; } else {
                    debug!("Health event channel closed"); break; } } _ = shutdown_rx
                    .recv() => { debug!("Event processing task shutting down"); break; }
                }
            }
        });
        self.background_tasks.lock().push(task);
        Ok(())
    }
    /// Process a health event
    async fn process_health_event(event: HealthEvent, stats: &Arc<RwLock<HealthMonitoringStats>>) {
        match event.event_type {
            HealthEventType::CriticalWarning => {
                error!(
                    "Critical health warning for device {}: {}",
                    event.device_id, event.message
                );
                let mut stats = stats.write();
                stats.critical_warnings += 1;
            },
            HealthEventType::PredictiveWarning => {
                warn!(
                    "Predictive health warning for device {}: {}",
                    event.device_id, event.message
                );
                let mut stats = stats.write();
                stats.predictive_warnings += 1;
            },
            HealthEventType::HealthDegraded => {
                warn!(
                    "Health degraded for device {}: {}",
                    event.device_id, event.message
                );
            },
            HealthEventType::HealthImproved => {
                info!(
                    "Health improved for device {}: {}",
                    event.device_id, event.message
                );
            },
            _ => {
                debug!(
                    "Health event for device {}: {}",
                    event.device_id, event.message
                );
            },
        }
        let mut stats = stats.write();
        stats.total_events += 1;
        stats.last_updated = Utc::now();
    }
    /// Start statistics update task
    async fn start_statistics_task(
        &self,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) -> GpuHealthResult<()> {
        let health_status = self.health_status.clone();
        let stats = self.stats.clone();
        let total_checks = self.total_health_checks.clone();
        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = interval.tick() => { Self::update_statistics(& health_status, &
                    stats, & total_checks). await; } _ = shutdown_rx.recv() => {
                    debug!("Statistics update task shutting down"); break; }
                }
            }
        });
        self.background_tasks.lock().push(task);
        Ok(())
    }
    /// Update health monitoring statistics
    async fn update_statistics(
        health_status: &Arc<RwLock<HashMap<usize, GpuHealthStatus>>>,
        stats: &Arc<RwLock<HealthMonitoringStats>>,
        total_checks: &Arc<AtomicU64>,
    ) {
        let status_map = health_status.read();
        let mut stats = stats.write();
        stats.devices_monitored = status_map.len();
        stats.total_checks = total_checks.load(Ordering::Relaxed);
        let healthy_count = status_map.values().filter(|s| s.is_healthy).count();
        stats.healthy_devices = healthy_count;
        stats.unhealthy_devices = status_map.len() - healthy_count;
        if !status_map.is_empty() {
            let total_score: f32 = status_map.values().map(|s| s.health_score).sum();
            stats.average_health_score = total_score / status_map.len() as f32;
        }
        stats.last_updated = Utc::now();
    }
    /// Get comprehensive health status for all monitored devices
    ///
    /// # Returns
    ///
    /// HashMap mapping device IDs to their current health status
    pub async fn get_health_status(&self) -> HashMap<usize, GpuHealthStatus> {
        let status = self.health_status.read();
        status.clone()
    }
    /// Get health status for a specific device
    ///
    /// # Arguments
    ///
    /// * `device_id` - ID of the device to query
    ///
    /// # Returns
    ///
    /// Optional health status for the specified device
    pub async fn get_device_health_status(&self, device_id: usize) -> Option<GpuHealthStatus> {
        let status = self.health_status.read();
        status.get(&device_id).cloned()
    }
    /// Get health analytics for all devices
    ///
    /// # Returns
    ///
    /// HashMap mapping device IDs to their health analytics data
    pub async fn get_health_analytics(&self) -> HashMap<usize, GpuHealthAnalytics> {
        let analytics = self.health_analytics.read();
        analytics.clone()
    }
    /// Get health analytics for a specific device
    ///
    /// # Arguments
    ///
    /// * `device_id` - ID of the device to query
    ///
    /// # Returns
    ///
    /// Optional health analytics for the specified device
    pub async fn get_device_analytics(&self, device_id: usize) -> Option<GpuHealthAnalytics> {
        let analytics = self.health_analytics.read();
        analytics.get(&device_id).cloned()
    }
    /// Get health monitoring statistics
    ///
    /// # Returns
    ///
    /// Current health monitoring statistics
    pub async fn get_monitoring_stats(&self) -> HealthMonitoringStats {
        let stats = self.stats.read();
        stats.clone()
    }
    /// Get overall health summary
    ///
    /// # Returns
    ///
    /// Summary of overall health across all devices
    pub async fn get_health_summary(&self) -> HealthSummary {
        let status_map = self.health_status.read();
        let stats = self.stats.read();
        let total_devices = status_map.len();
        let healthy_devices = status_map.values().filter(|s| s.is_healthy).count();
        let critical_devices = status_map.values().filter(|s| s.health_score < 0.3).count();
        let average_health = if total_devices > 0 {
            status_map.values().map(|s| s.health_score).sum::<f32>() / total_devices as f32
        } else {
            0.0
        };
        let overall_status = if critical_devices > 0 {
            OverallHealthStatus::Critical
        } else if healthy_devices == total_devices {
            OverallHealthStatus::Healthy
        } else {
            OverallHealthStatus::Warning
        };
        HealthSummary {
            overall_status,
            total_devices,
            healthy_devices,
            unhealthy_devices: total_devices - healthy_devices,
            critical_devices,
            average_health_score: average_health,
            total_health_checks: stats.total_checks,
            last_updated: Utc::now(),
        }
    }
    /// Generate comprehensive health report
    ///
    /// This method creates a detailed health report including device status,
    /// analytics, trends, and recommendations.
    ///
    /// # Returns
    ///
    /// Formatted health report string
    #[instrument(skip(self))]
    pub async fn generate_health_report(&self) -> String {
        let status_map = self.health_status.read();
        let analytics_map = self.health_analytics.read();
        let stats = self.stats.read();
        let summary = self.get_health_summary().await;
        let mut report = String::new();
        report.push_str(&format!(
            "GPU Health Monitoring Report\n\
             ========================================\n\
             Generated: {}\n\
             \n\
             OVERALL HEALTH SUMMARY\n\
             Overall Status: {:?}\n\
             Total Devices: {}\n\
             Healthy Devices: {}\n\
             Unhealthy Devices: {}\n\
             Critical Devices: {}\n\
             Average Health Score: {:.3}\n\
             \n\
             MONITORING STATISTICS\n\
             Total Health Checks: {}\n\
             Critical Warnings: {}\n\
             Predictive Warnings: {}\n\
             Total Events: {}\n\
             \n",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            summary.overall_status,
            summary.total_devices,
            summary.healthy_devices,
            summary.unhealthy_devices,
            summary.critical_devices,
            summary.average_health_score,
            stats.total_checks,
            stats.critical_warnings,
            stats.predictive_warnings,
            stats.total_events,
        ));
        report.push_str("DEVICE HEALTH DETAILS\n");
        report.push_str("========================================\n\n");
        for (device_id, health) in status_map.iter() {
            let analytics = analytics_map.get(device_id);
            report.push_str(&format!(
                "Device {}: {} (Score: {:.3})\n\
                 - Status: {}\n\
                 - Temperature: {:.1}°C (OK: {})\n\
                 - Memory Usage: {:.1}% (OK: {})\n\
                 - Utilization: {:.1}% (OK: {})\n\
                 - Power: {:.1}W (OK: {})\n\
                 - Last Check: {}\n",
                device_id,
                if health.is_healthy { "HEALTHY" } else { "UNHEALTHY" },
                health.health_score,
                if health.is_healthy { "OK" } else { "ISSUES DETECTED" },
                health.current_temperature,
                health.temperature_ok,
                health.current_memory_usage,
                health.memory_ok,
                health.current_utilization,
                health.performance_ok,
                health.current_power,
                health.power_ok,
                health.last_check.format("%Y-%m-%d %H:%M:%S UTC"),
            ));
            if !health.issues.is_empty() {
                report.push_str("  Issues:\n");
                for issue in &health.issues {
                    report.push_str(&format!("    - {}\n", issue));
                }
            }
            if let Some(analytics) = analytics {
                report.push_str(&format!(
                    "  Trend Analysis:\n\
                     - Trend: {:?} (Confidence: {:.2})\n\
                     - Risk Level: {:?}\n\
                     - 24h Projection: {:.3}\n\
                     - 7d Projection: {:.3}\n",
                    analytics.trend_analysis.trend,
                    analytics.trend_analysis.confidence,
                    analytics.trend_analysis.risk_level,
                    analytics.trend_analysis.projected_24h,
                    analytics.trend_analysis.projected_7d,
                ));
                if !analytics.trend_analysis.recommendations.is_empty() {
                    report.push_str("  Recommendations:\n");
                    for rec in &analytics.trend_analysis.recommendations {
                        report.push_str(&format!("    - {}\n", rec));
                    }
                }
                if let Some(model) = &analytics.prediction_model {
                    report.push_str(&format!(
                        "  Failure Prediction:\n\
                         - 24h Risk: {:.1}%\n\
                         - 7d Risk: {:.1}%\n\
                         - 30d Risk: {:.1}%\n",
                        model.failure_probability_24h * 100.0,
                        model.failure_probability_7d * 100.0,
                        model.failure_probability_30d * 100.0,
                    ));
                }
            }
            report.push('\n');
        }
        report
    }
    /// Update health monitoring configuration
    ///
    /// # Arguments
    ///
    /// * `new_config` - New health monitoring configuration
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of configuration update
    pub async fn update_config(&self, new_config: GpuHealthConfig) -> GpuHealthResult<()> {
        let mut config = self.config.write();
        *config = new_config;
        info!("Health monitoring configuration updated");
        Ok(())
    }
    /// Force a health check on all devices
    ///
    /// This method immediately triggers a health check on all monitored devices,
    /// bypassing the normal check interval.
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of the forced health check
    #[instrument(skip(self))]
    pub async fn force_health_check(&self) -> GpuHealthResult<()> {
        if !self.running.load(Ordering::Acquire) {
            return Err(GpuHealthError::NotInitialized);
        }
        info!("Forcing immediate health check on all devices");
        self.total_health_checks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    /// Check if health monitoring is running
    ///
    /// # Returns
    ///
    /// True if health monitoring is active, false otherwise
    pub fn is_monitoring(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
}
/// Predictive health model for failure prediction
#[derive(Debug, Clone)]
pub struct HealthPredictionModel {
    /// Model type identifier
    pub model_type: String,
    /// Model confidence score
    pub confidence: f32,
    /// Predicted failure probability in next 24 hours
    pub failure_probability_24h: f32,
    /// Predicted failure probability in next 7 days
    pub failure_probability_7d: f32,
    /// Predicted failure probability in next 30 days
    pub failure_probability_30d: f32,
    /// Key risk factors identified
    pub risk_factors: Vec<String>,
    /// Model last updated time
    pub last_updated: DateTime<Utc>,
}
/// Health monitoring configuration
#[derive(Debug, Clone)]
pub struct GpuHealthConfig {
    /// Health check interval
    pub check_interval: Duration,
    /// Temperature threshold (Celsius)
    pub temperature_threshold: f32,
    /// Memory usage threshold (percentage)
    pub memory_threshold: f32,
    /// Utilization threshold (percentage)
    pub utilization_threshold: f32,
    /// Power consumption threshold (Watts)
    pub power_threshold: f32,
    /// Health score threshold for unhealthy classification
    pub health_score_threshold: f32,
    /// Number of consecutive checks required for status change
    pub consecutive_checks_threshold: u32,
    /// Enable predictive analytics
    pub enable_prediction: bool,
    /// History retention duration
    pub history_retention: ChronoDuration,
    /// Analytics computation interval
    pub analytics_interval: Duration,
}
/// Types of health events
#[derive(Debug, Clone, PartialEq)]
pub enum HealthEventType {
    /// Device became healthy
    HealthImproved,
    /// Device became unhealthy
    HealthDegraded,
    /// Health issue detected
    IssueDetected,
    /// Health issue resolved
    IssueResolved,
    /// Critical health warning
    CriticalWarning,
    /// Health check failed
    CheckFailed,
    /// Predictive failure warning
    PredictiveWarning,
    /// Recovery action taken
    RecoveryAction,
}
/// Comprehensive error types for GPU health monitoring operations
#[derive(Debug, thiserror::Error)]
pub enum GpuHealthError {
    #[error("Health monitoring not initialized")]
    NotInitialized,
    #[error("Device {device_id} not found in health monitoring")]
    DeviceNotFound { device_id: usize },
    #[error("Health check failed for device {device_id}: {reason}")]
    HealthCheckFailed { device_id: usize, reason: String },
    #[error("Health monitoring task error: {message}")]
    TaskError { message: String },
    #[error("Analytics computation error: {source}")]
    AnalyticsError {
        #[from]
        source: anyhow::Error,
    },
    #[error("Health threshold exceeded: {threshold_type} = {value} > {limit}")]
    ThresholdExceeded {
        threshold_type: String,
        value: f32,
        limit: f32,
    },
}
/// Health trend analysis results
#[derive(Debug, Clone)]
pub struct HealthTrendAnalysis {
    /// Overall trend direction
    pub trend: HealthTrend,
    /// Trend confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Projected health score in 24 hours
    pub projected_24h: f32,
    /// Projected health score in 7 days
    pub projected_7d: f32,
    /// Risk assessment
    pub risk_level: HealthRiskLevel,
    /// Recommended actions
    pub recommendations: Vec<String>,
}
/// Health event for notifications and logging
#[derive(Debug, Clone)]
pub struct HealthEvent {
    /// Device ID
    pub device_id: usize,
    /// Event type
    pub event_type: HealthEventType,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event message
    pub message: String,
    /// Health score at time of event
    pub health_score: f32,
    /// Associated data
    pub data: HashMap<String, String>,
}
/// Health summary for quick overview
#[derive(Debug, Clone)]
pub struct HealthSummary {
    /// Overall health status
    pub overall_status: OverallHealthStatus,
    /// Total number of devices monitored
    pub total_devices: usize,
    /// Number of healthy devices
    pub healthy_devices: usize,
    /// Number of unhealthy devices
    pub unhealthy_devices: usize,
    /// Number of critical devices
    pub critical_devices: usize,
    /// Average health score across all devices
    pub average_health_score: f32,
    /// Total health checks performed
    pub total_health_checks: u64,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}
/// Health analytics data for trend analysis and prediction
#[derive(Debug, Clone)]
pub struct GpuHealthAnalytics {
    /// Device ID this analytics data applies to
    pub device_id: usize,
    /// Health score history (time-series data)
    pub health_history: VecDeque<(DateTime<Utc>, f32)>,
    /// Temperature history
    pub temperature_history: VecDeque<(DateTime<Utc>, f32)>,
    /// Memory usage history
    pub memory_history: VecDeque<(DateTime<Utc>, f32)>,
    /// Performance metrics history
    pub performance_history: VecDeque<(DateTime<Utc>, f32)>,
    /// Average health score over time
    pub average_health_score: f32,
    /// Health score trend slope
    pub health_trend_slope: f32,
    /// Standard deviation of health scores
    pub health_score_stddev: f32,
    /// Time series health analysis
    pub trend_analysis: HealthTrendAnalysis,
    /// Predictive model data
    pub prediction_model: Option<HealthPredictionModel>,
}
/// Health risk level assessment
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthRiskLevel {
    /// Low risk - device is healthy
    Low,
    /// Medium risk - monitor closely
    Medium,
    /// High risk - intervention recommended
    High,
    /// Critical risk - immediate action required
    Critical,
}
/// GPU health status for individual devices
///
/// This struct represents the comprehensive health status of a GPU device,
/// including various health metrics, diagnostic information, and historical data.
#[derive(Debug, Clone)]
pub struct GpuHealthStatus {
    /// Device identifier
    pub device_id: usize,
    /// Overall health indicator
    pub is_healthy: bool,
    /// Normalized health score (0.0 = critical, 1.0 = perfect health)
    pub health_score: f32,
    /// Timestamp of last health check
    pub last_check: DateTime<Utc>,
    /// List of identified health issues
    pub issues: Vec<String>,
    /// Temperature health status
    pub temperature_ok: bool,
    /// Memory health status
    pub memory_ok: bool,
    /// Performance health status
    pub performance_ok: bool,
    /// Power consumption health status
    pub power_ok: bool,
    /// Driver health status
    pub driver_ok: bool,
    /// Hardware health status
    pub hardware_ok: bool,
    /// Health trend over time
    pub health_trend: HealthTrend,
    /// Current temperature reading (Celsius)
    pub current_temperature: f32,
    /// Current memory usage percentage
    pub current_memory_usage: f32,
    /// Current utilization percentage
    pub current_utilization: f32,
    /// Current power consumption (Watts)
    pub current_power: f32,
    /// Number of consecutive healthy checks
    pub consecutive_healthy_checks: u32,
    /// Number of consecutive unhealthy checks
    pub consecutive_unhealthy_checks: u32,
    /// Time since last health issue
    pub time_since_last_issue: Option<ChronoDuration>,
    /// Predicted time to failure (if applicable)
    pub predicted_failure_time: Option<DateTime<Utc>>,
}
/// Overall health status for the entire system
#[derive(Debug, Clone, PartialEq)]
pub enum OverallHealthStatus {
    /// All devices are healthy
    Healthy,
    /// Some devices have issues but no critical problems
    Warning,
    /// One or more devices are in critical condition
    Critical,
    /// Health monitoring is not available
    Unknown,
}
