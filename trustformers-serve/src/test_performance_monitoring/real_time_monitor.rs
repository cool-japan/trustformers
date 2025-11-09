//! Real-Time Performance Monitoring
//!
//! This module provides real-time monitoring capabilities for test performance,
//! including live data collection, streaming metrics, and continuous monitoring.

use super::metrics::*;
use super::types::*;
use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::time::interval;

/// Real-time performance monitor for continuous test monitoring
pub struct RealTimePerformanceMonitor {
    config: RealTimeMonitoringConfig,
    monitoring_state: Arc<MonitoringState>,
    metric_collectors: HashMap<String, Box<dyn MetricCollector + Send + Sync>>,
    stream_manager: Arc<StreamManager>,
    alert_manager: Arc<RwLock<AlertManager>>,
    data_buffer: Arc<Mutex<CircularBuffer<StreamingMetrics>>>,
    subscribers: Arc<RwLock<HashMap<String, SubscriberInfo>>>,
    monitoring_task_handle: Option<tokio::task::JoinHandle<()>>,
}

/// Monitoring state tracking
#[derive(Debug)]
pub struct MonitoringState {
    pub is_monitoring: AtomicBool,
    pub start_time: RwLock<Option<SystemTime>>,
    pub active_tests: RwLock<HashMap<String, ActiveTestInfo>>,
    pub total_tests_monitored: AtomicU64,
    pub total_metrics_collected: AtomicU64,
    pub monitoring_errors: AtomicU64,
    pub last_heartbeat: RwLock<Option<SystemTime>>,
}

/// Information about actively monitored test
#[derive(Debug, Clone)]
pub struct ActiveTestInfo {
    pub test_id: String,
    pub test_name: String,
    pub start_time: SystemTime,
    pub current_phase: TestPhase,
    pub progress_percent: f64,
    pub last_update: SystemTime,
    pub resource_usage: ResourceUsageSnapshot,
    pub performance_indicators: Vec<LivePerformanceIndicator>,
    pub anomaly_flags: Vec<AnomalyFlag>,
}

/// Current resource usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageSnapshot {
    pub cpu_percent: f64,
    pub memory_mb: u64,
    pub io_rate_mbps: f64,
    pub network_rate_mbps: f64,
    pub disk_usage_percent: f64,
    pub open_files: u32,
    pub active_threads: u32,
    pub timestamp: SystemTime,
}

/// Trait for metric collection components
pub trait MetricCollector {
    fn collect_metrics(&self, test_id: &str) -> Result<StreamingMetrics, MonitoringError>;
    fn get_collector_name(&self) -> &str;
    fn is_enabled(&self) -> bool;
    fn get_collection_interval(&self) -> Duration;
}

/// System resource metric collector
#[derive(Debug)]
pub struct SystemResourceCollector {
    pub name: String,
    pub enabled: bool,
    pub collection_interval: Duration,
    pub last_collection: RwLock<Option<Instant>>,
}

/// Process-specific metric collector
#[derive(Debug)]
pub struct ProcessMetricCollector {
    pub name: String,
    pub enabled: bool,
    pub collection_interval: Duration,
    pub process_id: u32,
    pub baseline_metrics: RwLock<Option<ProcessBaseline>>,
}

/// Network performance metric collector
#[derive(Debug)]
pub struct NetworkMetricCollector {
    pub name: String,
    pub enabled: bool,
    pub collection_interval: Duration,
    pub interface_filter: Vec<String>,
    pub connection_tracker: RwLock<HashMap<String, ConnectionInfo>>,
}

/// I/O performance metric collector
#[derive(Debug)]
pub struct IOMetricCollector {
    pub name: String,
    pub enabled: bool,
    pub collection_interval: Duration,
    pub monitored_paths: Vec<String>,
    pub io_tracker: RwLock<IOTracker>,
}

/// Process baseline metrics for comparison
#[derive(Debug, Clone)]
pub struct ProcessBaseline {
    pub cpu_baseline: f64,
    pub memory_baseline: u64,
    pub thread_count_baseline: u32,
    pub file_handle_baseline: u32,
    pub established_at: SystemTime,
}

/// Connection tracking information
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub connection_id: String,
    pub remote_address: String,
    pub local_port: u16,
    pub protocol: String,
    pub established_at: SystemTime,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub last_activity: SystemTime,
}

/// I/O operation tracker
#[derive(Debug, Clone)]
pub struct IOTracker {
    pub read_operations: u64,
    pub write_operations: u64,
    pub total_bytes_read: u64,
    pub total_bytes_written: u64,
    pub average_read_latency: Duration,
    pub average_write_latency: Duration,
    pub last_reset: SystemTime,
}

/// Stream management for real-time data distribution
#[derive(Debug)]
pub struct StreamManager {
    pub active_streams: RwLock<HashMap<String, StreamInfo>>,
    pub stream_sender: mpsc::UnboundedSender<StreamingEvent>,
    pub stream_receiver: Mutex<mpsc::UnboundedReceiver<StreamingEvent>>,
    pub stream_config: StreamConfig,
    pub compression_enabled: bool,
}

/// Information about active data stream
#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub stream_id: String,
    pub test_id: String,
    pub subscriber_count: u32,
    pub start_time: SystemTime,
    pub last_activity: SystemTime,
    pub total_messages_sent: u64,
    pub total_bytes_sent: u64,
    pub stream_quality: StreamQuality,
    pub throttling_active: bool,
}

/// Stream quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamQuality {
    pub latency_ms: f64,
    pub throughput_mbps: f64,
    pub packet_loss_percent: f64,
    pub jitter_ms: f64,
    pub quality_score: f64,
}

/// Streaming event types
#[derive(Debug, Clone)]
pub enum StreamingEvent {
    MetricsUpdate {
        test_id: String,
        metrics: StreamingMetrics,
        timestamp: SystemTime,
    },
    TestStarted {
        test_id: String,
        test_info: ActiveTestInfo,
    },
    TestCompleted {
        test_id: String,
        final_metrics: Box<ComprehensiveTestMetrics>,
    },
    AnomalyDetected {
        test_id: String,
        anomaly: AnomalyFlag,
    },
    AlertTriggered {
        test_id: String,
        alert: AlertInfo,
    },
    SystemPressureUpdate {
        pressure_level: PressureLevel,
        system_metrics: SystemMetrics,
    },
    HeartBeat {
        timestamp: SystemTime,
        active_tests: u32,
        system_health: SystemHealth,
    },
}

/// Alert management for real-time monitoring
pub struct AlertManager {
    pub alert_rules: HashMap<String, AlertRule>,
    pub active_alerts: HashMap<String, ActiveAlert>,
    pub alert_history: VecDeque<AlertEvent>,
    pub notification_channels: Vec<Box<dyn NotificationChannel + Send + Sync>>,
    pub cooldown_periods: HashMap<String, SystemTime>,
    pub escalation_policies: HashMap<String, EscalationPolicy>,
}

impl fmt::Debug for RealTimePerformanceMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let collector_count = self.metric_collectors.len();
        let subscriber_count =
            self.subscribers.try_read().map(|guard| guard.len()).unwrap_or_default();
        let alert_rule_count = self
            .alert_manager
            .try_read()
            .map(|guard| guard.alert_rules.len())
            .unwrap_or_default();

        f.debug_struct("RealTimePerformanceMonitor")
            .field("config", &self.config)
            .field("collector_count", &collector_count)
            .field("subscriber_count", &subscriber_count)
            .field("alert_rule_count", &alert_rule_count)
            .field(
                "monitoring_task_active",
                &self.monitoring_task_handle.is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for AlertManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rule_ids: Vec<_> = self.alert_rules.keys().cloned().collect();
        let active_alert_ids: Vec<_> = self.active_alerts.keys().cloned().collect();

        f.debug_struct("AlertManager")
            .field("alert_rule_ids", &rule_ids)
            .field("active_alert_ids", &active_alert_ids)
            .field("alert_history_len", &self.alert_history.len())
            .field(
                "notification_channel_count",
                &self.notification_channels.len(),
            )
            .field("cooldown_policy_count", &self.cooldown_periods.len())
            .field("escalation_policy_count", &self.escalation_policies.len())
            .finish_non_exhaustive()
    }
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub rule_id: String,
    pub rule_name: String,
    pub description: String,
    pub metric_name: String,
    pub condition: AlertCondition,
    pub threshold_value: f64,
    pub severity: SeverityLevel,
    pub cooldown_duration: Duration,
    pub escalation_policy_id: Option<String>,
    pub enabled: bool,
}

/// Alert condition types
#[derive(Debug, Clone)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
    PercentageIncrease(f64),
    PercentageDecrease(f64),
    ThresholdBreach,
    AnomalyDetected,
    TrendDetected(TrendDirection),
}

/// Active alert information
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    pub alert_id: String,
    pub rule_id: String,
    pub test_id: String,
    pub triggered_at: SystemTime,
    pub current_value: f64,
    pub threshold_value: f64,
    pub severity: SeverityLevel,
    pub escalation_level: u32,
    pub acknowledgment_status: AcknowledgmentStatus,
    pub notifications_sent: u32,
    pub last_notification: Option<SystemTime>,
}

/// Alert event for history tracking
#[derive(Debug, Clone)]
pub struct AlertEvent {
    pub event_id: String,
    pub alert_id: String,
    pub event_type: AlertEventType,
    pub timestamp: SystemTime,
    pub details: HashMap<String, String>,
}

/// Types of alert events
#[derive(Debug, Clone)]
pub enum AlertEventType {
    Triggered,
    Acknowledged,
    Resolved,
    Escalated,
    Suppressed,
    NotificationSent,
    NotificationFailed,
}

/// Acknowledgment status of alerts
#[derive(Debug, Clone, PartialEq)]
pub enum AcknowledgmentStatus {
    Unacknowledged,
    Acknowledged { by: String, at: SystemTime },
    AutoResolved { at: SystemTime },
    ManuallyResolved { by: String, at: SystemTime },
}

/// Notification channel trait
pub trait NotificationChannel {
    fn send_notification(
        &self,
        alert: &ActiveAlert,
        message: &str,
    ) -> Result<(), NotificationError>;
    fn get_channel_name(&self) -> &str;
    fn is_enabled(&self) -> bool;
    fn supports_escalation(&self) -> bool;
}

/// Escalation policy for alert management
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    pub policy_id: String,
    pub policy_name: String,
    pub escalation_steps: Vec<EscalationStep>,
    pub max_escalation_level: u32,
    pub auto_resolve_timeout: Option<Duration>,
}

/// Escalation step definition
#[derive(Debug, Clone)]
pub struct EscalationStep {
    pub step_level: u32,
    pub delay_before_escalation: Duration,
    pub notification_channels: Vec<String>,
    pub additional_recipients: Vec<String>,
    pub escalation_message_template: String,
}

/// Circular buffer for efficient metric storage
#[derive(Debug)]
pub struct CircularBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
    total_items_added: u64,
}

/// Subscriber information for stream management
#[derive(Debug, Clone)]
pub struct SubscriberInfo {
    pub subscriber_id: String,
    pub subscription_type: SubscriptionType,
    pub filter_criteria: SubscriptionFilter,
    pub last_activity: SystemTime,
    pub total_messages_received: u64,
    pub connection_quality: ConnectionQuality,
    pub rate_limit: Option<RateLimit>,
}

/// Connection quality for subscribers
#[derive(Debug, Clone)]
pub struct ConnectionQuality {
    pub latency_ms: f64,
    pub bandwidth_mbps: f64,
    pub reliability_score: f64,
    pub last_heartbeat: SystemTime,
}

/// System health indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_status: HealthStatus,
    pub component_health: HashMap<String, HealthStatus>,
    pub resource_pressure: PressureLevel,
    pub error_rate_percent: f64,
    pub performance_score: f64,
    pub uptime: Duration,
    pub last_check: SystemTime,
}

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Degraded,
    Offline,
}

/// Monitoring errors
#[derive(Debug, Clone)]
pub enum MonitoringError {
    CollectionFailed { collector: String, reason: String },
    StreamingError { stream_id: String, error: String },
    AlertProcessingError { alert_id: String, error: String },
    ConfigurationError { parameter: String, reason: String },
    ResourceExhausted { resource: String },
    NetworkError { endpoint: String, error: String },
    DataCorruption { data_type: String, details: String },
}

/// Notification errors
#[derive(Debug, Clone)]
pub enum NotificationError {
    ChannelUnavailable { channel: String },
    MessageFormatError { reason: String },
    DeliveryFailed { target: String, reason: String },
    RateLimitExceeded { channel: String },
    AuthenticationFailed { channel: String },
    ConfigurationError { parameter: String },
}

impl RealTimePerformanceMonitor {
    /// Create a new real-time performance monitor
    pub fn new(config: RealTimeMonitoringConfig) -> Self {
        let (stream_sender, stream_receiver) = mpsc::unbounded_channel();

        Self {
            config: config.clone(),
            monitoring_state: Arc::new(MonitoringState::new()),
            metric_collectors: HashMap::new(),
            stream_manager: Arc::new(StreamManager {
                active_streams: RwLock::new(HashMap::new()),
                stream_sender,
                stream_receiver: Mutex::new(stream_receiver),
                stream_config: config.stream_config,
                compression_enabled: config.compression_enabled,
            }),
            alert_manager: Arc::new(RwLock::new(AlertManager::new())),
            data_buffer: Arc::new(Mutex::new(CircularBuffer::new(config.buffer_size))),
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            monitoring_task_handle: None,
        }
    }

    /// Start real-time monitoring
    pub async fn start_monitoring(&mut self) -> Result<(), MonitoringError> {
        if self.monitoring_state.is_monitoring.load(Ordering::Relaxed) {
            return Err(MonitoringError::ConfigurationError {
                parameter: "monitoring_state".to_string(),
                reason: "Monitoring is already active".to_string(),
            });
        }

        self.monitoring_state.is_monitoring.store(true, Ordering::Relaxed);
        *self.monitoring_state.start_time.write().await = Some(SystemTime::now());

        // Start monitoring task
        let state = Arc::clone(&self.monitoring_state);
        let collectors = self.create_metric_collectors();
        let stream_manager = Arc::clone(&self.stream_manager);
        let alert_manager = Arc::clone(&self.alert_manager);
        let buffer = Arc::clone(&self.data_buffer);
        let interval_duration = self.config.monitoring_interval;

        let handle = tokio::spawn(async move {
            let mut monitoring_interval = interval(interval_duration);

            while state.is_monitoring.load(Ordering::Relaxed) {
                monitoring_interval.tick().await;

                // Collect metrics from all active tests
                let active_tests = state.active_tests.read().await.clone();
                for (test_id, test_info) in active_tests {
                    if let Err(e) = Self::collect_and_process_metrics(
                        &test_id,
                        &test_info,
                        &collectors,
                        &stream_manager,
                        &alert_manager,
                        &buffer,
                    )
                    .await
                    {
                        state.monitoring_errors.fetch_add(1, Ordering::Relaxed);
                        log::error!("Metric collection failed for test {}: {:?}", test_id, e);
                    }
                }

                // Update heartbeat
                *state.last_heartbeat.write().await = Some(SystemTime::now());
            }
        });

        self.monitoring_task_handle = Some(handle);
        Ok(())
    }

    /// Stop real-time monitoring
    pub async fn stop_monitoring(&mut self) -> Result<(), MonitoringError> {
        self.monitoring_state.is_monitoring.store(false, Ordering::Relaxed);

        if let Some(handle) = self.monitoring_task_handle.take() {
            handle.abort();
        }

        // Clean up active streams
        let mut streams = self.stream_manager.active_streams.write().await;
        streams.clear();

        Ok(())
    }

    /// Register a test for monitoring
    pub async fn register_test(&self, test_info: ActiveTestInfo) -> Result<(), MonitoringError> {
        let mut active_tests = self.monitoring_state.active_tests.write().await;
        active_tests.insert(test_info.test_id.clone(), test_info.clone());

        // Create stream for this test
        let stream_info = StreamInfo {
            stream_id: format!("stream-{}", test_info.test_id),
            test_id: test_info.test_id.clone(),
            subscriber_count: 0,
            start_time: SystemTime::now(),
            last_activity: SystemTime::now(),
            total_messages_sent: 0,
            total_bytes_sent: 0,
            stream_quality: StreamQuality::default(),
            throttling_active: false,
        };

        let mut streams = self.stream_manager.active_streams.write().await;
        streams.insert(stream_info.stream_id.clone(), stream_info);

        // Send test started event
        let test_id = test_info.test_id.clone();
        let _ = self
            .stream_manager
            .stream_sender
            .send(StreamingEvent::TestStarted { test_id, test_info });

        self.monitoring_state.total_tests_monitored.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Unregister a test from monitoring
    pub async fn unregister_test(&self, test_id: &str) -> Result<(), MonitoringError> {
        let mut active_tests = self.monitoring_state.active_tests.write().await;
        if let Some(_test_info) = active_tests.remove(test_id) {
            // Remove associated stream
            let mut streams = self.stream_manager.active_streams.write().await;
            let stream_id = format!("stream-{}", test_id);
            streams.remove(&stream_id);

            // Send test completed event if we have final metrics
            // Note: In real implementation, we'd collect final comprehensive metrics here
        }

        Ok(())
    }

    /// Add a subscriber to monitoring streams
    pub async fn add_subscriber(
        &self,
        subscriber_info: SubscriberInfo,
    ) -> Result<String, MonitoringError> {
        let subscriber_id = subscriber_info.subscriber_id.clone();
        let mut subscribers = self.subscribers.write().await;
        subscribers.insert(subscriber_id.clone(), subscriber_info);
        Ok(subscriber_id)
    }

    /// Remove a subscriber from monitoring streams
    pub async fn remove_subscriber(&self, subscriber_id: &str) -> Result<(), MonitoringError> {
        let mut subscribers = self.subscribers.write().await;
        subscribers.remove(subscriber_id);
        Ok(())
    }

    /// Get current monitoring status
    pub async fn get_monitoring_status(&self) -> MonitoringStatus {
        let active_tests = self.monitoring_state.active_tests.read().await;
        let streams = self.stream_manager.active_streams.read().await;
        let subscribers = self.subscribers.read().await;

        MonitoringStatus {
            is_active: self.monitoring_state.is_monitoring.load(Ordering::Relaxed),
            start_time: *self.monitoring_state.start_time.read().await,
            active_test_count: active_tests.len(),
            active_stream_count: streams.len(),
            subscriber_count: subscribers.len(),
            total_tests_monitored: self
                .monitoring_state
                .total_tests_monitored
                .load(Ordering::Relaxed),
            total_metrics_collected: self
                .monitoring_state
                .total_metrics_collected
                .load(Ordering::Relaxed),
            monitoring_errors: self.monitoring_state.monitoring_errors.load(Ordering::Relaxed),
            last_heartbeat: *self.monitoring_state.last_heartbeat.read().await,
            system_health: self.assess_system_health().await,
        }
    }

    /// Assess current system health
    async fn assess_system_health(&self) -> SystemHealth {
        // This would implement comprehensive system health assessment
        // For now, returning a basic implementation
        SystemHealth {
            overall_status: HealthStatus::Healthy,
            component_health: HashMap::new(),
            resource_pressure: PressureLevel::Low,
            error_rate_percent: 0.0,
            performance_score: 95.0,
            uptime: SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default(),
            last_check: SystemTime::now(),
        }
    }

    /// Create metric collectors based on configuration
    fn create_metric_collectors(&self) -> HashMap<String, Box<dyn MetricCollector + Send + Sync>> {
        let mut collectors: HashMap<String, Box<dyn MetricCollector + Send + Sync>> =
            HashMap::new();

        // Add system resource collector
        collectors.insert(
            "system".to_string(),
            Box::new(SystemResourceCollector {
                name: "SystemResourceCollector".to_string(),
                enabled: true,
                collection_interval: self.config.monitoring_interval,
                last_collection: RwLock::new(None),
            }),
        );

        // Add process metric collector
        collectors.insert(
            "process".to_string(),
            Box::new(ProcessMetricCollector {
                name: "ProcessMetricCollector".to_string(),
                enabled: true,
                collection_interval: self.config.monitoring_interval,
                process_id: std::process::id(),
                baseline_metrics: RwLock::new(None),
            }),
        );

        collectors
    }

    /// Collect and process metrics for a test
    async fn collect_and_process_metrics(
        test_id: &str,
        _test_info: &ActiveTestInfo,
        collectors: &HashMap<String, Box<dyn MetricCollector + Send + Sync>>,
        stream_manager: &StreamManager,
        alert_manager: &RwLock<AlertManager>,
        buffer: &Mutex<CircularBuffer<StreamingMetrics>>,
    ) -> Result<(), MonitoringError> {
        for (collector_name, collector) in collectors {
            if !collector.is_enabled() {
                continue;
            }

            match collector.collect_metrics(test_id) {
                Ok(metrics) => {
                    // Store in buffer
                    {
                        let mut buf = buffer.lock().await;
                        buf.push(metrics.clone());
                    }

                    // Send streaming event
                    let _ = stream_manager.stream_sender.send(StreamingEvent::MetricsUpdate {
                        test_id: test_id.to_string(),
                        metrics: metrics.clone(),
                        timestamp: SystemTime::now(),
                    });

                    // Check for alerts
                    let mut alert_mgr = alert_manager.write().await;
                    alert_mgr.process_metrics(test_id, &metrics).await;
                },
                Err(e) => {
                    return Err(MonitoringError::CollectionFailed {
                        collector: collector_name.clone(),
                        reason: format!("{:?}", e),
                    });
                },
            }
        }

        Ok(())
    }
}

/// Monitoring status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringStatus {
    pub is_active: bool,
    pub start_time: Option<SystemTime>,
    pub active_test_count: usize,
    pub active_stream_count: usize,
    pub subscriber_count: usize,
    pub total_tests_monitored: u64,
    pub total_metrics_collected: u64,
    pub monitoring_errors: u64,
    pub last_heartbeat: Option<SystemTime>,
    pub system_health: SystemHealth,
}

impl MonitoringState {
    fn new() -> Self {
        Self {
            is_monitoring: AtomicBool::new(false),
            start_time: RwLock::new(None),
            active_tests: RwLock::new(HashMap::new()),
            total_tests_monitored: AtomicU64::new(0),
            total_metrics_collected: AtomicU64::new(0),
            monitoring_errors: AtomicU64::new(0),
            last_heartbeat: RwLock::new(None),
        }
    }
}

impl AlertManager {
    fn new() -> Self {
        Self {
            alert_rules: HashMap::new(),
            active_alerts: HashMap::new(),
            alert_history: VecDeque::new(),
            notification_channels: Vec::new(),
            cooldown_periods: HashMap::new(),
            escalation_policies: HashMap::new(),
        }
    }

    async fn process_metrics(&mut self, _test_id: &str, _metrics: &StreamingMetrics) {
        // Process metrics against alert rules
        // This would be implemented with actual alert logic
    }
}

impl<T> CircularBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            total_items_added: 0,
        }
    }

    fn push(&mut self, item: T) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
        self.total_items_added += 1;
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn total_added(&self) -> u64 {
        self.total_items_added
    }
}

impl Default for StreamQuality {
    fn default() -> Self {
        Self {
            latency_ms: 0.0,
            throughput_mbps: 0.0,
            packet_loss_percent: 0.0,
            jitter_ms: 0.0,
            quality_score: 100.0,
        }
    }
}

impl MetricCollector for SystemResourceCollector {
    fn collect_metrics(&self, test_id: &str) -> Result<StreamingMetrics, MonitoringError> {
        // This would implement actual system resource collection
        // For now, returning a mock implementation
        Ok(StreamingMetrics {
            stream_id: format!("system-{}", test_id),
            test_id: test_id.to_string(),
            timestamp: SystemTime::now(),
            elapsed_time: Duration::from_millis(100),
            current_phase: TestPhase::Execution,
            progress_percent: 50.0,
            instantaneous_cpu: 25.0,
            instantaneous_memory: 100 * 1024 * 1024, // 100MB
            instantaneous_io_rate: 10.0,
            instantaneous_network_rate: 5.0,
            live_error_count: 0,
            live_warning_count: 0,
            performance_indicators: vec![],
            anomaly_flags: vec![],
            prediction_metrics: None,
        })
    }

    fn get_collector_name(&self) -> &str {
        &self.name
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn get_collection_interval(&self) -> Duration {
        self.collection_interval
    }
}

impl MetricCollector for ProcessMetricCollector {
    fn collect_metrics(&self, test_id: &str) -> Result<StreamingMetrics, MonitoringError> {
        // This would implement actual process metric collection
        // For now, returning a mock implementation
        Ok(StreamingMetrics {
            stream_id: format!("process-{}", test_id),
            test_id: test_id.to_string(),
            timestamp: SystemTime::now(),
            elapsed_time: Duration::from_millis(150),
            current_phase: TestPhase::Execution,
            progress_percent: 60.0,
            instantaneous_cpu: 30.0,
            instantaneous_memory: 150 * 1024 * 1024, // 150MB
            instantaneous_io_rate: 15.0,
            instantaneous_network_rate: 8.0,
            live_error_count: 0,
            live_warning_count: 1,
            performance_indicators: vec![],
            anomaly_flags: vec![],
            prediction_metrics: None,
        })
    }

    fn get_collector_name(&self) -> &str {
        &self.name
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn get_collection_interval(&self) -> Duration {
        self.collection_interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_real_time_monitor_creation() {
        let config = RealTimeMonitoringConfig::default();
        let monitor = RealTimePerformanceMonitor::new(config);

        assert!(!monitor.monitoring_state.is_monitoring.load(Ordering::Relaxed));
        assert_eq!(monitor.metric_collectors.len(), 0);
    }

    #[tokio::test]
    async fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(3);

        buffer.push("item1");
        buffer.push("item2");
        buffer.push("item3");
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.total_added(), 3);

        buffer.push("item4");
        assert_eq!(buffer.len(), 3); // Should not exceed capacity
        assert_eq!(buffer.total_added(), 4);
    }

    #[tokio::test]
    async fn test_monitoring_state() {
        let state = MonitoringState::new();

        assert!(!state.is_monitoring.load(Ordering::Relaxed));
        assert_eq!(state.total_tests_monitored.load(Ordering::Relaxed), 0);
        assert_eq!(state.total_metrics_collected.load(Ordering::Relaxed), 0);
        assert_eq!(state.monitoring_errors.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_alert_manager_creation() {
        let alert_manager = AlertManager::new();

        assert!(alert_manager.alert_rules.is_empty());
        assert!(alert_manager.active_alerts.is_empty());
        assert!(alert_manager.alert_history.is_empty());
    }

    #[tokio::test]
    async fn test_stream_quality_default() {
        let quality = StreamQuality::default();

        assert_eq!(quality.latency_ms, 0.0);
        assert_eq!(quality.throughput_mbps, 0.0);
        assert_eq!(quality.packet_loss_percent, 0.0);
        assert_eq!(quality.jitter_ms, 0.0);
        assert_eq!(quality.quality_score, 100.0);
    }
}
