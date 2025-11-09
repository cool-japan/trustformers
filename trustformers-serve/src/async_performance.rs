//! Async Task Performance Tracking
//!
//! Provides comprehensive async task performance monitoring and analysis
//! for runtime contention monitoring, stack trace analysis, and task health metrics.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};
use tokio::{
    sync::{broadcast, Mutex, RwLock},
    time::interval,
};
use uuid::Uuid;

/// Async task performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncPerformanceConfig {
    /// Enable async task performance tracking
    pub enabled: bool,

    /// Tracking interval in seconds
    pub tracking_interval_seconds: u64,

    /// Maximum number of task metrics to keep in history
    pub max_history_size: usize,

    /// Performance alert thresholds
    pub alert_thresholds: PerformanceThresholds,

    /// Enable stack trace collection for slow tasks
    pub enable_stack_traces: bool,

    /// Slow task threshold in milliseconds
    pub slow_task_threshold_ms: u64,

    /// Enable runtime contention monitoring
    pub enable_contention_monitoring: bool,

    /// Task sampling rate (0.0-1.0)
    pub sampling_rate: f32,
}

impl Default for AsyncPerformanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tracking_interval_seconds: 5,
            max_history_size: 1000,
            alert_thresholds: PerformanceThresholds::default(),
            enable_stack_traces: true,
            slow_task_threshold_ms: 1000,
            enable_contention_monitoring: true,
            sampling_rate: 0.1, // 10% sampling
        }
    }
}

/// Performance alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Maximum average task duration in milliseconds
    pub max_avg_duration_ms: u64,

    /// Maximum task queue size
    pub max_queue_size: usize,

    /// Maximum number of blocked tasks
    pub max_blocked_tasks: usize,

    /// Maximum runtime contention percentage
    pub max_contention_percentage: f32,

    /// Maximum memory usage per task in bytes
    pub max_memory_per_task_bytes: u64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_avg_duration_ms: 5000,
            max_queue_size: 1000,
            max_blocked_tasks: 100,
            max_contention_percentage: 0.8,
            max_memory_per_task_bytes: 100 * 1024 * 1024, // 100MB
        }
    }
}

/// Task performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetrics {
    /// Task ID
    pub task_id: String,

    /// Task name/type
    pub task_name: String,

    /// Start time
    pub start_time: DateTime<Utc>,

    /// End time (if completed)
    pub end_time: Option<DateTime<Utc>>,

    /// Duration in milliseconds
    pub duration_ms: Option<u64>,

    /// Task status
    pub status: TaskStatus,

    /// Memory usage in bytes
    pub memory_usage_bytes: u64,

    /// CPU time used (estimated)
    pub cpu_time_ms: u64,

    /// Stack trace (if enabled and available)
    pub stack_trace: Option<String>,

    /// Spawn location
    pub spawn_location: Option<String>,

    /// Associated runtime worker ID
    pub worker_id: Option<usize>,

    /// Contention events
    pub contention_events: Vec<ContentionEvent>,
}

/// Task status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    /// Task is running
    Running,

    /// Task completed successfully
    Completed,

    /// Task failed with error
    Failed,

    /// Task was cancelled
    Cancelled,

    /// Task is blocked/waiting
    Blocked,

    /// Task is yielding
    Yielding,
}

/// Runtime contention event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Contention type
    pub contention_type: ContentionType,

    /// Duration of contention in microseconds
    pub duration_us: u64,

    /// Resource involved in contention
    pub resource: String,
}

/// Types of contention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentionType {
    /// Mutex contention
    Mutex,

    /// RwLock contention
    RwLock,

    /// Channel contention
    Channel,

    /// Runtime scheduler contention
    Scheduler,

    /// Memory allocation contention
    Memory,

    /// I/O contention
    IoWait,
}

/// Runtime performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeStats {
    /// Total number of active tasks
    pub active_tasks: usize,

    /// Total number of completed tasks
    pub completed_tasks: usize,

    /// Total number of failed tasks
    pub failed_tasks: usize,

    /// Average task duration
    pub avg_duration_ms: f64,

    /// P50, P95, P99 latencies
    pub latency_percentiles: LatencyPercentiles,

    /// Current queue size
    pub queue_size: usize,

    /// Number of blocked tasks
    pub blocked_tasks: usize,

    /// Runtime contention percentage
    pub contention_percentage: f32,

    /// Memory usage statistics
    pub memory_stats: TaskMemoryStats,

    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Latency percentiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Task memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMemoryStats {
    /// Total memory used by all tasks
    pub total_memory_bytes: u64,

    /// Average memory per task
    pub avg_memory_per_task_bytes: u64,

    /// Peak memory usage
    pub peak_memory_bytes: u64,

    /// Memory allocated rate (bytes/second)
    pub allocation_rate_bps: u64,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert ID
    pub id: String,

    /// Alert type
    pub alert_type: AlertType,

    /// Alert severity
    pub severity: AlertSeverity,

    /// Alert message
    pub message: String,

    /// Alert timestamp
    pub timestamp: DateTime<Utc>,

    /// Related task ID (if applicable)
    pub task_id: Option<String>,

    /// Metric value that triggered the alert
    pub metric_value: f64,

    /// Threshold that was exceeded
    pub threshold: f64,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    /// Slow task execution
    SlowTask,

    /// High memory usage
    HighMemoryUsage,

    /// Runtime contention
    Contention,

    /// Queue size exceeded
    QueueSizeExceeded,

    /// Too many blocked tasks
    TooManyBlockedTasks,

    /// Task failure rate high
    HighFailureRate,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Async task performance tracker
pub struct AsyncPerformanceTracker {
    config: AsyncPerformanceConfig,

    /// Current task metrics
    active_tasks: Arc<RwLock<HashMap<String, TaskMetrics>>>,

    /// Completed task history
    task_history: Arc<Mutex<VecDeque<TaskMetrics>>>,

    /// Runtime statistics
    runtime_stats: Arc<RwLock<RuntimeStats>>,

    /// Performance alerts
    alerts: Arc<Mutex<VecDeque<PerformanceAlert>>>,

    /// Alert broadcaster
    alert_sender: broadcast::Sender<PerformanceAlert>,

    /// Background task handles
    task_handles: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

impl AsyncPerformanceTracker {
    /// Create new async performance tracker
    pub fn new(config: AsyncPerformanceConfig) -> Self {
        let (alert_sender, _) = broadcast::channel(1000);

        let initial_stats = RuntimeStats {
            active_tasks: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            avg_duration_ms: 0.0,
            latency_percentiles: LatencyPercentiles {
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
            },
            queue_size: 0,
            blocked_tasks: 0,
            contention_percentage: 0.0,
            memory_stats: TaskMemoryStats {
                total_memory_bytes: 0,
                avg_memory_per_task_bytes: 0,
                peak_memory_bytes: 0,
                allocation_rate_bps: 0,
            },
            last_updated: Utc::now(),
        };

        Self {
            config,
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            task_history: Arc::new(Mutex::new(VecDeque::new())),
            runtime_stats: Arc::new(RwLock::new(initial_stats)),
            alerts: Arc::new(Mutex::new(VecDeque::new())),
            alert_sender,
            task_handles: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Start performance tracking
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            tracing::info!("Async performance tracking is disabled");
            return Ok(());
        }

        tracing::info!("Starting async task performance tracking");

        // Start monitoring loop
        let tracker = self.clone();
        let monitoring_task = tokio::spawn(async move {
            tracker.monitoring_loop().await;
        });

        // Start cleanup loop
        let tracker = self.clone();
        let cleanup_task = tokio::spawn(async move {
            tracker.cleanup_loop().await;
        });

        let mut handles = self.task_handles.lock().await;
        handles.push(monitoring_task);
        handles.push(cleanup_task);

        Ok(())
    }

    /// Track task start
    pub async fn track_task_start(
        &self,
        task_name: &str,
        spawn_location: Option<String>,
    ) -> String {
        if !self.should_sample() {
            return String::new();
        }

        let task_id = Uuid::new_v4().to_string();
        let metrics = TaskMetrics {
            task_id: task_id.clone(),
            task_name: task_name.to_string(),
            start_time: Utc::now(),
            end_time: None,
            duration_ms: None,
            status: TaskStatus::Running,
            memory_usage_bytes: 0, // Would need heap profiling for accurate measurement
            cpu_time_ms: 0,
            stack_trace: None,
            spawn_location,
            worker_id: None, // Would need runtime introspection
            contention_events: Vec::new(),
        };

        let mut active_tasks = self.active_tasks.write().await;
        active_tasks.insert(task_id.clone(), metrics);

        task_id
    }

    /// Track task completion
    pub async fn track_task_completion(&self, task_id: &str, status: TaskStatus) {
        if task_id.is_empty() {
            return;
        }

        let mut active_tasks = self.active_tasks.write().await;
        if let Some(mut metrics) = active_tasks.remove(task_id) {
            let end_time = Utc::now();
            let duration = end_time.signed_duration_since(metrics.start_time);

            metrics.end_time = Some(end_time);
            metrics.duration_ms = Some(duration.num_milliseconds() as u64);
            metrics.status = status.clone();

            // Check for slow task alert
            if let Some(duration_ms) = metrics.duration_ms {
                if duration_ms > self.config.slow_task_threshold_ms {
                    self.send_alert(
                        AlertType::SlowTask,
                        AlertSeverity::Warning,
                        format!("Task {} took {}ms", metrics.task_name, duration_ms),
                        Some(task_id.to_string()),
                        duration_ms as f64,
                        self.config.slow_task_threshold_ms as f64,
                    )
                    .await;
                }
            }

            // Add to history
            let mut history = self.task_history.lock().await;
            history.push_back(metrics);

            // Maintain history size limit
            while history.len() > self.config.max_history_size {
                history.pop_front();
            }
        }
    }

    /// Record contention event
    pub async fn record_contention(
        &self,
        task_id: &str,
        contention_type: ContentionType,
        duration_us: u64,
        resource: String,
    ) {
        if task_id.is_empty() {
            return;
        }

        let event = ContentionEvent {
            timestamp: Utc::now(),
            contention_type,
            duration_us,
            resource,
        };

        let mut active_tasks = self.active_tasks.write().await;
        if let Some(metrics) = active_tasks.get_mut(task_id) {
            metrics.contention_events.push(event);
        }
    }

    /// Get current runtime statistics
    pub async fn get_runtime_stats(&self) -> RuntimeStats {
        let stats = self.runtime_stats.read().await;
        stats.clone()
    }

    /// Get performance alerts
    pub async fn get_alerts(&self) -> Vec<PerformanceAlert> {
        let alerts = self.alerts.lock().await;
        alerts.iter().cloned().collect()
    }

    /// Subscribe to performance alerts
    pub fn subscribe_to_alerts(&self) -> broadcast::Receiver<PerformanceAlert> {
        self.alert_sender.subscribe()
    }

    /// Check if task should be sampled
    fn should_sample(&self) -> bool {
        fastrand::f32() < self.config.sampling_rate
    }

    /// Send performance alert
    async fn send_alert(
        &self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: String,
        task_id: Option<String>,
        metric_value: f64,
        threshold: f64,
    ) {
        let alert = PerformanceAlert {
            id: Uuid::new_v4().to_string(),
            alert_type,
            severity,
            message,
            timestamp: Utc::now(),
            task_id,
            metric_value,
            threshold,
        };

        // Add to alerts queue
        let mut alerts = self.alerts.lock().await;
        alerts.push_back(alert.clone());

        // Maintain alerts history
        while alerts.len() > 100 {
            alerts.pop_front();
        }

        // Broadcast alert
        let _ = self.alert_sender.send(alert);
    }

    /// Background monitoring loop
    async fn monitoring_loop(&self) {
        let mut interval = interval(Duration::from_secs(self.config.tracking_interval_seconds));

        loop {
            interval.tick().await;

            if let Err(e) = self.update_runtime_stats().await {
                tracing::error!("Error updating runtime stats: {}", e);
            }

            if let Err(e) = self.check_performance_thresholds().await {
                tracing::error!("Error checking performance thresholds: {}", e);
            }
        }
    }

    /// Update runtime statistics
    async fn update_runtime_stats(&self) -> Result<()> {
        let active_tasks = self.active_tasks.read().await;
        let history = self.task_history.lock().await;

        // Calculate statistics
        let active_count = active_tasks.len();
        let completed_count = history.iter().filter(|t| t.status == TaskStatus::Completed).count();
        let failed_count = history.iter().filter(|t| t.status == TaskStatus::Failed).count();
        let blocked_count =
            active_tasks.values().filter(|t| t.status == TaskStatus::Blocked).count();

        // Calculate average duration
        let durations: Vec<u64> = history.iter().filter_map(|t| t.duration_ms).collect();

        let avg_duration = if !durations.is_empty() {
            durations.iter().sum::<u64>() as f64 / durations.len() as f64
        } else {
            0.0
        };

        // Calculate percentiles
        let mut sorted_durations = durations.clone();
        sorted_durations.sort_unstable();

        let latency_percentiles = if !sorted_durations.is_empty() {
            let len = sorted_durations.len();
            LatencyPercentiles {
                p50: sorted_durations[len * 50 / 100] as f64,
                p95: sorted_durations[len * 95 / 100] as f64,
                p99: sorted_durations[len * 99 / 100] as f64,
            }
        } else {
            LatencyPercentiles {
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
            }
        };

        // Calculate memory statistics
        let total_memory: u64 = active_tasks.values().map(|t| t.memory_usage_bytes).sum();
        let avg_memory = if active_count > 0 { total_memory / active_count as u64 } else { 0 };

        // Calculate contention percentage
        let total_contention_events: usize =
            active_tasks.values().map(|t| t.contention_events.len()).sum();
        let contention_percentage = if active_count > 0 {
            (total_contention_events as f32 / active_count as f32).min(1.0)
        } else {
            0.0
        };

        let new_stats = RuntimeStats {
            active_tasks: active_count,
            completed_tasks: completed_count,
            failed_tasks: failed_count,
            avg_duration_ms: avg_duration,
            latency_percentiles,
            queue_size: active_count, // Simplified - would need runtime queue size
            blocked_tasks: blocked_count,
            contention_percentage,
            memory_stats: TaskMemoryStats {
                total_memory_bytes: total_memory,
                avg_memory_per_task_bytes: avg_memory,
                peak_memory_bytes: total_memory, // Simplified
                allocation_rate_bps: 0,          // Would need allocation tracking
            },
            last_updated: Utc::now(),
        };

        let mut stats = self.runtime_stats.write().await;
        *stats = new_stats;

        Ok(())
    }

    /// Check performance thresholds and generate alerts
    async fn check_performance_thresholds(&self) -> Result<()> {
        let stats = self.runtime_stats.read().await;
        let thresholds = &self.config.alert_thresholds;

        // Check average duration
        if stats.avg_duration_ms > thresholds.max_avg_duration_ms as f64 {
            self.send_alert(
                AlertType::SlowTask,
                AlertSeverity::Warning,
                format!(
                    "Average task duration {}ms exceeds threshold {}ms",
                    stats.avg_duration_ms, thresholds.max_avg_duration_ms
                ),
                None,
                stats.avg_duration_ms,
                thresholds.max_avg_duration_ms as f64,
            )
            .await;
        }

        // Check queue size
        if stats.queue_size > thresholds.max_queue_size {
            self.send_alert(
                AlertType::QueueSizeExceeded,
                AlertSeverity::Critical,
                format!(
                    "Queue size {} exceeds threshold {}",
                    stats.queue_size, thresholds.max_queue_size
                ),
                None,
                stats.queue_size as f64,
                thresholds.max_queue_size as f64,
            )
            .await;
        }

        // Check blocked tasks
        if stats.blocked_tasks > thresholds.max_blocked_tasks {
            self.send_alert(
                AlertType::TooManyBlockedTasks,
                AlertSeverity::Warning,
                format!(
                    "Blocked tasks {} exceeds threshold {}",
                    stats.blocked_tasks, thresholds.max_blocked_tasks
                ),
                None,
                stats.blocked_tasks as f64,
                thresholds.max_blocked_tasks as f64,
            )
            .await;
        }

        // Check contention
        if stats.contention_percentage > thresholds.max_contention_percentage {
            self.send_alert(
                AlertType::Contention,
                AlertSeverity::Critical,
                format!(
                    "Runtime contention {:.2}% exceeds threshold {:.2}%",
                    stats.contention_percentage * 100.0,
                    thresholds.max_contention_percentage * 100.0
                ),
                None,
                stats.contention_percentage as f64,
                thresholds.max_contention_percentage as f64,
            )
            .await;
        }

        Ok(())
    }

    /// Background cleanup loop
    async fn cleanup_loop(&self) {
        let mut interval = interval(Duration::from_secs(60)); // Cleanup every minute

        loop {
            interval.tick().await;

            // Clean up old completed tasks if history is too large
            let mut history = self.task_history.lock().await;
            while history.len() > self.config.max_history_size {
                history.pop_front();
            }

            // Clean up old alerts
            let mut alerts = self.alerts.lock().await;
            while alerts.len() > 100 {
                alerts.pop_front();
            }
        }
    }
}

impl Clone for AsyncPerformanceTracker {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            active_tasks: Arc::clone(&self.active_tasks),
            task_history: Arc::clone(&self.task_history),
            runtime_stats: Arc::clone(&self.runtime_stats),
            alerts: Arc::clone(&self.alerts),
            alert_sender: self.alert_sender.clone(),
            task_handles: Arc::clone(&self.task_handles),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_performance_tracker_creation() {
        let config = AsyncPerformanceConfig::default();
        let tracker = AsyncPerformanceTracker::new(config);

        let stats = tracker.get_runtime_stats().await;
        assert_eq!(stats.active_tasks, 0);
        assert_eq!(stats.completed_tasks, 0);
    }

    #[tokio::test]
    async fn test_task_tracking() {
        let config = AsyncPerformanceConfig {
            sampling_rate: 1.0, // Track all tasks for testing
            ..Default::default()
        };
        let tracker = AsyncPerformanceTracker::new(config);

        let task_id = tracker.track_task_start("test_task", None).await;
        assert!(!task_id.is_empty());

        // Simulate some work
        tokio::time::sleep(Duration::from_millis(10)).await;

        tracker.track_task_completion(&task_id, TaskStatus::Completed).await;

        let stats = tracker.get_runtime_stats().await;
        assert_eq!(stats.active_tasks, 0);
    }

    #[tokio::test]
    async fn test_contention_recording() {
        let config = AsyncPerformanceConfig {
            sampling_rate: 1.0,
            ..Default::default()
        };
        let tracker = AsyncPerformanceTracker::new(config);

        let task_id = tracker.track_task_start("test_task", None).await;

        tracker
            .record_contention(
                &task_id,
                ContentionType::Mutex,
                1000,
                "test_mutex".to_string(),
            )
            .await;

        let active_tasks = tracker.active_tasks.read().await;
        let task = active_tasks.get(&task_id).unwrap();
        assert_eq!(task.contention_events.len(), 1);
    }
}
