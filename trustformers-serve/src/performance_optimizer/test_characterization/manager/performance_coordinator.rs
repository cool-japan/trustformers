//! Performance Coordinator
//!
//! Coordinator for performance monitoring across all modules.

use super::*;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex as TokioMutex;
use tokio::task::{spawn, JoinHandle};
use tokio::time::interval;
use tracing::{error, info};

#[derive(Debug)]
pub struct PerformanceCoordinator {
    /// Performance metrics
    metrics: Arc<TokioMutex<PerformanceMetrics>>,
    /// Monitoring configuration
    config: Arc<PerformanceMonitoringConfig>,
    /// Engine statistics reference
    engine_stats: Arc<EngineStatistics>,
    /// Monitoring task handle
    monitoring_task: Arc<TokioMutex<Option<JoinHandle<()>>>>,
    /// Performance alerts
    alerts: Arc<TokioMutex<Vec<PerformanceAlert>>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

/// Performance monitoring configuration
#[derive(Debug, Clone)]
pub struct PerformanceMonitoringConfig {
    /// Monitoring interval in milliseconds
    pub monitoring_interval_ms: u64,
    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
    /// Alert configuration
    pub alert_config: AlertConfig,
    /// Metrics retention period
    pub metrics_retention_seconds: u64,
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval_ms: PERFORMANCE_MONITORING_INTERVAL_MS,
            thresholds: PerformanceThresholds::default(),
            alert_config: AlertConfig::default(),
            metrics_retention_seconds: 3600, // 1 hour
        }
    }
}

/// Performance thresholds
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum average analysis duration in milliseconds
    pub max_average_analysis_duration_ms: u64,
    /// Maximum cache miss rate (0.0 to 1.0)
    pub max_cache_miss_rate: f64,
    /// Maximum error rate (0.0 to 1.0)
    pub max_error_rate: f64,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Maximum active tasks
    pub max_active_tasks: usize,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_average_analysis_duration_ms: 30000, // 30 seconds
            max_cache_miss_rate: 0.3,                // 30%
            max_error_rate: 0.1,                     // 10%
            max_queue_size: 500,
            max_active_tasks: 50,
        }
    }
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Enable alerts
    pub enabled: bool,
    /// Alert cooldown period in seconds
    pub cooldown_seconds: u64,
    /// Maximum alerts per hour
    pub max_alerts_per_hour: usize,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cooldown_seconds: 300, // 5 minutes
            max_alerts_per_hour: 10,
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Network I/O in bytes per second
    /// Host network throughput in bytes per second, when it can be measured.
    ///
    /// Always `None` on this build: `sysinfo` is declared without its `network`
    /// feature, so no interface counters are available. Reporting `0` would be
    /// indistinguishable from an idle link.
    pub network_io_bps: Option<u64>,
    /// Disk I/O in bytes per second
    /// This process's disk throughput in bytes per second, measured over one
    /// sampling interval. `None` when the platform does not report the process.
    pub disk_io_bps: Option<u64>,
    /// Analysis throughput (analyses per minute)
    pub analysis_throughput: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Error rate
    pub error_rate: f64,
    /// Average response time in milliseconds
    pub average_response_time_ms: f64,
    /// Queue depth
    pub queue_depth: usize,
    /// Active connections
    pub active_connections: usize,
    /// Timestamp of last update
    pub last_updated: SystemTime,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            network_io_bps: None,
            disk_io_bps: None,
            analysis_throughput: 0.0,
            cache_hit_rate: 0.0,
            error_rate: 0.0,
            average_response_time_ms: 0.0,
            queue_depth: 0,
            active_connections: 0,
            last_updated: SystemTime::now(),
        }
    }
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
    pub timestamp: SystemTime,
    /// Metric value that triggered the alert
    pub metric_value: f64,
    /// Threshold that was exceeded
    pub threshold: f64,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighCpuUsage,
    HighMemoryUsage,
    HighErrorRate,
    HighCacheMissRate,
    SlowResponseTime,
    HighQueueDepth,
    ComponentFailure,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl PerformanceCoordinator {
    /// Create a new performance coordinator
    pub async fn new(
        monitoring_interval_ms: u64,
        engine_stats: Arc<EngineStatistics>,
    ) -> Result<Self> {
        let config = PerformanceMonitoringConfig {
            monitoring_interval_ms,
            ..Default::default()
        };

        Ok(Self {
            metrics: Arc::new(TokioMutex::new(PerformanceMetrics::default())),
            config: Arc::new(config),
            engine_stats,
            monitoring_task: Arc::new(TokioMutex::new(None)),
            alerts: Arc::new(TokioMutex::new(Vec::new())),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Start performance monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        let metrics = self.metrics.clone();
        let config = self.config.clone();
        let engine_stats = self.engine_stats.clone();
        let alerts = self.alerts.clone();
        let shutdown = self.shutdown.clone();

        let task = spawn(async move {
            let mut interval = interval(Duration::from_millis(config.monitoring_interval_ms));

            while !shutdown.load(Ordering::Acquire) {
                interval.tick().await;

                // Collect performance metrics
                if let Err(e) = Self::collect_metrics(&metrics, &engine_stats).await {
                    error!("Failed to collect performance metrics: {}", e);
                    continue;
                }

                // Check for performance issues and generate alerts
                if let Err(e) = Self::check_performance_thresholds(&metrics, &config, &alerts).await
                {
                    error!("Failed to check performance thresholds: {}", e);
                }
            }
        });

        let mut monitoring_task = self.monitoring_task.lock().await;
        *monitoring_task = Some(task);

        Ok(())
    }

    /// Collect performance metrics
    async fn collect_metrics(
        metrics: &Arc<TokioMutex<PerformanceMetrics>>,
        engine_stats: &Arc<EngineStatistics>,
    ) -> Result<()> {
        let mut metrics_guard = metrics.lock().await;

        // Update metrics from engine statistics
        let total_analyses = engine_stats.total_analyses.load(Ordering::Relaxed);
        let _successful_analyses = engine_stats.successful_analyses.load(Ordering::Relaxed);
        let failed_analyses = engine_stats.failed_analyses.load(Ordering::Relaxed);
        let cache_hits = engine_stats.cache_hits.load(Ordering::Relaxed);
        let cache_misses = engine_stats.cache_misses.load(Ordering::Relaxed);

        // Calculate rates
        if total_analyses > 0 {
            metrics_guard.error_rate = failed_analyses as f64 / total_analyses as f64;
        }

        let total_cache_requests = cache_hits + cache_misses;
        if total_cache_requests > 0 {
            metrics_guard.cache_hit_rate = cache_hits as f64 / total_cache_requests as f64;
        }

        metrics_guard.average_response_time_ms =
            engine_stats.average_analysis_duration_ms.load(Ordering::Relaxed) as f64;

        metrics_guard.queue_depth = engine_stats.active_analyses.load(Ordering::Relaxed);

        // Collect system metrics. Every one of these used to be a hash of the
        // wall clock dressed up as a reading; they are measurements now, and
        // the ones that cannot be measured say so.
        let host = crate::server::system_stats::measure_host_async().await;
        metrics_guard.cpu_usage_percent = host.cpu_percent;
        metrics_guard.memory_usage_mb = host.used_memory_bytes as f64 / (1024.0 * 1024.0);
        metrics_guard.network_io_bps = Self::network_io_bytes_per_second();
        metrics_guard.disk_io_bps = Self::process_disk_bytes_per_second().await;

        // Calculate throughput (analyses per minute)
        // This is a simplified calculation
        if total_analyses > 0 {
            metrics_guard.analysis_throughput = total_analyses as f64 / 60.0; // Assume 1 minute window
        }

        metrics_guard.last_updated = SystemTime::now();

        Ok(())
    }

    /// Host network throughput, in bytes per second.
    ///
    /// Always `None`: reading interface counters needs `sysinfo`'s `network`
    /// feature, and the workspace declares `sysinfo` with
    /// `default-features = false, features = ["system", "component"]`. Enabling
    /// it is a manifest change, so until then this reports the absence of a
    /// measurement rather than a number.
    fn network_io_bytes_per_second() -> Option<u64> {
        None
    }

    /// Disk throughput of this process, in bytes per second.
    ///
    /// Measured by refreshing the process twice around a fixed interval and
    /// dividing the bytes `sysinfo` attributes to it by the elapsed time.
    /// Returns `None` when the platform does not report this process.
    async fn process_disk_bytes_per_second() -> Option<u64> {
        tokio::task::spawn_blocking(|| {
            use sysinfo::{Pid, ProcessesToUpdate, System};

            const SAMPLE: Duration = Duration::from_millis(200);

            let pid = Pid::from_u32(std::process::id());
            let mut system = System::new();
            system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
            system.process(pid)?;

            std::thread::sleep(SAMPLE);
            let started = std::time::Instant::now();
            system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
            let usage = system.process(pid)?.disk_usage();
            let elapsed = started.elapsed().max(Duration::from_millis(1));

            let bytes = usage.read_bytes.saturating_add(usage.written_bytes);
            Some((bytes as f64 / elapsed.as_secs_f64()) as u64)
        })
        .await
        .ok()
        .flatten()
    }

    /// Check performance thresholds and generate alerts
    async fn check_performance_thresholds(
        metrics: &Arc<TokioMutex<PerformanceMetrics>>,
        config: &Arc<PerformanceMonitoringConfig>,
        alerts: &Arc<TokioMutex<Vec<PerformanceAlert>>>,
    ) -> Result<()> {
        let metrics_guard = metrics.lock().await;
        let mut alerts_guard = alerts.lock().await;

        // Check error rate threshold
        if metrics_guard.error_rate > config.thresholds.max_error_rate {
            let alert = PerformanceAlert {
                id: format!(
                    "error_rate_{}",
                    SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                ),
                alert_type: AlertType::HighErrorRate,
                severity: AlertSeverity::Warning,
                message: format!(
                    "High error rate detected: {:.2}%",
                    metrics_guard.error_rate * 100.0
                ),
                timestamp: SystemTime::now(),
                metric_value: metrics_guard.error_rate,
                threshold: config.thresholds.max_error_rate,
            };
            alerts_guard.push(alert);
        }

        // Check cache miss rate threshold
        let cache_miss_rate = 1.0 - metrics_guard.cache_hit_rate;
        if cache_miss_rate > config.thresholds.max_cache_miss_rate {
            let alert = PerformanceAlert {
                id: format!(
                    "cache_miss_{}",
                    SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                ),
                alert_type: AlertType::HighCacheMissRate,
                severity: AlertSeverity::Info,
                message: format!(
                    "High cache miss rate detected: {:.2}%",
                    cache_miss_rate * 100.0
                ),
                timestamp: SystemTime::now(),
                metric_value: cache_miss_rate,
                threshold: config.thresholds.max_cache_miss_rate,
            };
            alerts_guard.push(alert);
        }

        // Check response time threshold
        if metrics_guard.average_response_time_ms
            > config.thresholds.max_average_analysis_duration_ms as f64
        {
            let alert = PerformanceAlert {
                id: format!(
                    "response_time_{}",
                    SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                ),
                alert_type: AlertType::SlowResponseTime,
                severity: AlertSeverity::Warning,
                message: format!(
                    "Slow response time detected: {:.0}ms",
                    metrics_guard.average_response_time_ms
                ),
                timestamp: SystemTime::now(),
                metric_value: metrics_guard.average_response_time_ms,
                threshold: config.thresholds.max_average_analysis_duration_ms as f64,
            };
            alerts_guard.push(alert);
        }

        // Check queue depth threshold
        if metrics_guard.queue_depth > config.thresholds.max_queue_size {
            let alert = PerformanceAlert {
                id: format!(
                    "queue_depth_{}",
                    SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
                ),
                alert_type: AlertType::HighQueueDepth,
                severity: AlertSeverity::Error,
                message: format!("High queue depth detected: {}", metrics_guard.queue_depth),
                timestamp: SystemTime::now(),
                metric_value: metrics_guard.queue_depth as f64,
                threshold: config.thresholds.max_queue_size as f64,
            };
            alerts_guard.push(alert);
        }

        // Clean up old alerts
        let retention_cutoff =
            SystemTime::now() - Duration::from_secs(config.metrics_retention_seconds);
        alerts_guard.retain(|alert| alert.timestamp > retention_cutoff);

        Ok(())
    }

    /// Get current performance metrics
    pub async fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.lock().await.clone()
    }

    /// Get current alerts
    pub async fn get_alerts(&self) -> Vec<PerformanceAlert> {
        self.alerts.lock().await.clone()
    }

    /// Clear alerts
    pub async fn clear_alerts(&self) {
        let mut alerts = self.alerts.lock().await;
        alerts.clear();
    }

    /// Shutdown performance coordinator
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down PerformanceCoordinator");

        self.shutdown.store(true, Ordering::Release);

        // Cancel monitoring task
        let mut monitoring_task = self.monitoring_task.lock().await;
        if let Some(task) = monitoring_task.take() {
            task.abort();
        }

        info!("PerformanceCoordinator shutdown completed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Before 0.2.1 every one of these getters returned
    /// `DefaultHasher(SystemTime::now()) % N`, so two calls a second apart gave
    /// unrelated "measurements". These assertions pin the numbers to something
    /// the host can actually be asked about.
    #[tokio::test]
    async fn cpu_and_memory_come_from_the_host_not_the_clock() {
        let first = crate::server::system_stats::measure_host_async().await;
        let second = crate::server::system_stats::measure_host_async().await;

        assert!(
            (0.0..=100.0).contains(&first.cpu_percent),
            "CPU utilization must be a percentage, got {}",
            first.cpu_percent
        );
        assert!(first.total_memory_bytes > 0, "the host reports its RAM");
        assert!(first.used_memory_bytes > 0);
        assert!(first.used_memory_bytes <= first.total_memory_bytes);

        // The clock moved between the two samples; total RAM must not have.
        assert_eq!(
            first.total_memory_bytes, second.total_memory_bytes,
            "total memory is a property of the host, not of the current time"
        );
    }

    #[test]
    fn network_throughput_is_absent_rather_than_invented() {
        assert_eq!(
            PerformanceCoordinator::network_io_bytes_per_second(),
            None,
            "sysinfo is built without its `network` feature, so there are no \
             interface counters to read and none may be manufactured"
        );
    }

    #[tokio::test]
    async fn disk_throughput_is_a_measurement_of_this_process() {
        let rate = PerformanceCoordinator::process_disk_bytes_per_second().await;

        // The platform either reports this process or it does not; what it must
        // never do is return a number derived from the wall clock.
        if let Some(bytes_per_second) = rate {
            assert!(
                bytes_per_second < 100 * 1024 * 1024 * 1024,
                "an idle test process cannot be moving {bytes_per_second} B/s"
            );
        }
    }
}
