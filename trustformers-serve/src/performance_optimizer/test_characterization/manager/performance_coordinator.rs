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
    pub network_io_bps: u64,
    /// Disk I/O in bytes per second
    pub disk_io_bps: u64,
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
            network_io_bps: 0,
            disk_io_bps: 0,
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

        // Collect system metrics (simplified)
        metrics_guard.cpu_usage_percent = Self::get_cpu_usage().await;
        metrics_guard.memory_usage_mb = Self::get_memory_usage().await;
        metrics_guard.network_io_bps = Self::get_network_io().await;
        metrics_guard.disk_io_bps = Self::get_disk_io().await;

        // Calculate throughput (analyses per minute)
        // This is a simplified calculation
        if total_analyses > 0 {
            metrics_guard.analysis_throughput = total_analyses as f64 / 60.0; // Assume 1 minute window
        }

        metrics_guard.last_updated = SystemTime::now();

        Ok(())
    }

    /// Get CPU usage (simplified implementation)
    async fn get_cpu_usage() -> f64 {
        // In a real implementation, this would query system metrics
        // For now, return a random value between 0 and 100
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        (hasher.finish() % 100) as f64
    }

    /// Get memory usage (simplified implementation)
    async fn get_memory_usage() -> f64 {
        // In a real implementation, this would query system metrics
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        ((hasher.finish() % 8192) + 1024) as f64 // Return between 1GB and 8GB
    }

    /// Get network I/O (simplified implementation)
    async fn get_network_io() -> u64 {
        // In a real implementation, this would query system metrics
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        hasher.finish() % 1000000 // Return up to 1MB/s
    }

    /// Get disk I/O (simplified implementation)
    async fn get_disk_io() -> u64 {
        // In a real implementation, this would query system metrics
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        hasher.finish() % 500000 // Return up to 500KB/s
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
