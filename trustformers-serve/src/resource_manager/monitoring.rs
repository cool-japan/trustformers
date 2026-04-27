//! Resource monitoring and performance tracking.

// Re-export types for external access
pub use super::types::{
    DistributionEvent, ExecutionPerformanceMetrics, ExecutionState, HealthStatus, LoadMetrics,
    SystemPerformanceSnapshot, SystemResourceStatistics, WorkerStatus,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::{debug, info};

use crate::test_parallelization::ResourceMonitoringConfig;

/// Resource monitor for tracking system performance
pub struct ResourceMonitor {
    /// Configuration
    config: Arc<Mutex<ResourceMonitoringConfig>>,
    /// System metrics
    system_metrics: Arc<Mutex<SystemMetrics>>,
    /// Resource usage history
    usage_history: Arc<Mutex<Vec<ResourceUsageSnapshot>>>,
    /// Alert thresholds
    alert_thresholds: Arc<Mutex<AlertThresholds>>,
}

/// System statistics tracker
#[derive(Debug)]
pub struct SystemStatistics {
    /// Resource efficiency metrics
    efficiency_metrics: Arc<Mutex<HashMap<String, f32>>>,
    /// System performance history
    performance_history: Arc<Mutex<Vec<SystemPerformanceSnapshot>>>,
    /// Statistics collection interval
    collection_interval: Duration,
}

/// System metrics collection
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Memory utilization percentage
    pub memory_utilization: f32,
    /// GPU utilization percentage
    pub gpu_utilization: Option<f32>,
    /// Network utilization percentage
    pub network_utilization: f32,
    /// Disk utilization percentage
    pub disk_utilization: f32,
    /// Active process count
    pub active_processes: usize,
    /// System load average
    pub load_average: f32,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            gpu_utilization: None,
            network_utilization: 0.0,
            disk_utilization: 0.0,
            active_processes: 0,
            load_average: 0.0,
            last_updated: Utc::now(),
        }
    }
}

/// Resource usage snapshot
#[derive(Debug, Clone)]
pub struct ResourceUsageSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,
    /// System metrics at time of snapshot
    pub system_metrics: SystemMetrics,
    /// Resource pool utilization
    pub pool_utilization: PoolUtilization,
    /// Active test count
    pub active_tests: usize,
    /// Queue length
    pub queue_length: usize,
}

/// Resource pool utilization
#[derive(Debug, Clone)]
pub struct PoolUtilization {
    /// Network port pool utilization
    pub port_pool: f32,
    /// Temporary directory pool utilization
    pub temp_dir_pool: f32,
    /// GPU device pool utilization
    pub gpu_pool: f32,
    /// Database connection pool utilization
    pub database_pool: f32,
    /// Overall pool efficiency
    pub overall_efficiency: f32,
}

impl Default for PoolUtilization {
    fn default() -> Self {
        Self {
            port_pool: 0.0,
            temp_dir_pool: 0.0,
            gpu_pool: 0.0,
            database_pool: 0.0,
            overall_efficiency: 0.0,
        }
    }
}

/// Alert thresholds for monitoring
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// CPU utilization threshold
    pub cpu_threshold: f32,
    /// Memory utilization threshold
    pub memory_threshold: f32,
    /// GPU utilization threshold
    pub gpu_threshold: f32,
    /// Disk utilization threshold
    pub disk_threshold: f32,
    /// Queue length threshold
    pub queue_threshold: usize,
    /// Response time threshold
    pub response_time_threshold: Duration,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 0.8,
            memory_threshold: 0.85,
            gpu_threshold: 0.9,
            disk_threshold: 0.9,
            queue_threshold: 100,
            response_time_threshold: Duration::from_secs(30),
        }
    }
}

/// Worker pool for managing test execution
#[derive(Debug)]
pub struct WorkerPool {
    workers: Arc<Mutex<Vec<Worker>>>,
    capacity: usize,
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self {
            workers: Arc::new(Mutex::new(Vec::new())),
            capacity: 4, // Default to 4 workers
        }
    }
}

/// Worker representation
#[derive(Debug)]
pub struct Worker {
    id: String,
    status: WorkerStatus,
    current_task: Option<String>,
}

/// Health checker for system components
#[derive(Debug)]
pub struct HealthChecker {
    checks: Arc<Mutex<Vec<HealthCheck>>>,
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            checks: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

/// Health check record
#[derive(Debug, Clone)]
pub struct HealthCheck {
    name: String,
    status: HealthStatus,
    last_check: DateTime<Utc>,
}

/// Alert system for notifications
#[derive(Debug)]
pub struct AlertSystem {
    alerts: Arc<Mutex<Vec<Alert>>>,
}

impl Default for AlertSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl AlertSystem {
    pub fn new() -> Self {
        Self {
            alerts: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

/// Alert record
#[derive(Debug, Clone)]
pub struct Alert {
    id: String,
    severity: super::types::AlertSeverity,
    message: String,
    timestamp: DateTime<Utc>,
}

/// Performance analyzer for resource optimization
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    /// Analysis history
    analysis_history: Arc<Mutex<Vec<PerformanceAnalysis>>>,
    /// Optimization recommendations
    recommendations: Arc<Mutex<Vec<OptimizationRecommendation>>>,
}

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceAnalyzer {
    pub fn new() -> Self {
        Self {
            analysis_history: Arc::new(Mutex::new(Vec::new())),
            recommendations: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

/// Performance analysis result
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Overall efficiency score
    pub efficiency_score: f32,
    /// Bottleneck identification
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// Resource utilization trends
    pub utilization_trends: HashMap<String, f32>,
    /// Performance metrics
    pub metrics: ExecutionPerformanceMetrics,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    /// Resource type causing bottleneck
    pub resource_type: String,
    /// Severity of bottleneck
    pub severity: f32,
    /// Description of the issue
    pub description: String,
    /// Recommended action
    pub recommendation: String,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: OptimizationType,
    /// Expected impact
    pub expected_impact: f32,
    /// Implementation difficulty
    pub difficulty: f32,
    /// Priority score
    pub priority: f32,
    /// Description
    pub description: String,
}

/// Types of optimizations
#[derive(Debug, Clone)]
pub enum OptimizationType {
    /// Scale up resources
    ScaleUp,
    /// Scale down resources
    ScaleDown,
    /// Redistribute load
    RedistributeLoad,
    /// Optimize scheduling
    OptimizeScheduling,
    /// Tune configuration
    TuneConfiguration,
    /// Update algorithms
    UpdateAlgorithms,
}

impl ResourceMonitor {
    /// Create new resource monitor
    pub async fn new(config: ResourceMonitoringConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(Mutex::new(config)),
            system_metrics: Arc::new(Mutex::new(SystemMetrics::default())),
            usage_history: Arc::new(Mutex::new(Vec::new())),
            alert_thresholds: Arc::new(Mutex::new(AlertThresholds::default())),
        })
    }

    /// Update system metrics
    pub async fn update_metrics(&self, metrics: SystemMetrics) -> Result<()> {
        debug!("Updating system metrics");

        let mut system_metrics = self.system_metrics.lock();
        *system_metrics = metrics;

        // Check for threshold violations
        self.check_alert_thresholds(&*system_metrics).await?;

        Ok(())
    }

    /// Take a resource usage snapshot
    pub async fn take_snapshot(&self) -> Result<ResourceUsageSnapshot> {
        let system_metrics = (*self.system_metrics.lock()).clone();

        let snapshot = ResourceUsageSnapshot {
            timestamp: Utc::now(),
            system_metrics,
            pool_utilization: PoolUtilization::default(), // Would be calculated from actual pools
            active_tests: 0, // Would be retrieved from active test tracking
            queue_length: 0, // Would be retrieved from allocation queue
        };

        // Add to history
        let mut usage_history = self.usage_history.lock();
        usage_history.push(snapshot.clone());

        // Keep only recent history (last 1000 snapshots)
        if usage_history.len() > 1000 {
            usage_history.remove(0);
        }

        Ok(snapshot)
    }

    /// Check alert thresholds
    async fn check_alert_thresholds(&self, metrics: &SystemMetrics) -> Result<()> {
        let thresholds = self.alert_thresholds.lock();

        if metrics.cpu_utilization > thresholds.cpu_threshold {
            info!(
                "CPU utilization threshold exceeded: {:.2}%",
                metrics.cpu_utilization * 100.0
            );
        }

        if metrics.memory_utilization > thresholds.memory_threshold {
            info!(
                "Memory utilization threshold exceeded: {:.2}%",
                metrics.memory_utilization * 100.0
            );
        }

        if let Some(gpu_util) = metrics.gpu_utilization {
            if gpu_util > thresholds.gpu_threshold {
                info!(
                    "GPU utilization threshold exceeded: {:.2}%",
                    gpu_util * 100.0
                );
            }
        }

        Ok(())
    }

    /// Get current system metrics
    pub async fn get_current_metrics(&self) -> Result<SystemMetrics> {
        let system_metrics = self.system_metrics.lock();
        Ok(system_metrics.clone())
    }

    /// Get usage history
    pub async fn get_usage_history(&self) -> Result<Vec<ResourceUsageSnapshot>> {
        let usage_history = self.usage_history.lock();
        Ok(usage_history.clone())
    }

    /// Update alert thresholds
    pub async fn update_alert_thresholds(&self, thresholds: AlertThresholds) -> Result<()> {
        let mut alert_thresholds = self.alert_thresholds.lock();
        *alert_thresholds = thresholds;
        info!("Updated alert thresholds");
        Ok(())
    }
}

impl Default for SystemStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemStatistics {
    /// Create new system statistics tracker
    pub fn new() -> Self {
        Self {
            efficiency_metrics: Arc::new(Mutex::new(HashMap::new())),
            performance_history: Arc::new(Mutex::new(Vec::new())),
            collection_interval: Duration::from_secs(60),
        }
    }

    /// Calculate overall system efficiency
    pub async fn calculate_efficiency(&self) -> f32 {
        // In a real implementation, this would:
        // 1. Analyze resource utilization patterns
        // 2. Calculate efficiency based on allocation success rates
        // 3. Factor in system performance metrics
        // 4. Return a composite efficiency score

        0.75 // Placeholder value
    }

    /// Add performance snapshot
    pub async fn add_performance_snapshot(
        &self,
        snapshot: SystemPerformanceSnapshot,
    ) -> Result<()> {
        let mut performance_history = self.performance_history.lock();
        performance_history.push(snapshot);

        // Keep only recent history (last 1000 snapshots)
        if performance_history.len() > 1000 {
            performance_history.remove(0);
        }

        Ok(())
    }

    /// Get performance history
    pub async fn get_performance_history(&self) -> Result<Vec<SystemPerformanceSnapshot>> {
        let performance_history = self.performance_history.lock();
        Ok(performance_history.clone())
    }

    /// Update efficiency metric
    pub async fn update_efficiency_metric(&self, metric_name: String, value: f32) -> Result<()> {
        let mut efficiency_metrics = self.efficiency_metrics.lock();
        efficiency_metrics.insert(metric_name, value);
        Ok(())
    }

    /// Get efficiency metrics
    pub async fn get_efficiency_metrics(&self) -> Result<HashMap<String, f32>> {
        let efficiency_metrics = self.efficiency_metrics.lock();
        Ok(efficiency_metrics.clone())
    }
}

impl WorkerPool {
    /// Create new worker pool
    pub fn new(capacity: usize) -> Self {
        Self {
            workers: Arc::new(Mutex::new(Vec::new())),
            capacity,
        }
    }

    /// Add worker to pool
    pub async fn add_worker(&self, worker_id: String) -> Result<()> {
        let mut workers = self.workers.lock();

        if workers.len() < self.capacity {
            let worker = Worker {
                id: worker_id.clone(),
                status: WorkerStatus::Idle,
                current_task: None,
            };
            workers.push(worker);
            info!("Added worker to pool: {}", worker_id);
        }

        Ok(())
    }

    /// Get worker status
    pub async fn get_worker_status(&self) -> Result<HashMap<String, WorkerStatus>> {
        let workers = self.workers.lock();
        let mut status_map = HashMap::new();

        for worker in workers.iter() {
            status_map.insert(worker.id.clone(), worker.status.clone());
        }

        Ok(status_map)
    }

    /// Get available worker count
    pub async fn get_available_count(&self) -> Result<usize> {
        let workers = self.workers.lock();
        Ok(workers.iter().filter(|w| matches!(w.status, WorkerStatus::Idle)).count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_metrics_default() {
        let m = SystemMetrics::default();
        assert_eq!(m.cpu_utilization, 0.0);
        assert_eq!(m.memory_utilization, 0.0);
        assert!(m.gpu_utilization.is_none());
        assert_eq!(m.network_utilization, 0.0);
        assert_eq!(m.disk_utilization, 0.0);
        assert_eq!(m.active_processes, 0);
        assert_eq!(m.load_average, 0.0);
    }

    #[test]
    fn test_system_metrics_clone() {
        let mut m = SystemMetrics::default();
        m.cpu_utilization = 0.5;
        m.memory_utilization = 0.7;
        m.active_processes = 10;
        let c = m.clone();
        assert_eq!(c.cpu_utilization, 0.5);
        assert_eq!(c.memory_utilization, 0.7);
        assert_eq!(c.active_processes, 10);
    }

    #[test]
    fn test_pool_utilization_default() {
        let p = PoolUtilization::default();
        assert_eq!(p.port_pool, 0.0);
        assert_eq!(p.temp_dir_pool, 0.0);
        assert_eq!(p.gpu_pool, 0.0);
        assert_eq!(p.database_pool, 0.0);
        assert_eq!(p.overall_efficiency, 0.0);
    }

    #[test]
    fn test_pool_utilization_custom() {
        let p = PoolUtilization {
            port_pool: 0.3,
            temp_dir_pool: 0.5,
            gpu_pool: 0.8,
            database_pool: 0.2,
            overall_efficiency: 0.6,
        };
        assert!((p.gpu_pool - 0.8).abs() < 1e-6);
        assert!((p.overall_efficiency - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_alert_thresholds_default() {
        let t = AlertThresholds::default();
        assert!((t.cpu_threshold - 0.8).abs() < 1e-6);
        assert!((t.memory_threshold - 0.85).abs() < 1e-6);
        assert!((t.gpu_threshold - 0.9).abs() < 1e-6);
        assert!((t.disk_threshold - 0.9).abs() < 1e-6);
        assert_eq!(t.queue_threshold, 100);
        assert_eq!(t.response_time_threshold, Duration::from_secs(30));
    }

    #[test]
    fn test_alert_thresholds_clone() {
        let t = AlertThresholds::default();
        let c = t.clone();
        assert_eq!(c.queue_threshold, 100);
    }

    #[test]
    fn test_worker_pool_default() {
        let wp = WorkerPool::default();
        assert_eq!(wp.capacity, 4);
    }

    #[test]
    fn test_worker_pool_custom_capacity() {
        let wp = WorkerPool::new(8);
        assert_eq!(wp.capacity, 8);
    }

    #[tokio::test]
    async fn test_worker_pool_add_worker() {
        let wp = WorkerPool::new(3);
        let r = wp.add_worker("w-1".to_string()).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn test_worker_pool_get_available_count_empty() {
        let wp = WorkerPool::new(4);
        let count = wp.get_available_count().await.unwrap_or(99);
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_worker_pool_add_then_count() {
        let wp = WorkerPool::new(4);
        wp.add_worker("w-1".to_string()).await.unwrap_or(());
        wp.add_worker("w-2".to_string()).await.unwrap_or(());
        let count = wp.get_available_count().await.unwrap_or(0);
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_worker_pool_respects_capacity() {
        let wp = WorkerPool::new(2);
        wp.add_worker("w-1".to_string()).await.unwrap_or(());
        wp.add_worker("w-2".to_string()).await.unwrap_or(());
        wp.add_worker("w-3".to_string()).await.unwrap_or(());
        let status = wp.get_worker_status().await.unwrap_or_default();
        assert!(status.len() <= 2);
    }

    #[test]
    fn test_health_checker_default() {
        let hc = HealthChecker::default();
        let checks = hc.checks.lock();
        assert!(checks.is_empty());
    }

    #[test]
    fn test_alert_system_default() {
        let _a = AlertSystem::default();
    }

    #[test]
    fn test_performance_analyzer_default() {
        let _pa = PerformanceAnalyzer::default();
    }

    #[test]
    fn test_optimization_type_variants() {
        let variants = [
            OptimizationType::ScaleUp,
            OptimizationType::ScaleDown,
            OptimizationType::RedistributeLoad,
            OptimizationType::OptimizeScheduling,
            OptimizationType::TuneConfiguration,
            OptimizationType::UpdateAlgorithms,
        ];
        assert_eq!(variants.len(), 6);
    }

    #[test]
    fn test_optimization_recommendation_creation() {
        let rec = OptimizationRecommendation {
            recommendation_type: OptimizationType::ScaleUp,
            expected_impact: 0.8,
            difficulty: 0.4,
            priority: 0.9,
            description: "Add more GPU capacity".to_string(),
        };
        assert!((rec.expected_impact - 0.8).abs() < 1e-6);
        assert!((rec.priority - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_performance_bottleneck_creation() {
        let pb = PerformanceBottleneck {
            resource_type: "GPU".to_string(),
            severity: 0.9,
            description: "GPU saturated".to_string(),
            recommendation: "Add more GPUs".to_string(),
        };
        assert_eq!(pb.resource_type, "GPU");
        assert!((pb.severity - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_resource_usage_snapshot_creation() {
        let snap = ResourceUsageSnapshot {
            timestamp: chrono::Utc::now(),
            system_metrics: SystemMetrics::default(),
            pool_utilization: PoolUtilization::default(),
            active_tests: 5,
            queue_length: 10,
        };
        assert_eq!(snap.active_tests, 5);
        assert_eq!(snap.queue_length, 10);
    }

    #[tokio::test]
    async fn test_resource_monitor_new() {
        use crate::test_parallelization::ResourceMonitoringConfig;
        let config = ResourceMonitoringConfig::default();
        let monitor = ResourceMonitor::new(config).await;
        assert!(monitor.is_ok());
    }

    #[tokio::test]
    async fn test_resource_monitor_get_current_metrics() {
        use crate::test_parallelization::ResourceMonitoringConfig;
        let config = ResourceMonitoringConfig::default();
        let monitor = ResourceMonitor::new(config).await.unwrap_or_else(|_| panic!("failed"));
        let metrics = monitor.get_current_metrics().await.unwrap_or(SystemMetrics::default());
        assert_eq!(metrics.cpu_utilization, 0.0);
    }

    #[tokio::test]
    async fn test_resource_monitor_update_metrics() {
        use crate::test_parallelization::ResourceMonitoringConfig;
        let config = ResourceMonitoringConfig::default();
        let monitor = ResourceMonitor::new(config).await.unwrap_or_else(|_| panic!("failed"));
        let mut m = SystemMetrics::default();
        m.cpu_utilization = 0.3;
        let r = monitor.update_metrics(m).await;
        assert!(r.is_ok());
        let current = monitor.get_current_metrics().await.unwrap_or(SystemMetrics::default());
        assert!((current.cpu_utilization - 0.3).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_resource_monitor_take_snapshot() {
        use crate::test_parallelization::ResourceMonitoringConfig;
        let config = ResourceMonitoringConfig::default();
        let monitor = ResourceMonitor::new(config).await.unwrap_or_else(|_| panic!("failed"));
        let snapshot = monitor.take_snapshot().await;
        assert!(snapshot.is_ok());
    }

    #[tokio::test]
    async fn test_resource_monitor_usage_history_grows() {
        use crate::test_parallelization::ResourceMonitoringConfig;
        let config = ResourceMonitoringConfig::default();
        let monitor = ResourceMonitor::new(config).await.unwrap_or_else(|_| panic!("failed"));
        monitor.take_snapshot().await.unwrap_or_else(|_| panic!("snap failed"));
        monitor.take_snapshot().await.unwrap_or_else(|_| panic!("snap failed"));
        let history = monitor.get_usage_history().await.unwrap_or_default();
        assert_eq!(history.len(), 2);
    }

    #[tokio::test]
    async fn test_resource_monitor_update_alert_thresholds() {
        use crate::test_parallelization::ResourceMonitoringConfig;
        let config = ResourceMonitoringConfig::default();
        let monitor = ResourceMonitor::new(config).await.unwrap_or_else(|_| panic!("failed"));
        let mut thresholds = AlertThresholds::default();
        thresholds.cpu_threshold = 0.6;
        let r = monitor.update_alert_thresholds(thresholds).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn test_system_statistics_calculate_efficiency() {
        let stats = SystemStatistics::new();
        let eff = stats.calculate_efficiency().await;
        assert!((eff - 0.75).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_system_statistics_get_performance_history_empty() {
        let stats = SystemStatistics::new();
        let history = stats.get_performance_history().await.unwrap_or_default();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_system_statistics_update_efficiency_metric() {
        let stats = SystemStatistics::new();
        let r = stats.update_efficiency_metric("cpu".to_string(), 0.85).await;
        assert!(r.is_ok());
        let metrics = stats.get_efficiency_metrics().await.unwrap_or_default();
        let val = metrics.get("cpu").copied().unwrap_or(0.0);
        assert!((val - 0.85).abs() < 1e-6);
    }
}
