//! GPU resource management for test parallelization.
//!
//! This module provides comprehensive GPU resource management capabilities including
//! device allocation, performance monitoring, alert systems, and usage tracking
//! for parallel test execution in machine learning and compute-intensive workloads.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::HashMap,
    sync::Arc,
    time::Duration,
};
use tracing::{debug, info, warn, error};

use super::types::{
    GpuDeviceInfo, GpuAllocation, GpuUsageType, GpuPerformanceRequirements,
    GpuCapability, GpuDeviceStatus, GpuConstraint, GpuRealTimeMetrics,
    GpuClockSpeeds, GpuHistoricalMetric, GpuMetricType, GpuAlert, GpuAlertType,
    GpuAlertEvent, GpuAlertEventType, AlertSeverity, GpuPerformanceBenchmark,
    GpuBenchmarkType, GpuPerformanceRecord, GpuPerformanceBaseline,
    GpuPerformanceAnalysis, PerformanceTrend, PerformanceRegression,
    PerformanceRecommendation, TrendDirection, RegressionSeverity,
    RecommendationType, RecommendationComplexity, RecommendationPriority,
    GpuUsageStatistics, GpuMonitoringConfig, GpuAlertConfig, GpuAlertHandler,
    GpuPoolConfig,
};

/// GPU resource management system
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

/// GPU monitoring system for real-time device tracking
pub struct GpuMonitoringSystem {
    /// Monitoring configuration
    config: Arc<RwLock<GpuMonitoringConfig>>,
    /// Real-time metrics
    realtime_metrics: Arc<Mutex<HashMap<usize, GpuRealTimeMetrics>>>,
    /// Historical metrics
    historical_metrics: Arc<Mutex<Vec<GpuHistoricalMetric>>>,
    /// Alert system
    alert_system: Arc<GpuAlertSystem>,
    /// Monitoring enabled
    monitoring_enabled: bool,
}

/// GPU alert system for proactive monitoring
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

/// GPU performance tracker for benchmarking and analysis
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

impl GpuResourceManager {
    /// Create new GPU resource manager
    pub async fn new(config: GpuPoolConfig) -> Result<Self> {
        let available_devices = Self::discover_gpu_devices().await?;

        info!(
            "Initialized GPU resource manager with {} available devices",
            available_devices.len()
        );

        let monitoring_config = GpuMonitoringConfig {
            enable_realtime: true,
            monitoring_interval: Duration::from_secs(5),
            enable_performance_tracking: true,
            enable_alerts: true,
            alert_config: GpuAlertConfig::default(),
        };

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            available_devices: Arc::new(Mutex::new(available_devices)),
            allocated_resources: Arc::new(Mutex::new(HashMap::new())),
            monitoring_system: Arc::new(GpuMonitoringSystem::new(monitoring_config).await?),
            performance_tracker: Arc::new(GpuPerformanceTracker::new()),
            usage_stats: Arc::new(Mutex::new(GpuUsageStatistics::default())),
        })
    }

    /// Discover available GPU devices
    async fn discover_gpu_devices() -> Result<Vec<GpuDeviceInfo>> {
        let mut devices = Vec::new();

        // In a real implementation, this would use GPU discovery libraries
        // For now, create mock devices for testing
        for device_id in 0..2 {
            let device = GpuDeviceInfo {
                device_id,
                device_name: format!("Mock GPU Device {}", device_id),
                total_memory_mb: 8192, // 8GB
                available_memory_mb: 8192,
                utilization_percent: 0.0,
                capabilities: vec![
                    GpuCapability::Cuda("11.8".to_string()),
                    GpuCapability::MachineLearning(vec!["PyTorch".to_string(), "TensorFlow".to_string()]),
                ],
                status: GpuDeviceStatus::Available,
                last_updated: Utc::now(),
            };
            devices.push(device);
        }

        Ok(devices)
    }

    /// Allocate GPU devices for a test
    pub async fn allocate_devices(
        &self,
        requirements: &[GpuPerformanceRequirements],
        test_id: &str,
    ) -> Result<Vec<String>> {
        if requirements.is_empty() {
            return Ok(vec![]);
        }

        let mut available_devices = self.available_devices.lock();
        let mut allocated_resources = self.allocated_resources.lock();
        let mut usage_stats = self.usage_stats.lock();

        let mut allocated_device_ids = Vec::new();

        for req in requirements {
            // Find suitable device
            let suitable_device_index = available_devices.iter().position(|device| {
                device.status == GpuDeviceStatus::Available
                    && device.available_memory_mb >= req.min_memory_mb
                    && self.check_device_capabilities(device, req)
            });

            if let Some(index) = suitable_device_index {
                let mut device = available_devices.remove(index);
                device.status = GpuDeviceStatus::Busy;
                device.available_memory_mb -= req.min_memory_mb;

                let allocation_id = format!("gpu_{}_{}", device.device_id, test_id);
                let allocation = GpuAllocation {
                    device: device.clone(),
                    test_id: test_id.to_string(),
                    memory_allocated_mb: req.min_memory_mb,
                    allocated_at: Utc::now(),
                    expected_release: None,
                    usage_type: GpuUsageType::Custom("test".to_string()),
                    performance_requirements: req.clone(),
                };

                allocated_resources.insert(allocation_id.clone(), allocation);
                allocated_device_ids.push(allocation_id);

                // Update device in available list (now marked as busy)
                available_devices.push(device);
            } else {
                // Rollback partial allocations
                for alloc_id in &allocated_device_ids {
                    if let Some(allocation) = allocated_resources.remove(alloc_id) {
                        self.return_device_to_available_pool(allocation.device, &mut available_devices);
                    }
                }
                return Err(anyhow::anyhow!(
                    "No suitable GPU device found for requirements: min_memory_mb={}",
                    req.min_memory_mb
                ));
            }
        }

        // Update statistics
        usage_stats.total_allocations += allocated_device_ids.len() as u64;
        usage_stats.currently_allocated = allocated_resources.len();
        usage_stats.peak_usage = usage_stats.peak_usage.max(allocated_resources.len());

        info!(
            "Allocated {} GPU devices for test {}: {:?}",
            allocated_device_ids.len(),
            test_id,
            allocated_device_ids
        );

        Ok(allocated_device_ids)
    }

    /// Check if device meets capability requirements
    fn check_device_capabilities(&self, device: &GpuDeviceInfo, requirements: &GpuPerformanceRequirements) -> bool {
        // Check required frameworks
        for required_framework in &requirements.required_frameworks {
            let has_framework = device.capabilities.iter().any(|capability| {
                matches!(capability, GpuCapability::MachineLearning(frameworks) if frameworks.contains(required_framework))
            });
            if !has_framework {
                return false;
            }
        }

        // Check constraints
        for constraint in &requirements.constraints {
            if !self.check_constraint(device, constraint) {
                return false;
            }
        }

        true
    }

    /// Check individual constraint
    fn check_constraint(&self, device: &GpuDeviceInfo, constraint: &GpuConstraint) -> bool {
        match &constraint.constraint_type {
            super::types::GpuConstraintType::MaxMemoryUsage => {
                let memory_usage_ratio = (device.total_memory_mb - device.available_memory_mb) as f64 / device.total_memory_mb as f64;
                memory_usage_ratio <= constraint.value
            }
            super::types::GpuConstraintType::MaxUtilization => {
                device.utilization_percent as f64 <= constraint.value
            }
            super::types::GpuConstraintType::MinPerformance => {
                // Simplified performance check - in practice, would use benchmarks
                true
            }
            _ => true, // Other constraints not implemented in mock
        }
    }

    /// Return device to available pool
    fn return_device_to_available_pool(&self, mut device: GpuDeviceInfo, available_devices: &mut Vec<GpuDeviceInfo>) {
        device.status = GpuDeviceStatus::Available;

        // Find and update existing device or add new one
        if let Some(existing_device) = available_devices.iter_mut().find(|d| d.device_id == device.device_id) {
            *existing_device = device;
        } else {
            available_devices.push(device);
        }
    }

    /// Deallocate GPU device
    pub async fn deallocate_device(&self, allocation_id: &str) -> Result<()> {
        let mut available_devices = self.available_devices.lock();
        let mut allocated_resources = self.allocated_resources.lock();
        let mut usage_stats = self.usage_stats.lock();

        if let Some(allocation) = allocated_resources.remove(allocation_id) {
            // Return memory to device
            let mut device = allocation.device;
            device.available_memory_mb += allocation.memory_allocated_mb;
            device.status = GpuDeviceStatus::Available;

            self.return_device_to_available_pool(device, &mut available_devices);

            usage_stats.currently_allocated = allocated_resources.len();

            // Update GPU hours usage
            let usage_duration = allocation.allocated_at.signed_duration_since(Utc::now()).abs();
            let hours = usage_duration.num_seconds() as f64 / 3600.0;
            usage_stats.total_gpu_hours += hours;

            info!("Deallocated GPU device for allocation: {}", allocation_id);
            Ok(())
        } else {
            warn!("Attempted to deallocate GPU allocation {} that was not found", allocation_id);
            Err(anyhow::anyhow!("GPU allocation {} was not found", allocation_id))
        }
    }

    /// Deallocate all GPU devices for a specific test
    pub async fn deallocate_devices_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating GPU devices for test: {}", test_id);

        let mut available_devices = self.available_devices.lock();
        let mut allocated_resources = self.allocated_resources.lock();
        let mut usage_stats = self.usage_stats.lock();

        let mut deallocated_devices = Vec::new();

        // Find and collect devices to deallocate
        allocated_resources.retain(|allocation_id, allocation| {
            if allocation.test_id == test_id {
                // Return device to available pool
                let mut device = allocation.device.clone();
                device.available_memory_mb += allocation.memory_allocated_mb;
                device.status = GpuDeviceStatus::Available;

                self.return_device_to_available_pool(device, &mut available_devices);
                deallocated_devices.push(allocation_id.clone());

                // Update GPU hours usage
                let usage_duration = allocation.allocated_at.signed_duration_since(Utc::now()).abs();
                let hours = usage_duration.num_seconds() as f64 / 3600.0;
                usage_stats.total_gpu_hours += hours;

                false // Remove from allocated_resources
            } else {
                true // Keep in allocated_resources
            }
        });

        usage_stats.currently_allocated = allocated_resources.len();

        if !deallocated_devices.is_empty() {
            info!(
                "Released {} GPU devices for test {}: {:?}",
                deallocated_devices.len(),
                test_id,
                deallocated_devices
            );
        }

        Ok(())
    }

    /// Check if requested GPU devices are available
    pub async fn check_availability(&self, requirements: &[GpuPerformanceRequirements]) -> Result<bool> {
        let available_devices = self.available_devices.lock();

        for req in requirements {
            let has_suitable_device = available_devices.iter().any(|device| {
                device.status == GpuDeviceStatus::Available
                    && device.available_memory_mb >= req.min_memory_mb
                    && self.check_device_capabilities(device, req)
            });

            if !has_suitable_device {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Get current GPU usage statistics
    pub async fn get_statistics(&self) -> Result<GpuUsageStatistics> {
        let stats = self.usage_stats.lock();
        Ok(stats.clone())
    }

    /// Get available GPU devices
    pub async fn get_available_devices(&self) -> Vec<GpuDeviceInfo> {
        let available_devices = self.available_devices.lock();
        available_devices.iter()
            .filter(|device| device.status == GpuDeviceStatus::Available)
            .cloned()
            .collect()
    }

    /// Get allocated GPU resources
    pub async fn get_allocated_resources(&self) -> HashMap<String, GpuAllocation> {
        let allocated_resources = self.allocated_resources.lock();
        allocated_resources.clone()
    }

    /// Get GPU device information by ID
    pub async fn get_device_info(&self, device_id: usize) -> Option<GpuDeviceInfo> {
        let available_devices = self.available_devices.lock();
        available_devices.iter().find(|device| device.device_id == device_id).cloned()
    }

    /// Update GPU device status
    pub async fn update_device_status(&self, device_id: usize, status: GpuDeviceStatus) -> Result<()> {
        let mut available_devices = self.available_devices.lock();

        if let Some(device) = available_devices.iter_mut().find(|device| device.device_id == device_id) {
            device.status = status;
            device.last_updated = Utc::now();
            info!("Updated GPU device {} status to {:?}", device_id, device.status);
            Ok(())
        } else {
            Err(anyhow::anyhow!("GPU device {} not found", device_id))
        }
    }

    /// Get GPU utilization percentage
    pub async fn get_utilization(&self) -> f32 {
        let available_count = self.get_available_devices().await.len();
        let allocated_count = {
            let allocated_resources = self.allocated_resources.lock();
            allocated_resources.len()
        };
        let total_count = available_count + allocated_count;

        if total_count == 0 {
            0.0
        } else {
            allocated_count as f32 / total_count as f32
        }
    }

    /// Start monitoring system
    pub async fn start_monitoring(&self) -> Result<()> {
        self.monitoring_system.start_monitoring().await
    }

    /// Stop monitoring system
    pub async fn stop_monitoring(&self) -> Result<()> {
        self.monitoring_system.stop_monitoring().await
    }

    /// Get real-time GPU metrics
    pub async fn get_realtime_metrics(&self) -> HashMap<usize, GpuRealTimeMetrics> {
        self.monitoring_system.get_realtime_metrics().await
    }

    /// Run performance benchmark
    pub async fn run_benchmark(&self, device_id: usize, benchmark_type: GpuBenchmarkType) -> Result<GpuPerformanceBenchmark> {
        self.performance_tracker.run_benchmark(device_id, benchmark_type).await
    }

    /// Get performance analysis
    pub async fn get_performance_analysis(&self) -> GpuPerformanceAnalysis {
        self.performance_tracker.get_analysis().await
    }

    /// Generate GPU allocation report
    pub async fn generate_allocation_report(&self) -> String {
        let stats = self.get_statistics().await.unwrap_or_default();
        let available_devices = self.get_available_devices().await;
        let allocated_count = {
            let allocated_resources = self.allocated_resources.lock();
            allocated_resources.len()
        };
        let utilization = self.get_utilization().await;

        format!(
            "GPU Allocation Report:\n\
             - Available devices: {}\n\
             - Allocated devices: {}\n\
             - Total allocations: {}\n\
             - Peak usage: {}\n\
             - Current utilization: {:.1}%\n\
             - Total GPU hours: {:.2}\n\
             - Usage efficiency: {:.1}%",
            available_devices.len(),
            allocated_count,
            stats.total_allocations,
            stats.peak_usage,
            utilization * 100.0,
            stats.total_gpu_hours,
            stats.efficiency * 100.0
        )
    }
}

impl GpuMonitoringSystem {
    /// Create new GPU monitoring system
    pub async fn new(config: GpuMonitoringConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(config.clone())),
            realtime_metrics: Arc::new(Mutex::new(HashMap::new())),
            historical_metrics: Arc::new(Mutex::new(Vec::new())),
            alert_system: Arc::new(GpuAlertSystem::new(config.alert_config).await?),
            monitoring_enabled: false,
        })
    }

    /// Start monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        info!("Starting GPU monitoring system");
        // In a real implementation, this would start background monitoring threads
        Ok(())
    }

    /// Stop monitoring
    pub async fn stop_monitoring(&self) -> Result<()> {
        info!("Stopping GPU monitoring system");
        Ok(())
    }

    /// Get real-time metrics
    pub async fn get_realtime_metrics(&self) -> HashMap<usize, GpuRealTimeMetrics> {
        let metrics = self.realtime_metrics.lock();
        metrics.clone()
    }

    /// Update real-time metrics
    pub async fn update_metrics(&self, device_id: usize, metrics: GpuRealTimeMetrics) -> Result<()> {
        let mut realtime_metrics = self.realtime_metrics.lock();
        realtime_metrics.insert(device_id, metrics.clone());

        // Check for alerts
        self.alert_system.check_metrics_for_alerts(device_id, &metrics).await?;

        // Add to historical data
        let mut historical_metrics = self.historical_metrics.lock();
        historical_metrics.push(GpuHistoricalMetric {
            device_id,
            metric_type: GpuMetricType::Utilization,
            value: metrics.utilization_percent as f64,
            timestamp: metrics.timestamp,
        });

        // Limit historical data size
        if historical_metrics.len() > 10000 {
            historical_metrics.remove(0);
        }

        Ok(())
    }

    /// Get historical metrics
    pub async fn get_historical_metrics(&self, device_id: Option<usize>) -> Vec<GpuHistoricalMetric> {
        let historical_metrics = self.historical_metrics.lock();

        match device_id {
            Some(id) => historical_metrics.iter()
                .filter(|metric| metric.device_id == id)
                .cloned()
                .collect(),
            None => historical_metrics.clone(),
        }
    }
}

impl GpuAlertSystem {
    /// Create new GPU alert system
    pub async fn new(config: GpuAlertConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            active_alerts: Arc::new(Mutex::new(HashMap::new())),
            alert_history: Arc::new(Mutex::new(Vec::new())),
            alert_handlers: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Check metrics for alert conditions
    pub async fn check_metrics_for_alerts(&self, device_id: usize, metrics: &GpuRealTimeMetrics) -> Result<()> {
        let config = self.config.read();
        let mut alerts_to_trigger = Vec::new();

        // Check temperature alerts
        if config.enable_temperature_alerts {
            if metrics.temperature_celsius >= config.thresholds.temperature_critical {
                alerts_to_trigger.push(self.create_alert(
                    device_id,
                    GpuAlertType::HighTemperature,
                    AlertSeverity::Critical,
                    format!("Critical temperature: {:.1}°C", metrics.temperature_celsius),
                    metrics.temperature_celsius as f64,
                    config.thresholds.temperature_critical as f64,
                ));
            } else if metrics.temperature_celsius >= config.thresholds.temperature_warning {
                alerts_to_trigger.push(self.create_alert(
                    device_id,
                    GpuAlertType::HighTemperature,
                    AlertSeverity::Warning,
                    format!("High temperature: {:.1}°C", metrics.temperature_celsius),
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
                    format!("Critical utilization: {:.1}%", metrics.utilization_percent),
                    metrics.utilization_percent as f64,
                    config.thresholds.utilization_critical_percent as f64,
                ));
            } else if metrics.utilization_percent >= config.thresholds.utilization_warning_percent {
                alerts_to_trigger.push(self.create_alert(
                    device_id,
                    GpuAlertType::HighUtilization,
                    AlertSeverity::Warning,
                    format!("High utilization: {:.1}%", metrics.utilization_percent),
                    metrics.utilization_percent as f64,
                    config.thresholds.utilization_warning_percent as f64,
                ));
            }
        }

        // Trigger alerts
        for alert in alerts_to_trigger {
            self.trigger_alert(alert).await?;
        }

        Ok(())
    }

    /// Create alert
    fn create_alert(
        &self,
        device_id: usize,
        alert_type: GpuAlertType,
        severity: AlertSeverity,
        message: String,
        current_value: f64,
        threshold_value: f64,
    ) -> GpuAlert {
        GpuAlert {
            alert_id: format!("alert_{}_{}", device_id, Utc::now().timestamp_millis()),
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

    /// Trigger alert
    pub async fn trigger_alert(&self, alert: GpuAlert) -> Result<()> {
        let alert_id = alert.alert_id.clone();

        // Add to active alerts
        {
            let mut active_alerts = self.active_alerts.lock();
            active_alerts.insert(alert_id.clone(), alert.clone());
        }

        // Add to history
        {
            let mut alert_history = self.alert_history.lock();
            alert_history.push(GpuAlertEvent {
                timestamp: Utc::now(),
                event_type: GpuAlertEventType::Triggered,
                alert: alert.clone(),
                details: HashMap::new(),
            });

            // Limit history size
            if alert_history.len() > 1000 {
                alert_history.remove(0);
            }
        }

        // Notify handlers
        {
            let alert_handlers = self.alert_handlers.lock();
            for handler in alert_handlers.iter() {
                if handler.can_handle(&alert.alert_type) {
                    if let Err(e) = handler.handle_alert(&alert) {
                        error!("Alert handler {} failed: {}", handler.name(), e);
                    }
                }
            }
        }

        warn!("GPU Alert triggered: {} - {}", alert_id, alert.message);
        Ok(())
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> HashMap<String, GpuAlert> {
        let active_alerts = self.active_alerts.lock();
        active_alerts.clone()
    }

    /// Acknowledge alert
    pub async fn acknowledge_alert(&self, alert_id: &str) -> Result<()> {
        let mut active_alerts = self.active_alerts.lock();

        if let Some(alert) = active_alerts.get_mut(alert_id) {
            alert.acknowledged = true;
            info!("Alert {} acknowledged", alert_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Alert {} not found", alert_id))
        }
    }
}

impl GpuPerformanceTracker {
    /// Create new GPU performance tracker
    pub fn new() -> Self {
        Self {
            benchmarks: Arc::new(Mutex::new(HashMap::new())),
            performance_history: Arc::new(Mutex::new(HashMap::new())),
            baselines: Arc::new(Mutex::new(HashMap::new())),
            analysis: Arc::new(Mutex::new(GpuPerformanceAnalysis::default())),
        }
    }

    /// Run performance benchmark
    pub async fn run_benchmark(&self, device_id: usize, benchmark_type: GpuBenchmarkType) -> Result<GpuPerformanceBenchmark> {
        let start_time = Utc::now();

        // Simulate benchmark execution
        let score = match benchmark_type {
            GpuBenchmarkType::Compute => 1000.0 + (device_id as f64 * 100.0),
            GpuBenchmarkType::MemoryBandwidth => 500.0 + (device_id as f64 * 50.0),
            GpuBenchmarkType::MatrixOperations => 2000.0 + (device_id as f64 * 200.0),
            GpuBenchmarkType::MLInference => 800.0 + (device_id as f64 * 80.0),
            GpuBenchmarkType::MLTraining => 600.0 + (device_id as f64 * 60.0),
            GpuBenchmarkType::Custom(_) => 1000.0,
        };

        let execution_time = Duration::from_millis(1000 + (device_id as u64 * 100));

        let benchmark = GpuPerformanceBenchmark {
            name: format!("{:?} Benchmark", benchmark_type),
            device_id,
            benchmark_type,
            score,
            execution_time,
            timestamp: start_time,
            parameters: HashMap::new(),
        };

        // Store benchmark result
        {
            let mut benchmarks = self.benchmarks.lock();
            benchmarks.insert(device_id, benchmark.clone());
        }

        info!("Completed GPU benchmark for device {}: score {:.2}", device_id, score);
        Ok(benchmark)
    }

    /// Get performance analysis
    pub async fn get_analysis(&self) -> GpuPerformanceAnalysis {
        let analysis = self.analysis.lock();
        analysis.clone()
    }

    /// Update performance baseline
    pub async fn update_baseline(&self, device_id: usize, metrics: HashMap<String, f64>) -> Result<()> {
        let mut baselines = self.baselines.lock();

        let baseline = GpuPerformanceBaseline {
            device_id,
            baseline_metrics: metrics,
            established_at: Utc::now(),
            sample_count: 1,
            confidence_level: 0.8,
        };

        baselines.insert(device_id, baseline);
        info!("Updated performance baseline for GPU device {}", device_id);
        Ok(())
    }
}

impl Default for GpuPerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}