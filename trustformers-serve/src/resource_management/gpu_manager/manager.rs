//! Main GPU Resource Manager Implementation
//!
//! This module provides the core GpuResourceManager that orchestrates all GPU management
//! operations including device allocation, monitoring system coordination, alert management,
//! performance tracking, health monitoring, and load balancing.
//!
//! The GpuResourceManager serves as the central coordinator that brings together all the
//! specialized GPU management components to provide a unified interface for GPU resource
//! management in the TrustformeRS framework.

use anyhow::Context;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{
    sync::broadcast,
    task::JoinHandle,
    time::{interval, sleep},
};
use tracing::{debug, error, info, instrument, warn};

// Import types and specialized components
use super::types::*;
use super::{
    GpuAlertSystem, GpuHealthMonitor, GpuLoadBalancer, GpuMonitoringSystem, GpuPerformanceTracker,
};
// Import health_monitor's GpuHealthStatus which has more fields
use super::health_monitor::GpuHealthStatus as HealthMonitorGpuHealthStatus;

// Import config types from resource_management::types for components that expect them
// GpuMonitoringSystem uses resource_management::types
use crate::resource_management::types::{
    GpuAlertConfig as ResourceGpuAlertConfig, GpuClockSpeeds as ResourceGpuClockSpeeds,
    GpuMonitoringConfig, GpuRealTimeMetrics,
};

// GpuManagerError and GpuResult are imported from super::types::*

/// Comprehensive GPU resource management system
///
/// This is the main entry point for all GPU-related operations in the system.
/// It coordinates device discovery, allocation, monitoring, and health management.
///
/// The GpuResourceManager orchestrates multiple specialized components:
/// - **Monitoring System**: Real-time metrics collection and device monitoring
/// - **Alert System**: Proactive monitoring with configurable alerts for hardware issues
/// - **Performance Tracker**: Benchmarking, performance analysis, and baseline establishment
/// - **Health Monitor**: Device health monitoring with failure detection and recovery
/// - **Load Balancer**: Intelligent distribution of workloads across available devices
#[derive(Debug)]
pub struct GpuResourceManager {
    /// Configuration settings
    config: Arc<RwLock<GpuPoolConfig>>,

    /// Available GPU devices catalog
    available_devices: Arc<RwLock<HashMap<usize, GpuDeviceInfo>>>,

    /// Currently allocated GPU resources
    allocated_resources: Arc<RwLock<HashMap<String, GpuAllocation>>>,

    /// GPU monitoring system for real-time tracking
    monitoring_system: Arc<GpuMonitoringSystem>,

    /// GPU alert system for proactive health monitoring
    alert_system: Arc<GpuAlertSystem>,

    /// Performance tracking and benchmarking system
    performance_tracker: Arc<GpuPerformanceTracker>,

    /// Usage statistics and analytics
    usage_stats: Arc<RwLock<GpuUsageStatistics>>,

    /// Device health monitor
    health_monitor: Arc<GpuHealthMonitor>,

    /// Load balancer for optimal device distribution
    load_balancer: Arc<GpuLoadBalancer>,

    /// Background task handles
    background_tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,

    /// Shutdown signal for graceful cleanup
    shutdown_sender: broadcast::Sender<()>,

    /// System running state
    running: Arc<AtomicBool>,

    /// Total operation counter
    operation_counter: Arc<AtomicU64>,
}

impl GpuResourceManager {
    /// Create a new GPU resource manager with the specified configuration
    ///
    /// This performs comprehensive system initialization including:
    /// - Device discovery and capability detection
    /// - Monitoring system setup
    /// - Alert system configuration
    /// - Background task initialization
    ///
    /// # Arguments
    ///
    /// * `config` - GPU pool configuration parameters
    ///
    /// # Returns
    ///
    /// A configured GPU resource manager ready for operation
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - GPU device discovery fails
    /// - Monitoring system initialization fails
    /// - Alert system setup fails
    #[instrument(skip(config))]
    pub async fn new(config: GpuPoolConfig) -> GpuResult<Self> {
        info!("Initializing GPU resource manager");

        // Validate configuration
        Self::validate_config(&config)?;

        // Discover available GPU devices
        let available_devices = Self::discover_gpu_devices(&config).await?;
        info!("Discovered {} GPU devices", available_devices.len());

        // Initialize monitoring system
        let monitoring_config = GpuMonitoringConfig {
            monitoring_interval: Duration::from_secs(config.monitoring_interval_secs),
            retention_period: Duration::from_secs(3600),
            real_time_monitoring: config.enable_monitoring,
            alert_thresholds: Default::default(),
            monitored_metrics: vec![],
            alert_config: ResourceGpuAlertConfig::default(),
            enable_alerts: true,
            enable_performance_tracking: config.enable_performance_tracking,
        };

        let monitoring_system = Arc::new(
            GpuMonitoringSystem::new(monitoring_config)
                .await
                .context("Failed to initialize monitoring system")?,
        );

        // Initialize alert system
        let alert_system = Arc::new(
            GpuAlertSystem::new(GpuAlertConfig::default())
                .await
                .context("Failed to initialize alert system")?,
        );

        // Initialize performance tracker
        let performance_tracker = Arc::new(GpuPerformanceTracker::new());

        // Initialize health monitor
        let health_monitor = Arc::new(GpuHealthMonitor::new());

        // Initialize load balancer
        let load_balancer = Arc::new(GpuLoadBalancer::new());

        // Set up shutdown channel
        let (shutdown_sender, _) = broadcast::channel(1);

        let manager = Self {
            config: Arc::new(RwLock::new(config)),
            available_devices: Arc::new(RwLock::new(available_devices)),
            allocated_resources: Arc::new(RwLock::new(HashMap::new())),
            monitoring_system,
            alert_system,
            performance_tracker,
            usage_stats: Arc::new(RwLock::new(GpuUsageStatistics::default())),
            health_monitor,
            load_balancer,
            background_tasks: Arc::new(Mutex::new(Vec::new())),
            shutdown_sender,
            running: Arc::new(AtomicBool::new(false)),
            operation_counter: Arc::new(AtomicU64::new(0)),
        };

        info!("GPU resource manager initialized successfully");
        Ok(manager)
    }

    /// Validate configuration parameters
    fn validate_config(config: &GpuPoolConfig) -> GpuResult<()> {
        if config.max_devices == 0 {
            return Err(GpuManagerError::ConfigurationError {
                field: "max_devices".to_string(),
                message: "Must be greater than 0".to_string(),
            });
        }

        if config.memory_threshold < 0.0 || config.memory_threshold > 1.0 {
            return Err(GpuManagerError::ConfigurationError {
                field: "memory_threshold".to_string(),
                message: "Must be between 0.0 and 1.0".to_string(),
            });
        }

        if config.temperature_threshold < 0.0 || config.temperature_threshold > 150.0 {
            return Err(GpuManagerError::ConfigurationError {
                field: "temperature_threshold".to_string(),
                message: "Must be between 0.0 and 150.0 Celsius".to_string(),
            });
        }

        Ok(())
    }

    /// Discover and catalog available GPU devices
    ///
    /// This performs comprehensive device discovery including:
    /// - Hardware detection via multiple APIs (CUDA, OpenCL, etc.)
    /// - Capability assessment and compatibility checking
    /// - Driver version verification
    /// - Initial health assessment
    #[instrument(skip(config))]
    async fn discover_gpu_devices(
        config: &GpuPoolConfig,
    ) -> GpuResult<HashMap<usize, GpuDeviceInfo>> {
        let mut devices = HashMap::new();

        info!("Starting GPU device discovery");

        // In a real implementation, this would use multiple GPU discovery methods:
        // - NVIDIA Management Library (NVML) for NVIDIA GPUs
        // - ROCm for AMD GPUs
        // - OpenCL for general GPU detection
        // - Vulkan API for graphics capabilities

        // For this implementation, we'll create mock devices for testing
        // This allows the system to work without actual GPU hardware
        let device_count = std::cmp::min(config.max_devices, 4); // Limit for testing

        for device_id in 0..device_count {
            let device = Self::create_mock_device(device_id).await?;

            // Perform initial capability assessment
            if Self::assess_device_health(&device).await? {
                info!(
                    "Discovered GPU device {}: {} with {}MB memory",
                    device.device_id, device.device_name, device.total_memory_mb
                );
                devices.insert(device_id, device);
            } else {
                warn!(
                    "GPU device {} failed health assessment, skipping",
                    device_id
                );
            }
        }

        // TODO: In production, add real GPU discovery:
        // - Use nvidia-ml-py for NVIDIA GPU detection
        // - Use ROCm APIs for AMD GPU detection
        // - Use OpenCL for cross-vendor detection
        // - Implement driver compatibility checking

        info!(
            "GPU device discovery completed, found {} healthy devices",
            devices.len()
        );
        Ok(devices)
    }

    /// Create a mock GPU device for testing purposes
    async fn create_mock_device(device_id: usize) -> GpuResult<GpuDeviceInfo> {
        // Simulate different GPU types and capabilities
        let (device_name, total_memory_mb, capabilities) = match device_id {
            0 => (
                "NVIDIA GeForce RTX 4090".to_string(),
                24576, // 24GB
                vec![
                    GpuCapability::Cuda("12.0".to_string()),
                    GpuCapability::MachineLearning(vec![
                        "PyTorch".to_string(),
                        "TensorFlow".to_string(),
                        "JAX".to_string(),
                    ]),
                    GpuCapability::Vulkan("1.3".to_string()),
                ],
            ),
            1 => (
                "NVIDIA Tesla V100".to_string(),
                32768, // 32GB
                vec![
                    GpuCapability::Cuda("11.8".to_string()),
                    GpuCapability::MachineLearning(vec![
                        "PyTorch".to_string(),
                        "TensorFlow".to_string(),
                    ]),
                ],
            ),
            2 => (
                "AMD Radeon RX 7900 XTX".to_string(),
                24576, // 24GB
                vec![
                    GpuCapability::OpenCl("3.0".to_string()),
                    GpuCapability::Vulkan("1.3".to_string()),
                    GpuCapability::MachineLearning(vec!["PyTorch".to_string()]),
                ],
            ),
            _ => (
                format!("Generic GPU Device {}", device_id),
                8192, // 8GB
                vec![
                    GpuCapability::OpenCl("2.0".to_string()),
                    GpuCapability::Vulkan("1.2".to_string()),
                ],
            ),
        };

        // Simulate device discovery delay
        sleep(Duration::from_millis(100)).await;

        Ok(GpuDeviceInfo {
            device_id,
            device_name,
            total_memory_mb,
            available_memory_mb: total_memory_mb, // Initially all memory available
            utilization_percent: 0.0,
            capabilities,
            status: GpuDeviceStatus::Available,
            last_updated: Utc::now(),
        })
    }

    /// Assess device health and compatibility
    async fn assess_device_health(device: &GpuDeviceInfo) -> GpuResult<bool> {
        // In a real implementation, this would:
        // - Check driver compatibility
        // - Verify device functionality with test operations
        // - Check temperature and power status
        // - Validate memory integrity
        // - Test basic compute operations

        // For mock implementation, randomly simulate some devices being unhealthy
        let health_score = (device.device_id as f32 * 17.0) % 1.0;
        let is_healthy = health_score > 0.1; // 90% of devices are healthy

        if is_healthy {
            debug!("Device {} passed health assessment", device.device_id);
        } else {
            warn!("Device {} failed health assessment", device.device_id);
        }

        Ok(is_healthy)
    }

    /// Start all monitoring and background systems
    ///
    /// This initiates:
    /// - Real-time device monitoring
    /// - Health monitoring and failure detection
    /// - Performance tracking and benchmarking
    /// - Alert system activation
    /// - Load balancing optimization
    #[instrument(skip(self))]
    pub async fn start_monitoring(&self) -> GpuResult<()> {
        if self.running.load(Ordering::Acquire) {
            warn!("GPU monitoring is already running");
            return Ok(());
        }

        info!("Starting GPU monitoring systems");

        // Start monitoring system
        self.monitoring_system
            .start_monitoring()
            .await
            .context("Failed to start monitoring system")?;

        // Start alert system
        self.alert_system.start().await.context("Failed to start alert system")?;

        // Start health monitoring
        self.health_monitor
            .start_monitoring(
                self.available_devices.clone(),
                self.shutdown_sender.subscribe(),
            )
            .await
            .map_err(|e| GpuManagerError::MonitoringError {
                source: anyhow::anyhow!("Failed to start health monitoring: {}", e),
            })?;

        // Start performance tracking
        self.start_performance_tracking().await?;

        // Start metrics collection
        self.start_metrics_collection().await?;

        self.running.store(true, Ordering::Release);
        info!("GPU monitoring systems started successfully");

        Ok(())
    }

    /// Stop all monitoring and background systems
    #[instrument(skip(self))]
    pub async fn stop_monitoring(&self) -> GpuResult<()> {
        if !self.running.load(Ordering::Acquire) {
            warn!("GPU monitoring is not running");
            return Ok(());
        }

        info!("Stopping GPU monitoring systems");

        // Signal shutdown to all background tasks
        let _ = self.shutdown_sender.send(());

        // Stop monitoring system
        self.monitoring_system
            .stop_monitoring()
            .await
            .context("Failed to stop monitoring system")?;

        // Stop alert system
        self.alert_system.stop().await.context("Failed to stop alert system")?;

        // Wait for background tasks to complete
        let mut tasks = self.background_tasks.lock();
        for task in tasks.drain(..) {
            if !task.is_finished() {
                task.abort();
            }
        }

        self.running.store(false, Ordering::Release);
        info!("GPU monitoring systems stopped successfully");

        Ok(())
    }

    /// Start performance tracking background task
    async fn start_performance_tracking(&self) -> GpuResult<()> {
        let performance_tracker = self.performance_tracker.clone();
        let available_devices = self.available_devices.clone();
        let mut shutdown_rx = self.shutdown_sender.subscribe();

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(300)); // Run every 5 minutes

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let available_device_ids: Vec<_> = {
                            let devices = available_devices.read();
                            devices.values()
                                .filter(|device| device.status == GpuDeviceStatus::Available)
                                .map(|device| device.device_id)
                                .collect()
                        };

                        for device_id in available_device_ids {
                            // Run lightweight performance benchmark
                            if let Err(e) = performance_tracker
                                .run_benchmark(device_id, GpuBenchmarkType::Compute)
                                .await
                            {
                                error!("Failed to run benchmark for device {}: {}", device_id, e);
                            }
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        debug!("Performance tracking task shutting down");
                        break;
                    }
                }
            }
        });

        self.background_tasks.lock().push(task);
        Ok(())
    }

    /// Start metrics collection background task
    async fn start_metrics_collection(&self) -> GpuResult<()> {
        let monitoring_system = self.monitoring_system.clone();
        let available_devices = self.available_devices.clone();
        let mut shutdown_rx = self.shutdown_sender.subscribe();
        let config = {
            let guard = self.config.read();
            guard.clone()
        };

        let task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.monitoring_interval_secs));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let devices_to_monitor: Vec<_> = {
                            let devices = available_devices.read();
                            devices.values().cloned().collect()
                        };

                        for device in devices_to_monitor {
                            // Simulate collecting real-time metrics
                            let metrics = Self::collect_device_metrics(&device).await;
                            if let Err(e) = monitoring_system.update_metrics(device.device_id, metrics).await {
                                error!("Failed to update metrics for device {}: {}", device.device_id, e);
                            }
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        debug!("Metrics collection task shutting down");
                        break;
                    }
                }
            }
        });

        self.background_tasks.lock().push(task);
        Ok(())
    }

    /// Collect real-time metrics for a device
    async fn collect_device_metrics(device: &GpuDeviceInfo) -> GpuRealTimeMetrics {
        // In a real implementation, this would use:
        // - NVML for NVIDIA GPUs
        // - ROCm for AMD GPUs
        // - System monitoring APIs

        // For mock implementation, simulate realistic metrics
        let utilization =
            (device.device_id as f32 * 7.0 + Utc::now().timestamp() as f32 * 0.1) % 100.0;
        let temperature = 45.0 + (utilization * 0.4) + (device.device_id as f32 * 2.0);
        let memory_usage = (device.total_memory_mb as f32 * utilization / 100.0) as u64;
        let power_consumption = 150.0 + (utilization * 2.0);

        GpuRealTimeMetrics {
            device_id: device.device_id,
            timestamp: Utc::now(),
            memory_usage_mb: memory_usage,
            utilization_percent: utilization,
            temperature_celsius: temperature,
            power_consumption_watts: power_consumption,
            clock_speeds: ResourceGpuClockSpeeds {
                core_clock_mhz: 1800 + (utilization as u32 * 5),
                memory_clock_mhz: 7000,
                shader_clock_mhz: Some(1900 + (utilization as u32 * 6)),
            },
            fan_speeds: vec![40.0 + (temperature - 45.0) * 1.5],
        }
    }

    /// Allocate GPU devices based on performance requirements
    ///
    /// This method performs intelligent device allocation considering:
    /// - Performance requirements and constraints
    /// - Device capabilities and compatibility
    /// - Current load balancing and utilization
    /// - Health status and availability
    ///
    /// # Arguments
    ///
    /// * `requirements` - List of GPU performance requirements
    /// * `test_id` - Unique identifier for the test requesting resources
    ///
    /// # Returns
    ///
    /// List of allocation IDs for successfully allocated devices
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No suitable devices available
    /// - Device capabilities don't match requirements
    /// - Hardware failure detected during allocation
    #[instrument(skip(self, requirements))]
    pub async fn allocate_devices(
        &self,
        requirements: &[GpuPerformanceRequirements],
        test_id: &str,
    ) -> GpuResult<Vec<String>> {
        if requirements.is_empty() {
            return Ok(vec![]);
        }

        self.operation_counter.fetch_add(1, Ordering::Relaxed);
        info!(
            "Allocating {} GPU devices for test {}",
            requirements.len(),
            test_id
        );

        let mut available_devices = self.available_devices.write();
        let mut allocated_resources = self.allocated_resources.write();
        let mut usage_stats = self.usage_stats.write();

        let mut allocated_device_ids = Vec::new();

        for (idx, req) in requirements.iter().enumerate() {
            // Use load balancer to select optimal device
            let device_id = self
                .load_balancer
                .select_device(&available_devices, req)
                .await
                .ok_or_else(|| GpuManagerError::DeviceNotFound { device_id: 0 })?;

            // Get and validate the device
            let device = available_devices
                .get_mut(&device_id)
                .ok_or_else(|| GpuManagerError::DeviceNotFound { device_id })?;

            // Verify device meets requirements
            self.verify_device_requirements(device, req)?;

            // Check device health
            let health_status = self.health_monitor.get_health_status().await;
            if let Some(health) = health_status.get(&device_id) {
                if !health.is_healthy {
                    return Err(GpuManagerError::HardwareFailure {
                        device_id,
                        details: health.issues.join(", "),
                    });
                }
            }

            // Allocate the device
            device.status = GpuDeviceStatus::Busy;
            device.available_memory_mb =
                device.available_memory_mb.saturating_sub(req.min_memory_mb);
            device.last_updated = Utc::now();

            let allocation_id = format!("gpu_{}_{}_req_{}", device_id, test_id, idx);
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

            debug!("Allocated GPU device {} for test {}", device_id, test_id);
        }

        // Update statistics
        usage_stats.total_allocations += allocated_device_ids.len() as u64;
        usage_stats.currently_allocated = allocated_resources.len();
        usage_stats.peak_usage = usage_stats.peak_usage.max(allocated_resources.len());

        // Calculate efficiency
        let total_devices = available_devices.len();
        if total_devices > 0 {
            usage_stats.efficiency = allocated_resources.len() as f32 / total_devices as f32;
        }

        info!(
            "Successfully allocated {} GPU devices for test {}: {:?}",
            allocated_device_ids.len(),
            test_id,
            allocated_device_ids
        );

        Ok(allocated_device_ids)
    }

    /// Verify device meets performance requirements
    fn verify_device_requirements(
        &self,
        device: &GpuDeviceInfo,
        requirements: &GpuPerformanceRequirements,
    ) -> GpuResult<()> {
        // Check memory requirements
        if device.available_memory_mb < requirements.min_memory_mb {
            return Err(GpuManagerError::InsufficientMemory {
                required_mb: requirements.min_memory_mb,
                available_mb: device.available_memory_mb,
            });
        }

        // Check device status
        if device.status != GpuDeviceStatus::Available {
            return Err(GpuManagerError::DeviceUnavailable {
                device_id: device.device_id,
                status: format!("{:?}", device.status),
            });
        }

        // Check framework requirements
        for required_framework in &requirements.required_frameworks {
            let has_framework = device.capabilities.iter().any(|capability| match capability {
                GpuCapability::MachineLearning(frameworks) => {
                    frameworks.contains(required_framework)
                },
                GpuCapability::Cuda(_version) => required_framework == "CUDA",
                GpuCapability::OpenCl(_version) => required_framework == "OpenCL",
                GpuCapability::Vulkan(_version) => required_framework == "Vulkan",
                _ => false,
            });

            if !has_framework {
                return Err(GpuManagerError::FrameworkNotSupported {
                    framework: required_framework.clone(),
                });
            }
        }

        // Check constraints
        for constraint in &requirements.constraints {
            self.verify_constraint(device, constraint)?;
        }

        Ok(())
    }

    /// Verify individual constraint
    fn verify_constraint(
        &self,
        device: &GpuDeviceInfo,
        constraint: &GpuConstraint,
    ) -> GpuResult<()> {
        match &constraint.constraint_type {
            GpuConstraintType::MaxMemoryUsage => {
                let memory_usage_ratio = (device.total_memory_mb - device.available_memory_mb)
                    as f64
                    / device.total_memory_mb as f64;
                if memory_usage_ratio > constraint.value {
                    return Err(GpuManagerError::ConstraintViolated {
                        constraint: format!(
                            "Memory usage {:.1}% exceeds limit {:.1}%",
                            memory_usage_ratio * 100.0,
                            constraint.value * 100.0
                        ),
                    });
                }
            },
            GpuConstraintType::MaxUtilization => {
                if device.utilization_percent as f64 > constraint.value {
                    return Err(GpuManagerError::ConstraintViolated {
                        constraint: format!(
                            "Utilization {:.1}% exceeds limit {:.1}%",
                            device.utilization_percent, constraint.value
                        ),
                    });
                }
            },
            GpuConstraintType::MinPerformance => {
                // In a real implementation, this would check against benchmark scores
                // For now, assume all devices meet minimum performance
            },
            GpuConstraintType::PowerLimit => {
                // Would check current power consumption
            },
            GpuConstraintType::TemperatureLimit => {
                // Would check current temperature
            },
            GpuConstraintType::Custom(_name) => {
                // Handle custom constraints
            },
        }

        Ok(())
    }

    /// Deallocate a specific GPU device allocation
    #[instrument(skip(self))]
    pub async fn deallocate_device(&self, allocation_id: &str) -> GpuResult<()> {
        info!("Deallocating GPU device: {}", allocation_id);

        let mut available_devices = self.available_devices.write();
        let mut allocated_resources = self.allocated_resources.write();
        let mut usage_stats = self.usage_stats.write();

        if let Some(allocation) = allocated_resources.remove(allocation_id) {
            // Return memory to device and mark as available
            if let Some(device) = available_devices.get_mut(&allocation.device.device_id) {
                device.available_memory_mb += allocation.memory_allocated_mb;
                device.status = GpuDeviceStatus::Available;
                device.last_updated = Utc::now();
            }

            // Update statistics
            usage_stats.currently_allocated = allocated_resources.len();

            // Calculate GPU hours usage
            let usage_duration = Utc::now().signed_duration_since(allocation.allocated_at);
            let hours = usage_duration.num_seconds() as f64 / 3600.0;
            usage_stats.total_gpu_hours += hours;

            // Update efficiency
            let total_devices = available_devices.len();
            if total_devices > 0 {
                usage_stats.efficiency = allocated_resources.len() as f32 / total_devices as f32;
            }

            info!("Successfully deallocated GPU device: {}", allocation_id);
            Ok(())
        } else {
            warn!(
                "Attempted to deallocate unknown GPU allocation: {}",
                allocation_id
            );
            Err(GpuManagerError::DeviceNotFound { device_id: 0 })
        }
    }

    /// Deallocate all GPU devices for a specific test
    #[instrument(skip(self))]
    pub async fn deallocate_devices_for_test(&self, test_id: &str) -> GpuResult<()> {
        info!("Deallocating all GPU devices for test: {}", test_id);

        let mut available_devices = self.available_devices.write();
        let mut allocated_resources = self.allocated_resources.write();
        let mut usage_stats = self.usage_stats.write();

        let mut deallocated_count = 0;
        let mut total_gpu_hours = 0.0;

        // Find and deallocate devices for this test
        allocated_resources.retain(|_allocation_id, allocation| {
            if allocation.test_id == test_id {
                // Return device to available pool
                if let Some(device) = available_devices.get_mut(&allocation.device.device_id) {
                    device.available_memory_mb += allocation.memory_allocated_mb;
                    device.status = GpuDeviceStatus::Available;
                    device.last_updated = Utc::now();
                }

                // Calculate usage time
                let usage_duration = Utc::now().signed_duration_since(allocation.allocated_at);
                let hours = usage_duration.num_seconds() as f64 / 3600.0;
                total_gpu_hours += hours;

                deallocated_count += 1;
                debug!(
                    "Deallocated GPU device {} for test {}",
                    allocation.device.device_id, test_id
                );

                false // Remove from allocated_resources
            } else {
                true // Keep in allocated_resources
            }
        });

        // Update statistics
        usage_stats.currently_allocated = allocated_resources.len();
        usage_stats.total_gpu_hours += total_gpu_hours;

        // Update efficiency
        let total_devices = available_devices.len();
        if total_devices > 0 {
            usage_stats.efficiency = allocated_resources.len() as f32 / total_devices as f32;
        }

        if deallocated_count > 0 {
            info!(
                "Successfully deallocated {} GPU devices for test {} (total GPU hours: {:.2})",
                deallocated_count, test_id, total_gpu_hours
            );
        } else {
            debug!("No GPU devices were allocated to test {}", test_id);
        }

        Ok(())
    }

    /// Check if requested GPU devices are available
    #[instrument(skip(self, requirements))]
    pub async fn check_availability(
        &self,
        requirements: &[GpuPerformanceRequirements],
    ) -> GpuResult<bool> {
        let available_devices = self.available_devices.read();
        let health_status = self.health_monitor.get_health_status().await;

        for req in requirements {
            let has_suitable_device = available_devices.values().any(|device| {
                // Check basic availability
                if device.status != GpuDeviceStatus::Available {
                    return false;
                }

                // Check memory requirements
                if device.available_memory_mb < req.min_memory_mb {
                    return false;
                }

                // Check health status
                if let Some(health) = health_status.get(&device.device_id) {
                    if !health.is_healthy {
                        return false;
                    }
                }

                // Check other requirements
                self.verify_device_requirements(device, req).is_ok()
            });

            if !has_suitable_device {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Get current GPU usage statistics
    pub async fn get_statistics(&self) -> GpuResult<GpuUsageStatistics> {
        let stats = self.usage_stats.read();
        Ok(stats.clone())
    }

    /// Get available GPU devices
    pub async fn get_available_devices(&self) -> Vec<GpuDeviceInfo> {
        let available_devices = self.available_devices.read();
        available_devices
            .values()
            .filter(|device| device.status == GpuDeviceStatus::Available)
            .cloned()
            .collect()
    }

    /// Get all GPU devices (available and allocated)
    pub async fn get_all_devices(&self) -> Vec<GpuDeviceInfo> {
        let available_devices = self.available_devices.read();
        available_devices.values().cloned().collect()
    }

    /// Get allocated GPU resources
    pub async fn get_allocated_resources(&self) -> HashMap<String, GpuAllocation> {
        let allocated_resources = self.allocated_resources.read();
        allocated_resources.clone()
    }

    /// Get GPU device information by ID
    pub async fn get_device_info(&self, device_id: usize) -> Option<GpuDeviceInfo> {
        let available_devices = self.available_devices.read();
        available_devices.get(&device_id).cloned()
    }

    /// Update GPU device status
    #[instrument(skip(self))]
    pub async fn update_device_status(
        &self,
        device_id: usize,
        status: GpuDeviceStatus,
    ) -> GpuResult<()> {
        let mut available_devices = self.available_devices.write();

        if let Some(device) = available_devices.get_mut(&device_id) {
            let old_status = device.status.clone();
            device.status = status;
            device.last_updated = Utc::now();

            info!(
                "Updated GPU device {} status from {:?} to {:?}",
                device_id, old_status, device.status
            );
            Ok(())
        } else {
            Err(GpuManagerError::DeviceNotFound { device_id })
        }
    }

    /// Get GPU utilization percentage across all devices
    pub async fn get_utilization(&self) -> f32 {
        let available_devices = self.available_devices.read();
        let allocated_resources = self.allocated_resources.read();

        let total_devices = available_devices.len();
        let allocated_count = allocated_resources.len();

        if total_devices == 0 {
            0.0
        } else {
            allocated_count as f32 / total_devices as f32
        }
    }

    /// Get real-time GPU metrics for all devices
    pub async fn get_realtime_metrics(&self) -> HashMap<usize, GpuRealTimeMetrics> {
        self.monitoring_system.get_realtime_metrics().await
    }

    /// Run performance benchmark on a specific device
    #[instrument(skip(self))]
    pub async fn run_benchmark(
        &self,
        device_id: usize,
        benchmark_type: GpuBenchmarkType,
    ) -> GpuResult<GpuPerformanceBenchmark> {
        // Verify device exists and is available
        let available_devices = self.available_devices.read();
        if !available_devices.contains_key(&device_id) {
            return Err(GpuManagerError::DeviceNotFound { device_id });
        }

        // Run benchmark through performance tracker
        self.performance_tracker
            .run_benchmark(device_id, benchmark_type)
            .await
            .map_err(|e| GpuManagerError::MonitoringError {
                source: anyhow::anyhow!("Benchmark failed: {}", e),
            })
    }

    /// Get performance analysis for all devices
    pub async fn get_performance_analysis(&self) -> GpuPerformanceAnalysis {
        self.performance_tracker.get_analysis().await
    }

    /// Get health status for all devices
    pub async fn get_health_status(&self) -> HashMap<usize, HealthMonitorGpuHealthStatus> {
        self.health_monitor.get_health_status().await
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> HashMap<String, GpuAlert> {
        self.alert_system.get_active_alerts().await
    }

    /// Acknowledge an alert
    #[instrument(skip(self))]
    pub async fn acknowledge_alert(&self, alert_id: &str) -> GpuResult<()> {
        self.alert_system.acknowledge_alert(alert_id).await.map_err(|e| {
            GpuManagerError::MonitoringError {
                source: anyhow::anyhow!("Alert acknowledgment failed: {}", e),
            }
        })
    }

    /// Update GPU pool configuration
    ///
    /// This method allows dynamic reconfiguration of the GPU pool without
    /// requiring a restart of the system.
    #[instrument(skip(self, new_config))]
    pub async fn update_config(&self, new_config: GpuPoolConfig) -> GpuResult<()> {
        info!("Updating GPU pool configuration");

        // Validate new configuration
        Self::validate_config(&new_config)?;

        // Save new device count before moving
        let new_device_count = new_config.max_devices;

        // Update configuration
        {
            let mut config = self.config.write();
            *config = new_config;
        }

        // If device count changed, trigger rediscovery
        let current_device_count = {
            let devices = self.available_devices.read();
            devices.len()
        };

        if new_device_count != current_device_count {
            warn!(
                "Device count changed from {} to {}, restart required for full effect",
                current_device_count, new_device_count
            );
        }

        info!("GPU pool configuration updated successfully");
        Ok(())
    }

    /// Get current configuration
    pub async fn get_config(&self) -> GpuPoolConfig {
        let config = self.config.read();
        config.clone()
    }

    /// Generate comprehensive GPU allocation report
    pub async fn generate_allocation_report(&self) -> String {
        let stats = self.get_statistics().await.unwrap_or_default();
        let available_devices = self.get_available_devices().await;
        let all_devices = self.get_all_devices().await;
        let allocated_resources = self.get_allocated_resources().await;
        let utilization = self.get_utilization().await;
        let health_status = self.get_health_status().await;
        let active_alerts = self.get_active_alerts().await;

        // Count healthy devices
        let healthy_devices = health_status.values().filter(|h| h.is_healthy).count();
        let unhealthy_devices = health_status.len() - healthy_devices;

        // Calculate average GPU hours per allocation
        let avg_gpu_hours = if stats.total_allocations > 0 {
            stats.total_gpu_hours / stats.total_allocations as f64
        } else {
            0.0
        };

        format!(
            "GPU Resource Management Report\n\
             =====================================\n\
             \n\
             Device Overview:\n\
             - Total devices: {}\n\
             - Available devices: {}\n\
             - Allocated devices: {}\n\
             - Healthy devices: {}\n\
             - Unhealthy devices: {}\n\
             \n\
             Allocation Statistics:\n\
             - Total allocations: {}\n\
             - Peak usage: {} devices\n\
             - Current utilization: {:.1}%\n\
             - Total GPU hours: {:.2}\n\
             - Average hours per allocation: {:.2}\n\
             - Allocation efficiency: {:.1}%\n\
             \n\
             System Health:\n\
             - Active alerts: {}\n\
             - Operations completed: {}\n\
             - Monitoring active: {}\n\
             \n\
             Performance:\n\
             - Average memory per device: {:.0}MB\n\
             - Peak memory usage: {:.1}%\n\
             \n\
             Detailed Device Information:\n",
            all_devices.len(),
            available_devices.len(),
            allocated_resources.len(),
            healthy_devices,
            unhealthy_devices,
            stats.total_allocations,
            stats.peak_usage,
            utilization * 100.0,
            stats.total_gpu_hours,
            avg_gpu_hours,
            stats.efficiency * 100.0,
            active_alerts.len(),
            self.operation_counter.load(Ordering::Relaxed),
            self.running.load(Ordering::Relaxed),
            stats.average_memory_allocated_mb,
            stats.peak_memory_usage_percent
        ) + &self.generate_device_details_report(all_devices, health_status).await
    }

    /// Generate detailed device information
    async fn generate_device_details_report(
        &self,
        devices: Vec<GpuDeviceInfo>,
        health_status: HashMap<usize, HealthMonitorGpuHealthStatus>,
    ) -> String {
        let mut report = String::new();

        for device in devices {
            let health = health_status.get(&device.device_id);
            let health_score = health.map(|h| h.health_score).unwrap_or(0.0);
            let issues = health.map(|h| h.issues.join(", ")).unwrap_or_default();

            report.push_str(&format!(
                "Device {}: {} ({:?})\n\
                 - Memory: {}MB total, {}MB available\n\
                 - Utilization: {:.1}%\n\
                 - Health Score: {:.2}\n\
                 - Issues: {}\n\
                 - Capabilities: {:?}\n\
                 - Last Updated: {}\n\
                 \n",
                device.device_id,
                device.device_name,
                device.status,
                device.total_memory_mb,
                device.available_memory_mb,
                device.utilization_percent,
                health_score,
                if issues.is_empty() { "None" } else { &issues },
                device.capabilities,
                device.last_updated.format("%Y-%m-%d %H:%M:%S UTC")
            ));
        }

        report
    }

    /// Force device refresh - rediscover and update device information
    ///
    /// This is useful when hardware changes have occurred or when devices
    /// need to be re-evaluated for health and capabilities.
    #[instrument(skip(self))]
    pub async fn refresh_devices(&self) -> GpuResult<()> {
        info!("Refreshing GPU device information");

        let config = {
            let guard = self.config.read();
            guard.clone()
        };
        let new_devices = Self::discover_gpu_devices(&config).await?;

        // Update device information while preserving allocations
        {
            let mut available_devices = self.available_devices.write();
            let allocated_resources = self.allocated_resources.read();

            // Preserve allocation status for currently allocated devices
            for (device_id, new_device) in new_devices {
                if let Some(existing_device) = available_devices.get(&device_id) {
                    if existing_device.status == GpuDeviceStatus::Busy {
                        // Keep the device as busy if it's currently allocated
                        let mut updated_device = new_device;
                        updated_device.status = GpuDeviceStatus::Busy;

                        // Update available memory based on current allocations
                        let allocated_memory: u64 = allocated_resources
                            .values()
                            .filter(|alloc| alloc.device.device_id == device_id)
                            .map(|alloc| alloc.memory_allocated_mb)
                            .sum();

                        updated_device.available_memory_mb =
                            updated_device.total_memory_mb.saturating_sub(allocated_memory);

                        available_devices.insert(device_id, updated_device);
                    } else {
                        available_devices.insert(device_id, new_device);
                    }
                } else {
                    available_devices.insert(device_id, new_device);
                }
            }
        }

        info!("GPU device refresh completed");
        Ok(())
    }

    /// Graceful shutdown of the GPU manager
    #[instrument(skip(self))]
    pub async fn shutdown(&self) -> GpuResult<()> {
        info!("Shutting down GPU resource manager");

        // Stop monitoring systems
        if self.running.load(Ordering::Acquire) {
            self.stop_monitoring().await?;
        }

        // Deallocate all remaining resources
        let allocated_resources: Vec<_> = {
            let resources = self.allocated_resources.read();
            resources.keys().cloned().collect()
        };

        for allocation_id in allocated_resources {
            if let Err(e) = self.deallocate_device(&allocation_id).await {
                warn!("Failed to deallocate device during shutdown: {}", e);
            }
        }

        info!("GPU resource manager shutdown completed");
        Ok(())
    }
}

impl Drop for GpuResourceManager {
    fn drop(&mut self) {
        // Ensure clean shutdown when the manager is dropped
        if self.running.load(Ordering::Acquire) {
            let _ = self.shutdown_sender.send(());
        }
    }
}
