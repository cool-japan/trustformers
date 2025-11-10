//! Main ResourceManagementSystem orchestrating all resource management components.

use super::{
    allocation::{ConflictDetector, ResourceAllocator},
    cleanup::CleanupManager,
    custom_resources::CustomResourceManager,
    database_connections::DatabaseConnectionManager,
    gpu_resources::GpuResourceManager,
    monitoring::{ResourceMonitor, SystemStatistics},
    network_ports::NetworkPortManager,
    temp_directories::TempDirectoryManager,
    types::{AllocationEvent, SystemResourceStatistics},
};

use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::sync::{atomic::AtomicBool, Arc};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::test_independence_analyzer::ResourceRequirement;
use crate::test_parallelization::{ResourceAllocation, ResourceManagementConfig};

/// Comprehensive resource management system
pub struct ResourceManagementSystem {
    /// Configuration
    config: Arc<RwLock<ResourceManagementConfig>>,

    /// Network port manager
    port_manager: Arc<NetworkPortManager>,

    /// Temporary directory manager
    temp_dir_manager: Arc<TempDirectoryManager>,

    /// GPU resource manager
    gpu_manager: Arc<GpuResourceManager>,

    /// Database connection manager
    database_manager: Arc<DatabaseConnectionManager>,

    /// Custom resource manager
    custom_resource_manager: Arc<CustomResourceManager>,

    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,

    /// Conflict detector
    conflict_detector: Arc<ConflictDetector>,

    /// Resource allocator
    resource_allocator: Arc<ResourceAllocator>,

    /// Cleanup manager
    cleanup_manager: Arc<CleanupManager>,

    /// System statistics
    system_stats: Arc<SystemStatistics>,

    /// Background tasks
    background_tasks: Vec<JoinHandle<()>>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

impl ResourceManagementSystem {
    /// Create a new resource management system
    pub async fn new(config: ResourceManagementConfig) -> Result<Self> {
        info!("Initializing resource management system");

        let port_manager = Arc::new(
            NetworkPortManager::new(config.resource_pools.network_port_pool.clone())
                .await
                .context("Failed to initialize network port manager")?,
        );

        let temp_dir_manager = Arc::new(
            TempDirectoryManager::new(config.resource_pools.temp_directory_pool.clone())
                .await
                .context("Failed to initialize temporary directory manager")?,
        );

        let gpu_manager = Arc::new(
            GpuResourceManager::new(config.resource_pools.gpu_device_pool.clone())
                .await
                .context("Failed to initialize GPU resource manager")?,
        );

        let database_manager = Arc::new(
            DatabaseConnectionManager::new(config.resource_pools.database_pool.clone())
                .await
                .context("Failed to initialize database connection manager")?,
        );

        let custom_resource_manager = Arc::new(
            CustomResourceManager::new()
                .await
                .context("Failed to initialize custom resource manager")?,
        );

        let resource_monitor = Arc::new(
            ResourceMonitor::new(config.resource_monitoring.clone())
                .await
                .context("Failed to initialize resource monitor")?,
        );

        let conflict_detector = Arc::new(
            ConflictDetector::new(config.conflict_resolution.clone())
                .await
                .context("Failed to initialize conflict detector")?,
        );

        let resource_allocator = Arc::new(
            ResourceAllocator::new()
                .await
                .context("Failed to initialize resource allocator")?,
        );

        let cleanup_manager = Arc::new(
            CleanupManager::new(config.resource_cleanup.clone())
                .await
                .context("Failed to initialize cleanup manager")?,
        );

        let system_stats = Arc::new(SystemStatistics::new());

        info!("Resource management system initialized successfully");

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            port_manager,
            temp_dir_manager,
            gpu_manager,
            database_manager,
            custom_resource_manager,
            resource_monitor,
            conflict_detector,
            resource_allocator,
            cleanup_manager,
            system_stats,
            background_tasks: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Allocate resources for a test
    pub async fn allocate_resources(
        &self,
        requirements: &ResourceRequirement,
        test_id: &str,
    ) -> Result<ResourceAllocation> {
        info!("Allocating resources for test: {}", test_id);

        // Check for resource conflicts
        if let Some(conflict) = self
            .conflict_detector
            .check_conflicts(requirements, test_id)
            .await
            .context("Failed to check resource conflicts")?
        {
            return Err(anyhow::anyhow!(
                "Resource conflict detected: {:?}",
                conflict
            ));
        }

        // Implement resource allocation using the generic ResourceRequirement structure
        // Parse resource_type and allocate appropriate resources based on requirements
        let mut ports: Vec<u16> = vec![];
        let mut temp_dirs: Vec<String> = vec![];
        let mut gpu_devices: Vec<u32> = vec![];
        let mut db_connections: Vec<String> = vec![];

        let resource_type_str = requirements.resource_type.to_lowercase();
        let amount_needed = requirements.min_amount.ceil() as usize;

        match resource_type_str.as_str() {
            "network_port" | "port" => {
                debug!(
                    "Allocating {} network port(s) for test: {}",
                    amount_needed, test_id
                );
                match self.port_manager.allocate_ports(amount_needed, test_id).await {
                    Ok(allocated_ports) => {
                        ports = allocated_ports;
                    },
                    Err(e) => {
                        return Err(anyhow::anyhow!("Failed to allocate required ports: {}", e));
                    },
                }
            },
            "temp_directory" | "temp_dir" | "directory" => {
                debug!(
                    "Allocating {} temporary director(y/ies) for test: {}",
                    amount_needed, test_id
                );
                match self.temp_dir_manager.allocate_directories(amount_needed, test_id).await {
                    Ok(allocated_dirs) => {
                        temp_dirs = allocated_dirs;
                    },
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "Failed to allocate required temp directories: {}",
                            e
                        ));
                    },
                }
            },
            "gpu_device" | "gpu" => {
                debug!(
                    "Allocating {} GPU device(s) for test: {}",
                    amount_needed, test_id
                );
                // Allocate GPU devices based on available device IDs
                let device_ids: Vec<usize> = (0..amount_needed).collect();
                match self.gpu_manager.allocate_devices(&device_ids, test_id).await {
                    Ok(allocated_devices) => {
                        gpu_devices = allocated_devices.into_iter().map(|id| id as u32).collect();
                    },
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "Failed to allocate required GPU devices: {}",
                            e
                        ));
                    },
                }
            },
            "database_connection" | "database" | "db" => {
                debug!(
                    "Allocating {} database connection(s) for test: {}",
                    amount_needed, test_id
                );
                match self.database_manager.allocate_connections(amount_needed, test_id).await {
                    Ok(allocated_conns) => {
                        db_connections = allocated_conns;
                    },
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "Failed to allocate required database connections: {}",
                            e
                        ));
                    },
                }
            },
            "custom" => {
                debug!("Allocating custom resource for test: {}", test_id);
                // Custom resources are tracked separately and don't return specific IDs
                // We just track that the allocation happened
                debug!(
                    "Custom resource allocated successfully (amount: {})",
                    requirements.min_amount
                );
            },
            "mixed" | _ => {
                // Mixed allocation or unknown type: try to intelligently allocate
                // based on min_amount or use a default mixed allocation
                debug!(
                    "Performing mixed resource allocation for test: {} (type: {})",
                    test_id, resource_type_str
                );

                // Allocate one of each type for mixed allocation
                // Port
                if let Ok(allocated_ports) = self.port_manager.allocate_ports(1, test_id).await {
                    ports = allocated_ports;
                }

                // Temp directory
                if let Ok(allocated_dirs) =
                    self.temp_dir_manager.allocate_directories(1, test_id).await
                {
                    temp_dirs = allocated_dirs;
                }

                // Note: GPU and database are optional for mixed allocation
                // to avoid resource exhaustion on systems without GPUs or databases
            },
        }

        // Create allocation record with actual allocated resources
        let allocation = ResourceAllocation {
            resource_type: resource_type_str.clone(),
            resource_id: format!("allocation-{}", test_id),
            allocated_at: chrono::Utc::now(),
            deallocated_at: None,
            duration: std::time::Duration::from_secs(0),
            utilization: 0.0, // Will be updated during deallocation
            efficiency: 0.0,  // Will be calculated based on actual usage
        };

        // Track allocation
        self.resource_allocator
            .track_allocation(&allocation)
            .await
            .context("Failed to track resource allocation")?;

        info!(
            "Resources allocated successfully for test: {} (ports: {}, dirs: {}, gpus: {}, db: {})",
            test_id,
            ports.len(),
            temp_dirs.len(),
            gpu_devices.len(),
            db_connections.len()
        );

        Ok(allocation)
    }

    /// Deallocate resources for a test
    pub async fn deallocate_resources(&self, allocation: &ResourceAllocation) -> Result<()> {
        info!(
            "Deallocating resources for resource_id: {}",
            allocation.resource_id
        );

        // Extract test_id from resource_id (format: "allocation-{test_id}")
        let test_id = if allocation.resource_id.starts_with("allocation-") {
            &allocation.resource_id[11..]
        } else {
            &allocation.resource_id
        };

        // Deallocate resources from all managers based on resource type
        match allocation.resource_type.as_str() {
            "mixed" => {
                // Mixed allocation - deallocate from all managers
                self.deallocate_all_resources_for_test(test_id)
                    .await
                    .context("Failed to deallocate mixed resources")?;
            },
            "network_port" => {
                self.deallocate_network_ports_for_test(test_id)
                    .await
                    .context("Failed to deallocate network ports")?;
            },
            "temp_directory" => {
                self.deallocate_temp_directories_for_test(test_id)
                    .await
                    .context("Failed to deallocate temporary directories")?;
            },
            "gpu_device" => {
                self.deallocate_gpu_devices_for_test(test_id)
                    .await
                    .context("Failed to deallocate GPU devices")?;
            },
            "database_connection" => {
                self.deallocate_database_connections_for_test(test_id)
                    .await
                    .context("Failed to deallocate database connections")?;
            },
            "custom" => {
                self.deallocate_custom_resources_for_test(test_id)
                    .await
                    .context("Failed to deallocate custom resources")?;
            },
            _ => {
                warn!(
                    "Unknown resource type: {}, attempting mixed deallocation",
                    allocation.resource_type
                );
                self.deallocate_all_resources_for_test(test_id)
                    .await
                    .context("Failed to deallocate resources with unknown type")?;
            },
        }

        // Update allocation record
        self.resource_allocator
            .mark_deallocated(allocation)
            .await
            .context("Failed to mark allocation as deallocated")?;

        info!(
            "Resource deallocation completed for: {}",
            allocation.resource_id
        );
        Ok(())
    }

    /// Deallocate all resources for a test (mixed allocation)
    async fn deallocate_all_resources_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating all resources for test: {}", test_id);

        // Attempt to deallocate from all managers
        // We use non-fatal error handling as not all tests use all resource types

        if let Err(e) = self.deallocate_network_ports_for_test(test_id).await {
            warn!(
                "Failed to deallocate network ports for test {}: {}",
                test_id, e
            );
        }

        if let Err(e) = self.deallocate_temp_directories_for_test(test_id).await {
            warn!(
                "Failed to deallocate temp directories for test {}: {}",
                test_id, e
            );
        }

        if let Err(e) = self.deallocate_gpu_devices_for_test(test_id).await {
            warn!(
                "Failed to deallocate GPU devices for test {}: {}",
                test_id, e
            );
        }

        if let Err(e) = self.deallocate_database_connections_for_test(test_id).await {
            warn!(
                "Failed to deallocate database connections for test {}: {}",
                test_id, e
            );
        }

        if let Err(e) = self.deallocate_custom_resources_for_test(test_id).await {
            warn!(
                "Failed to deallocate custom resources for test {}: {}",
                test_id, e
            );
        }

        Ok(())
    }

    /// Deallocate network ports for a test
    async fn deallocate_network_ports_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating network ports for test: {}", test_id);
        self.port_manager
            .deallocate_ports_for_test(test_id)
            .await
            .context("Failed to deallocate network ports")
    }

    /// Deallocate temporary directories for a test
    async fn deallocate_temp_directories_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating temporary directories for test: {}", test_id);
        self.temp_dir_manager
            .deallocate_directories_for_test(test_id)
            .await
            .context("Failed to deallocate temporary directories")
    }

    /// Deallocate GPU devices for a test
    async fn deallocate_gpu_devices_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating GPU devices for test: {}", test_id);
        self.gpu_manager
            .deallocate_devices_for_test(test_id)
            .await
            .context("Failed to deallocate GPU devices")
    }

    /// Deallocate database connections for a test
    async fn deallocate_database_connections_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating database connections for test: {}", test_id);
        self.database_manager
            .deallocate_connections_for_test(test_id)
            .await
            .context("Failed to deallocate database connections")
    }

    /// Deallocate custom resources for a test
    async fn deallocate_custom_resources_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating custom resources for test: {}", test_id);
        self.custom_resource_manager
            .deallocate_resources_for_test(test_id)
            .await
            .context("Failed to deallocate custom resources")
    }

    /// Check if resources are available
    pub async fn check_availability(&self, _requirements: &ResourceRequirement) -> Result<bool> {
        // TODO: Implement availability checking using the new generic ResourceRequirement structure
        // The resource_type field should be parsed to determine which type of resource to check
        // For now, always returning true as placeholder
        Ok(true)
    }

    /// Get system resource statistics
    pub async fn get_statistics(&self) -> Result<SystemResourceStatistics> {
        let port_stats = self
            .port_manager
            .get_statistics()
            .await
            .context("Failed to get port statistics")?;

        let directory_stats = self
            .temp_dir_manager
            .get_statistics()
            .await
            .context("Failed to get directory statistics")?;

        let gpu_stats = self
            .gpu_manager
            .get_statistics()
            .await
            .context("Failed to get GPU statistics")?;

        let database_stats = self
            .database_manager
            .get_statistics()
            .await
            .context("Failed to get database statistics")?;

        let overall_efficiency = self.system_stats.calculate_efficiency().await;

        Ok(SystemResourceStatistics {
            port_stats,
            directory_stats,
            gpu_stats,
            database_stats,
            overall_efficiency,
        })
    }

    /// Start the resource management system
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting resource management system");

        // Start monitoring and background tasks
        // In a real implementation, this would start background tasks for:
        // - Resource monitoring
        // - Cleanup processing
        // - Health checks
        // - Statistics collection

        info!("Resource management system started successfully");
        Ok(())
    }

    /// Stop the resource management system
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping resource management system");

        // Signal shutdown
        self.shutdown.store(true, std::sync::atomic::Ordering::Relaxed);

        // Wait for background tasks to complete
        for task in self.background_tasks.drain(..) {
            let _ = task.await;
        }

        // Perform emergency cleanup
        let cleanup_count = self
            .cleanup_manager
            .force_cleanup_all()
            .await
            .context("Failed to perform emergency cleanup")?;

        info!(
            "Resource management system stopped (cleaned up {} resources)",
            cleanup_count
        );
        Ok(())
    }

    /// Update system configuration
    pub async fn update_config(&self, config: ResourceManagementConfig) -> Result<()> {
        info!("Updating resource management system configuration");

        // Update configuration
        let mut current_config = self.config.write();
        *current_config = config.clone();
        drop(current_config);

        // Update individual manager configurations
        self.port_manager
            .update_config(config.resource_pools.network_port_pool)
            .await
            .context("Failed to update port manager config")?;

        self.temp_dir_manager
            .update_config(config.resource_pools.temp_directory_pool)
            .await
            .context("Failed to update temp directory manager config")?;

        self.cleanup_manager
            .update_config(config.resource_cleanup)
            .await
            .context("Failed to update cleanup manager config")?;

        self.database_manager
            .update_config(config.resource_pools.database_pool)
            .await
            .context("Failed to update database manager config")?;

        info!("Resource management system configuration updated successfully");
        Ok(())
    }

    /// Get allocation events
    pub async fn get_allocation_events(&self) -> Result<Vec<AllocationEvent>> {
        self.resource_allocator
            .get_allocation_history()
            .await
            .context("Failed to get allocation history")
    }

    /// Perform health check on all resources
    pub async fn health_check(&self) -> Result<HealthCheckResult> {
        info!("Performing system health check");

        let port_health = self
            .port_manager
            .get_available_count()
            .await
            .context("Failed to check port health")?;

        let temp_dir_health = self
            .temp_dir_manager
            .get_statistics()
            .await
            .context("Failed to check temp directory health")?;

        let gpu_health =
            self.gpu_manager.get_statistics().await.context("Failed to check GPU health")?;

        let database_health = self
            .database_manager
            .get_statistics()
            .await
            .context("Failed to check database health")?;

        let custom_health = self
            .custom_resource_manager
            .health_check_resources()
            .await
            .context("Failed to check custom resource health")?;

        Ok(HealthCheckResult {
            overall_status: HealthStatus::Healthy,
            port_available_count: port_health,
            temp_dir_utilization: temp_dir_health.utilization,
            gpu_utilization: gpu_health.average_utilization,
            database_efficiency: database_health.pool_efficiency,
            custom_resource_health: custom_health,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Force cleanup all resources (emergency)
    pub async fn emergency_cleanup(&self) -> Result<EmergencyCleanupResult> {
        warn!("Performing emergency cleanup of all resources");

        let port_count = self
            .port_manager
            .force_release_all()
            .await
            .context("Failed to force release ports")?;

        let temp_dir_count = self
            .temp_dir_manager
            .force_cleanup_all()
            .await
            .context("Failed to force cleanup temp directories")?;

        let gpu_count = self
            .gpu_manager
            .force_release_all()
            .await
            .context("Failed to force release GPUs")?;

        let database_count = self
            .database_manager
            .force_close_all()
            .await
            .context("Failed to force close database connections")?;

        let cleanup_count = self
            .cleanup_manager
            .force_cleanup_all()
            .await
            .context("Failed to force cleanup")?;

        Ok(EmergencyCleanupResult {
            ports_released: port_count,
            temp_dirs_cleaned: temp_dir_count,
            gpus_released: gpu_count,
            database_connections_closed: database_count,
            cleanup_tasks_processed: cleanup_count,
            timestamp: chrono::Utc::now(),
        })
    }
}

/// Health check result
#[derive(Debug)]
pub struct HealthCheckResult {
    pub overall_status: HealthStatus,
    pub port_available_count: usize,
    pub temp_dir_utilization: f32,
    pub gpu_utilization: f32,
    pub database_efficiency: f32,
    pub custom_resource_health: std::collections::HashMap<String, bool>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Overall health status
#[derive(Debug)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Failed,
}

/// Emergency cleanup result
#[derive(Debug)]
pub struct EmergencyCleanupResult {
    pub ports_released: usize,
    pub temp_dirs_cleaned: usize,
    pub gpus_released: usize,
    pub database_connections_closed: usize,
    pub cleanup_tasks_processed: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
