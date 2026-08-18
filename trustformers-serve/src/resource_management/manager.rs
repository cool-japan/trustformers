//! Main resource management system coordinator.
//!
//! This module provides the ResourceManagementSystem that orchestrates all
//! resource management components including ports, directories, GPUs, databases,
//! custom resources, monitoring, allocation, cleanup, and statistics.

use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::sync::{atomic::AtomicBool, Arc};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::parallel_execution_engine::ResourceRequirement;
use crate::test_parallelization::ResourceAllocation;

use super::custom_resources::CustomResourceManager;
use super::database_management::DatabaseConnectionManager;
use super::directory_management::TempDirectoryManager;
use super::gpu_manager::types::GpuPoolConfig as GpuManagerPoolConfig;
use super::gpu_manager::GpuResourceManager;
use super::port_management::NetworkPortManager;
use super::types::*;

/// Comprehensive resource management system.
///
/// ## Removed in 0.2.1: components that were held but never consulted
///
/// `ResourceManagementSystem` used to also own a `config`, a
/// [`ResourceMonitor`], a `CleanupManager` and a `SystemStatistics`. All four
/// were constructed in `new`, stored, and never read again by any code path in
/// this crate — the system never sampled the monitor, never ran the cleanup
/// manager, and never published the statistics. Holding them made the struct
/// read as if it monitored and cleaned up; it did not.
///
/// They were removed rather than wired up, because wiring them means deciding
/// what a monitoring cadence and a cleanup policy should be, and that decision
/// belongs to whoever needs the behaviour — at which point the components can
/// come back as fields that something actually reads. What remains is the set
/// of managers the allocation path really calls.
pub struct ResourceManagementSystem {
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

    /// Conflict detector
    conflict_detector: Arc<ConflictDetector>,

    /// Resource allocator
    resource_allocator: Arc<ResourceAllocator>,

    /// Live claims, shared with the allocator and the conflict detector
    allocation_ledger: Arc<AllocationLedger>,

    /// Background tasks
    background_tasks: Vec<JoinHandle<()>>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

/// Resource monitor for system health
pub struct ResourceMonitor {}

/// A live claim on the resources one test holds between allocation and
/// deallocation.
///
/// Every field is copied from the [`ResourceRequirement`] that was granted, so
/// a claim describes what was actually asked for rather than an estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceClaim {
    /// Identifier of the allocation record this claim belongs to.
    pub resource_id: String,
    /// Test that owns the claim.
    pub test_id: String,
    /// GPU device indices the test named explicitly.
    pub gpu_devices: Vec<usize>,
    /// Number of network ports granted.
    pub network_ports: usize,
    /// Number of temporary directories granted.
    pub temp_directories: usize,
    /// Number of database connections granted.
    pub database_connections: usize,
    /// When the claim was recorded.
    pub claimed_at: chrono::DateTime<chrono::Utc>,
}

/// Ledger of live [`ResourceClaim`]s, shared by [`ResourceAllocator`] (which
/// writes it) and [`ConflictDetector`] (which reads it).
///
/// This is the single source of truth for "what is currently held". Without it
/// conflict detection has nothing to compare against, which is why the detector
/// and the allocator are constructed around one shared instance.
#[derive(Debug, Default)]
pub struct AllocationLedger {
    /// Live claims keyed by `resource_id`.
    claims: RwLock<std::collections::HashMap<String, ResourceClaim>>,
}

impl AllocationLedger {
    /// Record `claim`, returning the claim that blocked it, if any.
    ///
    /// A non-`None` return means the same `resource_id` was already live. The
    /// new claim is *rejected*, not merged: the stored claim stays exactly as it
    /// was, so the caller can report the collision without losing the original.
    pub fn record(&self, claim: ResourceClaim) -> Option<ResourceClaim> {
        let mut claims = self.claims.write();
        if let Some(existing) = claims.get(&claim.resource_id) {
            return Some(existing.clone());
        }
        claims.insert(claim.resource_id.clone(), claim);
        None
    }

    /// Release the claim under `resource_id`, returning it when one was live.
    pub fn release(&self, resource_id: &str) -> Option<ResourceClaim> {
        self.claims.write().remove(resource_id)
    }

    /// Release every claim owned by `test_id`, returning how many were live.
    pub fn release_for_test(&self, test_id: &str) -> usize {
        let mut claims = self.claims.write();
        let doomed: Vec<String> = claims
            .values()
            .filter(|claim| claim.test_id == test_id)
            .map(|claim| claim.resource_id.clone())
            .collect();
        for resource_id in &doomed {
            claims.remove(resource_id);
        }
        doomed.len()
    }

    /// Snapshot of every live claim.
    pub fn snapshot(&self) -> Vec<ResourceClaim> {
        self.claims.read().values().cloned().collect()
    }

    /// Number of live claims.
    pub fn len(&self) -> usize {
        self.claims.read().len()
    }

    /// Whether no claim is live.
    pub fn is_empty(&self) -> bool {
        self.claims.read().is_empty()
    }

    /// The conflict `requirements` would hit if granted to `test_id`, if any.
    ///
    /// Two conditions are checked, both by exact comparison against live
    /// claims:
    ///
    /// 1. `test_id` already holds a live allocation. The allocation identifier
    ///    is derived from the test id, so a second grant would overwrite the
    ///    first one's bookkeeping and leak its resources.
    /// 2. A GPU device index in `requirements.gpu_devices` is already held by
    ///    another test. GPU devices are named by index, so exclusivity is
    ///    decidable here.
    ///
    /// Ports, temporary directories and database connections are deliberately
    /// *not* checked: a [`ResourceRequirement`] carries only a count for them,
    /// and the concrete port numbers and paths are chosen by
    /// [`NetworkPortManager`](super::port_management::NetworkPortManager) and
    /// [`TempDirectoryManager`](super::directory_management::TempDirectoryManager)
    /// *after* this check runs. Those managers hand out disjoint resources and
    /// fail loudly when their pool is exhausted, so the exclusivity guarantee
    /// lives there rather than being guessed at here.
    pub fn conflict_with(
        &self,
        requirements: &ResourceRequirement,
        test_id: &str,
    ) -> Option<String> {
        let claims = self.claims.read();

        if let Some(existing) = claims.values().find(|claim| claim.test_id == test_id) {
            return Some(format!(
                "test '{}' already holds live allocation '{}' (claimed at {})",
                test_id, existing.resource_id, existing.claimed_at
            ));
        }

        for device in &requirements.gpu_devices {
            if let Some(holder) = claims
                .values()
                .find(|claim| claim.test_id != test_id && claim.gpu_devices.contains(device))
            {
                return Some(format!(
                    "GPU device {} is already held by test '{}' under allocation '{}'",
                    device, holder.test_id, holder.resource_id
                ));
            }
        }

        None
    }
}

/// Conflict detector for resource allocation.
///
/// Detection is an exact comparison against the live claims in the shared
/// [`AllocationLedger`]; there is no heuristic and no scoring. Consequently the
/// `detection_sensitivity` field of [`ConflictResolutionConfig`] is not
/// consulted, and neither is `enable_auto_resolution`: no automatic resolution
/// strategy is implemented, so a detected conflict is returned to the caller as
/// an error instead of being silently worked around.
pub struct ConflictDetector {
    /// Configuration this detector was built with.
    config: ConflictResolutionConfig,
    /// Live claims to compare against.
    ledger: Arc<AllocationLedger>,
}

/// Resource allocator for distribution.
///
/// Owns the write side of the shared [`AllocationLedger`].
pub struct ResourceAllocator {
    /// Live claims recorded by this allocator.
    ledger: Arc<AllocationLedger>,
}

/// Cleanup manager for resource cleanup
pub struct CleanupManager;

/// System statistics collector
pub struct SystemStatistics;

/// Health checker for system components
pub struct HealthChecker;

/// Alert system for notifications
pub struct AlertSystem;

impl ResourceManagementSystem {
    /// Create new resource management system
    pub async fn new(config: ResourceManagementConfig) -> Result<Self> {
        let port_manager = Arc::new(
            NetworkPortManager::new(config.resource_pools.network_port_pool.clone())
                .await
                .context("Failed to create port manager")?,
        );

        let temp_dir_manager = Arc::new(
            TempDirectoryManager::new(config.resource_pools.temp_directory_pool.clone())
                .await
                .context("Failed to create temp directory manager")?,
        );

        let gpu_manager = Arc::new(
            GpuResourceManager::new(GpuManagerPoolConfig {
                max_devices: config.resource_pools.gpu_device_pool.max_devices,
                enable_monitoring: config.resource_pools.gpu_device_pool.enable_monitoring,
                monitoring_interval_secs: config
                    .resource_pools
                    .gpu_device_pool
                    .monitoring_interval_secs,
                memory_threshold: config.resource_pools.gpu_device_pool.memory_threshold,
                temperature_threshold: config.resource_pools.gpu_device_pool.temperature_threshold,
                enable_performance_tracking: config
                    .resource_pools
                    .gpu_device_pool
                    .enable_performance_tracking,
                enable_health_monitoring: true,
                min_memory_mb: 1024,
                enable_alerts: true,
                enable_load_balancing: true,
                allocation_timeout_secs: 30,
                alert_thresholds: Default::default(),
                memory_allocation_threshold: config.resource_pools.gpu_device_pool.memory_threshold,
            })
            .await
            .context("Failed to create GPU manager")?,
        );

        let database_manager = Arc::new(
            DatabaseConnectionManager::new(config.resource_pools.database_pool.clone())
                .await
                .context("Failed to create database manager")?,
        );

        let custom_resource_manager = Arc::new(
            CustomResourceManager::new()
                .await
                .context("Failed to create custom resource manager")?,
        );

        // One ledger, shared: the allocator writes the claims the detector reads.
        // Handing each component its own copy would make every conflict check
        // look at an empty table and answer "no conflict" forever.
        let allocation_ledger = Arc::new(AllocationLedger::default());

        let conflict_detector = Arc::new(ConflictDetector::new(
            config.conflict_resolution.clone(),
            Arc::clone(&allocation_ledger),
        ));

        let resource_allocator = Arc::new(ResourceAllocator::new(Arc::clone(&allocation_ledger)));

        info!("Initialized resource management system");

        Ok(Self {
            port_manager,
            temp_dir_manager,
            gpu_manager,
            database_manager,
            custom_resource_manager,
            conflict_detector,
            resource_allocator,
            allocation_ledger,
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
        if let Some(conflict) =
            self.conflict_detector.check_conflicts(requirements, test_id).await?
        {
            return Err(anyhow::anyhow!(
                "Resource conflict detected: {:?}",
                conflict
            ));
        }

        // Dispatch allocation to each sub-manager based on the fields
        // present in the ResourceRequirement.  Failures are propagated immediately
        // so that partially-allocated resources can be cleaned up by the caller.
        if requirements.network_ports > 0 {
            self.port_manager
                .allocate_ports(requirements.network_ports, test_id)
                .await
                .context("Failed to allocate network ports")?;
        }

        if requirements.temp_directories > 0 {
            self.temp_dir_manager
                .allocate_directories(requirements.temp_directories, test_id)
                .await
                .context("Failed to allocate temporary directories")?;
        }

        if requirements.database_connections > 0 {
            self.database_manager
                .allocate_connections(requirements.database_connections, test_id)
                .await
                .context("Failed to allocate database connections")?;
        }

        // Determine resource_type label from whichever sub-resource is primary.
        let resource_type = if !requirements.resource_type.is_empty() {
            requirements.resource_type.clone()
        } else {
            "mixed".to_string()
        };

        // Create allocation record
        let allocation = ResourceAllocation {
            resource_type,
            resource_id: format!("allocation-{}", test_id),
            allocated_at: chrono::Utc::now(),
            deallocated_at: None,
            duration: std::time::Duration::from_secs(0),
            utilization: 0.0,
            efficiency: 0.0,
        };

        // Track allocation
        self.resource_allocator
            .track_allocation(&allocation, requirements, test_id)
            .await?;

        info!("Resources allocated successfully for test: {}", test_id);
        Ok(allocation)
    }

    /// Snapshot of every resource claim currently held.
    ///
    /// This is the same table [`Self::allocate_resources`] checks against, so a
    /// caller can see exactly why a conflict was reported.
    pub fn active_claims(&self) -> Vec<ResourceClaim> {
        self.allocation_ledger.snapshot()
    }

    /// Number of allocations currently live.
    pub fn active_allocation_count(&self) -> usize {
        self.allocation_ledger.len()
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
                self.deallocate_all_resources_for_test(test_id).await?;
            },
            "network_port" => {
                self.deallocate_network_ports_for_test(test_id).await?;
            },
            "temp_directory" => {
                self.deallocate_temp_directories_for_test(test_id).await?;
            },
            "gpu_device" => {
                self.deallocate_gpu_devices_for_test(test_id).await?;
            },
            "database_connection" => {
                self.deallocate_database_connections_for_test(test_id).await?;
            },
            "custom" => {
                self.deallocate_custom_resources_for_test(test_id).await?;
            },
            _ => {
                warn!(
                    "Unknown resource type: {}, attempting mixed deallocation",
                    allocation.resource_type
                );
                self.deallocate_all_resources_for_test(test_id).await?;
            },
        }

        // Update allocation record
        self.resource_allocator.mark_deallocated(allocation).await?;

        info!(
            "Resource deallocation completed for: {}",
            allocation.resource_id
        );
        Ok(())
    }

    /// Deallocate all resources for a test (mixed allocation)
    async fn deallocate_all_resources_for_test(&self, test_id: &str) -> Result<()> {
        // Attempt to deallocate from all managers
        // Use non-fatal error handling as not all tests use all resource types

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

    /// Deallocate network ports for test
    async fn deallocate_network_ports_for_test(&self, test_id: &str) -> Result<()> {
        self.port_manager.deallocate_ports_for_test(test_id).await
    }

    /// Deallocate temporary directories for test
    async fn deallocate_temp_directories_for_test(&self, test_id: &str) -> Result<()> {
        self.temp_dir_manager.deallocate_directories_for_test(test_id).await
    }

    /// Deallocate GPU devices for test
    async fn deallocate_gpu_devices_for_test(&self, test_id: &str) -> Result<()> {
        self.gpu_manager
            .deallocate_devices_for_test(test_id)
            .await
            .context("Failed to deallocate GPU devices")
    }

    /// Deallocate database connections for test
    async fn deallocate_database_connections_for_test(&self, test_id: &str) -> Result<()> {
        self.database_manager.deallocate_connections_for_test(test_id).await
    }

    /// Deallocate custom resources for test
    async fn deallocate_custom_resources_for_test(&self, test_id: &str) -> Result<()> {
        self.custom_resource_manager.deallocate_resources_for_test(test_id).await
    }

    /// Get system performance snapshot
    pub async fn get_performance_snapshot(&self) -> Result<SystemPerformanceSnapshot> {
        let gpu_manager_stats = self.gpu_manager.get_statistics().await?;
        let database_stats = self.database_manager.get_statistics().await?;
        let port_stats = self.port_manager.get_statistics().await?;
        let directory_stats = self.temp_dir_manager.get_statistics().await?;

        // Calculate average GPU utilization based on currently allocated vs peak usage
        let average_utilization = if gpu_manager_stats.peak_usage > 0 {
            (gpu_manager_stats.currently_allocated as f32 / gpu_manager_stats.peak_usage as f32)
                * 100.0
        } else {
            0.0_f32
        };

        // Calculate performance index based on efficiency and memory usage
        // Higher efficiency and lower memory usage percentage indicate better performance
        let performance_index = if gpu_manager_stats.peak_memory_usage_percent > 0.0 {
            let memory_efficiency = 1.0 - (gpu_manager_stats.peak_memory_usage_percent / 100.0);
            (gpu_manager_stats.efficiency * 0.6 + memory_efficiency * 0.4).min(1.0)
        } else {
            gpu_manager_stats.efficiency
        };

        // Convert gpu_manager::types::GpuUsageStatistics to resource_management::types::GpuUsageStatistics
        let gpu_stats = GpuUsageStatistics {
            total_allocations: gpu_manager_stats.total_allocations,
            currently_allocated: gpu_manager_stats.currently_allocated,
            peak_usage: gpu_manager_stats.peak_usage,
            average_utilization,
            total_memory_allocated_mb: (gpu_manager_stats.average_memory_allocated_mb
                * gpu_manager_stats.total_allocations as f64)
                as u64,
            allocation_efficiency: gpu_manager_stats.efficiency,
            performance_index,
        };

        // Collect actual CPU and memory utilization using sysinfo
        let mut system = sysinfo::System::new_all();
        system.refresh_all();

        // Calculate CPU utilization (average across all CPUs)
        let cpu_utilization = {
            let cpus = system.cpus();
            if !cpus.is_empty() {
                cpus.iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() / cpus.len() as f32
            } else {
                0.0
            }
        };

        // Calculate memory utilization percentage
        let memory_utilization = {
            let total_memory = system.total_memory();
            let used_memory = system.used_memory();
            if total_memory > 0 {
                (used_memory as f32 / total_memory as f32) * 100.0
            } else {
                0.0_f32
            }
        };

        let network_utilization =
            port_stats.currently_allocated as f32 / port_stats.peak_usage.max(1) as f32;

        // Overall efficiency is the mean of the subsystem occupancy signals that
        // have actually recorded activity: GPU device occupancy, temp-directory
        // utilization and port-pool occupancy. A subsystem that has never been
        // used contributes nothing rather than a zero that would drag the mean
        // down, and when nothing at all has been allocated the result is 0.0 —
        // this value is measured, never a nominal constant.
        let overall_efficiency = {
            let mut signals: Vec<f32> = Vec::new();
            if gpu_stats.total_allocations > 0 {
                signals.push(gpu_stats.allocation_efficiency);
            }
            if directory_stats.total_created > 0 {
                signals.push(directory_stats.utilization);
            }
            if port_stats.total_allocated > 0 {
                signals.push(network_utilization);
            }
            if signals.is_empty() {
                0.0_f32
            } else {
                signals.iter().sum::<f32>() / signals.len() as f32
            }
        };

        let snapshot = SystemPerformanceSnapshot {
            timestamp: chrono::Utc::now(),
            cpu_utilization,
            memory_utilization,
            gpu_utilization: if gpu_stats.total_allocations > 0 {
                // Use the calculated average_utilization
                Some(average_utilization)
            } else {
                None
            },
            network_utilization,
            disk_utilization: directory_stats.utilization,
            overall_efficiency,
            system_stats: SystemResourceStatistics::default(),
            gpu_stats,
            database_stats,
            port_stats,
            directory_stats,
        };

        Ok(snapshot)
    }

    /// Generate comprehensive resource report
    pub async fn generate_resource_report(&self) -> String {
        let mut report = String::from("Resource Management System Report\n");
        report.push_str("=======================================\n\n");

        // Port management report
        report.push_str("Network Port Management:\n");
        report.push_str(&self.port_manager.generate_allocation_report().await);
        report.push_str("\n\n");

        // Directory management report
        report.push_str("Temporary Directory Management:\n");
        report.push_str(&self.temp_dir_manager.generate_allocation_report().await);
        report.push_str("\n\n");

        // GPU management report
        report.push_str("GPU Resource Management:\n");
        report.push_str(&self.gpu_manager.generate_allocation_report().await);
        report.push_str("\n\n");

        // Database management report
        report.push_str("Database Connection Management:\n");
        report.push_str(&self.database_manager.generate_connection_report().await);
        report.push_str("\n\n");

        // Custom resource management report
        report.push_str("Custom Resource Management:\n");
        report.push_str(&self.custom_resource_manager.generate_report().await);

        report
    }

    /// Shutdown the resource management system
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down resource management system");

        // Signal shutdown
        self.shutdown.store(true, std::sync::atomic::Ordering::Relaxed);

        // Wait for background tasks to complete
        for task in self.background_tasks.drain(..) {
            if let Err(e) = task.await {
                warn!("Background task failed during shutdown: {}", e);
            }
        }

        info!("Resource management system shutdown complete");
        Ok(())
    }
}

impl ResourceMonitor {}

impl ConflictDetector {
    /// Build a detector sharing `ledger` with the system's [`ResourceAllocator`].
    fn new(config: ConflictResolutionConfig, ledger: Arc<AllocationLedger>) -> Self {
        Self { config, ledger }
    }

    /// The configuration this detector was built with.
    pub fn config(&self) -> &ConflictResolutionConfig {
        &self.config
    }

    /// Check `requirements` against every live claim in the ledger.
    ///
    /// Returns `Ok(None)` only after both checks below have run and found
    /// nothing; it never reports "no conflict" without looking.
    async fn check_conflicts(
        &self,
        requirements: &ResourceRequirement,
        test_id: &str,
    ) -> Result<Option<String>> {
        Ok(self.ledger.conflict_with(requirements, test_id))
    }
}

impl ResourceAllocator {
    /// Build an allocator writing into `ledger`.
    fn new(ledger: Arc<AllocationLedger>) -> Self {
        Self { ledger }
    }

    /// Record a live claim for `test_id`.
    ///
    /// # Errors
    ///
    /// Fails when the ledger already holds a claim under the same
    /// `resource_id`, which means the conflict check that precedes this call
    /// was bypassed. The pre-existing claim is left untouched.
    async fn track_allocation(
        &self,
        allocation: &ResourceAllocation,
        requirements: &ResourceRequirement,
        test_id: &str,
    ) -> Result<()> {
        let claim = ResourceClaim {
            resource_id: allocation.resource_id.clone(),
            test_id: test_id.to_string(),
            gpu_devices: requirements.gpu_devices.clone(),
            network_ports: requirements.network_ports,
            temp_directories: requirements.temp_directories,
            database_connections: requirements.database_connections,
            claimed_at: allocation.allocated_at,
        };
        if let Some(existing) = self.ledger.record(claim) {
            return Err(anyhow::anyhow!(
                "allocation '{}' is already claimed by test '{}' since {}",
                existing.resource_id,
                existing.test_id,
                existing.claimed_at
            ));
        }
        Ok(())
    }

    /// Release the claim recorded for `allocation`.
    ///
    /// Releasing an allocation the ledger does not know about is reported as a
    /// warning rather than an error: deallocation is idempotent, and an
    /// allocation record can outlive the process that tracked it.
    async fn mark_deallocated(&self, allocation: &ResourceAllocation) -> Result<()> {
        if self.ledger.release(&allocation.resource_id).is_none() {
            warn!(
                "No live claim recorded for allocation '{}'; nothing to release",
                allocation.resource_id
            );
        }
        Ok(())
    }
}

impl CleanupManager {}

impl SystemStatistics {}

impl HealthChecker {}

impl AlertSystem {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel_execution_engine::ResourceRequirement;
    use std::collections::HashMap;

    /// A requirement that touches no external resource, so allocating it is a
    /// pure bookkeeping operation.
    fn bookkeeping_only(gpu_devices: Vec<usize>) -> ResourceRequirement {
        ResourceRequirement {
            resource_type: "mixed".to_string(),
            min_amount: 0.0,
            cpu_cores: 0.0,
            memory_mb: 0,
            gpu_devices,
            network_ports: 0,
            temp_directories: 0,
            database_connections: 0,
            custom_resources: HashMap::new(),
        }
    }

    /// A configuration whose temporary directories live under the OS temp dir.
    ///
    /// `case` keeps concurrently running tests in separate subtrees: nextest
    /// runs each test in its own process, so a shared base path would have them
    /// creating and cleaning the same directory at the same time.
    fn test_config(case: &str) -> ResourceManagementConfig {
        let mut config = ResourceManagementConfig::default();
        config.resource_pools.temp_directory_pool.base_path =
            std::env::temp_dir().join("trustformers-resource-management-tests").join(case);
        config
    }

    fn claim(resource_id: &str, test_id: &str, gpu_devices: Vec<usize>) -> ResourceClaim {
        ResourceClaim {
            resource_id: resource_id.to_string(),
            test_id: test_id.to_string(),
            gpu_devices,
            network_ports: 0,
            temp_directories: 0,
            database_connections: 0,
            claimed_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn ledger_records_releases_and_reports_claims() {
        let ledger = AllocationLedger::default();
        assert!(ledger.is_empty());

        assert!(ledger.record(claim("allocation-a", "a", vec![])).is_none());
        assert_eq!(ledger.len(), 1);

        // A second claim on the same allocation id is refused, and the original
        // survives it.
        let displaced = ledger.record(claim("allocation-a", "b", vec![]));
        assert_eq!(displaced.map(|c| c.test_id), Some("a".to_string()));
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.snapshot()[0].test_id, "a");

        assert!(ledger.release("allocation-a").is_some());
        assert!(ledger.release("allocation-a").is_none());
        assert!(ledger.is_empty());
    }

    #[test]
    fn ledger_releases_every_claim_of_one_test() {
        let ledger = AllocationLedger::default();
        ledger.record(claim("allocation-a", "a", vec![]));
        ledger.record(claim("allocation-a-retry", "a", vec![]));
        ledger.record(claim("allocation-b", "b", vec![]));

        assert_eq!(ledger.release_for_test("a"), 2);
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.release_for_test("a"), 0);
    }

    /// Regression: `ConflictDetector::check_conflicts` used to return `Ok(None)`
    /// unconditionally, so a test already holding an allocation was granted a
    /// second one whose identifier collided with the first.
    #[test]
    fn ledger_detects_a_second_allocation_for_the_same_test() {
        let ledger = AllocationLedger::default();
        ledger.record(claim("allocation-alpha", "alpha", vec![]));

        let conflict = ledger.conflict_with(&bookkeeping_only(vec![]), "alpha");
        let message = conflict.expect("a live claim for 'alpha' must be reported");
        assert!(message.contains("alpha"), "{message}");
        assert!(message.contains("allocation-alpha"), "{message}");

        assert!(
            ledger.conflict_with(&bookkeeping_only(vec![]), "beta").is_none(),
            "an unrelated test must not be blocked"
        );
    }

    /// Regression: GPU devices are named by index, so two tests naming the same
    /// index conflict. The previous placeholder reported no conflict.
    #[test]
    fn ledger_detects_gpu_device_already_held_by_another_test() {
        let ledger = AllocationLedger::default();
        ledger.record(claim("allocation-alpha", "alpha", vec![0, 2]));

        let message = ledger
            .conflict_with(&bookkeeping_only(vec![2]), "beta")
            .expect("device 2 is held by 'alpha'");
        assert!(message.contains("GPU device 2"), "{message}");
        assert!(message.contains("alpha"), "{message}");

        assert!(
            ledger.conflict_with(&bookkeeping_only(vec![1]), "beta").is_none(),
            "a free device must not be reported as a conflict"
        );
    }

    /// Regression: the whole allocate → conflict → deallocate cycle through the
    /// public API. Against the placeholder implementation the second
    /// `allocate_resources` call returned `Ok`.
    #[tokio::test]
    async fn allocating_twice_for_one_test_is_reported_as_a_conflict() {
        let system = ResourceManagementSystem::new(test_config("duplicate-allocation"))
            .await
            .expect("system initialises");
        assert_eq!(system.active_allocation_count(), 0);

        let requirements = bookkeeping_only(vec![]);
        let allocation = system
            .allocate_resources(&requirements, "conflict-test")
            .await
            .expect("first allocation succeeds");
        assert_eq!(system.active_allocation_count(), 1);
        assert_eq!(system.active_claims()[0].test_id, "conflict-test");

        let second = system.allocate_resources(&requirements, "conflict-test").await;
        let error = second.expect_err("the second allocation must be refused");
        assert!(
            error.to_string().contains("Resource conflict detected"),
            "unexpected error: {error}"
        );
        assert_eq!(system.active_allocation_count(), 1);

        system.deallocate_resources(&allocation).await.expect("deallocation succeeds");
        assert_eq!(
            system.active_allocation_count(),
            0,
            "deallocation must release the claim"
        );

        // With the claim released the same test can allocate again.
        system
            .allocate_resources(&requirements, "conflict-test")
            .await
            .expect("re-allocation after release succeeds");
    }

    /// Regression: `overall_efficiency` was the hard-coded constant `0.85`,
    /// documented as "calculated based on all subsystem efficiencies". On a
    /// system that has allocated nothing there is no efficiency to report.
    #[tokio::test]
    async fn performance_snapshot_reports_measured_efficiency() {
        let system = ResourceManagementSystem::new(test_config("performance-snapshot"))
            .await
            .expect("system initialises");

        let snapshot = system.get_performance_snapshot().await.expect("snapshot is produced");
        assert_eq!(
            snapshot.overall_efficiency, 0.0,
            "an idle system has no measured efficiency to report"
        );
    }
}
