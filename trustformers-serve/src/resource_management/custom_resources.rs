//! Custom resource management for test parallelization.
//!
//! This module provides generic resource management capabilities for handling
//! custom resource types that don't fit into standard categories like ports,
//! directories, GPUs, or database connections.

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, info, warn};

/// Custom resource management system
pub struct CustomResourceManager {
    /// Custom resource pools by type
    resource_pools: Arc<Mutex<HashMap<String, CustomResourcePool>>>,
    /// Global resource allocations
    allocations: Arc<Mutex<HashMap<String, CustomResourceAllocation>>>,
}

/// Custom resource pool for a specific resource type
#[derive(Debug, Clone)]
pub struct CustomResourcePool {
    /// Resource type name
    pub resource_type: String,
    /// Available resources
    pub available_resources: Vec<CustomResource>,
    /// Pool configuration
    pub config: CustomResourceConfig,
    /// Usage statistics
    pub statistics: CustomResourceStatistics,
}

/// Custom resource configuration
#[derive(Debug, Clone)]
pub struct CustomResourceConfig {
    /// Maximum number of resources
    pub max_resources: usize,
    /// Resource lifetime limit
    pub max_lifetime: Option<std::time::Duration>,
    /// Auto-cleanup enabled
    pub auto_cleanup: bool,
    /// Resource properties
    pub properties: HashMap<String, String>,
}

/// Custom resource definition
#[derive(Debug, Clone)]
pub struct CustomResource {
    /// Resource ID
    pub resource_id: String,
    /// Resource type
    pub resource_type: String,
    /// Resource properties
    pub properties: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Current status
    pub status: CustomResourceStatus,
}

/// Custom resource status
#[derive(Debug, Clone)]
pub enum CustomResourceStatus {
    /// Available for allocation
    Available,
    /// Currently allocated
    Allocated,
    /// Under maintenance
    Maintenance,
    /// Error state
    Error(String),
    /// Retired/unusable
    Retired,
}

/// Custom resource allocation
#[derive(Debug, Clone)]
pub struct CustomResourceAllocation {
    /// Allocation ID
    pub allocation_id: String,
    /// Resource being allocated
    pub resource: CustomResource,
    /// Test ID that allocated the resource
    pub test_id: String,
    /// Allocation timestamp
    pub allocated_at: DateTime<Utc>,
    /// Expected release time
    pub expected_release: Option<DateTime<Utc>>,
    /// Allocation metadata
    pub metadata: HashMap<String, String>,
}

/// Custom resource usage statistics
#[derive(Debug, Clone, Default)]
pub struct CustomResourceStatistics {
    /// Total allocations
    pub total_allocations: u64,
    /// Currently allocated
    pub currently_allocated: usize,
    /// Peak usage
    pub peak_usage: usize,
    /// Average allocation duration
    pub average_duration: std::time::Duration,
    /// Resource utilization
    pub utilization: f32,
    /// Total resources created
    pub total_created: u64,
}

impl CustomResourceManager {
    /// Create new custom resource manager
    pub async fn new() -> Result<Self> {
        info!("Initialized custom resource manager");

        Ok(Self {
            resource_pools: Arc::new(Mutex::new(HashMap::new())),
            allocations: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Register a new resource pool
    pub async fn register_resource_pool(
        &self,
        resource_type: &str,
        config: CustomResourceConfig,
    ) -> Result<()> {
        let mut resource_pools = self.resource_pools.lock();

        let pool = CustomResourcePool {
            resource_type: resource_type.to_string(),
            available_resources: Vec::new(),
            config,
            statistics: CustomResourceStatistics::default(),
        };

        resource_pools.insert(resource_type.to_string(), pool);

        info!("Registered custom resource pool: {}", resource_type);
        Ok(())
    }

    /// Add custom resource to a pool
    pub async fn add_resource(&self, resource: CustomResource) -> Result<()> {
        let mut resource_pools = self.resource_pools.lock();

        if let Some(pool) = resource_pools.get_mut(&resource.resource_type) {
            if pool.available_resources.len() < pool.config.max_resources {
                pool.available_resources.push(resource.clone());
                pool.statistics.total_created += 1;

                info!(
                    "Added custom resource {} to pool {}",
                    resource.resource_id, resource.resource_type
                );
                Ok(())
            } else {
                Err(anyhow::anyhow!(
                    "Resource pool {} is at maximum capacity",
                    resource.resource_type
                ))
            }
        } else {
            Err(anyhow::anyhow!(
                "Resource pool {} not found",
                resource.resource_type
            ))
        }
    }

    /// Allocate custom resources
    pub async fn allocate_resources(
        &self,
        resource_type: &str,
        count: usize,
        test_id: &str,
    ) -> Result<Vec<String>> {
        if count == 0 {
            return Ok(vec![]);
        }

        let mut resource_pools = self.resource_pools.lock();
        let mut allocations = self.allocations.lock();

        if let Some(pool) = resource_pools.get_mut(resource_type) {
            if pool.available_resources.len() < count {
                return Err(anyhow::anyhow!(
                    "Insufficient available resources: requested {}, available {}",
                    count,
                    pool.available_resources.len()
                ));
            }

            let mut allocated_ids = Vec::new();
            let now = Utc::now();

            // Allocate resources
            for _ in 0..count {
                if let Some(mut resource) = pool.available_resources.pop() {
                    resource.status = CustomResourceStatus::Allocated;

                    let allocation_id = format!("{}_{}_alloc", resource.resource_id, test_id);
                    let allocation = CustomResourceAllocation {
                        allocation_id: allocation_id.clone(),
                        resource,
                        test_id: test_id.to_string(),
                        allocated_at: now,
                        expected_release: None,
                        metadata: HashMap::new(),
                    };

                    allocations.insert(allocation_id.clone(), allocation);
                    allocated_ids.push(allocation_id);
                } else {
                    // Rollback partial allocation
                    for alloc_id in &allocated_ids {
                        if let Some(allocation) = allocations.remove(alloc_id) {
                            let mut resource = allocation.resource;
                            resource.status = CustomResourceStatus::Available;
                            pool.available_resources.push(resource);
                        }
                    }
                    return Err(anyhow::anyhow!("Failed to allocate custom resources"));
                }
            }

            // Update statistics
            pool.statistics.total_allocations += count as u64;
            pool.statistics.currently_allocated = pool.statistics.currently_allocated + count;
            pool.statistics.peak_usage =
                pool.statistics.peak_usage.max(pool.statistics.currently_allocated);

            info!(
                "Allocated {} custom resources of type {} for test {}: {:?}",
                allocated_ids.len(),
                resource_type,
                test_id,
                allocated_ids
            );

            Ok(allocated_ids)
        } else {
            Err(anyhow::anyhow!("Resource pool {} not found", resource_type))
        }
    }

    /// Deallocate a specific resource
    pub async fn deallocate_resource(&self, allocation_id: &str) -> Result<()> {
        let mut resource_pools = self.resource_pools.lock();
        let mut allocations = self.allocations.lock();

        if let Some(allocation) = allocations.remove(allocation_id) {
            let mut resource = allocation.resource;
            resource.status = CustomResourceStatus::Available;

            if let Some(pool) = resource_pools.get_mut(&resource.resource_type) {
                pool.available_resources.push(resource);
                pool.statistics.currently_allocated =
                    pool.statistics.currently_allocated.saturating_sub(1);

                // Update average duration statistics
                let duration = allocation.allocated_at.signed_duration_since(Utc::now()).abs();
                let duration_std =
                    std::time::Duration::from_secs(duration.num_seconds().max(0) as u64);

                if pool.statistics.total_allocations > 0 {
                    let total_duration = pool.statistics.average_duration.as_secs() as f64
                        * (pool.statistics.total_allocations - 1) as f64;
                    let new_average = (total_duration + duration_std.as_secs() as f64)
                        / pool.statistics.total_allocations as f64;
                    pool.statistics.average_duration =
                        std::time::Duration::from_secs(new_average as u64);
                }

                info!(
                    "Deallocated custom resource {} for test {}",
                    allocation_id, allocation.test_id
                );
                Ok(())
            } else {
                Err(anyhow::anyhow!(
                    "Resource pool {} not found",
                    resource.resource_type
                ))
            }
        } else {
            warn!(
                "Attempted to deallocate custom resource {} that was not allocated",
                allocation_id
            );
            Err(anyhow::anyhow!(
                "Custom resource allocation {} was not found",
                allocation_id
            ))
        }
    }

    /// Deallocate all resources for a specific test
    pub async fn deallocate_resources_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating custom resources for test: {}", test_id);

        let mut resource_pools = self.resource_pools.lock();
        let mut allocations = self.allocations.lock();

        let mut deallocated_resources = Vec::new();

        // Find and collect allocations to deallocate
        allocations.retain(|allocation_id, allocation| {
            if allocation.test_id == test_id {
                let mut resource = allocation.resource.clone();
                resource.status = CustomResourceStatus::Available;

                // Return resource to appropriate pool
                if let Some(pool) = resource_pools.get_mut(&resource.resource_type) {
                    pool.available_resources.push(resource);
                    pool.statistics.currently_allocated =
                        pool.statistics.currently_allocated.saturating_sub(1);
                }

                deallocated_resources.push(allocation_id.clone());
                false // Remove from allocations
            } else {
                true // Keep in allocations
            }
        });

        if !deallocated_resources.is_empty() {
            info!(
                "Released {} custom resources for test {}: {:?}",
                deallocated_resources.len(),
                test_id,
                deallocated_resources
            );
        }

        Ok(())
    }

    /// Check resource availability
    pub async fn check_availability(&self, resource_type: &str, count: usize) -> Result<bool> {
        let resource_pools = self.resource_pools.lock();

        if let Some(pool) = resource_pools.get(resource_type) {
            Ok(pool.available_resources.len() >= count)
        } else {
            Err(anyhow::anyhow!("Resource pool {} not found", resource_type))
        }
    }

    /// Get resource pool statistics
    pub async fn get_pool_statistics(
        &self,
        resource_type: &str,
    ) -> Result<CustomResourceStatistics> {
        let resource_pools = self.resource_pools.lock();

        if let Some(pool) = resource_pools.get(resource_type) {
            Ok(pool.statistics.clone())
        } else {
            Err(anyhow::anyhow!("Resource pool {} not found", resource_type))
        }
    }

    /// Get all resource pools
    pub async fn get_resource_pools(&self) -> HashMap<String, CustomResourcePool> {
        let resource_pools = self.resource_pools.lock();
        resource_pools.clone()
    }

    /// Get all allocations
    pub async fn get_allocations(&self) -> HashMap<String, CustomResourceAllocation> {
        let allocations = self.allocations.lock();
        allocations.clone()
    }

    /// Remove resource pool
    pub async fn remove_resource_pool(&self, resource_type: &str) -> Result<()> {
        let mut resource_pools = self.resource_pools.lock();
        let allocations = self.allocations.lock();

        // Check if any resources are currently allocated
        let has_allocated_resources = allocations
            .values()
            .any(|allocation| allocation.resource.resource_type == resource_type);

        if has_allocated_resources {
            return Err(anyhow::anyhow!(
                "Cannot remove resource pool {} with active allocations",
                resource_type
            ));
        }

        if resource_pools.remove(resource_type).is_some() {
            info!("Removed custom resource pool: {}", resource_type);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Resource pool {} not found", resource_type))
        }
    }

    /// Clean up expired allocations
    pub async fn cleanup_expired_allocations(&self) -> Result<usize> {
        let mut resource_pools = self.resource_pools.lock();
        let mut allocations = self.allocations.lock();
        let now = Utc::now();
        let mut cleaned_count = 0;

        // Collect expired allocations
        let mut expired_allocations = Vec::new();
        for (allocation_id, allocation) in allocations.iter() {
            if let Some(expected_release) = allocation.expected_release {
                if now > expected_release {
                    expired_allocations.push(allocation_id.clone());
                }
            }
        }

        // Clean up expired allocations
        for allocation_id in expired_allocations {
            if let Some(allocation) = allocations.remove(&allocation_id) {
                let mut resource = allocation.resource;
                resource.status = CustomResourceStatus::Available;

                if let Some(pool) = resource_pools.get_mut(&resource.resource_type) {
                    pool.available_resources.push(resource);
                    pool.statistics.currently_allocated =
                        pool.statistics.currently_allocated.saturating_sub(1);
                }

                cleaned_count += 1;
                warn!(
                    "Cleaned up expired custom resource allocation: {} from test {}",
                    allocation_id, allocation.test_id
                );
            }
        }

        if cleaned_count > 0 {
            info!(
                "Cleaned up {} expired custom resource allocations",
                cleaned_count
            );
        }

        Ok(cleaned_count)
    }

    /// Generate resource management report
    pub async fn generate_report(&self) -> String {
        let resource_pools = self.resource_pools.lock();
        let allocations = self.allocations.lock();

        let mut report = String::from("Custom Resource Management Report:\n");

        for (resource_type, pool) in resource_pools.iter() {
            let utilization = if pool.config.max_resources > 0 {
                pool.statistics.currently_allocated as f32 / pool.config.max_resources as f32
                    * 100.0
            } else {
                0.0
            };

            report.push_str(&format!(
                "\n{} Pool:\n\
                 - Available: {}\n\
                 - Allocated: {}\n\
                 - Total created: {}\n\
                 - Peak usage: {}\n\
                 - Utilization: {:.1}%\n\
                 - Average allocation duration: {}s\n",
                resource_type,
                pool.available_resources.len(),
                pool.statistics.currently_allocated,
                pool.statistics.total_created,
                pool.statistics.peak_usage,
                utilization,
                pool.statistics.average_duration.as_secs()
            ));
        }

        report.push_str(&format!(
            "\nTotal active allocations: {}\n",
            allocations.len()
        ));

        report
    }
}

impl Default for CustomResourceConfig {
    fn default() -> Self {
        Self {
            max_resources: 100,
            max_lifetime: Some(std::time::Duration::from_secs(3600)), // 1 hour
            auto_cleanup: true,
            properties: HashMap::new(),
        }
    }
}
