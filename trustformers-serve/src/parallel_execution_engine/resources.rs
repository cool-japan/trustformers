//! Capacity accounting for the parallel execution engine.
//!
//! Split out of `types.rs` in 0.2.1 to keep every file in this module under the
//! 2000-line limit.

use super::types::{
    monitoring_config_from, AvailableResources, ResourceAllocationState, ResourceRequirement,
};
use crate::resource_management::{AllocationEvent, ResourceMonitor};
use crate::test_parallelization::ResourceAllocation;
use anyhow::Result;
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc};

/// Whether a [`ResourceRequirement`] can be satisfied right now.
///
/// The distinction matters to the execution loop: `WaitForCapacity` means
/// requeueing the test will eventually work because live allocations will be
/// released, whereas `ExceedsCapacity` can never be satisfied and must be
/// reported as an error instead of spun on forever.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationFeasibility {
    /// Enough uncommitted capacity exists to grant the request immediately.
    Grantable,
    /// The request fits the engine's total capacity but not what is free right
    /// now. The string names the resource that is currently exhausted.
    WaitForCapacity(String),
    /// The request is larger than the engine's total capacity and can never be
    /// granted. The string names the limit that is exceeded.
    ExceedsCapacity(String),
}
/// Sum of everything currently reserved by live allocations.
#[derive(Debug, Default, Clone, Copy)]
struct CommittedResources {
    cpu_cores: f32,
    memory_mb: u64,
    gpu_devices: usize,
    network_ports: usize,
    temp_directories: usize,
    database_connections: usize,
}
/// Resource manager for tracking and allocating test resources.
///
/// 0.2.1: `can_allocate` used to return `Ok(true)` unconditionally and
/// `allocate_resources` stamped a hardcoded `utilization: 0.8` /
/// `efficiency: 1.0` onto every allocation. Both are now real: capacity is
/// measured/configured once at construction (see [`AvailableResources::detect`]),
/// live allocations are summed against it on every request, and the
/// utilization/efficiency fields are left structurally absent (`None`) because
/// nothing in this crate measures them.
pub struct ResourceManager {
    /// Capacity this manager may hand out. Fixed for the manager's lifetime.
    capacity: AvailableResources,
    /// Live resource allocations, keyed by allocation ID
    allocations: Arc<Mutex<HashMap<String, ResourceAllocationState>>>,
    /// Resource monitoring
    _resource_monitor: Arc<ResourceMonitor>,
    /// Allocation history
    allocation_history: Arc<Mutex<Vec<AllocationEvent>>>,
}
impl ResourceManager {
    /// Build a manager whose capacity is measured/configured from `config`.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying resource monitor cannot start.
    pub async fn new(
        config: crate::test_parallelization::ResourceManagementConfig,
    ) -> Result<Self> {
        Ok(Self {
            capacity: AvailableResources::detect(&config.resource_pools),
            allocations: Arc::new(Mutex::new(HashMap::new())),
            _resource_monitor: Arc::new(
                ResourceMonitor::new(monitoring_config_from(&config.resource_monitoring)).await?,
            ),
            allocation_history: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Capacity this manager was built with.
    pub fn capacity(&self) -> &AvailableResources {
        &self.capacity
    }

    /// Total of everything currently reserved by live allocations.
    fn committed(&self) -> CommittedResources {
        let allocations = self.allocations.lock();
        let mut committed = CommittedResources::default();
        for state in allocations.values() {
            let requirement = &state.requirement;
            committed.cpu_cores += requirement.cpu_cores;
            committed.memory_mb = committed.memory_mb.saturating_add(requirement.memory_mb);
            committed.gpu_devices += requirement.gpu_devices.len();
            committed.network_ports += requirement.network_ports;
            committed.temp_directories += requirement.temp_directories;
            committed.database_connections += requirement.database_connections;
        }
        committed
    }

    /// Decide whether `requirements` can be granted, and if not, why.
    ///
    /// This is the honest core of the manager: every branch compares the
    /// request against a real limit (measured host CPU/memory, configured pool
    /// sizes) minus what live allocations already hold.
    pub fn feasibility(&self, requirements: &ResourceRequirement) -> AllocationFeasibility {
        let capacity = &self.capacity;
        let committed = self.committed();

        // (label, requested, total capacity, already committed)
        let checks: [(&str, f64, f64, f64); 6] = [
            (
                "CPU cores",
                f64::from(requirements.cpu_cores),
                f64::from(capacity.cpu_cores),
                f64::from(committed.cpu_cores),
            ),
            (
                "memory (MB)",
                requirements.memory_mb as f64,
                capacity.memory_mb as f64,
                committed.memory_mb as f64,
            ),
            (
                "GPU devices",
                requirements.gpu_devices.len() as f64,
                capacity.gpu_device_ids.len() as f64,
                committed.gpu_devices as f64,
            ),
            (
                "network ports",
                requirements.network_ports as f64,
                capacity.network_ports.len() as f64,
                committed.network_ports as f64,
            ),
            (
                "temporary directories",
                requirements.temp_directories as f64,
                capacity.temp_directory_slots as f64,
                committed.temp_directories as f64,
            ),
            (
                "database connections",
                requirements.database_connections as f64,
                capacity.database_connections as f64,
                committed.database_connections as f64,
            ),
        ];

        for (label, requested, total, in_use) in checks {
            if requested > total {
                return AllocationFeasibility::ExceedsCapacity(format!(
                    "{label}: requested {requested}, engine capacity {total}"
                ));
            }
            if requested > total - in_use {
                return AllocationFeasibility::WaitForCapacity(format!(
                    "{label}: requested {requested}, {} free of {total}",
                    total - in_use
                ));
            }
        }

        // Requested GPU device IDs must be members of the configured pool.
        for device_id in &requirements.gpu_devices {
            if !capacity.gpu_device_ids.contains(device_id) {
                return AllocationFeasibility::ExceedsCapacity(format!(
                    "GPU device {device_id} is not in the configured device pool {:?}",
                    capacity.gpu_device_ids
                ));
            }
        }

        for (name, requested) in &requirements.custom_resources {
            let total = capacity.custom_resources.get(name).copied().unwrap_or(0.0);
            if *requested > total {
                return AllocationFeasibility::ExceedsCapacity(format!(
                    "custom resource {name}: requested {requested}, engine capacity {total}"
                ));
            }
        }

        AllocationFeasibility::Grantable
    }

    /// Whether `requirements` can be granted right now.
    pub async fn can_allocate(&self, requirements: &ResourceRequirement) -> Result<bool> {
        Ok(self.feasibility(requirements) == AllocationFeasibility::Grantable)
    }

    /// Reserve `requirements` for `test_id`.
    ///
    /// Fails rather than over-committing if the request no longer fits — the
    /// feasibility check and the reservation are performed under the same lock
    /// so two concurrent callers cannot both pass a check for the last slot.
    pub async fn allocate_resources(
        &self,
        requirements: &ResourceRequirement,
        test_id: &str,
    ) -> Result<ResourceAllocation> {
        let allocation_id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now();
        let allocation = crate::test_parallelization::ResourceAllocation {
            resource_type: requirements.resource_type.clone(),
            resource_id: allocation_id.clone(),
            allocated_at: now,
            deallocated_at: None,
            duration: std::time::Duration::from_secs(0),
            // Not observed: nothing samples per-allocation utilization or
            // efficiency in this crate. See `ResourceAllocation`'s field docs.
            utilization: None,
            efficiency: None,
        };
        let allocation_state = ResourceAllocationState {
            allocated: allocation.clone(),
            requirement: requirements.clone(),
            allocated_at: now,
            expected_deallocation: None,
            efficiency: None,
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("test_id".to_string(), test_id.to_string());
                metadata
            },
        };
        {
            let mut allocations = self.allocations.lock();
            // Re-check under the lock so the reservation is atomic with respect
            // to other callers.
            let mut committed = CommittedResources::default();
            for state in allocations.values() {
                let held = &state.requirement;
                committed.cpu_cores += held.cpu_cores;
                committed.memory_mb = committed.memory_mb.saturating_add(held.memory_mb);
                committed.gpu_devices += held.gpu_devices.len();
                committed.network_ports += held.network_ports;
                committed.temp_directories += held.temp_directories;
                committed.database_connections += held.database_connections;
            }
            let fits = requirements.cpu_cores <= self.capacity.cpu_cores - committed.cpu_cores
                && requirements.memory_mb
                    <= self.capacity.memory_mb.saturating_sub(committed.memory_mb)
                && requirements.gpu_devices.len()
                    <= self.capacity.gpu_device_ids.len().saturating_sub(committed.gpu_devices)
                && requirements.network_ports
                    <= self.capacity.network_ports.len().saturating_sub(committed.network_ports)
                && requirements.temp_directories
                    <= self
                        .capacity
                        .temp_directory_slots
                        .saturating_sub(committed.temp_directories)
                && requirements.database_connections
                    <= self
                        .capacity
                        .database_connections
                        .saturating_sub(committed.database_connections);
            if !fits {
                return Err(anyhow::anyhow!(
                    "resource allocation for test {test_id} no longer fits available capacity"
                ));
            }
            allocations.insert(allocation_id.clone(), allocation_state);
        }
        let event = AllocationEvent {
            timestamp: now,
            resource_id: allocation_id.clone(),
            resource_type: allocation.resource_type.clone(),
            test_id: test_id.to_string(),
            event_type: "Allocated".to_string(),
            details: HashMap::new(),
        };
        self.allocation_history.lock().push(event);
        Ok(allocation)
    }

    /// Release exactly one allocation by its id, returning it with its real
    /// elapsed duration filled in.
    ///
    /// Prefer this over [`Self::release_resources_for_test`] whenever the
    /// allocation id is known: the `test_id` a reservation is filed under comes
    /// from `metadata.resource_usage.test_id`, which is not guaranteed to equal
    /// the `base_context.test_name` a completed execution reports, so releasing
    /// by name can miss the reservation and leak capacity.
    pub async fn release_allocation(&self, allocation_id: &str) -> Option<ResourceAllocation> {
        let now = chrono::Utc::now();
        let (allocation, test_id) = {
            let mut allocations = self.allocations.lock();
            let state = allocations.remove(allocation_id)?;
            let test_id = state.metadata.get("test_id").cloned().unwrap_or_default();
            let mut allocation = state.allocated;
            allocation.deallocated_at = Some(now);
            allocation.duration =
                (now - state.allocated_at).to_std().unwrap_or(std::time::Duration::ZERO);
            (allocation, test_id)
        };
        self.allocation_history.lock().push(AllocationEvent {
            timestamp: now,
            resource_id: allocation.resource_id.clone(),
            resource_type: allocation.resource_type.clone(),
            test_id,
            event_type: "Deallocated".to_string(),
            details: HashMap::new(),
        });
        Some(allocation)
    }

    /// Ids of every allocation currently held.
    pub fn live_allocation_ids(&self) -> Vec<String> {
        self.allocations.lock().keys().cloned().collect()
    }

    /// Release every allocation held for `test_id`, returning the released
    /// allocations with their real elapsed durations filled in.
    ///
    /// This is what makes the capacity accounting a loop rather than a ratchet:
    /// without it, `can_allocate` would refuse forever once the pool filled up.
    pub async fn release_resources_for_test(&self, test_id: &str) -> Vec<ResourceAllocation> {
        let now = chrono::Utc::now();
        let mut released = Vec::new();
        {
            let mut allocations = self.allocations.lock();
            let ids: Vec<String> = allocations
                .iter()
                .filter(|(_, state)| {
                    state.metadata.get("test_id").map(String::as_str) == Some(test_id)
                })
                .map(|(id, _)| id.clone())
                .collect();
            for id in ids {
                if let Some(state) = allocations.remove(&id) {
                    let mut allocation = state.allocated;
                    allocation.deallocated_at = Some(now);
                    allocation.duration =
                        (now - state.allocated_at).to_std().unwrap_or(std::time::Duration::ZERO);
                    released.push(allocation);
                }
            }
        }
        let mut history = self.allocation_history.lock();
        for allocation in &released {
            history.push(AllocationEvent {
                timestamp: now,
                resource_id: allocation.resource_id.clone(),
                resource_type: allocation.resource_type.clone(),
                test_id: test_id.to_string(),
                event_type: "Deallocated".to_string(),
                details: HashMap::new(),
            });
        }
        released
    }
}
