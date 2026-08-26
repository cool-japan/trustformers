//! Tests for the resource manager's real capacity accounting.
//!
//! These lock in the 0.2.1 fix: before it, `can_allocate` returned `Ok(true)`
//! for every request no matter how large, and `allocate_resources` stamped a
//! hardcoded `utilization: 0.8` / `efficiency: 1.0` onto every allocation it
//! handed back.

use super::resources::{AllocationFeasibility, ResourceManager};
use super::types::ResourceRequirement;
use crate::test_parallelization::ResourceManagementConfig;
use std::collections::HashMap;

/// A requirement that asks only for database connections, so the assertions
/// below turn purely on the configured pool size.
fn database_requirement(connections: usize) -> ResourceRequirement {
    ResourceRequirement {
        resource_type: "database_connection".to_string(),
        min_amount: connections as f64,
        cpu_cores: 0.0,
        memory_mb: 0,
        gpu_devices: Vec::new(),
        network_ports: 0,
        temp_directories: 0,
        database_connections: connections,
        custom_resources: HashMap::new(),
    }
}

/// A manager whose database pool holds exactly `connections` slots.
async fn manager_with_database_pool(connections: usize) -> ResourceManager {
    let mut config = ResourceManagementConfig::default();
    config.resource_pools.database_pool.max_connections = connections;
    ResourceManager::new(config).await.expect("resource manager should build")
}

#[tokio::test]
async fn a_request_larger_than_capacity_is_refused() {
    let manager = manager_with_database_pool(2).await;

    assert_eq!(
        manager.capacity().database_connections,
        2,
        "capacity must come from the configured pool"
    );
    assert!(
        matches!(
            manager.feasibility(&database_requirement(3)),
            AllocationFeasibility::ExceedsCapacity(_)
        ),
        "3 connections can never fit a 2-connection pool"
    );
    assert!(
        !manager
            .can_allocate(&database_requirement(3))
            .await
            .expect("feasibility check should succeed"),
        "can_allocate must not answer true for an impossible request"
    );
}

#[tokio::test]
async fn live_allocations_consume_capacity() {
    let manager = manager_with_database_pool(2).await;

    assert!(
        manager
            .can_allocate(&database_requirement(2))
            .await
            .expect("feasibility check should succeed"),
        "the whole pool must be grantable while nothing is held"
    );

    let allocation = manager
        .allocate_resources(&database_requirement(2), "holder")
        .await
        .expect("the whole pool should be allocatable");

    // The reservation is now live, so nothing further fits until it is released.
    assert!(
        matches!(
            manager.feasibility(&database_requirement(1)),
            AllocationFeasibility::WaitForCapacity(_)
        ),
        "a live allocation must reduce what remains"
    );
    assert!(
        manager.allocate_resources(&database_requirement(1), "second").await.is_err(),
        "over-committing must fail rather than silently succeed"
    );

    manager
        .release_allocation(&allocation.resource_id)
        .await
        .expect("release should find it");
    assert!(
        matches!(
            manager.feasibility(&database_requirement(2)),
            AllocationFeasibility::Grantable
        ),
        "releasing must give the capacity back"
    );
}

/// Nothing in this crate samples per-allocation utilization or efficiency, so
/// both must be structurally absent -- never the old `0.8` / `1.0` constants.
#[tokio::test]
async fn an_allocation_reports_no_unmeasured_numbers() {
    let manager = manager_with_database_pool(1).await;

    let allocation = manager
        .allocate_resources(&database_requirement(1), "measure-me")
        .await
        .expect("a single connection should be allocatable");

    assert!(
        allocation.utilization.is_none(),
        "utilization was never measured"
    );
    assert!(
        allocation.efficiency.is_none(),
        "efficiency was never measured"
    );
}
