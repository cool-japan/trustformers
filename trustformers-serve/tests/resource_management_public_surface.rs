//! Regression guard for the unprefixed resource-management surface.
//!
//! Until 0.2.1 this crate carried two resource-management module trees. The real
//! one, `resource_management`, was exported from the crate root under *prefixed*
//! names (`ModularResourceManagementSystem`, `ModularTempDirectoryManager`, …).
//! The unprefixed names — `ResourceManagementSystem`, `TempDirectoryManager`,
//! `TempDirectoryInfo`, `DirectoryStatus`, `PortUsageType` — resolved instead to
//! a second tree, `resource_manager`, whose sub-managers reported allocations
//! they never performed:
//!
//! * `allocate_ports` returned `vec![8080]` for every request, whatever the
//!   count, and reserved nothing, so a pool could never be exhausted;
//! * `allocate_directories` returned `/tmp/test-{id}-dir-{n}` path strings
//!   without creating a single directory, then recorded them as
//!   `DirectoryStatus::Allocated`;
//! * `allocate_connections` synthesised connection identifiers backed by no
//!   connection, and `allocate_devices` echoed the requested GPU indices back
//!   unchecked.
//!
//! The placeholder tree was deleted in 0.2.1 and the unprefixed names now point
//! at the real implementations. Every test below fails against the pre-0.2.1
//! exports: the two type-identity checks do not compile, and the two behavioural
//! checks observe fabricated results (a "successful" over-allocation, and
//! directories that do not exist).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use trustformers_serve::parallel_execution_engine::ResourceRequirement;
use trustformers_serve::resource_management::{ResourceManagementConfig, TempDirPoolConfig};
use trustformers_serve::{ResourceManagementSystem, TempDirectoryManager};

/// A base directory for one test case, under the OS temporary directory.
///
/// `case` keeps concurrently running tests in separate subtrees: nextest runs
/// each test in its own process, so a shared base path would have them creating
/// and removing the same directories at the same time.
fn case_base_path(case: &str) -> PathBuf {
    std::env::temp_dir()
        .join("trustformers-serve-resource-management-surface")
        .join(case)
}

/// Remove a case directory, ignoring the "it was never created" case.
fn remove_case_dir(path: &Path) {
    let _ = std::fs::remove_dir_all(path);
}

/// A requirement asking only for `network_ports` ports.
fn port_requirement(count: usize) -> ResourceRequirement {
    ResourceRequirement {
        resource_type: "network_port".to_string(),
        min_amount: count as f64,
        cpu_cores: 0.0,
        memory_mb: 0,
        gpu_devices: Vec::new(),
        network_ports: count,
        temp_directories: 0,
        database_connections: 0,
        custom_resources: HashMap::new(),
    }
}

/// A requirement asking only for `temp_directories` directories.
fn directory_requirement(count: usize) -> ResourceRequirement {
    ResourceRequirement {
        resource_type: "temp_directory".to_string(),
        min_amount: count as f64,
        cpu_cores: 0.0,
        memory_mb: 0,
        gpu_devices: Vec::new(),
        network_ports: 0,
        temp_directories: count,
        database_connections: 0,
        custom_resources: HashMap::new(),
    }
}

/// A configuration whose port pool holds exactly the ports in `port_range` and
/// whose temporary directories live under `base_path`.
fn config_for(case: &str, port_range: (u16, u16)) -> ResourceManagementConfig {
    let mut config = ResourceManagementConfig::default();
    config.resource_pools.network_port_pool.port_range = port_range;
    config.resource_pools.network_port_pool.reserved_ranges = Vec::new();
    config.resource_pools.temp_directory_pool.base_path = case_base_path(case);
    config
}

/// The unprefixed `ResourceManagementSystem` must *be* the modular one, not a
/// second implementation that merely shares its name.
///
/// This is a compile-time assertion: it type-checks only while both names
/// resolve to the same item. Against the pre-0.2.1 exports it did not compile,
/// because the two names denoted two different structs.
fn resource_management_system_identity(
    value: trustformers_serve::ModularResourceManagementSystem,
) -> ResourceManagementSystem {
    value
}

/// Same assertion for the temporary-directory manager, whose placeholder twin
/// was the one that reported directories it never created.
fn temp_directory_manager_identity(
    value: trustformers_serve::ModularTempDirectoryManager,
) -> TempDirectoryManager {
    value
}

/// Same assertion for the three remaining re-exported names, against the module
/// they are supposed to come from.
fn plain_data_identities(
    info: trustformers_serve::resource_management::TempDirectoryInfo,
    status: trustformers_serve::resource_management::DirectoryStatus,
    usage: trustformers_serve::resource_management::PortUsageType,
) -> (
    trustformers_serve::TempDirectoryInfo,
    trustformers_serve::DirectoryStatus,
    trustformers_serve::PortUsageType,
) {
    (info, status, usage)
}

#[tokio::test]
async fn unprefixed_names_denote_the_modular_implementations() {
    // Exercising the identity functions is what keeps them from being dead
    // code; the assertion they carry is discharged by the compiler.
    let case = "identity";
    let base = case_base_path(case);
    remove_case_dir(&base);

    let system = ResourceManagementSystem::new(config_for(case, (18_000, 18_010)))
        .await
        .expect("system initialises");
    let system = resource_management_system_identity(system);
    assert_eq!(system.active_allocation_count(), 0);

    let manager = TempDirectoryManager::new(TempDirPoolConfig {
        base_path: base.join("manager"),
        ..TempDirPoolConfig::default()
    })
    .await
    .expect("temp directory manager initialises");
    let manager = temp_directory_manager_identity(manager);
    assert_eq!(manager.get_allocated_directory_count().await, 0);

    let paths = manager.allocate_directories(1, "identity").await.expect("one directory");
    let info = manager
        .get_directory_allocations()
        .await
        .into_values()
        .next()
        .expect("the allocation is tracked")
        .directory;
    let (info, status, usage) = plain_data_identities(
        info,
        trustformers_serve::resource_management::DirectoryStatus::Allocated,
        trustformers_serve::resource_management::PortUsageType::HttpServer,
    );
    assert_eq!(info.path, PathBuf::from(&paths[0]));
    assert!(matches!(
        status,
        trustformers_serve::DirectoryStatus::Allocated
    ));
    assert!(matches!(
        usage,
        trustformers_serve::PortUsageType::HttpServer
    ));

    remove_case_dir(&base);
}

/// Regression: the placeholder port manager returned `vec![8080]` for every
/// request and reserved nothing, so asking a two-port pool for three ports
/// succeeded. The real manager fails, and says how many ports it had.
#[tokio::test]
async fn allocating_more_ports_than_the_pool_holds_is_refused() {
    let case = "port-exhaustion";
    let base = case_base_path(case);
    remove_case_dir(&base);

    // 18_100..=18_101 is two ports, neither well-known nor reserved.
    let system = ResourceManagementSystem::new(config_for(case, (18_100, 18_101)))
        .await
        .expect("system initialises");

    let error = system
        .allocate_resources(&port_requirement(3), "greedy-test")
        .await
        .expect_err("a two-port pool cannot satisfy a three-port request");
    let message = format!("{error:#}");
    assert!(
        message.contains("Insufficient available ports"),
        "the failure must name the exhausted pool, got: {message}"
    );
    assert_eq!(
        system.active_allocation_count(),
        0,
        "a refused allocation must not be recorded as live"
    );

    // The pool is intact, so a request it *can* satisfy still succeeds.
    system
        .allocate_resources(&port_requirement(2), "modest-test")
        .await
        .expect("two ports are available");
    assert_eq!(system.active_allocation_count(), 1);

    remove_case_dir(&base);
}

/// Regression: the placeholder directory manager returned
/// `/tmp/test-{id}-dir-{n}` strings and created nothing, while recording them as
/// `DirectoryStatus::Allocated`. The real manager creates each directory under
/// the configured base path before reporting it.
#[tokio::test]
async fn allocated_temporary_directories_exist_on_disk() {
    let case = "directories-exist";
    let base = case_base_path(case);
    remove_case_dir(&base);

    let system = ResourceManagementSystem::new(config_for(case, (18_200, 18_210)))
        .await
        .expect("system initialises");

    system
        .allocate_resources(&directory_requirement(3), "dir-test")
        .await
        .expect("three directories are available");

    for index in 0..3 {
        let expected = base.join(format!("test_dir-test_{index}"));
        assert!(
            expected.is_dir(),
            "directory {expected:?} was reported as allocated but does not exist"
        );
    }

    remove_case_dir(&base);
}

/// Regression: the placeholder manager handed the same `8080` to every caller,
/// so two tests could hold "the same" port simultaneously. The real manager
/// hands out disjoint ports and reports them individually.
#[tokio::test]
async fn concurrently_allocated_ports_are_disjoint() {
    let case = "disjoint-ports";
    let base = case_base_path(case);
    remove_case_dir(&base);

    let system = ResourceManagementSystem::new(config_for(case, (18_300, 18_309)))
        .await
        .expect("system initialises");

    system
        .allocate_resources(&port_requirement(3), "first")
        .await
        .expect("first allocation");
    system
        .allocate_resources(&port_requirement(3), "second")
        .await
        .expect("second allocation");
    assert_eq!(system.active_allocation_count(), 2);

    // Ten ports minus two three-port grants leaves four; a five-port request
    // must therefore fail, which is only true if the first two really reserved
    // six distinct ports.
    let error = system
        .allocate_resources(&port_requirement(5), "third")
        .await
        .expect_err("only four ports remain");
    assert!(
        format!("{error:#}").contains("Insufficient available ports"),
        "unexpected error: {error:#}"
    );

    remove_case_dir(&base);
}
