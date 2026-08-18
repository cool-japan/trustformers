//! Resource Management System for Test Parallelization
//!
//! This module provides comprehensive resource management capabilities including
//! resource tracking, allocation, conflict detection, and cleanup for parallel
//! test execution in the TrustformeRS framework.
//!
//! ## Architecture
//!
//! The resource management system is organized into several focused modules:
//! - [`types`] - Core types and data structures for resource management
//! - [`network_ports`] - Network port allocation and management
//! - [`temp_directories`] - Temporary directory lifecycle management
//! - [`gpu_resources`] - GPU device allocation and monitoring
//! - [`database_connections`] - Database connection pool management
//! - [`custom_resources`] - Custom resource type management
//! - [`allocation`] - Resource allocation coordination and conflict detection
//! - [`monitoring`] - Resource monitoring and performance tracking
//! - [`cleanup`] - Resource cleanup and lifecycle management
//! - [`manager`] - Main ResourceManagementSystem orchestrating all components
//!
//! ## Status of the sub-managers: placeholders, not implementations
//!
//! Read this before depending on anything in this module. The coordination
//! layer here is real — `allocation` performs genuine conflict detection
//! against registered claims, and the allocation ledger and history it keeps are
//! accurate — but four of the sub-managers do not allocate the resources they
//! report:
//!
//! * `network_ports::NetworkPortManager::allocate_ports` returns `vec![8080]`
//!   regardless of how many ports were asked for, and reserves nothing.
//! * `temp_directories::TempDirectoryManager::allocate_directories` returns
//!   `/tmp/test-{id}-dir-{n}` path strings without creating any directory, then
//!   records them as `DirectoryStatus::Allocated`.
//! * `database_connections::DatabaseConnectionManager::allocate_connections`
//!   returns synthesised connection identifiers backed by no connection.
//! * `gpu_resources::GpuResourceManager::allocate_devices` echoes the requested
//!   device indices back without checking availability, and
//!   `monitoring` reports a constant `0.75` where it documents a computed
//!   efficiency score.
//!
//! Each of those sites carries a "For now …/In a real implementation" comment
//! at the point of the shortcut. The genuinely implemented equivalents live in
//! the sibling `crate::resource_management` module, whose port manager keeps a
//! real pool and fails when it is exhausted, and whose directory manager creates
//! and cleans real directories. Note that `crate::resource_management` is
//! re-exported from the crate root under prefixed names
//! (`ModularResourceManagementSystem`, `ModularNetworkPortManager`, …) while the
//! unprefixed `ResourceManagementSystem` currently refers to *this* module.
//!
//! Resolving that — either by implementing these four managers for real or by
//! deleting this tree in favour of `resource_management` and re-pointing the
//! unprefixed exports — is a breaking change to the crate's public surface and
//! is deliberately left for a coordinated pass rather than done silently here.

// Allow dead code for resource manager infrastructure under development.
// Placed after the module documentation: an inner attribute wedged between two
// `//!` lines splits the rendered module doc in half.
#![allow(dead_code)]

pub mod allocation;
pub mod cleanup;
pub mod custom_resources;
pub mod database_connections;
pub mod gpu_resources;
pub mod manager;
pub mod monitoring;
pub mod network_ports;
pub mod temp_directories;
pub mod types;

// Re-export main types for backward compatibility
pub use manager::ResourceManagementSystem;
pub use types::*;

// Re-export component types for easy access
pub use allocation::{AllocationEvent, ConflictDetector, ResourceAllocator};
pub use cleanup::CleanupManager;
pub use custom_resources::CustomResourceManager;
pub use database_connections::{DatabaseConnectionManager, DatabaseUsageStatistics};
pub use gpu_resources::{GpuAllocation, GpuDeviceInfo, GpuResourceManager, GpuUsageStatistics};
pub use monitoring::{
    AlertSystem, HealthChecker, LoadMetrics, ResourceMonitor, SystemResourceStatistics,
    SystemStatistics, WorkerPool,
};
pub use network_ports::{NetworkPortManager, PortAllocation, PortUsageStatistics, PortUsageType};
pub use temp_directories::{
    DirectoryStatus, DirectoryUsageStatistics, TempDirectoryInfo, TempDirectoryManager,
};
