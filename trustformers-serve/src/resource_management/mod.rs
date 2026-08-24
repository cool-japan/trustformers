//! Resource Management System for Test Parallelization
//!
//! This module provides comprehensive resource management capabilities including
//! resource tracking, allocation, conflict detection, and cleanup for parallel
//! test execution in the TrustformeRS framework.
//!
//! ## Architecture
//!
//! The resource management system is organized into several focused modules:
//! - [`types`] - Core types, configurations, and shared data structures
//! - [`manager`] - Main ResourceManagementSystem coordinating all components
//! - [`port_management`] - Network port allocation and reservation
//! - [`directory_management`] - Temporary directory management and cleanup
//! - [`gpu_manager`] - GPU resource allocation and monitoring
//! - [`database_management`] - Database connection pool management
//! - [`custom_resources`] - Generic custom resource handling
//! - [`monitoring`] - Resource monitoring and health checks
//! - [`allocation`] - Resource allocation strategies and tracking
//! - [`cleanup`] - Resource cleanup and garbage collection
//! - [`statistics`] - Performance metrics and analytics
//!
//! ## Removed in 0.2.1: superseded orphan trees
//!
//! Three parallel implementations were deleted rather than mounted, because each
//! duplicated a module that is already declared here and in use:
//!
//! * `port_manager.rs` and the `port_manager/` subtree (~6,000 lines). Neither
//!   was ever declared, so neither had been compiled; `port_manager.rs` declared
//!   a `port_manager` submodule whose file does not exist, so it could not have
//!   compiled if it had been. [`port_management`] provides the
//!   [`NetworkPortManager`] that [`manager::ResourceManagementSystem`] actually
//!   allocates ports through.
//! * `gpu_management.rs` (~770 lines), superseded by the declared and live
//!   [`gpu_manager`] tree.
//! * `temp_dir_manager_legacy.rs`, superseded by the declared and live
//!   [`temp_dir_manager`] tree, and `types_data_tests.rs`, a test file for
//!   [`types_data`] that was never included by any `mod` declaration and so had
//!   never run.
//!
//! Keeping unreachable duplicates in a published crate is worse than removing
//! them: they cannot be exercised by any test, they drift silently against the
//! code that is live, and a reader cannot tell which of the two implementations
//! is the real one.
//!
//! ## Removed in 0.2.1: the module-wide `#![allow(dead_code)]`
//!
//! The blanket allow at the top of this module was hiding 19 warnings. Every one was a
//! private item that nothing read: the fields have been removed together with
//! the constructor arguments that fed them. The lint is enabled now, so the
//! next unread field is reported instead of accumulating.
//!
//! One targeted `#[allow(dead_code)]` outlived that sweep, on
//! `gpu_manager::manager::GpuResourceManager::create_mock_device`. Its doc
//! comment said it was "retained for unit-test use", but nothing called it —
//! not the discovery path, not a test. What it did was manufacture a GPU
//! inventory: device 0 was an "NVIDIA GeForce RTX 4090" with 24 GB and CUDA
//! 12.0, device 1 a "Tesla V100" with 32 GB, each with a capability list
//! naming PyTorch, TensorFlow and JAX. Sixty-five lines of invented hardware
//! one call away from the real `query_nvidia_devices` path. Deleted in 0.2.1,
//! and with it the last `allow(dead_code)` in this module tree.

pub mod allocation;
pub mod cleanup;
pub mod custom_resources;
pub mod database_management;
pub mod directory_management;
pub mod gpu_manager;
pub mod manager;
pub mod monitoring;
pub mod port_management;
pub mod statistics;
pub mod temp_dir_manager;
pub mod types;
pub mod types_data;

// Re-export main types for backward compatibility
pub use manager::ResourceManagementSystem;
pub use types::*;
// types_data is re-exported via types.rs

// Re-export component types for easy access
pub use allocation::{LoadMetrics, ResourceAllocator, WorkerPool};
pub use cleanup::{CleanupEvent, CleanupManager, CleanupTask};
pub use custom_resources::CustomResourceManager;
pub use database_management::{DatabaseSlot, DatabaseSlotAllocator, DatabaseType};
pub use directory_management::{DirectoryUsageTracking, TempDirectoryInfo, TempDirectoryManager};
pub use gpu_manager::{
    GpuAllocation, GpuDeviceInfo, GpuMonitoringSystem, GpuPerformanceTracker, GpuResourceManager,
};
pub use monitoring::{AlertSystem, HealthChecker, ResourceMonitor};
pub use port_management::{NetworkPortManager, PortAllocation, PortReservationSystem};
pub use statistics::{
    AnalyticsEngine, MetricsAggregator, PerformanceAnomaly, PerformanceBottleneck,
    PerformancePrediction, ReportGenerator, ResourceUtilizationSnapshot, StatisticsCollector,
    SystemMetrics,
};
