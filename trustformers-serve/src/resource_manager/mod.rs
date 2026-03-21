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

// Allow dead code for resource manager infrastructure under development
#![allow(dead_code)]
//! - [`network_ports`] - Network port allocation and management
//! - [`temp_directories`] - Temporary directory lifecycle management
//! - [`gpu_resources`] - GPU device allocation and monitoring
//! - [`database_connections`] - Database connection pool management
//! - [`custom_resources`] - Custom resource type management
//! - [`allocation`] - Resource allocation coordination and conflict detection
//! - [`monitoring`] - Resource monitoring and performance tracking
//! - [`cleanup`] - Resource cleanup and lifecycle management
//! - [`manager`] - Main ResourceManagementSystem orchestrating all components

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
