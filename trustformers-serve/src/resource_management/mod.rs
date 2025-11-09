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
//! - [`gpu_management`] - GPU resource allocation and monitoring
//! - [`database_management`] - Database connection pool management

// Allow dead code for resource management infrastructure under development
#![allow(dead_code)]
//! - [`custom_resources`] - Generic custom resource handling
//! - [`monitoring`] - Resource monitoring and health checks
//! - [`allocation`] - Resource allocation strategies and tracking
//! - [`cleanup`] - Resource cleanup and garbage collection
//! - [`statistics`] - Performance metrics and analytics

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

// Re-export main types for backward compatibility
pub use manager::ResourceManagementSystem;
pub use types::*;

// Re-export component types for easy access
pub use allocation::{LoadMetrics, ResourceAllocator, WorkerPool};
pub use cleanup::{CleanupEvent, CleanupManager, CleanupTask};
pub use custom_resources::CustomResourceManager;
pub use database_management::DatabaseConnectionManager;
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
