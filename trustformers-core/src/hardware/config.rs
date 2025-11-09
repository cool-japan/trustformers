// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hardware configuration types and utilities
//!
//! This module provides configuration structs and enums for hardware management,
//! including allocation strategies, load balancing, and memory management settings.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

use super::traits::DeviceStatus;
use super::{HardwareCapabilities, HardwareType};

/// Hardware manager configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareManagerConfig {
    /// Auto-discovery enabled
    pub auto_discovery: bool,
    /// Device timeout in seconds
    pub device_timeout: u64,
    /// Health check interval in seconds
    pub health_check_interval: u64,
    /// Resource allocation strategy
    pub allocation_strategy: AllocationStrategy,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Performance monitoring enabled
    pub performance_monitoring: bool,
    /// Failover enabled
    pub failover_enabled: bool,
    /// Maximum concurrent operations per device
    pub max_concurrent_operations: usize,
    /// Memory management settings
    pub memory_management: MemoryManagementConfig,
}

/// Resource allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// First available device
    FirstAvailable,
    /// Best fit based on requirements
    BestFit,
    /// Round robin allocation
    RoundRobin,
    /// Load-aware allocation
    LoadAware,
    /// Performance-optimized allocation
    PerformanceOptimized,
    /// Power-efficient allocation
    PowerEfficient,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round robin
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Least utilization
    LeastUtilization,
    /// Weighted round robin
    WeightedRoundRobin,
    /// Performance-based
    PerformanceBased,
    /// Adaptive load balancing
    Adaptive,
}

/// Memory management configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryManagementConfig {
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Pool size per device (bytes)
    pub pool_size: usize,
    /// Memory fragmentation threshold
    pub fragmentation_threshold: f64,
    /// Garbage collection enabled
    pub gc_enabled: bool,
    /// GC interval in seconds
    pub gc_interval: u64,
    /// Memory pressure threshold
    pub pressure_threshold: f64,
}

/// Device information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device ID
    pub id: String,
    /// Hardware type
    pub hardware_type: HardwareType,
    /// Device capabilities
    pub capabilities: HardwareCapabilities,
    /// Current status
    pub status: DeviceStatus,
    /// Last seen timestamp
    pub last_seen: SystemTime,
    /// Device weight for load balancing
    pub weight: f64,
    /// Priority level
    pub priority: i32,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Operation statistics for performance tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationStats {
    /// Total operations executed
    pub total_operations: u64,
    /// Successful operations
    pub successful_operations: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// Average execution time in milliseconds
    pub avg_execution_time: f64,
    /// Peak execution time in milliseconds
    pub peak_execution_time: f64,
    /// Min execution time in milliseconds
    pub min_execution_time: f64,
    /// Operations per second
    pub operations_per_second: f64,
    /// Last updated timestamp
    pub last_updated: SystemTime,
}

/// Memory usage statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    /// Total memory in bytes
    pub total_memory: usize,
    /// Used memory in bytes
    pub used_memory: usize,
    /// Free memory in bytes
    pub free_memory: usize,
    /// Memory utilization percentage
    pub utilization: f64,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
}

impl Default for HardwareManagerConfig {
    fn default() -> Self {
        Self {
            auto_discovery: true,
            device_timeout: 30,
            health_check_interval: 60,
            allocation_strategy: AllocationStrategy::BestFit,
            load_balancing: LoadBalancingStrategy::LeastUtilization,
            performance_monitoring: true,
            failover_enabled: true,
            max_concurrent_operations: 4,
            memory_management: MemoryManagementConfig::default(),
        }
    }
}

impl Default for MemoryManagementConfig {
    fn default() -> Self {
        Self {
            enable_pooling: true,
            pool_size: 1024 * 1024 * 1024, // 1GB
            fragmentation_threshold: 0.3,
            gc_enabled: true,
            gc_interval: 300, // 5 minutes
            pressure_threshold: 0.8,
        }
    }
}

impl Default for OperationStats {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            avg_execution_time: 0.0,
            peak_execution_time: 0.0,
            min_execution_time: f64::INFINITY,
            operations_per_second: 0.0,
            last_updated: SystemTime::now(),
        }
    }
}

impl Default for MemoryUsageStats {
    fn default() -> Self {
        Self {
            total_memory: 0,
            used_memory: 0,
            free_memory: 0,
            utilization: 0.0,
            peak_usage: 0,
        }
    }
}
