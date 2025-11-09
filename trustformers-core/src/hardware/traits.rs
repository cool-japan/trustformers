// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hardware abstraction traits for TrustformeRS
//!
//! This module defines the core traits that hardware backends must implement
//! to integrate with the TrustformeRS ecosystem. These traits provide a unified
//! interface for tensor operations, memory management, and device control.

use super::{HardwareCapabilities, HardwareConfig, HardwareMetrics, HardwareResult, HardwareType};
use crate::tensor::Tensor;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core trait for hardware devices
#[async_trait]
pub trait HardwareDevice: Send + Sync {
    /// Get device identifier
    fn device_id(&self) -> &str;

    /// Get hardware type
    fn hardware_type(&self) -> HardwareType;

    /// Get device capabilities
    fn capabilities(&self) -> &HardwareCapabilities;

    /// Initialize the device
    async fn initialize(&mut self, config: &HardwareConfig) -> HardwareResult<()>;

    /// Shutdown the device
    async fn shutdown(&mut self) -> HardwareResult<()>;

    /// Check if device is available
    fn is_available(&self) -> bool;

    /// Get current device status
    fn status(&self) -> DeviceStatus;

    /// Get current metrics
    async fn metrics(&self) -> HardwareResult<HardwareMetrics>;

    /// Reset device state
    async fn reset(&mut self) -> HardwareResult<()>;

    /// Allocate memory on device
    async fn allocate_memory(&mut self, size: usize) -> HardwareResult<DeviceMemory>;

    /// Free memory on device
    async fn free_memory(&mut self, memory: DeviceMemory) -> HardwareResult<()>;

    /// Synchronize device operations
    async fn synchronize(&self) -> HardwareResult<()>;
}

/// Backend trait for hardware implementations
#[async_trait]
pub trait HardwareBackend: Send + Sync {
    /// Backend name
    fn name(&self) -> &str;

    /// Backend version
    fn version(&self) -> &str;

    /// Discover available devices
    async fn discover_devices(&self) -> HardwareResult<Vec<Box<dyn HardwareDevice>>>;

    /// Create device from configuration
    async fn create_device(
        &self,
        config: &HardwareConfig,
    ) -> HardwareResult<Box<dyn HardwareDevice>>;

    /// Check backend compatibility
    fn is_compatible(&self, hardware_type: HardwareType) -> bool;

    /// Get supported operations
    fn supported_operations(&self) -> &[String];

    /// Validate configuration
    fn validate_config(&self, config: &HardwareConfig) -> HardwareResult<()>;
}

/// Hardware operation trait
#[async_trait]
pub trait HardwareOperation: Send + Sync {
    /// Operation name
    fn name(&self) -> &str;

    /// Execute operation on device
    async fn execute(
        &self,
        device: &mut dyn HardwareDevice,
        inputs: &[Tensor],
        outputs: &mut [Tensor],
        params: &HashMap<String, OperationParameter>,
    ) -> HardwareResult<()>;

    /// Validate operation parameters
    fn validate_params(&self, params: &HashMap<String, OperationParameter>) -> HardwareResult<()>;

    /// Get operation requirements
    fn requirements(&self) -> OperationRequirements;

    /// Estimate operation cost
    fn estimate_cost(&self, inputs: &[Tensor], params: &HashMap<String, OperationParameter>)
        -> f64;
}

/// Device status information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceStatus {
    /// Device is online and ready
    pub online: bool,
    /// Device is busy processing
    pub busy: bool,
    /// Error state
    pub error: Option<String>,
    /// Memory usage
    pub memory_usage: MemoryUsage,
    /// Temperature
    pub temperature: Option<f64>,
    /// Power consumption
    pub power_consumption: Option<f64>,
    /// Utilization percentage
    pub utilization: f64,
}

/// Memory usage information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Total memory in bytes
    pub total: usize,
    /// Used memory in bytes
    pub used: usize,
    /// Free memory in bytes
    pub free: usize,
    /// Fragmentation ratio
    pub fragmentation: f64,
}

/// Device memory handle
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceMemory {
    /// Memory address
    pub address: usize,
    /// Memory size in bytes
    pub size: usize,
    /// Memory type
    pub memory_type: MemoryType,
    /// Device identifier
    pub device_id: String,
}

/// Memory types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// Device local memory
    Local,
    /// Host memory
    Host,
    /// Shared memory
    Shared,
    /// Unified memory
    Unified,
    /// Persistent memory
    Persistent,
    /// Cache memory
    Cache,
}

/// Operation parameter types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperationParameter {
    /// Integer parameter
    Integer(i64),
    /// Float parameter
    Float(f64),
    /// String parameter
    String(String),
    /// Boolean parameter
    Boolean(bool),
    /// Array parameter
    Array(Vec<OperationParameter>),
    /// Object parameter
    Object(HashMap<String, OperationParameter>),
}

/// Operation requirements
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationRequirements {
    /// Minimum memory required
    pub min_memory: usize,
    /// Required compute units
    pub compute_units: Option<u32>,
    /// Required data types
    pub data_types: Vec<super::DataType>,
    /// Required capabilities
    pub capabilities: Vec<String>,
    /// Performance characteristics
    pub performance: PerformanceRequirements,
}

/// Performance requirements
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PerformanceRequirements {
    /// Maximum latency in milliseconds
    pub max_latency: Option<f64>,
    /// Minimum throughput in operations per second
    pub min_throughput: Option<f64>,
    /// Memory bandwidth requirements
    pub memory_bandwidth: Option<f64>,
    /// Power consumption limit
    pub power_limit: Option<f64>,
}

/// Async hardware operation trait
#[async_trait]
pub trait AsyncHardwareOperation: Send + Sync {
    /// Start async operation
    async fn start(
        &self,
        device: &mut dyn HardwareDevice,
        inputs: &[Tensor],
        params: &HashMap<String, OperationParameter>,
    ) -> HardwareResult<AsyncOperationHandle>;

    /// Check operation status
    async fn status(&self, handle: &AsyncOperationHandle) -> HardwareResult<AsyncOperationStatus>;

    /// Get operation results
    async fn results(&self, handle: &AsyncOperationHandle) -> HardwareResult<Vec<Tensor>>;

    /// Cancel operation
    async fn cancel(&self, handle: &AsyncOperationHandle) -> HardwareResult<()>;
}

/// Async operation handle
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsyncOperationHandle {
    /// Operation identifier
    pub id: String,
    /// Device identifier
    pub device_id: String,
    /// Operation name
    pub operation_name: String,
    /// Start time
    pub start_time: std::time::SystemTime,
}

/// Async operation status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AsyncOperationStatus {
    /// Operation is queued
    Queued,
    /// Operation is running
    Running,
    /// Operation completed successfully
    Completed,
    /// Operation failed
    Failed(String),
    /// Operation was cancelled
    Cancelled,
}

/// Hardware scheduler trait
pub trait HardwareScheduler: Send + Sync + std::fmt::Debug {
    /// Schedule operation on best available device
    fn schedule_operation(
        &self,
        operation: &dyn HardwareOperation,
        inputs: &[Tensor],
        params: &HashMap<String, OperationParameter>,
    ) -> HardwareResult<String>; // Returns device_id

    /// Get scheduler statistics
    fn statistics(&self) -> SchedulerStatistics;

    /// Update device priorities
    fn update_priorities(&mut self, priorities: HashMap<String, f64>);
}

/// Scheduler statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerStatistics {
    /// Total operations scheduled
    pub total_operations: u64,
    /// Operations per device
    pub operations_per_device: HashMap<String, u64>,
    /// Average scheduling time
    pub avg_scheduling_time: f64,
    /// Device utilization
    pub device_utilization: HashMap<String, f64>,
    /// Failed operations
    pub failed_operations: u64,
}

impl Default for DeviceStatus {
    fn default() -> Self {
        Self {
            online: false,
            busy: false,
            error: None,
            memory_usage: MemoryUsage::default(),
            temperature: None,
            power_consumption: None,
            utilization: 0.0,
        }
    }
}

impl Default for MemoryUsage {
    fn default() -> Self {
        Self {
            total: 0,
            used: 0,
            free: 0,
            fragmentation: 0.0,
        }
    }
}

impl Default for OperationRequirements {
    fn default() -> Self {
        Self {
            min_memory: 0,
            compute_units: None,
            data_types: vec![super::DataType::F32],
            capabilities: vec![],
            performance: PerformanceRequirements::default(),
        }
    }
}

impl Default for SchedulerStatistics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            operations_per_device: HashMap::new(),
            avg_scheduling_time: 0.0,
            device_utilization: HashMap::new(),
            failed_operations: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_status_default() {
        let status = DeviceStatus::default();
        assert!(!status.online);
        assert!(!status.busy);
        assert!(status.error.is_none());
        assert_eq!(status.utilization, 0.0);
    }

    #[test]
    fn test_memory_usage_calculation() {
        let mut usage = MemoryUsage::default();
        usage.total = 1000;
        usage.used = 600;
        usage.free = usage.total - usage.used;
        assert_eq!(usage.free, 400);
    }

    #[test]
    fn test_operation_parameter_types() {
        let int_param = OperationParameter::Integer(42);
        let float_param = OperationParameter::Float(std::f64::consts::PI);
        let _string_param = OperationParameter::String("test".to_string());
        let _bool_param = OperationParameter::Boolean(true);

        match int_param {
            OperationParameter::Integer(val) => assert_eq!(val, 42),
            _ => assert!(false, "Expected Integer parameter but got {:?}", int_param),
        }

        match float_param {
            OperationParameter::Float(val) => assert_eq!(val, std::f64::consts::PI),
            _ => assert!(false, "Expected Float parameter but got {:?}", float_param),
        }
    }

    #[test]
    fn test_async_operation_status() {
        let status = AsyncOperationStatus::Queued;
        assert_eq!(status, AsyncOperationStatus::Queued);

        let failed_status = AsyncOperationStatus::Failed("test error".to_string());
        match failed_status {
            AsyncOperationStatus::Failed(msg) => assert_eq!(msg, "test error"),
            _ => assert!(false, "Expected Failed status but got {:?}", failed_status),
        }
    }

    #[test]
    fn test_memory_type_equality() {
        assert_eq!(MemoryType::Local, MemoryType::Local);
        assert_ne!(MemoryType::Local, MemoryType::Host);
        assert_eq!(MemoryType::Shared, MemoryType::Shared);
    }
}
