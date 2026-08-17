// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Hardware manager for TrustformeRS
//!
//! This module provides a centralized manager for hardware devices, backends,
//! and operations. It coordinates between different specialized components
//! to provide a unified hardware management interface.

#![allow(unused_variables)] // Hardware manager

use super::allocation::{LoadBalancer, MemoryManager, ResourceAllocator};
use super::backends::{CPUBackend, GPUBackend};
use super::config::{DeviceInfo, HardwareManagerConfig};
use super::devices::{CPUDevice, GPUBackendType, GPUDevice};
use super::monitoring::{HealthChecker, PerformanceMonitor};
use super::registry::HardwareRegistry;
use super::scheduling::{AdvancedScheduler, DefaultScheduler, SchedulingAlgorithm};
use super::traits::{HardwareBackend, HardwareOperation, HardwareScheduler};
use super::{
    HardwareMetrics, HardwareResult, HardwareType, OperationMode, OperationParameter, PrecisionMode,
};
use crate::errors::TrustformersError;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use tokio::sync::Mutex as AsyncMutex;

/// Hardware manager implementation
#[derive(Debug)]
pub struct HardwareManager {
    /// Manager configuration
    config: HardwareManagerConfig,
    /// Hardware registry
    #[allow(dead_code)]
    registry: Arc<RwLock<HardwareRegistry>>,
    /// CPU backend
    cpu_backend: Arc<AsyncMutex<CPUBackend>>,
    /// GPU backend
    gpu_backend: Arc<Mutex<Option<GPUBackend>>>,
    /// Device information cache
    device_info: Arc<RwLock<HashMap<String, DeviceInfo>>>,
    /// Device metrics cache
    device_metrics: Arc<RwLock<HashMap<String, HardwareMetrics>>>,
    /// Operation scheduler
    scheduler: Arc<Mutex<Box<dyn HardwareScheduler>>>,
    /// Performance monitor
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
    /// Health checker
    health_checker: Arc<Mutex<HealthChecker>>,
    /// Resource allocator
    #[allow(dead_code)]
    resource_allocator: Arc<Mutex<ResourceAllocator>>,
    /// Load balancer
    #[allow(dead_code)]
    load_balancer: Arc<Mutex<LoadBalancer>>,
    /// Memory manager
    #[allow(dead_code)]
    memory_manager: Arc<Mutex<MemoryManager>>,
}

impl HardwareManager {
    /// Create a new hardware manager
    pub fn new(config: HardwareManagerConfig) -> Self {
        let scheduler: Box<dyn HardwareScheduler> = if config.performance_monitoring {
            Box::new(AdvancedScheduler::new(SchedulingAlgorithm::LoadAware))
        } else {
            Box::new(DefaultScheduler::new())
        };

        Self {
            cpu_backend: Arc::new(AsyncMutex::new(CPUBackend::new())),
            gpu_backend: Arc::new(Mutex::new(None)),
            registry: Arc::new(RwLock::new(HardwareRegistry::new())),
            device_info: Arc::new(RwLock::new(HashMap::new())),
            device_metrics: Arc::new(RwLock::new(HashMap::new())),
            scheduler: Arc::new(Mutex::new(scheduler)),
            performance_monitor: Arc::new(Mutex::new(PerformanceMonitor::new())),
            health_checker: Arc::new(Mutex::new(HealthChecker::new())),
            resource_allocator: Arc::new(Mutex::new(ResourceAllocator::new(
                config.allocation_strategy,
            ))),
            load_balancer: Arc::new(Mutex::new(LoadBalancer::new(config.load_balancing))),
            memory_manager: Arc::new(Mutex::new(MemoryManager::new())),
            config,
        }
    }

    /// Initialize the hardware manager
    pub async fn initialize(&mut self) -> HardwareResult<()> {
        // Initialize CPU backend
        {
            let cpu_backend = self.cpu_backend.lock().await;
            // CPU backend initialization - no longer needed with new trait design
            self.register_backend_devices(&*cpu_backend).await?;
        }

        // Initialize GPU backend if available
        if self.detect_gpu_backend().is_some() {
            self.initialize_gpu_backend().await?;
        }

        // Start background tasks
        if self.config.performance_monitoring {
            self.start_monitoring().await?;
        }

        if self.config.health_check_interval > 0 {
            self.start_health_checks().await?;
        }

        Ok(())
    }

    /// Detect available GPU backend
    fn detect_gpu_backend(&self) -> Option<GPUBackendType> {
        #[cfg(feature = "cuda")]
        if self.is_cuda_available() {
            return Some(GPUBackendType::CUDA);
        }

        #[cfg(feature = "rocm")]
        if self.is_rocm_available() {
            return Some(GPUBackendType::ROCm);
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        if self.is_metal_available() {
            return Some(GPUBackendType::Metal);
        }

        #[cfg(feature = "opencl")]
        if self.is_opencl_available() {
            return Some(GPUBackendType::OpenCL);
        }

        #[cfg(feature = "vulkan")]
        if self.is_vulkan_available() {
            return Some(GPUBackendType::Vulkan);
        }

        None
    }

    // Delegate to `GPUBackend`'s real runtime probes (driver CLI exit
    // status / runtime library presence) instead of unconditionally
    // claiming every backend is available whenever its Cargo feature is
    // compiled in.
    #[cfg(feature = "cuda")]
    fn is_cuda_available(&self) -> bool {
        GPUBackend::is_cuda_available()
    }

    #[cfg(feature = "rocm")]
    fn is_rocm_available(&self) -> bool {
        GPUBackend::is_rocm_available()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn is_metal_available(&self) -> bool {
        // `is_metal_available` is an instance probe on `GPUBackend` (not yet
        // migrated to a static probe); constructing one here is cheap and
        // side-effect-free (see `GPUBackend::new`).
        GPUBackend::new(GPUBackendType::Metal).is_metal_available()
    }

    #[cfg(feature = "opencl")]
    fn is_opencl_available(&self) -> bool {
        GPUBackend::is_opencl_available()
    }

    #[cfg(feature = "vulkan")]
    fn is_vulkan_available(&self) -> bool {
        GPUBackend::is_vulkan_available()
    }

    /// Initialize GPU backend
    async fn initialize_gpu_backend(&mut self) -> HardwareResult<()> {
        if let Some(backend_type) = self.detect_gpu_backend() {
            let gpu_backend = GPUBackend::new(backend_type);
            // GPU backend initialization - no longer needed with new trait design
            self.register_backend_devices(&gpu_backend).await?;

            *self.gpu_backend.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) =
                Some(gpu_backend);
        }
        Ok(())
    }

    /// Register devices from a backend
    async fn register_backend_devices(&self, backend: &dyn HardwareBackend) -> HardwareResult<()> {
        let devices = backend.discover_devices().await?;
        let mut device_info =
            self.device_info.write().unwrap_or_else(|poisoned| poisoned.into_inner());

        for device in devices {
            let device_id = device.device_id().to_string();
            let info = DeviceInfo {
                id: device_id.clone(),
                hardware_type: device.hardware_type(),
                capabilities: device.capabilities().clone(),
                status: device.status(),
                last_seen: std::time::SystemTime::now(),
                weight: 1.0,
                priority: 0,
                tags: vec![],
            };

            device_info.insert(device_id, info);
        }

        Ok(())
    }

    /// Start performance monitoring
    async fn start_monitoring(&self) -> HardwareResult<()> {
        // Start background monitoring task (placeholder)
        Ok(())
    }

    /// Start health checking
    async fn start_health_checks(&self) -> HardwareResult<()> {
        // Start background health checking task (placeholder)
        Ok(())
    }

    /// Check if a device exists
    pub fn has_device(&self, device_id: &str) -> bool {
        self.device_info
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(device_id)
    }

    /// Get device information
    pub fn get_device_info(&self, device_id: &str) -> Option<DeviceInfo> {
        self.device_info
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(device_id)
            .cloned()
    }

    /// Get device metrics
    pub fn get_device_metrics(&self, device_id: &str) -> Option<HardwareMetrics> {
        self.device_metrics
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(device_id)
            .cloned()
    }

    /// List all available devices
    pub fn list_devices(&self) -> Vec<DeviceInfo> {
        self.device_info
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .cloned()
            .collect()
    }

    /// List devices by hardware type
    pub fn list_devices_by_type(&self, hardware_type: HardwareType) -> Vec<DeviceInfo> {
        self.device_info
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .filter(|info| info.hardware_type == hardware_type)
            .cloned()
            .collect()
    }

    /// Get the best device for an operation
    pub fn get_best_device(&self, operation: &dyn HardwareOperation) -> HardwareResult<String> {
        // Use scheduler to find the best device
        if let Ok(scheduler) = self.scheduler.lock() {
            let inputs = vec![]; // Placeholder - would pass actual inputs
            let params = HashMap::new(); // Placeholder - would pass actual params
            scheduler.schedule_operation(operation, &inputs, &params)
        } else {
            Err(TrustformersError::hardware_error(
                "Failed to lock scheduler",
                "schedule_operation",
            ))
        }
    }

    /// Execute an operation on the best available device
    pub fn execute_operation(
        &self,
        operation: &dyn HardwareOperation,
        inputs: &[Tensor],
        params: &HashMap<String, OperationParameter>,
    ) -> HardwareResult<Vec<Tensor>> {
        // Get the best device for this operation
        let device_id = self.get_best_device(operation)?;

        // Execute on the selected device
        self.execute_on_device(&device_id, operation, inputs, params)
    }

    /// Execute an operation on a specific device
    ///
    /// Dispatches by operation name to `CPUDevice::execute_operation` /
    /// `GPUDevice::execute_operation`, which perform real Tensor
    /// computation (see `hardware::devices`) for the operations they
    /// support, and return a structured error for anything they don't -
    /// never a copy of the input passed off as a computed result.
    pub fn execute_on_device(
        &self,
        device_id: &str,
        operation: &dyn HardwareOperation,
        inputs: &[Tensor],
        _params: &HashMap<String, OperationParameter>,
    ) -> HardwareResult<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(TrustformersError::hardware_error(
                "execute_on_device requires at least one input tensor",
                "execute_on_device",
            ));
        }

        // Determine which backend owns this device
        let device_info = self.get_device_info(device_id).ok_or_else(|| {
            TrustformersError::hardware_error("Device not found", "execute_on_device")
        })?;

        match device_info.hardware_type {
            HardwareType::CPU => {
                let device = CPUDevice::new(device_id.to_string());
                device.execute_operation(
                    operation.name(),
                    inputs,
                    OperationMode::Balanced,
                    PrecisionMode::Single,
                )
            },
            HardwareType::GPU => {
                // `DeviceInfo` does not currently record which
                // `GPUBackendType` a device id belongs to; fall back to
                // re-detecting one. This does not affect correctness of the
                // computed result: `GPUDevice::execute_operation` performs
                // the same real host computation for every backend type,
                // since none of them open a live accelerator context (see
                // `GPUDevice::new`).
                let backend_type = self.detect_gpu_backend().unwrap_or(GPUBackendType::Unknown);
                let device = GPUDevice::new(device_id.to_string(), backend_type);
                device.execute_operation(
                    operation.name(),
                    inputs,
                    OperationMode::Balanced,
                    PrecisionMode::Single,
                )
            },
            _ => Err(TrustformersError::hardware_error(
                "Unsupported hardware type",
                "execute_on_device",
            )),
        }
    }

    /// Update device metrics
    pub fn update_device_metrics(&self, device_id: &str, metrics: HardwareMetrics) {
        {
            let mut device_metrics =
                self.device_metrics.write().unwrap_or_else(|poisoned| poisoned.into_inner());
            device_metrics.insert(device_id.to_string(), metrics.clone());
        }

        // Update performance monitor
        if let Ok(mut monitor) = self.performance_monitor.lock() {
            monitor.update_metrics(device_id, &metrics);
        }
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> HashMap<String, f64> {
        if let Ok(monitor) = self.performance_monitor.lock() {
            monitor.efficiency_scores.clone()
        } else {
            HashMap::new()
        }
    }

    /// Get health status for all devices
    pub fn get_health_status(&self) -> HashMap<String, super::monitoring::HealthStatus> {
        if let Ok(checker) = self.health_checker.lock() {
            checker
                .get_all_results()
                .iter()
                .map(|(id, result)| (id.clone(), result.status))
                .collect()
        } else {
            HashMap::new()
        }
    }

    /// Cleanup and shutdown the hardware manager
    pub async fn cleanup(&mut self) -> HardwareResult<()> {
        // Backend cleanup - no longer needed with new trait design
        // Individual devices handle their own cleanup through the shutdown method

        // Clear caches
        self.device_info
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.device_metrics
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();

        Ok(())
    }
}

impl Default for HardwareManager {
    fn default() -> Self {
        Self::new(HardwareManagerConfig::default())
    }
}

// Re-export commonly used types
pub use super::config::{AllocationStrategy, LoadBalancingStrategy};
pub use super::monitoring::{AnomalySeverity, AnomalyType, HealthStatus};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hardware_manager_creation() {
        let config = HardwareManagerConfig::default();
        let manager = HardwareManager::new(config);

        // Basic creation test
        assert_eq!(manager.list_devices().len(), 0);
    }

    #[tokio::test]
    async fn test_cpu_backend_initialization() {
        let mut manager = HardwareManager::default();

        // Should successfully initialize CPU backend
        assert!(manager.initialize().await.is_ok());

        // Should have at least one CPU device
        let cpu_devices = manager.list_devices_by_type(HardwareType::CPU);
        assert!(!cpu_devices.is_empty());
    }

    #[tokio::test]
    async fn test_device_metrics_update() {
        let manager = HardwareManager::default();

        let metrics = HardwareMetrics {
            ops_per_second: 1000.0,
            memory_bandwidth: 100.0,
            utilization: 50.0,
            power_consumption: 100.0,
            temperature: Some(45.0),
            error_rate: 0.001,
            latency: 1.0,
            throughput: 1000.0,
        };

        manager.update_device_metrics("test_device", metrics.clone());

        let retrieved_metrics = manager.get_device_metrics("test_device");
        assert!(retrieved_metrics.is_some());
        assert_eq!(
            retrieved_metrics.expect("operation failed in test").utilization,
            50.0
        );
    }

    /// Minimal `HardwareOperation` used only to exercise
    /// `HardwareManager::execute_on_device` by name; `execute`/
    /// `validate_params`/`estimate_cost` are not called by
    /// `execute_on_device` (it dispatches by `name()` directly to
    /// `CPUDevice`/`GPUDevice::execute_operation`), so they are stubbed.
    struct NamedOp(&'static str);

    #[async_trait::async_trait]
    impl HardwareOperation for NamedOp {
        fn name(&self) -> &str {
            self.0
        }

        async fn execute(
            &self,
            _device: &mut dyn super::super::traits::HardwareDevice,
            _inputs: &[Tensor],
            _outputs: &mut [Tensor],
            _params: &HashMap<String, OperationParameter>,
        ) -> HardwareResult<()> {
            Ok(())
        }

        fn validate_params(
            &self,
            _params: &HashMap<String, OperationParameter>,
        ) -> HardwareResult<()> {
            Ok(())
        }

        fn requirements(&self) -> super::super::traits::OperationRequirements {
            super::super::traits::OperationRequirements {
                min_memory: 0,
                compute_units: None,
                data_types: vec![],
                capabilities: vec![],
                performance: Default::default(),
            }
        }

        fn estimate_cost(
            &self,
            _inputs: &[Tensor],
            _params: &HashMap<String, OperationParameter>,
        ) -> f64 {
            0.0
        }
    }

    /// Regression test: `execute_on_device` used to return
    /// `Ok(vec![inputs[0].clone()])` ("a mock result") for every CPU/GPU
    /// operation, regardless of what the operation actually was. This
    /// asserts the dispatched device really computes `add`, and that an
    /// unsupported operation name honestly errors instead of echoing back
    /// the input.
    #[tokio::test]
    async fn test_execute_on_device_computes_real_result_for_cpu() {
        let mut manager = HardwareManager::default();
        manager.initialize().await.expect("initialize failed");

        let cpu_devices = manager.list_devices_by_type(HardwareType::CPU);
        assert!(
            !cpu_devices.is_empty(),
            "expected at least one CPU device to be registered"
        );
        let device_id = cpu_devices[0].id.clone();

        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).expect("create failed");
        let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], &[3]).expect("create failed");
        let op = NamedOp("add");

        let result = manager
            .execute_on_device(&device_id, &op, &[a, b], &HashMap::new())
            .expect("execute_on_device should succeed for add");

        let data = result[0].data().expect("read result");
        assert_eq!(
            data,
            vec![11.0, 22.0, 33.0],
            "must be a real sum, not an echo of input[0]"
        );
    }

    #[tokio::test]
    async fn test_execute_on_device_unsupported_operation_errors() {
        let mut manager = HardwareManager::default();
        manager.initialize().await.expect("initialize failed");

        let cpu_devices = manager.list_devices_by_type(HardwareType::CPU);
        assert!(!cpu_devices.is_empty());
        let device_id = cpu_devices[0].id.clone();

        let a = Tensor::from_vec(vec![1.0f32], &[1]).expect("create failed");
        let op = NamedOp("definitely_not_a_real_op");

        let result = manager.execute_on_device(&device_id, &op, &[a], &HashMap::new());
        assert!(
            result.is_err(),
            "unsupported operations must error, not silently succeed"
        );
    }

    #[tokio::test]
    async fn test_execute_on_device_empty_inputs_errors_not_panics() {
        let mut manager = HardwareManager::default();
        manager.initialize().await.expect("initialize failed");

        let cpu_devices = manager.list_devices_by_type(HardwareType::CPU);
        assert!(!cpu_devices.is_empty());
        let device_id = cpu_devices[0].id.clone();

        let op = NamedOp("add");
        let result = manager.execute_on_device(&device_id, &op, &[], &HashMap::new());
        assert!(result.is_err());
    }
}
