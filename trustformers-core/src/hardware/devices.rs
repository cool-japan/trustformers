// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Hardware device implementations
//!
//! This module provides specific implementations for different hardware device types,
//! including CPU and GPU devices with their respective capabilities and operations.

use super::traits::{DeviceMemory, DeviceStatus, HardwareDevice, MemoryType, MemoryUsage};
use super::{
    DataType, HardwareCapabilities, HardwareConfig, HardwareMetrics, HardwareResult, HardwareType,
    OperationMode, PrecisionMode,
};
use crate::errors::TrustformersError;
use crate::tensor::Tensor;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// CPU device implementation
#[derive(Debug, Clone)]
pub struct CPUDevice {
    /// Device identifier
    id: String,
    /// Device capabilities
    capabilities: HardwareCapabilities,
    /// Initialization status
    is_initialized: bool,
    /// Real-time metrics
    metrics: Arc<Mutex<HardwareMetrics>>,
    /// Memory pools for different allocations
    memory_pools: Arc<Mutex<HashMap<usize, Vec<u8>>>>,
    /// Next memory allocation ID
    next_memory_id: Arc<Mutex<usize>>,
    /// Device status
    status: Arc<Mutex<DeviceStatus>>,
}

/// GPU device implementation
#[derive(Debug, Clone)]
pub struct GPUDevice {
    /// Device identifier
    id: String,
    /// GPU backend type
    backend_type: GPUBackendType,
    /// Device capabilities
    capabilities: HardwareCapabilities,
    /// Initialization status
    is_initialized: bool,
    /// Real-time metrics
    metrics: Arc<Mutex<HardwareMetrics>>,
    /// Memory pools for different allocations
    memory_pools: Arc<Mutex<HashMap<usize, Vec<u8>>>>,
    /// Next memory allocation ID
    next_memory_id: Arc<Mutex<usize>>,
    /// Device status
    status: Arc<Mutex<DeviceStatus>>,
}

/// GPU backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GPUBackendType {
    /// NVIDIA CUDA backend
    CUDA,
    /// AMD ROCm backend
    ROCm,
    /// OpenCL backend
    OpenCL,
    /// Apple Metal backend
    Metal,
    /// Vulkan backend
    Vulkan,
    /// Unknown or unsupported backend
    Unknown,
}

impl CPUDevice {
    /// Create a new CPU device
    pub fn new(id: String) -> Self {
        let capabilities = Self::detect_cpu_capabilities();
        let metrics = Arc::new(Mutex::new(HardwareMetrics {
            ops_per_second: 1_000_000.0, // 1M ops/sec baseline for CPU
            memory_bandwidth: Self::detect_memory_bandwidth(),
            utilization: 0.0,
            power_consumption: 65.0, // Typical CPU TDP
            temperature: Some(45.0), // Typical idle temperature
            error_rate: 0.0001,
            latency: 1.0,
            throughput: 1000.0,
        }));

        Self {
            id,
            capabilities,
            is_initialized: false,
            metrics,
            memory_pools: Arc::new(Mutex::new(HashMap::new())),
            next_memory_id: Arc::new(Mutex::new(1)),
            status: Arc::new(Mutex::new(DeviceStatus {
                online: true,
                busy: false,
                error: None,
                memory_usage: MemoryUsage {
                    used: 0,
                    total: Self::get_system_memory(),
                    free: Self::get_system_memory(),
                    fragmentation: 0.0,
                },
                temperature: Some(45.0),
                power_consumption: Some(65.0),
                utilization: 0.0,
            })),
        }
    }

    /// Detect CPU capabilities and specifications
    fn detect_cpu_capabilities() -> HardwareCapabilities {
        let core_count = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4); // Default to 4 cores if detection fails

        HardwareCapabilities {
            data_types: vec![
                DataType::F32,
                DataType::F64,
                DataType::I8,
                DataType::I16,
                DataType::I32,
                DataType::I64,
                DataType::U8,
                DataType::U16,
                DataType::U32,
                DataType::U64,
                DataType::Bool,
            ],
            max_dimensions: 8, // Reasonable limit for CPU operations
            memory_size: Some(Self::get_system_memory()),
            clock_frequency: Some(2_400_000_000), // 2.4 GHz base frequency
            compute_units: Some(core_count as u32),
            operations: vec![
                "add",
                "sub",
                "mul",
                "div",
                "matmul",
                "conv2d",
                "relu",
                "softmax",
                "batch_norm",
                "layer_norm",
                "transpose",
                "reshape",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            power_consumption: Some(65.0),    // Watts
            thermal_design_power: Some(95.0), // Watts
        }
    }

    /// Detect system memory size
    fn get_system_memory() -> usize {
        #[cfg(target_os = "linux")]
        {
            // Read from /proc/meminfo on Linux
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Use sysctl on macOS
            use std::process::Command;
            if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
                if let Ok(mem_str) = String::from_utf8(output.stdout) {
                    if let Ok(mem_bytes) = mem_str.trim().parse::<usize>() {
                        return mem_bytes;
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Use WMI or GetPhysicallyInstalledSystemMemory on Windows
            // For now, return a reasonable default
        }

        // Default fallback: 8GB
        8 * 1024 * 1024 * 1024
    }

    /// Detect memory bandwidth
    fn detect_memory_bandwidth() -> f64 {
        // Estimate based on typical DDR4/DDR5 specs
        // This would ideally be measured or detected from system specs
        25.6e9 // 25.6 GB/s for DDR4-3200
    }

    /// Update CPU metrics based on current system state
    fn update_metrics(&self) -> HardwareResult<()> {
        let mut metrics = self.metrics.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock metrics", "update_metrics")
        })?;

        // Update utilization (simplified - would use actual CPU monitoring)
        metrics.utilization = self.get_cpu_utilization();

        // Update temperature (simplified - would read from sensors)
        metrics.temperature = Some(self.get_cpu_temperature());

        // Update power usage based on utilization
        metrics.power_consumption = 65.0 + (metrics.utilization * 30.0); // Base + load-dependent

        Ok(())
    }

    fn get_cpu_utilization(&self) -> f64 {
        // Placeholder - would implement actual CPU utilization monitoring
        // Could use /proc/stat on Linux, performance counters on Windows, etc.
        25.0 // 25% utilization as example
    }

    fn get_cpu_temperature(&self) -> f64 {
        // Placeholder - would read from thermal sensors
        // Could use lm-sensors on Linux, CoreTemp on Windows, etc.
        55.0 // 55°C as example
    }

    /// Execute operation on CPU device
    pub fn execute_operation(
        &self,
        operation: &str,
        inputs: &[Tensor],
        _mode: OperationMode,
        _precision: PrecisionMode,
    ) -> HardwareResult<Vec<Tensor>> {
        // Mark device as busy
        {
            let mut status = self.status.lock().map_err(|_| {
                TrustformersError::hardware_error(
                    "Failed to lock device status",
                    "execute_operation",
                )
            })?;
            status.busy = true;
        }

        // Execute the operation (placeholder implementation)
        let result = match operation {
            "add" => {
                if inputs.len() >= 2 {
                    vec![inputs[0].add(&inputs[1])?]
                } else {
                    return Err(TrustformersError::hardware_error(
                        "Add operation requires at least 2 inputs",
                        "execute_operation",
                    ));
                }
            },
            "mul" => {
                if inputs.len() >= 2 {
                    vec![inputs[0].mul(&inputs[1])?]
                } else {
                    return Err(TrustformersError::hardware_error(
                        "Mul operation requires at least 2 inputs",
                        "execute_operation",
                    ));
                }
            },
            "matmul" => {
                if inputs.len() >= 2 {
                    vec![inputs[0].matmul(&inputs[1])?]
                } else {
                    return Err(TrustformersError::hardware_error(
                        "Matmul operation requires at least 2 inputs",
                        "execute_operation",
                    ));
                }
            },
            _ => {
                return Err(TrustformersError::hardware_error(
                    &format!("Unsupported operation: {}", operation),
                    "execute_operation",
                ));
            },
        };

        // Mark device as not busy
        {
            let mut status = self.status.lock().map_err(|_| {
                TrustformersError::hardware_error(
                    "Failed to lock device status",
                    "execute_operation",
                )
            })?;
            status.busy = false;
        }

        Ok(result)
    }
}

#[async_trait]
impl HardwareDevice for CPUDevice {
    fn device_id(&self) -> &str {
        &self.id
    }

    fn hardware_type(&self) -> HardwareType {
        HardwareType::CPU
    }

    fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    async fn initialize(&mut self, _config: &HardwareConfig) -> HardwareResult<()> {
        if self.is_initialized {
            return Ok(());
        }

        // Initialize CPU device (minimal setup required)
        {
            let mut status = self.status.lock().expect("Lock poisoned");
            status.online = true;
            status.busy = false;
        }

        // Perform any necessary CPU-specific initialization
        self.update_metrics()?;

        self.is_initialized = true;

        Ok(())
    }

    async fn shutdown(&mut self) -> HardwareResult<()> {
        // Clear memory pools
        if let Ok(mut pools) = self.memory_pools.lock() {
            pools.clear();
        }

        {
            let mut status = self.status.lock().expect("Lock poisoned");
            status.online = false;
            status.busy = false;
        }

        self.is_initialized = false;

        Ok(())
    }

    fn is_available(&self) -> bool {
        self.is_initialized && self.status.lock().expect("Lock poisoned").online
    }

    fn status(&self) -> DeviceStatus {
        self.status.lock().expect("Lock poisoned").clone()
    }

    async fn metrics(&self) -> HardwareResult<HardwareMetrics> {
        self.update_metrics()?;
        Ok(self.metrics.lock().expect("Lock poisoned").clone())
    }

    async fn reset(&mut self) -> HardwareResult<()> {
        // Reset device state
        {
            let mut status = self.status.lock().expect("Lock poisoned");
            status.busy = false;
            status.error = None;
        }

        // Clear memory pools
        if let Ok(mut pools) = self.memory_pools.lock() {
            pools.clear();
        }

        Ok(())
    }

    async fn allocate_memory(&mut self, size: usize) -> HardwareResult<DeviceMemory> {
        let memory_id = {
            let mut id_counter = self.next_memory_id.lock().expect("Lock poisoned");
            let id = *id_counter;
            *id_counter += 1;
            id
        };

        // Allocate memory buffer
        let buffer = vec![0u8; size];

        {
            let mut pools = self.memory_pools.lock().expect("Lock poisoned");
            pools.insert(memory_id, buffer);
        }

        Ok(DeviceMemory {
            address: memory_id,
            size,
            memory_type: MemoryType::Host,
            device_id: self.id.clone(),
        })
    }

    async fn free_memory(&mut self, memory: DeviceMemory) -> HardwareResult<()> {
        let mut pools = self.memory_pools.lock().expect("Lock poisoned");
        pools.remove(&memory.address);
        Ok(())
    }

    async fn synchronize(&self) -> HardwareResult<()> {
        // CPU operations are synchronous by nature
        Ok(())
    }
}

impl CPUDevice {
    #[allow(dead_code)]
    fn execute_add(&self, inputs: &[Tensor]) -> HardwareResult<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(TrustformersError::hardware_error(
                "Add operation requires exactly 2 inputs",
                "allocate_memory",
            ));
        }

        let result = inputs[0].add(&inputs[1])?;
        Ok(vec![result])
    }

    #[allow(dead_code)]
    fn execute_multiply(&self, inputs: &[Tensor]) -> HardwareResult<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(TrustformersError::hardware_error(
                "Multiply operation requires exactly 2 inputs",
                "execute_multiply",
            ));
        }

        let result = inputs[0].mul(&inputs[1])?;
        Ok(vec![result])
    }

    #[allow(dead_code)]
    fn execute_matmul(&self, inputs: &[Tensor]) -> HardwareResult<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(TrustformersError::hardware_error(
                "MatMul operation requires exactly 2 inputs",
                "execute_matmul",
            ));
        }

        let result = inputs[0].matmul(&inputs[1])?;
        Ok(vec![result])
    }

    #[allow(dead_code)]
    fn execute_relu(&self, inputs: &[Tensor]) -> HardwareResult<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(TrustformersError::hardware_error(
                "ReLU operation requires exactly 1 input",
                "execute_relu",
            ));
        }

        let result = inputs[0].relu()?;
        Ok(vec![result])
    }

    #[allow(dead_code)]
    fn execute_softmax(&self, inputs: &[Tensor]) -> HardwareResult<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(TrustformersError::hardware_error(
                "Softmax operation requires exactly 1 input",
                "execute_softmax",
            ));
        }

        let result = inputs[0].softmax(-1)?; // Apply softmax along last dimension
        Ok(vec![result])
    }
}

impl GPUDevice {
    /// Create a new GPU device
    pub fn new(id: String, backend_type: GPUBackendType) -> Self {
        let capabilities = Self::detect_gpu_capabilities(&backend_type);

        // Extract memory size and power consumption from backend type
        let (memory_size, _compute_units, power_consumption) = match backend_type {
            GPUBackendType::CUDA => (8 * 1024 * 1024 * 1024, 2048, 250.0),
            GPUBackendType::ROCm => (16 * 1024 * 1024 * 1024, 3840, 300.0),
            GPUBackendType::OpenCL => (4 * 1024 * 1024 * 1024, 1024, 150.0),
            GPUBackendType::Metal => (8 * 1024 * 1024 * 1024, 1024, 200.0),
            GPUBackendType::Vulkan => (6 * 1024 * 1024 * 1024, 1536, 180.0),
            GPUBackendType::Unknown => (2 * 1024 * 1024 * 1024, 512, 100.0),
        };

        let metrics = Arc::new(Mutex::new(HardwareMetrics {
            ops_per_second: 50_000_000.0, // 50M ops/sec for GPU
            memory_bandwidth: Self::detect_gpu_memory_bandwidth(&backend_type),
            utilization: 0.0,
            power_consumption,       // Typical GPU TDP
            temperature: Some(35.0), // Typical idle temperature
            error_rate: 0.00001,
            latency: 0.5,
            throughput: 50_000.0,
        }));

        Self {
            id,
            backend_type,
            capabilities,
            is_initialized: false,
            metrics,
            memory_pools: Arc::new(Mutex::new(HashMap::new())),
            next_memory_id: Arc::new(Mutex::new(1)),
            status: Arc::new(Mutex::new(DeviceStatus {
                online: true,
                busy: false,
                error: None,
                memory_usage: MemoryUsage {
                    used: 0,
                    total: memory_size,
                    free: memory_size,
                    fragmentation: 0.0,
                },
                temperature: Some(35.0),
                power_consumption: Some(power_consumption),
                utilization: 0.0,
            })),
        }
    }

    /// Detect GPU capabilities based on backend type
    fn detect_gpu_capabilities(backend_type: &GPUBackendType) -> HardwareCapabilities {
        let (memory_size, compute_units, power_consumption) = match backend_type {
            GPUBackendType::CUDA => (8 * 1024 * 1024 * 1024, 2048, 250.0), // 8GB VRAM, 2048 CUDA cores
            GPUBackendType::ROCm => (16 * 1024 * 1024 * 1024, 3840, 300.0), // 16GB VRAM, 3840 Stream processors
            GPUBackendType::OpenCL => (4 * 1024 * 1024 * 1024, 1024, 150.0), // 4GB VRAM, 1024 cores
            GPUBackendType::Metal => (8 * 1024 * 1024 * 1024, 1024, 200.0), // 8GB unified memory
            GPUBackendType::Vulkan => (6 * 1024 * 1024 * 1024, 1536, 180.0), // 6GB VRAM
            GPUBackendType::Unknown => (2 * 1024 * 1024 * 1024, 512, 100.0), // Minimal fallback
        };

        HardwareCapabilities {
            data_types: vec![
                DataType::F32,
                DataType::F16,
                DataType::BF16,
                DataType::I8,
                DataType::I16,
                DataType::I32,
                DataType::U8,
                DataType::U16,
                DataType::U32,
                DataType::Bool,
            ],
            max_dimensions: 12, // GPUs can handle higher dimensions
            memory_size: Some(memory_size),
            clock_frequency: Some(1_800_000_000), // 1.8 GHz boost clock
            compute_units: Some(compute_units),
            operations: vec![
                "add",
                "sub",
                "mul",
                "div",
                "matmul",
                "conv2d",
                "conv3d",
                "relu",
                "gelu",
                "softmax",
                "batch_norm",
                "layer_norm",
                "group_norm",
                "attention",
                "flash_attention",
                "transpose",
                "reshape",
                "slice",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            power_consumption: Some(power_consumption),
            thermal_design_power: Some(power_consumption + 50.0),
        }
    }

    /// Detect GPU memory bandwidth based on backend
    fn detect_gpu_memory_bandwidth(backend_type: &GPUBackendType) -> f64 {
        match backend_type {
            GPUBackendType::CUDA => 900.0e9,    // 900 GB/s for high-end CUDA
            GPUBackendType::ROCm => 1600.0e9,   // 1.6 TB/s for high-end ROCm
            GPUBackendType::OpenCL => 400.0e9,  // 400 GB/s for OpenCL
            GPUBackendType::Metal => 400.0e9,   // 400 GB/s for Metal
            GPUBackendType::Vulkan => 500.0e9,  // 500 GB/s for Vulkan
            GPUBackendType::Unknown => 200.0e9, // 200 GB/s fallback
        }
    }

    /// Execute operation on GPU device
    pub fn execute_operation(
        &self,
        operation: &str,
        inputs: &[Tensor],
        _mode: OperationMode,
        _precision: PrecisionMode,
    ) -> HardwareResult<Vec<Tensor>> {
        // Mark device as busy
        {
            let mut status = self.status.lock().map_err(|_| {
                TrustformersError::hardware_error(
                    "Failed to lock device status",
                    "execute_operation",
                )
            })?;
            status.busy = true;
        }

        // Execute the operation (placeholder implementation)
        let result = match operation {
            "add" => {
                if inputs.len() >= 2 {
                    vec![inputs[0].add(&inputs[1])?]
                } else {
                    return Err(TrustformersError::hardware_error(
                        "Add operation requires at least 2 inputs",
                        "execute_operation",
                    ));
                }
            },
            "mul" => {
                if inputs.len() >= 2 {
                    vec![inputs[0].mul(&inputs[1])?]
                } else {
                    return Err(TrustformersError::hardware_error(
                        "Mul operation requires at least 2 inputs",
                        "execute_operation",
                    ));
                }
            },
            "matmul" => {
                if inputs.len() >= 2 {
                    vec![inputs[0].matmul(&inputs[1])?]
                } else {
                    return Err(TrustformersError::hardware_error(
                        "Matmul operation requires at least 2 inputs",
                        "execute_operation",
                    ));
                }
            },
            _ => {
                return Err(TrustformersError::hardware_error(
                    &format!("Unsupported operation: {}", operation),
                    "execute_operation",
                ));
            },
        };

        // Mark device as not busy
        {
            let mut status = self.status.lock().map_err(|_| {
                TrustformersError::hardware_error(
                    "Failed to lock device status",
                    "execute_operation",
                )
            })?;
            status.busy = false;
        }

        Ok(result)
    }
}

#[async_trait]
impl HardwareDevice for GPUDevice {
    fn device_id(&self) -> &str {
        &self.id
    }

    fn hardware_type(&self) -> HardwareType {
        HardwareType::GPU
    }

    fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    async fn initialize(&mut self, _config: &HardwareConfig) -> HardwareResult<()> {
        if self.is_initialized {
            return Ok(());
        }

        {
            let mut status = self.status.lock().expect("Lock poisoned");
            status.online = false;
            status.busy = true;
        }

        // Initialize GPU device based on backend type
        match self.backend_type {
            GPUBackendType::CUDA => self.initialize_cuda()?,
            GPUBackendType::ROCm => self.initialize_rocm()?,
            GPUBackendType::OpenCL => self.initialize_opencl()?,
            GPUBackendType::Metal => self.initialize_metal()?,
            GPUBackendType::Vulkan => self.initialize_vulkan()?,
            GPUBackendType::Unknown => {
                return Err(TrustformersError::hardware_error(
                    "Cannot initialize unknown GPU backend",
                    "initialize",
                ));
            },
        }

        self.is_initialized = true;
        {
            let mut status = self.status.lock().expect("Lock poisoned");
            status.online = true;
            status.busy = false;
        }

        Ok(())
    }

    async fn shutdown(&mut self) -> HardwareResult<()> {
        // Clear GPU memory pools
        if let Ok(mut pools) = self.memory_pools.lock() {
            pools.clear();
        }

        // Backend-specific cleanup
        match self.backend_type {
            GPUBackendType::CUDA => self.cleanup_cuda()?,
            GPUBackendType::ROCm => self.cleanup_rocm()?,
            GPUBackendType::OpenCL => self.cleanup_opencl()?,
            GPUBackendType::Metal => self.cleanup_metal()?,
            GPUBackendType::Vulkan => self.cleanup_vulkan()?,
            GPUBackendType::Unknown => {},
        }

        {
            let mut status = self.status.lock().expect("Lock poisoned");
            status.online = false;
            status.busy = false;
        }

        self.is_initialized = false;

        Ok(())
    }

    async fn metrics(&self) -> HardwareResult<HardwareMetrics> {
        // Update metrics from GPU
        let mut metrics = self.metrics.lock().expect("Lock poisoned");

        // Update GPU-specific metrics
        metrics.utilization = self.get_gpu_utilization();
        metrics.temperature = Some(self.get_gpu_temperature());
        metrics.power_consumption = self.get_gpu_power_usage();

        Ok(metrics.clone())
    }

    fn is_available(&self) -> bool {
        self.is_initialized && self.status.lock().expect("Lock poisoned").online
    }

    fn status(&self) -> DeviceStatus {
        self.status.lock().expect("Lock poisoned").clone()
    }

    async fn reset(&mut self) -> HardwareResult<()> {
        // Reset device state
        {
            let mut status = self.status.lock().expect("Lock poisoned");
            status.busy = false;
            status.error = None;
        }

        // Clear memory pools
        if let Ok(mut pools) = self.memory_pools.lock() {
            pools.clear();
        }

        Ok(())
    }

    async fn allocate_memory(&mut self, size: usize) -> HardwareResult<DeviceMemory> {
        let memory_id = {
            let mut id_counter = self.next_memory_id.lock().expect("Lock poisoned");
            let id = *id_counter;
            *id_counter += 1;
            id
        };

        // Allocate GPU memory (simplified - would use actual GPU memory allocation)
        let buffer = vec![0u8; size];

        {
            let mut pools = self.memory_pools.lock().expect("Lock poisoned");
            pools.insert(memory_id, buffer);
        }

        Ok(DeviceMemory {
            address: memory_id,
            size,
            memory_type: MemoryType::Local,
            device_id: self.id.clone(),
        })
    }

    async fn free_memory(&mut self, memory: DeviceMemory) -> HardwareResult<()> {
        let mut pools = self.memory_pools.lock().expect("Lock poisoned");
        pools.remove(&memory.address);
        Ok(())
    }

    async fn synchronize(&self) -> HardwareResult<()> {
        // Synchronize GPU operations
        match self.backend_type {
            GPUBackendType::CUDA => Ok(()),   // CUDA sync placeholder
            GPUBackendType::ROCm => Ok(()),   // ROCm sync placeholder
            GPUBackendType::OpenCL => Ok(()), // OpenCL sync placeholder
            GPUBackendType::Metal => Ok(()),  // Metal sync placeholder
            GPUBackendType::Vulkan => Ok(()), // Vulkan sync placeholder
            GPUBackendType::Unknown => Err(TrustformersError::hardware_error(
                "Cannot sync unknown backend",
                "sync_memory",
            )),
        }
    }
}

impl GPUDevice {
    fn initialize_cuda(&self) -> HardwareResult<()> {
        // CUDA initialization (placeholder)
        Ok(())
    }

    fn initialize_rocm(&self) -> HardwareResult<()> {
        // ROCm initialization (placeholder)
        Ok(())
    }

    fn initialize_opencl(&self) -> HardwareResult<()> {
        // OpenCL initialization (placeholder)
        Ok(())
    }

    fn initialize_metal(&self) -> HardwareResult<()> {
        // Metal initialization (placeholder)
        Ok(())
    }

    fn initialize_vulkan(&self) -> HardwareResult<()> {
        // Vulkan initialization (placeholder)
        Ok(())
    }

    fn cleanup_cuda(&self) -> HardwareResult<()> {
        // CUDA cleanup (placeholder)
        Ok(())
    }

    fn cleanup_rocm(&self) -> HardwareResult<()> {
        // ROCm cleanup (placeholder)
        Ok(())
    }

    fn cleanup_opencl(&self) -> HardwareResult<()> {
        // OpenCL cleanup (placeholder)
        Ok(())
    }

    fn cleanup_metal(&self) -> HardwareResult<()> {
        // Metal cleanup (placeholder)
        Ok(())
    }

    fn cleanup_vulkan(&self) -> HardwareResult<()> {
        // Vulkan cleanup (placeholder)
        Ok(())
    }

    fn get_gpu_utilization(&self) -> f64 {
        // Placeholder - would query actual GPU utilization
        35.0 // 35% utilization
    }

    fn get_gpu_temperature(&self) -> f64 {
        // Placeholder - would query GPU temperature sensors
        65.0 // 65°C
    }

    fn get_gpu_power_usage(&self) -> f64 {
        // Placeholder - would query actual GPU power usage
        180.0 // 180W
    }

    #[allow(dead_code)]
    fn execute_gpu_matmul(&self, inputs: &[Tensor]) -> HardwareResult<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(TrustformersError::hardware_error(
                "GPU MatMul requires exactly 2 inputs",
                "execute_gpu_matmul",
            ));
        }

        // GPU-accelerated matrix multiplication
        let result = inputs[0].matmul(&inputs[1])?;
        Ok(vec![result])
    }

    #[allow(dead_code)]
    fn execute_gpu_conv2d(&self, inputs: &[Tensor]) -> HardwareResult<Vec<Tensor>> {
        // CPU fallback for 2D convolution using im2col + matmul approach
        // inputs: [input (N,C_in,H,W), kernel (C_out,C_in,kH,kW)]
        tracing::debug!("GPU Conv2D not available - using CPU fallback");

        if inputs.len() < 2 {
            return Err(TrustformersError::hardware_error(
                "Conv2D requires at least 2 inputs: [input, kernel]",
                "execute_gpu_conv2d",
            ));
        }

        let input = &inputs[0];
        let kernel = &inputs[1];
        let input_shape = input.shape();
        let kernel_shape = kernel.shape();

        if input_shape.len() != 4 || kernel_shape.len() != 4 {
            return Err(TrustformersError::hardware_error(
                "Conv2D expects 4D input (N,C_in,H,W) and 4D kernel (C_out,C_in,kH,kW)",
                "execute_gpu_conv2d",
            ));
        }

        let batch_size = input_shape[0];
        let c_in = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];
        let c_out = kernel_shape[0];
        let kc_in = kernel_shape[1];
        let k_h = kernel_shape[2];
        let k_w = kernel_shape[3];

        if c_in != kc_in {
            return Err(TrustformersError::hardware_error(
                &format!(
                    "Conv2D channel mismatch: input has {} channels, kernel expects {}",
                    c_in, kc_in
                ),
                "execute_gpu_conv2d",
            ));
        }

        // Stride=1, padding=0 convolution
        let h_out = h_in.saturating_sub(k_h) + 1;
        let w_out = w_in.saturating_sub(k_w) + 1;

        if h_out == 0 || w_out == 0 {
            return Err(TrustformersError::hardware_error(
                "Conv2D output dimensions are zero - kernel larger than input",
                "execute_gpu_conv2d",
            ));
        }

        let input_data = input.data().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to extract input data: {}", e),
                "execute_gpu_conv2d",
            )
        })?;
        let kernel_data = kernel.data().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to extract kernel data: {}", e),
                "execute_gpu_conv2d",
            )
        })?;

        // im2col + matmul approach
        let col_rows = c_in * k_h * k_w;
        let col_cols = h_out * w_out;
        let mut output_data = vec![0.0f32; batch_size * c_out * h_out * w_out];

        for n in 0..batch_size {
            // Build im2col matrix for this batch element
            let mut col_matrix = vec![0.0f32; col_rows * col_cols];

            for c in 0..c_in {
                for kh in 0..k_h {
                    for kw in 0..k_w {
                        let col_row = c * k_h * k_w + kh * k_w + kw;
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let ih = oh + kh;
                                let iw = ow + kw;
                                let col_col = oh * w_out + ow;
                                col_matrix[col_row * col_cols + col_col] = input_data
                                    [n * c_in * h_in * w_in + c * h_in * w_in + ih * w_in + iw];
                            }
                        }
                    }
                }
            }

            // kernel reshaped to (C_out, C_in*kH*kW) matmul with col_matrix (C_in*kH*kW, H_out*W_out)
            for co in 0..c_out {
                for spatial in 0..col_cols {
                    let mut sum = 0.0f32;
                    for kr in 0..col_rows {
                        sum +=
                            kernel_data[co * col_rows + kr] * col_matrix[kr * col_cols + spatial];
                    }
                    output_data[n * c_out * h_out * w_out + co * h_out * w_out + spatial] = sum;
                }
            }
        }

        let result =
            Tensor::from_vec(output_data, &[batch_size, c_out, h_out, w_out]).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create output tensor: {}", e),
                    "execute_gpu_conv2d",
                )
            })?;

        Ok(vec![result])
    }

    #[allow(dead_code)]
    fn execute_gpu_attention(&self, inputs: &[Tensor]) -> HardwareResult<Vec<Tensor>> {
        // CPU fallback for standard scaled dot-product attention
        // inputs: [Q, K, V] each of shape (..., seq_len, d_k)
        // Computes: softmax(Q * K^T / sqrt(d_k)) * V
        tracing::debug!("GPU Attention not available - using CPU fallback");

        if inputs.len() < 3 {
            return Err(TrustformersError::hardware_error(
                "Attention requires 3 inputs: [Q, K, V]",
                "execute_gpu_attention",
            ));
        }

        let q = &inputs[0];
        let k = &inputs[1];
        let v = &inputs[2];

        let q_shape = q.shape();
        if q_shape.len() < 2 {
            return Err(TrustformersError::hardware_error(
                "Attention Q must have at least 2 dimensions",
                "execute_gpu_attention",
            ));
        }

        let d_k = q_shape[q_shape.len() - 1];
        if d_k == 0 {
            return Err(TrustformersError::hardware_error(
                "Attention d_k dimension must be > 0",
                "execute_gpu_attention",
            ));
        }

        let scale = 1.0 / (d_k as f32).sqrt();

        // Transpose K: swap last two dimensions
        let k_shape = k.shape();
        let k_ndim = k_shape.len();
        let k_t = k.transpose(k_ndim - 2, k_ndim - 1).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to transpose K: {}", e),
                "execute_gpu_attention",
            )
        })?;

        // Q * K^T
        let scores = q.matmul(&k_t).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compute Q*K^T: {}", e),
                "execute_gpu_attention",
            )
        })?;

        // Scale by 1/sqrt(d_k)
        let scaled_scores = scores.scalar_mul(scale).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to scale scores: {}", e),
                "execute_gpu_attention",
            )
        })?;

        // Softmax along last dimension
        let attn_weights = scaled_scores.softmax(-1).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compute softmax: {}", e),
                "execute_gpu_attention",
            )
        })?;

        // Multiply by V
        let output = attn_weights.matmul(v).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compute attention * V: {}", e),
                "execute_gpu_attention",
            )
        })?;

        Ok(vec![output])
    }

    #[allow(dead_code)]
    fn execute_gpu_flash_attention(&self, inputs: &[Tensor]) -> HardwareResult<Vec<Tensor>> {
        // CPU fallback for flash attention using a tiled/chunked approach
        // inputs: [Q, K, V] each of shape (seq_len, d_k) or (batch, seq_len, d_k)
        // Falls back to tiled computation for memory efficiency
        tracing::debug!("GPU Flash Attention not available - using CPU tiled fallback");

        if inputs.len() < 3 {
            return Err(TrustformersError::hardware_error(
                "Flash Attention requires 3 inputs: [Q, K, V]",
                "execute_gpu_flash_attention",
            ));
        }

        let q = &inputs[0];
        let k = &inputs[1];
        let v = &inputs[2];

        let q_shape = q.shape();
        let k_shape = k.shape();

        if q_shape.len() < 2 {
            return Err(TrustformersError::hardware_error(
                "Flash Attention Q must have at least 2 dimensions",
                "execute_gpu_flash_attention",
            ));
        }

        let d_k = q_shape[q_shape.len() - 1];
        if d_k == 0 {
            return Err(TrustformersError::hardware_error(
                "Flash Attention d_k dimension must be > 0",
                "execute_gpu_flash_attention",
            ));
        }

        let seq_len_q = q_shape[q_shape.len() - 2];
        let seq_len_k = k_shape[k_shape.len() - 2];
        let scale = 1.0 / (d_k as f32).sqrt();

        // Tile size for chunked computation (memory-efficient approach)
        let tile_size = 64.min(seq_len_k);
        if tile_size == 0 {
            return Err(TrustformersError::hardware_error(
                "Flash Attention sequence length must be > 0",
                "execute_gpu_flash_attention",
            ));
        }

        // For small sequences or when tiling overhead is not worth it,
        // delegate to standard attention
        if seq_len_q * seq_len_k <= tile_size * tile_size * 4 {
            return self.execute_gpu_attention(inputs);
        }

        // Tiled flash attention: process K/V in chunks to reduce peak memory
        // This is the CPU emulation of the flash attention algorithm
        let q_data = q.data().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to extract Q data: {}", e),
                "execute_gpu_flash_attention",
            )
        })?;
        let k_data = k.data().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to extract K data: {}", e),
                "execute_gpu_flash_attention",
            )
        })?;
        let v_data = v.data().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to extract V data: {}", e),
                "execute_gpu_flash_attention",
            )
        })?;

        // Determine batch dimensions (everything before the last 2 dims)
        let batch_dims: Vec<usize> = q_shape[..q_shape.len() - 2].to_vec();
        let batch_size: usize = if batch_dims.is_empty() { 1 } else { batch_dims.iter().product() };

        let mut output_data = vec![0.0f32; batch_size * seq_len_q * d_k];

        for b in 0..batch_size {
            let q_offset = b * seq_len_q * d_k;
            let k_offset = b * seq_len_k * d_k;
            let v_offset = b * seq_len_k * d_k;
            let o_offset = b * seq_len_q * d_k;

            // Per-row running max and sum for numerically stable softmax (online softmax)
            let mut row_max = vec![f32::NEG_INFINITY; seq_len_q];
            let mut row_sum = vec![0.0f32; seq_len_q];

            // Process K/V in tiles
            let num_tiles = seq_len_k.div_ceil(tile_size);

            for tile_idx in 0..num_tiles {
                let k_start = tile_idx * tile_size;
                let k_end = (k_start + tile_size).min(seq_len_k);
                let tile_len = k_end - k_start;

                for qi in 0..seq_len_q {
                    // Compute scores for this tile: Q[qi] dot K[k_start..k_end]^T * scale
                    let mut tile_scores = vec![0.0f32; tile_len];
                    for ki in 0..tile_len {
                        let mut dot = 0.0f32;
                        for di in 0..d_k {
                            dot += q_data[q_offset + qi * d_k + di]
                                * k_data[k_offset + (k_start + ki) * d_k + di];
                        }
                        tile_scores[ki] = dot * scale;
                    }

                    // Find tile max
                    let tile_max = tile_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

                    let prev_max = row_max[qi];
                    let new_max = prev_max.max(tile_max);

                    // Rescale previous accumulated output and sum
                    let correction =
                        if prev_max.is_finite() { (prev_max - new_max).exp() } else { 0.0 };

                    // Rescale existing output
                    for di in 0..d_k {
                        output_data[o_offset + qi * d_k + di] *= correction;
                    }
                    row_sum[qi] *= correction;

                    // Compute exp(score - new_max) and accumulate
                    for ki in 0..tile_len {
                        let w = (tile_scores[ki] - new_max).exp();
                        row_sum[qi] += w;
                        for di in 0..d_k {
                            output_data[o_offset + qi * d_k + di] +=
                                w * v_data[v_offset + (k_start + ki) * d_k + di];
                        }
                    }

                    row_max[qi] = new_max;
                }
            }

            // Normalize by row_sum
            for qi in 0..seq_len_q {
                let s = if row_sum[qi] > 0.0 { row_sum[qi] } else { 1.0 };
                for di in 0..d_k {
                    output_data[o_offset + qi * d_k + di] /= s;
                }
            }
        }

        // Reconstruct output shape matching Q's shape
        let out_shape = q_shape.clone();
        // Last two dims are seq_len_q x d_k, which matches Q
        let result = Tensor::from_vec(output_data, &out_shape).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create flash attention output tensor: {}", e),
                "execute_gpu_flash_attention",
            )
        })?;

        Ok(vec![result])
    }
}

/// CPU memory implementation
#[derive(Debug)]
pub struct CPUMemory {
    #[allow(dead_code)]
    id: usize,
    #[allow(dead_code)]
    size: usize,
    #[allow(dead_code)]
    memory_type: MemoryType,
    #[allow(dead_code)]
    device_id: String,
    #[allow(dead_code)]
    pools: Arc<Mutex<HashMap<usize, Vec<u8>>>>,
}

// Removed DeviceMemory impl for CPUMemory - DeviceMemory is a struct, not a trait

/// GPU memory implementation
#[derive(Debug)]
pub struct GPUMemory {
    #[allow(dead_code)]
    id: usize,
    #[allow(dead_code)]
    size: usize,
    #[allow(dead_code)]
    memory_type: MemoryType,
    #[allow(dead_code)]
    device_id: String,
    #[allow(dead_code)]
    backend_type: GPUBackendType,
    #[allow(dead_code)]
    pools: Arc<Mutex<HashMap<usize, Vec<u8>>>>,
}

// Removed DeviceMemory impl for GPUMemory - DeviceMemory is a struct, not a trait

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device_creation() {
        let dev = CPUDevice::new("cpu-0".to_string());
        assert_eq!(dev.id, "cpu-0");
        assert!(!dev.is_initialized);
    }

    #[test]
    fn test_cpu_device_capabilities() {
        let dev = CPUDevice::new("cpu-test".to_string());
        assert!(dev.capabilities.max_dimensions > 0);
        assert!(!dev.capabilities.data_types.is_empty());
        assert!(!dev.capabilities.operations.is_empty());
    }

    #[test]
    fn test_cpu_device_has_f32_support() {
        let dev = CPUDevice::new("cpu-f32".to_string());
        assert!(dev.capabilities.data_types.contains(&DataType::F32));
    }

    #[test]
    fn test_cpu_device_has_f64_support() {
        let dev = CPUDevice::new("cpu-f64".to_string());
        assert!(dev.capabilities.data_types.contains(&DataType::F64));
    }

    #[test]
    fn test_cpu_device_operations_include_add() {
        let dev = CPUDevice::new("cpu-ops".to_string());
        assert!(dev.capabilities.operations.contains(&"add".to_string()));
    }

    #[test]
    fn test_cpu_device_operations_include_matmul() {
        let dev = CPUDevice::new("cpu-ops2".to_string());
        assert!(dev.capabilities.operations.contains(&"matmul".to_string()));
    }

    #[test]
    fn test_cpu_memory_detection() {
        let mem = CPUDevice::get_system_memory();
        assert!(mem > 0);
    }

    #[test]
    fn test_cpu_memory_bandwidth() {
        let bw = CPUDevice::detect_memory_bandwidth();
        assert!(bw > 0.0);
    }

    #[test]
    fn test_gpu_device_cuda_creation() {
        let dev = GPUDevice::new("gpu-0".to_string(), GPUBackendType::CUDA);
        assert_eq!(dev.id, "gpu-0");
        assert_eq!(dev.backend_type, GPUBackendType::CUDA);
        assert!(!dev.is_initialized);
    }

    #[test]
    fn test_gpu_device_metal_creation() {
        let dev = GPUDevice::new("gpu-metal".to_string(), GPUBackendType::Metal);
        assert_eq!(dev.backend_type, GPUBackendType::Metal);
    }

    #[test]
    fn test_gpu_device_rocm_creation() {
        let dev = GPUDevice::new("gpu-rocm".to_string(), GPUBackendType::ROCm);
        assert_eq!(dev.backend_type, GPUBackendType::ROCm);
    }

    #[test]
    fn test_gpu_device_vulkan_creation() {
        let dev = GPUDevice::new("gpu-vulkan".to_string(), GPUBackendType::Vulkan);
        assert_eq!(dev.backend_type, GPUBackendType::Vulkan);
    }

    #[test]
    fn test_gpu_device_opencl_creation() {
        let dev = GPUDevice::new("gpu-cl".to_string(), GPUBackendType::OpenCL);
        assert_eq!(dev.backend_type, GPUBackendType::OpenCL);
    }

    #[test]
    fn test_gpu_device_unknown_creation() {
        let dev = GPUDevice::new("gpu-unknown".to_string(), GPUBackendType::Unknown);
        assert_eq!(dev.backend_type, GPUBackendType::Unknown);
    }

    #[test]
    fn test_gpu_backend_type_equality() {
        assert_eq!(GPUBackendType::CUDA, GPUBackendType::CUDA);
        assert_ne!(GPUBackendType::CUDA, GPUBackendType::Metal);
    }

    #[test]
    fn test_cpu_device_initial_status() {
        let dev = CPUDevice::new("cpu-status".to_string());
        let status_lock = dev.status.lock();
        if let Ok(status) = status_lock {
            assert!(status.online);
            assert!(!status.busy);
            assert!(status.error.is_none());
        }
    }

    #[test]
    fn test_gpu_device_initial_status() {
        let dev = GPUDevice::new("gpu-status".to_string(), GPUBackendType::CUDA);
        let status_lock = dev.status.lock();
        if let Ok(status) = status_lock {
            assert!(status.online);
            assert!(!status.busy);
            assert!(status.error.is_none());
        }
    }

    #[test]
    fn test_cpu_device_memory_pools_initially_empty() {
        let dev = CPUDevice::new("cpu-mem".to_string());
        let pools_lock = dev.memory_pools.lock();
        if let Ok(pools) = pools_lock {
            assert!(pools.is_empty());
        }
    }

    #[test]
    fn test_gpu_device_memory_pools_initially_empty() {
        let dev = GPUDevice::new("gpu-mem".to_string(), GPUBackendType::Metal);
        let pools_lock = dev.memory_pools.lock();
        if let Ok(pools) = pools_lock {
            assert!(pools.is_empty());
        }
    }

    #[test]
    fn test_cpu_execute_add() {
        let dev = CPUDevice::new("cpu-exec".to_string());
        let a = Tensor::from_data(vec![1.0, 2.0], &[2]).expect("create failed");
        let b = Tensor::from_data(vec![3.0, 4.0], &[2]).expect("create failed");
        let result = dev.execute_operation(
            "add",
            &[a, b],
            OperationMode::Performance,
            PrecisionMode::Single,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_cpu_execute_mul() {
        let dev = CPUDevice::new("cpu-mul".to_string());
        let a = Tensor::from_data(vec![2.0, 3.0], &[2]).expect("create failed");
        let b = Tensor::from_data(vec![4.0, 5.0], &[2]).expect("create failed");
        let result = dev.execute_operation(
            "mul",
            &[a, b],
            OperationMode::Performance,
            PrecisionMode::Single,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_cpu_execute_insufficient_inputs() {
        let dev = CPUDevice::new("cpu-err".to_string());
        let a = Tensor::from_data(vec![1.0], &[1]).expect("create failed");
        let result = dev.execute_operation(
            "add",
            &[a],
            OperationMode::Performance,
            PrecisionMode::Single,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_capabilities_include_f32() {
        let dev = GPUDevice::new("gpu-caps".to_string(), GPUBackendType::CUDA);
        assert!(dev.capabilities.data_types.contains(&DataType::F32));
    }

    #[test]
    fn test_cpu_device_clone() {
        let dev = CPUDevice::new("cpu-clone".to_string());
        let cloned = dev.clone();
        assert_eq!(cloned.id, "cpu-clone");
    }

    #[test]
    fn test_gpu_device_clone() {
        let dev = GPUDevice::new("gpu-clone".to_string(), GPUBackendType::Metal);
        let cloned = dev.clone();
        assert_eq!(cloned.id, "gpu-clone");
        assert_eq!(cloned.backend_type, GPUBackendType::Metal);
    }
}
