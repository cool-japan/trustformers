// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Hardware backend implementations
//!
//! This module provides backend implementations for different hardware types,
//! managing the interface between high-level operations and low-level hardware APIs.

#![allow(unused_variables)] // Hardware backend implementations

use super::devices::{CPUDevice, GPUBackendType, GPUDevice};
use super::traits::{HardwareBackend, HardwareDevice};
use super::{HardwareConfig, HardwareResult, HardwareType, OperationMode, PrecisionMode};
use crate::errors::TrustformersError;
use crate::tensor::Tensor;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// CPU backend implementation
#[derive(Debug)]
pub struct CPUBackend {
    /// Available CPU devices
    devices: Arc<Mutex<HashMap<String, CPUDevice>>>,
    /// Backend configuration
    config: CPUBackendConfig,
    /// Supported operations
    supported_ops: Vec<String>,
}

/// CPU backend configuration
#[derive(Debug, Clone)]
pub struct CPUBackendConfig {
    /// Number of threads to use
    pub num_threads: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Memory pool size per device
    pub memory_pool_size: usize,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
}

/// GPU backend implementation
#[derive(Debug)]
pub struct GPUBackend {
    /// Available GPU devices
    devices: Arc<Mutex<HashMap<String, GPUDevice>>>,
    /// Backend type (CUDA, ROCm, etc.)
    backend_type: GPUBackendType,
    /// Backend configuration
    config: GPUBackendConfig,
    /// Supported operations
    supported_ops: Vec<String>,
}

/// GPU backend configuration
#[derive(Debug, Clone)]
pub struct GPUBackendConfig {
    /// Memory pool size per device
    pub memory_pool_size: usize,
    /// Enable unified memory
    pub enable_unified_memory: bool,
    /// Stream/queue count per device
    pub stream_count: usize,
    /// Enable kernel fusion
    pub enable_kernel_fusion: bool,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
}

impl CPUBackend {
    /// Create a new CPU backend
    pub fn new() -> Self {
        let supported_ops = vec![
            "add".to_string(),
            "mul".to_string(),
            "matmul".to_string(),
            "conv2d".to_string(),
            "relu".to_string(),
            "softmax".to_string(),
        ];

        Self {
            devices: Arc::new(Mutex::new(HashMap::new())),
            config: CPUBackendConfig::default(),
            supported_ops,
        }
    }

    /// Create CPU backend with custom configuration
    pub fn with_config(config: CPUBackendConfig) -> Self {
        let supported_ops = vec![
            "add".to_string(),
            "mul".to_string(),
            "matmul".to_string(),
            "conv2d".to_string(),
            "relu".to_string(),
            "softmax".to_string(),
        ];

        Self {
            devices: Arc::new(Mutex::new(HashMap::new())),
            config,
            supported_ops,
        }
    }

    /// Discover available CPU devices
    pub fn discover_devices(&self) -> HardwareResult<Vec<String>> {
        let mut device_ids = Vec::new();

        // Create a single CPU device (most systems have one logical CPU)
        let device_id = "cpu_0".to_string();
        let device = CPUDevice::new(device_id.clone());

        {
            let mut devices = self.devices.lock().map_err(|_| {
                TrustformersError::hardware_error("Failed to lock devices", "device_discovery")
            })?;
            devices.insert(device_id.clone(), device);
        }

        device_ids.push(device_id);

        // Could potentially enumerate NUMA nodes as separate devices
        if self.config.enable_monitoring {
            self.setup_monitoring()?;
        }

        Ok(device_ids)
    }

    /// Setup performance monitoring for CPU devices
    fn setup_monitoring(&self) -> HardwareResult<()> {
        // Setup CPU performance monitoring using system information
        #[cfg(target_os = "linux")]
        {
            // Check if perf counters are available
            if std::path::Path::new("/proc/cpuinfo").exists() {
                log::info!("CPU performance monitoring enabled for Linux");
            }
        }
        #[cfg(target_os = "macos")]
        {
            // macOS system monitoring
            log::info!("CPU performance monitoring enabled for macOS");
        }
        #[cfg(target_os = "windows")]
        {
            // Windows performance counters
            log::info!("CPU performance monitoring enabled for Windows");
        }
        Ok(())
    }

    /// Get CPU device by ID
    pub fn get_device(&self, device_id: &str) -> Option<CPUDevice> {
        if let Ok(devices) = self.devices.lock() {
            devices.get(device_id).cloned()
        } else {
            None
        }
    }

    /// Execute operation on specific CPU device
    pub fn execute_on_device(
        &self,
        device_id: &str,
        operation: &str,
        inputs: &[Tensor],
        mode: OperationMode,
        precision: PrecisionMode,
    ) -> HardwareResult<Vec<Tensor>> {
        let device = self.get_device(device_id).ok_or_else(|| {
            TrustformersError::hardware_error("Device not found", "execute_on_device")
        })?;

        device.execute_operation(operation, inputs, mode, precision)
    }

    /// Get device count
    pub fn device_count(&self) -> usize {
        self.devices.lock().map(|d| d.len()).unwrap_or(0)
    }
}

#[async_trait]
impl HardwareBackend for CPUBackend {
    fn name(&self) -> &str {
        "CPU Backend"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    async fn discover_devices(&self) -> HardwareResult<Vec<Box<dyn HardwareDevice>>> {
        let mut devices = Vec::new();

        // Create CPU devices based on system configuration
        for cpu_id in 0..num_cpus::get() {
            let device_id = format!("cpu-{}", cpu_id);
            let cpu_device = CPUDevice::new(device_id.clone());
            devices.push(Box::new(cpu_device) as Box<dyn HardwareDevice>);
        }

        // Store devices in the backend
        if let Ok(device_map) = self.devices.lock() {
            for (i, device) in devices.iter().enumerate() {
                let device_id = format!("cpu-{}", i);
                // We can't store the boxed device directly due to ownership
                // This would need to be refactored for proper device management
            }
        }

        Ok(devices)
    }

    async fn create_device(
        &self,
        config: &HardwareConfig,
    ) -> HardwareResult<Box<dyn HardwareDevice>> {
        let device_id = if config.device_id.is_empty() {
            "cpu-0".to_string()
        } else {
            config.device_id.clone()
        };
        let cpu_device = CPUDevice::new(device_id);
        Ok(Box::new(cpu_device))
    }

    fn is_compatible(&self, hardware_type: HardwareType) -> bool {
        hardware_type == HardwareType::CPU
    }

    fn supported_operations(&self) -> &[String] {
        &self.supported_ops
    }

    fn validate_config(&self, _config: &HardwareConfig) -> HardwareResult<()> {
        // CPU devices are generally always valid
        Ok(())
    }
}

impl GPUBackend {
    /// Create a new GPU backend
    pub fn new(backend_type: GPUBackendType) -> Self {
        let supported_ops = vec![
            "add".to_string(),
            "mul".to_string(),
            "matmul".to_string(),
            "conv2d".to_string(),
            "relu".to_string(),
            "softmax".to_string(),
        ];

        Self {
            devices: Arc::new(Mutex::new(HashMap::new())),
            backend_type,
            config: GPUBackendConfig::default(),
            supported_ops,
        }
    }

    /// Create GPU backend with custom configuration
    pub fn with_config(backend_type: GPUBackendType, config: GPUBackendConfig) -> Self {
        let supported_ops = vec![
            "add".to_string(),
            "mul".to_string(),
            "matmul".to_string(),
            "conv2d".to_string(),
            "relu".to_string(),
            "softmax".to_string(),
        ];

        Self {
            devices: Arc::new(Mutex::new(HashMap::new())),
            backend_type,
            config,
            supported_ops,
        }
    }

    /// Discover available GPU devices
    pub fn discover_devices(&self) -> HardwareResult<Vec<String>> {
        let device_ids = match self.backend_type {
            GPUBackendType::CUDA => self.discover_cuda_devices()?,
            GPUBackendType::ROCm => self.discover_rocm_devices()?,
            GPUBackendType::OpenCL => self.discover_opencl_devices()?,
            GPUBackendType::Metal => self.discover_metal_devices()?,
            GPUBackendType::Vulkan => self.discover_vulkan_devices()?,
            GPUBackendType::Unknown => {
                return Err(TrustformersError::hardware_error(
                    "Unknown GPU backend type",
                    "discover_devices",
                ));
            },
        };

        // Create GPU device instances
        {
            let mut devices = self.devices.lock().map_err(|_| {
                TrustformersError::hardware_error("Failed to lock devices", "discover_devices")
            })?;

            for device_id in &device_ids {
                let device = GPUDevice::new(device_id.clone(), self.backend_type);
                devices.insert(device_id.clone(), device);
            }
        }

        if self.config.enable_monitoring {
            self.setup_monitoring()?;
        }

        Ok(device_ids)
    }

    fn discover_cuda_devices(&self) -> HardwareResult<Vec<String>> {
        // Enhanced CUDA device discovery
        #[cfg(feature = "cuda")]
        {
            use std::process::Command;

            // Try to get GPU count using nvidia-smi
            if let Ok(output) = Command::new("nvidia-smi")
                .args([
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ])
                .output()
            {
                if output.status.success() {
                    let devices_str = String::from_utf8_lossy(&output.stdout);
                    let devices: Vec<String> = devices_str
                        .lines()
                        .enumerate()
                        .map(|(i, line)| {
                            let parts: Vec<&str> = line.split(',').collect();
                            let name = parts.first().unwrap_or(&"Unknown GPU").trim();
                            let memory = parts.get(1).unwrap_or(&"0").trim();
                            format!("cuda_{}_{}_{}MB", i, name.replace(' ', "_"), memory)
                        })
                        .collect();

                    if !devices.is_empty() {
                        return Ok(devices);
                    }
                }
            }

            // `nvidia-smi --query-gpu` above is the authoritative source: if
            // it didn't report a device (missing tool, non-zero exit, or an
            // empty device list), there is nothing real to report. A coarse
            // "is a CUDA runtime library present" signal (`is_cuda_available`)
            // is not proof a GPU is attached, so it must not be used to
            // synthesize a phantom device here.
            Ok(vec![])
        }

        #[cfg(not(feature = "cuda"))]
        Ok(vec![])
    }

    fn discover_rocm_devices(&self) -> HardwareResult<Vec<String>> {
        // Enhanced ROCm device discovery
        #[cfg(feature = "rocm")]
        {
            use std::process::Command;

            // Try to get ROCm device info using rocm-smi
            if let Ok(output) = Command::new("rocm-smi")
                .args(["--showproductname", "--showmeminfo", "vram", "--csv"])
                .output()
            {
                if output.status.success() {
                    let devices_str = String::from_utf8_lossy(&output.stdout);
                    let mut devices = Vec::new();

                    for (i, line) in devices_str.lines().enumerate() {
                        if line.starts_with("GPU") || line.contains("card") {
                            let parts: Vec<&str> = line.split(',').collect();
                            if parts.len() >= 2 {
                                let device_info = parts[1].trim();
                                devices.push(format!(
                                    "rocm_{}_{}",
                                    i,
                                    device_info.replace(' ', "_")
                                ));
                            }
                        }
                    }

                    if !devices.is_empty() {
                        return Ok(devices);
                    }
                }
            }

            // Alternative: Try using /sys filesystem for AMD GPUs
            if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
                let mut amd_devices = Vec::new();
                for (i, entry) in entries.enumerate() {
                    if let Ok(entry) = entry {
                        let name = entry.file_name();
                        if let Some(name_str) = name.to_str() {
                            if name_str.starts_with("card") && !name_str.contains("-") {
                                // Check if it's an AMD GPU
                                let vendor_path =
                                    format!("/sys/class/drm/{}/device/vendor", name_str);
                                if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                                    if vendor.trim() == "0x1002" {
                                        // AMD vendor ID
                                        amd_devices.push(format!("rocm_{}_AMD_GPU", i));
                                    }
                                }
                            }
                        }
                    }
                }

                if !amd_devices.is_empty() {
                    return Ok(amd_devices);
                }
            }

            // As with CUDA above: `rocm-smi` output and the `/sys/class/drm`
            // vendor-ID scan are the authoritative sources. If neither found
            // a device, report none rather than synthesizing one from the
            // coarser `is_rocm_available` library-presence check.
            Ok(vec![])
        }

        #[cfg(not(feature = "rocm"))]
        Ok(vec![])
    }

    fn discover_opencl_devices(&self) -> HardwareResult<Vec<String>> {
        // OpenCL device discovery using system calls
        if Self::is_opencl_available() {
            #[cfg(feature = "opencl")]
            {
                // Try to use clinfo command if available
                use std::process::Command;
                if let Ok(output) = Command::new("clinfo").arg("--list").output() {
                    if output.status.success() {
                        let output_str = String::from_utf8_lossy(&output.stdout);
                        let devices: Vec<String> = output_str
                            .lines()
                            .filter(|line| line.contains("Device"))
                            .enumerate()
                            .map(|(i, _)| format!("gpu_opencl_{}", i))
                            .collect();
                        return Ok(devices);
                    }
                }
                // `clinfo` is missing, failed, or reported zero devices: the
                // OpenCL *loader* being present (`is_opencl_available`) does
                // not mean a usable OpenCL device exists, so report none
                // rather than fabricating "gpu_opencl_0".
                Ok(vec![])
            }
            #[cfg(not(feature = "opencl"))]
            Ok(vec![])
        } else {
            Ok(vec![])
        }
    }

    fn discover_metal_devices(&self) -> HardwareResult<Vec<String>> {
        // Query the real Metal API directly (already a dependency of this
        // exact cfg block) instead of shelling out to `system_profiler`
        // (1-3s per call) and string-parsing its output, or falling back to
        // a `sysctl` CPU-brand guess that has no arm for anything newer
        // than M3. This mirrors `Device::metal_if_available` in
        // `crate::device`, which already does it right.
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let devices: Vec<String> = metal::Device::all()
                .iter()
                .enumerate()
                .map(|(i, device)| format!("metal_{}_{}", i, device.name().replace(' ', "_")))
                .collect();
            Ok(devices)
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        Ok(vec![])
    }

    fn discover_vulkan_devices(&self) -> HardwareResult<Vec<String>> {
        // Vulkan device discovery using vulkaninfo if available
        if Self::is_vulkan_available() {
            #[cfg(feature = "vulkan")]
            {
                use std::process::Command;
                if let Ok(output) = Command::new("vulkaninfo").arg("--summary").output() {
                    if output.status.success() {
                        let output_str = String::from_utf8_lossy(&output.stdout);
                        let devices: Vec<String> = output_str
                            .lines()
                            .filter(|line| line.contains("deviceName"))
                            .enumerate()
                            .map(|(i, _)| format!("gpu_vulkan_{}", i))
                            .collect();
                        if !devices.is_empty() {
                            return Ok(devices);
                        }
                    }
                }
                // `vulkaninfo` is missing, failed, or reported zero devices:
                // the Vulkan loader being present (`is_vulkan_available`)
                // does not mean a usable Vulkan device exists, so report
                // none rather than fabricating "gpu_vulkan_0".
                Ok(vec![])
            }
            #[cfg(not(feature = "vulkan"))]
            Ok(vec![])
        } else {
            Ok(vec![])
        }
    }

    /// Real availability probe: checks the driver CLI's exit status (not
    /// merely that the process could be spawned) before checking for the
    /// runtime library on disk.
    #[allow(dead_code)]
    pub(crate) fn is_cuda_available() -> bool {
        // Check CUDA availability with runtime detection
        #[cfg(feature = "cuda")]
        {
            // Check for nvidia-smi command succeeding (not just spawning:
            // e.g. a `nvidia-smi` shim that always exits non-zero when no
            // driver is loaded must not be read as "available").
            if std::process::Command::new("nvidia-smi")
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
            {
                return true;
            }
            // Check for CUDA runtime library
            #[cfg(target_os = "linux")]
            {
                std::path::Path::new("/usr/local/cuda/lib64/libcudart.so").exists()
                    || std::path::Path::new("/usr/lib/x86_64-linux-gnu/libcudart.so").exists()
            }
            #[cfg(target_os = "windows")]
            {
                // Check for CUDA installation on Windows
                std::env::var("CUDA_PATH").is_ok()
            }
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            false
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    #[allow(dead_code)]
    pub(crate) fn is_rocm_available() -> bool {
        // Check ROCm availability with runtime detection
        #[cfg(feature = "rocm")]
        {
            // Check for rocm-smi command succeeding
            if std::process::Command::new("rocm-smi")
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
            {
                return true;
            }
            // Check for ROCm runtime library
            #[cfg(target_os = "linux")]
            {
                std::path::Path::new("/opt/rocm/lib/libhip_hcc.so").exists()
                    || std::path::Path::new("/opt/rocm/lib/libamdhip64.so").exists()
            }
            #[cfg(not(target_os = "linux"))]
            false
        }
        #[cfg(not(feature = "rocm"))]
        false
    }

    pub(crate) fn is_opencl_available() -> bool {
        // Check OpenCL availability with runtime detection
        #[cfg(feature = "opencl")]
        {
            // Check for clinfo command succeeding
            if std::process::Command::new("clinfo").output().is_ok_and(|o| o.status.success()) {
                return true;
            }
            // Check for OpenCL runtime library
            #[cfg(target_os = "linux")]
            {
                std::path::Path::new("/usr/lib/x86_64-linux-gnu/libOpenCL.so").exists()
            }
            #[cfg(target_os = "macos")]
            {
                std::path::Path::new("/System/Library/Frameworks/OpenCL.framework").exists()
            }
            #[cfg(target_os = "windows")]
            {
                // No command-line probe used here; check for the loader DLL
                // instead of assuming availability.
                std::path::Path::new("C:\\Windows\\System32\\OpenCL.dll").exists()
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            false
        }
        #[cfg(not(feature = "opencl"))]
        false
    }

    #[allow(dead_code)]
    pub(crate) fn is_metal_available(&self) -> bool {
        // Actually obtain a Metal device handle rather than checking that
        // the framework bundle exists on disk (which is true on every
        // shipping macOS install, including ones with no usable GPU, e.g.
        // some virtualized/headless CI environments).
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            metal::Device::system_default().is_some()
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        false
    }

    pub(crate) fn is_vulkan_available() -> bool {
        // Check Vulkan availability with runtime detection
        #[cfg(feature = "vulkan")]
        {
            // Check for vulkaninfo command succeeding
            if std::process::Command::new("vulkaninfo")
                .output()
                .is_ok_and(|o| o.status.success())
            {
                return true;
            }
            // Check for Vulkan loader library
            #[cfg(target_os = "linux")]
            {
                std::path::Path::new("/usr/lib/x86_64-linux-gnu/libvulkan.so").exists()
                    || std::path::Path::new("/usr/lib/libvulkan.so").exists()
            }
            #[cfg(target_os = "macos")]
            {
                std::path::Path::new("/usr/local/lib/libvulkan.dylib").exists()
                    || std::path::Path::new("/opt/homebrew/lib/libvulkan.dylib").exists()
            }
            #[cfg(target_os = "windows")]
            {
                // No command-line probe used here; check for the loader DLL
                // instead of assuming availability.
                std::path::Path::new("C:\\Windows\\System32\\vulkan-1.dll").exists()
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            false
        }
        #[cfg(not(feature = "vulkan"))]
        false
    }

    fn setup_monitoring(&self) -> HardwareResult<()> {
        // Setup GPU performance monitoring based on backend type
        match self.backend_type {
            GPUBackendType::CUDA => {
                #[cfg(feature = "cuda")]
                {
                    log::info!("Setting up CUDA performance monitoring");
                    // NVML initialization for GPU monitoring would go here
                }
            },
            GPUBackendType::ROCm => {
                #[cfg(feature = "rocm")]
                {
                    log::info!("Setting up ROCm performance monitoring");
                    // ROCm SMI monitoring setup would go here
                }
            },
            GPUBackendType::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    log::info!("Setting up OpenCL performance monitoring");
                    // OpenCL profiling setup would go here
                }
            },
            GPUBackendType::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    log::info!("Setting up Metal performance monitoring");
                    // Metal performance counters setup would go here
                }
            },
            GPUBackendType::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    log::info!("Setting up Vulkan performance monitoring");
                    // Vulkan timestamp queries setup would go here
                }
            },
            GPUBackendType::Unknown => {
                log::warn!("GPU backend type unknown, skipping performance monitoring setup");
                // No monitoring setup for unknown backends
            },
        }
        Ok(())
    }

    /// Get GPU device by ID
    pub fn get_device(&self, device_id: &str) -> Option<GPUDevice> {
        if let Ok(devices) = self.devices.lock() {
            devices.get(device_id).cloned()
        } else {
            None
        }
    }

    /// Execute operation on specific GPU device
    pub fn execute_on_device(
        &self,
        device_id: &str,
        operation: &str,
        inputs: &[Tensor],
        mode: OperationMode,
        precision: PrecisionMode,
    ) -> HardwareResult<Vec<Tensor>> {
        let device = self.get_device(device_id).ok_or_else(|| {
            TrustformersError::hardware_error("Device not found", "execute_on_device")
        })?;

        device.execute_operation(operation, inputs, mode, precision)
    }

    /// Get device count
    pub fn device_count(&self) -> usize {
        self.devices.lock().map(|d| d.len()).unwrap_or(0)
    }

    /// Get backend type
    pub fn backend_type(&self) -> GPUBackendType {
        self.backend_type
    }
}

#[async_trait]
impl HardwareBackend for GPUBackend {
    fn name(&self) -> &str {
        match self.backend_type {
            GPUBackendType::CUDA => "CUDA GPU Backend",
            GPUBackendType::ROCm => "ROCm GPU Backend",
            GPUBackendType::OpenCL => "OpenCL GPU Backend",
            GPUBackendType::Metal => "Metal GPU Backend",
            GPUBackendType::Vulkan => "Vulkan GPU Backend",
            GPUBackendType::Unknown => "Unknown GPU Backend",
        }
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    async fn discover_devices(&self) -> HardwareResult<Vec<Box<dyn HardwareDevice>>> {
        let mut devices = Vec::new();

        match self.backend_type {
            GPUBackendType::CUDA => {
                if cfg!(feature = "cuda") {
                    let device = GPUDevice::new("gpu-cuda-0".to_string(), self.backend_type);
                    devices.push(Box::new(device) as Box<dyn HardwareDevice>);
                }
            },
            GPUBackendType::ROCm => {
                if cfg!(feature = "rocm") {
                    let device = GPUDevice::new("gpu-rocm-0".to_string(), self.backend_type);
                    devices.push(Box::new(device) as Box<dyn HardwareDevice>);
                }
            },
            GPUBackendType::OpenCL => {
                if cfg!(feature = "opencl") {
                    let device = GPUDevice::new("gpu-opencl-0".to_string(), self.backend_type);
                    devices.push(Box::new(device) as Box<dyn HardwareDevice>);
                }
            },
            GPUBackendType::Metal => {
                if cfg!(all(target_os = "macos", feature = "metal")) {
                    let device = GPUDevice::new("gpu-metal-0".to_string(), self.backend_type);
                    devices.push(Box::new(device) as Box<dyn HardwareDevice>);
                }
            },
            GPUBackendType::Vulkan => {
                if cfg!(feature = "vulkan") {
                    let device = GPUDevice::new("gpu-vulkan-0".to_string(), self.backend_type);
                    devices.push(Box::new(device) as Box<dyn HardwareDevice>);
                }
            },
            GPUBackendType::Unknown => {
                return Err(TrustformersError::hardware_error(
                    "Unknown GPU backend type",
                    "discover_devices",
                ));
            },
        }

        Ok(devices)
    }

    async fn create_device(
        &self,
        config: &HardwareConfig,
    ) -> HardwareResult<Box<dyn HardwareDevice>> {
        let device_id = if config.device_id.is_empty() {
            "gpu-0".to_string()
        } else {
            config.device_id.clone()
        };
        let gpu_device = GPUDevice::new(device_id, self.backend_type);
        Ok(Box::new(gpu_device))
    }

    fn is_compatible(&self, hardware_type: HardwareType) -> bool {
        hardware_type == HardwareType::GPU
    }

    fn supported_operations(&self) -> &[String] {
        &self.supported_ops
    }

    fn validate_config(&self, config: &HardwareConfig) -> HardwareResult<()> {
        if config.hardware_type != HardwareType::GPU {
            return Err(TrustformersError::hardware_error(
                "Config not for GPU hardware",
                "is_compatible",
            ));
        }
        Ok(())
    }
}

impl GPUBackend {
    #[allow(dead_code)]
    fn synchronize_cuda(&self) -> HardwareResult<()> {
        // CUDA device synchronization (placeholder)
        Ok(())
    }

    #[allow(dead_code)]
    fn synchronize_rocm(&self) -> HardwareResult<()> {
        // ROCm device synchronization (placeholder)
        Ok(())
    }

    #[allow(dead_code)]
    fn synchronize_opencl(&self) -> HardwareResult<()> {
        // OpenCL device synchronization (placeholder)
        Ok(())
    }

    #[allow(dead_code)]
    fn synchronize_metal(&self) -> HardwareResult<()> {
        // Metal device synchronization (placeholder)
        Ok(())
    }

    #[allow(dead_code)]
    fn synchronize_vulkan(&self) -> HardwareResult<()> {
        // Vulkan device synchronization (placeholder)
        Ok(())
    }
}

impl Default for CPUBackendConfig {
    fn default() -> Self {
        Self {
            num_threads: std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4),
            enable_simd: true,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            enable_monitoring: true,
        }
    }
}

impl Default for GPUBackendConfig {
    fn default() -> Self {
        Self {
            memory_pool_size: 2 * 1024 * 1024 * 1024, // 2GB
            enable_unified_memory: false,
            stream_count: 4,
            enable_kernel_fusion: true,
            enable_monitoring: true,
        }
    }
}

impl Default for CPUBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CPUBackend::new();
        assert!(!backend.supported_ops.is_empty());
    }

    #[test]
    fn test_cpu_backend_default() {
        let backend = CPUBackend::default();
        assert!(!backend.supported_ops.is_empty());
    }

    #[test]
    fn test_cpu_backend_config_default() {
        let config = CPUBackendConfig::default();
        assert!(config.num_threads > 0);
        assert!(config.enable_simd);
    }

    #[test]
    fn test_cpu_backend_with_config() {
        let config = CPUBackendConfig {
            num_threads: 8,
            enable_simd: true,
            memory_pool_size: 1024 * 1024,
            enable_monitoring: false,
        };
        let backend = CPUBackend::with_config(config);
        assert!(!backend.supported_ops.is_empty());
    }

    #[test]
    fn test_cpu_backend_supported_ops_include_add() {
        let backend = CPUBackend::new();
        assert!(backend.supported_ops.contains(&"add".to_string()));
    }

    #[test]
    fn test_cpu_backend_supported_ops_include_matmul() {
        let backend = CPUBackend::new();
        assert!(backend.supported_ops.contains(&"matmul".to_string()));
    }

    #[test]
    fn test_cpu_backend_supported_ops_include_relu() {
        let backend = CPUBackend::new();
        assert!(backend.supported_ops.contains(&"relu".to_string()));
    }

    #[test]
    fn test_cpu_backend_discover_devices() {
        let backend = CPUBackend::new();
        let result = backend.discover_devices();
        assert!(result.is_ok());
        if let Ok(devices) = result {
            assert!(!devices.is_empty());
        }
    }

    #[test]
    fn test_cpu_backend_device_count_initially_zero() {
        let backend = CPUBackend::new();
        assert_eq!(backend.device_count(), 0);
    }

    #[test]
    fn test_cpu_backend_get_nonexistent_device() {
        let backend = CPUBackend::new();
        let dev = backend.get_device("nonexistent");
        assert!(dev.is_none());
    }

    #[test]
    fn test_gpu_backend_cuda_creation() {
        let backend = GPUBackend::new(GPUBackendType::CUDA);
        assert_eq!(backend.backend_type(), GPUBackendType::CUDA);
    }

    #[test]
    fn test_gpu_backend_metal_creation() {
        let backend = GPUBackend::new(GPUBackendType::Metal);
        assert_eq!(backend.backend_type(), GPUBackendType::Metal);
    }

    #[test]
    fn test_gpu_backend_rocm_creation() {
        let backend = GPUBackend::new(GPUBackendType::ROCm);
        assert_eq!(backend.backend_type(), GPUBackendType::ROCm);
    }

    #[test]
    fn test_gpu_backend_config_default() {
        let config = GPUBackendConfig::default();
        assert!(config.stream_count > 0);
    }

    #[test]
    fn test_gpu_backend_with_config() {
        let config = GPUBackendConfig {
            memory_pool_size: 4 * 1024 * 1024,
            enable_unified_memory: true,
            stream_count: 4,
            enable_kernel_fusion: true,
            enable_monitoring: false,
        };
        let backend = GPUBackend::with_config(GPUBackendType::Vulkan, config);
        assert_eq!(backend.backend_type(), GPUBackendType::Vulkan);
    }

    #[test]
    fn test_gpu_backend_device_count_initially_zero() {
        let backend = GPUBackend::new(GPUBackendType::CUDA);
        assert_eq!(backend.device_count(), 0);
    }

    #[test]
    fn test_gpu_backend_get_nonexistent_device() {
        let backend = GPUBackend::new(GPUBackendType::Metal);
        let dev = backend.get_device("nonexistent");
        assert!(dev.is_none());
    }

    #[test]
    fn test_gpu_backend_supported_ops() {
        let backend = GPUBackend::new(GPUBackendType::CUDA);
        assert!(!backend.supported_ops.is_empty());
    }

    #[test]
    fn test_cpu_backend_config_clone() {
        let config = CPUBackendConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.num_threads, config.num_threads);
        assert_eq!(cloned.enable_simd, config.enable_simd);
    }

    #[test]
    fn test_gpu_backend_config_clone() {
        let config = GPUBackendConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.stream_count, config.stream_count);
    }

    #[test]
    fn test_cpu_backend_discover_and_get() {
        let backend = CPUBackend::new();
        if let Ok(device_ids) = backend.discover_devices() {
            if let Some(first_id) = device_ids.first() {
                let dev = backend.get_device(first_id);
                assert!(dev.is_some());
            }
        }
    }

    #[test]
    fn test_cpu_backend_execute_add() {
        let backend = CPUBackend::new();
        let _ = backend.discover_devices();
        let a = Tensor::from_data(vec![1.0, 2.0], &[2]).expect("create failed");
        let b = Tensor::from_data(vec![3.0, 4.0], &[2]).expect("create failed");
        if let Ok(device_ids) = backend.discover_devices() {
            if let Some(first_id) = device_ids.first() {
                let result = backend.execute_on_device(
                    first_id,
                    "add",
                    &[a, b],
                    OperationMode::Performance,
                    PrecisionMode::Single,
                );
                assert!(result.is_ok());
            }
        }
    }

    /// Regression test: `discover_metal_devices`/`is_metal_available` used
    /// to shell out to `system_profiler`/`sysctl` and string-match chipset
    /// names (with no arm past "M3"), or merely check that the Metal
    /// framework bundle exists on disk. They must now query the real Metal
    /// API and agree with each other: on any Mac with a default Metal
    /// device, discovery must report at least one device.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_metal_discovery_uses_real_metal_api() {
        let backend = GPUBackend::new(GPUBackendType::Metal);
        let available = backend.is_metal_available();
        assert_eq!(
            available,
            metal::Device::system_default().is_some(),
            "is_metal_available must agree with a direct Metal API probe"
        );

        if available {
            let devices =
                backend.discover_metal_devices().expect("metal discovery should not error");
            assert!(
                !devices.is_empty(),
                "a real default Metal device exists but discovery reported none"
            );
            // The old sysctl-brand-string fallback only ever emitted names
            // matching "metal_M1_GPU" / "metal_M2_GPU" / "metal_M3_GPU" /
            // "metal_Apple_Silicon_GPU"; the real API path is not
            // constrained to that fixed set (it echoes `device.name()`).
            for d in &devices {
                assert!(d.starts_with("metal_"), "unexpected device id format: {d}");
            }
        }
    }
}
