// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Intel oneAPI implementation for TrustformeRS hardware acceleration
//!
//! This module provides actual Intel oneAPI/DPC++/SYCL runtime API bindings
//! for Intel GPUs including Arc, Xe, and Data Center GPU Max Series.

use crate::errors::{Result, TrustformersError};
use crate::kernels::intel_kernels::{
    IntelDevice, IntelKernel, IntelKernelConfig, IntelPrecision, IntelUtils,
};
use crate::tensor::Tensor;
use std::sync::{Arc, Mutex, OnceLock};

/// Intel oneAPI implementation with hardware bindings
pub struct IntelImpl {
    /// Intel kernel manager
    kernel_manager: Arc<Mutex<IntelKernel>>,
    /// Device information
    device: IntelDevice,
    /// Available devices
    available_devices: Vec<IntelDevice>,
    /// Performance statistics
    stats: Arc<Mutex<IntelStats>>,
}

/// Intel oneAPI performance statistics
#[derive(Debug, Clone, Default)]
pub struct IntelStats {
    /// Total operations executed
    pub total_operations: u64,
    /// Total execution time (microseconds)
    pub total_time_us: u64,
    /// Memory transfers to device (bytes)
    pub memory_h2d_bytes: u64,
    /// Memory transfers from device (bytes)
    pub memory_d2h_bytes: u64,
    /// Kernel compilation time (microseconds)
    pub compilation_time_us: u64,
    /// Number of kernel launches
    pub kernel_launches: u64,
}

/// Global Intel oneAPI instance
static INTEL_INSTANCE: OnceLock<Arc<IntelImpl>> = OnceLock::new();

impl IntelImpl {
    /// Initialize Intel oneAPI with the first available device
    pub fn new() -> Result<Self> {
        // Detect available Intel GPU devices
        let available_devices = IntelUtils::detect_devices()?;

        if available_devices.is_empty() {
            return Err(TrustformersError::hardware_error(
                "No Intel GPU devices found",
                "intel_device_detection",
            ));
        }

        let device = available_devices[0].clone();

        // Create kernel configuration optimized for the detected device
        let config = IntelKernelConfig {
            device_id: device.id,
            workgroup_size: IntelUtils::get_optimal_workgroup_size(1024, device.max_workgroup_size),
            preferred_workgroup_size_multiple: if device.sub_group_sizes.contains(&32) {
                32
            } else {
                16
            },
            max_workgroup_size: device.max_workgroup_size,
            local_memory_size: device.local_memory_size,
            global_memory_size: device.global_memory_size,
            compute_units: device.compute_units,
            max_clock_frequency: device.max_clock_frequency,
            sub_group_size: device.sub_group_sizes[0],
            enable_profiling: true,
            enable_fp16: device.supports_fp16,
            enable_dpas: device.supports_dpas,
        };

        // Initialize kernel manager
        let kernel_manager = IntelKernel::new(config).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to initialize Intel kernels: {}", e),
                "intel_kernel_init",
            )
        })?;

        Ok(Self {
            kernel_manager: Arc::new(Mutex::new(kernel_manager)),
            device,
            available_devices,
            stats: Arc::new(Mutex::new(IntelStats::default())),
        })
    }

    /// Get global Intel oneAPI instance
    pub fn global() -> Result<&'static Arc<IntelImpl>> {
        INTEL_INSTANCE.get_or_init(|| {
            Arc::new(Self::new().unwrap_or_else(|_| {
                // Create a fallback instance with CPU emulation
                Self::create_fallback()
            }))
        });
        Ok(INTEL_INSTANCE.get().expect("Intel instance should exist after initialization"))
    }

    /// Create fallback instance when Intel GPU is not available
    fn create_fallback() -> Self {
        // Create a mock device for CPU fallback
        let mock_device = IntelDevice {
            id: 0,
            name: "Intel CPU Fallback".to_string(),
            vendor: "Intel Corporation".to_string(),
            driver_version: "fallback".to_string(),
            device_type: crate::kernels::intel_kernels::IntelDeviceType::Unknown,
            compute_units: 1,
            max_clock_frequency: 3000,
            local_memory_size: 32768,
            global_memory_size: 32 * 1024 * 1024 * 1024, // 32GB system RAM
            max_workgroup_size: 256,
            sub_group_sizes: vec![1],
            extensions: vec![],
            supports_fp16: false,
            supports_dpas: false,
            supports_systolic_arrays: false,
        };

        let config = IntelKernelConfig::default();
        let kernel_manager = IntelKernel::new(config).unwrap();

        Self {
            kernel_manager: Arc::new(Mutex::new(kernel_manager)),
            device: mock_device.clone(),
            available_devices: vec![mock_device],
            stats: Arc::new(Mutex::new(IntelStats::default())),
        }
    }

    /// Check if Intel oneAPI is available
    pub fn is_available() -> bool {
        // Try to detect Intel devices
        IntelUtils::detect_devices().map(|devices| !devices.is_empty()).unwrap_or(false)
    }

    /// Execute matrix multiplication using Intel oneAPI
    pub fn matmul(&self, a: &Tensor, b: &Tensor, c: &mut Tensor) -> Result<()> {
        let start_time = std::time::Instant::now();

        let mut kernel_manager = self.kernel_manager.lock().expect("Lock poisoned");
        let precision = IntelUtils::get_recommended_precision(&self.device);

        // Execute GEMM operation
        let result = kernel_manager.gemm(a, b, c, 1.0, 0.0, precision);

        // Update statistics
        let elapsed = start_time.elapsed();
        let mut stats = self.stats.lock().expect("Lock poisoned");
        stats.total_operations += 1;
        stats.total_time_us += elapsed.as_micros() as u64;
        stats.kernel_launches += 1;

        result
    }

    /// Execute Flash Attention using Intel oneAPI
    pub fn flash_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        let mut kernel_manager = self.kernel_manager.lock().expect("Lock poisoned");
        let precision = IntelUtils::get_recommended_precision(&self.device);

        // Calculate attention scale
        let head_dim = query.shape().last().copied().unwrap_or(64) as f32;
        let scale = 1.0 / head_dim.sqrt();

        // Execute attention operation
        let result = kernel_manager.attention(query, key, value, output, scale, precision);

        // Update statistics
        let elapsed = start_time.elapsed();
        let mut stats = self.stats.lock().expect("Lock poisoned");
        stats.total_operations += 1;
        stats.total_time_us += elapsed.as_micros() as u64;
        stats.kernel_launches += 1;

        result
    }

    /// Execute layer normalization using Intel oneAPI
    pub fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        output: &mut Tensor,
        eps: f32,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        let mut kernel_manager = self.kernel_manager.lock().expect("Lock poisoned");
        let precision = IntelUtils::get_recommended_precision(&self.device);

        // Execute layer normalization
        let result = kernel_manager.layer_norm(input, weight, bias, output, eps, precision);

        // Update statistics
        let elapsed = start_time.elapsed();
        let mut stats = self.stats.lock().expect("Lock poisoned");
        stats.total_operations += 1;
        stats.total_time_us += elapsed.as_micros() as u64;
        stats.kernel_launches += 1;

        result
    }

    /// Get device information
    pub fn device_info(&self) -> String {
        format!(
            "Intel {} (Driver: {}, Compute Units: {}, Memory: {:.1} GB, FP16: {}, DPAS: {})",
            self.device.name,
            self.device.driver_version,
            self.device.compute_units,
            self.device.global_memory_size as f64 / (1024.0 * 1024.0 * 1024.0),
            self.device.supports_fp16,
            self.device.supports_dpas
        )
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> Result<(usize, usize)> {
        let kernel_manager = self.kernel_manager.lock().expect("Lock poisoned");
        let memory_stats = kernel_manager.memory_stats()?;

        // Return (used_memory, total_memory)
        Ok((memory_stats.total_allocated, self.device.global_memory_size))
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> IntelStats {
        self.stats.lock().expect("Lock poisoned").clone()
    }

    /// Reset performance statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().expect("Lock poisoned");
        *stats = IntelStats::default();
    }

    /// List available Intel devices
    pub fn list_devices(&self) -> &[IntelDevice] {
        &self.available_devices
    }

    /// Get current device
    pub fn current_device(&self) -> &IntelDevice {
        &self.device
    }

    /// Check if XMX (Xe Matrix Extensions) is supported
    pub fn has_xmx_support(&self) -> bool {
        IntelUtils::has_xmx_support(&self.device)
    }

    /// Get recommended precision for current device
    pub fn recommended_precision(&self) -> IntelPrecision {
        IntelUtils::get_recommended_precision(&self.device)
    }
}

/// Public API for Intel oneAPI hardware acceleration
pub mod api {
    use super::*;

    /// Initialize Intel oneAPI backend
    pub fn init_intel() -> Result<()> {
        IntelImpl::global()?;
        Ok(())
    }

    /// Check if Intel oneAPI is available
    pub fn is_intel_available() -> bool {
        IntelImpl::is_available()
    }

    /// Execute matrix multiplication using Intel oneAPI
    pub fn intel_matmul(a: &Tensor, b: &Tensor, c: &mut Tensor) -> Result<()> {
        let intel = IntelImpl::global()?;
        intel.matmul(a, b, c)
    }

    /// Execute Flash Attention using Intel oneAPI
    pub fn intel_flash_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
    ) -> Result<()> {
        let intel = IntelImpl::global()?;
        intel.flash_attention(query, key, value, output)
    }

    /// Execute layer normalization using Intel oneAPI
    pub fn intel_layer_norm(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        output: &mut Tensor,
        eps: f32,
    ) -> Result<()> {
        let intel = IntelImpl::global()?;
        intel.layer_norm(input, weight, bias, output, eps)
    }

    /// Get Intel device information
    pub fn intel_device_info() -> Result<String> {
        let intel = IntelImpl::global()?;
        Ok(intel.device_info())
    }

    /// Get Intel memory statistics
    pub fn intel_memory_stats() -> Result<(usize, usize)> {
        let intel = IntelImpl::global()?;
        intel.memory_stats()
    }

    /// Get Intel performance statistics
    pub fn intel_performance_stats() -> Result<IntelStats> {
        let intel = IntelImpl::global()?;
        Ok(intel.get_stats())
    }

    /// Reset Intel performance statistics
    pub fn intel_reset_stats() -> Result<()> {
        let intel = IntelImpl::global()?;
        intel.reset_stats();
        Ok(())
    }

    /// List available Intel devices
    pub fn intel_list_devices() -> Result<Vec<IntelDevice>> {
        let intel = IntelImpl::global()?;
        Ok(intel.list_devices().to_vec())
    }

    /// Check if Intel XMX (Xe Matrix Extensions) is supported
    pub fn intel_has_xmx() -> Result<bool> {
        let intel = IntelImpl::global()?;
        Ok(intel.has_xmx_support())
    }

    /// Get recommended precision for Intel device
    pub fn intel_recommended_precision() -> Result<IntelPrecision> {
        let intel = IntelImpl::global()?;
        Ok(intel.recommended_precision())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_intel_initialization() {
        let result = api::init_intel();
        // Should not fail even if Intel GPU is not available (uses fallback)
        assert!(result.is_ok());
    }

    #[test]
    fn test_intel_device_info() {
        let _ = api::init_intel();
        let info = api::intel_device_info();
        assert!(info.is_ok());
        let info_str = info.unwrap();
        assert!(info_str.contains("Intel"));
    }

    #[test]
    fn test_intel_matmul() {
        let _ = api::init_intel();

        let a = Tensor::ones(&[4, 4]).expect("Failed to create ones tensor");
        let b = Tensor::ones(&[4, 4]).expect("Failed to create ones tensor");
        let mut c = Tensor::zeros(&[4, 4]).expect("Failed to create zero tensor");

        let result = api::intel_matmul(&a, &b, &mut c);
        assert!(result.is_ok());
    }

    #[test]
    fn test_intel_stats() {
        let _ = api::init_intel();

        // Reset stats
        let _ = api::intel_reset_stats();

        // Perform an operation
        let a = Tensor::ones(&[2, 2]).expect("Failed to create ones tensor");
        let b = Tensor::ones(&[2, 2]).expect("Failed to create ones tensor");
        let mut c = Tensor::zeros(&[2, 2]).expect("Failed to create zero tensor");
        let _ = api::intel_matmul(&a, &b, &mut c);

        // Check stats
        let stats = api::intel_performance_stats().unwrap();
        assert!(stats.total_operations > 0);
        assert!(stats.kernel_launches > 0);
    }

    #[test]
    fn test_intel_memory_stats() {
        let _ = api::init_intel();
        let stats = api::intel_memory_stats();
        assert!(stats.is_ok());

        let (_used, total) = stats.unwrap();
        assert!(total > 0); // Should have some memory available
    }

    #[test]
    fn test_intel_device_listing() {
        let _ = api::init_intel();
        let devices = api::intel_list_devices();
        assert!(devices.is_ok());

        let device_list = devices.unwrap();
        assert!(!device_list.is_empty()); // Should have at least fallback device
    }

    #[test]
    fn test_intel_precision_recommendation() {
        let _ = api::init_intel();
        let precision = api::intel_recommended_precision();
        assert!(precision.is_ok());

        // Should return some valid precision
        match precision.unwrap() {
            IntelPrecision::FP32 | IntelPrecision::FP16 | IntelPrecision::BF16 => (),
            other => panic!("Unexpected precision recommendation: {:?}", other),
        }
    }

    #[test]
    fn test_intel_flash_attention() {
        let _ = api::init_intel();

        let query = Tensor::ones(&[1, 4, 64]).expect("Failed to create ones tensor");
        let key = Tensor::ones(&[1, 4, 64]).expect("Failed to create ones tensor");
        let value = Tensor::ones(&[1, 4, 64]).expect("Failed to create ones tensor");
        let mut output = Tensor::zeros(&[1, 4, 64]).expect("Failed to create zero tensor");

        let result = api::intel_flash_attention(&query, &key, &value, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_intel_layer_norm() {
        let _ = api::init_intel();

        let input = Tensor::ones(&[2, 128]).expect("Failed to create ones tensor");
        let weight = Tensor::ones(&[128]).expect("Failed to create ones tensor");
        let bias = Tensor::zeros(&[128]).expect("Failed to create zero tensor");
        let mut output = Tensor::zeros(&[2, 128]).expect("Failed to create zero tensor");

        let result = api::intel_layer_norm(&input, &weight, Some(&bias), &mut output, 1e-5);
        assert!(result.is_ok());
    }
}
