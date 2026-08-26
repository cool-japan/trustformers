// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
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

/// Global Intel oneAPI instance: `Ok` once real detection succeeds, or the
/// (stringified, since `TrustformersError` is not `Clone`) error from the
/// one and only detection attempt otherwise. `OnceLock` runs its init
/// closure at most once regardless of outcome, so a failed detection is
/// cached as a failure rather than silently retried into a fabricated
/// fallback on every subsequent call.
static INTEL_INSTANCE: OnceLock<std::result::Result<Arc<IntelImpl>, String>> = OnceLock::new();

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

    /// Get the global Intel oneAPI instance.
    ///
    /// Returns the real, once-computed detection result. Before this fix, a
    /// failed detection (no Intel GPU present, which is every machine today;
    /// see `IntelUtils::detect_devices`) was silently replaced with an
    /// "Intel CPU Fallback" `IntelDevice` reported as vendor "Intel
    /// Corporation", a phantom accelerator. Callers now get the real error
    /// honestly instead.
    pub fn global() -> Result<&'static Arc<IntelImpl>> {
        let slot =
            INTEL_INSTANCE.get_or_init(|| Self::new().map(Arc::new).map_err(|e| e.to_string()));
        match slot {
            Ok(instance) => Ok(instance),
            Err(message) => Err(TrustformersError::hardware_error(
                message,
                "intel_global_init",
            )),
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

        let mut kernel_manager =
            self.kernel_manager.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let precision = IntelUtils::get_recommended_precision(&self.device);

        // Execute GEMM operation
        let result = kernel_manager.gemm(a, b, c, 1.0, 0.0, precision);

        // Update statistics
        let elapsed = start_time.elapsed();
        let mut stats = self.stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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

        let mut kernel_manager =
            self.kernel_manager.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let precision = IntelUtils::get_recommended_precision(&self.device);

        // Calculate attention scale
        let head_dim = query.shape().last().copied().unwrap_or(64) as f32;
        let scale = 1.0 / head_dim.sqrt();

        // Execute attention operation
        let result = kernel_manager.attention(query, key, value, output, scale, precision);

        // Update statistics
        let elapsed = start_time.elapsed();
        let mut stats = self.stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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

        let mut kernel_manager =
            self.kernel_manager.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let precision = IntelUtils::get_recommended_precision(&self.device);

        // Execute layer normalization
        let result = kernel_manager.layer_norm(input, weight, bias, output, eps, precision);

        // Update statistics
        let elapsed = start_time.elapsed();
        let mut stats = self.stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
        let kernel_manager =
            self.kernel_manager.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let memory_stats = kernel_manager.memory_stats()?;

        // Return (used_memory, total_memory)
        Ok((memory_stats.total_allocated, self.device.global_memory_size))
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> IntelStats {
        self.stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }

    /// Reset performance statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
    use crate::kernels::intel_kernels::IntelUtils;
    use crate::tensor::Tensor;

    /// Regression test: `IntelUtils::detect_devices` used to unconditionally
    /// fabricate a fixed "Intel Arc A770" entry regardless of whether any
    /// Intel GPU was actually attached. There is no real oneAPI/Level-Zero
    /// binding in this build, so honest detection on this machine (and any
    /// machine, until a real backend is wired up) must report zero devices,
    /// never a phantom one.
    #[test]
    fn test_detect_devices_reports_no_phantom_devices() {
        let devices = IntelUtils::detect_devices().expect("detect_devices should not error");
        assert!(
            devices.is_empty(),
            "must not report a fabricated device when no real Intel GPU runtime is wired up"
        );
    }

    /// Regression test: `IntelImpl::global()` used to silently substitute a
    /// fabricated "Intel CPU Fallback" `IntelDevice` (reported as vendor
    /// "Intel Corporation") whenever real initialization failed, so every
    /// caller believed a real Intel accelerator was present. It must now
    /// honestly propagate the failure instead.
    #[test]
    fn test_intel_initialization_honestly_fails_without_real_hardware() {
        let result = api::init_intel();
        assert!(
            result.is_err(),
            "must not fabricate a fallback device when no real Intel GPU is present"
        );
    }

    #[test]
    fn test_is_intel_available_is_false_without_real_hardware() {
        assert!(!api::is_intel_available());
    }

    /// Downstream API calls must propagate the same honest error rather
    /// than silently computing against a fabricated device (or panicking).
    #[test]
    fn test_intel_matmul_propagates_honest_error() {
        let a = Tensor::ones(&[4, 4]).expect("Failed to create ones tensor");
        let b = Tensor::ones(&[4, 4]).expect("Failed to create ones tensor");
        let mut c = Tensor::zeros(&[4, 4]).expect("Failed to create zero tensor");

        let result = api::intel_matmul(&a, &b, &mut c);
        assert!(result.is_err());
    }

    #[test]
    fn test_intel_list_devices_propagates_honest_error() {
        assert!(api::intel_list_devices().is_err());
    }
}
