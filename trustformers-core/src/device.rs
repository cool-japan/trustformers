//! Device abstraction for hardware acceleration
//!
//! This module provides a simple Device enum for specifying where computations
//! should be executed (CPU, CUDA GPU, Metal GPU, etc.).

use serde::{Deserialize, Serialize};

/// Device specification for tensor operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Device {
    /// CPU execution
    #[default]
    CPU,
    /// NVIDIA CUDA GPU (with device ID)
    CUDA(usize),
    /// Apple Metal GPU (with device ID)
    Metal(usize),
    /// AMD ROCm GPU (with device ID)
    ROCm(usize),
    /// WebGPU
    WebGPU,
}

impl Device {
    /// Returns true if this device is a GPU
    pub fn is_gpu(&self) -> bool {
        !matches!(self, Device::CPU)
    }

    /// Returns true if this device is Metal GPU
    pub fn is_metal(&self) -> bool {
        matches!(self, Device::Metal(_))
    }

    /// Returns true if this device is CUDA GPU
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::CUDA(_))
    }

    /// Returns true if this device is CPU
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::CPU)
    }

    /// Returns the device ID for GPU devices
    pub fn device_id(&self) -> Option<usize> {
        match self {
            Device::CUDA(id) | Device::Metal(id) | Device::ROCm(id) => Some(*id),
            _ => None,
        }
    }

    /// Create a Metal device, or CPU if Metal is not available
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn metal_if_available(device_id: usize) -> Device {
        // Try to initialize Metal backend to check availability
        // This is more reliable than scirs2's platform detection
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            use metal::Device as MetalDevice;
            if MetalDevice::system_default().is_some() {
                Device::Metal(device_id)
            } else {
                Device::CPU
            }
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        Device::CPU
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    pub fn metal_if_available(_device_id: usize) -> Device {
        Device::CPU
    }

    /// Create a CUDA device, or CPU if CUDA is not available.
    ///
    /// Availability is determined by a genuine runtime probe
    /// (`crate::gpu_ops::cuda::oxicuda_cuda_available()`), which `dlopen`s the CUDA
    /// driver and queries the device count. This is deliberately *not* based on
    /// `scirs2_core::simd_ops::PlatformCapabilities`: as of scirs2-core 0.6.0 its
    /// `cuda_available` flag is hardcoded to `false` (CUDA support was retired
    /// upstream), so relying on it here would make this constructor always return
    /// `Device::CPU` even on machines with a working CUDA GPU.
    #[cfg(feature = "cuda")]
    pub fn cuda_if_available(device_id: usize) -> Device {
        if crate::gpu_ops::cuda::oxicuda_cuda_available() {
            Device::CUDA(device_id)
        } else {
            Device::CPU
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn cuda_if_available(_device_id: usize) -> Device {
        Device::CPU
    }

    /// Get the best available device, preferring CUDA, then Metal, then CPU.
    ///
    /// Like [`Device::cuda_if_available`] and [`Device::metal_if_available`], this
    /// probes trustformers' own GPU backends directly rather than going through
    /// `scirs2_core::simd_ops::PlatformCapabilities`, whose `cuda_available` is
    /// hardcoded `false` upstream and whose `metal_available` requires a scirs2
    /// `metal` feature that trustformers never enables. Using it here would silently
    /// always fall back to `Device::CPU`, even on CUDA/Metal-capable machines.
    pub fn best_available() -> Device {
        #[cfg(feature = "cuda")]
        if crate::gpu_ops::cuda::oxicuda_cuda_available() {
            return Device::CUDA(0);
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            use metal::Device as MetalDevice;
            if MetalDevice::system_default().is_some() {
                return Device::Metal(0);
            }
        }

        Device::CPU
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CPU => write!(f, "CPU"),
            Device::CUDA(id) => write!(f, "CUDA:{}", id),
            Device::Metal(id) => write!(f, "Metal:{}", id),
            Device::ROCm(id) => write!(f, "ROCm:{}", id),
            Device::WebGPU => write!(f, "WebGPU"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_checks() {
        assert!(Device::CPU.is_cpu());
        assert!(!Device::CPU.is_gpu());
        assert!(!Device::CPU.is_metal());

        assert!(Device::Metal(0).is_metal());
        assert!(Device::Metal(0).is_gpu());
        assert!(!Device::Metal(0).is_cpu());

        assert!(Device::CUDA(0).is_cuda());
        assert!(Device::CUDA(0).is_gpu());
    }

    #[test]
    fn test_device_id() {
        assert_eq!(Device::CPU.device_id(), None);
        assert_eq!(Device::Metal(0).device_id(), Some(0));
        assert_eq!(Device::CUDA(1).device_id(), Some(1));
    }

    #[test]
    fn test_device_display() {
        assert_eq!(Device::CPU.to_string(), "CPU");
        assert_eq!(Device::Metal(0).to_string(), "Metal:0");
        assert_eq!(Device::CUDA(1).to_string(), "CUDA:1");
    }

    /// `cuda_if_available` must never panic, regardless of whether the host
    /// actually has a CUDA-capable GPU. On a machine without CUDA it must fall
    /// back to `Device::CPU`; this test intentionally does not assert which
    /// branch is taken since that depends on the machine running the test.
    #[test]
    fn test_cuda_if_available_does_not_panic() {
        let device = Device::cuda_if_available(0);
        assert!(matches!(device, Device::CPU | Device::CUDA(0)));
    }

    /// `metal_if_available` must never panic, regardless of whether the host
    /// actually has a Metal-capable GPU.
    #[test]
    fn test_metal_if_available_does_not_panic() {
        let device = Device::metal_if_available(0);
        assert!(matches!(device, Device::CPU | Device::Metal(0)));
    }

    /// `best_available` must never panic and must always return a valid
    /// `Device`, without asserting a specific GPU is present (CI/dev machines
    /// running this test are not guaranteed to have CUDA or Metal hardware).
    #[test]
    fn test_best_available_does_not_panic() {
        let device = Device::best_available();
        assert!(matches!(
            device,
            Device::CPU | Device::CUDA(0) | Device::Metal(0)
        ));
    }
}
