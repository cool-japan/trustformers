//! Device abstraction for hardware acceleration
//!
//! This module provides a simple Device enum for specifying where computations
//! should be executed (CPU, CUDA GPU, Metal GPU, etc.).

use serde::{Deserialize, Serialize};

/// Device specification for tensor operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    /// CPU execution
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
        use scirs2_core::simd_ops::PlatformCapabilities;

        let caps = PlatformCapabilities::detect();
        if caps.metal_available {
            Device::Metal(device_id)
        } else {
            Device::CPU
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    pub fn metal_if_available(_device_id: usize) -> Device {
        Device::CPU
    }

    /// Create a CUDA device, or CPU if CUDA is not available
    #[cfg(feature = "cuda")]
    pub fn cuda_if_available(device_id: usize) -> Device {
        use scirs2_core::simd_ops::PlatformCapabilities;

        let caps = PlatformCapabilities::detect();
        if caps.cuda_available {
            Device::CUDA(device_id)
        } else {
            Device::CPU
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn cuda_if_available(_device_id: usize) -> Device {
        Device::CPU
    }

    /// Get the best available device (prefers GPU over CPU)
    pub fn best_available() -> Device {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            use scirs2_core::simd_ops::PlatformCapabilities;
            let caps = PlatformCapabilities::detect();
            if caps.metal_available {
                return Device::Metal(0);
            }
        }

        #[cfg(feature = "cuda")]
        {
            use scirs2_core::simd_ops::PlatformCapabilities;
            let caps = PlatformCapabilities::detect();
            if caps.cuda_available {
                return Device::CUDA(0);
            }
        }

        Device::CPU
    }
}

impl Default for Device {
    fn default() -> Self {
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
}
