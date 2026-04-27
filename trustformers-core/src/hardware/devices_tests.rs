// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for hardware device implementations.

#[cfg(test)]
mod tests {
    use super::super::devices::{CPUDevice, GPUBackendType, GPUDevice};
    use super::super::traits::{DeviceStatus, HardwareDevice, MemoryUsage};
    use super::super::{HardwareType, OperationMode, PrecisionMode};
    use crate::tensor::Tensor;

    #[test]
    fn test_cpu_device_creation() {
        let device = CPUDevice::new("cpu_test_0".to_string());
        assert_eq!(device.device_id(), "cpu_test_0");
        assert_eq!(device.hardware_type(), HardwareType::CPU);
    }

    #[test]
    fn test_cpu_device_capabilities() {
        let device = CPUDevice::new("cpu_cap".to_string());
        let caps = device.capabilities();
        assert!(!caps.data_types.is_empty());
        assert!(caps.max_dimensions >= 1);
        assert!(caps.memory_size.is_some());
    }

    #[test]
    fn test_cpu_device_operations_list() {
        let device = CPUDevice::new("cpu_ops".to_string());
        let caps = device.capabilities();
        assert!(!caps.operations.is_empty());
        assert!(caps.operations.iter().any(|op| op == "matmul"));
        assert!(caps.operations.iter().any(|op| op == "add"));
        assert!(caps.operations.iter().any(|op| op == "softmax"));
    }

    #[test]
    fn test_cpu_device_not_initialized() {
        let device = CPUDevice::new("cpu_uninit".to_string());
        assert!(!device.is_available());
    }

    #[test]
    fn test_cpu_device_status() {
        let device = CPUDevice::new("cpu_status".to_string());
        let status = device.status();
        assert!(status.online);
        assert!(!status.busy);
        assert!(status.error.is_none());
    }

    #[test]
    fn test_cpu_device_status_memory() {
        let device = CPUDevice::new("cpu_mem".to_string());
        let status = device.status();
        assert!(status.memory_usage.total > 0);
        assert_eq!(status.memory_usage.used, 0);
        assert!(status.memory_usage.free > 0);
    }

    #[test]
    fn test_cpu_device_execute_add() {
        let device = CPUDevice::new("cpu_exec_add".to_string());
        if let (Ok(a), Ok(b)) = (
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]),
            Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]),
        ) {
            let result = device.execute_operation(
                "add",
                &[a, b],
                OperationMode::Balanced,
                PrecisionMode::Single,
            );
            assert!(result.is_ok());
            if let Ok(outputs) = result {
                assert_eq!(outputs.len(), 1);
            }
        }
    }

    #[test]
    fn test_cpu_device_execute_mul() {
        let device = CPUDevice::new("cpu_exec_mul".to_string());
        if let (Ok(a), Ok(b)) = (
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]),
            Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]),
        ) {
            let result = device.execute_operation(
                "mul",
                &[a, b],
                OperationMode::Balanced,
                PrecisionMode::Single,
            );
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_cpu_device_execute_matmul() {
        let device = CPUDevice::new("cpu_exec_matmul".to_string());
        if let (Ok(a), Ok(b)) = (
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]),
            Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]),
        ) {
            let result = device.execute_operation(
                "matmul",
                &[a, b],
                OperationMode::Balanced,
                PrecisionMode::Single,
            );
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_cpu_device_execute_unsupported() {
        let device = CPUDevice::new("cpu_exec_unsup".to_string());
        if let Ok(a) = Tensor::from_vec(vec![1.0], &[1]) {
            let result = device.execute_operation(
                "nonexistent_op",
                &[a],
                OperationMode::Balanced,
                PrecisionMode::Single,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_cpu_device_add_insufficient_inputs() {
        let device = CPUDevice::new("cpu_few_inputs".to_string());
        if let Ok(a) = Tensor::from_vec(vec![1.0], &[1]) {
            let result = device.execute_operation(
                "add",
                &[a],
                OperationMode::Balanced,
                PrecisionMode::Single,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_gpu_backend_type_variants() {
        let types = [
            GPUBackendType::CUDA,
            GPUBackendType::ROCm,
            GPUBackendType::OpenCL,
            GPUBackendType::Metal,
            GPUBackendType::Vulkan,
            GPUBackendType::Unknown,
        ];
        // Ensure all variants are distinct
        for i in 0..types.len() {
            for j in (i + 1)..types.len() {
                assert_ne!(types[i], types[j]);
            }
        }
    }

    #[test]
    fn test_gpu_device_creation_cuda() {
        let device = GPUDevice::new("gpu_cuda_0".to_string(), GPUBackendType::CUDA);
        assert_eq!(device.device_id(), "gpu_cuda_0");
        assert_eq!(device.hardware_type(), HardwareType::GPU);
    }

    #[test]
    fn test_gpu_device_creation_metal() {
        let device = GPUDevice::new("gpu_metal_0".to_string(), GPUBackendType::Metal);
        assert_eq!(device.device_id(), "gpu_metal_0");
    }

    #[test]
    fn test_gpu_device_creation_rocm() {
        let device = GPUDevice::new("gpu_rocm_0".to_string(), GPUBackendType::ROCm);
        assert_eq!(device.device_id(), "gpu_rocm_0");
    }

    #[test]
    fn test_gpu_device_capabilities() {
        let device = GPUDevice::new("gpu_caps".to_string(), GPUBackendType::CUDA);
        let caps = device.capabilities();
        assert!(!caps.data_types.is_empty());
        assert!(caps.max_dimensions >= 1);
        assert!(caps.memory_size.is_some());
        assert!(caps.compute_units.is_some());
    }

    #[test]
    fn test_gpu_device_operations_list() {
        let device = GPUDevice::new("gpu_ops".to_string(), GPUBackendType::CUDA);
        let caps = device.capabilities();
        assert!(caps.operations.iter().any(|op| op == "attention"));
        assert!(caps.operations.iter().any(|op| op == "flash_attention"));
    }

    #[test]
    fn test_gpu_device_status() {
        let device = GPUDevice::new("gpu_status".to_string(), GPUBackendType::Metal);
        let status = device.status();
        assert!(status.online);
        assert!(!status.busy);
        assert!(status.error.is_none());
        assert!(status.temperature.is_some());
    }

    #[test]
    fn test_gpu_device_not_initialized() {
        let device = GPUDevice::new("gpu_uninit".to_string(), GPUBackendType::Vulkan);
        assert!(!device.is_available());
    }

    #[test]
    fn test_device_status_clone() {
        let status = DeviceStatus {
            online: true,
            busy: false,
            error: None,
            memory_usage: MemoryUsage {
                used: 100,
                total: 1000,
                free: 900,
                fragmentation: 0.05,
            },
            temperature: Some(55.0),
            power_consumption: Some(150.0),
            utilization: 25.0,
        };
        let cloned = status.clone();
        assert_eq!(cloned.online, status.online);
        assert_eq!(cloned.busy, status.busy);
        assert_eq!(cloned.memory_usage.total, status.memory_usage.total);
    }

    #[test]
    fn test_memory_usage_defaults() {
        let usage = MemoryUsage {
            used: 0,
            total: 8_000_000_000,
            free: 8_000_000_000,
            fragmentation: 0.0,
        };
        assert_eq!(usage.used, 0);
        assert_eq!(usage.total, 8_000_000_000);
    }

    #[test]
    fn test_gpu_backend_type_eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(GPUBackendType::CUDA);
        set.insert(GPUBackendType::Metal);
        set.insert(GPUBackendType::CUDA);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_gpu_device_power_consumption() {
        let device = GPUDevice::new("gpu_power".to_string(), GPUBackendType::CUDA);
        let caps = device.capabilities();
        assert!(caps.power_consumption.is_some());
        if let Some(power) = caps.power_consumption {
            assert!(power > 0.0);
        }
    }

    #[test]
    fn test_gpu_device_thermal_design_power() {
        let device = GPUDevice::new("gpu_tdp".to_string(), GPUBackendType::CUDA);
        let caps = device.capabilities();
        assert!(caps.thermal_design_power.is_some());
        if let Some(tdp) = caps.thermal_design_power {
            assert!(tdp > 0.0);
        }
    }

    #[test]
    fn test_cpu_device_clock_frequency() {
        let device = CPUDevice::new("cpu_freq".to_string());
        let caps = device.capabilities();
        assert!(caps.clock_frequency.is_some());
        if let Some(freq) = caps.clock_frequency {
            assert!(freq > 0);
        }
    }
}
