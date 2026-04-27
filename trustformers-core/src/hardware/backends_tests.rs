// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for hardware backend implementations.

#[cfg(test)]
mod tests {
    use super::super::backends::{CPUBackend, CPUBackendConfig, GPUBackend, GPUBackendConfig};
    use super::super::devices::GPUBackendType;
    use super::super::traits::HardwareBackend;
    use super::super::{HardwareConfig, HardwareType, OperationMode, PrecisionMode};
    use crate::tensor::Tensor;

    #[test]
    fn test_cpu_backend_config_default() {
        let config = CPUBackendConfig::default();
        assert!(config.num_threads > 0);
        assert!(config.enable_simd);
        assert!(config.memory_pool_size > 0);
    }

    #[test]
    fn test_cpu_backend_config_custom() {
        let config = CPUBackendConfig {
            num_threads: 8,
            enable_simd: false,
            memory_pool_size: 1024 * 1024,
            enable_monitoring: true,
        };
        assert_eq!(config.num_threads, 8);
        assert!(!config.enable_simd);
        assert!(config.enable_monitoring);
    }

    #[test]
    fn test_gpu_backend_config_default() {
        let config = GPUBackendConfig::default();
        assert!(config.memory_pool_size > 0);
        assert!(config.stream_count > 0);
    }

    #[test]
    fn test_gpu_backend_config_custom() {
        let config = GPUBackendConfig {
            memory_pool_size: 2 * 1024 * 1024 * 1024,
            enable_unified_memory: true,
            stream_count: 4,
            enable_kernel_fusion: true,
            enable_monitoring: false,
        };
        assert!(config.enable_unified_memory);
        assert_eq!(config.stream_count, 4);
        assert!(config.enable_kernel_fusion);
        assert!(!config.enable_monitoring);
    }

    #[test]
    fn test_cpu_backend_new() {
        let backend = CPUBackend::new();
        assert_eq!(backend.name(), "CPU Backend");
        assert_eq!(backend.version(), "1.0.0");
    }

    #[test]
    fn test_cpu_backend_with_config() {
        let config = CPUBackendConfig {
            num_threads: 4,
            enable_simd: true,
            memory_pool_size: 512 * 1024,
            enable_monitoring: false,
        };
        let backend = CPUBackend::with_config(config);
        assert_eq!(backend.name(), "CPU Backend");
    }

    #[test]
    fn test_cpu_backend_is_compatible() {
        let backend = CPUBackend::new();
        assert!(backend.is_compatible(HardwareType::CPU));
        assert!(!backend.is_compatible(HardwareType::GPU));
    }

    #[test]
    fn test_cpu_backend_supported_operations() {
        let backend = CPUBackend::new();
        let ops = backend.supported_operations();
        assert!(!ops.is_empty());
        assert!(ops.iter().any(|op| op == "matmul"));
        assert!(ops.iter().any(|op| op == "add"));
        assert!(ops.iter().any(|op| op == "softmax"));
    }

    #[test]
    fn test_cpu_backend_validate_config() {
        let backend = CPUBackend::new();
        let config = HardwareConfig {
            device_id: "cpu-0".to_string(),
            hardware_type: HardwareType::CPU,
            operation_mode: OperationMode::Balanced,
            precision_mode: PrecisionMode::Single,
            memory_pool_size: None,
            batch_size_limits: None,
            custom_params: std::collections::HashMap::new(),
        };
        let result = backend.validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cpu_backend_discover_devices() {
        let backend = CPUBackend::new();
        let result = backend.discover_devices();
        assert!(result.is_ok());
        if let Ok(ids) = result {
            assert!(!ids.is_empty());
            assert!(ids[0].contains("cpu"));
        }
    }

    #[test]
    fn test_cpu_backend_device_count_initial() {
        let backend = CPUBackend::new();
        assert_eq!(backend.device_count(), 0);
    }

    #[test]
    fn test_cpu_backend_device_count_after_discover() {
        let backend = CPUBackend::new();
        if backend.discover_devices().is_ok() {
            assert!(backend.device_count() > 0);
        }
    }

    #[test]
    fn test_cpu_backend_get_device_not_found() {
        let backend = CPUBackend::new();
        let device = backend.get_device("nonexistent");
        assert!(device.is_none());
    }

    #[test]
    fn test_cpu_backend_get_device_after_discover() {
        let backend = CPUBackend::new();
        if let Ok(ids) = backend.discover_devices() {
            if let Some(first_id) = ids.first() {
                let device = backend.get_device(first_id);
                assert!(device.is_some());
            }
        }
    }

    #[test]
    fn test_cpu_backend_execute_on_device() {
        let backend = CPUBackend::new();
        if let Ok(ids) = backend.discover_devices() {
            if let Some(first_id) = ids.first() {
                if let (Ok(a), Ok(b)) = (
                    Tensor::from_vec(vec![1.0, 2.0], &[2]),
                    Tensor::from_vec(vec![3.0, 4.0], &[2]),
                ) {
                    let result = backend.execute_on_device(
                        first_id,
                        "add",
                        &[a, b],
                        OperationMode::Balanced,
                        PrecisionMode::Single,
                    );
                    assert!(result.is_ok());
                }
            }
        }
    }

    #[test]
    fn test_cpu_backend_execute_on_nonexistent_device() {
        let backend = CPUBackend::new();
        if let Ok(a) = Tensor::from_vec(vec![1.0], &[1]) {
            let result = backend.execute_on_device(
                "nonexistent",
                "add",
                &[a],
                OperationMode::Balanced,
                PrecisionMode::Single,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_gpu_backend_new_cuda() {
        let backend = GPUBackend::new(GPUBackendType::CUDA);
        assert!(backend.name().contains("GPU Backend"));
    }

    #[test]
    fn test_gpu_backend_new_metal() {
        let backend = GPUBackend::new(GPUBackendType::Metal);
        assert!(backend.name().contains("GPU Backend"));
    }

    #[test]
    fn test_gpu_backend_new_rocm() {
        let backend = GPUBackend::new(GPUBackendType::ROCm);
        assert!(backend.name().contains("GPU Backend"));
    }

    #[test]
    fn test_gpu_backend_is_compatible() {
        let backend = GPUBackend::new(GPUBackendType::CUDA);
        assert!(backend.is_compatible(HardwareType::GPU));
        assert!(!backend.is_compatible(HardwareType::CPU));
    }

    #[test]
    fn test_gpu_backend_supported_operations() {
        let backend = GPUBackend::new(GPUBackendType::CUDA);
        let ops = backend.supported_operations();
        assert!(!ops.is_empty());
        assert!(ops.iter().any(|op| op == "matmul"));
    }

    #[test]
    fn test_gpu_backend_with_config() {
        let config = GPUBackendConfig {
            memory_pool_size: 4 * 1024 * 1024 * 1024,
            enable_unified_memory: true,
            stream_count: 8,
            enable_kernel_fusion: true,
            enable_monitoring: true,
        };
        let backend = GPUBackend::with_config(GPUBackendType::Metal, config);
        assert!(backend.name().contains("GPU Backend"));
    }

    #[test]
    fn test_gpu_backend_validate_config() {
        let backend = GPUBackend::new(GPUBackendType::CUDA);
        let config = HardwareConfig {
            device_id: "gpu-0".to_string(),
            hardware_type: HardwareType::GPU,
            operation_mode: OperationMode::Balanced,
            precision_mode: PrecisionMode::Single,
            memory_pool_size: None,
            batch_size_limits: None,
            custom_params: std::collections::HashMap::new(),
        };
        let result = backend.validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_backend_config_clone() {
        let config = GPUBackendConfig {
            memory_pool_size: 1024,
            enable_unified_memory: false,
            stream_count: 2,
            enable_kernel_fusion: false,
            enable_monitoring: false,
        };
        let cloned = config.clone();
        assert_eq!(cloned.memory_pool_size, config.memory_pool_size);
        assert_eq!(cloned.stream_count, config.stream_count);
    }

    #[test]
    fn test_cpu_backend_config_clone() {
        let config = CPUBackendConfig {
            num_threads: 16,
            enable_simd: true,
            memory_pool_size: 2048,
            enable_monitoring: true,
        };
        let cloned = config.clone();
        assert_eq!(cloned.num_threads, 16);
        assert!(cloned.enable_simd);
    }
}
