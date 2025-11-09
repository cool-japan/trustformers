// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hardware acceleration abstraction layer for TrustformeRS
//!
//! This module provides a unified interface for various hardware acceleration
//! platforms, including custom ASICs, neuromorphic processors, and specialized
//! AI accelerators. It enables seamless integration of new hardware backends
//! while maintaining compatibility with existing tensor operations.

pub mod allocation;
pub mod asic;
pub mod backends;
pub mod config;
pub mod devices;
pub mod manager;
pub mod monitoring;
pub mod registry;
pub mod scheduling;
pub mod traits;

pub use allocation::{LoadBalancer, MemoryManager, ResourceAllocator};
pub use asic::{AsicBackend, AsicDevice, AsicOperationSet};
pub use backends::{CPUBackend, CPUBackendConfig, GPUBackend, GPUBackendConfig};
pub use config::{AllocationStrategy, LoadBalancingStrategy};
pub use config::{DeviceInfo, HardwareManagerConfig};
pub use devices::{CPUDevice, GPUBackendType, GPUDevice};
pub use manager::HardwareManager;
pub use monitoring::{
    AnomalyDetector, AnomalySeverity, AnomalyType, HealthChecker, HealthStatus, PerformanceMonitor,
};
pub use registry::HardwareRegistry;
pub use scheduling::{AdvancedScheduler, DefaultScheduler, SchedulingAlgorithm};

use crate::errors::TrustformersError;
use serde::{Deserialize, Serialize};
pub use traits::{
    HardwareBackend, HardwareDevice, HardwareOperation, HardwareScheduler, OperationParameter,
    SchedulerStatistics,
};

/// Supported hardware types in TrustformeRS
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareType {
    /// Central Processing Unit
    CPU,
    /// Graphics Processing Unit (CUDA, ROCm, Metal, etc.)
    GPU,
    /// Custom Application-Specific Integrated Circuit
    ASIC,
    /// Neuromorphic processing unit
    Neuromorphic,
    /// Quantum processing unit
    Quantum,
    /// Field-Programmable Gate Array
    FPGA,
    /// Digital Signal Processor
    DSP,
    /// Tensor Processing Unit
    TPU,
    /// Vision Processing Unit
    VPU,
    /// Custom accelerator
    Custom(String),
}

/// Hardware capability flags
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    /// Supported data types
    pub data_types: Vec<DataType>,
    /// Maximum tensor dimensions
    pub max_dimensions: usize,
    /// Memory size in bytes
    pub memory_size: Option<usize>,
    /// Clock frequency in Hz
    pub clock_frequency: Option<u64>,
    /// Compute units
    pub compute_units: Option<u32>,
    /// Supported operations
    pub operations: Vec<String>,
    /// Power consumption in watts
    pub power_consumption: Option<f64>,
    /// Thermal design power
    pub thermal_design_power: Option<f64>,
}

/// Supported data types for hardware operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub enum DataType {
    F32,
    F16,
    BF16,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    Complex64,
    Complex128,
    Custom(u8), // Custom bit width
}

/// Hardware performance metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareMetrics {
    /// Operations per second
    pub ops_per_second: f64,
    /// Memory bandwidth in bytes/second
    pub memory_bandwidth: f64,
    /// Utilization percentage (0.0 to 1.0)
    pub utilization: f64,
    /// Power consumption in watts
    pub power_consumption: f64,
    /// Temperature in Celsius
    pub temperature: Option<f64>,
    /// Error rate
    pub error_rate: f64,
    /// Latency in microseconds
    pub latency: f64,
    /// Throughput in operations per second
    pub throughput: f64,
}

/// Hardware configuration for different operation modes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Hardware type
    pub hardware_type: HardwareType,
    /// Device identifier
    pub device_id: String,
    /// Operation mode (Performance, Efficiency, Balanced)
    pub operation_mode: OperationMode,
    /// Memory pool size
    pub memory_pool_size: Option<usize>,
    /// Batch size limits
    pub batch_size_limits: Option<(usize, usize)>,
    /// Precision mode
    pub precision_mode: PrecisionMode,
    /// Custom parameters
    pub custom_params: std::collections::HashMap<String, String>,
}

/// Operation modes for hardware optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationMode {
    /// Maximum performance
    Performance,
    /// Maximum efficiency
    Efficiency,
    /// Balanced performance and efficiency
    Balanced,
    /// Low power consumption
    LowPower,
    /// High precision
    HighPrecision,
    /// Custom mode
    Custom,
}

/// Precision modes for hardware operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionMode {
    /// Single precision floating point
    Single,
    /// Half precision floating point
    Half,
    /// Brain floating point
    BFloat16,
    /// Double precision floating point
    Double,
    /// Mixed precision
    Mixed,
    /// Integer precision
    Integer(u8),
    /// Custom precision
    Custom(u8),
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self {
            data_types: vec![DataType::F32],
            max_dimensions: 8,
            memory_size: None,
            clock_frequency: None,
            compute_units: None,
            operations: vec![],
            power_consumption: None,
            thermal_design_power: None,
        }
    }
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            hardware_type: HardwareType::CPU,
            device_id: "default".to_string(),
            operation_mode: OperationMode::Balanced,
            memory_pool_size: None,
            batch_size_limits: None,
            precision_mode: PrecisionMode::Single,
            custom_params: std::collections::HashMap::new(),
        }
    }
}

impl std::fmt::Display for HardwareType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HardwareType::CPU => write!(f, "CPU"),
            HardwareType::GPU => write!(f, "GPU"),
            HardwareType::ASIC => write!(f, "ASIC"),
            HardwareType::Neuromorphic => write!(f, "Neuromorphic"),
            HardwareType::Quantum => write!(f, "Quantum"),
            HardwareType::FPGA => write!(f, "FPGA"),
            HardwareType::DSP => write!(f, "DSP"),
            HardwareType::TPU => write!(f, "TPU"),
            HardwareType::VPU => write!(f, "VPU"),
            HardwareType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::F32 => write!(f, "f32"),
            DataType::F16 => write!(f, "f16"),
            DataType::BF16 => write!(f, "bf16"),
            DataType::F64 => write!(f, "f64"),
            DataType::I8 => write!(f, "i8"),
            DataType::I16 => write!(f, "i16"),
            DataType::I32 => write!(f, "i32"),
            DataType::I64 => write!(f, "i64"),
            DataType::U8 => write!(f, "u8"),
            DataType::U16 => write!(f, "u16"),
            DataType::U32 => write!(f, "u32"),
            DataType::U64 => write!(f, "u64"),
            DataType::Bool => write!(f, "bool"),
            DataType::Complex64 => write!(f, "complex64"),
            DataType::Complex128 => write!(f, "complex128"),
            DataType::Custom(bits) => write!(f, "custom({})", bits),
        }
    }
}

/// Hardware abstraction result type
pub type HardwareResult<T> = Result<T, TrustformersError>;

#[cfg(test)]
mod tests {
    use super::asic::*;
    use super::traits::DeviceStatus as TraitsDeviceStatus;
    use super::*;

    use std::collections::HashMap;

    #[test]
    fn test_hardware_type_display() {
        assert_eq!(HardwareType::CPU.to_string(), "CPU");
        assert_eq!(HardwareType::ASIC.to_string(), "ASIC");
        assert_eq!(
            HardwareType::Custom("TPU".to_string()).to_string(),
            "Custom(TPU)"
        );
    }

    #[test]
    fn test_data_type_display() {
        assert_eq!(DataType::F32.to_string(), "f32");
        assert_eq!(DataType::BF16.to_string(), "bf16");
        assert_eq!(DataType::Custom(8).to_string(), "custom(8)");
    }

    #[test]
    fn test_hardware_capabilities_default() {
        let caps = HardwareCapabilities::default();
        assert_eq!(caps.data_types, vec![DataType::F32]);
        assert_eq!(caps.max_dimensions, 8);
        assert!(caps.memory_size.is_none());
    }

    #[test]
    fn test_hardware_config_default() {
        let config = HardwareConfig::default();
        assert_eq!(config.hardware_type, HardwareType::CPU);
        assert_eq!(config.device_id, "default");
        assert_eq!(config.operation_mode, OperationMode::Balanced);
        assert_eq!(config.precision_mode, PrecisionMode::Single);
    }

    #[test]
    fn test_hardware_types_equality() {
        assert_eq!(HardwareType::CPU, HardwareType::CPU);
        assert_ne!(HardwareType::CPU, HardwareType::GPU);
        assert_eq!(
            HardwareType::Custom("TPU".to_string()),
            HardwareType::Custom("TPU".to_string())
        );
    }

    #[test]
    fn test_asic_type_varieties() {
        let asic_types = [
            AsicType::AIInference,
            AsicType::NPU,
            AsicType::TPU,
            AsicType::DSP,
            AsicType::VPU,
            AsicType::Crypto,
            AsicType::EdgeAI,
            AsicType::Custom("CustomAccelerator".to_string()),
        ];

        assert_eq!(asic_types.len(), 8);
        assert_eq!(asic_types[0], AsicType::AIInference);
        assert_eq!(
            asic_types[7],
            AsicType::Custom("CustomAccelerator".to_string())
        );
    }

    #[test]
    fn test_asic_vendor_creation() {
        let vendor = AsicVendor {
            name: "TrustformeRS Chips".to_string(),
            id: 0x1234,
            driver_version: "2.1.0".to_string(),
            firmware_version: "1.5.2".to_string(),
            support_contact: Some("support@trustformers.ai".to_string()),
        };

        assert_eq!(vendor.name, "TrustformeRS Chips");
        assert_eq!(vendor.id, 0x1234);
        assert!(vendor.support_contact.is_some());
    }

    #[test]
    fn test_device_status_and_memory_usage() {
        use super::traits::MemoryUsage;

        let memory_usage = MemoryUsage {
            total: 8192,
            used: 4096,
            free: 4096,
            fragmentation: 0.1,
        };

        let status = TraitsDeviceStatus {
            online: true,
            busy: false,
            error: None,
            memory_usage,
            temperature: Some(70.5),
            power_consumption: Some(150.0),
            utilization: 0.8,
        };

        assert!(status.online);
        assert!(!status.busy);
        assert!(status.error.is_none());
        assert_eq!(status.memory_usage.total, 8192);
        assert_eq!(status.memory_usage.used, 4096);
        assert_eq!(status.memory_usage.free, 4096);
        assert_eq!(status.temperature, Some(70.5));
        assert_eq!(status.utilization, 0.8);
    }

    #[test]
    fn test_operation_parameters() {
        use super::traits::OperationParameter;

        let mut params = HashMap::new();
        params.insert(
            "learning_rate".to_string(),
            OperationParameter::Float(0.001),
        );
        params.insert("batch_size".to_string(), OperationParameter::Integer(32));
        params.insert(
            "model_name".to_string(),
            OperationParameter::String("bert-base".to_string()),
        );
        params.insert("use_fp16".to_string(), OperationParameter::Boolean(true));

        let array_param = OperationParameter::Array(vec![
            OperationParameter::Integer(1),
            OperationParameter::Integer(2),
            OperationParameter::Integer(3),
        ]);
        params.insert("dimensions".to_string(), array_param);

        assert_eq!(params.len(), 5);

        match params.get("learning_rate").unwrap() {
            OperationParameter::Float(val) => assert_eq!(*val, 0.001),
            _ => assert!(
                false,
                "Expected Float parameter but got {:?}",
                params.get("learning_rate")
            ),
        }

        match params.get("batch_size").unwrap() {
            OperationParameter::Integer(val) => assert_eq!(*val, 32),
            _ => assert!(
                false,
                "Expected Integer parameter but got {:?}",
                params.get("batch_size")
            ),
        }
    }

    #[test]
    fn test_memory_types() {
        use super::traits::{DeviceMemory, MemoryType};

        let memory_types = [
            MemoryType::Local,
            MemoryType::Host,
            MemoryType::Shared,
            MemoryType::Unified,
            MemoryType::Persistent,
            MemoryType::Cache,
        ];

        assert_eq!(memory_types.len(), 6);
        assert_eq!(memory_types[0], MemoryType::Local);
        assert_ne!(memory_types[0], MemoryType::Host);

        let device_memory = DeviceMemory {
            address: 0x10000000,
            size: 1024 * 1024, // 1MB
            memory_type: MemoryType::Local,
            device_id: "gpu_0".to_string(),
        };

        assert_eq!(device_memory.address, 0x10000000);
        assert_eq!(device_memory.size, 1024 * 1024);
        assert_eq!(device_memory.memory_type, MemoryType::Local);
        assert_eq!(device_memory.device_id, "gpu_0");
    }

    #[test]
    fn test_hardware_metrics() {
        let metrics = HardwareMetrics {
            ops_per_second: 1000.0,
            memory_bandwidth: 500.0,
            utilization: 0.5,
            power_consumption: 100.0,
            temperature: Some(65.0),
            error_rate: 0.001,
            latency: 10.0,
            throughput: 1000.0,
        };

        assert_eq!(metrics.ops_per_second, 1000.0);
        assert_eq!(metrics.utilization, 0.5);
        assert_eq!(metrics.temperature, Some(65.0));
        assert!(metrics.error_rate < 0.01);
    }

    #[test]
    fn test_precision_modes() {
        let precision_modes = [
            PrecisionMode::Single,
            PrecisionMode::Half,
            PrecisionMode::BFloat16,
            PrecisionMode::Double,
            PrecisionMode::Mixed,
            PrecisionMode::Integer(8),
            PrecisionMode::Custom(12),
        ];

        assert_eq!(precision_modes.len(), 7);
        assert_eq!(precision_modes[0], PrecisionMode::Single);
        assert_eq!(precision_modes[5], PrecisionMode::Integer(8));
        assert_eq!(precision_modes[6], PrecisionMode::Custom(12));
    }

    #[test]
    fn test_operation_modes() {
        let operation_modes = [
            OperationMode::Performance,
            OperationMode::Efficiency,
            OperationMode::Balanced,
            OperationMode::LowPower,
            OperationMode::HighPrecision,
            OperationMode::Custom,
        ];

        assert_eq!(operation_modes.len(), 6);
        assert_eq!(operation_modes[0], OperationMode::Performance);
        assert_eq!(operation_modes[2], OperationMode::Balanced);
        assert_eq!(operation_modes[5], OperationMode::Custom);
    }

    #[test]
    fn test_hardware_serialization() {
        // Test HardwareType serialization
        let hardware_type = HardwareType::Custom("TestAccelerator".to_string());
        let serialized = serde_json::to_string(&hardware_type).unwrap();
        let deserialized: HardwareType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(hardware_type, deserialized);

        // Test DataType serialization
        let data_type = DataType::Custom(12);
        let serialized = serde_json::to_string(&data_type).unwrap();
        let deserialized: DataType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(data_type, deserialized);

        // Test OperationMode serialization
        let operation_mode = OperationMode::Performance;
        let serialized = serde_json::to_string(&operation_mode).unwrap();
        let deserialized: OperationMode = serde_json::from_str(&serialized).unwrap();
        assert_eq!(operation_mode, deserialized);
    }

    #[test]
    fn test_hardware_capabilities_custom() {
        let mut caps = HardwareCapabilities::default();
        caps.data_types = vec![DataType::F32, DataType::F16, DataType::I8];
        caps.max_dimensions = 16;
        caps.memory_size = Some(8 * 1024 * 1024 * 1024); // 8GB
        caps.clock_frequency = Some(2_500_000_000); // 2.5 GHz
        caps.compute_units = Some(64);
        caps.operations = vec![
            "matmul".to_string(),
            "conv2d".to_string(),
            "attention".to_string(),
        ];
        caps.power_consumption = Some(250.0);
        caps.thermal_design_power = Some(300.0);

        assert_eq!(caps.data_types.len(), 3);
        assert_eq!(caps.max_dimensions, 16);
        assert_eq!(caps.memory_size, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(caps.operations.len(), 3);
        assert!(caps.operations.contains(&"matmul".to_string()));
    }

    #[test]
    fn test_hardware_config_custom() {
        let mut custom_params = HashMap::new();
        custom_params.insert("vendor".to_string(), "TrustformeRS".to_string());
        custom_params.insert("model".to_string(), "TF-1000".to_string());
        custom_params.insert("revision".to_string(), "A1".to_string());

        let config = HardwareConfig {
            hardware_type: HardwareType::ASIC,
            device_id: "asic_0".to_string(),
            operation_mode: OperationMode::Performance,
            memory_pool_size: Some(1024 * 1024 * 1024), // 1GB
            batch_size_limits: Some((1, 256)),
            precision_mode: PrecisionMode::Mixed,
            custom_params,
        };

        assert_eq!(config.hardware_type, HardwareType::ASIC);
        assert_eq!(config.device_id, "asic_0");
        assert_eq!(config.operation_mode, OperationMode::Performance);
        assert_eq!(config.memory_pool_size, Some(1024 * 1024 * 1024));
        assert_eq!(config.batch_size_limits, Some((1, 256)));
        assert_eq!(config.precision_mode, PrecisionMode::Mixed);
        assert_eq!(config.custom_params.len(), 3);
        assert_eq!(
            config.custom_params.get("vendor"),
            Some(&"TrustformeRS".to_string())
        );
    }
}
