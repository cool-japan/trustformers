//! WebGPU backend for tensor operations
//!
//! This module provides full WebGPU support for GPU-accelerated tensor operations.

pub mod advanced_fusion_patterns;
pub mod advanced_optimizations;
pub mod async_executor;
pub mod backend;
pub mod buffer_pool;
pub mod device_selector;
pub mod kernel_fusion;
pub mod shader_manager;
pub mod shaders;
pub mod simple_ops;
pub mod tensor_ops;
pub mod types;
pub mod workgroup_tuner;

// Re-export the main types
pub use advanced_fusion_patterns::{
    AdvancedFusionOptimizer, FusionPatternConfig, FusionStats, TransformerFusionPattern,
};
pub use advanced_optimizations::{
    AdvancedGPUConfig, AdvancedGPUOptimizer, GPUPerformanceMetrics, KernelOptimization,
};
pub use async_executor::{AsyncExecutor, ExecutionStatus, OperationResult, Priority};
pub use backend::WebGPUBackend;
pub use buffer_pool::BufferPool;
pub use device_selector::{DeviceCapabilities, DeviceSelector, DeviceType};
pub use kernel_fusion::{FusableOp, FusedKernel, KernelFusion, OpNode};
pub use shader_manager::ShaderManager;
pub use tensor_ops::TensorOps;
pub use types::{
    GpuAdapterExt, GpuBufferExt, GpuCommandEncoderExt, GpuComputePassEncoderExt, GpuDeviceExt,
    GpuExt, GpuQueueExt,
};
pub use workgroup_tuner::{OperationType, WorkgroupConfig, WorkgroupTuner};

use crate::compute::ComputeError;

/// Initialize WebGPU subsystem
pub fn initialize() -> Result<(), ComputeError> {
    // Placeholder - actual initialization is done lazily when creating WebGPUBackend
    Ok(())
}

/// Check if WebGPU is supported
pub async fn is_webgpu_supported() -> bool {
    crate::compute::webgpu_simple::is_webgpu_available()
}

/// Get device capabilities
pub async fn get_device_capabilities() -> Result<DeviceCapabilities, ComputeError> {
    // Return default capabilities as placeholder
    Ok(DeviceCapabilities::default())
}
