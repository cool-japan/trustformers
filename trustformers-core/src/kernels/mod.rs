// Custom optimized kernels for performance-critical operations
// Note: Glob re-exports may have name collisions across different backend implementations.
// This is intentional as each backend provides its own implementation of common types.
#![allow(ambiguous_glob_reexports)]

pub mod fused_ops;
pub mod rope;
pub mod simd;
pub mod simd_ops;

#[cfg(feature = "cuda")]
pub mod cuda_kernels;

#[cfg(feature = "cuda")]
pub mod cuda_impl;

#[cfg(feature = "rocm")]
pub mod rocm_kernels;

#[cfg(feature = "rocm")]
pub mod rocm_impl;

#[cfg(feature = "intel")]
pub mod intel_kernels;

#[cfg(feature = "intel")]
pub mod intel_impl;

#[cfg(feature = "vulkan")]
pub mod vulkan_kernels;

#[cfg(feature = "vulkan")]
pub mod vulkan_impl;

#[cfg(feature = "xla")]
pub mod xla_impl;

#[cfg(feature = "tpu")]
pub mod tpu_impl;

#[cfg(feature = "oneapi")]
pub mod oneapi_impl;

#[cfg(feature = "riscv")]
pub mod riscv_impl;

#[cfg(feature = "metal")]
pub mod metal_impl;

pub use fused_ops::*;
pub use rope::*;
pub use simd::CpuFeatures as SIMDCpuFeatures;
pub use simd::{SIMDLayerNorm, SIMDMatrixOps, SIMDSoftmax};
pub use simd_ops::CpuFeatures;

#[cfg(feature = "cuda")]
pub use cuda_kernels::*;

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

#[cfg(feature = "rocm")]
pub use rocm_kernels::*;

#[cfg(feature = "rocm")]
pub use rocm_impl::*;

#[cfg(feature = "intel")]
pub use intel_kernels::*;

#[cfg(feature = "intel")]
pub use intel_impl::*;

#[cfg(feature = "vulkan")]
pub use vulkan_kernels::*;

#[cfg(feature = "vulkan")]
pub use vulkan_impl::*;

#[cfg(feature = "xla")]
pub use xla_impl::*;

#[cfg(feature = "tpu")]
pub use tpu_impl::*;

#[cfg(feature = "oneapi")]
pub use oneapi_impl::*;

#[cfg(feature = "riscv")]
pub use riscv_impl::*;

#[cfg(feature = "metal")]
pub use metal_impl::*;
