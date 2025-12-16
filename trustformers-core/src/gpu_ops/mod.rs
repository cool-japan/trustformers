//! GPU operations for hardware acceleration
//!
//! This module provides GPU-accelerated operations for tensors using various
//! GPU backends.
//!
//! Supported backends:
//! - **Metal**: Apple Silicon (macOS) - M1/M2/M3 chips
//! - **CUDA**: NVIDIA GPUs (Linux/Windows)
//! - **WebGPU**: Cross-platform (browsers, desktop, mobile)
//! - **ROCm**: AMD GPUs (Linux)
//! - **OpenCL**: Universal GPU support (Intel, AMD, NVIDIA)
//!
//! Each backend provides:
//! - Matrix multiplication (matmul)
//! - GELU activation
//! - LayerNorm
//! - RoPE (Rotary Position Embedding)
//! - Softmax with causal masking
//! - Persistent buffer caching for zero-copy operations

// Advanced CUDA kernels - temporarily disabled, needs cudarc 0.17.7 API migration
// #[cfg(feature = "cuda")]
// pub mod advanced_kernels;

// GPU backend modules
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;
#[cfg(feature = "opencl")]
pub mod opencl;
#[cfg(feature = "rocm")]
pub mod rocm;
#[cfg(feature = "wgpu_backend")]
pub mod webgpu;

// Default dispatch (Metal on macOS, others on respective platforms)
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use metal::dispatch_matmul;

// Export persistent buffer types and functions
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use metal::BufferId;

#[cfg(feature = "cuda")]
pub use cuda::{dispatch_cuda_matmul, BufferId as CudaBufferId};

#[cfg(feature = "wgpu_backend")]
pub use webgpu::{dispatch_webgpu_matmul, BufferId as WebGpuBufferId};

#[cfg(feature = "rocm")]
pub use rocm::{dispatch_rocm_matmul, BufferId as RocmBufferId};

#[cfg(feature = "opencl")]
pub use opencl::{dispatch_opencl_matmul, BufferId as OpenClBufferId};
