//! GPU operations for hardware acceleration
//!
//! This module provides GPU-accelerated operations for tensors using various
//! GPU backends (Metal, CUDA, etc.).

#[cfg(feature = "cuda")]
pub mod advanced_kernels;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod metal;

pub use metal::dispatch_matmul;

// Export persistent buffer types and functions
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use metal::BufferId;

#[cfg(feature = "cuda")]
pub use cuda::dispatch_cuda_matmul;
