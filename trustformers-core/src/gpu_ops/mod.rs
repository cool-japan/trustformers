//! GPU operations for hardware acceleration
//!
//! This module provides GPU-accelerated operations for tensors using various
//! GPU backends (Metal, CUDA, etc.).

pub mod metal;

pub use metal::dispatch_matmul;
