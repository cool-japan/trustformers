//! Common imports and utilities for Metal backend modules
//!
//! This module provides shared imports and helper functions used across all Metal backend modules.

// Re-export all common types and traits needed by Metal backend modules
pub(crate) use crate::device::Device;
pub(crate) use crate::errors::{Result, TrustformersError};
pub(crate) use crate::tensor::Tensor;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use metal::{
    foreign_types::ForeignType, Buffer, CommandQueue, CompileOptions, Device as MetalDevice,
    MTLResourceOptions,
};

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use std::collections::HashMap;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use std::mem;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use std::sync::Arc;

// Import scirs2-core MPS for GPU-to-GPU optimized operations
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use scirs2_core::gpu::backends::MPSOperations;

// Re-export objc2 types commonly used in Metal backend
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use objc2::rc::Retained;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use objc2::runtime::ProtocolObject;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use objc2_metal::MTLBuffer as ObjC2Buffer;
