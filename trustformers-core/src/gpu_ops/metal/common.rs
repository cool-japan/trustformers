//! Common imports and utilities for Metal backend modules
//!
//! This module provides shared imports and helper functions used across all Metal backend modules.

// Re-export all common types and traits needed by Metal backend modules
pub(crate) use crate::device::Device;
pub(crate) use crate::errors::{Result, TrustformersError};
pub(crate) use crate::tensor::Tensor;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use metal::{
    Buffer, CommandQueue, CompileOptions, Device as MetalDevice, MTLResourceOptions,
};

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use std::collections::HashMap;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use std::mem;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use std::sync::Arc;

// NOTE: the objc2 / objc2-metal re-exports that used to live here existed solely
// for `buffer_to_objc2`, a dead `#[allow(dead_code)]` unsafe helper with no
// callers. Both the helper and the two dependencies are gone; everything in this
// backend goes through metal-rs.
