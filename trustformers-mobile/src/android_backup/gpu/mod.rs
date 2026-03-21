//! GPU Acceleration Support for Android
//!
//! This module provides GPU compute acceleration using Vulkan and OpenGL ES
//! for neural network inference on Android devices.

pub mod vulkan;
pub mod opengl_es;

// Re-export main types and functions
pub use vulkan::*;
pub use opengl_es::*;