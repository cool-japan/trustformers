//! Neural Networks API (NNAPI) Integration
//!
//! This module provides comprehensive Android NNAPI support for hardware-accelerated
//! neural network inference on compatible Android devices.

pub mod bindings;
pub mod model;
pub mod execution;

// Re-export main types and functions
pub use bindings::*;
pub use model::*;
pub use execution::*;