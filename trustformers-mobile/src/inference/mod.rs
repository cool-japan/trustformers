//! Mobile Inference Engine
//!
//! This module provides a unified mobile inference engine that integrates
//! platform-specific optimizations, quantization, and memory management
//! for efficient transformer inference on mobile devices.
//!
//! Split into cohesive submodules: [`formats`] (model format/execution
//! planning), [`tensor_conversion`] (weight-format decoding helpers),
//! [`engine`] (the `MobileInferenceEngine`), [`cache`], [`memory_info`] and
//! [`builder`].

pub mod builder;
pub mod cache;
pub mod engine;
pub mod formats;
pub mod memory_info;
pub mod tensor_conversion;
#[cfg(test)]
mod tests;

// Re-export all types
pub use builder::*;
pub use cache::*;
pub use engine::*;
pub use formats::*;
pub use memory_info::*;
pub use tensor_conversion::*;
