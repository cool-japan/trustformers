//! Data Pipeline Enhancements for TrustformeRS Training
//!
//! This module provides advanced data pipeline capabilities including streaming datasets,
//! dynamic augmentation, curriculum learning, active learning, and multi-modal data handling.
//!
//! Split into cohesive submodules: [`pipeline`] (core orchestration), [`streaming`],
//! [`augmentation`], [`curriculum`], [`active_learning`], [`multimodal`] and [`validation`].

pub mod active_learning;
pub mod augmentation;
pub mod curriculum;
pub mod multimodal;
pub mod pipeline;
pub mod streaming;
#[cfg(test)]
mod tests;
pub mod validation;

// Re-export all types
pub use active_learning::*;
pub use augmentation::*;
pub use curriculum::*;
pub use multimodal::*;
pub use pipeline::*;
pub use streaming::*;
pub use validation::*;
