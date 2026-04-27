//! Swin Transformer — Hierarchical Vision Transformer with shifted windows.
//!
//! This module provides:
//! - [`SwinConfig`]: Configuration (preset variants: tiny, small, base, base-384).
//! - [`SwinModel`]: Base model returning globally pooled features `(B, final_dim)`.
//! - [`SwinForImageClassification`]: Classification head.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use trustformers_models::swin::{SwinConfig, SwinForImageClassification};
//! use scirs2_core::ndarray::Array4;
//!
//! let config = SwinConfig::swin_tiny_patch4_window7_224();
//! let model = SwinForImageClassification::new(config).expect("valid config constructs model");
//! let images = Array4::<f32>::zeros((1, 224, 224, 3));
//! let logits = model.forward(&images).expect("well-formed input tensor succeeds");
//! ```

pub mod config;
pub mod model;
pub mod tasks;

#[cfg(test)]
mod tests;

pub use config::SwinConfig;
pub use model::{
    window_partition, window_reverse, cyclic_shift,
    SwinPatchEmbedding, PatchMerging, WindowAttention,
    SwinTransformerBlock, SwinStage, SwinModel,
};
pub use tasks::SwinForImageClassification;
