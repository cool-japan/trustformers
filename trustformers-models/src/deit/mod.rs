//! DeiT — Data-efficient Image Transformers.
//!
//! This module provides:
//! - [`DeiTConfig`]: Configuration (preset variants: tiny, small, base, base-384).
//! - [`DeiTModel`]: Base model returning full hidden states.
//! - [`DeiTForImageClassification`]: Classification head with optional distillation.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use trustformers_models::deit::{DeiTConfig, DeiTForImageClassification};
//! use scirs2_core::ndarray::Array4;
//!
//! let config = DeiTConfig::deit_tiny_patch16_224();
//! let model = DeiTForImageClassification::new(config).expect("valid config constructs model");
//! let images = Array4::<f32>::zeros((1, 224, 224, 3));
//! let logits = model.forward(&images).expect("well-formed input tensor succeeds");
//! ```

pub mod config;
pub mod model;
pub mod tasks;

#[cfg(test)]
mod tests;

pub use config::DeiTConfig;
pub use model::{DeiTAttention, DeiTEmbeddings, DeiTEncoder, DeiTLayer, DeiTMLP, DeiTModel, DeiTPatchEmbedding};
pub use tasks::DeiTForImageClassification;
