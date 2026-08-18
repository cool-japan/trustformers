//! Attention mechanism implementations and utilities.
//!
//! This module provides various attention mechanisms used in transformer architectures,
//! organized into submodules for better maintainability and code reuse.

pub mod common;
pub mod flash;
pub(crate) mod flash_kernel;
pub mod mask;
pub mod multi_head;

pub use common::{
    join_name, AttentionConfig, AttentionOptimizationHints, AttentionProjections, AttentionUtils,
};
pub use flash::FlashAttention;
pub use mask::MaskSemantics;
pub use multi_head::MultiHeadAttention;
