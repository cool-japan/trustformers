//! Attention mechanism implementations and utilities.
//!
//! This module provides various attention mechanisms used in transformer architectures,
//! organized into submodules for better maintainability and code reuse.

pub mod common;
pub mod flash;
pub mod multi_head;

pub use common::{
    AttentionConfig, AttentionOptimizationHints, AttentionProjections, AttentionUtils,
};
pub use flash::FlashAttention;
pub use multi_head::MultiHeadAttention;
