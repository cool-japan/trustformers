//! GPT-2 model implementation
//!
//! Split into submodules.

mod model_blocks;
mod model_core;
mod model_ops;

pub use model_core::*;
// model_blocks items are pub(crate) and accessed via direct imports.
// Re-exported here because `model_blocks` itself is a private module.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use model_blocks::metal_attention_call_count;
