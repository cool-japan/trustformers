//! GPT-2 model implementation
//!
//! Split into submodules.

mod model_blocks;
mod model_core;
mod model_ops;

pub use model_core::*;
// model_blocks items are pub(crate) and accessed via direct imports
