//! GPT-2 model implementation
//!
//! Split into submodules.

mod model_blocks;
mod model_core;

pub use model_core::*;
// model_blocks items are pub(crate) and accessed via direct imports
