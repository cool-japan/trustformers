//! WASM tensor implementation
//!
//! Split into submodules for maintainability.

mod tensor_core;
mod tensor_ops;

pub use tensor_core::*;
pub use tensor_ops::*;
