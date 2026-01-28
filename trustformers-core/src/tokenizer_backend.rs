//! Tokenizer Backend Re-exports
//!
//! This module re-exports types from the HuggingFace `tokenizers` library.
//! According to the SciRS2 Integration Policy, only `trustformers-core` can import
//! external dependencies directly. Other crates (like `trustformers-tokenizers`) must
//! use these re-exported types.
//!
//! ## SciRS2 Integration Policy Compliance
//!
//! This module exists to enforce the policy that external dependencies are only imported
//! in `trustformers-core`. The `trustformers-tokenizers` crate uses these re-exports
//! instead of directly importing from the `tokenizers` crate.

// Re-export core tokenizers types
pub use tokenizers::{Encoding, Error as TokenizerError, Tokenizer};

// Re-export commonly used tokenizer traits and types if needed in the future
// pub use tokenizers::{...};
