//! Tokenizer C API for TrustformeRS
//!
//! This module provides comprehensive tokenizer loading, encoding, decoding, and management
//! capabilities for the TrustformeRS C API.
//!
//! The module is organized into several sub-modules for better maintainability:
//! - `types`: Type definitions for tokenizer configuration, encoding, and training
//! - `c_api`: C Foreign Function Interface exports for tokenizer operations

// Sub-modules
pub mod c_api;
pub mod types;

// Re-export public types and functions
pub use types::*;

// C API functions are automatically exported via #[no_mangle]
pub use c_api::*;
