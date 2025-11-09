//! Model C API for TrustformeRS
//!
//! This module provides comprehensive model loading, inference, and management capabilities
//! for the TrustformeRS C API.
//!
//! The module is organized into several sub-modules for better maintainability:
//! - `types`: Type definitions for model configuration, tensor info, metadata, and validation
//! - `c_api`: C Foreign Function Interface exports for model operations

// Sub-modules
pub mod c_api;
pub mod types;

// Re-export public types and functions
pub use types::*;

// C API functions are automatically exported via #[no_mangle]
pub use c_api::*;
