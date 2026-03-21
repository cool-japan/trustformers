//! # CpuVendor - Trait Implementations
//!
//! This module contains trait implementations for `CpuVendor`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CpuVendor;

impl Default for CpuVendor {
    fn default() -> Self {
        CpuVendor::Other("Unknown".to_string())
    }
}
