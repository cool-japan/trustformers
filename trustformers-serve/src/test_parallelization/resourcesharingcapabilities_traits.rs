//! # ResourceSharingCapabilities - Trait Implementations
//!
//! This module contains trait implementations for `ResourceSharingCapabilities`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ResourceSharingCapabilities;

impl Default for ResourceSharingCapabilities {
    fn default() -> Self {
        Self {
            cpu_sharing: true,
            memory_sharing: false,
            gpu_sharing: false,
            network_sharing: true,
            filesystem_sharing: true,
        }
    }
}
