//! # MetalBackend - clear_buffer_cache_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::common::*;

use super::metalbackend_type::MetalBackend;

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Clear all persistent buffers
    pub fn clear_buffer_cache(&self) -> Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "clear_buffer_cache")
        })?;
        cache.clear();
        Ok(())
    }
}
