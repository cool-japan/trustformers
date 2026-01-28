//! # MetalBackend - remove_persistent_buffer_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::common::*;

use super::metalbackend_type::MetalBackend;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::types::{BufferCache, BufferId};

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Remove a persistent buffer from cache
    pub fn remove_persistent_buffer(&self, id: &BufferId) -> Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "remove_persistent_buffer",
            )
        })?;
        cache.remove(id);
        Ok(())
    }
}
