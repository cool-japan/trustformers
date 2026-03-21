//! # EventStore - Trait Implementations
//!
//! This module contains trait implementations for `EventStore`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::EventStore;

impl fmt::Debug for EventStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let buffer_stats = self
            .event_buffer
            .try_lock()
            .map(|buffer| (buffer.len(), buffer.capacity()))
            .unwrap_or((0, 0));
        f.debug_struct("EventStore")
            .field("storage_config", &self.storage_config)
            .field("has_persistent_storage", &self.persistent_storage.is_some())
            .field("compression_enabled", &self.compression_enabled)
            .field("buffer_len", &buffer_stats.0)
            .field("buffer_capacity", &buffer_stats.1)
            .field("event_indexer", &self.event_indexer)
            .field("retention_manager", &self.retention_manager)
            .finish()
    }
}
