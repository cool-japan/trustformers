//! # AlertStore - Trait Implementations
//!
//! This module contains trait implementations for `AlertStore`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AlertStore;

impl std::fmt::Debug for AlertStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlertStore")
            .field("active_alerts", &self.active_alerts)
            .field("alert_history", &self.alert_history)
            .field("alert_index", &self.alert_index)
            .field(
                "storage_backend",
                &self.storage_backend.as_ref().map(|_| "<dyn AlertPersistence>"),
            )
            .field("retention_policy", &self.retention_policy)
            .field("compression_enabled", &self.compression_enabled)
            .finish()
    }
}

impl Default for AlertStore {
    fn default() -> Self {
        Self::new()
    }
}
