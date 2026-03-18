//! # EventEnricher - Trait Implementations
//!
//! This module contains trait implementations for `EventEnricher`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::EventEnricher;

impl fmt::Debug for EventEnricher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let provider_count = self.enrichment_providers.len();
        let cache_size =
            self.enrichment_cache.try_lock().map(|guard| guard.len()).unwrap_or_default();
        f.debug_struct("EventEnricher")
            .field("provider_count", &provider_count)
            .field("cache_size", &cache_size)
            .field("config", &self.enrichment_config)
            .finish()
    }
}
