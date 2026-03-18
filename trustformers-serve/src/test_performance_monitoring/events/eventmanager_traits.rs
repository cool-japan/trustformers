//! # EventManager - Trait Implementations
//!
//! This module contains trait implementations for `EventManager`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::EventManager;

impl std::fmt::Debug for EventManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventManager")
            .field("config", &self.config)
            .field("event_dispatcher", &self.event_dispatcher)
            .field("event_store", &self.event_store)
            .field("subscription_manager", &self.subscription_manager)
            .field("event_processor", &self.event_processor)
            .field("event_statistics", &self.event_statistics)
            .field("event_filters", &self.event_filters)
            .field("event_transformers", &"<dyn trait objects>")
            .finish()
    }
}
