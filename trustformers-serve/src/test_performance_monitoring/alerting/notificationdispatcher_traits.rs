//! # NotificationDispatcher - Trait Implementations
//!
//! This module contains trait implementations for `NotificationDispatcher`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::NotificationDispatcher;

impl std::fmt::Debug for NotificationDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NotificationDispatcher")
            .field(
                "notification_channels",
                &format!("<{} channels>", self.notification_channels.len()),
            )
            .field("sent_per_channel", &self.sent_per_channel)
            .finish()
    }
}

impl Default for NotificationDispatcher {
    fn default() -> Self {
        Self::new()
    }
}
