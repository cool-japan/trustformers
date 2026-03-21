//! # SubscriptionManager - Trait Implementations
//!
//! This module contains trait implementations for `SubscriptionManager`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::SubscriptionManager;

impl fmt::Debug for SubscriptionManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let subscription_count =
            self.active_subscriptions.try_read().map(|subs| subs.len()).unwrap_or_default();
        let group_count = self
            .subscription_groups
            .try_read()
            .map(|groups| groups.len())
            .unwrap_or_default();
        let registry_available = self.subscriber_registry.try_read().is_ok();
        let rate_limit_configured =
            self.rate_limiter.try_read().map(|rl| rl.is_some()).unwrap_or(false);
        f.debug_struct("SubscriptionManager")
            .field("subscription_count", &subscription_count)
            .field("group_count", &group_count)
            .field("registry_available", &registry_available)
            .field("rate_limit_configured", &rate_limit_configured)
            .finish()
    }
}
