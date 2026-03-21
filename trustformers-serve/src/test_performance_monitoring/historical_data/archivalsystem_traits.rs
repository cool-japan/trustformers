//! # ArchivalSystem - Trait Implementations
//!
//! This module contains trait implementations for `ArchivalSystem`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::ArchivalSystem;

impl fmt::Debug for ArchivalSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let policy_count =
            self.archival_policies.try_read().map(|policies| policies.len()).unwrap_or(0);
        f.debug_struct("ArchivalSystem")
            .field("policy_count", &policy_count)
            .field("backend_count", &self.archival_backends.len())
            .field("archival_scheduler", &self.archival_scheduler)
            .field("archival_index", &self.archival_index)
            .field("retrieval_cache", &self.retrieval_cache)
            .finish()
    }
}
