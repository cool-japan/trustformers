//! # ProfilingState - Trait Implementations
//!
//! This module contains trait implementations for `ProfilingState`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ProfilingState;

impl Default for ProfilingState {
    fn default() -> Self {
        Self {
            is_active: false,
            current_session_id: None,
            start_time: None,
            total_duration: Duration::ZERO,
            events_recorded: 0,
            snapshots_taken: 0,
            last_error: None,
        }
    }
}

