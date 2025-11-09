//! Time and duration utilities.
//!
//! This module provides utilities for time calculation, formatting, and
//! relative time descriptions for conversational AI systems.

use chrono::{DateTime, Utc};

/// Time and duration utilities
pub struct TimeUtils;

impl TimeUtils {
    /// Calculate time elapsed since timestamp
    pub fn time_elapsed(timestamp: DateTime<Utc>) -> std::time::Duration {
        let now = Utc::now();
        let diff = now - timestamp;
        std::time::Duration::from_secs(diff.num_seconds().max(0) as u64)
    }

    /// Format timestamp for display
    pub fn format_timestamp(timestamp: DateTime<Utc>) -> String {
        timestamp.format("%Y-%m-%d %H:%M:%S UTC").to_string()
    }

    /// Calculate relative time description
    pub fn relative_time(timestamp: DateTime<Utc>) -> String {
        let now = Utc::now();
        let diff = now - timestamp;

        if diff.num_seconds() < 60 {
            "just now".to_string()
        } else if diff.num_minutes() < 60 {
            format!("{} minutes ago", diff.num_minutes())
        } else if diff.num_hours() < 24 {
            format!("{} hours ago", diff.num_hours())
        } else if diff.num_days() < 7 {
            format!("{} days ago", diff.num_days())
        } else if diff.num_weeks() < 4 {
            format!("{} weeks ago", diff.num_weeks())
        } else {
            format!("{} months ago", diff.num_days() / 30)
        }
    }

    /// Check if timestamp is within a duration
    pub fn is_within_duration(timestamp: DateTime<Utc>, duration: std::time::Duration) -> bool {
        let elapsed = Self::time_elapsed(timestamp);
        elapsed <= duration
    }

    /// Get conversation session duration
    pub fn session_duration(
        start_time: DateTime<Utc>,
        end_time: Option<DateTime<Utc>>,
    ) -> std::time::Duration {
        let end = end_time.unwrap_or_else(Utc::now);
        let diff = end - start_time;
        std::time::Duration::from_secs(diff.num_seconds().max(0) as u64)
    }
}
