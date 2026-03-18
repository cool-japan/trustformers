//! # ActiveSpan - Trait Implementations
//!
//! This module contains trait implementations for `ActiveSpan`.
//!
//! ## Implemented Traits
//!
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use chrono::Utc;

use super::types::ActiveSpan;

impl Drop for ActiveSpan {
    fn drop(&mut self) {
        if let Some(span) = self.span.try_lock() {
            if span.end_time.is_none() {
                drop(span);
                if let Some(mut span) = self.span.try_lock() {
                    span.end_time = Some(Utc::now());
                    self.manager.queue_span_for_export(span.clone());
                }
            }
        } else {
            let span = self.span.clone();
            let manager = self.manager.clone();
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    let mut span_guard = span.lock();
                    if span_guard.end_time.is_none() {
                        span_guard.end_time = Some(Utc::now());
                        manager.queue_span_for_export(span_guard.clone());
                    }
                });
            } else {
                tracing::warn!(
                    "No tokio runtime available during ActiveSpan drop, span may not be properly finished"
                );
            }
        }
    }
}
