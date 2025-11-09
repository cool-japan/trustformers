//! # StreamingDebugConfig - Trait Implementations
//!
//! This module contains trait implementations for `StreamingDebugConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for StreamingDebugConfig {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            stream_interval_ms: 100,
            max_concurrent_streams: 10,
            stream_buffer_size: 1000,
            enable_compression: true,
            supported_formats: vec![StreamFormat::Json, StreamFormat::MessagePack],
            max_retention_seconds: 3600,
            enable_authentication: false,
            rate_limit_per_second: 100,
        }
    }
}
