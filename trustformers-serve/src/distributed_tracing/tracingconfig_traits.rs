//! # TracingConfig - Trait Implementations
//!
//! This module contains trait implementations for `TracingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

use super::types::{BatchConfig, SamplingStrategy, TracingBackend, TracingConfig};

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "trustformers-serve".to_string(),
            service_version: "0.1.0".to_string(),
            backend: TracingBackend::Console,
            sampling: SamplingStrategy::Always,
            max_span_queue_size: 10000,
            export_timeout: Duration::from_secs(30),
            batch_config: BatchConfig::default(),
            resource_attributes: HashMap::new(),
            auto_http_instrumentation: true,
            auto_db_instrumentation: true,
        }
    }
}
