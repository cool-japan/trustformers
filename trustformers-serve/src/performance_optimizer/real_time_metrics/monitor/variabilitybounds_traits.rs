//! # VariabilityBounds - Trait Implementations
//!
//! This module contains trait implementations for `VariabilityBounds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::VariabilityBounds;

impl Default for VariabilityBounds {
    fn default() -> Self {
        Self {
            throughput_lower: 80.0,
            throughput_upper: 120.0,
            latency_lower: 0.04,
            latency_upper: 0.06,
            cpu_lower: 0.3,
            cpu_upper: 0.7,
            memory_lower: 0.5,
            memory_upper: 0.8,
            efficiency_lower: 0.7,
            efficiency_upper: 0.95,
            network_lower: 1_000_000.0,
            network_upper: 10_000_000.0,
            io_lower: 100.0,
            io_upper: 1000.0,
            response_time_lower: 0.01,
            response_time_upper: 0.1,
            error_rate_lower: 0.0,
            error_rate_upper: 5.0,
        }
    }
}
