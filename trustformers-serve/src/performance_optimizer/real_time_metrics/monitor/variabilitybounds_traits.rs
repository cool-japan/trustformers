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
    /// Bounds with nothing measured in them.
    ///
    /// ## Changed in 0.2.1
    ///
    /// Every bound used to be a plausible-looking constant -- throughput
    /// 80..120, latency 40..60ms, CPU 0.3..0.7, network 1..10 MB/s, error rate
    /// 0..5% -- so a baseline that had never seen a sample still handed
    /// `check_deviation` a full set of limits to judge live metrics against.
    /// Empty bounds are what an unmeasured baseline actually has.
    fn default() -> Self {
        Self {
            throughput_lower: 0.0,
            throughput_upper: 0.0,
            latency_lower: 0.0,
            latency_upper: 0.0,
            cpu_lower: 0.0,
            cpu_upper: 0.0,
            memory_lower: 0.0,
            memory_upper: 0.0,
            efficiency_lower: 0.0,
            efficiency_upper: 0.0,
            network_lower: 0.0,
            network_upper: 0.0,
            io_lower: 0.0,
            io_upper: 0.0,
            response_time_lower: 0.0,
            response_time_upper: 0.0,
            error_rate_lower: 0.0,
            error_rate_upper: 0.0,
        }
    }
}
