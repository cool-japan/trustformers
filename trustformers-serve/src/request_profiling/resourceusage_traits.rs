//! # ResourceUsage - Trait Implementations
//!
//! This module contains trait implementations for `ResourceUsage`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ResourceUsage;

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            peak_cpu_percent: None,
            avg_cpu_percent: None,
            peak_memory_bytes: None,
            avg_memory_bytes: None,
            gpu_memory_bytes: None,
            gpu_utilization_percent: None,
            disk_read_bytes: None,
            disk_write_bytes: None,
            network_rx_bytes: None,
            network_tx_bytes: None,
            file_descriptors: None,
            thread_count: None,
        }
    }
}
