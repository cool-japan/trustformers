//! Mobile memory usage reporting.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};

/// Mobile memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileMemoryInfo {
    /// Model memory usage in MB
    pub model_memory_mb: usize,
    /// Runtime memory overhead in MB
    pub runtime_memory_mb: usize,
    /// Total memory usage in MB
    pub total_memory_mb: usize,
    /// Memory limit in MB
    pub memory_limit_mb: usize,
    /// Memory savings percentage
    pub memory_savings_percent: f32,
    /// Cache memory usage in MB
    pub cache_memory_mb: usize,
}
impl MobileMemoryInfo {
    /// Check if memory usage is within limits
    pub fn is_within_limits(&self) -> bool {
        self.total_memory_mb <= self.memory_limit_mb
    }
    /// Get memory utilization percentage
    pub fn memory_utilization_percent(&self) -> f32 {
        (self.total_memory_mb as f32 / self.memory_limit_mb as f32) * 100.0
    }
    /// Get available memory in MB
    pub fn available_memory_mb(&self) -> usize {
        self.memory_limit_mb.saturating_sub(self.total_memory_mb)
    }
}
