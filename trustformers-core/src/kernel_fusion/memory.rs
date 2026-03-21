//! Memory access patterns for optimization
//!
//! This module provides structures for representing and analyzing memory
//! access patterns to optimize kernel fusion decisions.

/// Memory access patterns for optimization
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAccessPattern {
    Sequential,
    Strided { strides: Vec<usize> },
    Random,
    Broadcast,
}

impl MemoryAccessPattern {
    /// Check if the access pattern is cache-friendly
    pub fn is_cache_friendly(&self) -> bool {
        match self {
            MemoryAccessPattern::Sequential => true,
            MemoryAccessPattern::Strided { strides } => {
                // Small strides are generally cache-friendly
                strides.iter().all(|&stride| stride <= 64)
            },
            MemoryAccessPattern::Random => false,
            MemoryAccessPattern::Broadcast => true, // One value used multiple times
        }
    }

    /// Estimate memory bandwidth utilization (0.0 to 1.0)
    pub fn bandwidth_utilization(&self) -> f64 {
        match self {
            MemoryAccessPattern::Sequential => 1.0,
            MemoryAccessPattern::Strided { strides } => {
                // Assume utilization inversely proportional to stride
                let max_stride = strides.iter().max().unwrap_or(&1);
                1.0 / (*max_stride as f64).max(1.0)
            },
            MemoryAccessPattern::Random => 0.1, // Very poor utilization
            MemoryAccessPattern::Broadcast => 0.8, // Good utilization due to reuse
        }
    }

    /// Check if the pattern supports vectorization
    pub fn supports_vectorization(&self) -> bool {
        match self {
            MemoryAccessPattern::Sequential => true,
            MemoryAccessPattern::Strided { strides } => {
                // Only unit stride in the innermost dimension supports vectorization
                strides.last().is_some_and(|&stride| stride == 1)
            },
            MemoryAccessPattern::Random => false,
            MemoryAccessPattern::Broadcast => false, // Different semantics
        }
    }
}
