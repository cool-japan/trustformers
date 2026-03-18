//! # LoadBalancingStrategy - Trait Implementations
//!
//! This module contains trait implementations for `LoadBalancingStrategy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::LoadBalancingStrategy;

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        Self::LeastLoaded
    }
}

impl std::fmt::Display for LoadBalancingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LeastLoaded => write!(f, "Least Loaded"),
            Self::RoundRobin => write!(f, "Round Robin"),
            Self::BestFit => write!(f, "Best Fit"),
            Self::Random => write!(f, "Random"),
            Self::Weighted => write!(f, "Weighted"),
            Self::PerformanceBased => write!(f, "Performance Based"),
            Self::MemoryOptimized => write!(f, "Memory Optimized"),
            Self::PowerAware => write!(f, "Power Aware"),
            Self::Hybrid(strategies) => write!(f, "Hybrid({})", strategies.len()),
            Self::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}
