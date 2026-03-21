//! Configuration for Dynamic Batching

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Main batching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingConfig {
    /// Maximum batch size
    pub max_batch_size: usize,

    /// Minimum batch size before timeout
    pub min_batch_size: usize,

    /// Maximum time to wait for batch formation
    pub max_wait_time: Duration,

    /// Enable adaptive batching
    pub enable_adaptive_batching: bool,

    /// Dynamic batch configuration
    pub dynamic_config: DynamicBatchConfig,

    /// Batching mode
    pub mode: BatchingMode,

    /// Optimization target
    pub optimization_target: OptimizationTarget,

    /// Memory limit for batching (bytes)
    pub memory_limit: Option<usize>,

    /// Priority-based scheduling
    pub enable_priority_scheduling: bool,

    /// Timeout policy
    pub timeout_policy: TimeoutPolicy,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            min_batch_size: 1,
            max_wait_time: Duration::from_millis(50),
            enable_adaptive_batching: true,
            dynamic_config: DynamicBatchConfig::default(),
            mode: BatchingMode::Dynamic,
            optimization_target: OptimizationTarget::Throughput,
            memory_limit: None,
            enable_priority_scheduling: true,
            timeout_policy: TimeoutPolicy::default(),
        }
    }
}

/// Dynamic batching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBatchConfig {
    /// Minimum requests before forming a batch
    pub min_requests: usize,

    /// Target batch utilization (0.0 - 1.0)
    pub target_utilization: f32,

    /// Batch size increment step
    pub size_increment: usize,

    /// Latency SLO in milliseconds
    pub latency_slo_ms: Option<u64>,

    /// Enable memory-aware batching
    pub memory_aware: bool,

    /// Padding strategy
    pub padding_strategy: PaddingStrategy,

    /// Sequence bucketing
    pub enable_bucketing: bool,

    /// Bucket boundaries for sequence lengths
    pub bucket_boundaries: Vec<usize>,
}

impl Default for DynamicBatchConfig {
    fn default() -> Self {
        Self {
            min_requests: 1,
            target_utilization: 0.8,
            size_increment: 4,
            latency_slo_ms: Some(100),
            memory_aware: true,
            padding_strategy: PaddingStrategy::Minimal,
            enable_bucketing: true,
            bucket_boundaries: vec![128, 256, 512, 1024, 2048],
        }
    }
}

/// Adaptive configuration that adjusts based on load
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Enable load-based adaptation
    pub enable_load_adaptation: bool,

    /// Low load threshold (requests per second)
    pub low_load_threshold: f32,

    /// High load threshold (requests per second)
    pub high_load_threshold: f32,

    /// Adjustment interval
    pub adjustment_interval: Duration,

    /// Maximum batch size under high load
    pub high_load_batch_size: usize,

    /// Timeout under low load
    pub low_load_timeout: Duration,

    /// Enable predictive batching
    pub enable_prediction: bool,

    /// History window for predictions
    pub prediction_window: Duration,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enable_load_adaptation: true,
            low_load_threshold: 10.0,
            high_load_threshold: 100.0,
            adjustment_interval: Duration::from_secs(30),
            high_load_batch_size: 64,
            low_load_timeout: Duration::from_millis(10),
            enable_prediction: true,
            prediction_window: Duration::from_secs(60),
        }
    }
}

/// Batching mode
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BatchingMode {
    /// Fixed batch size
    Fixed,
    /// Dynamic batch size based on queue
    Dynamic,
    /// Adaptive based on load patterns
    Adaptive,
    /// Continuous batching (for LLMs)
    Continuous,
}

/// Optimization target
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OptimizationTarget {
    /// Optimize for throughput
    Throughput,
    /// Optimize for latency
    Latency,
    /// Balance between throughput and latency
    Balanced,
    /// Optimize for cost (cloud deployments)
    Cost,
}

/// Padding strategy for variable-length inputs
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PaddingStrategy {
    /// Pad to maximum length in batch
    Maximum,
    /// Minimal padding with bucketing
    Minimal,
    /// No padding (requires attention mask)
    None,
    /// Dynamic padding based on distribution
    Dynamic,
}

/// Timeout policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutPolicy {
    /// Base timeout
    pub base_timeout: Duration,

    /// Timeout multiplier based on queue depth
    pub queue_depth_multiplier: f32,

    /// Maximum timeout
    pub max_timeout: Duration,

    /// Enable exponential backoff
    pub exponential_backoff: bool,

    /// Priority boost for waiting requests
    pub priority_boost_ms: u64,
}

impl Default for TimeoutPolicy {
    fn default() -> Self {
        Self {
            base_timeout: Duration::from_millis(50),
            queue_depth_multiplier: 0.1,
            max_timeout: Duration::from_millis(200),
            exponential_backoff: false,
            priority_boost_ms: 10,
        }
    }
}

/// Batch formation strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BatchFormationStrategy {
    /// First-come, first-served
    FCFS,
    /// Priority-based
    Priority,
    /// Similarity-based (group similar inputs)
    Similarity,
    /// Length-based bucketing
    LengthBased,
    /// Model-specific strategy
    ModelSpecific,
}

/// Request priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory per batch (bytes)
    pub max_batch_memory: usize,

    /// Reserved memory for processing (bytes)
    pub reserved_memory: usize,

    /// Enable memory profiling
    pub enable_profiling: bool,

    /// Memory allocation strategy
    pub allocation_strategy: AllocationStrategy,
}

/// Memory allocation strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AllocationStrategy {
    /// Pre-allocate maximum size
    PreAllocate,
    /// Dynamic allocation
    Dynamic,
    /// Pool-based allocation
    Pooled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BatchingConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.mode, BatchingMode::Dynamic);
        assert!(config.enable_adaptive_batching);
    }

    #[test]
    fn test_adaptive_config() {
        let config = AdaptiveConfig::default();
        assert!(config.enable_load_adaptation);
        assert_eq!(config.low_load_threshold, 10.0);
        assert_eq!(config.high_load_threshold, 100.0);
    }
}
