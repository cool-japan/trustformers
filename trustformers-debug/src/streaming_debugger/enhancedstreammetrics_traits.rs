//! # EnhancedStreamMetrics - Trait Implementations
//!
//! This module contains trait implementations for `EnhancedStreamMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for EnhancedStreamMetrics {
    fn default() -> Self {
        Self {
            base_metrics: StreamMetrics::default(),
            adaptive_metrics: AdaptiveStreamingMetrics {
                quality_adjustments: 0,
                average_quality: 1.0,
                bandwidth_utilization: 0.0,
                quality_stability_score: 1.0,
            },
            aggregation_metrics: AggregationMetrics {
                windows_processed: 0,
                average_window_size: 0.0,
                aggregation_latency: 0.0,
                data_reduction_ratio: 0.0,
            },
            buffer_metrics: BufferMetrics {
                buffer_adjustments: 0,
                average_utilization: 0.0,
                memory_efficiency: 1.0,
                buffer_overflow_count: 0,
            },
            network_metrics: NetworkMetrics {
                average_bandwidth: 1_000_000,
                average_latency: 50,
                connection_stability: 1.0,
                quality_score_trend: 0.0,
            },
        }
    }
}
