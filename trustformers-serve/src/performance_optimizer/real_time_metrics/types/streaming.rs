//! Streaming Types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// Import common types

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

// Streaming Aggregation Types
// ============================================================================

/// Anomaly tracker for detecting anomalous data patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyTracker {
    /// Anomaly detection enabled
    pub enabled: bool,
    /// Anomaly threshold
    pub threshold: f64,
    /// Detected anomalies count
    pub anomalies_detected: u64,
}

impl Default for AnomalyTracker {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 3.0,
            anomalies_detected: 0,
        }
    }
}

/// Stream statistics for real-time data streams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStatistics {
    /// Total items processed
    pub items_processed: u64,
    /// Processing rate (items/sec)
    pub processing_rate: f64,
    /// Average latency
    pub avg_latency: Duration,
    /// Stream start time
    pub stream_start: DateTime<Utc>,
}

impl Default for StreamStatistics {
    fn default() -> Self {
        Self {
            items_processed: 0,
            processing_rate: 0.0,
            avg_latency: Duration::from_secs(0),
            stream_start: Utc::now(),
        }
    }
}

/// Backpressure controller for stream flow control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureController {
    /// Backpressure enabled
    pub enabled: bool,
    /// Current backpressure level (0.0-1.0)
    pub pressure_level: f64,
    /// Buffer high watermark
    pub high_watermark: usize,
    /// Buffer low watermark
    pub low_watermark: usize,
}

impl Default for BackpressureController {
    fn default() -> Self {
        Self {
            enabled: true,
            pressure_level: 0.0,
            high_watermark: 10000,
            low_watermark: 1000,
        }
    }
}

/// Flow controller for managing data flow rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowController {
    /// Target throughput (items/sec)
    pub target_throughput: f64,
    /// Current throughput (items/sec)
    pub current_throughput: f64,
    /// Throttling enabled
    pub throttling_enabled: bool,
}

impl Default for FlowController {
    fn default() -> Self {
        Self {
            target_throughput: 1000.0,
            current_throughput: 0.0,
            throttling_enabled: false,
        }
    }
}

/// Publishing statistics for result publication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishingStatistics {
    /// Total results published
    pub results_published: u64,
    /// Publishing rate (results/sec)
    pub publishing_rate: f64,
    /// Failed publications
    pub failed_publications: u64,
    /// Average publish latency
    pub avg_publish_latency: Duration,
}

impl Default for PublishingStatistics {
    fn default() -> Self {
        Self {
            results_published: 0,
            publishing_rate: 0.0,
            failed_publications: 0,
            avg_publish_latency: Duration::from_secs(0),
        }
    }
}

impl PublishingStatistics {
    /// Create new publishing statistics with default values
    pub fn new() -> Self {
        Self::default()
    }
}

/// Result formatter for formatting aggregation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultFormatter {
    /// Output format
    pub format: String,
    /// Include metadata
    pub include_metadata: bool,
    /// Precision for floating point values
    pub precision: u8,
}

impl Default for ResultFormatter {
    fn default() -> Self {
        Self {
            format: "json".to_string(),
            include_metadata: true,
            precision: 6,
        }
    }
}

impl ResultFormatter {
    /// Create new result formatter with default values
    pub fn new() -> Self {
        Self::default()
    }
}

// ============================================================================
