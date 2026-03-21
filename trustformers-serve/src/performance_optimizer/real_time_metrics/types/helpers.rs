//! Helper Types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

// Import common types
use super::common::AtomicF32;

// Import types from sibling modules
use super::metrics::PerformanceBaseline;

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

// ADDITIONAL HELPER TYPES AND CONFIGURATION
// =============================================================================

/// Sample rate configuration
///
/// Configuration for adaptive sample rate control including bounds,
/// adjustment policies, and performance targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleRateConfig {
    /// Minimum sample rate
    pub min_rate: f32,

    /// Maximum sample rate
    pub max_rate: f32,

    /// Target accuracy
    pub target_accuracy: f32,

    /// Adjustment sensitivity
    pub adjustment_sensitivity: f32,

    /// Rate adjustment interval
    pub adjustment_interval: Duration,

    /// Stability threshold
    pub stability_threshold: f32,

    /// Adaptive mode enabled
    pub adaptive_mode: bool,
}

impl Default for SampleRateConfig {
    fn default() -> Self {
        Self {
            min_rate: 0.1,
            max_rate: 100.0,
            target_accuracy: 0.95,
            adjustment_sensitivity: 0.1,
            adjustment_interval: Duration::from_secs(30),
            stability_threshold: 0.05,
            adaptive_mode: true,
        }
    }
}

/// Rate adjustment record
///
/// Record of sample rate adjustments including reasons, effectiveness,
/// and performance impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateAdjustment {
    /// Adjustment timestamp
    pub timestamp: DateTime<Utc>,

    /// Previous rate
    pub previous_rate: f32,

    /// New rate
    pub new_rate: f32,

    /// Adjustment reason
    pub reason: String,

    /// Effectiveness score
    pub effectiveness: Option<f32>,

    /// Performance before adjustment
    pub performance_before: Option<PerformanceMeasurement>,

    /// Performance after adjustment
    pub performance_after: Option<PerformanceMeasurement>,
}

impl RateAdjustment {
    /// Create new rate adjustment
    pub fn new(previous_rate: f32, new_rate: f32, reason: String) -> Self {
        Self {
            timestamp: Utc::now(),
            previous_rate,
            new_rate,
            reason,
            effectiveness: None,
            performance_before: None,
            performance_after: None,
        }
    }

    /// Calculate adjustment magnitude
    pub fn adjustment_magnitude(&self) -> f32 {
        (self.new_rate - self.previous_rate).abs()
    }

    /// Get adjustment direction
    pub fn adjustment_direction(&self) -> &'static str {
        if self.new_rate > self.previous_rate {
            "increase"
        } else if self.new_rate < self.previous_rate {
            "decrease"
        } else {
            "no_change"
        }
    }
}

/// Rate controller statistics
///
/// Performance statistics for the sample rate controller including
/// adjustment frequency and effectiveness metrics.
#[derive(Debug, Default)]
pub struct RateControllerStats {
    /// Total adjustments made
    pub total_adjustments: AtomicU64,

    /// Average adjustment effectiveness
    pub avg_effectiveness: AtomicF32,

    /// Current rate stability
    pub rate_stability: AtomicF32,

    /// Controller accuracy
    pub accuracy: AtomicF32,

    /// Last adjustment timestamp
    pub last_adjustment: parking_lot::Mutex<Option<DateTime<Utc>>>,
}

impl RateControllerStats {
    /// Update effectiveness
    pub fn update_effectiveness(&self, effectiveness: f32) {
        let current_avg = self.avg_effectiveness.load(Ordering::Acquire);
        let new_avg = if current_avg == 0.0 {
            effectiveness
        } else {
            (current_avg * 0.9) + (effectiveness * 0.1)
        };
        self.avg_effectiveness.store(new_avg, Ordering::Release);
    }

    /// Get adjustment frequency (adjustments per hour)
    pub fn adjustment_frequency(&self) -> f64 {
        if let Some(last_adjustment) = *self.last_adjustment.lock() {
            let hours_since = (Utc::now() - last_adjustment).num_hours() as f64;
            if hours_since > 0.0 {
                self.total_adjustments.load(Ordering::Acquire) as f64 / hours_since
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Record new adjustment
    pub fn record_adjustment(&self) {
        self.total_adjustments.fetch_add(1, Ordering::AcqRel);
        *self.last_adjustment.lock() = Some(Utc::now());
    }
}

/// Overhead measurement for impact monitoring
///
/// Measurement of performance overhead caused by metrics collection
/// and monitoring operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadMeasurement {
    /// Measurement timestamp
    pub timestamp: DateTime<Utc>,

    /// CPU overhead percentage
    pub cpu_overhead: f32,

    /// Memory overhead (bytes)
    pub memory_overhead: u64,

    /// I/O overhead
    pub io_overhead: f32,

    /// Network overhead
    pub network_overhead: f32,

    /// Collection latency overhead
    pub latency_overhead: Duration,

    /// Throughput impact
    pub throughput_impact: f32,
}

impl OverheadMeasurement {
    /// Create new overhead measurement
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            cpu_overhead: 0.0,
            memory_overhead: 0,
            io_overhead: 0.0,
            network_overhead: 0.0,
            latency_overhead: Duration::from_micros(0),
            throughput_impact: 0.0,
        }
    }

    /// Calculate overall overhead score
    pub fn overall_overhead(&self) -> f32 {
        (self.cpu_overhead * 0.3
            + (self.memory_overhead as f32 / 1_000_000.0) * 0.2
            + self.io_overhead * 0.2
            + self.network_overhead * 0.1
            + (self.latency_overhead.as_micros() as f32 / 1000.0) * 0.1
            + self.throughput_impact * 0.1)
            .min(100.0)
    }

    /// Check if overhead is acceptable
    pub fn is_acceptable(&self, max_cpu: f32, max_memory: u64, max_throughput_impact: f32) -> bool {
        self.cpu_overhead <= max_cpu
            && self.memory_overhead <= max_memory
            && self.throughput_impact <= max_throughput_impact
    }
}

impl Default for OverheadMeasurement {
    fn default() -> Self {
        Self::new()
    }
}

/// Impact analysis for performance monitoring
///
/// Analysis of the impact of monitoring operations on system performance.
#[derive(Debug, Clone)]
pub struct ImpactAnalysis {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,

    /// Overhead measurements
    pub overhead_measurements: Vec<OverheadMeasurement>,

    /// Performance baseline without monitoring
    pub baseline_without_monitoring: Option<PerformanceBaseline>,

    /// Performance baseline with monitoring
    pub baseline_with_monitoring: Option<PerformanceBaseline>,

    /// Impact severity
    pub impact_severity: SeverityLevel,

    /// Recommendations for impact reduction
    pub recommendations: Vec<String>,
}

impl ImpactAnalysis {
    /// Create new impact analysis
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            overhead_measurements: Vec::new(),
            baseline_without_monitoring: None,
            baseline_with_monitoring: None,
            impact_severity: SeverityLevel::Info,
            recommendations: Vec::new(),
        }
    }

    /// Add overhead measurement
    pub fn add_measurement(&mut self, measurement: OverheadMeasurement) {
        self.overhead_measurements.push(measurement);
        self.update_severity();
    }

    /// Update impact severity based on measurements
    fn update_severity(&mut self) {
        if self.overhead_measurements.is_empty() {
            return;
        }

        let avg_overhead: f32 =
            self.overhead_measurements.iter().map(|m| m.overall_overhead()).sum::<f32>()
                / self.overhead_measurements.len() as f32;

        self.impact_severity = match avg_overhead {
            x if x >= 20.0 => SeverityLevel::Critical,
            x if x >= 10.0 => SeverityLevel::High,
            x if x >= 5.0 => SeverityLevel::Medium,
            x if x >= 2.0 => SeverityLevel::Low,
            _ => SeverityLevel::Info,
        };
    }

    /// Get average overhead
    pub fn average_overhead(&self) -> f32 {
        if self.overhead_measurements.is_empty() {
            0.0
        } else {
            self.overhead_measurements.iter().map(|m| m.overall_overhead()).sum::<f32>()
                / self.overhead_measurements.len() as f32
        }
    }
}

impl Default for ImpactAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Impact monitor configuration
///
/// Configuration for monitoring the impact of metrics collection on system performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactMonitorConfig {
    /// Impact monitoring enabled
    pub enabled: bool,

    /// Monitoring interval
    pub monitoring_interval: Duration,

    /// Maximum acceptable CPU overhead
    pub max_cpu_overhead: f32,

    /// Maximum acceptable memory overhead
    pub max_memory_overhead: u64,

    /// Maximum acceptable throughput impact
    pub max_throughput_impact: f32,

    /// Alert on high impact
    pub alert_on_high_impact: bool,

    /// Auto-adjustment enabled
    pub auto_adjustment: bool,
}

impl Default for ImpactMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_secs(60),
            max_cpu_overhead: 5.0,
            max_memory_overhead: 100_000_000, // 100MB
            max_throughput_impact: 2.0,
            alert_on_high_impact: true,
            auto_adjustment: true,
        }
    }
}

/// Impact alert for high overhead conditions
///
/// Alert generated when monitoring overhead exceeds acceptable thresholds.
#[derive(Debug, Clone)]
pub struct ImpactAlert {
    /// Alert timestamp
    pub timestamp: DateTime<Utc>,

    /// Alert ID
    pub alert_id: String,

    /// Impact measurement
    pub measurement: OverheadMeasurement,

    /// Alert severity
    pub severity: SeverityLevel,

    /// Alert message
    pub message: String,

    /// Recommended actions
    pub actions: Vec<String>,

    /// Alert metadata
    pub metadata: HashMap<String, String>,
}

impl ImpactAlert {
    /// Create new impact alert
    pub fn new(
        alert_id: String,
        measurement: OverheadMeasurement,
        severity: SeverityLevel,
        message: String,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            alert_id,
            measurement,
            severity,
            message,
            actions: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add recommended action
    pub fn add_action(&mut self, action: String) {
        self.actions.push(action);
    }

    /// Check if alert requires immediate attention
    pub fn requires_attention(&self) -> bool {
        matches!(self.severity, SeverityLevel::High | SeverityLevel::Critical)
    }
}

// ============================================================================
