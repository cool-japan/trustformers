//! Monitoring Types

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

// Import common types
use super::common::AtomicF32;

use tokio::task::JoinHandle;

// Import types from sibling modules
use super::enums::{MonitoringEventType, MonitoringScope};

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

// MONITORING TYPES
// =============================================================================

/// Monitor thread for parallel performance monitoring
///
/// Individual monitoring thread responsible for specific aspects of performance
/// monitoring with dedicated responsibilities and optimized data collection.
#[derive(Debug)]
pub struct MonitorThread {
    /// Thread identifier
    pub id: String,

    /// Thread handle
    pub handle: JoinHandle<()>,

    /// Monitoring scope
    pub scope: MonitoringScope,

    /// Thread statistics
    pub stats: ThreadStatistics,

    /// Last activity timestamp
    pub last_activity: Arc<RwLock<DateTime<Utc>>>,

    /// Thread health status
    pub health_status: Arc<AtomicBool>,
}

impl MonitorThread {
    /// Create new monitor thread
    pub fn new(id: String, handle: JoinHandle<()>, scope: MonitoringScope) -> Self {
        Self {
            id,
            handle,
            scope,
            stats: ThreadStatistics::default(),
            last_activity: Arc::new(RwLock::new(Utc::now())),
            health_status: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Update last activity
    pub fn update_activity(&self) {
        *self.last_activity.write() = Utc::now();
    }

    /// Check if thread is healthy
    pub fn is_healthy(&self) -> bool {
        self.health_status.load(Ordering::Acquire)
    }

    /// Get time since last activity
    pub fn time_since_activity(&self) -> Duration {
        let now = Utc::now();
        let last_activity = *self.last_activity.read();
        (now - last_activity).to_std().unwrap_or(Duration::from_secs(0))
    }
}

/// Thread performance statistics
///
/// Performance statistics for individual monitoring threads including
/// collection rates, processing times, and resource utilization.
#[derive(Debug, Default)]
pub struct ThreadStatistics {
    /// Data points collected
    pub data_points_collected: AtomicU64,

    /// Average collection time (nanoseconds)
    pub avg_collection_time: AtomicF32,

    /// Processing errors
    pub processing_errors: AtomicU64,

    /// Thread CPU usage
    pub cpu_usage: AtomicF32,

    /// Thread memory usage (bytes)
    pub memory_usage: AtomicU64,
}

impl ThreadStatistics {
    /// Get collection rate (points per second)
    pub fn collection_rate(&self) -> f64 {
        let _total_points = self.data_points_collected.load(Ordering::Acquire);
        let avg_time_nanos = self.avg_collection_time.load(Ordering::Acquire);

        if avg_time_nanos > 0.0 {
            1_000_000_000.0 / avg_time_nanos as f64
        } else {
            0.0
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        let total_points = self.data_points_collected.load(Ordering::Acquire);
        let errors = self.processing_errors.load(Ordering::Acquire);

        if total_points > 0 {
            errors as f64 / total_points as f64
        } else {
            0.0
        }
    }

    /// Update collection time
    pub fn update_collection_time(&self, time_nanos: f32) {
        let current_avg = self.avg_collection_time.load(Ordering::Acquire);
        let new_avg = if current_avg == 0.0 {
            time_nanos
        } else {
            (current_avg * 0.9) + (time_nanos * 0.1) // Exponential moving average
        };
        self.avg_collection_time.store(new_avg, Ordering::Release);
    }
}

/// Monitoring event for system communication
///
/// Event structure for communicating monitoring information, alerts,
/// and system status updates across monitoring components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Event type
    pub event_type: MonitoringEventType,

    /// Event source
    pub source: String,

    /// Event data
    pub data: HashMap<String, String>,

    /// Event severity
    pub severity: SeverityLevel,

    /// Event metadata
    pub metadata: HashMap<String, String>,
}

impl MonitoringEvent {
    /// Create new monitoring event
    pub fn new(event_type: MonitoringEventType, source: String, severity: SeverityLevel) -> Self {
        Self {
            timestamp: Utc::now(),
            event_type,
            source,
            data: HashMap::new(),
            severity,
            metadata: HashMap::new(),
        }
    }

    /// Add data to event
    pub fn add_data(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }

    /// Add metadata to event
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Check if event requires immediate attention
    pub fn requires_attention(&self) -> bool {
        matches!(self.severity, SeverityLevel::High | SeverityLevel::Critical)
    }
}

// =============================================================================
