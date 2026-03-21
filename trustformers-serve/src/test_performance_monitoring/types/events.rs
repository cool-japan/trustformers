//! Events Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, AtomicU64},
    time::Duration,
};

// Import types from sibling modules
use super::config::{
    AggregationConfig, CorrelationConfig, EnrichmentConfig, EventConfig, EventIndexingConfig,
    EventRetentionConfig, EventStorageConfig, PatternConfig, RateLimitConfig,
};
use super::enums::{EventSeverity, MetricValue};

#[derive(Debug, Serialize, Deserialize)]
pub struct EventQuery {
    /// Filter criteria
    pub filters: std::collections::HashMap<String, String>,
    /// Time range for query
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Maximum number of events to return
    pub limit: Option<usize>,
    /// Event types to include
    pub event_types: Option<Vec<String>>,
    /// Sort order
    pub sort_order: Option<String>,
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 1000,
            storage_config: EventStorageConfig::default(),
            buffer_size: 10000,
            compression_enabled: true,
            indexing_config: EventIndexingConfig::default(),
            retention_config: EventRetentionConfig::default(),
            rate_limit_config: RateLimitConfig::default(),
            correlation_config: CorrelationConfig::default(),
            pattern_config: PatternConfig::default(),
            aggregation_config: AggregationConfig::default(),
            enrichment_config: EnrichmentConfig::default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DispatchStatistics {
    pub total_events: AtomicU64,
    pub processed_events: AtomicU64,
    pub failed_events: AtomicU64,
    pub avg_processing_time: Duration,
}

impl Default for DispatchStatistics {
    fn default() -> Self {
        Self {
            total_events: AtomicU64::new(0),
            processed_events: AtomicU64::new(0),
            failed_events: AtomicU64::new(0),
            avg_processing_time: Duration::from_secs(0),
        }
    }
}

impl DispatchStatistics {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubscriberRegistry {
    subscribers: HashMap<String, Vec<String>>,
    active_subscriptions: HashMap<String, AtomicBool>,
}

impl Default for SubscriberRegistry {
    fn default() -> Self {
        Self {
            subscribers: HashMap::new(),
            active_subscriptions: HashMap::new(),
        }
    }
}

impl SubscriberRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceEventData {
    pub event_type: String,
    pub metrics: HashMap<String, MetricValue>,
    pub severity: EventSeverity,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CorrelationEngine {
    pub engine_id: String,
    pub correlation_window: Duration,
    pub enabled: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CorrelationCache {
    pub cache_size: usize,
    pub ttl: Duration,
    pub hit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPattern {
    pub pattern_type: String,
    pub confidence: f64,
    pub detected_at: DateTime<Utc>,
}

impl Default for NetworkPattern {
    fn default() -> Self {
        Self {
            pattern_type: String::new(),
            confidence: 0.0,
            detected_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailurePatternAnalysis {
    pub pattern_detected: bool,
    pub pattern_type: String,
    pub frequency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    pub period: usize,
    pub amplitude: f64,
    pub detected: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GrowthPattern {
    pub growth_rate: f64,
    pub trend: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IoPattern {
    pub read_ops: u64,
    pub write_ops: u64,
    pub pattern_type: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiskUsagePattern {
    pub reads_per_sec: f64,
    pub writes_per_sec: f64,
    pub usage_trend: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CorrelationAnalysis {
    pub correlation_matrix: Vec<Vec<f64>>,
    pub variables: Vec<String>,
    pub method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CyclicalPattern {
    // TODO: Re-enable once utils::duration_serde is implemented
    // #[serde(with = "crate::utils::duration_serde")]
    pub period: Duration,
    pub amplitude: f64,
    pub phase: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TemporalAnomalyPattern {
    pub pattern_type: String,
    pub start_time: DateTime<Utc>,
    pub duration: Duration,
    pub severity: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueuedEvent {
    pub event_id: String,
    pub priority: i32,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DispatchWorker {
    pub worker_id: String,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CorrelationContext {
    pub context_id: String,
    pub data: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CorrelationStatistics {
    pub correlation_count: u64,
    pub avg_correlation_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationLogic {
    pub logic_type: String,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAction {
    pub action_type: String,
    pub target: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PatternMatchState {
    pub matched: bool,
    pub partial_matches: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMatcher {
    pub matcher_id: String,
    pub pattern: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AggregatedEvent {
    pub event_id: String,
    pub event_type: String,
    pub count: usize,
    pub aggregation_time: DateTime<Utc>,
    pub data: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EnrichmentData {
    pub source: String,
    pub data: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EnrichmentCost {
    pub cpu_time: Duration,
    pub memory_bytes: usize,
    pub io_operations: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EventIndex {
    pub index_id: String,
    pub indexed_fields: Vec<String>,
    pub index_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LifecycleEventManager {
    pub manager_id: String,
    pub event_handlers: Vec<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPatternRequirements {
    pub expected_iops: f64,
    pub access_frequency: String,
    // TODO: Re-enable once utils::duration_serde is implemented
    // #[serde(with = "crate::utils::duration_serde")]
    pub latency_requirement: Duration,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CorrelationNetwork {
    pub nodes: Vec<String>,
    pub edges: Vec<(String, String, f64)>,
    pub correlation_threshold: f64,
    pub network_density: f64,
}
