//! Resource-related types for test characterization

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Instant};

use super::super::resources::ResourceUsageSnapshot;

pub struct DirectoryUsageInfo {
    pub path: String,
    pub size_bytes: u64,
    pub file_count: usize,
}

pub struct AvailabilityLevel {
    pub level: String,
    pub percentage: f64,
    pub classification: String,
}

pub struct AvailabilityStatus {
    pub status: String,
    pub uptime_percentage: f64,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub is_available: bool,
}

pub struct ComponentHealthSummary {
    pub component_name: String,
    pub health_score: f64,
    pub status: String,
}

pub struct CoolingCurve {
    pub temperature_points: Vec<f64>,
    pub time_points: Vec<f64>,
}

pub struct FanController {
    pub fan_speed: f64,
    pub temperature_threshold: f64,
}

pub struct LeakIndicator {
    pub leak_detected: bool,
    pub leak_rate: f64,
    pub resource_type: String,
}

pub struct RateLimit {
    pub max_requests: usize,
    pub time_window: std::time::Duration,
}

pub struct RateLimiter {
    pub limits: Vec<RateLimit>,
    pub current_count: usize,
}

pub struct RateLimits {
    pub limits: HashMap<String, RateLimit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeResourceMetrics {
    /// Current resource usage
    pub current_usage: ResourceUsageSnapshot,
    /// Usage trends
    pub trends: HashMap<String, super::super::analysis::TrendAnalysis>,
    /// Anomaly indicators
    pub anomalies: Vec<super::super::analysis::AnomalyIndicator>,
    /// Performance indicators
    pub performance_indicators: HashMap<String, f64>,
    /// Capacity utilization
    pub capacity_utilization: HashMap<String, f64>,
    /// Bottleneck indicators
    pub bottlenecks: Vec<super::super::analysis::BottleneckIndicator>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<super::super::optimization::OptimizationOpportunity>,
    /// Quality assessment
    pub quality_assessment: super::super::quality::QualityAssessment,
    /// Alert conditions
    pub alert_conditions: Vec<super::super::alerts::AlertCondition>,
    /// Predictive metrics
    pub predictive_metrics: HashMap<String, f64>,
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl Default for RealTimeResourceMetrics {
    fn default() -> Self {
        Self {
            current_usage: ResourceUsageSnapshot {
                timestamp: Instant::now(),
                cpu_usage: 0.0,
                memory_usage: 0,
                available_memory: 0,
                io_read_rate: 0.0,
                io_write_rate: 0.0,
                network_in_rate: 0.0,
                network_out_rate: 0.0,
                network_rx_rate: 0.0,
                network_tx_rate: 0.0,
                gpu_utilization: 0.0,
                gpu_usage: 0.0,
                gpu_memory_usage: 0,
                disk_usage: 0.0,
                load_average: [0.0, 0.0, 0.0],
                process_count: 0,
                thread_count: 0,
                memory_pressure: 0.0,
                io_wait: 0.0,
            },
            trends: HashMap::new(),
            anomalies: Vec::new(),
            performance_indicators: HashMap::new(),
            capacity_utilization: HashMap::new(),
            bottlenecks: Vec::new(),
            optimization_opportunities: Vec::new(),
            quality_assessment: super::super::quality::QualityAssessment {
                overall_score: 0.0,
                completeness: 0.0,
                accuracy: 0.0,
                consistency: 0.0,
                timeliness: 0.0,
                reliability: 0.0,
                confidence_intervals: HashMap::new(),
                indicators: HashMap::new(),
                assessed_at: Instant::now(),
                assessment_method: String::from("unknown"),
            },
            alert_conditions: Vec::new(),
            predictive_metrics: HashMap::new(),
            last_updated: chrono::Utc::now(),
        }
    }
}

pub struct ThresholdValue {
    pub value: f64,
    pub unit: String,
    pub threshold_type: String,
}

pub struct TimeBucket {
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    pub duration: std::time::Duration,
    pub data_points: Vec<f64>,
}
