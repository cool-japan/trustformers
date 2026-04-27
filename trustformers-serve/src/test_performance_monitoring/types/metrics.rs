//! Metrics Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

// Import types from sibling modules
use super::enums::MetricValue;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QueryPerformanceMetrics {
    pub avg_query_time: Duration,
    pub query_count: u64,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StorageEfficiencyMetrics {
    pub compression_ratio: f64,
    pub storage_used: u64,
    pub storage_capacity: u64,
}

impl Default for StorageEfficiencyMetrics {
    fn default() -> Self {
        Self {
            compression_ratio: 1.0,
            storage_used: 0,
            storage_capacity: 0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RetentionMetrics {
    pub retention_period: Duration,
    pub cleanup_frequency: Duration,
    pub purged_records: u64,
}

impl Default for RetentionMetrics {
    fn default() -> Self {
        Self {
            retention_period: Duration::from_secs(7 * 24 * 3600), // 7 days
            cleanup_frequency: Duration::from_secs(24 * 3600),    // Daily
            purged_records: 0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatabaseStatistics {
    pub query_count: u64,
    pub avg_query_time: Duration,
    pub connection_count: u32,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CurrentPerformanceMetrics {
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub active_tests: usize,
    pub throughput: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub report_id: String,
    pub title: String,
    pub generated_at: DateTime<Utc>,
    pub metrics: HashMap<String, MetricValue>,
    pub summary: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceStream {
    pub stream_id: String,
    pub start_time: DateTime<Utc>,
    pub metrics_buffer: Vec<CurrentPerformanceMetrics>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemResourceMetrics {
    pub timestamp: DateTime<Utc>,
    pub cpu_percent: f64,
    pub memory_bytes: u64,
    pub memory_percent: f64,
    pub disk_io_read: u64,
    pub disk_io_write: u64,
    pub network_rx: u64,
    pub network_tx: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestExecutionMetrics {
    pub test_id: String,
    pub execution_time: Duration,
    pub success: bool,
    pub cpu_time: Duration,
    pub memory_peak: u64,
    pub io_operations: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub evaluations_count: u64,
    pub avg_duration_ms: f64,
    pub success_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StorageStatistics {
    pub total_size_bytes: u64,
    pub compressed_size_bytes: u64,
    pub record_count: u64,
    pub compression_ratio: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetricIndex {
    pub metric_name: String,
    pub indexed_fields: Vec<String>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexStatistics {
    pub index_count: usize,
    pub total_entries: u64,
    pub index_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCriteria {
    pub metric_name: String,
    pub operator: String,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcImpactMetrics {
    pub gc_count: u64,
    pub total_pause_time_ms: f64,
    pub avg_pause_time_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RetentionStatistics {
    pub total_size: usize,
    pub retained_count: usize,
    pub deleted_count: usize,
    pub last_cleanup: DateTime<Utc>,
}

impl Default for RetentionStatistics {
    fn default() -> Self {
        Self {
            total_size: 0,
            retained_count: 0,
            deleted_count: 0,
            last_cleanup: chrono::Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionStatistics {
    pub partition_id: String,
    pub size: usize,
    pub record_count: usize,
    pub last_modified: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStatistics {
    pub stream_id: String,
    pub event_count: usize,
    pub bytes_processed: usize,
    pub throughput: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompressionStatistics {
    pub total_compressed: usize,
    pub total_saved: usize,
    pub compression_ratio: f64,
    pub last_compression: DateTime<Utc>,
}

impl Default for CompressionStatistics {
    fn default() -> Self {
        Self {
            total_compressed: 0,
            total_saved: 0,
            compression_ratio: 1.0,
            last_compression: chrono::Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    // TODO: Re-enable once utils::duration_serde is implemented
    // #[serde(with = "crate::utils::duration_serde")]
    pub max_latency: Duration,
    pub min_throughput: f64,
    pub max_cpu_usage: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            max_latency: Duration::from_secs(1),
            min_throughput: 0.0,
            max_cpu_usage: 0.8,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryStatistics {
    pub execution_time: Duration,
    pub rows_scanned: usize,
    pub rows_returned: usize,
    pub cache_hit: bool,
}

impl Default for QueryStatistics {
    fn default() -> Self {
        Self {
            execution_time: Duration::from_secs(0),
            rows_scanned: 0,
            rows_returned: 0,
            cache_hit: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubscriptionMetrics {
    pub total_subscriptions: usize,
    pub active_subscriptions: usize,
    pub notifications_sent: usize,
    pub notifications_failed: usize,
    pub last_notification: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UsageStatistics {
    pub total_requests: usize,
    pub total_bytes: usize,
    pub average_latency: Duration,
    pub peak_throughput: f64,
    pub error_rate: f64,
}

impl Default for UsageStatistics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_bytes: 0,
            average_latency: Duration::from_secs(0),
            peak_throughput: 0.0,
            error_rate: 0.0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_throughput: f64,
    pub request_latency: Duration,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            disk_usage: 0.0,
            network_throughput: 0.0,
            request_latency: Duration::from_secs(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_performance_metrics_default() {
        let m = QueryPerformanceMetrics::default();
        assert_eq!(m.query_count, 0);
        assert!((m.cache_hit_rate - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_storage_efficiency_metrics_default() {
        let m = StorageEfficiencyMetrics::default();
        assert!((m.compression_ratio - 1.0).abs() < 1e-9);
        assert_eq!(m.storage_used, 0);
    }

    #[test]
    fn test_retention_metrics_default() {
        let m = RetentionMetrics::default();
        assert_eq!(m.retention_period, Duration::from_secs(7 * 24 * 3600));
        assert_eq!(m.purged_records, 0);
    }

    #[test]
    fn test_retention_statistics_default() {
        let m = RetentionStatistics::default();
        assert_eq!(m.total_size, 0);
        assert_eq!(m.retained_count, 0);
        assert_eq!(m.deleted_count, 0);
    }

    #[test]
    fn test_compression_statistics_default() {
        let m = CompressionStatistics::default();
        assert_eq!(m.total_compressed, 0);
        assert!((m.compression_ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_performance_targets_default() {
        let m = PerformanceTargets::default();
        assert_eq!(m.max_latency, Duration::from_secs(1));
        assert!((m.max_cpu_usage - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_query_statistics_default() {
        let m = QueryStatistics::default();
        assert_eq!(m.rows_scanned, 0);
        assert!(!m.cache_hit);
    }

    #[test]
    fn test_usage_statistics_default() {
        let m = UsageStatistics::default();
        assert_eq!(m.total_requests, 0);
        assert!((m.error_rate - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_performance_metrics_default() {
        let m = PerformanceMetrics::default();
        assert!((m.cpu_usage - 0.0).abs() < 1e-9);
        assert!((m.memory_usage - 0.0).abs() < 1e-9);
        assert!((m.disk_usage - 0.0).abs() < 1e-9);
    }
}
