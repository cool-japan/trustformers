//! Storage Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

// Import types from sibling modules
use super::config::{ReportConfig, TestPerformanceMonitoringConfig};

#[derive(Debug)]
pub struct ReportStorage {
    storage_path: String,
    retention_policy: Duration,
}

impl ReportStorage {
    pub fn new(config: &TestPerformanceMonitoringConfig) -> Self {
        Self {
            storage_path: "/tmp/reports".to_string(),
            retention_policy: config.retention_period,
        }
    }

    pub fn from_report_config(config: &ReportConfig) -> Self {
        let retention_policy =
            config.auto_generate_interval.unwrap_or_else(|| Duration::from_secs(24 * 3600));

        Self {
            storage_path: "/tmp/reports".to_string(),
            retention_policy,
        }
    }

    /// Get a report by ID
    /// TODO: Implement actual report retrieval from storage
    pub async fn get_report(
        &self,
        _report_id: &str,
    ) -> Result<crate::test_performance_monitoring::reporting::Report, anyhow::Error> {
        Ok(crate::test_performance_monitoring::reporting::Report {
            report_id: "stub".to_string(),
            test_id: "stub".to_string(),
            report_type: crate::test_performance_monitoring::reporting::ReportType::Summary,
            content: "Stub report".to_string(),
            generated_at: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimeSeriesIndex {
    index_entries: HashMap<String, u64>,
    last_updated: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CompressionInfo {
    pub algorithm: String,
    pub compression_ratio: f64,
    pub original_size: u64,
    pub compressed_size: u64,
}

pub struct ArchivalData {
    pub data_id: String,
    pub archived_at: DateTime<Utc>,
    pub data_size_bytes: u64,
}

pub struct ArchivalResult {
    pub success: bool,
    pub archived_count: usize,
    pub total_size_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchivalFilter {
    pub age_threshold: Duration,
    pub size_threshold: u64,
    pub include_patterns: Vec<String>,
}

pub struct ArchivalMetadata {
    pub archive_id: String,
    pub created_at: DateTime<Utc>,
    pub record_count: u64,
    pub checksum: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TagIndex {
    pub tag_name: String,
    pub tag_values: Vec<String>,
    pub usage_count: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BloomFilter {
    pub size_bits: usize,
    pub hash_count: usize,
    pub false_positive_rate: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QueryResultMetadata {
    pub query_time_ms: f64,
    pub result_count: usize,
    pub cache_hit: bool,
}

pub struct StorageOptimizationResult {
    pub optimized_size_bytes: u64,
    pub space_saved_percent: f64,
    pub optimization_method: String,
    pub space_reclaimed: u64,
    pub optimization_time: Duration,
    pub operations_performed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionKey {
    pub key_type: String,
    pub key_value: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartitioningStrategy {
    pub strategy_type: String,
    pub partition_key: String,
    pub partition_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StorageOptimization {
    pub compression_enabled: bool,
    pub deduplication_enabled: bool,
    pub archival_enabled: bool,
}

impl Default for StorageOptimization {
    fn default() -> Self {
        Self {
            compression_enabled: true,
            deduplication_enabled: false,
            archival_enabled: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetadata {
    pub algorithm: String,
    pub compression_ratio: f64,
    pub compressed_size: usize,
    pub original_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitioningScheme {
    pub scheme_id: String,
    pub partition_key: String,
    pub partition_count: usize,
    pub strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionIndex {
    pub partition_id: String,
    pub index_type: String,
    pub indexed_fields: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RetentionExecutor {
    pub executor_id: String,
    pub schedule: String,
    pub last_run: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompressionScheduler {
    pub scheduler_id: String,
    pub schedule: String,
    pub compression_level: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionTrigger {
    pub trigger_type: String,
    pub threshold: f64,
    pub enabled: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArchivalScheduler {
    pub scheduler_id: String,
    pub schedule: String,
    pub retention_period: Duration,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArchivalIndex {
    pub archive_id: String,
    pub indexed_fields: Vec<String>,
    pub index_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivalTrigger {
    pub trigger_type: String,
    // TODO: Re-enable once utils::duration_serde is implemented
    // #[serde(with = "crate::utils::duration_serde")]
    pub age_threshold: Duration,
    pub size_threshold: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageRequirements {
    pub min_capacity: usize,
    pub storage_class: String,
    pub redundancy_level: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryParser {
    pub parser_id: String,
    pub syntax_version: String,
    pub strict_mode: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryOptimizer {
    pub optimizer_id: String,
    pub optimization_level: i32,
    pub cost_model: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryExecutionEngine {
    pub engine_id: String,
    pub max_parallelism: usize,
    pub timeout: Duration,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResultCache {
    pub cache_id: String,
    pub max_size: usize,
    pub ttl: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuery {
    pub query_id: String,
    pub query_text: String,
    pub parameters: HashMap<String, String>,
}
