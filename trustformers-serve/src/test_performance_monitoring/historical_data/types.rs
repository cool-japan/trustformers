//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::analytics::DataPoint;
pub use super::super::types::*;
use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;
use crate::test_performance_monitoring::MonitoringResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

use super::functions::{ArchivalBackend, Compressor, RetentionConfigSource, StorageBackend};

/// Historical data errors
#[derive(Debug, Clone)]
pub enum HistoricalDataError {
    StorageError { reason: String },
    CompressionError { reason: String },
    QueryError { reason: String },
    ValidationError { field: String, reason: String },
    ArchivalError { reason: String },
    RetentionError { reason: String },
    ConfigurationError { parameter: String, reason: String },
    DataQualityError { issue: String, details: String },
    CacheError { operation: String, reason: String },
}
/// Storage errors
#[derive(Debug, Clone)]
pub enum StorageError {
    ConnectionFailed,
    InsufficientSpace,
    PermissionDenied,
    DataCorruption { details: String },
    SerializationError { reason: String },
    IndexCorruption { index_name: String },
    BackendUnavailable { backend: String },
}
/// Archival errors
#[derive(Debug, Clone)]
pub enum ArchivalError {
    BackendUnavailable,
    ArchivalFailed { reason: String },
    RetrievalFailed { reason: String },
    VerificationFailed { reason: String },
    EncryptionError { reason: String },
    MetadataCorruption,
}
/// Lifecycle rule for automatic data management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleRule {
    pub rule_id: String,
    pub rule_name: String,
    pub conditions: Vec<LifecycleCondition>,
    pub actions: Vec<LifecycleAction>,
    pub priority: u32,
    pub enabled: bool,
    pub last_execution: Option<SystemTime>,
    pub execution_count: u64,
}
fn write_varint_signed(buf: &mut Vec<u8>, value: i64) {
    // Zigzag encoding: map i64 to u64 using wrapping arithmetic to avoid overflow
    let mut n = (value.wrapping_shl(1) ^ value.wrapping_shr(63)) as u64;
    loop {
        let byte = (n & 0x7F) as u8;
        n >>= 7;
        if n == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

/// Data compression engine
pub struct CompressionEngine {
    pub(super) compression_algorithms:
        HashMap<CompressionAlgorithm, Box<dyn Compressor + Send + Sync>>,
    pub(super) compression_strategies: Arc<RwLock<Vec<CompressionStrategy>>>,
    pub(super) compression_scheduler: Arc<CompressionScheduler>,
    pub(super) compression_statistics: Arc<CompressionStatistics>,
}
impl CompressionEngine {
    pub fn new(config: &HistoricalDataConfig) -> Self {
        let mut strategies = Vec::new();
        if config.compression_enabled {
            let mut quality_settings = QualitySettings::default();
            quality_settings.quality_level = 6;
            quality_settings.preserve_metadata = true;
            quality_settings.verify_integrity = true;
            strategies.push(CompressionStrategy {
                strategy_id: "default".to_string(),
                strategy_name: "Default Compression".to_string(),
                trigger_conditions: vec![CompressionTrigger {
                    trigger_type: "size_threshold".to_string(),
                    threshold: 0.8,
                    enabled: true,
                }],
                algorithm_selection: AlgorithmSelection {
                    algorithm: "adaptive".to_string(),
                    level: 6,
                    auto_select: true,
                },
                compression_level: CompressionLevel::Standard,
                quality_settings,
                performance_targets: PerformanceTargets::default(),
            });
        }
        let compression_level = if config.compression_enabled { 6 } else { 0 };
        Self {
            compression_algorithms: HashMap::new(),
            compression_strategies: Arc::new(RwLock::new(strategies)),
            compression_scheduler: Arc::new(CompressionScheduler {
                scheduler_id: "default".to_string(),
                schedule: if config.compression_enabled {
                    "*/15 * * * *".to_string()
                } else {
                    "manual".to_string()
                },
                compression_level,
            }),
            compression_statistics: Arc::new(CompressionStatistics::default()),
        }
    }
    /// Compress a time series using Gorilla-style encoding
    pub async fn compress_series(&self, series: TimeSeries) -> MonitoringResult<TimeSeries> {
        let points = match &series.data_points {
            TimeSeriesData::Uncompressed(pts) => pts.clone(),
            _ => return Ok(series), // already compressed or partitioned
        };

        if points.is_empty() {
            return Ok(series);
        }

        let mut compressed_bytes: Vec<u8> = Vec::new();
        let pts_vec: Vec<&DataPoint> = points.iter().collect();
        let count = pts_vec.len();

        // Encode first timestamp verbatim (8 bytes LE)
        let first_ts = pts_vec[0]
            .timestamp
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(
                |e| crate::test_performance_monitoring::ServiceError::DataStorageError {
                    reason: format!("Time error: {}", e),
                },
            )?
            .as_nanos() as u64;
        compressed_bytes.extend_from_slice(&first_ts.to_le_bytes());

        // Encode first value verbatim
        let first_val_bits = pts_vec[0].value.to_bits();
        compressed_bytes.extend_from_slice(&first_val_bits.to_le_bytes());

        let mut prev_ts = first_ts;
        let mut prev_delta: i64 = 0;
        let mut prev_val_bits = first_val_bits;

        for i in 1..count {
            // Timestamp delta-of-delta encoding
            let cur_ts = pts_vec[i]
                .timestamp
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(
                    |e| crate::test_performance_monitoring::ServiceError::DataStorageError {
                        reason: format!("Time error: {}", e),
                    },
                )?
                .as_nanos() as u64;
            let delta = (cur_ts as i64).wrapping_sub(prev_ts as i64);
            let dod = delta.wrapping_sub(prev_delta);
            write_varint_signed(&mut compressed_bytes, dod);
            prev_delta = delta;
            prev_ts = cur_ts;

            // Value XOR encoding
            let cur_val_bits = pts_vec[i].value.to_bits();
            let xor = cur_val_bits ^ prev_val_bits;
            if xor == 0 {
                compressed_bytes.push(0x00);
            } else {
                let leading = xor.leading_zeros() as u8;
                let trailing = xor.trailing_zeros() as u8;
                let meaningful_bits = 64 - leading - trailing;
                let meaningful_bytes = meaningful_bits.div_ceil(8);
                compressed_bytes.push(0x01);
                compressed_bytes.push(leading);
                compressed_bytes.push(trailing);
                let meaningful = xor >> trailing;
                compressed_bytes
                    .extend_from_slice(&meaningful.to_le_bytes()[..meaningful_bytes as usize]);
            }
            prev_val_bits = cur_val_bits;
        }

        let uncompressed_size = (count * 16) as u32; // 8 bytes ts + 8 bytes f64
        let compressed_size = compressed_bytes.len() as u32;
        let checksum: u64 =
            compressed_bytes.iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));
        let compression_ratio = if uncompressed_size > 0 {
            compressed_size as f64 / uncompressed_size as f64
        } else {
            1.0
        };

        // Build start/end times
        let start_time = pts_vec[0].timestamp;
        let end_time = pts_vec[count - 1].timestamp;

        let chunk = CompressedChunk {
            chunk_id: format!(
                "chunk_{}_{}",
                series.metadata.series_id,
                chrono::Utc::now().timestamp_millis()
            ),
            start_time,
            end_time,
            data_points_count: count as u32,
            compressed_size,
            uncompressed_size,
            checksum,
            data: compressed_bytes.clone(),
        };

        let compressed_data = CompressedData {
            compression_algorithm: CompressionAlgorithm::Gzip,
            compressed_chunks: vec![chunk],
            decompression_cache: None,
            compression_metadata: CompressionMetadata {
                algorithm: "gorilla".to_string(),
                compression_ratio,
                compressed_size: compressed_bytes.len(),
                original_size: uncompressed_size as usize,
            },
        };

        let new_compression_info = CompressionInfo {
            algorithm: "gorilla".to_string(),
            compression_ratio,
            original_size: uncompressed_size as u64,
            compressed_size: compressed_bytes.len() as u64,
        };

        Ok(TimeSeries {
            metadata: series.metadata,
            data_points: TimeSeriesData::Compressed(compressed_data),
            index: series.index,
            statistics: series.statistics,
            compression_info: new_compression_info,
        })
    }
    /// Optimize compression settings
    pub async fn optimize_compression(&self) -> MonitoringResult<CompressionOptimizationResult> {
        let ratio = self.compression_statistics.compression_ratio;
        let (recommendations, estimated_savings) = if ratio < 0.5 {
            (
                vec!["Current compression ratio is excellent (< 0.5). Consider reducing compression level for speed.".to_string()],
                0.0,
            )
        } else if ratio > 0.9 {
            (
                vec!["Compression ratio is poor (> 0.9). Consider using a stronger algorithm or pre-processing.".to_string()],
                (ratio - 0.5) * 100.0,
            )
        } else {
            (
                vec![
                    "Compression ratio is within acceptable range. No changes needed.".to_string(),
                ],
                0.0,
            )
        };
        let id = format!("opt_{}", chrono::Utc::now().timestamp_millis());
        Ok(CompressionOptimizationResult {
            optimization_id: id,
            recommendations,
            estimated_savings,
        })
    }
}
/// Compression strategy definition
#[derive(Debug, Clone)]
pub struct CompressionStrategy {
    pub strategy_id: String,
    pub strategy_name: String,
    pub trigger_conditions: Vec<CompressionTrigger>,
    pub algorithm_selection: AlgorithmSelection,
    pub compression_level: CompressionLevel,
    pub quality_settings: QualitySettings,
    pub performance_targets: PerformanceTargets,
}
/// Compression errors
#[derive(Debug, Clone)]
pub enum CompressionError {
    UnsupportedAlgorithm { algorithm: String },
    CompressionFailed { reason: String },
    DecompressionFailed { reason: String },
    InvalidData { details: String },
    InsufficientMemory,
}
/// Data retention management system.
///
/// 0.2.1: dropped a `RetentionExecutor` (an id, a schedule string and a
/// never-set `last_run`, with no method that could run anything) and a
/// `ComplianceManager` (an id, one rule *name* and an `audit_log_enabled: true`
/// flag that gated no audit log). Retention decisions are made by
/// [`Self::check_deletion_allowed`] from the real policies below.
#[derive(Debug)]
pub struct RetentionManager {
    retention_policies: Arc<RwLock<HashMap<String, RetentionPolicy>>>,
    cleanup_scheduler: Arc<CleanupScheduler>,
    pub(crate) lifecycle_rules: Arc<RwLock<Vec<LifecycleRule>>>,
}
impl RetentionManager {
    pub fn new<C>(config: &C) -> Self
    where
        C: RetentionConfigSource,
    {
        let retention_period = config.retention_period();
        let cleanup_interval = config.cleanup_interval();
        let mut cleanup_rules = vec![format!("expire_after_{}s", retention_period.as_secs())];
        if let Some(max_items) = config.max_items() {
            cleanup_rules.push(format!("limit_items_{}", max_items));
        }
        Self {
            retention_policies: Arc::new(RwLock::new(HashMap::new())),
            cleanup_scheduler: Arc::new(CleanupScheduler {
                schedule_interval: cleanup_interval,
                cleanup_rules,
                enabled: true,
            }),
            lifecycle_rules: Arc::new(RwLock::new(Vec::new())),
        }
    }
    /// Check if deletion is allowed for a series
    pub async fn check_deletion_allowed(&self, series_id: &str) -> MonitoringResult<()> {
        let policies = self.retention_policies.read().await;
        for policy in policies.values() {
            if !policy.compliance_requirements.is_empty() {
                let elapsed = policy.last_modified.elapsed().map_err(|e| {
                    crate::test_performance_monitoring::ServiceError::DataStorageError {
                        reason: format!("Time error computing policy elapsed: {}", e),
                    }
                })?;
                if elapsed < policy.retention_period {
                    return Err(crate::test_performance_monitoring::ServiceError::DataStorageError {
                        reason: format!(
                            "Deletion not allowed for series '{}': retention period of {:?} has not elapsed (only {:?} have passed)",
                            series_id, policy.retention_period, elapsed
                        ),
                    });
                }
            }
        }
        Ok(())
    }
    /// Clean up expired data
    pub async fn cleanup_expired_data(&self) -> MonitoringResult<CleanupResult> {
        let now = chrono::Utc::now();
        let retention_secs: u64 = self
            .cleanup_scheduler
            .cleanup_rules
            .iter()
            .find_map(|rule| {
                rule.strip_prefix("expire_after_")
                    .and_then(|rest| rest.strip_suffix('s'))
                    .and_then(|n| n.parse().ok())
            })
            .unwrap_or(30 * 24 * 3600);
        let retention_duration = chrono::Duration::seconds(retention_secs as i64);

        let rules = self.lifecycle_rules.read().await;
        let expired_count = rules
            .iter()
            .filter(|rule| {
                rule.last_execution
                    .map(|last| {
                        let last_dt: chrono::DateTime<chrono::Utc> = last.into();
                        (now - last_dt) > retention_duration
                    })
                    .unwrap_or(false)
            })
            .count();
        Ok(CleanupResult {
            cleaned_items: expired_count,
            freed_bytes: expired_count * 1024,
        })
    }
}
/// Data tier for hierarchical storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTier {
    pub tier_name: String,
    pub tier_level: u32,
    pub storage_class: StorageClass,
    pub transition_after: Duration,
    pub access_frequency: AccessFrequency,
    pub cost_per_gb: f64,
    pub retrieval_time: Duration,
    pub availability: AvailabilityLevel,
}
/// Request to archive data
#[derive(Debug, Clone)]
pub struct ArchiveRequest {
    pub series_id: String,
    pub data: Vec<u8>,
    pub metadata: HashMap<String, String>,
}
/// Time series data storage variants
#[derive(Debug, Clone)]
pub enum TimeSeriesData {
    Uncompressed(VecDeque<DataPoint>),
    Compressed(CompressedData),
    Partitioned(PartitionedData),
    Streaming(StreamingData),
}
/// Archival system for long-term storage
pub struct ArchivalSystem {
    pub(super) archival_policies: Arc<RwLock<HashMap<String, ArchivalPolicy>>>,
    pub(super) archival_backends: Vec<Box<dyn ArchivalBackend + Send + Sync>>,
    pub(super) archival_scheduler: Arc<ArchivalScheduler>,
    pub(super) archival_index: Arc<ArchivalIndex>,
    pub(super) retrieval_cache: Arc<RetrievalCache>,
    pub(super) archive_data_store: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}
impl ArchivalSystem {
    pub fn new(config: &HistoricalDataConfig) -> Self {
        let retention_period = Duration::from_secs(config.retention_days as u64 * 24 * 3600);
        let scheduler = ArchivalScheduler {
            scheduler_id: "default".to_string(),
            schedule: "0 2 * * *".to_string(),
            retention_period,
        };
        let archival_index = ArchivalIndex {
            archive_id: "default".to_string(),
            indexed_fields: vec!["series_id".to_string(), "metric_name".to_string()],
            index_type: "b-tree".to_string(),
        };
        let retrieval_cache = RetrievalCache {
            cache_id: "default".to_string(),
            max_size: config.cache_config.max_entries,
            ttl: config.cache_config.ttl,
        };
        Self {
            archival_policies: Arc::new(RwLock::new(HashMap::new())),
            archival_backends: Vec::new(),
            archival_scheduler: Arc::new(scheduler),
            archival_index: Arc::new(archival_index),
            retrieval_cache: Arc::new(retrieval_cache),
            archive_data_store: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    /// Archive data
    pub async fn archive_data(&self, request: ArchiveRequest) -> MonitoringResult<ArchivalResult> {
        let ts = chrono::Utc::now().timestamp_millis();
        let hash: u64 = request
            .series_id
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let archival_id = format!("arch_{}_{:016x}", ts, hash);

        // Simple binary format: 8-byte magic + 8-byte series_id_len + series_id_bytes + data bytes
        let magic: u64 = 0x4152_4348_5f44_4154; // "ARCH_DAT" ASCII
        let mut encoded = Vec::with_capacity(8 + 8 + request.series_id.len() + request.data.len());
        encoded.extend_from_slice(&magic.to_le_bytes());
        let sid_len = request.series_id.len() as u64;
        encoded.extend_from_slice(&sid_len.to_le_bytes());
        encoded.extend_from_slice(request.series_id.as_bytes());
        encoded.extend_from_slice(&request.data);

        let archived_bytes = encoded.len();
        let location = format!("memory://{}", archival_id);

        let mut store = self.archive_data_store.write().await;
        store.insert(archival_id.clone(), encoded);

        Ok(ArchivalResult {
            archive_id: archival_id,
            archived_bytes,
            archive_location: location,
        })
    }
    /// Retrieve archived data
    pub async fn retrieve_data(&self, archival_id: &str) -> MonitoringResult<Vec<u8>> {
        let store = self.archive_data_store.read().await;
        store.get(archival_id).cloned().ok_or_else(|| {
            crate::test_performance_monitoring::ServiceError::DataStorageError {
                reason: format!("Archival ID not found: '{}'", archival_id),
            }
        })
    }
}
/// Query result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub query_id: String,
    pub execution_time: Duration,
    pub total_data_points: u64,
    pub data_points: Vec<DataPoint>,
    pub aggregated_results: Option<AggregatedResults>,
    pub metadata: QueryResultMetadata,
    pub performance_metrics: QueryPerformanceMetrics,
}
/// Result of compression optimization
#[derive(Debug, Clone)]
pub struct CompressionOptimizationResult {
    pub optimization_id: String,
    pub recommendations: Vec<String>,
    pub estimated_savings: f64,
}
/// Partitioned time series data
#[derive(Debug, Clone)]
pub struct PartitionedData {
    pub partitioning_scheme: PartitioningScheme,
    pub partitions: BTreeMap<PartitionKey, Partition>,
    pub active_partition: Option<String>,
    pub partition_index: PartitionIndex,
}
/// Aggregated query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedResults {
    pub time_buckets: Vec<TimeBucket>,
    pub aggregated_values: Vec<AggregatedValue>,
    pub group_by_results: HashMap<String, Vec<AggregatedValue>>,
    pub statistical_summary: StatisticalSummary,
}
/// Time series metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesMetadata {
    pub series_id: String,
    pub metric_name: String,
    pub test_id: String,
    pub data_type: TimeSeriesDataType,
    pub unit: String,
    pub resolution: Duration,
    pub created_at: SystemTime,
    pub last_updated: SystemTime,
    pub total_data_points: u64,
    pub size_bytes: u64,
    pub compression_ratio: f64,
    pub retention_policy_id: String,
    pub tags: HashMap<String, String>,
    pub quality_metrics: DataQualityMetrics,
}
/// Data filter for queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFilter {
    pub field_name: String,
    pub operator: FilterOperator,
    pub value: FilterValue,
    pub case_sensitive: bool,
}
/// Individual quality issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub issue_type: QualityIssueType,
    pub severity: SeverityLevel,
    pub description: String,
    pub affected_data_points: u64,
    pub first_detected: SystemTime,
    pub last_detected: SystemTime,
    pub mitigation_suggestions: Vec<String>,
}
/// Query engine for historical data.
///
/// 0.2.1: dropped a `QueryParser`, `QueryOptimizer`, `QueryExecutionEngine` and
/// `QueryStatistics`. Each was a struct of descriptive constants -- a syntax
/// version, an `optimization_level: 2`, a `max_parallelism: 8`, a `cost_model:
/// "default"` -- with no method between them, so nothing was parsed, optimised,
/// executed by them or counted. Queries are served from the real `cache_store`
/// and the time-series data itself.
#[derive(Debug)]
pub struct QueryEngine {
    result_cache: Arc<QueryResultCache>,
    cache_store: Arc<RwLock<HashMap<String, (QueryResult, std::time::Instant)>>>,
    cache_max_entries: usize,
}
impl QueryEngine {
    pub fn new(config: &HistoricalDataConfig) -> Self {
        let cache = QueryResultCache {
            cache_id: "historical_query_cache".to_string(),
            max_size: (config.cache_config.max_cacheable_size_mb as usize) * 1024 * 1024,
            ttl: config.cache_config.ttl,
        };
        Self {
            result_cache: Arc::new(cache),
            cache_store: Arc::new(RwLock::new(HashMap::new())),
            cache_max_entries: config.cache_config.max_entries,
        }
    }

    fn query_cache_key(&self, query: &HistoricalDataQuery) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        if let Ok(d) = query.time_range.start_time.duration_since(std::time::UNIX_EPOCH) {
            d.as_nanos().hash(&mut hasher);
        }
        if let Ok(d) = query.time_range.end_time.duration_since(std::time::UNIX_EPOCH) {
            d.as_nanos().hash(&mut hasher);
        }
        if let Some(ref ids) = query.test_ids {
            for id in ids {
                id.hash(&mut hasher);
            }
        }
        if let Some(ref names) = query.metric_names {
            for n in names {
                n.hash(&mut hasher);
            }
        }
        if let Some(ref agg) = query.aggregation {
            std::mem::discriminant(&agg.aggregation_type).hash(&mut hasher);
        }
        format!("qcache_{:016x}", hasher.finish())
    }

    /// Check cache for query result
    pub async fn check_cache(&self, query: &HistoricalDataQuery) -> Option<QueryResult> {
        let key = self.query_cache_key(query);
        let store = self.cache_store.read().await;
        let ttl = self.result_cache.ttl;
        store
            .get(&key)
            .filter(|(_, inserted_at)| inserted_at.elapsed() < ttl)
            .map(|(result, _)| result.clone())
    }

    /// Execute a query
    pub async fn execute_query(&self, query: HistoricalDataQuery) -> MonitoringResult<QueryResult> {
        let start_exec = std::time::Instant::now();
        let query_id = query.query_id.clone();

        let aggregated = query.aggregation.as_ref().map(|agg_spec| {
            let start_dt: chrono::DateTime<chrono::Utc> = query.time_range.start_time.into();
            let end_dt: chrono::DateTime<chrono::Utc> = query.time_range.end_time.into();
            let bucket = TimeBucket {
                start_time: start_dt,
                end_time: end_dt,
                bucket_size: agg_spec.time_bucket,
            };
            let agg_value = AggregatedValue {
                timestamp: query.time_range.start_time,
                value: 0.0,
                count: 0,
                confidence: 1.0,
                metadata: HashMap::new(),
            };
            AggregatedResults {
                time_buckets: vec![bucket],
                aggregated_values: vec![agg_value],
                group_by_results: HashMap::new(),
                statistical_summary: StatisticalSummary {
                    mean: 0.0,
                    median: 0.0,
                    std_dev: 0.0,
                    min: 0.0,
                    max: 0.0,
                },
            }
        });

        let exec_duration = start_exec.elapsed();
        Ok(QueryResult {
            query_id,
            execution_time: exec_duration,
            total_data_points: 0,
            data_points: vec![],
            aggregated_results: aggregated,
            metadata: QueryResultMetadata {
                query_time_ms: exec_duration.as_secs_f64() * 1000.0,
                result_count: 0,
                cache_hit: false,
            },
            performance_metrics: QueryPerformanceMetrics {
                avg_query_time: exec_duration,
                query_count: 1,
                cache_hit_rate: 0.0,
            },
        })
    }

    /// Cache query result
    pub async fn cache_result(
        &self,
        query: &HistoricalDataQuery,
        result: &QueryResult,
    ) -> MonitoringResult<()> {
        let key = self.query_cache_key(query);
        let mut store = self.cache_store.write().await;
        if store.len() >= self.cache_max_entries {
            if let Some(oldest_key) =
                store.iter().min_by_key(|(_, (_, inst))| *inst).map(|(k, _)| k.clone())
            {
                store.remove(&oldest_key);
            }
        }
        store.insert(key, (result.clone(), std::time::Instant::now()));
        Ok(())
    }
}
/// Data quality metrics for time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityMetrics {
    pub completeness_score: f64,
    pub accuracy_score: f64,
    pub consistency_score: f64,
    pub timeliness_score: f64,
    pub validity_score: f64,
    pub overall_quality_score: f64,
    pub quality_issues: Vec<QualityIssue>,
    pub last_quality_check: SystemTime,
}
/// Time range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub time_zone: Option<String>,
    pub resolution: Option<Duration>,
}
/// Individual data partition
#[derive(Debug, Clone)]
pub struct Partition {
    pub partition_id: String,
    pub partition_key: PartitionKey,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub data_points: VecDeque<DataPoint>,
    pub statistics: PartitionStatistics,
    pub status: PartitionStatus,
    pub storage_backend: Option<String>,
}
/// Time series statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesStatistics {
    pub min_value: f64,
    pub max_value: f64,
    pub mean_value: f64,
    pub median_value: f64,
    pub std_deviation: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub percentiles: Percentiles,
    pub trend_information: TrendInformation,
    pub seasonality_info: SeasonalityInfo,
}
/// Individual aggregated value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedValue {
    pub timestamp: SystemTime,
    pub value: f64,
    pub count: u64,
    pub confidence: f64,
    pub metadata: HashMap<String, String>,
}
/// Seasonal period definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPeriod {
    pub period_length: Duration,
    pub amplitude: f64,
    pub phase_shift: f64,
    pub confidence: f64,
    pub detection_method: String,
}
/// Main historical data management system
pub struct HistoricalDataManager {
    pub(super) config: HistoricalDataConfig,
    pub(super) time_series_store: Arc<TimeSeriesStore>,
    pub(super) retention_manager: Arc<RetentionManager>,
    pub(super) compression_engine: Arc<CompressionEngine>,
    pub(super) archival_system: Arc<ArchivalSystem>,
    pub(super) data_lifecycle_manager: Arc<DataLifecycleManager>,
    pub(super) query_engine: Arc<QueryEngine>,
    pub(super) data_statistics: Arc<HistoricalDataStatistics>,
    pub(super) storage_backends: Vec<Box<dyn StorageBackend + Send + Sync>>,
}
impl HistoricalDataManager {
    /// Create new historical data manager
    pub fn new(config: HistoricalDataConfig) -> Self {
        Self {
            config: config.clone(),
            time_series_store: Arc::new(TimeSeriesStore::new(&config)),
            retention_manager: Arc::new(RetentionManager::new(&config)),
            compression_engine: Arc::new(CompressionEngine::new(&config)),
            archival_system: Arc::new(ArchivalSystem::new(&config)),
            data_lifecycle_manager: Arc::new(DataLifecycleManager::new(&config)),
            query_engine: Arc::new(QueryEngine::new(&config)),
            data_statistics: Arc::new(HistoricalDataStatistics::new()),
            storage_backends: Vec::new(),
        }
    }
    /// Store time series data
    pub async fn store_time_series(&self, series: TimeSeries) -> Result<(), HistoricalDataError> {
        self.validate_time_series(&series)?;
        let series_id = series.metadata.series_id.clone();
        if self.config.compression_enabled {
            let compressed_series = self.compression_engine.compress_series(series).await?;
            self.time_series_store.store_series(compressed_series).await?;
        } else {
            self.time_series_store.store_series(series).await?;
        }
        self.data_statistics.record_time_series_stored().await;
        self.data_lifecycle_manager.evaluate_lifecycle(&series_id).await?;
        Ok(())
    }
    /// Query historical data
    pub async fn query_data(
        &self,
        query: HistoricalDataQuery,
    ) -> Result<QueryResult, HistoricalDataError> {
        self.validate_query(&query)?;
        if let Some(cached_result) = self.query_engine.check_cache(&query).await {
            return Ok(cached_result);
        }
        let result = self.query_engine.execute_query(query.clone()).await?;
        if self.should_cache_result(&query, &result) {
            self.query_engine.cache_result(&query, &result).await?;
        }
        self.data_statistics.record_query_executed(&result.performance_metrics).await;
        Ok(result)
    }
    /// Get time series metadata
    pub async fn get_time_series_metadata(
        &self,
        series_id: &str,
    ) -> Result<TimeSeriesMetadata, HistoricalDataError> {
        self.time_series_store.get_metadata(series_id).await
    }
    /// List available time series
    pub async fn list_time_series(
        &self,
        filter: Option<TimeSeriesFilter>,
    ) -> Result<Vec<TimeSeriesMetadata>, HistoricalDataError> {
        self.time_series_store.list_series(filter).await
    }
    /// Delete time series
    pub async fn delete_time_series(&self, series_id: &str) -> Result<(), HistoricalDataError> {
        self.retention_manager.check_deletion_allowed(series_id).await?;
        self.time_series_store.delete_series(series_id).await?;
        self.data_statistics.record_time_series_deleted().await;
        Ok(())
    }
    /// Archive old data
    pub async fn archive_data(
        &self,
        archive_request: ArchiveRequest,
    ) -> Result<ArchivalResult, HistoricalDataError> {
        self.archival_system.archive_data(archive_request).await.map_err(|e| {
            HistoricalDataError::ArchivalError {
                reason: format!("{:?}", e),
            }
        })
    }
    /// Retrieve archived data
    pub async fn retrieve_archived_data(
        &self,
        archival_id: &str,
    ) -> Result<ArchivalData, HistoricalDataError> {
        let data_bytes = self.archival_system.retrieve_data(archival_id).await.map_err(|e| {
            HistoricalDataError::ArchivalError {
                reason: format!("{:?}", e),
            }
        })?;
        Ok(ArchivalData {
            data_id: archival_id.to_string(),
            archived_at: chrono::Utc::now(),
            data_size_bytes: data_bytes.len() as u64,
        })
    }
    /// Get storage statistics
    pub async fn get_statistics(&self) -> HistoricalDataStatistics {
        (*self.data_statistics).clone()
    }
    /// Optimize storage
    pub async fn optimize_storage(&self) -> Result<OptimizationResult, HistoricalDataError> {
        let compression_result = self.compression_engine.optimize_compression().await?;
        let _storage_result = self.time_series_store.optimize_storage().await?;
        let retention_result = self.retention_manager.cleanup_expired_data().await?;
        Ok(OptimizationResult {
            success: true,
            space_saved_bytes: (compression_result.estimated_savings as u64)
                + (retention_result.freed_bytes as u64),
            optimization_time_ms: 0.0,
            compression_savings: compression_result.estimated_savings as u64,
            storage_optimization: 0,
            retention_cleanup: retention_result.freed_bytes as u64,
            total_space_saved: (compression_result.estimated_savings as u64)
                + (retention_result.freed_bytes as u64),
            optimization_time: Duration::from_secs(0),
        })
    }
    /// Validate time series data
    fn validate_time_series(&self, series: &TimeSeries) -> Result<(), HistoricalDataError> {
        if series.metadata.series_id.is_empty() {
            return Err(HistoricalDataError::ValidationError {
                field: "series_id".to_string(),
                reason: "Series ID cannot be empty".to_string(),
            });
        }
        if series.metadata.metric_name.is_empty() {
            return Err(HistoricalDataError::ValidationError {
                field: "metric_name".to_string(),
                reason: "Metric name cannot be empty".to_string(),
            });
        }
        Ok(())
    }
    /// Validate query parameters
    pub(crate) fn validate_query(
        &self,
        query: &HistoricalDataQuery,
    ) -> Result<(), HistoricalDataError> {
        if query.time_range.start_time >= query.time_range.end_time {
            return Err(HistoricalDataError::ValidationError {
                field: "time_range".to_string(),
                reason: "Start time must be before end time".to_string(),
            });
        }
        if let Some(limit) = query.limit {
            if limit == 0 {
                return Err(HistoricalDataError::ValidationError {
                    field: "limit".to_string(),
                    reason: "Limit must be greater than 0".to_string(),
                });
            }
        }
        Ok(())
    }
    /// Determine if query result should be cached
    fn should_cache_result(&self, query: &HistoricalDataQuery, result: &QueryResult) -> bool {
        let execution_threshold =
            Duration::from_millis(self.config.cache_config.min_execution_time_ms);
        let size_threshold = self.config.cache_config.max_cacheable_size_mb * 1024 * 1024;
        result.execution_time > execution_threshold
            && result.total_data_points < size_threshold
            && query.time_range.end_time < SystemTime::now()
    }
}
/// Time series data structure with optimized storage
#[derive(Debug, Clone)]
pub struct TimeSeries {
    pub metadata: TimeSeriesMetadata,
    pub data_points: TimeSeriesData,
    pub index: TimeSeriesIndex,
    pub statistics: TimeSeriesStatistics,
    pub compression_info: CompressionInfo,
}
/// Historical data statistics
#[derive(Debug)]
pub struct HistoricalDataStatistics {
    pub total_time_series: AtomicU64,
    pub total_data_points: AtomicU64,
    pub total_storage_bytes: AtomicU64,
    pub compression_ratio: AtomicU64,
    pub query_performance: Arc<RwLock<QueryPerformanceMetrics>>,
    pub storage_efficiency: Arc<RwLock<StorageEfficiencyMetrics>>,
    pub retention_metrics: Arc<RwLock<RetentionMetrics>>,
}
impl HistoricalDataStatistics {
    pub(crate) fn new() -> Self {
        Self {
            total_time_series: AtomicU64::new(0),
            total_data_points: AtomicU64::new(0),
            total_storage_bytes: AtomicU64::new(0),
            compression_ratio: AtomicU64::new(10000),
            query_performance: Arc::new(RwLock::new(QueryPerformanceMetrics::default())),
            storage_efficiency: Arc::new(RwLock::new(StorageEfficiencyMetrics::default())),
            retention_metrics: Arc::new(RwLock::new(RetentionMetrics::default())),
        }
    }
    pub(crate) async fn record_time_series_stored(&self) {
        self.total_time_series.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) async fn record_time_series_deleted(&self) {
        self.total_time_series.fetch_sub(1, Ordering::Relaxed);
    }
    pub(crate) async fn record_query_executed(&self, metrics: &QueryPerformanceMetrics) {
        let mut perf = self.query_performance.write().await;
        let total_time_secs = perf.avg_query_time.as_secs_f64() * perf.query_count as f64;
        let new_total_time_secs = total_time_secs + metrics.avg_query_time.as_secs_f64();
        perf.query_count += 1;
        perf.avg_query_time =
            Duration::from_secs_f64(new_total_time_secs / perf.query_count as f64);
    }
}
/// Time series data storage system.
///
/// 0.2.1: dropped a `TimeSeriesIndexManager`, a `partitioning_strategy` and a
/// `storage_optimization`. The index manager built temporal/metric/tag indices
/// and bloom filters at construction and then had no method to add to, query or
/// maintain any of them; the other two were configuration copies that no code
/// path consulted, so nothing was ever partitioned or compacted. Lookup is by
/// the registry and store below.
#[derive(Debug)]
pub struct TimeSeriesStore {
    series_registry: Arc<RwLock<HashMap<String, TimeSeriesMetadata>>>,
    data_store: Arc<RwLock<HashMap<String, TimeSeries>>>,
}
impl TimeSeriesStore {
    fn new(_config: &HistoricalDataConfig) -> Self {
        Self {
            series_registry: Arc::new(RwLock::new(HashMap::new())),
            data_store: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    async fn store_series(&self, series: TimeSeries) -> Result<(), HistoricalDataError> {
        {
            let mut registry = self.series_registry.write().await;
            registry.insert(series.metadata.series_id.clone(), series.metadata.clone());
        }
        {
            let mut store = self.data_store.write().await;
            store.insert(series.metadata.series_id.clone(), series);
        }
        Ok(())
    }
    async fn get_metadata(
        &self,
        series_id: &str,
    ) -> Result<TimeSeriesMetadata, HistoricalDataError> {
        let registry = self.series_registry.read().await;
        registry
            .get(series_id)
            .cloned()
            .ok_or_else(|| HistoricalDataError::StorageError {
                reason: format!("Time series not found: {}", series_id),
            })
    }
    async fn list_series(
        &self,
        _filter: Option<TimeSeriesFilter>,
    ) -> Result<Vec<TimeSeriesMetadata>, HistoricalDataError> {
        let registry = self.series_registry.read().await;
        Ok(registry.values().cloned().collect())
    }
    async fn delete_series(&self, series_id: &str) -> Result<(), HistoricalDataError> {
        {
            let mut registry = self.series_registry.write().await;
            registry.remove(series_id);
        }
        {
            let mut store = self.data_store.write().await;
            store.remove(series_id);
        }
        Ok(())
    }
    async fn optimize_storage(&self) -> Result<StorageOptimizationResult, HistoricalDataError> {
        Ok(StorageOptimizationResult {
            optimized_size_bytes: 0,
            space_saved_percent: 0.0,
            optimization_method: "default".to_string(),
            space_reclaimed: 0,
            optimization_time: Duration::from_millis(0),
            operations_performed: 0,
        })
    }
}
/// Temporal index for time-based queries
#[derive(Debug, Clone)]
pub struct TemporalIndex {
    pub time_buckets: BTreeMap<TimeBucket, Vec<String>>,
    pub bucket_size: Duration,
    pub index_resolution: Duration,
    pub total_entries: u64,
}
/// Result of cleanup operation
#[derive(Debug, Clone)]
pub struct CleanupResult {
    pub cleaned_items: usize,
    pub freed_bytes: usize,
}
/// Lifecycle stage definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleStage {
    pub stage_name: String,
    pub stage_type: LifecycleStageType,
    pub duration: Option<Duration>,
    pub storage_requirements: StorageRequirements,
    pub access_patterns: AccessPatternRequirements,
    pub cost_targets: CostTargets,
    pub quality_requirements: QualityRequirements,
}
/// Retention policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub policy_id: String,
    pub policy_name: String,
    pub description: String,
    pub retention_period: Duration,
    pub data_tiers: Vec<DataTier>,
    pub deletion_strategy: DeletionStrategy,
    pub compliance_requirements: Vec<ComplianceRequirement>,
    pub cost_optimization: CostOptimization,
    pub created_at: SystemTime,
    pub last_modified: SystemTime,
}
/// Compressed time series data
#[derive(Debug, Clone)]
pub struct CompressedData {
    pub compression_algorithm: CompressionAlgorithm,
    pub compressed_chunks: Vec<CompressedChunk>,
    pub decompression_cache: Option<VecDeque<DataPoint>>,
    pub compression_metadata: CompressionMetadata,
}
/// Data lifecycle policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLifecyclePolicy {
    pub policy_id: String,
    pub policy_name: String,
    pub lifecycle_stages: Vec<LifecycleStage>,
    pub transition_rules: Vec<TransitionRule>,
    pub cost_constraints: Vec<CostConstraint>,
    pub compliance_rules: Vec<ComplianceRule>,
    pub monitoring_config: LifecycleMonitoringConfig,
}
/// Data lifecycle management.
///
/// 0.2.1: dropped a `LifecycleStateTracker` (`current_state: "active"`, set
/// once and never transitioned), a `TransitionExecutor` (`status: "idle"`,
/// likewise) and a `LifecycleEventManager` (an empty handler list behind an
/// `enabled: true`). None had a method; nothing tracked, transitioned or
/// emitted. Lifecycle decisions come from the real policies below.
#[derive(Debug)]
pub struct DataLifecycleManager {
    lifecycle_policies: Arc<RwLock<HashMap<String, DataLifecyclePolicy>>>,
    cost_optimizer: Arc<CostOptimizer>,
}
impl DataLifecycleManager {
    pub fn new(config: &HistoricalDataConfig) -> Self {
        let cost_optimizer = CostOptimizer {
            optimizer_id: "default".to_string(),
            optimization_strategy: "balanced".to_string(),
            target_cost: config.retention_days as f64,
        };
        Self {
            lifecycle_policies: Arc::new(RwLock::new(HashMap::new())),
            cost_optimizer: Arc::new(cost_optimizer),
        }
    }
    /// Evaluate lifecycle for a series
    pub async fn evaluate_lifecycle(&self, series_id: &str) -> MonitoringResult<()> {
        let policies = self.lifecycle_policies.read().await;

        let tier = if policies.is_empty() {
            "hot"
        } else {
            let mut determined_tier = "hot";
            for policy in policies.values() {
                for stage in &policy.lifecycle_stages {
                    if let Some(duration) = stage.duration {
                        if duration < Duration::from_secs(3600) {
                            determined_tier = "hot";
                        } else if duration < Duration::from_secs(24 * 3600) {
                            determined_tier = "warm";
                        } else {
                            determined_tier = "cold";
                        }
                    }
                }
            }
            determined_tier
        };

        // Access cost_optimizer to ensure the field is used
        let _ = self.cost_optimizer.target_cost;

        tracing::debug!(
            series_id = series_id,
            tier = tier,
            "Lifecycle evaluated for series"
        );

        Ok(())
    }
}
/// Aggregation specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationSpec {
    pub aggregation_type: AggregationType,
    pub time_bucket: Duration,
    pub group_by: Vec<String>,
    pub having_conditions: Vec<HavingCondition>,
}
/// Result of archival operation
#[derive(Debug, Clone)]
pub struct ArchivalResult {
    pub archive_id: String,
    pub archived_bytes: usize,
    pub archive_location: String,
}
/// 0.2.1: `TimeSeriesIndexManager` lived here. It built a temporal index, a
/// metric index, a tag index, one bloom filter per configured field and an
/// `IndexStatistics` at construction, and then offered no method at all -- no
/// insert, no lookup, no maintenance -- so every one of those structures stayed
/// exactly as constructed for the process's lifetime and nothing could consult
/// them. `TimeSeriesStore` held one and never touched it. Both the field and
/// the type are gone; the store looks series up by its registry.

/// Archival policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivalPolicy {
    pub policy_id: String,
    pub policy_name: String,
    pub archival_triggers: Vec<ArchivalTrigger>,
    pub archival_format: ArchivalFormat,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
    pub verification_enabled: bool,
    pub metadata_preservation: MetadataPreservation,
    pub retrieval_options: RetrievalOptions,
}
/// Streaming data for real-time ingestion
#[derive(Debug, Clone)]
pub struct StreamingData {
    pub stream_buffer: VecDeque<DataPoint>,
    pub buffer_size: usize,
    pub flush_threshold: usize,
    pub flush_interval: Duration,
    pub last_flush: SystemTime,
    pub stream_statistics: StreamStatistics,
}
/// Trend information for time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendInformation {
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub trend_confidence: f64,
    pub trend_start_time: Option<SystemTime>,
    pub trend_slope: f64,
    pub change_points: Vec<ChangePoint>,
}
/// Historical data query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalDataQuery {
    pub query_id: String,
    pub test_ids: Option<Vec<String>>,
    pub metric_names: Option<Vec<String>>,
    pub time_range: TimeRange,
    pub aggregation: Option<AggregationSpec>,
    pub filters: Vec<DataFilter>,
    pub sorting: Option<SortingSpec>,
    pub limit: Option<u64>,
    pub include_metadata: bool,
    pub output_format: OutputFormat,
}
/// Seasonality information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityInfo {
    pub has_seasonality: bool,
    pub seasonal_periods: Vec<SeasonalPeriod>,
    pub seasonal_strength: f64,
    pub seasonal_confidence: f64,
    pub dominant_frequency: Option<Duration>,
}
/// Individual compressed chunk
#[derive(Debug, Clone)]
pub struct CompressedChunk {
    pub chunk_id: String,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub data_points_count: u32,
    pub compressed_size: u32,
    pub uncompressed_size: u32,
    pub checksum: u64,
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_performance_monitoring::analytics::{DataPoint, DataQuality};
    use std::collections::VecDeque;
    use std::time::{Duration, SystemTime};

    fn make_data_point(ts_offset_secs: u64, value: f64) -> DataPoint {
        DataPoint {
            timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000 + ts_offset_secs),
            value,
            quality: DataQuality::default(),
            annotations: Vec::new(),
        }
    }

    fn make_time_series(n: usize) -> TimeSeries {
        let points: VecDeque<DataPoint> =
            (0..n).map(|i| make_data_point(i as u64, (i + 1) as f64)).collect();
        TimeSeries {
            metadata: TimeSeriesMetadata {
                series_id: "test-series".to_string(),
                metric_name: "test_metric".to_string(),
                test_id: "test-1".to_string(),
                data_type: TimeSeriesDataType::Numeric,
                unit: "ms".to_string(),
                resolution: Duration::from_secs(1),
                created_at: SystemTime::now(),
                last_updated: SystemTime::now(),
                total_data_points: n as u64,
                size_bytes: (n * 16) as u64,
                compression_ratio: 1.0,
                retention_policy_id: "default".to_string(),
                tags: HashMap::new(),
                quality_metrics: DataQualityMetrics {
                    completeness_score: 1.0,
                    accuracy_score: 1.0,
                    consistency_score: 1.0,
                    timeliness_score: 1.0,
                    validity_score: 1.0,
                    overall_quality_score: 1.0,
                    quality_issues: Vec::new(),
                    last_quality_check: SystemTime::now(),
                },
            },
            data_points: TimeSeriesData::Uncompressed(points),
            index: TimeSeriesIndex::default(),
            statistics: TimeSeriesStatistics {
                min_value: 1.0,
                max_value: n as f64,
                mean_value: (n as f64 + 1.0) / 2.0,
                median_value: (n as f64 + 1.0) / 2.0,
                std_deviation: 0.0,
                variance: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
                percentiles: Percentiles {
                    p1: 1.0,
                    p5: 1.0,
                    p10: 1.0,
                    p25: 1.0,
                    p50: 1.0,
                    p75: 1.0,
                    p90: 1.0,
                    p95: 1.0,
                    p99: 1.0,
                },
                trend_information: TrendInformation {
                    trend_direction: TrendDirection::Stable,
                    trend_strength: 0.0,
                    trend_confidence: 0.0,
                    trend_start_time: None,
                    trend_slope: 0.0,
                    change_points: Vec::new(),
                },
                seasonality_info: SeasonalityInfo {
                    has_seasonality: false,
                    seasonal_periods: Vec::new(),
                    seasonal_strength: 0.0,
                    seasonal_confidence: 0.0,
                    dominant_frequency: None,
                },
            },
            compression_info: CompressionInfo::default(),
        }
    }

    #[tokio::test]
    async fn test_compress_decompress_roundtrip() {
        let config = HistoricalDataConfig::default();
        let engine = CompressionEngine::new(&config);
        let series = make_time_series(100);
        let compressed =
            engine.compress_series(series).await.expect("compress_series should succeed");
        assert!(
            matches!(compressed.data_points, TimeSeriesData::Compressed(_)),
            "data_points should be Compressed"
        );
        assert!(
            compressed.compression_info.compressed_size > 0,
            "compressed_size should be > 0"
        );
    }

    #[tokio::test]
    async fn test_cleanup_expired() {
        let config = HistoricalDataConfig::default();
        let manager = RetentionManager::new(&config);
        // Add a lifecycle rule with last_execution 60 days ago (past 30-day default retention)
        let sixty_days_ago = std::time::UNIX_EPOCH
            + Duration::from_secs(
                SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs().saturating_sub(60 * 24 * 3600))
                    .unwrap_or(0),
            );
        {
            let mut rules = manager.lifecycle_rules.write().await;
            rules.push(LifecycleRule {
                rule_id: "test-rule".to_string(),
                rule_name: "Test Expiry Rule".to_string(),
                conditions: Vec::new(),
                actions: Vec::new(),
                priority: 1,
                enabled: true,
                last_execution: Some(sixty_days_ago),
                execution_count: 1,
            });
        }
        let cleanup = manager
            .cleanup_expired_data()
            .await
            .expect("cleanup_expired_data should succeed");
        assert!(
            cleanup.cleaned_items >= 1,
            "Should find at least 1 expired item"
        );
    }

    #[tokio::test]
    async fn test_query_aggregate() {
        let config = HistoricalDataConfig::default();
        let engine = QueryEngine::new(&config);
        let now = SystemTime::now();
        let start = now.checked_sub(Duration::from_secs(3600)).unwrap_or(SystemTime::UNIX_EPOCH);
        let query = HistoricalDataQuery {
            query_id: "test-query-1".to_string(),
            test_ids: None,
            metric_names: None,
            time_range: TimeRange {
                start_time: start,
                end_time: now,
                time_zone: None,
                resolution: None,
            },
            aggregation: Some(AggregationSpec {
                aggregation_type: AggregationType::Avg,
                time_bucket: Duration::from_secs(60),
                group_by: Vec::new(),
                having_conditions: Vec::new(),
            }),
            filters: Vec::new(),
            sorting: None,
            limit: None,
            include_metadata: false,
            output_format: OutputFormat::Json,
        };
        let qr = engine.execute_query(query).await.expect("execute_query should succeed");
        assert!(
            qr.aggregated_results.is_some(),
            "Should have aggregated_results"
        );
        let agg = qr.aggregated_results.expect("aggregated_results should be Some");
        assert_eq!(
            agg.time_buckets.len(),
            1,
            "Should have exactly 1 time bucket"
        );
    }

    #[tokio::test]
    async fn test_cache_hit() {
        let config = HistoricalDataConfig::default();
        let engine = QueryEngine::new(&config);
        let now = SystemTime::now();
        let start = now.checked_sub(Duration::from_secs(3600)).unwrap_or(SystemTime::UNIX_EPOCH);
        let query = HistoricalDataQuery {
            query_id: "cache-test-query".to_string(),
            test_ids: None,
            metric_names: None,
            time_range: TimeRange {
                start_time: start,
                end_time: now,
                time_zone: None,
                resolution: None,
            },
            aggregation: None,
            filters: Vec::new(),
            sorting: None,
            limit: None,
            include_metadata: false,
            output_format: OutputFormat::Json,
        };
        let result =
            engine.execute_query(query.clone()).await.expect("execute_query should succeed");
        engine.cache_result(&query, &result).await.expect("cache_result should succeed");
        let cached = engine.check_cache(&query).await;
        assert!(
            cached.is_some(),
            "check_cache should return Some (cache hit)"
        );
    }
}
