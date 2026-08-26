//! Streaming dataset configuration: sources, shuffling, batching, caching and compression.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::Duration;

use super::pipeline::DataSample;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingConfig {
    /// Batch size
    pub batch_size: usize,
    /// Dynamic batching enabled
    pub dynamic: bool,
    /// Maximum batch size for dynamic batching
    pub max_batch_size: usize,
    /// Batching strategy
    pub strategy: BatchingStrategy,
    /// Drop last incomplete batch
    pub drop_last: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchingStrategy {
    /// Fixed size batches
    Fixed,
    /// Variable size based on sequence length
    SequenceLength { max_tokens: usize },
    /// Variable size based on memory usage
    MemoryAware { max_memory_mb: usize },
    /// Adaptive batching based on throughput
    Adaptive,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheType {
    /// In-memory cache
    Memory,
    /// Disk-based cache
    Disk { directory: PathBuf },
    /// Redis cache
    Redis { connection_string: String },
    /// Hybrid memory + disk
    Hybrid { memory_ratio: f64 },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Enable caching
    pub enabled: bool,
    /// Cache type
    pub cache_type: CacheType,
    /// Cache size limit
    pub max_size_gb: f64,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Cache compression
    pub compression: CompressionConfig,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Snappy,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level
    pub level: u8,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSource {
    /// Source identifier
    pub id: String,
    /// Source type
    pub source_type: DataSourceType,
    /// Source-specific configuration
    pub config: HashMap<String, String>,
    /// Weight for sampling from this source
    pub weight: f64,
    /// Quality score
    pub quality_score: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSourceType {
    /// Local file system
    LocalFiles { patterns: Vec<String> },
    /// Remote HTTP/HTTPS endpoints
    Http { urls: Vec<String> },
    /// Database connection
    Database { connection_string: String },
    /// Cloud storage (S3, GCS, Azure)
    CloudStorage { bucket: String, prefix: String },
    /// Kafka stream
    Kafka {
        topics: Vec<String>,
        brokers: Vec<String>,
    },
    /// Custom data source
    Custom { source_name: String },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Random replacement
    Random,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShuffleConfig {
    /// Whether to shuffle data
    pub enabled: bool,
    /// Shuffle buffer size
    pub buffer_size: usize,
    /// Shuffle strategy
    pub strategy: ShuffleStrategy,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShuffleStrategy {
    /// Random shuffle
    Random,
    /// Reservoir sampling
    Reservoir,
    /// Block-wise shuffle
    BlockWise { block_size: usize },
    /// Hash-based shuffle
    HashBased,
}
pub struct StreamingDataset {
    pub config: StreamingDatasetConfig,
    pub buffer: VecDeque<DataSample>,
    pub stats: StreamingStats,
    /// Whether [`DataPipeline::get_batch`](crate::data_pipeline::pipeline::DataPipeline::get_batch) may draw from this dataset.
    pub active: bool,
}
impl StreamingDataset {
    /// Create an inactive streaming dataset with an empty buffer.
    pub fn new(config: StreamingDatasetConfig) -> Self {
        Self {
            config,
            buffer: VecDeque::new(),
            stats: StreamingStats {
                samples_processed: 0,
                bytes_processed: 0,
                processing_time: Duration::from_secs(0),
                error_count: 0,
            },
            active: false,
        }
    }
    /// Append samples, respecting `config.buffer_size`.
    ///
    /// Returns the number of samples accepted. A full buffer is reported as an error rather
    /// than silently discarding data.
    pub fn push_samples(&mut self, samples: Vec<DataSample>) -> Result<usize> {
        let capacity = self.config.buffer_size.max(1);
        let free = capacity.saturating_sub(self.buffer.len());
        if samples.len() > free {
            return Err(anyhow::anyhow!(
                "streaming buffer is full: {} free slot(s) for {} sample(s) (buffer_size = {})",
                free,
                samples.len(),
                capacity
            ));
        }
        let accepted = samples.len();
        for sample in samples {
            let bytes: u64 = sample
                .data
                .values()
                .map(|t| (t.size() * std::mem::size_of::<f32>()) as u64)
                .sum();
            self.stats.bytes_processed += bytes;
            self.buffer.push_back(sample);
        }
        Ok(accepted)
    }
    /// Pop the oldest buffered sample, counting it as processed.
    pub fn pop_sample(&mut self) -> Option<DataSample> {
        let sample = self.buffer.pop_front()?;
        self.stats.samples_processed += 1;
        Some(sample)
    }
    /// Number of samples currently buffered.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }
}
/// Streaming dataset configuration and management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingDatasetConfig {
    /// Data sources for streaming
    pub sources: Vec<DataSource>,
    /// Buffer size for streaming
    pub buffer_size: usize,
    /// Prefetch buffer size
    pub prefetch_size: usize,
    /// Shuffle configuration
    pub shuffle: ShuffleConfig,
    /// Batching configuration
    pub batching: BatchingConfig,
    /// Caching configuration
    pub caching: CachingConfig,
}
#[derive(Debug, Clone)]
pub struct StreamingStats {
    pub samples_processed: usize,
    pub bytes_processed: u64,
    pub processing_time: Duration,
    pub error_count: usize,
}
