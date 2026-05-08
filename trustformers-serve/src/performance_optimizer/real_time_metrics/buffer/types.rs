//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

// use super::types::*;
use crate::performance_optimizer::real_time_metrics::types::common::AtomicF32;
use anyhow::{Context, Result};
use oxiarc_deflate::streaming::{GzipStreamDecoder, GzipStreamEncoder};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    io::{Read, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};
use tokio::{sync::Semaphore, task::JoinHandle};

/// Configuration for buffer manager
#[derive(Debug, Clone)]
pub struct BufferManagerConfig {
    /// Maximum number of managed buffers
    pub max_managed_buffers: usize,
    /// Default buffer capacity
    pub default_buffer_capacity: usize,
    /// Enable automatic cleanup
    pub auto_cleanup: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
    /// Performance monitoring enabled
    pub monitoring_enabled: bool,
}
/// Compression algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// LZ4 fast compression
    Lz4,
    /// Gzip compression
    Gzip,
    /// Zstd compression
    Zstd,
    /// Custom compression
    Custom,
}
/// Comprehensive buffer performance statistics and monitoring
///
/// Tracks detailed metrics about buffer usage, performance characteristics,
/// and operational statistics for optimization and debugging purposes.
#[derive(Debug, Default)]
pub struct BufferStatistics {
    /// Total number of insertions performed
    pub total_insertions: AtomicU64,
    /// Number of buffer overwrites (data loss events)
    pub overwrites: AtomicU64,
    /// Total read operations performed
    pub read_operations: AtomicU64,
    /// Average insertion time in nanoseconds
    pub avg_insertion_time_ns: AtomicU64,
    /// Average read time in nanoseconds
    pub avg_read_time_ns: AtomicU64,
    /// Peak buffer utilization percentage
    pub peak_utilization: AtomicF32,
    /// Current utilization percentage
    pub current_utilization: AtomicF32,
    /// Total memory allocated in bytes
    pub memory_allocated: AtomicU64,
    /// Number of memory reallocations
    pub reallocations: AtomicU64,
    /// Number of failed operations
    pub failed_operations: AtomicU64,
    /// Number of compression operations
    pub compression_ops: AtomicU64,
    /// Total compressed size in bytes
    pub compressed_size: AtomicU64,
    /// Original size before compression in bytes
    pub original_size: AtomicU64,
    /// Cache hit rate for read operations
    pub cache_hit_rate: AtomicF32,
    /// Cache miss count
    pub cache_misses: AtomicU64,
    /// Last access timestamp
    pub last_access: parking_lot::Mutex<Option<Instant>>,
    /// Performance degradation events
    pub degradation_events: AtomicU64,
}
/// Statistics for buffer manager operations
#[derive(Debug, Default)]
pub struct BufferManagerStats {
    /// Total buffers managed
    pub managed_buffers: AtomicUsize,
    /// Total operations performed
    pub total_operations: AtomicU64,
    /// Failed operations
    pub failed_operations: AtomicU64,
    /// Average operation latency
    pub avg_operation_latency_ns: AtomicU64,
    /// Peak concurrent operations
    pub peak_concurrent_ops: AtomicUsize,
    /// Current concurrent operations
    pub current_concurrent_ops: AtomicUsize,
}
/// Compression utility functions for data storage optimization
pub struct CompressionUtils;
impl CompressionUtils {
    /// Compress data using the specified algorithm
    pub fn compress(data: &[u8], config: &CompressionConfig) -> Result<Vec<u8>> {
        if data.len() < config.min_size {
            return Ok(data.to_vec());
        }
        match config.algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Gzip => Self::compress_gzip(data, config.level),
            CompressionAlgorithm::Lz4 => Self::compress_lz4(data),
            CompressionAlgorithm::Zstd => Self::compress_zstd(data, config.level),
            CompressionAlgorithm::Custom => {
                // Custom compression requires registering a handler at runtime.
                // No custom handler is registered; identity passthrough is used as a
                // safe fallback so existing data is never silently corrupted.
                // Register a handler or choose Gzip / Lz4 / Zstd explicitly.
                tracing::warn!(
                    "Custom compression algorithm selected but no handler is registered; \
                     falling back to identity (no-op). \
                     Use CompressionAlgorithm::Gzip, Lz4, or Zstd for actual compression."
                );
                Ok(data.to_vec())
            },
        }
    }
    /// Decompress data using the specified algorithm
    pub fn decompress(data: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Gzip => Self::decompress_gzip(data),
            CompressionAlgorithm::Lz4 => Self::decompress_lz4(data),
            CompressionAlgorithm::Zstd => Self::decompress_zstd(data),
            CompressionAlgorithm::Custom => {
                // Symmetric fallback: if the data was compressed with a custom
                // compressor the caller must handle decompression themselves.
                // Return the bytes unchanged so the caller can detect the no-op.
                tracing::warn!(
                    "Custom decompression algorithm selected but no handler is registered; \
                     returning bytes unchanged. \
                     Use CompressionAlgorithm::Gzip, Lz4, or Zstd for supported formats."
                );
                Ok(data.to_vec())
            },
        }
    }
    /// Compress using Gzip
    fn compress_gzip(data: &[u8], level: u32) -> Result<Vec<u8>> {
        let mut encoder = GzipStreamEncoder::new(Vec::new(), level as u8);
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }
    /// Decompress using Gzip
    fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = GzipStreamDecoder::new(data);
        let mut result = Vec::new();
        decoder.read_to_end(&mut result)?;
        Ok(result)
    }
    /// Compress using LZ4
    fn compress_lz4(data: &[u8]) -> Result<Vec<u8>> {
        oxiarc_lz4::compress(data).map_err(|e| anyhow::anyhow!("LZ4 compression failed: {}", e))
    }
    /// Decompress using LZ4
    fn decompress_lz4(data: &[u8]) -> Result<Vec<u8>> {
        // Use a generous max output size (64 MB) for decompression
        oxiarc_lz4::decompress(data, 64 * 1024 * 1024)
            .map_err(|e| anyhow::anyhow!("LZ4 decompression failed: {}", e))
    }
    /// Compress using Zstd
    fn compress_zstd(data: &[u8], level: u32) -> Result<Vec<u8>> {
        oxiarc_zstd::compress_with_level(data, level as i32)
            .map_err(|e| anyhow::anyhow!("Zstd compression failed: {}", e))
    }
    /// Decompress using Zstd
    fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>> {
        oxiarc_zstd::decompress(data)
            .map_err(|e| anyhow::anyhow!("Zstd decompression failed: {}", e))
    }
    /// Calculate compression ratio
    pub fn compression_ratio(original_size: usize, compressed_size: usize) -> f32 {
        if original_size == 0 {
            return 0.0;
        }
        ((original_size - compressed_size) as f32 / original_size as f32) * 100.0
    }
}
/// Storage operation statistics
#[derive(Debug, Default)]
pub struct StorageStats {
    /// Total store operations
    pub store_operations: AtomicU64,
    /// Total retrieve operations
    pub retrieve_operations: AtomicU64,
    /// Total delete operations
    pub delete_operations: AtomicU64,
    /// Total bytes stored
    pub bytes_stored: AtomicU64,
    /// Total bytes retrieved
    pub bytes_retrieved: AtomicU64,
    /// Average operation latency in nanoseconds
    pub avg_latency_ns: AtomicU64,
    /// Failed operations count
    pub failed_operations: AtomicU64,
    /// Storage utilization percentage
    pub utilization_percent: AtomicF32,
}
/// Buffer pool for efficient memory management and reuse
///
/// Maintains a pool of pre-allocated buffers to minimize allocation overhead
/// and improve performance for high-frequency buffer operations.
#[derive(Debug)]
pub struct BufferPool<T> {
    /// Pool of available buffers
    available_buffers: Arc<Mutex<VecDeque<Arc<Mutex<CircularBuffer<T>>>>>>,
    /// Pool configuration
    config: BufferPoolConfig,
    /// Pool statistics
    stats: Arc<BufferPoolStatistics>,
    /// Semaphore for limiting concurrent buffer acquisitions
    semaphore: Arc<Semaphore>,
    /// Background cleanup task handle
    cleanup_task: Arc<Mutex<Option<JoinHandle<()>>>>,
    /// Pool active flag
    active: Arc<AtomicBool>,
}
impl<T: Send + 'static> BufferPool<T> {
    /// Create a new buffer pool with specified configuration
    pub fn new(max_buffers: usize, default_capacity: usize) -> Self
    where
        T: Clone,
    {
        let config = BufferPoolConfig {
            max_buffers,
            min_buffers: max_buffers / 4,
            default_capacity,
            cleanup_interval: Duration::from_secs(60),
            max_idle_time: Duration::from_secs(300),
            monitoring_enabled: true,
            preallocation_strategy: PreallocationStrategy::Lazy,
        };
        Self::new_with_config(config)
    }
    /// Create a new buffer pool with custom configuration
    pub fn new_with_config(config: BufferPoolConfig) -> Self
    where
        T: Clone,
    {
        let pool = Self {
            available_buffers: Arc::new(Mutex::new(VecDeque::new())),
            semaphore: Arc::new(Semaphore::new(config.max_buffers)),
            stats: Arc::new(BufferPoolStatistics::default()),
            cleanup_task: Arc::new(Mutex::new(None)),
            active: Arc::new(AtomicBool::new(true)),
            config,
        };
        if pool.config.preallocation_strategy == PreallocationStrategy::Eager {
            pool.preallocate_buffers();
        }
        pool
    }
    /// Acquire a buffer from the pool
    pub async fn acquire(&self) -> Result<Arc<Mutex<CircularBuffer<T>>>>
    where
        T: Clone,
    {
        let start_time = Instant::now();
        let _permit =
            self.semaphore.acquire().await.context("Failed to acquire semaphore permit")?;
        self.stats.acquisition_requests.fetch_add(1, Ordering::Relaxed);
        if let Some(buffer) = self.available_buffers.lock().pop_front() {
            self.stats.hit_rate.store(self.calculate_hit_rate(), Ordering::Relaxed);
            self.update_acquisition_time(start_time.elapsed());
            return Ok(buffer);
        }
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        let buffer = Arc::new(Mutex::new(CircularBuffer::new(
            self.config.default_capacity,
        )));
        self.stats.buffers_created.fetch_add(1, Ordering::Relaxed);
        self.stats.active_buffers.fetch_add(1, Ordering::Relaxed);
        self.update_acquisition_time(start_time.elapsed());
        Ok(buffer)
    }
    /// Return a buffer to the pool
    pub fn release(&self, buffer: Arc<Mutex<CircularBuffer<T>>>) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        {
            let mut buf = buffer.lock();
            buf.clear();
        }
        let mut available = self.available_buffers.lock();
        if available.len() < self.config.max_buffers {
            available.push_back(buffer);
        } else {
            self.stats.buffers_destroyed.fetch_add(1, Ordering::Relaxed);
        }
        self.stats.active_buffers.fetch_sub(1, Ordering::Relaxed);
    }
    /// Get pool statistics
    pub fn stats(&self) -> Arc<BufferPoolStatistics> {
        Arc::clone(&self.stats)
    }
    /// Start background cleanup task
    pub fn start_cleanup_task(&mut self) {
        let mut cleanup_task = self.cleanup_task.lock();
        if cleanup_task.is_some() {
            return;
        }
        let available_buffers = Arc::clone(&self.available_buffers);
        let stats = Arc::clone(&self.stats);
        let active = Arc::clone(&self.active);
        let cleanup_interval = self.config.cleanup_interval;
        let max_idle_time = self.config.max_idle_time;
        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            while active.load(Ordering::Relaxed) {
                interval.tick().await;
                let mut buffers = available_buffers.lock();
                let initial_count = buffers.len();
                buffers.retain(|buffer| {
                    let buf = buffer.lock();
                    if let Some(last_access) = buf.stats().last_access.lock().as_ref() {
                        last_access.elapsed() < max_idle_time
                    } else {
                        true
                    }
                });
                let cleaned_count = initial_count - buffers.len();
                if cleaned_count > 0 {
                    stats.cleanup_operations.fetch_add(1, Ordering::Relaxed);
                    stats.buffers_destroyed.fetch_add(cleaned_count as u64, Ordering::Relaxed);
                }
            }
        });
        *cleanup_task = Some(task);
    }
    /// Shutdown the buffer pool
    pub async fn shutdown(&mut self) {
        self.active.store(false, Ordering::Relaxed);
        let task = self.cleanup_task.lock().take();
        if let Some(task) = task {
            task.abort();
        }
        let buffers_count = {
            let mut available = self.available_buffers.lock();
            let count = available.len();
            available.clear();
            count
        };
        self.stats.buffers_destroyed.fetch_add(buffers_count as u64, Ordering::Relaxed);
    }
    /// Pre-allocate buffers based on strategy
    fn preallocate_buffers(&self)
    where
        T: Clone,
    {
        let count = match self.config.preallocation_strategy {
            PreallocationStrategy::Eager => self.config.max_buffers,
            PreallocationStrategy::Lazy => 0,
            PreallocationStrategy::Adaptive => self.config.max_buffers / 2,
            PreallocationStrategy::Mixed => self.config.min_buffers,
        };
        let mut available = self.available_buffers.lock();
        for _ in 0..count {
            let buffer = Arc::new(Mutex::new(CircularBuffer::new(
                self.config.default_capacity,
            )));
            available.push_back(buffer);
            self.stats.buffers_created.fetch_add(1, Ordering::Relaxed);
        }
    }
    /// Calculate current hit rate
    fn calculate_hit_rate(&self) -> f32 {
        let requests = self.stats.acquisition_requests.load(Ordering::Relaxed);
        let misses = self.stats.misses.load(Ordering::Relaxed);
        if requests > 0 {
            ((requests - misses) as f32 / requests as f32) * 100.0
        } else {
            0.0
        }
    }
    /// Update acquisition time statistics
    fn update_acquisition_time(&self, duration: Duration) {
        let nanos = duration.as_nanos() as u64;
        let current_avg = self.stats.avg_acquisition_time_ns.load(Ordering::Relaxed);
        let requests = self.stats.acquisition_requests.load(Ordering::Relaxed);
        if let Some(new_avg) = ((current_avg * (requests - 1)) + nanos).checked_div(requests) {
            self.stats.avg_acquisition_time_ns.store(new_avg, Ordering::Relaxed);
        }
    }
}
/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression algorithm to use
    pub algorithm: CompressionAlgorithm,
    /// Compression level (algorithm-specific)
    pub level: u32,
    /// Minimum size threshold for compression
    pub min_size: usize,
    /// Enable compression statistics
    pub enable_stats: bool,
    /// Compression timeout
    pub timeout: Duration,
}
/// Thread-safe buffer manager for coordinating multiple buffers
#[derive(Debug)]
pub struct BufferManager<T> {
    /// Collection of managed buffers
    buffers: Arc<RwLock<HashMap<String, Arc<Mutex<CircularBuffer<T>>>>>>,
    /// Buffer pool for efficient allocation
    pool: Arc<BufferPool<T>>,
    /// Manager statistics
    stats: Arc<BufferManagerStats>,
    /// Configuration
    config: BufferManagerConfig,
}
impl<T: Send + 'static> BufferManager<T> {
    /// Create a new buffer manager
    pub fn new(config: BufferManagerConfig) -> Self
    where
        T: Clone,
    {
        let pool = Arc::new(BufferPool::new(
            config.max_managed_buffers,
            config.default_buffer_capacity,
        ));
        Self {
            buffers: Arc::new(RwLock::new(HashMap::new())),
            pool,
            stats: Arc::new(BufferManagerStats::default()),
            config,
        }
    }
    /// Get or create a buffer with the specified ID
    pub async fn get_buffer(&self, id: &str) -> Result<Arc<Mutex<CircularBuffer<T>>>>
    where
        T: Clone,
    {
        let start_time = Instant::now();
        self.stats.current_concurrent_ops.fetch_add(1, Ordering::Relaxed);
        {
            let buffers = self.buffers.read();
            if let Some(buffer) = buffers.get(id) {
                self.update_operation_stats(start_time.elapsed(), true);
                return Ok(Arc::clone(buffer));
            }
        }
        let buffer = self.pool.acquire().await?;
        {
            let mut buffers = self.buffers.write();
            if buffers.len() >= self.config.max_managed_buffers {
                self.stats.failed_operations.fetch_add(1, Ordering::Relaxed);
                return Err(anyhow::anyhow!("Maximum managed buffers exceeded"));
            }
            buffers.insert(id.to_string(), Arc::clone(&buffer));
        }
        self.stats.managed_buffers.fetch_add(1, Ordering::Relaxed);
        self.update_operation_stats(start_time.elapsed(), true);
        Ok(buffer)
    }
    /// Remove a buffer from management
    pub fn remove_buffer(&self, id: &str) -> Result<()> {
        let start_time = Instant::now();
        let removed = {
            let mut buffers = self.buffers.write();
            buffers.remove(id).is_some()
        };
        if removed {
            self.stats.managed_buffers.fetch_sub(1, Ordering::Relaxed);
            self.update_operation_stats(start_time.elapsed(), true);
            Ok(())
        } else {
            self.stats.failed_operations.fetch_add(1, Ordering::Relaxed);
            self.update_operation_stats(start_time.elapsed(), false);
            Err(anyhow::anyhow!("Buffer not found: {}", id))
        }
    }
    /// Get statistics for all managed buffers
    pub fn get_aggregate_stats(&self) -> HashMap<String, Arc<BufferStatistics>> {
        let buffers = self.buffers.read();
        buffers
            .iter()
            .map(|(id, buffer)| {
                let buf = buffer.lock();
                (id.clone(), buf.stats())
            })
            .collect()
    }
    /// Update operation statistics
    fn update_operation_stats(&self, duration: Duration, success: bool) {
        self.stats.current_concurrent_ops.fetch_sub(1, Ordering::Relaxed);
        self.stats.total_operations.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.stats.failed_operations.fetch_add(1, Ordering::Relaxed);
        }
        let nanos = duration.as_nanos() as u64;
        let current_avg = self.stats.avg_operation_latency_ns.load(Ordering::Relaxed);
        let total_ops = self.stats.total_operations.load(Ordering::Relaxed);
        if let Some(new_avg) = ((current_avg * (total_ops - 1)) + nanos).checked_div(total_ops) {
            self.stats.avg_operation_latency_ns.store(new_avg, Ordering::Relaxed);
        }
        let current_concurrent = self.stats.current_concurrent_ops.load(Ordering::Relaxed);
        let peak = self.stats.peak_concurrent_ops.load(Ordering::Relaxed);
        if current_concurrent > peak {
            self.stats.peak_concurrent_ops.store(current_concurrent, Ordering::Relaxed);
        }
    }
}
/// High-performance generic circular buffer with thread-safe operations
///
/// This implementation provides constant-time insertion and efficient memory management
/// with comprehensive statistics tracking and performance optimization features.
#[derive(Debug)]
pub struct CircularBuffer<T> {
    /// Internal buffer storage with optional entries
    buffer: Vec<Option<T>>,
    /// Current write position (atomic for thread safety)
    write_pos: AtomicUsize,
    /// Current buffer size (atomic for thread safety)
    size: AtomicUsize,
    /// Maximum buffer capacity
    capacity: usize,
    /// Buffer performance statistics
    stats: Arc<BufferStatistics>,
    /// Buffer creation timestamp
    created_at: Instant,
    /// Buffer ID for tracking and debugging
    id: String,
    /// Overflow strategy when buffer is full
    overflow_strategy: OverflowStrategy,
    /// Memory pool reference for efficient allocation
    pool_ref: Option<Weak<BufferPool<T>>>,
}
impl<T> CircularBuffer<T> {
    /// Create a new circular buffer with specified capacity
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of items the buffer can hold
    ///
    /// # Returns
    /// A new `CircularBuffer` instance
    ///
    /// # Example
    /// ```rust
    /// use trustformers_serve::performance_optimizer::real_time_metrics::buffer::CircularBuffer;
    /// let buffer = CircularBuffer::<i32>::new(1024);
    /// assert_eq!(buffer.capacity(), 1024);
    /// ```
    pub fn new(capacity: usize) -> Self
    where
        T: Clone,
    {
        let id = format!("buffer_{}", uuid::Uuid::new_v4());
        Self {
            buffer: vec![None; capacity],
            write_pos: AtomicUsize::new(0),
            size: AtomicUsize::new(0),
            capacity,
            stats: Arc::new(BufferStatistics::default()),
            created_at: Instant::now(),
            id,
            overflow_strategy: OverflowStrategy::Overwrite,
            pool_ref: None,
        }
    }
    /// Create a new buffer with custom configuration
    pub fn new_with_config(capacity: usize, strategy: OverflowStrategy, id: String) -> Self
    where
        T: Clone,
    {
        Self {
            buffer: vec![None; capacity],
            write_pos: AtomicUsize::new(0),
            size: AtomicUsize::new(0),
            capacity,
            stats: Arc::new(BufferStatistics::default()),
            created_at: Instant::now(),
            id,
            overflow_strategy: strategy,
            pool_ref: None,
        }
    }
    /// Push a new item into the buffer
    ///
    /// Handles overflow according to the configured strategy and updates statistics.
    ///
    /// # Arguments
    /// * `item` - The item to insert into the buffer
    ///
    /// # Returns
    /// `Ok(())` on success, `Err` if overflow strategy is `Error` and buffer is full
    pub fn push(&mut self, item: T) -> Result<()>
    where
        T: Clone,
    {
        let start_time = Instant::now();
        let current_size = self.size.load(Ordering::Relaxed);
        if current_size >= self.capacity {
            match self.overflow_strategy {
                OverflowStrategy::DropNew => {
                    self.stats.failed_operations.fetch_add(1, Ordering::Relaxed);
                    return Ok(());
                },
                OverflowStrategy::Error => {
                    self.stats.failed_operations.fetch_add(1, Ordering::Relaxed);
                    return Err(anyhow::anyhow!("Buffer overflow: capacity exceeded"));
                },
                OverflowStrategy::Grow => {
                    self.grow_capacity()?;
                },
                OverflowStrategy::Compress => {
                    self.compress_old_data()?;
                },
                OverflowStrategy::Overwrite => {
                    self.stats.overwrites.fetch_add(1, Ordering::Relaxed);
                },
            }
        }
        let pos = self.write_pos.load(Ordering::Relaxed);
        self.buffer[pos] = Some(item);
        let next_pos = (pos + 1) % self.capacity;
        self.write_pos.store(next_pos, Ordering::Relaxed);
        if current_size < self.capacity {
            self.size.store(current_size + 1, Ordering::Relaxed);
        }
        self.stats.total_insertions.fetch_add(1, Ordering::Relaxed);
        self.update_insertion_time(start_time.elapsed());
        self.update_utilization();
        self.update_last_access();
        Ok(())
    }
    /// Get an item at a specific index (relative to the oldest item)
    pub fn get(&self, index: usize) -> Option<&T> {
        let start_time = Instant::now();
        if index >= self.len() {
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let size = self.size.load(Ordering::Relaxed);
        let write_pos = self.write_pos.load(Ordering::Relaxed);
        let actual_pos =
            if size < self.capacity { index } else { (write_pos + index) % self.capacity };
        let result = self.buffer[actual_pos].as_ref();
        self.stats.read_operations.fetch_add(1, Ordering::Relaxed);
        self.update_read_time(start_time.elapsed());
        self.update_last_access();
        if result.is_some() {
            self.update_cache_hit_rate(true);
        } else {
            self.update_cache_hit_rate(false);
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
    /// Iterate over all items in the buffer (from oldest to newest)
    pub fn iter(&self) -> BufferIterator<'_, T> {
        BufferIterator::new(self)
    }
    /// Get the latest N items from the buffer
    pub fn latest(&self, n: usize) -> Vec<&T> {
        let size = self.len();
        if size == 0 || n == 0 {
            return Vec::new();
        }
        let start_index = size.saturating_sub(n);
        (start_index..size).filter_map(|i| self.get(i)).collect()
    }
    /// Clear all items from the buffer
    pub fn clear(&mut self) {
        for item in &mut self.buffer {
            *item = None;
        }
        self.write_pos.store(0, Ordering::Relaxed);
        self.size.store(0, Ordering::Relaxed);
        self.update_utilization();
    }
    /// Get current size of the buffer
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Get buffer utilization percentage
    pub fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            0.0
        } else {
            (self.len() as f32 / self.capacity as f32) * 100.0
        }
    }
    /// Get buffer statistics
    pub fn stats(&self) -> Arc<BufferStatistics> {
        Arc::clone(&self.stats)
    }
    /// Get buffer ID
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Get buffer age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
    /// Set overflow strategy
    pub fn set_overflow_strategy(&mut self, strategy: OverflowStrategy) {
        self.overflow_strategy = strategy;
    }
    /// Grow buffer capacity dynamically
    fn grow_capacity(&mut self) -> Result<()>
    where
        T: Clone,
    {
        let new_capacity = self.capacity * 2;
        let new_buffer = vec![None; new_capacity];
        for _item in self.iter() {}
        self.buffer = new_buffer;
        self.capacity = new_capacity;
        self.stats.reallocations.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    /// Compress old data to make space (placeholder for compression logic)
    fn compress_old_data(&mut self) -> Result<()> {
        self.stats.compression_ops.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    /// Update insertion time statistics
    fn update_insertion_time(&self, duration: Duration) {
        let nanos = duration.as_nanos() as u64;
        let current_avg = self.stats.avg_insertion_time_ns.load(Ordering::Relaxed);
        let insertions = self.stats.total_insertions.load(Ordering::Relaxed);
        if let Some(new_avg) = ((current_avg * (insertions - 1)) + nanos).checked_div(insertions) {
            self.stats.avg_insertion_time_ns.store(new_avg, Ordering::Relaxed);
        }
    }
    /// Update read time statistics
    fn update_read_time(&self, duration: Duration) {
        let nanos = duration.as_nanos() as u64;
        let current_avg = self.stats.avg_read_time_ns.load(Ordering::Relaxed);
        let reads = self.stats.read_operations.load(Ordering::Relaxed);
        if let Some(new_avg) = ((current_avg * (reads - 1)) + nanos).checked_div(reads) {
            self.stats.avg_read_time_ns.store(new_avg, Ordering::Relaxed);
        }
    }
    /// Update utilization statistics
    pub(crate) fn update_utilization(&self) {
        let current = self.utilization();
        self.stats.current_utilization.store(current, Ordering::Relaxed);
        let peak = self.stats.peak_utilization.load(Ordering::Relaxed);
        if current > peak {
            self.stats.peak_utilization.store(current, Ordering::Relaxed);
        }
    }
    /// Update cache hit rate
    fn update_cache_hit_rate(&self, _hit: bool) {
        let total_reads = self.stats.read_operations.load(Ordering::Relaxed);
        let misses = self.stats.cache_misses.load(Ordering::Relaxed);
        if total_reads > 0 {
            let hits = total_reads - misses;
            let hit_rate = (hits as f32 / total_reads as f32) * 100.0;
            self.stats.cache_hit_rate.store(hit_rate, Ordering::Relaxed);
        }
    }
    /// Update last access time
    fn update_last_access(&self) {
        let mut last_access = self.stats.last_access.lock();
        *last_access = Some(Instant::now());
    }
}
/// Statistics for buffer pool operations
#[derive(Debug, Default)]
pub struct BufferPoolStatistics {
    /// Total buffers created
    pub buffers_created: AtomicU64,
    /// Total buffers destroyed
    pub buffers_destroyed: AtomicU64,
    /// Current active buffers
    pub active_buffers: AtomicUsize,
    /// Pool hit rate (successful acquisitions)
    pub hit_rate: AtomicF32,
    /// Pool miss count
    pub misses: AtomicU64,
    /// Total acquisition requests
    pub acquisition_requests: AtomicU64,
    /// Average acquisition time in nanoseconds
    pub avg_acquisition_time_ns: AtomicU64,
    /// Memory pressure events
    pub memory_pressure_events: AtomicU64,
    /// Cleanup operations performed
    pub cleanup_operations: AtomicU64,
}
/// Pre-allocation strategy for buffer pools
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreallocationStrategy {
    /// Allocate all buffers upfront
    Eager,
    /// Allocate buffers on demand
    Lazy,
    /// Allocate based on usage patterns
    Adaptive,
    /// Mixed strategy based on system load
    Mixed,
}
/// Strategy for handling buffer overflow conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowStrategy {
    /// Overwrite oldest data (circular behavior)
    Overwrite,
    /// Drop new data when full
    DropNew,
    /// Grow buffer capacity dynamically
    Grow,
    /// Trigger error on overflow
    Error,
    /// Compress old data to make space
    Compress,
}
/// Memory-based storage backend for fast access
#[derive(Debug)]
pub struct MemoryStorage<T> {
    /// In-memory data store
    pub(super) data: Arc<RwLock<HashMap<String, Vec<T>>>>,
    /// Storage statistics
    pub(super) stats: Arc<StorageStats>,
    /// Maximum memory usage limit
    pub(super) max_memory_bytes: usize,
    /// Current memory usage estimation
    pub(super) current_memory_bytes: AtomicUsize,
}
impl<T> MemoryStorage<T> {
    /// Create a new memory storage backend
    pub fn new(max_memory_bytes: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(StorageStats::default()),
            max_memory_bytes,
            current_memory_bytes: AtomicUsize::new(0),
        }
    }
    /// Update operation latency statistics
    pub(crate) fn update_latency(&self, duration: Duration) {
        let nanos = duration.as_nanos() as u64;
        let current_avg = self.stats.avg_latency_ns.load(Ordering::Relaxed);
        let total_ops = self.stats.store_operations.load(Ordering::Relaxed)
            + self.stats.retrieve_operations.load(Ordering::Relaxed)
            + self.stats.delete_operations.load(Ordering::Relaxed);
        if let Some(new_avg) = ((current_avg * (total_ops - 1)) + nanos).checked_div(total_ops) {
            self.stats.avg_latency_ns.store(new_avg, Ordering::Relaxed);
        }
    }
    /// Update memory utilization percentage
    pub(crate) fn update_utilization(&self) {
        let current = self.current_memory_bytes.load(Ordering::Relaxed);
        let utilization = if self.max_memory_bytes > 0 {
            (current as f32 / self.max_memory_bytes as f32) * 100.0
        } else {
            0.0
        };
        self.stats.utilization_percent.store(utilization, Ordering::Relaxed);
    }
}
/// File-based storage backend for persistence
#[derive(Debug)]
pub struct FileStorage<T> {
    /// Base directory for file storage
    base_dir: PathBuf,
    /// Storage statistics
    stats: Arc<StorageStats>,
    /// Compression configuration
    compression: Option<CompressionConfig>,
    /// File rotation configuration
    rotation_config: FileRotationConfig,
    /// Concurrent file operations limit
    operation_semaphore: Arc<Semaphore>,
    /// Phantom data to maintain generic type parameter
    _phantom: std::marker::PhantomData<T>,
}
/// File rotation configuration for file storage
#[derive(Debug, Clone)]
pub struct FileRotationConfig {
    /// Maximum file size before rotation
    pub max_file_size: u64,
    /// Maximum number of files to keep
    pub max_files: usize,
    /// Rotation check interval
    pub check_interval: Duration,
    /// Enable compression for rotated files
    pub compress_rotated: bool,
}
/// Iterator for CircularBuffer that yields items from oldest to newest
pub struct BufferIterator<'a, T> {
    pub(super) buffer: &'a CircularBuffer<T>,
    pub(super) current_index: usize,
    pub(super) remaining: usize,
}
impl<'a, T> BufferIterator<'a, T> {
    fn new(buffer: &'a CircularBuffer<T>) -> Self {
        Self {
            buffer,
            current_index: 0,
            remaining: buffer.len(),
        }
    }
}
/// Configuration for buffer pool behavior
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// Maximum number of buffers in pool
    pub max_buffers: usize,
    /// Minimum number of buffers to maintain
    pub min_buffers: usize,
    /// Default buffer capacity
    pub default_capacity: usize,
    /// Cleanup interval for unused buffers
    pub cleanup_interval: Duration,
    /// Maximum idle time before buffer cleanup
    pub max_idle_time: Duration,
    /// Enable performance monitoring
    pub monitoring_enabled: bool,
    /// Pre-allocation strategy
    pub preallocation_strategy: PreallocationStrategy,
}
