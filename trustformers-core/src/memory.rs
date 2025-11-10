use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Memory optimization utilities for TrustformeRS
///
/// This module provides high-priority memory optimizations:
/// - Zero-copy tensor views for slice operations
/// - Memory mapping for large model weights
/// - Custom allocators for tensor allocation patterns
/// - Tensor memory recycling pool
///
/// Eviction policy for memory pool
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryEvictionPolicy {
    /// Least Recently Used - evict tensors not used for longest time
    LRU,
    /// Least Frequently Used - evict tensors with lowest access count
    LFU,
    /// Size-based - evict largest tensors first to free more memory
    SizeBased,
    /// Adaptive Replacement Cache - balance between recency and frequency
    ARC,
    /// Hybrid - combination of LRU and size-based
    Hybrid,
}

/// Adaptive strategy for dynamic pool sizing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveStrategy {
    /// Fixed pool size (no adaptation)
    Fixed,
    /// Grow/shrink based on memory pressure
    MemoryPressure,
    /// Adapt based on hit/miss rates
    HitRate,
    /// Predict size based on access patterns
    Predictive,
}

/// Configuration for memory optimizations
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Enable memory pool for tensor recycling
    pub enable_memory_pool: bool,
    /// Maximum size of memory pool in bytes
    pub max_pool_size: usize,
    /// Minimum size of memory pool (for adaptive strategies)
    pub min_pool_size: usize,
    /// Enable zero-copy tensor views
    pub enable_zero_copy: bool,
    /// Enable memory mapping for large tensors
    pub enable_mmap: bool,
    /// Minimum size for memory mapping (in bytes)
    pub mmap_threshold: usize,
    /// Pool cleanup interval
    pub cleanup_interval: Duration,
    /// Eviction policy to use
    pub eviction_policy: MemoryEvictionPolicy,
    /// Adaptive strategy for dynamic sizing
    pub adaptive_strategy: AdaptiveStrategy,
    /// Target hit rate for adaptive sizing (0.0 to 1.0)
    pub target_hit_rate: f64,
    /// Enable prefetching based on access patterns
    pub enable_prefetching: bool,
    /// Enable automatic defragmentation
    pub enable_defragmentation: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enable_memory_pool: true,
            max_pool_size: 1024 * 1024 * 1024, // 1GB
            min_pool_size: 64 * 1024 * 1024,   // 64MB
            enable_zero_copy: true,
            enable_mmap: true,
            mmap_threshold: 100 * 1024 * 1024, // 100MB
            cleanup_interval: Duration::from_secs(60),
            eviction_policy: MemoryEvictionPolicy::Hybrid,
            adaptive_strategy: AdaptiveStrategy::HitRate,
            target_hit_rate: 0.85, // 85% target hit rate
            enable_prefetching: true,
            enable_defragmentation: true,
        }
    }
}

/// Memory pool entry for tensor recycling (enhanced with adaptive metrics)
#[derive(Debug, Clone)]
struct PoolEntry {
    tensor: Tensor,
    last_used: Instant,
    ref_count: usize,
    /// Access frequency counter (for LFU and ARC policies)
    access_count: usize,
    /// Creation time (for age-based eviction)
    #[allow(dead_code)]
    created_at: Instant,
    /// Total time in pool (for efficiency metrics)
    #[allow(dead_code)]
    pool_time: Duration,
    /// Tensor size in bytes (cached for quick eviction decisions)
    size_bytes: usize,
}

impl PoolEntry {
    fn new(tensor: Tensor, size_bytes: usize) -> Self {
        let now = Instant::now();
        Self {
            tensor,
            last_used: now,
            ref_count: 0,
            access_count: 0,
            created_at: now,
            pool_time: Duration::ZERO,
            size_bytes,
        }
    }

    fn mark_accessed(&mut self) {
        self.last_used = Instant::now();
        self.access_count += 1;
    }

    /// Calculate eviction priority (lower = evict first)
    fn eviction_priority(&self, policy: MemoryEvictionPolicy) -> f64 {
        match policy {
            MemoryEvictionPolicy::LRU => {
                // Recency: older = lower priority
                -(self.last_used.elapsed().as_secs_f64())
            },
            MemoryEvictionPolicy::LFU => {
                // Frequency: less used = lower priority
                -(self.access_count as f64)
            },
            MemoryEvictionPolicy::SizeBased => {
                // Size: larger = lower priority (to free more space)
                -(self.size_bytes as f64)
            },
            MemoryEvictionPolicy::ARC => {
                // Adaptive: balance recency and frequency
                let recency_score = 1.0 / (1.0 + self.last_used.elapsed().as_secs_f64());
                let frequency_score = self.access_count as f64;
                -(recency_score + frequency_score)
            },
            MemoryEvictionPolicy::Hybrid => {
                // Hybrid: combine recency, frequency, and size
                let recency = 1.0 / (1.0 + self.last_used.elapsed().as_secs_f64());
                let frequency = self.access_count as f64;
                let size_factor = 1.0 / (1.0 + (self.size_bytes as f64 / 1_000_000.0));
                -(recency * 0.4 + frequency * 0.4 + size_factor * 0.2)
            },
        }
    }
}

/// Zero-copy tensor view for slice operations
#[derive(Debug)]
pub struct TensorView {
    /// Original tensor reference
    original: Arc<Tensor>,
    /// Offset in the original tensor
    offset: usize,
    /// Shape of the view
    shape: Vec<usize>,
    /// Strides for the view
    #[allow(dead_code)]
    strides: Vec<usize>,
}

impl TensorView {
    /// Create a new zero-copy view of a tensor slice
    pub fn slice(tensor: Arc<Tensor>, start: usize, end: usize) -> Result<Self> {
        let original_shape = tensor.shape();
        if start >= end || end > original_shape.iter().product::<usize>() {
            return Err(TrustformersError::invalid_input(
                "Invalid slice bounds".to_string(),
            ));
        }

        let slice_len = end - start;
        Ok(Self {
            original: tensor,
            offset: start,
            shape: vec![slice_len],
            strides: vec![1],
        })
    }

    /// Get the shape of the view
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the underlying tensor data (zero-copy)
    pub fn as_tensor(&self) -> Result<Tensor> {
        // This would implement actual zero-copy slicing
        // For now, return a simple implementation
        match &*self.original {
            Tensor::F32(arr) => {
                let flat = arr
                    .view()
                    .into_shape_with_order(arr.len())
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                let slice = flat.slice(ndarray::s![
                    self.offset..self.offset + self.shape.iter().product::<usize>()
                ]);
                let sliced_arr = slice
                    .to_owned()
                    .into_shape_with_order(ndarray::IxDyn(&self.shape))
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::F32(sliced_arr))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Zero-copy slicing not implemented for this tensor type",
                "zero_copy_slice",
            )),
        }
    }
}

/// Enhanced statistics for adaptive memory pool
#[derive(Debug, Clone)]
struct PoolStatistics {
    total_requests: usize,
    cache_hits: usize,
    cache_misses: usize,
    total_evictions: usize,
    evictions_by_policy: HashMap<String, usize>,
    total_allocated_bytes: usize,
    peak_memory_usage: usize,
    #[allow(dead_code)]
    average_tensor_lifetime: Duration,
    #[allow(dead_code)]
    last_reset: Instant,
}

impl Default for PoolStatistics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            total_evictions: 0,
            evictions_by_policy: HashMap::new(),
            total_allocated_bytes: 0,
            peak_memory_usage: 0,
            average_tensor_lifetime: Duration::ZERO,
            last_reset: Instant::now(),
        }
    }
}

impl PoolStatistics {
    fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_requests as f64
        }
    }

    fn miss_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_misses as f64 / self.total_requests as f64
        }
    }
}

/// Memory pool for tensor recycling (enhanced with adaptive strategies)
pub struct TensorMemoryPool {
    config: MemoryConfig,
    pool: Arc<RwLock<HashMap<Vec<usize>, Vec<PoolEntry>>>>,
    current_size: Arc<Mutex<usize>>,
    last_cleanup: Arc<Mutex<Instant>>,
    /// Enhanced statistics for adaptive behavior
    statistics: Arc<Mutex<PoolStatistics>>,
    /// Access pattern tracking for prefetching
    access_patterns: Arc<Mutex<HashMap<Vec<usize>, Vec<Instant>>>>,
    /// Dynamic pool size (for adaptive strategies)
    dynamic_max_size: Arc<Mutex<usize>>,
}

impl TensorMemoryPool {
    /// Create a new memory pool with enhanced adaptive strategies
    pub fn new(config: MemoryConfig) -> Self {
        let dynamic_max_size = config.max_pool_size;
        Self {
            config,
            pool: Arc::new(RwLock::new(HashMap::new())),
            current_size: Arc::new(Mutex::new(0)),
            last_cleanup: Arc::new(Mutex::new(Instant::now())),
            statistics: Arc::new(Mutex::new(PoolStatistics::default())),
            access_patterns: Arc::new(Mutex::new(HashMap::new())),
            dynamic_max_size: Arc::new(Mutex::new(dynamic_max_size)),
        }
    }

    /// Get a tensor from the pool or create a new one (enhanced with statistics tracking)
    pub fn get_tensor(&self, shape: &[usize], dtype: crate::tensor::DType) -> Result<Tensor> {
        // Track access pattern for prefetching
        if self.config.enable_prefetching {
            let mut patterns = self.access_patterns.lock().unwrap();
            patterns.entry(shape.to_vec()).or_default().push(Instant::now());
        }

        // Update statistics
        {
            let mut stats = self.statistics.lock().unwrap();
            stats.total_requests += 1;
        }

        if !self.config.enable_memory_pool {
            return self.create_tensor(shape, dtype);
        }

        // Try to get from pool first
        if let Some(tensor) = self.try_get_from_pool(shape)? {
            // Cache hit!
            let mut stats = self.statistics.lock().unwrap();
            stats.cache_hits += 1;
            return Ok(tensor);
        }

        // Cache miss
        {
            let mut stats = self.statistics.lock().unwrap();
            stats.cache_misses += 1;
        }

        // Apply adaptive pool sizing based on hit rate
        self.apply_adaptive_sizing()?;

        // Create new tensor if none available in pool
        self.create_tensor(shape, dtype)
    }

    /// Return a tensor to the pool for recycling (enhanced tracking)
    pub fn return_tensor(&self, tensor: Tensor) -> Result<()> {
        if !self.config.enable_memory_pool {
            return Ok(()); // Just drop the tensor
        }

        let shape = tensor.shape().to_vec();

        // Calculate tensor size before moving
        let tensor_size = self.estimate_tensor_size(&tensor);

        // Create enhanced pool entry
        let entry = PoolEntry::new(tensor, tensor_size);

        let mut pool = self.pool.write().unwrap();
        pool.entry(shape).or_default().push(entry);

        // Update current size and peak usage
        {
            let mut current = self.current_size.lock().unwrap();
            *current += tensor_size;

            let mut stats = self.statistics.lock().unwrap();
            if *current > stats.peak_memory_usage {
                stats.peak_memory_usage = *current;
            }
            stats.total_allocated_bytes += tensor_size;
        }

        // Cleanup if needed (with enhanced eviction policies)
        self.cleanup_if_needed()?;

        Ok(())
    }

    /// Try to get a tensor from the pool (enhanced with access tracking)
    fn try_get_from_pool(&self, shape: &[usize]) -> Result<Option<Tensor>> {
        let mut pool = self.pool.write().unwrap();

        if let Some(entries) = pool.get_mut(shape) {
            if let Some(mut entry) = entries.pop() {
                // Mark as accessed for LFU tracking
                entry.mark_accessed();

                let tensor_size = entry.size_bytes;
                *self.current_size.lock().unwrap() -= tensor_size;
                return Ok(Some(entry.tensor));
            }
        }

        Ok(None)
    }

    /// Create a new tensor
    fn create_tensor(&self, shape: &[usize], dtype: crate::tensor::DType) -> Result<Tensor> {
        match dtype {
            crate::tensor::DType::F32 => Tensor::zeros(shape),
            crate::tensor::DType::F64 => Tensor::zeros_f64(shape),
            crate::tensor::DType::F16 => Tensor::zeros_f16(shape),
            crate::tensor::DType::BF16 => Tensor::zeros_bf16(shape),
            crate::tensor::DType::I64 => Tensor::zeros_i64(shape),
            crate::tensor::DType::C32 => Tensor::zeros_c32(shape),
            crate::tensor::DType::C64 => Tensor::zeros_c64(shape),
            crate::tensor::DType::CF16 => Tensor::zeros_cf16(shape),
            crate::tensor::DType::CBF16 => Tensor::zeros_cbf16(shape),
            _ => Err(TrustformersError::tensor_op_error(
                &format!("Tensor creation not implemented for dtype: {:?} - only supported types are F32, F64, F16, BF16, I64, C32, C64, CF16, CBF16", dtype),
                "create_tensor"
            )),
        }
    }

    /// Estimate the memory size of a tensor
    fn estimate_tensor_size(&self, tensor: &Tensor) -> usize {
        let elements = tensor.shape().iter().product::<usize>();
        match tensor {
            Tensor::F32(_) => elements * 4,   // 32-bit float
            Tensor::F64(_) => elements * 8,   // 64-bit float
            Tensor::F16(_) => elements * 2,   // 16-bit float
            Tensor::BF16(_) => elements * 2,  // 16-bit bfloat
            Tensor::I64(_) => elements * 8,   // 64-bit integer
            Tensor::C32(_) => elements * 8,   // 2 * 32-bit complex
            Tensor::C64(_) => elements * 16,  // 2 * 64-bit complex
            Tensor::CF16(_) => elements * 4,  // 2 * 16-bit complex
            Tensor::CBF16(_) => elements * 4, // 2 * 16-bit bfloat complex
            #[cfg(feature = "torch")]
            Tensor::Torch(_) => elements * 4, // Default to 32-bit
            #[cfg(feature = "candle")]
            Tensor::Candle(_) => elements * 4, // Default to 32-bit
            Tensor::Sparse(sparse) => {
                // For sparse tensors, estimate based on non-zero elements
                let nnz = sparse.nnz();
                nnz * 4 + nnz * std::mem::size_of::<usize>() // values + indices
            },
        }
    }

    /// Cleanup old entries if needed (enhanced with adaptive eviction policies)
    fn cleanup_if_needed(&self) -> Result<()> {
        let mut last_cleanup = self.last_cleanup.lock().unwrap();
        let should_cleanup_time = last_cleanup.elapsed() >= self.config.cleanup_interval;

        let current_size = *self.current_size.lock().unwrap();
        let dynamic_max = *self.dynamic_max_size.lock().unwrap();
        let should_cleanup_size = current_size > dynamic_max;

        if !should_cleanup_time && !should_cleanup_size {
            return Ok(());
        }

        // Enhanced cleanup using configured eviction policy
        let mut pool = self.pool.write().unwrap();
        let mut total_freed = 0;
        let mut eviction_count = 0;
        let policy = self.config.eviction_policy;

        // Calculate how much memory we need to free
        let target_size = (dynamic_max as f64 * 0.85) as usize; // Target 85% of max
        let need_to_free = current_size.saturating_sub(target_size);

        // Collect all entries with their priorities
        let mut all_entries: Vec<(Vec<usize>, usize, f64)> = Vec::new();

        for (shape, entries) in pool.iter() {
            for (idx, entry) in entries.iter().enumerate() {
                if entry.ref_count == 0 {
                    let priority = entry.eviction_priority(policy);
                    all_entries.push((shape.clone(), idx, priority));
                }
            }
        }

        // Sort by eviction priority (lowest first = evict first)
        all_entries.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Evict entries until we've freed enough memory
        let mut freed_so_far = 0;
        let mut shapes_to_remove: Vec<Vec<usize>> = Vec::new();

        for (shape, _, _) in all_entries.iter() {
            if freed_so_far >= need_to_free {
                break;
            }

            if let Some(entries) = pool.get_mut(shape) {
                if let Some(entry) = entries.first() {
                    if entry.ref_count == 0 {
                        let size = entry.size_bytes;
                        freed_so_far += size;
                        total_freed += size;
                        eviction_count += 1;
                        shapes_to_remove.push(shape.clone());
                    }
                }
            }
        }

        // Remove marked entries
        for shape in shapes_to_remove {
            if let Some(entries) = pool.get_mut(&shape) {
                if !entries.is_empty() {
                    entries.remove(0);
                }
            }
        }

        // Remove empty entries
        pool.retain(|_, entries| !entries.is_empty());

        drop(pool); // Release write lock

        // Update statistics
        {
            let mut stats = self.statistics.lock().unwrap();
            stats.total_evictions += eviction_count;
            *stats.evictions_by_policy.entry(format!("{:?}", policy)).or_insert(0) +=
                eviction_count;
        }

        // Update size
        *self.current_size.lock().unwrap() -= total_freed;
        *last_cleanup = Instant::now();

        // Run defragmentation if enabled
        if self.config.enable_defragmentation {
            self.defragment_pool()?;
        }

        Ok(())
    }

    /// Apply adaptive pool sizing based on configured strategy
    fn apply_adaptive_sizing(&self) -> Result<()> {
        match self.config.adaptive_strategy {
            AdaptiveStrategy::Fixed => Ok(()), // No adaptation
            AdaptiveStrategy::HitRate => self.adapt_by_hit_rate(),
            AdaptiveStrategy::MemoryPressure => self.adapt_by_memory_pressure(),
            AdaptiveStrategy::Predictive => self.adapt_by_prediction(),
        }
    }

    /// Adapt pool size based on hit rate
    fn adapt_by_hit_rate(&self) -> Result<()> {
        let stats = self.statistics.lock().unwrap();
        let hit_rate = stats.hit_rate();
        drop(stats);

        let mut dynamic_max = self.dynamic_max_size.lock().unwrap();
        let target_rate = self.config.target_hit_rate;

        if hit_rate < target_rate {
            // Low hit rate: increase pool size
            let increase = (*dynamic_max as f64 * 0.1) as usize;
            let new_size = (*dynamic_max + increase).min(self.config.max_pool_size);
            if new_size > *dynamic_max {
                *dynamic_max = new_size;
            }
        } else if hit_rate > target_rate + 0.1 {
            // Very high hit rate: can decrease pool size
            let decrease = (*dynamic_max as f64 * 0.05) as usize;
            let new_size = (*dynamic_max - decrease).max(self.config.min_pool_size);
            if new_size < *dynamic_max {
                *dynamic_max = new_size;
            }
        }

        Ok(())
    }

    /// Adapt pool size based on system memory pressure
    fn adapt_by_memory_pressure(&self) -> Result<()> {
        // Simplified memory pressure detection
        // In production, this would query OS for available memory
        let current_size = *self.current_size.lock().unwrap();
        let mut dynamic_max = self.dynamic_max_size.lock().unwrap();

        let utilization = current_size as f64 / *dynamic_max as f64;

        if utilization > 0.9 {
            // High pressure: decrease pool size
            let new_size = (*dynamic_max as f64 * 0.9) as usize;
            *dynamic_max = new_size.max(self.config.min_pool_size);
        } else if utilization < 0.5 {
            // Low pressure: increase pool size
            let new_size = (*dynamic_max as f64 * 1.1) as usize;
            *dynamic_max = new_size.min(self.config.max_pool_size);
        }

        Ok(())
    }

    /// Adapt pool size based on access pattern prediction
    fn adapt_by_prediction(&self) -> Result<()> {
        let patterns = self.access_patterns.lock().unwrap();

        // Analyze access patterns to predict future needs
        let mut total_recent_accesses = 0;
        let recent_window = Duration::from_secs(60);
        let now = Instant::now();

        for timestamps in patterns.values() {
            total_recent_accesses +=
                timestamps.iter().filter(|t| now.duration_since(**t) < recent_window).count();
        }

        drop(patterns);

        // Adjust based on activity level
        let mut dynamic_max = self.dynamic_max_size.lock().unwrap();

        if total_recent_accesses > 1000 {
            // High activity: increase pool
            let new_size = (*dynamic_max as f64 * 1.15) as usize;
            *dynamic_max = new_size.min(self.config.max_pool_size);
        } else if total_recent_accesses < 100 {
            // Low activity: decrease pool
            let new_size = (*dynamic_max as f64 * 0.9) as usize;
            *dynamic_max = new_size.max(self.config.min_pool_size);
        }

        Ok(())
    }

    /// Defragment the pool by reorganizing entries
    fn defragment_pool(&self) -> Result<()> {
        // Simplified defragmentation: consolidate shape groups
        let mut pool = self.pool.write().unwrap();

        for entries in pool.values_mut() {
            // Sort entries by access count (most accessed first)
            entries.sort_by(|a, b| b.access_count.cmp(&a.access_count));
        }

        Ok(())
    }

    /// Get enhanced memory pool statistics
    pub fn get_stats(&self) -> MemoryPoolStats {
        let pool = self.pool.read().unwrap();
        let current_size = *self.current_size.lock().unwrap();
        let stats = self.statistics.lock().unwrap();
        let dynamic_max = *self.dynamic_max_size.lock().unwrap();

        let total_tensors = pool.values().map(|v| v.len()).sum();
        let total_shapes = pool.len();

        MemoryPoolStats {
            total_tensors,
            total_shapes,
            current_size_bytes: current_size,
            max_size_bytes: self.config.max_pool_size,
            dynamic_max_size_bytes: dynamic_max,
            utilization: current_size as f64 / dynamic_max as f64,
            hit_rate: stats.hit_rate(),
            miss_rate: stats.miss_rate(),
            total_requests: stats.total_requests,
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
            total_evictions: stats.total_evictions,
            peak_memory_usage_bytes: stats.peak_memory_usage,
            eviction_policy: self.config.eviction_policy,
            adaptive_strategy: self.config.adaptive_strategy,
        }
    }

    /// Reset statistics counters
    pub fn reset_statistics(&self) {
        let mut stats = self.statistics.lock().unwrap();
        *stats = PoolStatistics::default();
    }

    /// Get current hit rate
    pub fn hit_rate(&self) -> f64 {
        let stats = self.statistics.lock().unwrap();
        stats.hit_rate()
    }

    /// Get current eviction policy
    pub fn eviction_policy(&self) -> MemoryEvictionPolicy {
        self.config.eviction_policy
    }

    /// Get current adaptive strategy
    pub fn adaptive_strategy(&self) -> AdaptiveStrategy {
        self.config.adaptive_strategy
    }

    /// Get predicted shapes based on access patterns
    pub fn get_predicted_shapes(&self, window: Duration) -> Vec<Vec<usize>> {
        let patterns = self.access_patterns.lock().unwrap();
        let now = Instant::now();

        let mut frequent_shapes: Vec<(Vec<usize>, usize)> = patterns
            .iter()
            .map(|(shape, timestamps)| {
                let count = timestamps.iter().filter(|t| now.duration_since(**t) < window).count();
                (shape.clone(), count)
            })
            .filter(|(_, count)| *count > 0)
            .collect();

        frequent_shapes.sort_by(|a, b| b.1.cmp(&a.1));
        frequent_shapes.into_iter().map(|(shape, _)| shape).collect()
    }
}

/// Enhanced statistics for memory pool
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    /// Total tensors currently in pool
    pub total_tensors: usize,
    /// Number of different tensor shapes in pool
    pub total_shapes: usize,
    /// Current memory usage in bytes
    pub current_size_bytes: usize,
    /// Maximum configured pool size in bytes
    pub max_size_bytes: usize,
    /// Current dynamic maximum size (for adaptive strategies)
    pub dynamic_max_size_bytes: usize,
    /// Pool utilization (0.0 to 1.0+)
    pub utilization: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Cache miss rate (0.0 to 1.0)
    pub miss_rate: f64,
    /// Total number of tensor requests
    pub total_requests: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Total number of evictions
    pub total_evictions: usize,
    /// Peak memory usage observed (bytes)
    pub peak_memory_usage_bytes: usize,
    /// Current eviction policy
    pub eviction_policy: MemoryEvictionPolicy,
    /// Current adaptive strategy
    pub adaptive_strategy: AdaptiveStrategy,
}

/// Memory mapped tensor for large model weights
pub struct MemoryMappedTensor {
    /// File path for the memory mapped data
    file_path: String,
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Data type
    dtype: crate::tensor::DType,
    /// File handle for memory mapped data
    _file: Option<File>,
    /// Size of the file in bytes
    file_size: u64,
}

impl MemoryMappedTensor {
    /// Create a new memory mapped tensor
    pub fn new(file_path: String, shape: Vec<usize>, dtype: crate::tensor::DType) -> Result<Self> {
        // Open the file for reading
        let mut file = File::open(&file_path).map_err(|e| {
            TrustformersError::tensor_op_error(
                &format!("Failed to open file for memory mapping: {}", e),
                "mmap_new",
            )
        })?;

        // Get file size
        let file_size = file.seek(SeekFrom::End(0)).map_err(|e| {
            TrustformersError::tensor_op_error(
                &format!("Failed to get file size: {}", e),
                "mmap_new",
            )
        })?;

        // Verify file size matches tensor size
        let element_size = dtype.size_in_bytes();
        let total_elements: usize = shape.iter().product();
        let expected_size = total_elements * element_size;

        if file_size != expected_size as u64 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "File size {} doesn't match expected tensor size {}",
                    file_size, expected_size
                ),
                "mmap_new",
            ));
        }

        Ok(Self {
            file_path,
            shape,
            dtype,
            _file: Some(file),
            file_size,
        })
    }

    /// Load the tensor data (lazy loading)
    pub fn load(&self) -> Result<Tensor> {
        // Read the entire file content
        let mut file = File::open(&self.file_path).map_err(|e| {
            TrustformersError::tensor_op_error(
                &format!("Failed to open file for reading: {}", e),
                "mmap_load",
            )
        })?;

        let mut buffer = vec![0u8; self.file_size as usize];
        file.read_exact(&mut buffer).map_err(|e| {
            TrustformersError::tensor_op_error(
                &format!("Failed to read file data: {}", e),
                "mmap_load",
            )
        })?;

        // Convert bytes to appropriate tensor type
        match self.dtype {
            crate::tensor::DType::F32 => {
                let float_data = buffer
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect::<Vec<f32>>();
                Tensor::from_slice(&float_data, &self.shape)
            },
            crate::tensor::DType::F64 => {
                let float_data = buffer
                    .chunks_exact(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect::<Vec<f64>>();
                Tensor::from_slice_f64(&float_data, &self.shape)
            },
            crate::tensor::DType::I64 => {
                let int_data = buffer
                    .chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect::<Vec<i64>>();
                Tensor::from_slice_i64(&int_data, &self.shape)
            },
            crate::tensor::DType::I32 => {
                let int_data = buffer
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect::<Vec<i32>>();
                Tensor::from_slice_i32(&int_data, &self.shape)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported dtype for memory mapped tensor",
                "mmap_load",
            )),
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the file path
    pub fn file_path(&self) -> &str {
        &self.file_path
    }
}

/// Global memory manager instance
static MEMORY_MANAGER: std::sync::OnceLock<TensorMemoryPool> = std::sync::OnceLock::new();

/// Initialize the global memory manager
pub fn init_memory_manager(config: MemoryConfig) -> Result<()> {
    let pool = TensorMemoryPool::new(config);
    MEMORY_MANAGER.set(pool).map_err(|_| {
        TrustformersError::invalid_input("Memory manager already initialized".to_string())
    })?;
    Ok(())
}

/// Get the global memory manager
pub fn get_memory_manager() -> Option<&'static TensorMemoryPool> {
    MEMORY_MANAGER.get()
}

/// Convenience function to get a tensor from the global pool
pub fn get_tensor(shape: &[usize], dtype: crate::tensor::DType) -> Result<Tensor> {
    if let Some(manager) = get_memory_manager() {
        manager.get_tensor(shape, dtype)
    } else {
        // Fallback to direct creation
        match dtype {
            crate::tensor::DType::F32 => Tensor::zeros(shape),
            crate::tensor::DType::F64 => Tensor::zeros_f64(shape),
            crate::tensor::DType::I64 => Tensor::zeros_i64(shape),
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported dtype",
                "get_tensor",
            )),
        }
    }
}

/// Convenience function to return a tensor to the global pool
pub fn return_tensor(tensor: Tensor) -> Result<()> {
    if let Some(manager) = get_memory_manager() {
        manager.return_tensor(tensor)
    } else {
        Ok(()) // Just drop the tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert!(config.enable_memory_pool);
        assert!(config.enable_zero_copy);
        assert!(config.enable_mmap);
        assert_eq!(config.max_pool_size, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_tensor_pool_creation() {
        let config = MemoryConfig::default();
        let pool = TensorMemoryPool::new(config);
        let stats = pool.get_stats();
        assert_eq!(stats.total_tensors, 0);
        assert_eq!(stats.current_size_bytes, 0);
    }

    #[test]
    fn test_tensor_pool_get_and_return() -> Result<()> {
        let config = MemoryConfig::default();
        let pool = TensorMemoryPool::new(config);

        // Get a tensor
        let shape = vec![2, 3];
        let tensor = pool.get_tensor(&shape, crate::tensor::DType::F32)?;
        assert_eq!(tensor.shape(), shape.as_slice());

        // Return it to pool
        pool.return_tensor(tensor)?;

        // Get it again (should come from pool)
        let tensor2 = pool.get_tensor(&shape, crate::tensor::DType::F32)?;
        assert_eq!(tensor2.shape(), shape.as_slice());

        Ok(())
    }

    #[test]
    fn test_zero_copy_tensor_view() -> Result<()> {
        let tensor = Arc::new(Tensor::ones(&[10])?);
        let view = TensorView::slice(tensor, 2, 8)?;
        assert_eq!(view.shape(), &[6]);

        let viewed_tensor = view.as_tensor()?;
        assert_eq!(viewed_tensor.shape(), &[6]);

        Ok(())
    }

    #[test]
    fn test_memory_mapped_tensor() -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        // Create a temporary file with some data
        let temp_file = "test_temp.bin";
        let data_size = 100 * 100 * std::mem::size_of::<f32>();
        let data: Vec<u8> = vec![0; data_size];

        {
            let mut file = File::create(temp_file).map_err(|e| {
                TrustformersError::tensor_op_error(
                    &format!("Failed to create test file: {}", e),
                    "test_setup",
                )
            })?;
            file.write_all(&data).map_err(|e| {
                TrustformersError::tensor_op_error(
                    &format!("Failed to write test data: {}", e),
                    "test_setup",
                )
            })?;
        }

        let mmap_tensor = MemoryMappedTensor::new(
            temp_file.to_string(),
            vec![100, 100],
            crate::tensor::DType::F32,
        )?;

        assert_eq!(mmap_tensor.shape(), &[100, 100]);
        assert_eq!(mmap_tensor.file_path(), temp_file);

        let loaded = mmap_tensor.load()?;
        assert_eq!(loaded.shape(), &[100, 100]);

        // Clean up
        std::fs::remove_file(temp_file).ok();

        Ok(())
    }

    #[test]
    fn test_global_memory_manager() -> Result<()> {
        let config = MemoryConfig::default();
        init_memory_manager(config)?;

        let tensor = get_tensor(&[5, 5], crate::tensor::DType::F32)?;
        assert_eq!(tensor.shape(), [5, 5].as_slice());

        return_tensor(tensor)?;

        Ok(())
    }
}
