use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{
    cache_key::CacheKey,
    eviction::{EvictionPolicy, LRUEviction, SizeBasedEviction, TTLEviction},
    metrics::CacheMetrics,
};

/// Configuration for the inference cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries (for LRU)
    pub max_entries: Option<usize>,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: Option<usize>,
    /// Time-to-live for entries
    pub ttl: Option<Duration>,
    /// Whether to enable metrics collection
    pub enable_metrics: bool,
    /// Whether to compress cached values
    pub compress_values: bool,
    /// Minimum value size to compress (bytes)
    pub compression_threshold: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: Some(1000),
            max_memory_bytes: Some(1024 * 1024 * 1024), // 1GB
            ttl: Some(Duration::from_secs(3600)),       // 1 hour
            enable_metrics: true,
            compress_values: true,
            compression_threshold: 1024, // 1KB
        }
    }
}

/// A cached entry in the inference cache
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cached value (potentially compressed)
    pub value: Vec<u8>,
    /// Size of the uncompressed value
    pub uncompressed_size: usize,
    /// Whether the value is compressed
    pub is_compressed: bool,
    /// When the entry was created
    pub created_at: Instant,
    /// When the entry was last accessed
    pub last_accessed: Instant,
    /// Number of times accessed
    pub access_count: u64,
}

impl CacheEntry {
    fn new(value: Vec<u8>, is_compressed: bool, uncompressed_size: usize) -> Self {
        let now = Instant::now();
        Self {
            value,
            uncompressed_size,
            is_compressed,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }

    fn access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    fn memory_size(&self) -> usize {
        self.value.len() + std::mem::size_of::<Self>()
    }
}

/// Thread-safe inference cache with multiple eviction policies
pub struct InferenceCache {
    /// The main cache storage
    cache: Arc<DashMap<CacheKey, CacheEntry>>,
    /// Eviction policy
    eviction_policy: Arc<parking_lot::Mutex<Box<dyn EvictionPolicy>>>,
    /// Cache configuration
    config: CacheConfig,
    /// Metrics collector
    metrics: Option<Arc<CacheMetrics>>,
}

impl InferenceCache {
    /// Create a new inference cache with the given configuration
    pub fn new(config: CacheConfig) -> Self {
        // Create composite eviction policy based on config
        let eviction_policy = Self::create_eviction_policy(&config);

        let metrics =
            if config.enable_metrics { Some(Arc::new(CacheMetrics::new())) } else { None };

        Self {
            cache: Arc::new(DashMap::new()),
            eviction_policy: Arc::new(parking_lot::Mutex::new(eviction_policy)),
            config,
            metrics,
        }
    }

    fn create_eviction_policy(config: &CacheConfig) -> Box<dyn EvictionPolicy> {
        // Use size-based eviction if memory limit is set
        if let Some(max_bytes) = config.max_memory_bytes {
            Box::new(SizeBasedEviction::new(max_bytes))
        }
        // Otherwise use LRU if entry limit is set
        else if let Some(max_entries) = config.max_entries {
            Box::new(LRUEviction::new(max_entries))
        }
        // Otherwise use TTL if set
        else if let Some(ttl) = config.ttl {
            Box::new(TTLEviction::new(ttl))
        }
        // Default to LRU with 1000 entries
        else {
            Box::new(LRUEviction::new(1000))
        }
    }

    /// Get a value from the cache
    pub fn get(&self, key: &CacheKey) -> Option<Vec<u8>> {
        let start = Instant::now();

        if let Some(mut entry) = self.cache.get_mut(key) {
            entry.access();
            let value = entry.value.clone();
            let is_compressed = entry.is_compressed;
            drop(entry); // Release lock

            // Update eviction policy
            self.eviction_policy.lock().on_access(key);

            // Decompress if needed
            let result = if is_compressed { self.decompress(&value).ok() } else { Some(value) };

            // Record metrics
            if let Some(metrics) = &self.metrics {
                let elapsed = start.elapsed();
                if result.is_some() {
                    metrics.record_hit(elapsed);
                } else {
                    metrics.record_miss(elapsed);
                }
            }

            result
        } else {
            if let Some(metrics) = &self.metrics {
                metrics.record_miss(start.elapsed());
            }
            None
        }
    }

    /// Insert a value into the cache
    pub fn insert(&self, key: CacheKey, value: Vec<u8>) {
        let start = Instant::now();
        let uncompressed_size = value.len();

        // Compress if enabled and above threshold
        let (stored_value, is_compressed) = if self.config.compress_values
            && uncompressed_size >= self.config.compression_threshold
        {
            match self.compress(&value) {
                Ok(compressed) if compressed.len() < uncompressed_size => (compressed, true),
                _ => (value, false),
            }
        } else {
            (value, false)
        };

        let entry = CacheEntry::new(stored_value, is_compressed, uncompressed_size);
        let memory_size = entry.memory_size();

        // Insert the entry
        self.cache.insert(key.clone(), entry);

        // Update eviction policy
        self.eviction_policy.lock().on_insert(&key, memory_size);

        // Check if we need to evict after insertion
        self.maybe_evict();

        // Record metrics
        if let Some(metrics) = &self.metrics {
            metrics.record_insert(memory_size, start.elapsed());
        }
    }

    /// Remove a value from the cache
    pub fn remove(&self, key: &CacheKey) -> Option<Vec<u8>> {
        if let Some((_, entry)) = self.cache.remove(key) {
            let memory_size = entry.memory_size();

            // Update eviction policy
            self.eviction_policy.lock().on_remove(key);

            // Record metrics
            if let Some(metrics) = &self.metrics {
                metrics.record_eviction(memory_size);
            }

            // Decompress if needed
            if entry.is_compressed {
                self.decompress(&entry.value).ok()
            } else {
                Some(entry.value)
            }
        } else {
            None
        }
    }

    /// Clear all entries from the cache
    pub fn clear(&self) {
        self.cache.clear();

        if let Some(metrics) = &self.metrics {
            metrics.reset();
        }
    }

    /// Get the number of entries in the cache
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get cache metrics if enabled
    pub fn metrics(&self) -> Option<Arc<CacheMetrics>> {
        self.metrics.clone()
    }

    /// Helper method to handle eviction for a specific key
    fn handle_eviction(&self, key: &CacheKey) {
        if let Some((_, entry)) = self.cache.remove(key) {
            // Note: Don't call policy.on_remove() here since next_eviction
            // already removed it from the policy's tracking
            if let Some(metrics) = &self.metrics {
                metrics.record_eviction(entry.memory_size());
            }
        }
    }

    /// Perform eviction if necessary
    fn maybe_evict(&self) {
        let mut policy = self.eviction_policy.lock();

        while policy.should_evict() {
            if let Some(key) = policy.next_eviction() {
                self.handle_eviction(&key);
            } else {
                break;
            }
        }
    }

    /// Compress data using zstd
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
        use std::io::Write;
        let mut encoder = zstd::Encoder::new(Vec::new(), 3)?;
        encoder.write_all(data)?;
        encoder.finish()
    }

    /// Decompress data using zstd
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
        zstd::decode_all(data)
    }
}

/// Builder for creating inference caches with custom configuration
pub struct InferenceCacheBuilder {
    config: CacheConfig,
}

impl InferenceCacheBuilder {
    pub fn new() -> Self {
        Self {
            config: CacheConfig::default(),
        }
    }

    pub fn max_entries(mut self, max_entries: usize) -> Self {
        self.config.max_entries = Some(max_entries);
        // Clear memory limit to ensure LRU eviction is used
        self.config.max_memory_bytes = None;
        self
    }

    pub fn max_memory_mb(mut self, max_memory_mb: usize) -> Self {
        self.config.max_memory_bytes = Some(max_memory_mb * 1024 * 1024);
        // Clear entry limit to ensure size-based eviction is used
        self.config.max_entries = None;
        self
    }

    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.config.ttl = Some(ttl);
        self
    }

    pub fn enable_metrics(mut self, enable: bool) -> Self {
        self.config.enable_metrics = enable;
        self
    }

    pub fn enable_compression(mut self, enable: bool) -> Self {
        self.config.compress_values = enable;
        self
    }

    pub fn compression_threshold(mut self, threshold: usize) -> Self {
        self.config.compression_threshold = threshold;
        self
    }

    pub fn build(self) -> InferenceCache {
        InferenceCache::new(self.config)
    }
}

impl Default for InferenceCacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::cache_key::CacheKeyBuilder;

    #[test]
    fn test_basic_cache_operations() {
        let cache = InferenceCacheBuilder::new().max_entries(10).enable_metrics(true).build();

        let key = CacheKeyBuilder::new("test-model", "classification")
            .with_text("Hello world")
            .build();

        let value = b"prediction result".to_vec();

        // Test insert and get
        cache.insert(key.clone(), value.clone());
        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved, value);

        // Test metrics
        let metrics = cache.metrics().unwrap();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.hits, 1);
        assert_eq!(snapshot.misses, 0);
        assert_eq!(snapshot.total_entries, 1);
    }

    #[test]
    fn test_compression() {
        let cache = InferenceCacheBuilder::new()
            .enable_compression(true)
            .compression_threshold(10)
            .build();

        let key = CacheKeyBuilder::new("test-model", "generation")
            .with_text("Test prompt")
            .build();

        // Create a large value that should be compressed
        let value = vec![42u8; 1000];

        cache.insert(key.clone(), value.clone());
        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved, value);

        // Check that the stored value is smaller (compressed)
        let entry = cache.cache.get(&key).unwrap();
        assert!(entry.is_compressed);
        assert!(entry.value.len() < entry.uncompressed_size);
    }

    #[test]
    fn test_eviction() {
        let cache = InferenceCacheBuilder::new().max_entries(3).enable_metrics(true).build();

        let keys: Vec<_> = (0..5)
            .map(|i| CacheKeyBuilder::new("model", "task").with_text(&format!("text{}", i)).build())
            .collect();

        // Insert more entries than the cache can hold
        for (i, key) in keys.iter().enumerate() {
            cache.insert(key.clone(), vec![i as u8; 100]);
        }

        // First two entries should have been evicted
        assert!(cache.get(&keys[0]).is_none());
        assert!(cache.get(&keys[1]).is_none());

        // Last three should still be present
        assert!(cache.get(&keys[2]).is_some());
        assert!(cache.get(&keys[3]).is_some());
        assert!(cache.get(&keys[4]).is_some());

        // Check eviction metrics
        let metrics = cache.metrics().unwrap();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.evictions, 2);
    }
}
