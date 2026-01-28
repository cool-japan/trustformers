//! Adaptive Cache Manager with LRU and Smart Eviction
//!
//! This module provides an advanced caching system for mobile deployment with:
//! - LRU (Least Recently Used) eviction
//! - Adaptive eviction based on memory pressure
//! - Frequency-based retention
//! - Size-aware caching
//! - Access pattern learning
//! - Multi-tier caching (memory, disk)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use trustformers_core::errors::{Result, TrustformersError};

/// Cache eviction strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionStrategy {
    /// Least Recently Used - evict oldest accessed items
    LRU,
    /// Least Frequently Used - evict least accessed items
    LFU,
    /// Adaptive - combine LRU and LFU based on access patterns
    Adaptive,
    /// Size-aware - prioritize smaller items when memory is tight
    SizeAware,
    /// Priority-based - use explicit priorities
    Priority,
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// The cached data
    pub data: T,
    /// Size in bytes
    pub size_bytes: usize,
    /// Last access time
    pub last_accessed: Instant,
    /// Access count
    pub access_count: u64,
    /// Priority (higher = more important to keep)
    pub priority: u32,
    /// Creation time
    pub created_at: Instant,
    /// Expiry time (None = never expires)
    pub expires_at: Option<Instant>,
}

impl<T> CacheEntry<T> {
    /// Create new cache entry
    pub fn new(data: T, size_bytes: usize) -> Self {
        let now = Instant::now();
        Self {
            data,
            size_bytes,
            last_accessed: now,
            access_count: 0,
            priority: 0,
            created_at: now,
            expires_at: None,
        }
    }

    /// Mark as accessed
    pub fn mark_accessed(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Check if expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Instant::now() > expires_at
        } else {
            false
        }
    }

    /// Get age in seconds
    pub fn age_seconds(&self) -> f64 {
        self.created_at.elapsed().as_secs_f64()
    }

    /// Get time since last access in seconds
    pub fn idle_seconds(&self) -> f64 {
        self.last_accessed.elapsed().as_secs_f64()
    }
}

/// Adaptive cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCacheConfig {
    /// Maximum memory budget in bytes
    pub max_memory_bytes: usize,

    /// Soft limit (trigger eviction) as fraction of max
    pub soft_limit_fraction: f32,

    /// Hard limit (aggressive eviction) as fraction of max
    pub hard_limit_fraction: f32,

    /// Eviction strategy
    pub eviction_strategy: EvictionStrategy,

    /// Enable access pattern learning
    pub learn_patterns: bool,

    /// Default entry TTL (time to live)
    pub default_ttl: Option<Duration>,

    /// Enable disk-backed cache
    pub disk_cache_enabled: bool,

    /// Disk cache directory
    pub disk_cache_dir: Option<PathBuf>,

    /// Maximum disk cache size
    pub max_disk_cache_bytes: usize,
}

impl Default for AdaptiveCacheConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 512 * 1024 * 1024, // 512MB
            soft_limit_fraction: 0.75,
            hard_limit_fraction: 0.9,
            eviction_strategy: EvictionStrategy::Adaptive,
            learn_patterns: true,
            default_ttl: Some(Duration::from_secs(3600)), // 1 hour
            disk_cache_enabled: false,
            disk_cache_dir: None,
            max_disk_cache_bytes: 2 * 1024 * 1024 * 1024, // 2GB
        }
    }
}

/// Access pattern statistics
#[derive(Debug, Clone)]
struct AccessPatternStats {
    /// Total accesses
    total_accesses: u64,
    /// Sequential access count
    sequential_count: u64,
    /// Random access count
    random_count: u64,
    /// Average access frequency
    avg_frequency: f64,
    /// Last access times (for detecting patterns)
    recent_accesses: VecDeque<Instant>,
}

impl Default for AccessPatternStats {
    fn default() -> Self {
        Self {
            total_accesses: 0,
            sequential_count: 0,
            random_count: 0,
            avg_frequency: 0.0,
            recent_accesses: VecDeque::with_capacity(100),
        }
    }
}

/// Adaptive cache manager
pub struct AdaptiveCacheManager<K: Hash + Eq + Clone, V: Clone> {
    /// Configuration
    config: AdaptiveCacheConfig,

    /// Cache storage
    cache: Arc<Mutex<HashMap<K, CacheEntry<V>>>>,

    /// LRU order (most recent at back)
    lru_order: Arc<Mutex<VecDeque<K>>>,

    /// Current memory usage
    current_memory: Arc<Mutex<usize>>,

    /// Access pattern statistics
    patterns: Arc<Mutex<HashMap<K, AccessPatternStats>>>,

    /// Cache hit/miss statistics
    stats: Arc<Mutex<CacheStats>>,
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Total items added
    pub inserts: u64,
    /// Current item count
    pub item_count: usize,
    /// Current memory usage
    pub memory_bytes: usize,
    /// Peak memory usage
    pub peak_memory_bytes: usize,
}

impl CacheStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Calculate average item size
    pub fn avg_item_size(&self) -> usize {
        if self.item_count == 0 {
            0
        } else {
            self.memory_bytes / self.item_count
        }
    }
}

impl<K: Hash + Eq + Clone, V: Clone> AdaptiveCacheManager<K, V> {
    /// Create new adaptive cache manager
    pub fn new(config: AdaptiveCacheConfig) -> Self {
        Self {
            config,
            cache: Arc::new(Mutex::new(HashMap::new())),
            lru_order: Arc::new(Mutex::new(VecDeque::new())),
            current_memory: Arc::new(Mutex::new(0)),
            patterns: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Get value from cache
    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.lock().expect("Lock poisoned");
        let mut stats = self.stats.lock().expect("Lock poisoned");

        if let Some(entry) = cache.get_mut(key) {
            // Check if expired
            if entry.is_expired() {
                drop(cache);
                drop(stats);
                self.remove(key);
                return None;
            }

            // Update access metadata
            entry.mark_accessed();

            // Update LRU order
            let mut lru = self.lru_order.lock().expect("Lock poisoned");
            if let Some(pos) = lru.iter().position(|k| k == key) {
                lru.remove(pos);
            }
            lru.push_back(key.clone());

            // Update access patterns
            if self.config.learn_patterns {
                self.update_access_pattern(key);
            }

            stats.hits += 1;
            Some(entry.data.clone())
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Insert value into cache
    pub fn insert(&self, key: K, value: V, size_bytes: usize) -> Result<()> {
        self.insert_with_priority(key, value, size_bytes, 0)
    }

    /// Insert value with priority
    pub fn insert_with_priority(
        &self,
        key: K,
        value: V,
        size_bytes: usize,
        priority: u32,
    ) -> Result<()> {
        // Check if we need to evict
        let mut current_mem = self.current_memory.lock().expect("Lock poisoned");
        let soft_limit =
            (self.config.max_memory_bytes as f32 * self.config.soft_limit_fraction) as usize;

        if *current_mem + size_bytes > soft_limit {
            drop(current_mem);
            self.evict_to_fit(size_bytes)?;
            current_mem = self.current_memory.lock().expect("Lock poisoned");
        }

        // Create cache entry
        let mut entry = CacheEntry::new(value, size_bytes);
        entry.priority = priority;

        if let Some(ttl) = self.config.default_ttl {
            entry.expires_at = Some(Instant::now() + ttl);
        }

        // Insert into cache
        let mut cache = self.cache.lock().expect("Lock poisoned");
        cache.insert(key.clone(), entry);

        // Update LRU order
        let mut lru = self.lru_order.lock().expect("Lock poisoned");
        lru.push_back(key.clone());

        // Update memory tracking
        *current_mem += size_bytes;

        // Update statistics
        let mut stats = self.stats.lock().expect("Lock poisoned");
        stats.inserts += 1;
        stats.item_count = cache.len();
        stats.memory_bytes = *current_mem;
        stats.peak_memory_bytes = stats.peak_memory_bytes.max(*current_mem);

        Ok(())
    }

    /// Remove value from cache
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.lock().expect("Lock poisoned");

        if let Some(entry) = cache.remove(key) {
            // Update LRU order
            let mut lru = self.lru_order.lock().expect("Lock poisoned");
            if let Some(pos) = lru.iter().position(|k| k == key) {
                lru.remove(pos);
            }

            // Update memory tracking
            let mut current_mem = self.current_memory.lock().expect("Lock poisoned");
            *current_mem = current_mem.saturating_sub(entry.size_bytes);

            // Update statistics
            let mut stats = self.stats.lock().expect("Lock poisoned");
            stats.item_count = cache.len();
            stats.memory_bytes = *current_mem;

            Some(entry.data)
        } else {
            None
        }
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        let mut cache = self.cache.lock().expect("Lock poisoned");
        cache.clear();

        let mut lru = self.lru_order.lock().expect("Lock poisoned");
        lru.clear();

        let mut current_mem = self.current_memory.lock().expect("Lock poisoned");
        *current_mem = 0;

        let mut stats = self.stats.lock().expect("Lock poisoned");
        stats.item_count = 0;
        stats.memory_bytes = 0;
    }

    /// Evict entries to fit new data
    fn evict_to_fit(&self, required_bytes: usize) -> Result<()> {
        let target_memory =
            (self.config.max_memory_bytes as f32 * self.config.soft_limit_fraction) as usize;
        let current_mem = *self.current_memory.lock().expect("Lock poisoned");

        if current_mem + required_bytes <= target_memory {
            return Ok(());
        }

        let to_free = (current_mem + required_bytes) - target_memory;
        let mut freed = 0;

        match self.config.eviction_strategy {
            EvictionStrategy::LRU => {
                freed = self.evict_lru(to_free)?;
            },
            EvictionStrategy::LFU => {
                freed = self.evict_lfu(to_free)?;
            },
            EvictionStrategy::Adaptive => {
                freed = self.evict_adaptive(to_free)?;
            },
            EvictionStrategy::SizeAware => {
                freed = self.evict_size_aware(to_free)?;
            },
            EvictionStrategy::Priority => {
                freed = self.evict_by_priority(to_free)?;
            },
        }

        if freed < to_free {
            return Err(TrustformersError::runtime_error(
                "Failed to free enough memory through eviction".to_string(),
            ));
        }

        Ok(())
    }

    /// LRU eviction - evict least recently used
    fn evict_lru(&self, to_free: usize) -> Result<usize> {
        let mut freed = 0;
        let mut stats = self.stats.lock().expect("Lock poisoned");

        while freed < to_free {
            let key = {
                let mut lru = self.lru_order.lock().expect("Lock poisoned");
                lru.pop_front()
            };

            if let Some(key) = key {
                // Get the size BEFORE removing the entry
                let entry_size = {
                    let cache = self.cache.lock().expect("Lock poisoned");
                    cache.get(&key).map(|e| e.size_bytes).unwrap_or(0)
                };

                drop(stats);
                if self.remove(&key).is_some() {
                    freed += entry_size;
                    stats = self.stats.lock().expect("Lock poisoned");
                    stats.evictions += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(freed)
    }

    /// LFU eviction - evict least frequently used
    fn evict_lfu(&self, to_free: usize) -> Result<usize> {
        let cache = self.cache.lock().expect("Lock poisoned");

        // Sort by access count (ascending)
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by_key(|(_, entry)| entry.access_count);

        // Collect keys to evict
        let mut keys_to_evict = Vec::new();
        let mut freed_estimate = 0;
        for (k, e) in entries {
            if freed_estimate >= to_free {
                break;
            }
            freed_estimate += e.size_bytes;
            keys_to_evict.push((k.clone(), e.size_bytes));
        }

        drop(cache);

        let mut freed = 0;
        for (key, size) in keys_to_evict.iter() {
            self.remove(key);
            freed += size;
        }

        // Update stats after all removals (avoid deadlock)
        let mut stats = self.stats.lock().expect("Lock poisoned");
        stats.evictions += keys_to_evict.len() as u64;

        Ok(freed)
    }

    /// Adaptive eviction - combine LRU and LFU based on access patterns
    fn evict_adaptive(&self, to_free: usize) -> Result<usize> {
        let cache = self.cache.lock().expect("Lock poisoned");

        // Calculate adaptive score: combines recency and frequency
        let mut entries: Vec<_> = cache
            .iter()
            .map(|(k, e)| {
                let recency_score = 1.0 / (1.0 + e.idle_seconds());
                let frequency_score = e.access_count as f64 / (1.0 + e.age_seconds());
                let priority_score = e.priority as f64 / 100.0;

                // Combined score (higher = more important to keep)
                let score = recency_score * 0.4 + frequency_score * 0.4 + priority_score * 0.2;
                (k, e, score)
            })
            .collect();

        // Sort by score (ascending - evict lowest scores first)
        entries
            .sort_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Collect keys to evict
        let mut keys_to_evict = Vec::new();
        let mut freed_estimate = 0;
        for (k, e, _) in entries {
            if freed_estimate >= to_free {
                break;
            }
            freed_estimate += e.size_bytes;
            keys_to_evict.push((k.clone(), e.size_bytes));
        }

        drop(cache);

        let mut freed = 0;
        for (key, size) in keys_to_evict.iter() {
            self.remove(key);
            freed += size;
        }

        // Update stats after all removals (avoid deadlock)
        let mut stats = self.stats.lock().expect("Lock poisoned");
        stats.evictions += keys_to_evict.len() as u64;

        Ok(freed)
    }

    /// Size-aware eviction - prioritize smaller items when under pressure
    fn evict_size_aware(&self, to_free: usize) -> Result<usize> {
        let cache = self.cache.lock().expect("Lock poisoned");

        // Prefer evicting larger items first to free more space quickly
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by_key(|(_, entry)| std::cmp::Reverse(entry.size_bytes));

        // Collect keys to evict
        let mut keys_to_evict = Vec::new();
        let mut freed_estimate = 0;
        for (k, e) in entries {
            if freed_estimate >= to_free {
                break;
            }
            freed_estimate += e.size_bytes;
            keys_to_evict.push((k.clone(), e.size_bytes));
        }

        drop(cache);

        let mut freed = 0;
        for (key, size) in keys_to_evict.iter() {
            self.remove(key);
            freed += size;
        }

        // Update stats after all removals (avoid deadlock)
        let mut stats = self.stats.lock().expect("Lock poisoned");
        stats.evictions += keys_to_evict.len() as u64;

        Ok(freed)
    }

    /// Priority-based eviction - evict lowest priority first
    fn evict_by_priority(&self, to_free: usize) -> Result<usize> {
        let cache = self.cache.lock().expect("Lock poisoned");

        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by_key(|(_, entry)| entry.priority);

        // Collect keys to evict
        let mut keys_to_evict = Vec::new();
        let mut freed_estimate = 0;
        for (k, e) in entries {
            if freed_estimate >= to_free {
                break;
            }
            freed_estimate += e.size_bytes;
            keys_to_evict.push((k.clone(), e.size_bytes));
        }

        drop(cache);

        let mut freed = 0;
        for (key, size) in keys_to_evict.iter() {
            self.remove(key);
            freed += size;
        }

        // Update stats after all removals (avoid deadlock)
        let mut stats = self.stats.lock().expect("Lock poisoned");
        stats.evictions += keys_to_evict.len() as u64;

        Ok(freed)
    }

    /// Update access pattern statistics
    fn update_access_pattern(&self, key: &K) {
        let mut patterns = self.patterns.lock().expect("Lock poisoned");
        let stats = patterns.entry(key.clone()).or_default();

        stats.total_accesses += 1;
        stats.recent_accesses.push_back(Instant::now());

        // Keep only recent 100 accesses
        if stats.recent_accesses.len() > 100 {
            stats.recent_accesses.pop_front();
        }

        // Update frequency
        if stats.recent_accesses.len() >= 2 {
            let time_span = stats
                .recent_accesses
                .back()
                .expect("Time went backwards")
                .duration_since(*stats.recent_accesses.front().expect("No recent accesses"))
                .as_secs_f64();
            stats.avg_frequency = stats.recent_accesses.len() as f64 / time_span.max(1.0);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().expect("Lock poisoned").clone()
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        *self.current_memory.lock().expect("Lock poisoned")
    }

    /// Check if cache contains key
    pub fn contains(&self, key: &K) -> bool {
        self.cache.lock().expect("Lock poisoned").contains_key(key)
    }

    /// Get cache size (number of entries)
    pub fn len(&self) -> usize {
        self.cache.lock().expect("Lock poisoned").len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_eviction() {
        let config = AdaptiveCacheConfig {
            max_memory_bytes: 1000,
            // Use 1.0 so eviction only happens when exceeding max_memory_bytes
            soft_limit_fraction: 1.0,
            eviction_strategy: EvictionStrategy::LRU,
            ..Default::default()
        };

        let cache: AdaptiveCacheManager<String, Vec<u8>> = AdaptiveCacheManager::new(config);

        // Insert items (300 * 3 = 900 bytes, under 1000 limit)
        cache.insert("a".to_string(), vec![0u8; 300], 300).expect("Insert failed");
        cache.insert("b".to_string(), vec![0u8; 300], 300).expect("Insert failed");
        cache.insert("c".to_string(), vec![0u8; 300], 300).expect("Insert failed");

        // This should trigger eviction of "a" (oldest) to make room
        // 900 + 300 = 1200 > 1000, need to free at least 200 bytes -> evict "a" (300 bytes)
        cache.insert("d".to_string(), vec![0u8; 300], 300).expect("Insert failed");

        assert!(!cache.contains(&"a".to_string()));
        assert!(cache.contains(&"b".to_string()));
        assert!(cache.contains(&"c".to_string()));
        assert!(cache.contains(&"d".to_string()));
    }

    #[test]
    fn test_adaptive_eviction() {
        let config = AdaptiveCacheConfig {
            max_memory_bytes: 1000,
            // Use 1.0 so eviction only happens when exceeding max_memory_bytes
            soft_limit_fraction: 1.0,
            eviction_strategy: EvictionStrategy::Adaptive,
            ..Default::default()
        };

        let cache: AdaptiveCacheManager<String, Vec<u8>> = AdaptiveCacheManager::new(config);

        // Insert items (300 * 3 = 900 bytes, under 1000 limit)
        cache.insert("a".to_string(), vec![0u8; 300], 300).expect("Insert failed");
        cache.insert("b".to_string(), vec![0u8; 300], 300).expect("Insert failed");
        cache.insert("c".to_string(), vec![0u8; 300], 300).expect("Insert failed");

        // Access "a" multiple times to increase its frequency
        for _ in 0..10 {
            cache.get(&"a".to_string());
        }

        // This should evict "b" or "c" (lower score), not "a"
        cache.insert("d".to_string(), vec![0u8; 300], 300).expect("Insert failed");

        assert!(cache.contains(&"a".to_string()));
        assert!(cache.contains(&"d".to_string()));
    }

    #[test]
    fn test_cache_stats() {
        let config = AdaptiveCacheConfig::default();
        let cache: AdaptiveCacheManager<String, String> = AdaptiveCacheManager::new(config);

        cache
            .insert("key1".to_string(), "value1".to_string(), 100)
            .expect("Insert failed");

        cache.get(&"key1".to_string());
        cache.get(&"key2".to_string());

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_priority_eviction() {
        let config = AdaptiveCacheConfig {
            max_memory_bytes: 1000,
            // Use 1.0 so eviction only happens when exceeding max_memory_bytes
            soft_limit_fraction: 1.0,
            eviction_strategy: EvictionStrategy::Priority,
            ..Default::default()
        };

        let cache: AdaptiveCacheManager<String, Vec<u8>> = AdaptiveCacheManager::new(config);

        // Insert with different priorities (300 * 3 = 900 bytes, under 1000 limit)
        cache
            .insert_with_priority("low".to_string(), vec![0u8; 300], 300, 1)
            .expect("Insert failed");
        cache
            .insert_with_priority("high".to_string(), vec![0u8; 300], 300, 10)
            .expect("Insert failed");
        cache
            .insert_with_priority("medium".to_string(), vec![0u8; 300], 300, 5)
            .expect("Insert failed");

        // This should evict "low" (lowest priority)
        cache
            .insert_with_priority("new".to_string(), vec![0u8; 300], 300, 5)
            .expect("Insert failed");

        assert!(!cache.contains(&"low".to_string()));
        assert!(cache.contains(&"high".to_string()));
        assert!(cache.contains(&"medium".to_string()));
    }
}
