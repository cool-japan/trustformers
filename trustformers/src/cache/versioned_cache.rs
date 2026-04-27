//! TTL-based and versioned model cache.
//!
//! Extends basic LRU/LFU caching with TTL expiration and model versioning.
//! Thread-safe via `Arc<RwLock<…>>` guards.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, trace};

// ─── CacheError ───────────────────────────────────────────────────────────────

/// Errors that can occur during cache operations
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// The cache has reached its entry or byte limit
    #[error("Cache is full: {0} entries, {1} bytes")]
    CacheFull(usize, usize),
    /// The entry is larger than the configured maximum
    #[error("Entry too large: {0} bytes exceeds max {1}")]
    EntryTooLarge(usize, usize),
    /// Lock poisoning (should never happen in well-behaved code)
    #[error("Cache lock poisoned")]
    LockPoisoned,
}

// ─── CacheEvictionPolicy ──────────────────────────────────────────────────────

/// Eviction strategy used when the cache is at capacity
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used — evict the entry that was accessed longest ago (default)
    #[default]
    Lru,
    /// Least Frequently Used — evict the entry with the fewest accesses
    Lfu,
    /// TTL only — only expired entries are evicted; refuse new entries if full
    Ttl,
    /// Evict the largest entry first to reclaim the most space
    Size,
}

// ─── VersionedCacheConfig ─────────────────────────────────────────────────────

/// Configuration for the versioned cache
#[derive(Debug, Clone)]
pub struct VersionedCacheConfig {
    /// Maximum number of entries (0 = unlimited)
    pub max_entries: usize,
    /// Maximum total size in bytes across all entries (0 = unlimited)
    pub max_size_bytes: usize,
    /// Default TTL applied to new entries (`None` = no expiration)
    pub default_ttl: Option<Duration>,
    /// Eviction strategy when capacity is exceeded
    pub eviction_policy: CacheEvictionPolicy,
}

impl Default for VersionedCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            max_size_bytes: 512 * 1024 * 1024, // 512 MB
            default_ttl: None,
            eviction_policy: CacheEvictionPolicy::Lru,
        }
    }
}

// ─── VersionedCacheEntry ──────────────────────────────────────────────────────

/// A single entry stored inside the versioned cache
#[derive(Debug)]
pub struct VersionedCacheEntry<V> {
    /// The cached value
    pub value: V,
    /// Semantic version string, e.g. `"1.0.0"` or a git revision
    pub version: String,
    /// When this entry was first inserted
    pub created_at: Instant,
    /// When this entry was last accessed via `get()`
    pub last_accessed: Instant,
    /// Number of times this entry has been successfully retrieved
    pub access_count: u64,
    /// Optional TTL; if `None`, the entry never expires
    pub ttl: Option<Duration>,
    /// Size in bytes reported by the inserter
    pub size_bytes: usize,
}

impl<V> VersionedCacheEntry<V> {
    /// Returns `true` when the entry has a TTL and it has elapsed
    pub fn is_expired(&self) -> bool {
        match self.ttl {
            None => false,
            Some(ttl) => self.created_at.elapsed() > ttl,
        }
    }

    /// Returns `true` when the entry is not expired **and** its version
    /// satisfies `required_version` (or no version requirement is given)
    pub fn is_valid_version(&self, required_version: Option<&str>) -> bool {
        match required_version {
            None => true,
            Some(req) => self.version == req,
        }
    }
}

// ─── VersionedCacheStats ──────────────────────────────────────────────────────

/// Snapshot of cache usage and hit/miss counters
#[derive(Debug, Clone, Default)]
pub struct VersionedCacheStats {
    /// Number of successful `get()` calls that returned a value
    pub hits: u64,
    /// Number of `get()` calls that returned `None`
    pub misses: u64,
    /// Number of entries that were evicted to make room for new ones
    pub evictions: u64,
    /// Number of entries removed because their TTL elapsed
    pub ttl_expirations: u64,
    /// Number of `get()` calls rejected due to version mismatch
    pub version_mismatches: u64,
    /// Total bytes currently occupied by live entries
    pub total_size_bytes: usize,
    /// Number of live (non-expired) entries
    pub entry_count: usize,
}

impl VersionedCacheStats {
    /// Hit rate in the range `[0.0, 1.0]`; returns `0.0` if no requests yet
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ─── VersionedCache ───────────────────────────────────────────────────────────

/// Thread-safe TTL/versioned cache
///
/// Supports LRU, LFU, TTL-only, and size-based eviction policies and
/// optional per-entry TTL overrides.
pub struct VersionedCache<K, V>
where
    K: Hash + Eq + Clone,
{
    entries: Arc<RwLock<HashMap<K, VersionedCacheEntry<V>>>>,
    /// Insertion/access order, used for LRU eviction (most-recent last)
    access_order: Arc<RwLock<Vec<K>>>,
    config: VersionedCacheConfig,
    stats: Arc<RwLock<VersionedCacheStats>>,
}

impl<K, V> VersionedCache<K, V>
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: Clone,
{
    /// Create a new cache with the given configuration
    pub fn new(config: VersionedCacheConfig) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(Vec::new())),
            config,
            stats: Arc::new(RwLock::new(VersionedCacheStats::default())),
        }
    }

    // ── Private helpers ────────────────────────────────────────────────────

    fn update_access_order(&self, key: &K) {
        if let Ok(mut order) = self.access_order.write() {
            order.retain(|k| k != key);
            order.push(key.clone());
        }
    }

    fn remove_from_access_order(&self, key: &K) {
        if let Ok(mut order) = self.access_order.write() {
            order.retain(|k| k != key);
        }
    }

    /// Evict one entry to make space, using the configured policy.
    /// Returns `true` when an entry was evicted.
    fn evict_one(&self) -> bool {
        let entries_guard = match self.entries.write() {
            Ok(g) => g,
            Err(_) => return false,
        };

        // First, try to remove an expired entry (always preferable)
        let expired_key: Option<K> = entries_guard
            .iter()
            .find(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone());

        if let Some(key) = expired_key {
            drop(entries_guard);
            self.remove_entry_internal(&key, true);
            return true;
        }

        // Otherwise apply the configured eviction policy
        let victim_key: Option<K> = match self.config.eviction_policy {
            CacheEvictionPolicy::Ttl => {
                // Only evict expired entries; we already checked above
                None
            }

            CacheEvictionPolicy::Lru => {
                let order = match self.access_order.read() {
                    Ok(g) => g,
                    Err(_) => return false,
                };
                order.first().cloned()
            }

            CacheEvictionPolicy::Lfu => entries_guard
                .iter()
                .min_by_key(|(_, e)| e.access_count)
                .map(|(k, _)| k.clone()),

            CacheEvictionPolicy::Size => entries_guard
                .iter()
                .max_by_key(|(_, e)| e.size_bytes)
                .map(|(k, _)| k.clone()),
        };

        drop(entries_guard);

        if let Some(key) = victim_key {
            self.remove_entry_internal(&key, false);
            true
        } else {
            false
        }
    }

    /// Remove an entry, updating stats and access order bookkeeping
    fn remove_entry_internal(&self, key: &K, is_ttl_expiry: bool) {
        let removed_size = {
            let mut entries = match self.entries.write() {
                Ok(g) => g,
                Err(_) => return,
            };
            entries.remove(key).map(|e| e.size_bytes).unwrap_or(0)
        };

        self.remove_from_access_order(key);

        if let Ok(mut stats) = self.stats.write() {
            if removed_size > 0 {
                stats.total_size_bytes = stats.total_size_bytes.saturating_sub(removed_size);
                stats.entry_count = stats.entry_count.saturating_sub(1);
                if is_ttl_expiry {
                    stats.ttl_expirations += 1;
                } else {
                    stats.evictions += 1;
                }
            }
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// Insert an entry.
    ///
    /// If `ttl_override` is `Some`, it takes precedence over `config.default_ttl`.
    /// Returns `CacheError::EntryTooLarge` if the entry exceeds `max_size_bytes`.
    /// Returns `CacheError::CacheFull` when `eviction_policy == Ttl` and the cache
    /// is full and no expired entries are available.
    pub fn insert(
        &self,
        key: K,
        value: V,
        version: impl Into<String>,
        size_bytes: usize,
        ttl_override: Option<Duration>,
    ) -> Result<(), CacheError> {
        // Validate size
        if self.config.max_size_bytes > 0 && size_bytes > self.config.max_size_bytes {
            return Err(CacheError::EntryTooLarge(
                size_bytes,
                self.config.max_size_bytes,
            ));
        }

        // Make room if needed
        loop {
            let (entry_count, total_bytes) = {
                let s = self.stats.read().map_err(|_| CacheError::LockPoisoned)?;
                (s.entry_count, s.total_size_bytes)
            };

            let over_entries = self.config.max_entries > 0 && entry_count >= self.config.max_entries;
            let over_bytes =
                self.config.max_size_bytes > 0 && total_bytes + size_bytes > self.config.max_size_bytes;

            if !over_entries && !over_bytes {
                break;
            }

            if !self.evict_one() {
                return Err(CacheError::CacheFull(entry_count, total_bytes));
            }
        }

        let effective_ttl = ttl_override.or(self.config.default_ttl);
        let now = Instant::now();
        let entry = VersionedCacheEntry {
            value,
            version: version.into(),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            ttl: effective_ttl,
            size_bytes,
        };

        {
            let mut entries = self.entries.write().map_err(|_| CacheError::LockPoisoned)?;
            // If key already existed, remove its old size from stats
            if let Some(old) = entries.remove(&key) {
                if let Ok(mut stats) = self.stats.write() {
                    stats.total_size_bytes =
                        stats.total_size_bytes.saturating_sub(old.size_bytes);
                    stats.entry_count = stats.entry_count.saturating_sub(1);
                }
                self.remove_from_access_order(&key);
            }
            entries.insert(key.clone(), entry);
        }

        self.update_access_order(&key);

        if let Ok(mut stats) = self.stats.write() {
            stats.total_size_bytes += size_bytes;
            stats.entry_count += 1;
        }

        trace!(key = ?key, "Inserted entry into versioned cache");
        Ok(())
    }

    /// Retrieve a value from the cache.
    ///
    /// Returns `None` if the key is absent, the entry is expired, or the
    /// version does not match `required_version`.
    pub fn get(&self, key: &K, required_version: Option<&str>) -> Option<V> {
        // Check existence and validity without holding the write lock
        let (value, valid) = {
            let entries = self.entries.read().ok()?;
            match entries.get(key) {
                None => (None, false),
                Some(entry) => {
                    if entry.is_expired() || !entry.is_valid_version(required_version) {
                        (None, false)
                    } else {
                        (Some(entry.value.clone()), true)
                    }
                }
            }
        };

        if !valid {
            // Determine why it was invalid for statistics
            let is_version_mismatch = {
                let entries = self.entries.read().ok()?;
                entries.get(key).is_some_and(|e| {
                    !e.is_expired() && !e.is_valid_version(required_version)
                })
            };

            if let Ok(mut stats) = self.stats.write() {
                stats.misses += 1;
                if is_version_mismatch {
                    stats.version_mismatches += 1;
                }
            }

            // Clean up expired entry eagerly
            {
                let expired = self
                    .entries
                    .read()
                    .ok()?
                    .get(key)
                    .is_some_and(|e| e.is_expired());
                if expired {
                    self.remove_entry_internal(key, true);
                }
            }

            return None;
        }

        // Update access metadata on the entry
        if let Ok(mut entries) = self.entries.write() {
            if let Some(entry) = entries.get_mut(key) {
                entry.last_accessed = Instant::now();
                entry.access_count += 1;
            }
        }

        self.update_access_order(key);

        if let Ok(mut stats) = self.stats.write() {
            stats.hits += 1;
        }

        debug!(key = ?key, "Cache hit");
        value
    }

    /// Returns `true` when the key exists and the entry has not expired
    pub fn contains(&self, key: &K) -> bool {
        self.entries
            .read()
            .ok()
            .and_then(|e| e.get(key).map(|entry| !entry.is_expired()))
            .unwrap_or(false)
    }

    /// Remove an entry; returns `true` if an entry was present and removed
    pub fn remove(&self, key: &K) -> bool {
        let existed = {
            let entries = match self.entries.read() {
                Ok(g) => g,
                Err(_) => return false,
            };
            entries.contains_key(key)
        };

        if existed {
            self.remove_entry_internal(key, false);
            // Correct the eviction counter (remove_entry_internal counts it as eviction)
            if let Ok(mut stats) = self.stats.write() {
                stats.evictions = stats.evictions.saturating_sub(1);
            }
        }

        existed
    }

    /// Evict all entries whose TTL has elapsed.
    ///
    /// Returns the number of entries removed.
    pub fn evict_expired(&self) -> usize {
        let expired_keys: Vec<K> = {
            let entries = match self.entries.read() {
                Ok(g) => g,
                Err(_) => return 0,
            };
            entries
                .iter()
                .filter(|(_, e)| e.is_expired())
                .map(|(k, _)| k.clone())
                .collect()
        };

        let count = expired_keys.len();
        for key in expired_keys {
            self.remove_entry_internal(&key, true);
        }
        count
    }

    /// Clear all entries from the cache, resetting statistics
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.write() {
            entries.clear();
        }
        if let Ok(mut order) = self.access_order.write() {
            order.clear();
        }
        if let Ok(mut stats) = self.stats.write() {
            *stats = VersionedCacheStats::default();
        }
    }

    /// Snapshot of current cache statistics
    pub fn stats(&self) -> VersionedCacheStats {
        self.stats
            .read()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Number of live (non-expired) entries
    pub fn len(&self) -> usize {
        self.entries
            .read()
            .map(|e| e.values().filter(|v| !v.is_expired()).count())
            .unwrap_or(0)
    }

    /// Returns `true` when there are no live entries
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all entries that have a specific version string.
    ///
    /// Returns the number of entries invalidated.
    pub fn invalidate_version(&self, version: &str) -> usize {
        let keys_to_remove: Vec<K> = {
            let entries = match self.entries.read() {
                Ok(g) => g,
                Err(_) => return 0,
            };
            entries
                .iter()
                .filter(|(_, e)| e.version == version)
                .map(|(k, _)| k.clone())
                .collect()
        };

        let count = keys_to_remove.len();
        for key in keys_to_remove {
            self.remove_entry_internal(&key, false);
            // Don't count as eviction — it's an intentional invalidation
            if let Ok(mut stats) = self.stats.write() {
                stats.evictions = stats.evictions.saturating_sub(1);
            }
        }
        count
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn default_cache() -> VersionedCache<String, Vec<u8>> {
        VersionedCache::new(VersionedCacheConfig::default())
    }

    fn small_cache(max_entries: usize) -> VersionedCache<String, Vec<u8>> {
        VersionedCache::new(VersionedCacheConfig {
            max_entries,
            max_size_bytes: 0, // unlimited bytes
            default_ttl: None,
            eviction_policy: CacheEvictionPolicy::Lru,
        })
    }

    #[test]
    fn test_insert_and_get() {
        let cache = default_cache();
        cache
            .insert("key1".to_string(), vec![1, 2, 3], "1.0", 3, None)
            .unwrap();
        let v = cache.get(&"key1".to_string(), None).unwrap();
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn test_miss_returns_none() {
        let cache = default_cache();
        assert!(cache.get(&"missing".to_string(), None).is_none());
    }

    #[test]
    fn test_version_mismatch_returns_none() {
        let cache = default_cache();
        cache
            .insert("k".to_string(), vec![0], "1.0", 1, None)
            .unwrap();
        assert!(cache.get(&"k".to_string(), Some("2.0")).is_none());
    }

    #[test]
    fn test_version_match_returns_value() {
        let cache = default_cache();
        cache
            .insert("k".to_string(), vec![42], "1.0", 1, None)
            .unwrap();
        let v = cache.get(&"k".to_string(), Some("1.0")).unwrap();
        assert_eq!(v, vec![42]);
    }

    #[test]
    fn test_ttl_expiry() {
        let cache: VersionedCache<String, u32> = VersionedCache::new(VersionedCacheConfig {
            default_ttl: Some(Duration::from_millis(10)),
            ..Default::default()
        });
        cache.insert("k".to_string(), 99, "1.0", 4, None).unwrap();

        thread::sleep(Duration::from_millis(20));

        assert!(cache.get(&"k".to_string(), None).is_none());
    }

    #[test]
    fn test_ttl_override_per_entry() {
        let cache: VersionedCache<String, u32> = VersionedCache::new(VersionedCacheConfig {
            default_ttl: Some(Duration::from_secs(3600)), // 1 hour default
            ..Default::default()
        });
        // Override with a very short TTL
        cache
            .insert(
                "k".to_string(),
                99,
                "1.0",
                4,
                Some(Duration::from_millis(10)),
            )
            .unwrap();

        thread::sleep(Duration::from_millis(25));
        assert!(cache.get(&"k".to_string(), None).is_none());
    }

    #[test]
    fn test_contains() {
        let cache = default_cache();
        cache
            .insert("k".to_string(), vec![], "v1", 0, None)
            .unwrap();
        assert!(cache.contains(&"k".to_string()));
        assert!(!cache.contains(&"other".to_string()));
    }

    #[test]
    fn test_remove() {
        let cache = default_cache();
        cache
            .insert("k".to_string(), vec![1], "v1", 1, None)
            .unwrap();
        assert!(cache.remove(&"k".to_string()));
        assert!(!cache.contains(&"k".to_string()));
        assert!(!cache.remove(&"k".to_string())); // second remove returns false
    }

    #[test]
    fn test_evict_expired() {
        let cache: VersionedCache<String, u32> = VersionedCache::new(VersionedCacheConfig {
            default_ttl: Some(Duration::from_millis(10)),
            ..Default::default()
        });
        cache.insert("a".to_string(), 1, "v", 4, None).unwrap();
        cache.insert("b".to_string(), 2, "v", 4, None).unwrap();
        // c has no TTL
        cache
            .insert(
                "c".to_string(),
                3,
                "v",
                4,
                Some(Duration::from_secs(3600)),
            )
            .unwrap();

        thread::sleep(Duration::from_millis(25));
        let evicted = cache.evict_expired();
        assert_eq!(evicted, 2);
        assert!(cache.contains(&"c".to_string()));
    }

    #[test]
    fn test_clear() {
        let cache = default_cache();
        cache
            .insert("a".to_string(), vec![1], "v1", 1, None)
            .unwrap();
        cache
            .insert("b".to_string(), vec![2], "v1", 1, None)
            .unwrap();
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().entry_count, 0);
    }

    #[test]
    fn test_hit_rate() {
        let cache = default_cache();
        cache
            .insert("k".to_string(), vec![1], "v1", 1, None)
            .unwrap();

        cache.get(&"k".to_string(), None);
        cache.get(&"k".to_string(), None);
        cache.get(&"miss".to_string(), None);

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = small_cache(2);
        cache
            .insert("a".to_string(), vec![1], "v1", 1, None)
            .unwrap();
        cache
            .insert("b".to_string(), vec![2], "v1", 1, None)
            .unwrap();
        // Access "a" to make it most-recently-used
        cache.get(&"a".to_string(), None);
        // Insert "c" — should evict "b" (LRU)
        cache
            .insert("c".to_string(), vec![3], "v1", 1, None)
            .unwrap();

        assert!(cache.contains(&"a".to_string()));
        assert!(cache.contains(&"c".to_string()));
        assert!(!cache.contains(&"b".to_string()));
    }

    #[test]
    fn test_entry_too_large() {
        let cache: VersionedCache<String, Vec<u8>> = VersionedCache::new(VersionedCacheConfig {
            max_size_bytes: 10,
            ..Default::default()
        });
        let result = cache.insert("big".to_string(), vec![0; 100], "v1", 100, None);
        assert!(matches!(result, Err(CacheError::EntryTooLarge(100, 10))));
    }

    #[test]
    fn test_invalidate_version() {
        let cache = default_cache();
        cache
            .insert("a".to_string(), vec![1], "1.0", 1, None)
            .unwrap();
        cache
            .insert("b".to_string(), vec![2], "1.0", 1, None)
            .unwrap();
        cache
            .insert("c".to_string(), vec![3], "2.0", 1, None)
            .unwrap();

        let invalidated = cache.invalidate_version("1.0");
        assert_eq!(invalidated, 2);
        assert!(!cache.contains(&"a".to_string()));
        assert!(!cache.contains(&"b".to_string()));
        assert!(cache.contains(&"c".to_string()));
    }

    #[test]
    fn test_len_and_is_empty() {
        let cache = default_cache();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache
            .insert("x".to_string(), vec![1], "v", 1, None)
            .unwrap();
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_stats_version_mismatch_counter() {
        let cache = default_cache();
        cache
            .insert("k".to_string(), vec![1], "1.0", 1, None)
            .unwrap();
        cache.get(&"k".to_string(), Some("2.0")); // mismatch
        let stats = cache.stats();
        assert_eq!(stats.version_mismatches, 1);
    }

    #[test]
    fn test_overwrite_same_key() {
        let cache = default_cache();
        cache
            .insert("k".to_string(), vec![1], "1.0", 1, None)
            .unwrap();
        cache
            .insert("k".to_string(), vec![2], "2.0", 1, None)
            .unwrap();
        let v = cache.get(&"k".to_string(), Some("2.0")).unwrap();
        assert_eq!(v, vec![2]);
        // Stats entry count should still be 1 after overwrite
        assert_eq!(cache.stats().entry_count, 1);
    }

    #[test]
    fn test_lfu_eviction() {
        let cache: VersionedCache<String, u32> = VersionedCache::new(VersionedCacheConfig {
            max_entries: 2,
            eviction_policy: CacheEvictionPolicy::Lfu,
            ..Default::default()
        });
        cache.insert("a".to_string(), 1, "v", 4, None).unwrap();
        cache.insert("b".to_string(), 2, "v", 4, None).unwrap();
        // Access "a" twice, "b" zero times
        cache.get(&"a".to_string(), None);
        cache.get(&"a".to_string(), None);
        // Insert "c" — should evict "b" (LFU with 0 accesses)
        cache.insert("c".to_string(), 3, "v", 4, None).unwrap();

        assert!(cache.contains(&"a".to_string()));
        assert!(cache.contains(&"c".to_string()));
        assert!(!cache.contains(&"b".to_string()));
    }

    #[test]
    fn test_size_eviction_policy() {
        let cache: VersionedCache<String, Vec<u8>> = VersionedCache::new(VersionedCacheConfig {
            max_entries: 3,
            max_size_bytes: 1000,
            eviction_policy: CacheEvictionPolicy::Size,
            ..Default::default()
        });
        cache
            .insert("small".to_string(), vec![0; 10], "v", 10, None)
            .unwrap();
        cache
            .insert("large".to_string(), vec![0; 500], "v", 500, None)
            .unwrap();
        cache
            .insert("medium".to_string(), vec![0; 100], "v", 100, None)
            .unwrap();
        // Insert something that forces eviction (total would exceed 1000)
        cache
            .insert("new".to_string(), vec![0; 400], "v", 400, None)
            .unwrap();

        // "large" (500 bytes) should have been evicted first
        assert!(!cache.contains(&"large".to_string()));
    }
}
