//! Generic LRU cache with hit/miss/eviction statistics.
//!
//! Provides an O(n) LRU eviction implementation using a `HashMap` for O(1)
//! lookups and a `VecDeque` to track recency order (front = most-recently-used,
//! back = least-recently-used).

use std::collections::{HashMap, VecDeque};

// ── LruCacheStats ────────────────────────────────────────────────────────────────

/// Aggregate statistics for an `LruCache` instance.
#[derive(Debug, Clone, Default)]
pub struct LruCacheStats {
    /// Number of successful cache lookups.
    pub hits: u64,
    /// Number of unsuccessful cache lookups.
    pub misses: u64,
    /// Number of entries that have been evicted to make room for new ones.
    pub evictions: u64,
    /// Approximate total bytes stored (user-managed; not automatically updated).
    pub total_bytes: usize,
}

impl LruCacheStats {
    /// Hit rate as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no lookups have been performed.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Total number of lookup attempts (`hits + misses`).
    pub fn total_requests(&self) -> u64 {
        self.hits + self.misses
    }
}

// ── LruCache ──────────────────────────────────────────────────────────────────

/// A bounded, generic LRU cache keyed by `String`.
///
/// # Ordering invariant
/// `order.front()` is the most-recently-used key; `order.back()` is the
/// least-recently-used key and the next candidate for eviction.
pub struct LruCache<V> {
    capacity: usize,
    map: HashMap<String, V>,
    /// Recency order: front = MRU, back = LRU.
    order: VecDeque<String>,
    stats: LruCacheStats,
}

impl<V: Clone> LruCache<V> {
    /// Create a new empty cache with the given capacity.
    ///
    /// Panics if `capacity` is 0.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "LruCache capacity must be at least 1");
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            stats: LruCacheStats::default(),
        }
    }

    /// Look up a value by key, promoting it to most-recently-used on a hit.
    pub fn get(&mut self, key: &str) -> Option<V> {
        if let Some(value) = self.map.get(key).cloned() {
            // Promote to MRU: remove from current position and push to front.
            self.order.retain(|k| k != key);
            self.order.push_front(key.to_string());
            self.stats.hits += 1;
            Some(value)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert or update a key-value pair.
    ///
    /// If the key already exists its value is replaced in-place and it becomes
    /// the most-recently-used entry.  If the cache is at capacity, the
    /// least-recently-used entry is evicted first.
    pub fn put(&mut self, key: String, value: V) {
        if self.map.contains_key(&key) {
            // Update existing: move to MRU.
            self.order.retain(|k| k != &key);
            self.order.push_front(key.clone());
            self.map.insert(key, value);
        } else {
            // New key: evict LRU if at capacity.
            if self.map.len() >= self.capacity {
                self.evict();
            }
            self.order.push_front(key.clone());
            self.map.insert(key, value);
        }
    }

    /// Manually evict the least-recently-used entry.
    ///
    /// Returns the evicted `(key, value)` pair, or `None` if the cache is empty.
    pub fn evict(&mut self) -> Option<(String, V)> {
        if let Some(lru_key) = self.order.pop_back() {
            if let Some(value) = self.map.remove(&lru_key) {
                self.stats.evictions += 1;
                return Some((lru_key, value));
            }
        }
        None
    }

    /// Reference to the current statistics snapshot.
    pub fn stats(&self) -> &LruCacheStats {
        &self.stats
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns `true` if `len() >= capacity`.
    pub fn is_full(&self) -> bool {
        self.map.len() >= self.capacity
    }

    /// The maximum number of entries this cache can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. LruCache::new starts empty
    #[test]
    fn test_lru_new_is_empty() {
        let cache: LruCache<i32> = LruCache::new(4);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // 2. LruCache::new has correct capacity
    #[test]
    fn test_lru_new_capacity() {
        let cache: LruCache<i32> = LruCache::new(8);
        assert_eq!(cache.capacity(), 8);
    }

    // 3. put single item, len becomes 1
    #[test]
    fn test_lru_put_one_item() {
        let mut cache: LruCache<i32> = LruCache::new(4);
        cache.put("a".to_string(), 1);
        assert_eq!(cache.len(), 1);
    }

    // 4. get existing key returns Some(value)
    #[test]
    fn test_lru_get_existing_key() {
        let mut cache: LruCache<i32> = LruCache::new(4);
        cache.put("a".to_string(), 42);
        let v = cache.get("a");
        assert_eq!(v, Some(42));
    }

    // 5. get missing key returns None
    #[test]
    fn test_lru_get_missing_key() {
        let mut cache: LruCache<i32> = LruCache::new(4);
        assert_eq!(cache.get("nonexistent"), None);
    }

    // 6. get increments hit count
    #[test]
    fn test_lru_get_increments_hits() {
        let mut cache: LruCache<i32> = LruCache::new(4);
        cache.put("k".to_string(), 1);
        cache.get("k");
        cache.get("k");
        assert_eq!(cache.stats().hits, 2);
    }

    // 7. get on missing key increments miss count
    #[test]
    fn test_lru_get_missing_increments_misses() {
        let mut cache: LruCache<i32> = LruCache::new(4);
        cache.get("missing");
        assert_eq!(cache.stats().misses, 1);
    }

    // 8. hit_rate returns 0 with no requests
    #[test]
    fn test_cache_stats_hit_rate_no_requests() {
        let stats = LruCacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    // 9. hit_rate returns 1.0 with all hits
    #[test]
    fn test_cache_stats_hit_rate_all_hits() {
        let stats = LruCacheStats { hits: 5, misses: 0, evictions: 0, total_bytes: 0 };
        assert!((stats.hit_rate() - 1.0).abs() < 1e-9);
    }

    // 10. hit_rate returns 0.5 with equal hits and misses
    #[test]
    fn test_cache_stats_hit_rate_half() {
        let stats = LruCacheStats { hits: 4, misses: 4, evictions: 0, total_bytes: 0 };
        assert!((stats.hit_rate() - 0.5).abs() < 1e-9);
    }

    // 11. total_requests = hits + misses
    #[test]
    fn test_cache_stats_total_requests() {
        let stats = LruCacheStats { hits: 3, misses: 7, evictions: 0, total_bytes: 0 };
        assert_eq!(stats.total_requests(), 10);
    }

    // 12. put overwrites existing key (no eviction, no len change)
    #[test]
    fn test_lru_put_overwrites_existing() {
        let mut cache: LruCache<i32> = LruCache::new(4);
        cache.put("k".to_string(), 1);
        cache.put("k".to_string(), 99);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get("k"), Some(99));
        // No evictions should have happened.
        assert_eq!(cache.stats().evictions, 0);
    }

    // 13. put beyond capacity evicts LRU
    #[test]
    fn test_lru_put_beyond_capacity_evicts() {
        let mut cache: LruCache<i32> = LruCache::new(2);
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);
        cache.put("c".to_string(), 3); // evicts "a" (LRU)
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.get("a"), None);
        assert_eq!(cache.get("b"), Some(2));
        assert_eq!(cache.get("c"), Some(3));
    }

    // 14. eviction increments evictions counter
    #[test]
    fn test_lru_eviction_increments_counter() {
        let mut cache: LruCache<i32> = LruCache::new(1);
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2); // evicts "a"
        assert_eq!(cache.stats().evictions, 1);
    }

    // 15. get promotes item to most-recently-used position
    #[test]
    fn test_lru_get_promotes_to_mru() {
        let mut cache: LruCache<i32> = LruCache::new(3);
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);
        cache.put("c".to_string(), 3);
        // Access "a" so it becomes MRU.
        cache.get("a");
        // Now add "d": should evict "b" (the new LRU after "a" was promoted).
        cache.put("d".to_string(), 4);
        assert_eq!(cache.get("a"), Some(1), "a must still be present");
        assert_eq!(cache.get("b"), None, "b should have been evicted");
    }

    // 16. after get, the accessed item survives next eviction
    #[test]
    fn test_lru_accessed_item_survives_eviction() {
        let mut cache: LruCache<i32> = LruCache::new(2);
        cache.put("x".to_string(), 10);
        cache.put("y".to_string(), 20);
        // Access "x" to make it MRU.
        cache.get("x");
        // Insert "z"; "y" should be evicted (LRU).
        cache.put("z".to_string(), 30);
        assert_eq!(cache.get("x"), Some(10), "x must survive");
        assert_eq!(cache.get("y"), None, "y must be evicted");
        assert_eq!(cache.get("z"), Some(30));
    }

    // 17. evict on empty cache returns None
    #[test]
    fn test_lru_evict_empty_returns_none() {
        let mut cache: LruCache<i32> = LruCache::new(4);
        assert_eq!(cache.evict(), None);
    }

    // 18. is_full returns false when below capacity
    #[test]
    fn test_lru_is_full_false_below_capacity() {
        let mut cache: LruCache<i32> = LruCache::new(3);
        cache.put("a".to_string(), 1);
        assert!(!cache.is_full());
    }

    // 19. is_full returns true when at capacity
    #[test]
    fn test_lru_is_full_true_at_capacity() {
        let mut cache: LruCache<i32> = LruCache::new(2);
        cache.put("a".to_string(), 1);
        cache.put("b".to_string(), 2);
        assert!(cache.is_full());
    }

    // 20. is_empty returns true initially
    #[test]
    fn test_lru_is_empty_initially() {
        let cache: LruCache<String> = LruCache::new(4);
        assert!(cache.is_empty());
    }

    // 21. is_empty returns false after put
    #[test]
    fn test_lru_is_empty_false_after_put() {
        let mut cache: LruCache<i32> = LruCache::new(4);
        cache.put("k".to_string(), 1);
        assert!(!cache.is_empty());
    }

    // 22. LRU ordering: insert A B C (capacity=2); after inserting C, A should be evicted
    #[test]
    fn test_lru_ordering_fifo_eviction() {
        let mut cache: LruCache<i32> = LruCache::new(2);
        cache.put("A".to_string(), 1);
        cache.put("B".to_string(), 2);
        cache.put("C".to_string(), 3); // A is LRU, so A is evicted
        assert_eq!(cache.get("A"), None, "A must be evicted");
        assert_eq!(cache.get("B"), Some(2));
        assert_eq!(cache.get("C"), Some(3));
    }

    // 23. LRU ordering: insert A B (cap=2); get A; insert C — B evicted, not A
    #[test]
    fn test_lru_ordering_access_updates_recency() {
        let mut cache: LruCache<i32> = LruCache::new(2);
        cache.put("A".to_string(), 1);
        cache.put("B".to_string(), 2);
        cache.get("A"); // A becomes MRU
        cache.put("C".to_string(), 3); // B is now LRU, evict B
        assert_eq!(cache.get("A"), Some(1), "A must survive (it was accessed)");
        assert_eq!(cache.get("B"), None, "B must be evicted (it was LRU)");
        assert_eq!(cache.get("C"), Some(3));
    }

    // 24. Multiple puts same key don't increase len beyond 1
    #[test]
    fn test_lru_duplicate_put_no_len_growth() {
        let mut cache: LruCache<i32> = LruCache::new(4);
        cache.put("same".to_string(), 1);
        cache.put("same".to_string(), 2);
        cache.put("same".to_string(), 3);
        assert_eq!(cache.len(), 1);
    }

    // 25. LruCacheStats default has all zeros
    #[test]
    fn test_cache_stats_default_zeros() {
        let s = LruCacheStats::default();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.total_bytes, 0);
    }

    // 26. LruCacheStats hit_rate with only misses returns 0.0
    #[test]
    fn test_cache_stats_hit_rate_only_misses() {
        let s = LruCacheStats { hits: 0, misses: 10, evictions: 0, total_bytes: 0 };
        assert_eq!(s.hit_rate(), 0.0);
    }

    // 27. LruCacheStats hit_rate with only hits returns 1.0
    #[test]
    fn test_cache_stats_hit_rate_only_hits() {
        let s = LruCacheStats { hits: 7, misses: 0, evictions: 0, total_bytes: 0 };
        assert!((s.hit_rate() - 1.0).abs() < 1e-9);
    }

    // 28. Manual evict returns the LRU entry
    #[test]
    fn test_lru_manual_evict_returns_lru() {
        let mut cache: LruCache<i32> = LruCache::new(3);
        cache.put("first".to_string(), 1);
        cache.put("second".to_string(), 2);
        let evicted = cache.evict();
        assert!(evicted.is_some());
        let (key, _val) = evicted.expect("evicted entry");
        assert_eq!(key, "first", "first inserted should be LRU");
    }
}
