//! KV cache management for inference serving
//!
//! Stores key-value tensors from previous tokens to avoid recomputation
//! during incremental decoding.

mod kv_cache_extra_tests;

use std::collections::HashMap;

// ─── Entry ────────────────────────────────────────────────────────────────────

/// Cache entry for a single layer and request.
#[derive(Debug, Clone)]
pub struct KvCacheEntry {
    pub request_id: u64,
    pub layer_idx: usize,
    /// Cached keys: \[seq_len, num_kv_heads, head_dim\] flattened
    pub keys: Vec<f32>,
    /// Cached values: \[seq_len, num_kv_heads, head_dim\] flattened
    pub values: Vec<f32>,
    pub seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Size in bytes
    pub size_bytes: usize,
}

impl KvCacheEntry {
    pub fn new(
        request_id: u64,
        layer_idx: usize,
        keys: Vec<f32>,
        values: Vec<f32>,
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let size_bytes = (keys.len() + values.len()) * 4; // f32 = 4 bytes
        Self {
            request_id,
            layer_idx,
            keys,
            values,
            seq_len,
            num_kv_heads,
            head_dim,
            size_bytes,
        }
    }

    /// Append new key-value tensors (for incremental decoding).
    pub fn extend(
        &mut self,
        new_keys: &[f32],
        new_values: &[f32],
        new_tokens: usize,
    ) -> Result<(), KvCacheError> {
        let expected = new_tokens * self.num_kv_heads * self.head_dim;
        if new_keys.len() != expected {
            return Err(KvCacheError::DimensionMismatch {
                expected,
                got: new_keys.len(),
            });
        }
        if new_values.len() != expected {
            return Err(KvCacheError::DimensionMismatch {
                expected,
                got: new_values.len(),
            });
        }
        self.keys.extend_from_slice(new_keys);
        self.values.extend_from_slice(new_values);
        self.seq_len += new_tokens;
        self.size_bytes = (self.keys.len() + self.values.len()) * 4;
        Ok(())
    }

    /// Truncate to a maximum sequence length (for sliding window attention).
    pub fn truncate(&mut self, max_seq_len: usize) {
        if self.seq_len > max_seq_len {
            let tokens_to_remove = self.seq_len - max_seq_len;
            let stride = self.num_kv_heads * self.head_dim;
            let remove_len = tokens_to_remove * stride;
            self.keys.drain(0..remove_len);
            self.values.drain(0..remove_len);
            self.seq_len = max_seq_len;
            self.size_bytes = (self.keys.len() + self.values.len()) * 4;
        }
    }
}

// ─── Eviction Policy ──────────────────────────────────────────────────────────

/// Eviction policy for KV cache entries.
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// First In First Out
    Fifo,
    /// Size-based: evict largest entries first
    LargestFirst,
    /// Deterministic: evict by request_id modulo (avoids non-determinism)
    Deterministic,
}

impl std::fmt::Display for EvictionPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvictionPolicy::Lru => write!(f, "LRU"),
            EvictionPolicy::Lfu => write!(f, "LFU"),
            EvictionPolicy::Fifo => write!(f, "FIFO"),
            EvictionPolicy::LargestFirst => write!(f, "LargestFirst"),
            EvictionPolicy::Deterministic => write!(f, "Deterministic"),
        }
    }
}

// ─── Manager ──────────────────────────────────────────────────────────────────

/// KV cache manager covering all requests and all layers.
///
/// Internal layout: `request_id → layer_idx → KvCacheEntry`.
pub struct KvCacheManager {
    /// request_id → layer_idx → entry
    cache: HashMap<u64, HashMap<usize, KvCacheEntry>>,
    pub policy: EvictionPolicy,
    pub max_bytes: usize,
    current_bytes: usize,
    /// LRU / FIFO tracking: monotonic insertion-order counter per request_id.
    access_order: HashMap<u64, u64>,
    /// LFU tracking: access count per request_id.
    access_count: HashMap<u64, u64>,
    next_seq: u64,
    total_hits: u64,
    total_misses: u64,
    total_evictions: u64,
}

impl KvCacheManager {
    /// Create a new manager with the given eviction policy and memory budget.
    pub fn new(policy: EvictionPolicy, max_bytes: usize) -> Self {
        Self {
            cache: HashMap::new(),
            policy,
            max_bytes,
            current_bytes: 0,
            access_order: HashMap::new(),
            access_count: HashMap::new(),
            next_seq: 0,
            total_hits: 0,
            total_misses: 0,
            total_evictions: 0,
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn next_seq_id(&mut self) -> u64 {
        let s = self.next_seq;
        self.next_seq += 1;
        s
    }

    /// Total bytes occupied by all entries of a request.
    fn request_bytes(&self, request_id: u64) -> usize {
        self.cache
            .get(&request_id)
            .map(|layers| layers.values().map(|e| e.size_bytes).sum())
            .unwrap_or(0)
    }

    // ── public API ───────────────────────────────────────────────────────────

    /// Insert or replace a KV cache entry for (request_id, layer_idx).
    ///
    /// If the cache is over budget after insertion, entries are evicted
    /// according to the configured policy until the budget is satisfied.
    pub fn insert(&mut self, entry: KvCacheEntry) -> Result<(), KvCacheError> {
        let needed = entry.size_bytes;

        // If a previous entry exists for the same (request, layer), subtract
        // its bytes so we only account for the delta.
        let old_bytes = self
            .cache
            .get(&entry.request_id)
            .and_then(|layers| layers.get(&entry.layer_idx))
            .map(|e| e.size_bytes)
            .unwrap_or(0);

        // Check whether it can ever fit.
        if needed > self.max_bytes {
            return Err(KvCacheError::CapacityExceeded {
                needed,
                available: self.max_bytes,
            });
        }

        // Evict until there is room.
        let headroom = self.max_bytes - (self.current_bytes - old_bytes);
        if needed > headroom {
            self.evict_to_fit(needed - headroom);
        }

        // Track access for LRU / FIFO.
        let seq = self.next_seq_id();
        self.access_order.insert(entry.request_id, seq);
        *self.access_count.entry(entry.request_id).or_insert(0) += 1;

        self.current_bytes = self.current_bytes - old_bytes + needed;
        self.cache.entry(entry.request_id).or_default().insert(entry.layer_idx, entry);

        Ok(())
    }

    /// Retrieve a KV cache entry, updating access tracking.
    pub fn get(&mut self, request_id: u64, layer_idx: usize) -> Option<&KvCacheEntry> {
        let hit = self.cache.get(&request_id).and_then(|layers| layers.get(&layer_idx)).is_some();

        if hit {
            self.total_hits += 1;
            // Refresh LRU sequence.
            let seq = self.next_seq_id();
            self.access_order.insert(request_id, seq);
            *self.access_count.entry(request_id).or_insert(0) += 1;
            self.cache.get(&request_id).and_then(|layers| layers.get(&layer_idx))
        } else {
            self.total_misses += 1;
            None
        }
    }

    /// Evict entries until at least `needed_bytes` are freed.
    ///
    /// Returns the number of *entries* (not bytes) evicted.
    pub fn evict_to_fit(&mut self, needed_bytes: usize) -> usize {
        let mut freed = 0usize;
        let mut evicted_count = 0usize;

        while freed < needed_bytes && !self.cache.is_empty() {
            // Choose the victim request according to policy.
            let victim_id: u64 = match self.policy {
                EvictionPolicy::Lru | EvictionPolicy::Fifo => {
                    // Evict the request with the smallest (oldest) access_order seq.
                    let victim = self
                        .access_order
                        .iter()
                        .filter(|(id, _)| self.cache.contains_key(*id))
                        .min_by_key(|(_, &seq)| seq)
                        .map(|(&id, _)| id);
                    match victim {
                        Some(id) => id,
                        None => break,
                    }
                },
                EvictionPolicy::Lfu => {
                    // Evict the request with the lowest access count.
                    let victim = self
                        .access_count
                        .iter()
                        .filter(|(id, _)| self.cache.contains_key(*id))
                        .min_by_key(|(_, &cnt)| cnt)
                        .map(|(&id, _)| id);
                    match victim {
                        Some(id) => id,
                        None => break,
                    }
                },
                EvictionPolicy::LargestFirst => {
                    // Evict the request that occupies the most bytes.
                    let victim =
                        self.cache.keys().copied().max_by_key(|&id| self.request_bytes(id));
                    match victim {
                        Some(id) => id,
                        None => break,
                    }
                },
                EvictionPolicy::Deterministic => {
                    // Evict deterministically: smallest request_id % prime bucket.
                    let victim = self.cache.keys().copied().min_by_key(|&id| id % 65537);
                    match victim {
                        Some(id) => id,
                        None => break,
                    }
                },
            };

            freed += self.remove_request_internal(victim_id);
            evicted_count += 1;
            self.total_evictions += 1;
        }

        evicted_count
    }

    /// Remove all entries for a completed request.
    ///
    /// Returns the number of bytes freed.
    pub fn remove_request(&mut self, request_id: u64) -> usize {
        self.remove_request_internal(request_id)
    }

    fn remove_request_internal(&mut self, request_id: u64) -> usize {
        let bytes = self
            .cache
            .remove(&request_id)
            .map(|layers| layers.values().map(|e| e.size_bytes).sum::<usize>())
            .unwrap_or(0);
        self.current_bytes = self.current_bytes.saturating_sub(bytes);
        self.access_order.remove(&request_id);
        self.access_count.remove(&request_id);
        bytes
    }

    /// Current cache statistics snapshot.
    pub fn stats(&self) -> KvCacheStats {
        KvCacheStats {
            total_hits: self.total_hits,
            total_misses: self.total_misses,
            total_evictions: self.total_evictions,
            current_bytes: self.current_bytes,
            max_bytes: self.max_bytes,
            num_requests: self.cache.len(),
            hit_rate: self.hit_rate(),
            utilization: self.utilization(),
        }
    }

    /// Cache hit rate (0.0 – 1.0).
    pub fn hit_rate(&self) -> f32 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            self.total_hits as f32 / total as f32
        }
    }

    /// Bytes currently used by the cache.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Fraction of the memory budget currently used (0.0 – 1.0).
    pub fn utilization(&self) -> f32 {
        self.current_bytes as f32 / self.max_bytes.max(1) as f32
    }

    /// Number of distinct requests currently held in cache.
    pub fn num_requests_cached(&self) -> usize {
        self.cache.len()
    }

    /// Remove all entries from the cache, resetting statistics.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
        self.access_count.clear();
        self.current_bytes = 0;
        self.next_seq = 0;
        self.total_hits = 0;
        self.total_misses = 0;
        self.total_evictions = 0;
    }
}

// ─── Stats ────────────────────────────────────────────────────────────────────

/// Snapshot of KV cache statistics.
#[derive(Debug, Clone)]
pub struct KvCacheStats {
    pub total_hits: u64,
    pub total_misses: u64,
    pub total_evictions: u64,
    pub current_bytes: usize,
    pub max_bytes: usize,
    pub num_requests: usize,
    pub hit_rate: f32,
    pub utilization: f32,
}

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors that can occur during KV cache operations.
#[derive(Debug)]
pub enum KvCacheError {
    DimensionMismatch { expected: usize, got: usize },
    CapacityExceeded { needed: usize, available: usize },
    RequestNotFound(u64),
}

impl std::fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCacheError::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected} elements, got {got}"
            ),
            KvCacheError::CapacityExceeded { needed, available } => write!(
                f,
                "capacity exceeded: needed {needed} bytes, only {available} bytes available"
            ),
            KvCacheError::RequestNotFound(id) => {
                write!(f, "request {id} not found in cache")
            },
        }
    }
}

impl std::error::Error for KvCacheError {}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a minimal entry with all-zeros tensors.
    fn make_entry(request_id: u64, layer_idx: usize, seq_len: usize) -> KvCacheEntry {
        let num_kv_heads = 2;
        let head_dim = 4;
        let total = seq_len * num_kv_heads * head_dim;
        KvCacheEntry::new(
            request_id,
            layer_idx,
            vec![0.0_f32; total],
            vec![0.0_f32; total],
            seq_len,
            num_kv_heads,
            head_dim,
        )
    }

    #[test]
    fn test_kv_entry_new() {
        let entry = make_entry(1, 0, 4);
        assert_eq!(entry.request_id, 1);
        assert_eq!(entry.layer_idx, 0);
        assert_eq!(entry.seq_len, 4);
        assert_eq!(entry.num_kv_heads, 2);
        assert_eq!(entry.head_dim, 4);
    }

    #[test]
    fn test_kv_entry_size_bytes() {
        // seq=4, heads=2, dim=4 → 4*2*4 = 32 f32 per tensor, ×2 tensors × 4 bytes = 256
        let entry = make_entry(1, 0, 4);
        assert_eq!(entry.size_bytes, 256);
    }

    #[test]
    fn test_kv_entry_extend() {
        let mut entry = make_entry(1, 0, 2);
        let stride = 2 * 4; // num_kv_heads * head_dim
        let new_keys = vec![1.0_f32; stride];
        let new_vals = vec![2.0_f32; stride];
        entry.extend(&new_keys, &new_vals, 1).expect("extend ok");
        assert_eq!(entry.seq_len, 3);
        assert_eq!(entry.keys.len(), 3 * stride);
    }

    #[test]
    fn test_kv_entry_extend_wrong_size() {
        let mut entry = make_entry(1, 0, 2);
        // Provide wrong-size new_keys.
        let result = entry.extend(&[0.0_f32; 999], &[0.0_f32; 8], 1);
        assert!(result.is_err());
        match result.unwrap_err() {
            KvCacheError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 8);
                assert_eq!(got, 999);
            },
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_kv_entry_truncate() {
        let mut entry = make_entry(1, 0, 6);
        entry.truncate(4);
        assert_eq!(entry.seq_len, 4);
        let stride = 2 * 4;
        assert_eq!(entry.keys.len(), 4 * stride);
        assert_eq!(entry.size_bytes, 2 * 4 * stride * 4);
    }

    #[test]
    fn test_kv_entry_truncate_no_op() {
        let mut entry = make_entry(1, 0, 4);
        let original_size = entry.size_bytes;
        entry.truncate(8); // max > current → no-op
        assert_eq!(entry.seq_len, 4);
        assert_eq!(entry.size_bytes, original_size);
    }

    #[test]
    fn test_eviction_policy_display() {
        assert_eq!(EvictionPolicy::Lru.to_string(), "LRU");
        assert_eq!(EvictionPolicy::Lfu.to_string(), "LFU");
        assert_eq!(EvictionPolicy::Fifo.to_string(), "FIFO");
        assert_eq!(EvictionPolicy::LargestFirst.to_string(), "LargestFirst");
        assert_eq!(EvictionPolicy::Deterministic.to_string(), "Deterministic");
    }

    #[test]
    fn test_kv_cache_insert_and_get() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        let entry = make_entry(42, 0, 2);
        mgr.insert(entry).expect("insert ok");
        let retrieved = mgr.get(42, 0);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().request_id, 42);
    }

    #[test]
    fn test_kv_cache_hit_rate_empty() {
        let mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024);
        assert_eq!(mgr.hit_rate(), 0.0);
    }

    #[test]
    fn test_kv_cache_hit_tracking() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        mgr.insert(make_entry(1, 0, 2)).expect("insert ok");
        // Three hits.
        mgr.get(1, 0);
        mgr.get(1, 0);
        mgr.get(1, 0);
        assert_eq!(mgr.total_hits, 3);
        assert_eq!(mgr.total_misses, 0);
        assert!((mgr.hit_rate() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_kv_cache_miss_tracking() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        mgr.get(99, 0); // miss
        mgr.get(99, 1); // miss
        assert_eq!(mgr.total_misses, 2);
        assert_eq!(mgr.total_hits, 0);
        assert_eq!(mgr.hit_rate(), 0.0);
    }

    #[test]
    fn test_kv_cache_lru_eviction() {
        // Budget = 512 bytes; each entry is 256 bytes → fits 2.
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 512);
        mgr.insert(make_entry(1, 0, 4)).expect("insert req 1");
        mgr.insert(make_entry(2, 0, 4)).expect("insert req 2");
        // Access req 1 to make it recently used.
        mgr.get(1, 0);
        // Insert req 3 → must evict the LRU which is req 2.
        mgr.insert(make_entry(3, 0, 4)).expect("insert req 3");
        assert!(mgr.get(1, 0).is_some(), "req 1 should still be in cache");
        assert!(mgr.get(3, 0).is_some(), "req 3 should be in cache");
        // req 2 was evicted
        assert_eq!(mgr.total_evictions, 1);
    }

    #[test]
    fn test_kv_cache_fifo_eviction() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Fifo, 512);
        mgr.insert(make_entry(10, 0, 4)).expect("insert 10");
        mgr.insert(make_entry(11, 0, 4)).expect("insert 11");
        // Insert 12 → FIFO evicts request 10 (inserted first).
        mgr.insert(make_entry(12, 0, 4)).expect("insert 12");
        assert_eq!(mgr.total_evictions, 1);
        assert!(mgr.get(11, 0).is_some());
        assert!(mgr.get(12, 0).is_some());
    }

    #[test]
    fn test_kv_cache_largest_first_eviction() {
        // Budget = 512 bytes.
        // Entry with seq=4 → 256 bytes, seq=1 → 64 bytes.
        // After two inserts: 256 + 64 = 320 bytes used.
        let mut mgr = KvCacheManager::new(EvictionPolicy::LargestFirst, 512);
        mgr.insert(make_entry(1, 0, 4)).expect("insert large (256B)");
        mgr.insert(make_entry(2, 0, 1)).expect("insert small (64B)");
        // Insert another 256B entry → 320 + 256 = 576 > 512 budget.
        // LargestFirst should evict req 1 (256B), freeing enough room.
        mgr.insert(make_entry(3, 0, 4)).expect("insert req 3");
        // Exactly one eviction expected.
        assert_eq!(mgr.total_evictions, 1);
        // The small entry (req 2) and new entry (req 3) should remain.
        assert!(mgr.get(2, 0).is_some(), "small req 2 should survive");
        assert!(mgr.get(3, 0).is_some(), "new req 3 should be in cache");
    }

    #[test]
    fn test_kv_cache_remove_request() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        mgr.insert(make_entry(5, 0, 4)).expect("insert");
        mgr.insert(make_entry(5, 1, 4)).expect("insert layer 1");
        let before = mgr.current_bytes();
        let freed = mgr.remove_request(5);
        assert_eq!(freed, before);
        assert_eq!(mgr.current_bytes(), 0);
        assert_eq!(mgr.num_requests_cached(), 0);
    }

    #[test]
    fn test_kv_cache_utilization() {
        let max = 1024;
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, max);
        // Entry: seq=1, heads=2, dim=4 → 8 floats per tensor × 2 × 4B = 64 bytes.
        mgr.insert(make_entry(1, 0, 1)).expect("insert");
        let expected = 64.0 / max as f32;
        assert!((mgr.utilization() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_kv_cache_stats() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        mgr.insert(make_entry(1, 0, 2)).expect("insert");
        mgr.get(1, 0); // hit
        mgr.get(2, 0); // miss
        let stats = mgr.stats();
        assert_eq!(stats.total_hits, 1);
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.num_requests, 1);
        assert!((stats.hit_rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_kv_cache_error_display() {
        let e1 = KvCacheError::DimensionMismatch {
            expected: 8,
            got: 4,
        };
        assert!(e1.to_string().contains("dimension mismatch"));

        let e2 = KvCacheError::CapacityExceeded {
            needed: 1024,
            available: 512,
        };
        assert!(e2.to_string().contains("capacity exceeded"));

        let e3 = KvCacheError::RequestNotFound(99);
        assert!(e3.to_string().contains("99"));
    }

    // ── Insert too-large entry returns CapacityExceeded ────────────────────

    #[test]
    fn test_insert_too_large_returns_capacity_exceeded() {
        // Budget = 64 bytes; entry is 256 bytes → must fail.
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 64);
        let err = mgr.insert(make_entry(1, 0, 4)).unwrap_err();
        match err {
            KvCacheError::CapacityExceeded { needed, available } => {
                assert!(needed > available);
            },
            other => panic!("expected CapacityExceeded, got: {other}"),
        }
    }

    // ── Insert multiple layers for same request ────────────────────────────

    #[test]
    fn test_insert_multiple_layers_same_request() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        for layer in 0..4 {
            mgr.insert(make_entry(7, layer, 2)).expect("insert layer");
        }
        assert_eq!(
            mgr.num_requests_cached(),
            1,
            "all layers belong to one request"
        );
    }

    // ── Get missing layer returns None ─────────────────────────────────────

    #[test]
    fn test_get_missing_layer_returns_none() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        mgr.insert(make_entry(1, 0, 2)).expect("insert");
        let result = mgr.get(1, 99); // layer 99 never inserted
        assert!(result.is_none());
        assert_eq!(mgr.total_misses, 1);
    }

    // ── Remove non-existent request returns 0 ─────────────────────────────

    #[test]
    fn test_remove_nonexistent_request() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        let freed = mgr.remove_request(9999);
        assert_eq!(freed, 0);
    }

    // ── current_bytes increases on insert ────────────────────────────────

    #[test]
    fn test_current_bytes_increases_on_insert() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        assert_eq!(mgr.current_bytes(), 0);
        mgr.insert(make_entry(1, 0, 4)).expect("insert");
        assert!(mgr.current_bytes() > 0);
    }

    // ── Replacing same (request, layer) does not double-count bytes ────────

    #[test]
    fn test_replace_entry_updates_bytes_correctly() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        mgr.insert(make_entry(1, 0, 4)).expect("first insert");
        let after_first = mgr.current_bytes();
        // Re-insert same slot: size stays the same (same seq_len).
        mgr.insert(make_entry(1, 0, 4)).expect("second insert");
        assert_eq!(
            mgr.current_bytes(),
            after_first,
            "re-inserting same-size entry must not add bytes"
        );
    }

    // ── LFU eviction evicts least-frequently used ─────────────────────────

    #[test]
    fn test_kv_cache_lfu_eviction() {
        // Budget for 2 entries (256 bytes each).
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lfu, 512);
        mgr.insert(make_entry(1, 0, 4)).expect("insert req 1");
        mgr.insert(make_entry(2, 0, 4)).expect("insert req 2");

        // Access req 1 multiple times to boost its frequency.
        mgr.get(1, 0);
        mgr.get(1, 0);
        mgr.get(1, 0);

        // Now insert req 3 → LFU should evict req 2 (lower access count).
        mgr.insert(make_entry(3, 0, 4)).expect("insert req 3");
        assert_eq!(mgr.total_evictions, 1, "one eviction must have occurred");
        assert!(mgr.get(1, 0).is_some(), "req 1 (high freq) must survive");
        assert!(mgr.get(3, 0).is_some(), "req 3 (new) must be in cache");
    }

    // ── Deterministic eviction policy ────────────────────────────────────

    #[test]
    fn test_kv_cache_deterministic_eviction() {
        // Budget for 2 entries.
        let mut mgr = KvCacheManager::new(EvictionPolicy::Deterministic, 512);
        mgr.insert(make_entry(100, 0, 4)).expect("insert 100");
        mgr.insert(make_entry(200, 0, 4)).expect("insert 200");
        // Third insert triggers eviction; deterministic policy evicts by id % 65537.
        mgr.insert(make_entry(300, 0, 4)).expect("insert 300");
        assert_eq!(mgr.total_evictions, 1);
    }

    // ── utilization is 0 for empty cache ─────────────────────────────────

    #[test]
    fn test_utilization_zero_when_empty() {
        let mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024);
        assert_eq!(mgr.utilization(), 0.0);
    }

    // ── utilization approaches 1.0 at capacity ───────────────────────────

    #[test]
    fn test_utilization_near_full() {
        // Each entry is 256 bytes; budget is 512 → 2 entries → ~50% each.
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 512);
        mgr.insert(make_entry(1, 0, 4)).expect("insert");
        let util = mgr.utilization();
        assert!(util > 0.0 && util <= 1.0, "utilization must be in (0,1]");
    }

    // ── extend then get returns updated seq_len ───────────────────────────

    #[test]
    fn test_extend_entry_then_get() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        let mut entry = make_entry(1, 0, 2);
        let stride = 2 * 4; // num_kv_heads * head_dim
        entry
            .extend(&vec![0.0_f32; stride], &vec![0.0_f32; stride], 1)
            .expect("extend ok");
        assert_eq!(entry.seq_len, 3);
        mgr.insert(entry).expect("insert extended entry");
        let retrieved = mgr.get(1, 0).expect("entry must be in cache");
        assert_eq!(retrieved.seq_len, 3);
    }

    // ── KvCacheEntry extend wrong values size returns error ────────────────

    #[test]
    fn test_kv_entry_extend_wrong_values_size() {
        let mut entry = make_entry(1, 0, 2);
        let stride = 2 * 4;
        let result = entry.extend(&vec![0.0_f32; stride], &vec![0.0_f32; 999], 1);
        assert!(result.is_err(), "wrong values size must return error");
    }

    // ── hit_rate after mixed hits and misses ─────────────────────────────

    #[test]
    fn test_hit_rate_mixed() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        mgr.insert(make_entry(1, 0, 2)).expect("insert");
        mgr.get(1, 0); // hit
        mgr.get(1, 0); // hit
        mgr.get(2, 0); // miss
        mgr.get(2, 1); // miss
                       // 2 hits, 2 misses → 0.5
        let rate = mgr.hit_rate();
        assert!((rate - 0.5).abs() < 1e-6);
    }

    // ── stats snapshot matches individual fields ──────────────────────────

    #[test]
    fn test_stats_snapshot_matches_fields() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        mgr.insert(make_entry(1, 0, 4)).expect("insert");
        mgr.get(1, 0); // hit
        mgr.get(2, 0); // miss
        let stats = mgr.stats();
        assert_eq!(stats.total_hits, mgr.total_hits);
        assert_eq!(stats.total_misses, mgr.total_misses);
        assert_eq!(stats.current_bytes, mgr.current_bytes());
        assert_eq!(stats.max_bytes, mgr.max_bytes);
    }

    // ── Evict when already empty does nothing ──────────────────────────────

    #[test]
    fn test_evict_when_empty_does_nothing() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024);
        let evicted = mgr.evict_to_fit(512);
        assert_eq!(evicted, 0, "evicting from empty cache must return 0");
    }

    // ── Multiple requests, remove one, others remain ──────────────────────

    #[test]
    fn test_remove_one_request_others_remain() {
        let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
        for id in 1u64..=5 {
            mgr.insert(make_entry(id, 0, 2)).expect("insert");
        }
        mgr.remove_request(3);
        assert_eq!(mgr.num_requests_cached(), 4);
        for id in [1u64, 2, 4, 5] {
            assert!(
                mgr.get(id, 0).is_some(),
                "request {id} must still be cached"
            );
        }
    }

    // ── Truncate to same length is a no-op ───────────────────────────────

    #[test]
    fn test_truncate_to_same_length_no_op() {
        let mut entry = make_entry(1, 0, 4);
        let before_size = entry.size_bytes;
        entry.truncate(4);
        assert_eq!(entry.seq_len, 4);
        assert_eq!(entry.size_bytes, before_size);
    }
}
