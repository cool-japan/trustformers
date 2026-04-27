#![cfg(test)]
/// Extended tests for the KV cache module.
use super::*;

/// Build a simple cache entry with `seq_len` tokens, 2 heads, 4-dim head.
fn make_entry(request_id: u64, layer_idx: usize, seq_len: usize) -> KvCacheEntry {
    let stride = 2 * 4; // num_kv_heads * head_dim
    let keys = vec![0.5_f32; seq_len * stride];
    let values = vec![0.25_f32; seq_len * stride];
    KvCacheEntry::new(request_id, layer_idx, keys, values, seq_len, 2, 4)
}

// ── 36. KvCacheEntry::new — size_bytes is 4 × (keys + values) ─────────────
#[test]
fn test_kv_entry_new_size_bytes_correct() {
    let entry = make_entry(1, 0, 4); // 4 tokens × 2 heads × 4 dim = 32 floats × 2 = 64 × 4 = 256
    assert_eq!(entry.size_bytes, 256);
}

// ── 37. KvCacheEntry::extend — seq_len advances correctly ─────────────────
#[test]
fn test_kv_entry_extend_advances_seq_len() {
    let mut entry = make_entry(1, 0, 4);
    let stride = 2 * 4;
    entry
        .extend(&vec![0.0_f32; stride * 2], &vec![0.0_f32; stride * 2], 2)
        .expect("extend");
    assert_eq!(entry.seq_len, 6);
}

// ── 38. KvCacheEntry::extend — wrong keys length returns DimensionMismatch ─
#[test]
fn test_kv_entry_extend_wrong_keys_len_returns_error() {
    let mut entry = make_entry(1, 0, 2);
    let err = entry.extend(&[1.0_f32; 999], &[0.0_f32; 8], 1).unwrap_err();
    assert!(matches!(err, KvCacheError::DimensionMismatch { .. }));
}

// ── 39. KvCacheEntry::truncate — seq_len reduces to max ──────────────────
#[test]
fn test_kv_entry_truncate_reduces_seq_len() {
    let mut entry = make_entry(1, 0, 8);
    entry.truncate(4);
    assert_eq!(entry.seq_len, 4);
}

// ── 40. KvCacheEntry::truncate — size_bytes updates after truncation ──────
#[test]
fn test_kv_entry_truncate_updates_size_bytes() {
    let mut entry = make_entry(1, 0, 8);
    let before = entry.size_bytes;
    entry.truncate(4);
    assert!(
        entry.size_bytes < before,
        "size_bytes must decrease after truncation"
    );
}

// ── 41. KvCacheManager::new — empty cache, current_bytes = 0 ─────────────
#[test]
fn test_kv_cache_manager_starts_empty() {
    let mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
    assert_eq!(mgr.current_bytes(), 0);
    assert_eq!(mgr.num_requests_cached(), 0);
}

// ── 42. insert single entry and retrieve by same key ──────────────────────
#[test]
fn test_insert_and_get_single_entry() {
    let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
    mgr.insert(make_entry(5, 2, 4)).expect("insert");
    assert!(
        mgr.get(5, 2).is_some(),
        "inserted entry must be retrievable"
    );
}

// ── 43. get — miss increments total_misses ────────────────────────────────
#[test]
fn test_get_miss_increments_total_misses() {
    let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
    mgr.get(99, 0);
    assert_eq!(mgr.total_misses, 1);
}

// ── 44. get — hit increments total_hits ───────────────────────────────────
#[test]
fn test_get_hit_increments_total_hits() {
    let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
    mgr.insert(make_entry(1, 0, 2)).expect("insert");
    mgr.get(1, 0);
    assert_eq!(mgr.total_hits, 1);
}

// ── 45. hit_rate — 0.0 when no requests made ─────────────────────────────
#[test]
fn test_hit_rate_zero_when_no_requests() {
    let mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024);
    assert_eq!(mgr.hit_rate(), 0.0);
}

// ── 46. insert — current_bytes increases ──────────────────────────────────
#[test]
fn test_insert_increases_current_bytes() {
    let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
    let before = mgr.current_bytes();
    mgr.insert(make_entry(1, 0, 4)).expect("insert");
    assert!(mgr.current_bytes() > before);
}

// ── 47. remove_request — entry no longer retrievable ─────────────────────
#[test]
fn test_remove_request_no_longer_retrievable() {
    let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
    mgr.insert(make_entry(7, 0, 2)).expect("insert");
    mgr.remove_request(7);
    assert!(
        mgr.get(7, 0).is_none(),
        "removed request must not be retrievable"
    );
}

// ── 48. remove_request — bytes decrease ───────────────────────────────────
#[test]
fn test_remove_request_decreases_bytes() {
    let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
    mgr.insert(make_entry(8, 0, 4)).expect("insert");
    let before = mgr.current_bytes();
    mgr.remove_request(8);
    assert!(mgr.current_bytes() < before);
}

// ── 49. EvictionPolicy variants are distinguishable ───────────────────────
#[test]
fn test_eviction_policy_variants_differ() {
    assert_ne!(EvictionPolicy::Lru, EvictionPolicy::Lfu);
    assert_ne!(EvictionPolicy::Fifo, EvictionPolicy::Lru);
}

// ── 50. KvCacheEntry fields are correct after construction ────────────────
#[test]
fn test_kv_entry_fields_after_construction() {
    let entry = make_entry(42, 3, 8);
    assert_eq!(entry.request_id, 42);
    assert_eq!(entry.layer_idx, 3);
    assert_eq!(entry.seq_len, 8);
    assert_eq!(entry.num_kv_heads, 2);
    assert_eq!(entry.head_dim, 4);
}

// ── 51. stats snapshot — utilization matches current/max ─────────────────
#[test]
fn test_stats_utilization_field() {
    let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
    mgr.insert(make_entry(1, 0, 4)).expect("insert");
    let s = mgr.stats();
    let expected = s.current_bytes as f32 / s.max_bytes as f32;
    let actual = s.utilization;
    assert!(
        (actual - expected).abs() < 1e-5,
        "utilization mismatch: expected {expected}, got {actual}"
    );
}

// ── 52. multiple layers for same request are tracked separately ───────────
#[test]
fn test_multiple_layers_same_request_tracked_separately() {
    let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
    mgr.insert(make_entry(10, 0, 4)).expect("insert layer 0");
    mgr.insert(make_entry(10, 1, 4)).expect("insert layer 1");
    assert!(mgr.get(10, 0).is_some(), "layer 0 must be present");
    assert!(mgr.get(10, 1).is_some(), "layer 1 must be present");
}

// ── 53. clear — empties the cache completely ──────────────────────────────
#[test]
fn test_clear_empties_cache() {
    let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 1024 * 1024);
    for i in 0u64..5 {
        mgr.insert(make_entry(i, 0, 2)).expect("insert");
    }
    mgr.clear();
    assert_eq!(mgr.current_bytes(), 0);
    assert_eq!(mgr.num_requests_cached(), 0);
}

// ── 54. evict_to_fit — returns count of evicted entries ──────────────────
#[test]
fn test_evict_to_fit_returns_evicted_count() {
    let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 512);
    // 2 entries at 256 bytes each → full
    mgr.insert(make_entry(1, 0, 4)).expect("insert 1");
    mgr.insert(make_entry(2, 0, 4)).expect("insert 2");
    let evicted = mgr.evict_to_fit(256);
    assert!(
        evicted > 0,
        "must evict at least one entry to free 256 bytes"
    );
}

// ── 55. insert over capacity triggers eviction ────────────────────────────
#[test]
fn test_insert_over_capacity_triggers_eviction() {
    let mut mgr = KvCacheManager::new(EvictionPolicy::Lru, 512);
    mgr.insert(make_entry(1, 0, 4)).expect("insert 1");
    mgr.insert(make_entry(2, 0, 4)).expect("insert 2");
    mgr.insert(make_entry(3, 0, 4)).expect("insert 3");
    assert!(
        mgr.total_evictions >= 1,
        "inserting beyond capacity should trigger eviction"
    );
}
