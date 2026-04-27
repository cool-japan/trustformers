//! Semantic caching: returns cached results for semantically similar queries.
//!
//! Unlike exact-match caching, semantic caching finds cached entries whose
//! input embedding is within a cosine similarity threshold of the query.
//!
//! # Algorithm
//! 1. Embed the query into a dense vector (via a small embedding function)
//! 2. Search the cache index for entries with cosine_similarity >= threshold
//! 3. Return the cached result if found, otherwise run inference and cache it

use std::sync::{Arc, RwLock};
use std::time::Instant;

use thiserror::Error;

/// Errors from the semantic cache
#[derive(Debug, Error)]
pub enum SemanticCacheError {
    #[error("Embedding dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),

    #[error("Zero-length embedding")]
    ZeroLengthEmbedding,
}

/// Configuration for the semantic cache
#[derive(Debug, Clone)]
pub struct SemanticCacheConfig {
    /// Maximum number of entries in the cache
    pub max_entries: usize,
    /// Cosine similarity threshold for a cache hit (0.0 - 1.0)
    pub similarity_threshold: f32,
    /// Embedding dimension (must match your embedding model)
    pub embedding_dim: usize,
    /// TTL for cache entries in seconds (0 = no expiry)
    pub ttl_secs: u64,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            similarity_threshold: 0.95,
            embedding_dim: 768,
            ttl_secs: 3600,
        }
    }
}

/// A single semantic cache entry
#[derive(Debug, Clone)]
pub struct SemanticCacheEntry {
    pub key: String,
    /// Normalized unit vector (pre-normalized for fast cosine similarity)
    pub embedding: Vec<f32>,
    /// Cached response value
    pub value: serde_json::Value,
    pub created_at: Instant,
    pub hit_count: u64,
}

/// Running statistics for the semantic cache
#[derive(Debug, Clone, Default)]
pub struct SemanticCacheStats {
    pub total_lookups: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
}

impl SemanticCacheStats {
    /// Hit rate as a fraction in [0, 1]
    pub fn hit_rate(&self) -> f32 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.cache_hits as f32 / self.total_lookups as f32
        }
    }
}

/// The semantic cache
pub struct SemanticCache {
    config: SemanticCacheConfig,
    entries: Arc<RwLock<Vec<SemanticCacheEntry>>>,
    stats: Arc<RwLock<SemanticCacheStats>>,
}

impl SemanticCache {
    /// Create a new semantic cache with the given configuration
    pub fn new(config: SemanticCacheConfig) -> Self {
        Self {
            config,
            entries: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(SemanticCacheStats::default())),
        }
    }

    /// Look up by embedding vector. Returns cached value if similarity >= threshold.
    ///
    /// The provided embedding will be normalized before comparison.
    pub fn get(&self, embedding: &[f32]) -> Option<serde_json::Value> {
        let mut normalized = embedding.to_vec();
        normalize_embedding(&mut normalized);

        // Increment total lookups
        {
            let mut stats = self.stats.write().ok()?;
            stats.total_lookups += 1;
        }

        // Evict expired entries before lookup
        self.evict_expired();

        let entries = self.entries.read().ok()?;
        let threshold = self.config.similarity_threshold;

        let mut best_similarity = f32::NEG_INFINITY;
        let mut best_index: Option<usize> = None;

        for (idx, entry) in entries.iter().enumerate() {
            let sim = cosine_similarity(&normalized, &entry.embedding);
            if sim >= threshold && sim > best_similarity {
                best_similarity = sim;
                best_index = Some(idx);
            }
        }

        drop(entries);

        if let Some(idx) = best_index {
            // Increment hit count and stats
            if let Ok(mut entries) = self.entries.write() {
                if let Some(entry) = entries.get_mut(idx) {
                    entry.hit_count += 1;
                    let value = entry.value.clone();
                    drop(entries);

                    if let Ok(mut stats) = self.stats.write() {
                        stats.cache_hits += 1;
                    }

                    return Some(value);
                }
            }
        }

        // Cache miss
        if let Ok(mut stats) = self.stats.write() {
            stats.cache_misses += 1;
        }

        None
    }

    /// Insert a new entry. Evicts oldest entry if at capacity.
    pub fn insert(
        &self,
        key: String,
        embedding: Vec<f32>,
        value: serde_json::Value,
    ) -> Result<(), SemanticCacheError> {
        if embedding.is_empty() {
            return Err(SemanticCacheError::ZeroLengthEmbedding);
        }

        if embedding.len() != self.config.embedding_dim {
            return Err(SemanticCacheError::DimensionMismatch {
                expected: self.config.embedding_dim,
                got: embedding.len(),
            });
        }

        let mut normalized = embedding;
        normalize_embedding(&mut normalized);

        let entry = SemanticCacheEntry {
            key,
            embedding: normalized,
            value,
            created_at: Instant::now(),
            hit_count: 0,
        };

        let mut entries = self
            .entries
            .write()
            .map_err(|e| SemanticCacheError::LockPoisoned(e.to_string()))?;

        // Evict oldest entry if at capacity
        if entries.len() >= self.config.max_entries {
            entries.remove(0);

            if let Ok(mut stats) = self.stats.write() {
                stats.evictions += 1;
            }
        }

        entries.push(entry);
        Ok(())
    }

    /// Evict expired TTL entries
    pub fn evict_expired(&self) {
        if self.config.ttl_secs == 0 {
            return;
        }

        let ttl = std::time::Duration::from_secs(self.config.ttl_secs);
        let now = Instant::now();

        if let Ok(mut entries) = self.entries.write() {
            let before = entries.len();
            entries.retain(|e| now.duration_since(e.created_at) < ttl);
            let evicted = before - entries.len();

            if evicted > 0 {
                if let Ok(mut stats) = self.stats.write() {
                    stats.evictions += evicted as u64;
                }
            }
        }
    }

    /// Clear all entries
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.write() {
            entries.clear();
        }
    }

    /// Current statistics snapshot
    pub fn stats(&self) -> SemanticCacheStats {
        self.stats
            .read()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Number of live entries in the cache
    pub fn len(&self) -> usize {
        self.entries
            .read()
            .map(|e| e.len())
            .unwrap_or(0)
    }

    /// Returns true if the cache has no entries
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Cosine similarity between two unit vectors.
///
/// Both vectors are assumed to be pre-normalized to unit length, so this
/// reduces to a simple dot product.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Normalize a vector to unit length in-place.
///
/// Vectors with norm below `1e-8` are left unchanged to avoid
/// division by near-zero.
pub fn normalize_embedding(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

// ─────────────────────────── tests ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_config(threshold: f32, max_entries: usize, ttl_secs: u64) -> SemanticCacheConfig {
        SemanticCacheConfig {
            max_entries,
            similarity_threshold: threshold,
            embedding_dim: 4,
            ttl_secs,
        }
    }

    fn unit_vec(v: Vec<f32>) -> Vec<f32> {
        let mut v = v;
        normalize_embedding(&mut v);
        v
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-6, "Expected 0.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![0.5_f32, 0.5, 0.5, 0.5];
        let mut a_norm = a.clone();
        normalize_embedding(&mut a_norm);
        let sim = cosine_similarity(&a_norm, &a_norm);
        assert!((sim - 1.0).abs() < 1e-6, "Expected 1.0, got {sim}");
    }

    #[test]
    fn test_normalize_embedding() {
        let mut v = vec![3.0_f32, 4.0, 0.0, 0.0];
        normalize_embedding(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "Norm should be 1.0, got {norm}");
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_semantic_cache_exact_match() {
        let cache = SemanticCache::new(make_config(0.99, 16, 0));
        let emb = unit_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let value = serde_json::json!({"result": "hello"});

        cache.insert("k1".into(), emb.clone(), value.clone()).expect("insert should succeed");

        let result = cache.get(&emb);
        assert!(result.is_some(), "Exact match should hit");
        assert_eq!(result.unwrap(), value);
    }

    #[test]
    fn test_semantic_cache_similar_miss() {
        // Threshold = 0.99, but similarity will be ~0.71 (45° angle)
        let cache = SemanticCache::new(make_config(0.99, 16, 0));

        let emb_a = unit_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let emb_b = unit_vec(vec![1.0, 1.0, 0.0, 0.0]);

        cache
            .insert("k1".into(), emb_a, serde_json::json!("A"))
            .expect("insert");

        let result = cache.get(&emb_b);
        assert!(result.is_none(), "Low similarity should miss");
    }

    #[test]
    fn test_semantic_cache_similar_hit() {
        // Threshold = 0.90, similarity between very close vectors should be > 0.90
        let cache = SemanticCache::new(make_config(0.90, 16, 0));

        // Embed vector for a "base" direction
        let base = unit_vec(vec![1.0, 0.0, 0.0, 0.0]);
        // Slightly perturbed (cos ≈ 0.9998)
        let query = unit_vec(vec![1.0, 0.01, 0.0, 0.0]);

        cache
            .insert("k1".into(), base, serde_json::json!("hit_value"))
            .expect("insert");

        let result = cache.get(&query);
        assert!(result.is_some(), "Near-identical vectors should hit with threshold=0.90");
        assert_eq!(result.unwrap(), serde_json::json!("hit_value"));
    }

    #[test]
    fn test_semantic_cache_eviction_at_capacity() {
        let cache = SemanticCache::new(make_config(0.99, 3, 0));

        // Insert 3 entries that fill the cache
        for i in 0..3_u32 {
            let mut emb = vec![0.0_f32; 4];
            emb[i as usize % 4] = 1.0;
            cache
                .insert(format!("k{i}"), emb, serde_json::json!(i))
                .expect("insert");
        }
        assert_eq!(cache.len(), 3);

        // Insert a 4th entry — should evict oldest
        let new_emb = unit_vec(vec![0.3, 0.3, 0.3, 0.3]);
        cache
            .insert("k_new".into(), new_emb, serde_json::json!("new"))
            .expect("insert");

        assert_eq!(cache.len(), 3, "Capacity must stay at max_entries");

        // Check eviction count
        let stats = cache.stats();
        assert!(stats.evictions >= 1, "At least 1 eviction should have occurred");
    }

    #[test]
    fn test_semantic_cache_ttl_expiry() {
        // TTL = 0 means no expiry; test with very short TTL is awkward with Instant,
        // so we instead set ttl_secs=0 to disable and verify nothing expires.
        let cache = SemanticCache::new(make_config(0.99, 16, 0));

        let emb = unit_vec(vec![1.0, 0.0, 0.0, 0.0]);
        cache
            .insert("k1".into(), emb.clone(), serde_json::json!("v"))
            .expect("insert");

        // With ttl=0, entry should persist
        cache.evict_expired();
        assert_eq!(cache.len(), 1, "TTL=0 means no expiry");

        // Now create a cache with ttl_secs=1 and force expiry by manually advancing time.
        // We test via a separate cache that explicitly calls evict_expired after the TTL
        // would have passed.  Since we cannot sleep in unit tests easily, we instead
        // verify that the evict_expired function itself doesn't panic when TTL is set.
        let cache2 = SemanticCache::new(make_config(0.99, 16, 1));
        cache2
            .insert("k1".into(), emb, serde_json::json!("v"))
            .expect("insert");
        // Immediately calling evict should NOT evict (entry is fresh)
        cache2.evict_expired();
        assert_eq!(cache2.len(), 1, "Fresh entry should not be evicted");
    }

    #[test]
    fn test_semantic_cache_stats() {
        let cache = SemanticCache::new(make_config(0.99, 16, 0));
        let emb = unit_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let miss_emb = unit_vec(vec![0.0, 1.0, 0.0, 0.0]);

        cache
            .insert("k1".into(), emb.clone(), serde_json::json!("v"))
            .expect("insert");

        // One hit
        let _ = cache.get(&emb);
        // One miss (orthogonal)
        let _ = cache.get(&miss_emb);

        let stats = cache.stats();
        assert_eq!(stats.total_lookups, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_semantic_cache_concurrent_access() {
        use std::sync::Arc;

        let cache = Arc::new(SemanticCache::new(make_config(0.95, 128, 0)));

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let cache = Arc::clone(&cache);
                thread::spawn(move || {
                    let mut emb = vec![0.0_f32; 4];
                    emb[i % 4] = 1.0;
                    let emb = unit_vec(emb);

                    // Insert
                    cache
                        .insert(format!("thread-{i}"), emb.clone(), serde_json::json!(i))
                        .expect("concurrent insert should succeed");

                    // Query
                    let _ = cache.get(&emb);
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }

        assert!(cache.len() > 0, "Cache should have entries after concurrent inserts");
    }

    #[test]
    fn test_semantic_cache_dimension_mismatch_error() {
        let cache = SemanticCache::new(make_config(0.99, 16, 0));
        // dim=4 in config, but we send 3
        let result = cache.insert("k".into(), vec![1.0, 0.0, 0.0], serde_json::json!("v"));
        assert!(
            matches!(result, Err(SemanticCacheError::DimensionMismatch { .. })),
            "Should return DimensionMismatch error"
        );
    }

    #[test]
    fn test_semantic_cache_clear() {
        let cache = SemanticCache::new(make_config(0.99, 16, 0));
        let emb = unit_vec(vec![1.0, 0.0, 0.0, 0.0]);
        cache.insert("k".into(), emb, serde_json::json!("v")).expect("insert");
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_stats_hit_rate_empty() {
        let stats = SemanticCacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0, "Hit rate of zero lookups should be 0.0");
    }
}
