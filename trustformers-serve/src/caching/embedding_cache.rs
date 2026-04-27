//! Embedding Cache Implementation
//!
//! Caches embeddings and supports vector similarity search.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::config::TierConfig;
use super::metrics::CacheStatsCollector;

/// Embedding cache key
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbeddingKey {
    pub text: String,
    pub model_id: String,
    pub embedding_type: String,
}

/// Embedding cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingEntry {
    pub embedding: Vec<f32>,
    pub dimension: usize,
    pub created_at: u64,
    pub access_count: u64,
    pub last_accessed: u64,
}

impl EmbeddingEntry {
    pub fn new(embedding: Vec<f32>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("System time before UNIX_EPOCH")
            .as_secs();

        Self {
            dimension: embedding.len(),
            embedding,
            created_at: now,
            access_count: 0,
            last_accessed: now,
        }
    }

    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.last_accessed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("System time before UNIX_EPOCH")
            .as_secs();
    }
}

/// Vector index for similarity search
pub struct VectorIndex {
    embeddings: HashMap<EmbeddingKey, EmbeddingEntry>,
}

impl Default for VectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex {
    pub fn new() -> Self {
        Self {
            embeddings: HashMap::new(),
        }
    }

    /// Find similar embeddings
    pub fn find_similar(&self, query: &[f32], k: usize) -> Vec<(EmbeddingKey, f32)> {
        let mut similarities = Vec::new();

        for (key, entry) in &self.embeddings {
            let similarity = cosine_similarity(query, &entry.embedding);
            similarities.push((key.clone(), similarity));
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);

        similarities
    }
}

/// Embedding cache service
pub struct EmbeddingCacheService {
    cache: Arc<RwLock<HashMap<EmbeddingKey, EmbeddingEntry>>>,
    index: Arc<RwLock<VectorIndex>>,
    config: TierConfig,
    metrics: Arc<CacheStatsCollector>,
}

impl EmbeddingCacheService {
    pub fn new(config: TierConfig, metrics: Arc<CacheStatsCollector>) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            index: Arc::new(RwLock::new(VectorIndex::new())),
            config,
            metrics,
        }
    }

    /// Get embedding from cache
    pub async fn get(&self, key: &EmbeddingKey) -> Option<Vec<f32>> {
        let mut cache = self.cache.write().await;

        if let Some(entry) = cache.get_mut(key) {
            entry.mark_accessed();
            self.metrics.record_cache_hit("embedding_cache").await;
            Some(entry.embedding.clone())
        } else {
            self.metrics.record_cache_miss("embedding_cache", "not_found").await;
            None
        }
    }

    /// Store embedding in cache
    pub async fn put(&self, key: EmbeddingKey, embedding: Vec<f32>) -> Result<()> {
        let entry = EmbeddingEntry::new(embedding);

        // Store in cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(key.clone(), entry.clone());
        }

        // Update index
        {
            let mut index = self.index.write().await;
            index.embeddings.insert(key, entry);
        }

        self.metrics.record_cache_put("embedding_cache", 0).await;

        Ok(())
    }

    /// Find similar embeddings
    pub async fn find_similar(&self, query: &[f32], k: usize) -> Vec<(EmbeddingKey, f32)> {
        let index = self.index.read().await;
        index.find_similar(query, k)
    }

    /// Clear cache
    pub async fn clear(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        let mut index = self.index.write().await;

        cache.clear();
        index.embeddings.clear();

        Ok(())
    }

    /// Update configuration
    pub async fn update_config(&self, config: TierConfig) -> Result<()> {
        // Implement config update logic
        // Note: Since config is not mutable, we apply changes to cache behavior

        // If max entries decreased, trigger eviction to fit new limits
        let new_max_entries = config.max_entries;
        {
            let mut cache = self.cache.write().await;
            let mut index = self.index.write().await;

            if cache.len() > new_max_entries {
                let entries_to_remove = cache.len() - new_max_entries;

                // Collect entries to remove based on LRU (least recently used)
                let mut entries: Vec<(EmbeddingKey, u64)> =
                    cache.iter().map(|(k, v)| (k.clone(), v.last_accessed)).collect();

                // Sort by last accessed time (oldest first)
                entries.sort_by_key(|(_, last_accessed)| *last_accessed);

                // Remove oldest entries
                for (key, _) in entries.iter().take(entries_to_remove) {
                    cache.remove(key);
                    index.embeddings.remove(key);
                }

                tracing::info!(
                    "Evicted {} embedding entries due to config update",
                    entries_to_remove
                );
            }
        }

        // Apply new eviction policy effects immediately
        match &config.eviction_policy {
            super::config::EvictionPolicy::TTL => {
                // Clean up any old entries based on TTL
                self.cleanup_old_entries().await?;
            },
            super::config::EvictionPolicy::LRU => {
                tracing::info!("Switched to LRU eviction policy for embedding cache");
            },
            super::config::EvictionPolicy::LFU => {
                tracing::info!("Switched to LFU eviction policy for embedding cache");
            },
            super::config::EvictionPolicy::Priority => {
                tracing::info!("Switched to Priority-based eviction policy for embedding cache");
            },
            super::config::EvictionPolicy::Random => {
                tracing::info!("Switched to Random eviction policy for embedding cache");
            },
            super::config::EvictionPolicy::FIFO => {
                tracing::info!("Switched to FIFO eviction policy for embedding cache");
            },
        }

        tracing::info!(
            "Embedding cache configuration updated: max_entries={}, eviction_policy={:?}",
            new_max_entries,
            config.eviction_policy
        );

        Ok(())
    }

    /// Cleanup old entries based on TTL
    async fn cleanup_old_entries(&self) -> Result<()> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("System time before UNIX_EPOCH")
            .as_secs();

        let mut cache = self.cache.write().await;
        let mut index = self.index.write().await;

        // TTL-based cleanup - remove entries older than configured TTL
        let ttl_seconds = self.config.default_ttl.as_secs();
        let mut keys_to_remove = Vec::new();

        for (key, entry) in cache.iter() {
            if current_time.saturating_sub(entry.created_at) > ttl_seconds {
                keys_to_remove.push(key.clone());
            }
        }

        let removed_count = keys_to_remove.len();
        for key in keys_to_remove {
            cache.remove(&key);
            index.embeddings.remove(&key);
        }

        if removed_count > 0 {
            tracing::info!(
                "TTL cleanup removed {} old embedding entries",
                removed_count
            );
        }

        Ok(())
    }

    /// Run maintenance
    pub async fn run_maintenance(&self) -> Result<()> {
        // Implement maintenance tasks
        let mut entries_cleaned = 0;

        // Clean up old entries (older than 24 hours)
        {
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("System time before UNIX_EPOCH")
                .as_secs();

            let mut cache = self.cache.write().await;
            let mut index = self.index.write().await;

            let max_age_seconds = 24 * 60 * 60; // 24 hours
            let mut keys_to_remove = Vec::new();

            for (key, entry) in cache.iter() {
                if current_time.saturating_sub(entry.last_accessed) > max_age_seconds {
                    keys_to_remove.push(key.clone());
                }
            }

            for key in keys_to_remove {
                cache.remove(&key);
                index.embeddings.remove(&key);
                entries_cleaned += 1;
            }
        }

        // Enforce entry count limits
        if self.config.max_entries > 0 {
            let mut cache = self.cache.write().await;
            let mut index = self.index.write().await;

            while cache.len() > self.config.max_entries {
                // Find least recently used entry
                if let Some((lru_key, _)) = cache
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_accessed)
                    .map(|(k, v)| (k.clone(), v.last_accessed))
                {
                    cache.remove(&lru_key);
                    index.embeddings.remove(&lru_key);
                    entries_cleaned += 1;
                } else {
                    break;
                }
            }
        }

        // Rebuild vector index for consistency
        {
            let cache = self.cache.read().await;
            let mut index = self.index.write().await;

            // Clear and rebuild index from cache
            index.embeddings.clear();
            for (key, entry) in cache.iter() {
                index.embeddings.insert(key.clone(), entry.clone());
            }
        }

        if entries_cleaned > 0 {
            tracing::info!(
                "Embedding cache maintenance completed: cleaned {} entries",
                entries_cleaned
            );
        }

        Ok(())
    }

    /// Get statistics
    pub async fn get_stats(&self) -> EmbeddingCacheStats {
        let cache = self.cache.read().await;

        EmbeddingCacheStats {
            entry_count: cache.len(),
            total_dimension: cache.values().map(|e| e.dimension).sum(),
            avg_dimension: if cache.is_empty() {
                0.0
            } else {
                cache.values().map(|e| e.dimension as f32).sum::<f32>() / cache.len() as f32
            },
            hit_rate: self.metrics.get_hit_rate("embedding_cache").await,
        }
    }
}

/// Embedding cache statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmbeddingCacheStats {
    pub entry_count: usize,
    pub total_dimension: usize,
    pub avg_dimension: f32,
    pub hit_rate: f32,
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caching::config::{EvictionPolicy, TierConfig};
    use crate::caching::metrics::CacheStatsCollector;
    use std::time::Duration;

    fn make_tier_config(max_entries: usize) -> TierConfig {
        TierConfig {
            max_size_bytes: 64 * 1024 * 1024,
            max_entries,
            default_ttl: Duration::from_secs(3600),
            eviction_policy: EvictionPolicy::LFU,
            compression_enabled: false,
            tier_name: "test_embedding".to_string(),
        }
    }

    fn make_key(text: &str) -> EmbeddingKey {
        EmbeddingKey {
            text: text.to_string(),
            model_id: "test_model".to_string(),
            embedding_type: "dense".to_string(),
        }
    }

    // --- EmbeddingEntry tests ---

    #[test]
    fn test_embedding_entry_dimension_matches_vec_len() {
        let emb = vec![0.1f32, 0.2, 0.3, 0.4];
        let entry = EmbeddingEntry::new(emb.clone());
        assert_eq!(entry.dimension, 4);
        assert_eq!(entry.embedding.len(), 4);
    }

    #[test]
    fn test_embedding_entry_initial_access_count_zero() {
        let entry = EmbeddingEntry::new(vec![1.0, 2.0]);
        assert_eq!(entry.access_count, 0);
    }

    #[test]
    fn test_embedding_entry_mark_accessed_increments_count() {
        let mut entry = EmbeddingEntry::new(vec![1.0, 2.0]);
        entry.mark_accessed();
        assert_eq!(entry.access_count, 1);
        entry.mark_accessed();
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_embedding_entry_created_at_and_last_accessed_non_zero() {
        let entry = EmbeddingEntry::new(vec![0.5]);
        assert!(entry.created_at > 0);
        assert!(entry.last_accessed > 0);
    }

    // --- VectorIndex tests ---

    #[test]
    fn test_vector_index_find_similar_empty_returns_empty() {
        let index = VectorIndex::new();
        let results = index.find_similar(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_index_find_similar_returns_at_most_k() {
        let mut index = VectorIndex::new();
        for i in 0..10u32 {
            let key = make_key(&format!("query_{}", i));
            let emb = vec![i as f32, 0.0];
            index.embeddings.insert(key, EmbeddingEntry::new(emb));
        }
        let results = index.find_similar(&[1.0, 0.0], 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_vector_index_find_similar_sorted_by_similarity() {
        let mut index = VectorIndex::new();
        let key_a = make_key("a");
        let key_b = make_key("b");
        // key_a is more similar to [1,0] than key_b
        index.embeddings.insert(key_a, EmbeddingEntry::new(vec![1.0, 0.0]));
        index.embeddings.insert(key_b, EmbeddingEntry::new(vec![0.0, 1.0]));
        let results = index.find_similar(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // First result should have higher similarity
        assert!(results[0].1 >= results[1].1);
    }

    // --- cosine_similarity function tests (via VectorIndex) ---

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let mut index = VectorIndex::new();
        let key = make_key("same");
        index.embeddings.insert(key, EmbeddingEntry::new(vec![0.6, 0.8]));
        let results = index.find_similar(&[0.6, 0.8], 1);
        assert_eq!(results.len(), 1);
        assert!((results[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let mut index = VectorIndex::new();
        let key = make_key("ortho");
        index.embeddings.insert(key, EmbeddingEntry::new(vec![0.0, 1.0]));
        let results = index.find_similar(&[1.0, 0.0], 1);
        assert!((results[0].1 - 0.0).abs() < 1e-5);
    }

    // --- EmbeddingCacheService async tests ---

    #[tokio::test]
    async fn test_embedding_service_miss_on_empty() {
        let metrics = Arc::new(CacheStatsCollector::new());
        let service = EmbeddingCacheService::new(make_tier_config(100), metrics);
        let result = service.get(&make_key("missing")).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_embedding_service_put_and_get() {
        let metrics = Arc::new(CacheStatsCollector::new());
        let service = EmbeddingCacheService::new(make_tier_config(100), metrics);
        let key = make_key("hello");
        let emb = vec![0.1f32, 0.2, 0.3];
        service.put(key.clone(), emb.clone()).await.expect("put should succeed");
        let result = service.get(&key).await;
        assert!(result.is_some());
        let retrieved = result.expect("result should be Some");
        assert_eq!(retrieved.len(), 3);
        assert!((retrieved[0] - 0.1).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_embedding_service_clear_removes_all() {
        let metrics = Arc::new(CacheStatsCollector::new());
        let service = EmbeddingCacheService::new(make_tier_config(100), metrics);
        service.put(make_key("a"), vec![0.1, 0.2]).await.expect("put should succeed");
        service.put(make_key("b"), vec![0.3, 0.4]).await.expect("put should succeed");
        service.clear().await.expect("clear should succeed");
        assert!(service.get(&make_key("a")).await.is_none());
        assert!(service.get(&make_key("b")).await.is_none());
    }

    #[tokio::test]
    async fn test_embedding_service_find_similar() {
        let metrics = Arc::new(CacheStatsCollector::new());
        let service = EmbeddingCacheService::new(make_tier_config(100), metrics);
        service
            .put(make_key("close"), vec![1.0, 0.0])
            .await
            .expect("put should succeed");
        service.put(make_key("far"), vec![0.0, 1.0]).await.expect("put should succeed");
        let results = service.find_similar(&[1.0, 0.0], 1).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.text, "close");
    }

    #[tokio::test]
    async fn test_embedding_service_stats_entry_count() {
        let metrics = Arc::new(CacheStatsCollector::new());
        let service = EmbeddingCacheService::new(make_tier_config(100), metrics);
        service.put(make_key("a"), vec![1.0]).await.expect("put should succeed");
        service.put(make_key("b"), vec![2.0]).await.expect("put should succeed");
        let stats = service.get_stats().await;
        assert_eq!(stats.entry_count, 2);
    }

    #[tokio::test]
    async fn test_embedding_service_stats_avg_dimension() {
        let metrics = Arc::new(CacheStatsCollector::new());
        let service = EmbeddingCacheService::new(make_tier_config(100), metrics);
        service
            .put(make_key("d4"), vec![1.0, 2.0, 3.0, 4.0])
            .await
            .expect("put should succeed");
        service.put(make_key("d2"), vec![1.0, 2.0]).await.expect("put should succeed");
        let stats = service.get_stats().await;
        assert!((stats.avg_dimension - 3.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_embedding_service_run_maintenance_no_panic() {
        let metrics = Arc::new(CacheStatsCollector::new());
        let service = EmbeddingCacheService::new(make_tier_config(100), metrics);
        service.put(make_key("m"), vec![0.5]).await.expect("put should succeed");
        let result = service.run_maintenance().await;
        assert!(result.is_ok());
    }
}
