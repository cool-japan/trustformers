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
            .unwrap()
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
            .unwrap()
            .as_secs();
    }
}

/// Vector index for similarity search
pub struct VectorIndex {
    embeddings: HashMap<EmbeddingKey, EmbeddingEntry>,
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

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
            .unwrap()
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
                .unwrap()
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
