//! Result Cache Implementation
//!
//! Caches inference results with TTL support and intelligent eviction policies.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

use super::config::{EvictionPolicy, TierConfig};
use super::metrics::CacheStatsCollector;

/// Cache key for result storage
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Model identifier
    pub model_id: String,
    /// Input text or tokens hash
    pub input_hash: u64,
    /// Generation parameters hash
    pub params_hash: u64,
    /// Model version
    pub model_version: Option<String>,
}

impl CacheKey {
    pub fn new(
        model_id: String,
        input: &str,
        params: &HashMap<String, serde_json::Value>,
        model_version: Option<String>,
    ) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        input.hash(&mut hasher);
        let input_hash = hasher.finish();

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for (k, v) in params.iter() {
            k.hash(&mut hasher);
            v.to_string().hash(&mut hasher);
        }
        let params_hash = hasher.finish();

        Self {
            model_id,
            input_hash,
            params_hash,
            model_version,
        }
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Cached result data
    pub result: CacheResult,
    /// Entry creation timestamp
    pub created_at: u64,
    /// Time-to-live in seconds
    pub ttl_seconds: u64,
    /// Access count
    pub access_count: u64,
    /// Last access timestamp
    pub last_accessed: u64,
    /// Entry size in bytes
    pub size_bytes: usize,
    /// Priority score for eviction
    pub priority: f32,
}

impl CacheEntry {
    pub fn new(result: CacheResult, ttl_seconds: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        let size_bytes = serde_json::to_vec(&result).unwrap_or_default().len();

        Self {
            result,
            created_at: now,
            ttl_seconds,
            access_count: 0,
            last_accessed: now,
            size_bytes,
            priority: 1.0,
        }
    }

    /// Check if entry is expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();
        now > self.created_at + self.ttl_seconds
    }

    /// Update access metadata
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        // Update priority based on access pattern
        let age_factor = 1.0 / (1.0 + (self.last_accessed - self.created_at) as f32 / 3600.0);
        let frequency_factor = (self.access_count as f32).ln_1p();
        self.priority = age_factor * frequency_factor;
    }
}

/// Cached inference result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheResult {
    /// Generated tokens or text
    pub output: String,
    /// Output tokens if available
    pub tokens: Option<Vec<u32>>,
    /// Logits if requested
    pub logits: Option<Vec<f32>>,
    /// Generation metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Model used for generation
    pub model_used: String,
}

/// Cache hit result
#[derive(Debug, Clone)]
pub struct CacheHit {
    pub result: CacheResult,
    pub metadata: CacheHitMetadata,
}

/// Cache hit metadata
#[derive(Debug, Clone)]
pub struct CacheHitMetadata {
    pub age_seconds: u64,
    pub access_count: u64,
    pub cache_efficiency: f32,
    pub saved_processing_time_ms: u64,
}

/// Cache miss reason
#[derive(Debug, Clone)]
pub enum CacheMiss {
    NotFound,
    Expired,
    InvalidEntry,
    PolicyRejected,
}

/// Result cache service
pub struct ResultCacheService {
    cache: Arc<RwLock<HashMap<CacheKey, CacheEntry>>>,
    config: TierConfig,
    metrics: Arc<CacheStatsCollector>,
    max_size_bytes: usize,
    current_size_bytes: Arc<RwLock<usize>>,
}

impl ResultCacheService {
    pub fn new(config: TierConfig, metrics: Arc<CacheStatsCollector>) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size_bytes: config.max_size_bytes,
            config,
            metrics,
            current_size_bytes: Arc::new(RwLock::new(0)),
        }
    }

    /// Get cached result
    pub async fn get(&self, key: &CacheKey) -> Option<CacheHit> {
        let mut cache = self.cache.write().await;

        if let Some(entry) = cache.get_mut(key) {
            // Check if expired
            if entry.is_expired() {
                cache.remove(key);
                self.metrics.record_cache_miss("result_cache", "expired").await;
                return None;
            }

            // Update access metadata
            entry.mark_accessed();

            // Record hit
            self.metrics.record_cache_hit("result_cache").await;

            let hit_metadata = CacheHitMetadata {
                age_seconds: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("SystemTime should be after UNIX_EPOCH")
                    .as_secs()
                    - entry.created_at,
                access_count: entry.access_count,
                cache_efficiency: entry.priority,
                saved_processing_time_ms: entry.result.processing_time_ms,
            };

            Some(CacheHit {
                result: entry.result.clone(),
                metadata: hit_metadata,
            })
        } else {
            self.metrics.record_cache_miss("result_cache", "not_found").await;
            None
        }
    }

    /// Store result in cache
    pub async fn put(&self, key: CacheKey, result: CacheResult) -> Result<()> {
        let ttl_seconds = self.config.default_ttl.as_secs();
        let entry = CacheEntry::new(result, ttl_seconds);

        // Check if we need to evict entries
        let new_size = *self.current_size_bytes.read().await + entry.size_bytes;
        if new_size > self.max_size_bytes {
            self.evict_entries(entry.size_bytes).await?;
        }

        // Insert entry
        {
            let mut cache = self.cache.write().await;
            if let Some(old_entry) = cache.insert(key.clone(), entry.clone()) {
                // Update size tracking
                let mut current_size = self.current_size_bytes.write().await;
                *current_size = current_size.saturating_sub(old_entry.size_bytes);
                *current_size += entry.size_bytes;
            } else {
                let mut current_size = self.current_size_bytes.write().await;
                *current_size += entry.size_bytes;
            }
        }

        self.metrics.record_cache_put("result_cache", entry.size_bytes).await;

        Ok(())
    }

    /// Invalidate specific key
    pub async fn invalidate(&self, key: &CacheKey) -> Result<()> {
        let mut cache = self.cache.write().await;
        if let Some(entry) = cache.remove(key) {
            let mut current_size = self.current_size_bytes.write().await;
            *current_size = current_size.saturating_sub(entry.size_bytes);

            self.metrics.record_cache_invalidation("result_cache").await;
        }

        Ok(())
    }

    /// Clear all cached entries
    pub async fn clear(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        let count = cache.len();
        cache.clear();

        let mut current_size = self.current_size_bytes.write().await;
        *current_size = 0;

        self.metrics.record_cache_clear("result_cache", count).await;

        Ok(())
    }

    /// Run maintenance tasks
    pub async fn run_maintenance(&self) -> Result<()> {
        self.cleanup_expired_entries().await?;
        self.update_priorities().await?;
        Ok(())
    }

    /// Update cache configuration
    pub async fn update_config(&self, _config: TierConfig) -> Result<()> {
        // Implement config update logic\n        // Note: Since config is not mutable, we apply changes to cache behavior\n        \n        // If max size decreased, trigger eviction to fit new limits\n        let new_max_size = _config.max_size_bytes;\n        if new_max_size < self.max_size_bytes {\n            let current_size = *self.current_size_bytes.read().await;\n            if current_size > new_max_size {\n                let bytes_to_evict = current_size - new_max_size;\n                self.evict_entries(bytes_to_evict).await?;\n            }\n        }\n        \n        // Apply new eviction policy effects immediately\n        match &config.eviction_policy {\n            EvictionPolicy::Ttl => {\n                // Clean up any expired entries with potentially new TTL settings\n                self.cleanup_expired_entries().await?;\n            },\n            EvictionPolicy::Lru => {\n                // LRU policy will be applied during next eviction\n                tracing::info!(\"Switched to LRU eviction policy\");\n            },\n            EvictionPolicy::Lfu => {\n                // LFU policy will be applied during next eviction  \n                tracing::info!(\"Switched to LFU eviction policy\");\n            },\n            EvictionPolicy::Priority => {\n                // Update priorities based on new configuration\n                self.update_priorities().await?;\n                tracing::info!(\"Switched to Priority-based eviction policy\");\n            },\n        }\n        \n        // Log configuration update\n        tracing::info!(\n            \"Result cache configuration updated: max_size={} bytes, eviction_policy={:?}\",\n            new_max_size,\n            config.eviction_policy\n        );
        Ok(())
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> ResultCacheStats {
        let cache = self.cache.read().await;
        let current_size = *self.current_size_bytes.read().await;

        ResultCacheStats {
            entry_count: cache.len(),
            total_size_bytes: current_size,
            hit_rate: self.metrics.get_hit_rate("result_cache").await,
            avg_entry_size: if cache.is_empty() { 0 } else { current_size / cache.len() },
            oldest_entry_age: cache
                .values()
                .map(|e| {
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("SystemTime should be after UNIX_EPOCH")
                        .as_secs()
                        - e.created_at
                })
                .max()
                .unwrap_or(0),
        }
    }

    // Private methods
    async fn evict_entries(&self, needed_space: usize) -> Result<()> {
        let mut cache = self.cache.write().await;
        let mut current_size = self.current_size_bytes.write().await;

        // Collect entries with their eviction priority and keys
        let mut entries_to_evict: Vec<_> = cache
            .iter()
            .map(|(key, entry)| {
                let sort_key = match self.config.eviction_policy {
                    EvictionPolicy::LRU => entry.last_accessed,
                    EvictionPolicy::LFU => entry.access_count,
                    EvictionPolicy::TTL => entry.created_at + entry.ttl_seconds,
                    EvictionPolicy::Priority => entry.priority as u64,
                    EvictionPolicy::Random => {
                        use scirs2_core::random::*;
                        let mut rng = thread_rng();
                        rng.random::<u64>()
                    },
                    EvictionPolicy::FIFO => entry.created_at,
                };
                (key.clone(), sort_key, entry.size_bytes)
            })
            .collect();

        // Sort by the appropriate key
        entries_to_evict.sort_by_key(|(_, sort_key, _)| *sort_key);

        // Evict entries until we have enough space
        let mut freed_space = 0;
        let target_space = needed_space + (self.max_size_bytes / 10); // 10% buffer

        for (key, _, size) in entries_to_evict {
            if freed_space >= target_space {
                break;
            }

            if cache.remove(&key).is_some() {
                freed_space += size;
                *current_size = current_size.saturating_sub(size);

                self.metrics.record_cache_eviction("result_cache", "size_limit").await;
            }
        }

        Ok(())
    }

    async fn cleanup_expired_entries(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        let mut current_size = self.current_size_bytes.write().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        let expired_keys: Vec<_> = cache
            .iter()
            .filter(|(_, entry)| now > entry.created_at + entry.ttl_seconds)
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            if let Some(entry) = cache.remove(&key) {
                *current_size = current_size.saturating_sub(entry.size_bytes);
                self.metrics.record_cache_eviction("result_cache", "expired").await;
            }
        }

        Ok(())
    }

    async fn update_priorities(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        for entry in cache.values_mut() {
            // Recalculate priority based on current time
            let age_factor = 1.0 / (1.0 + (now - entry.created_at) as f32 / 3600.0);
            let frequency_factor = (entry.access_count as f32).ln_1p();
            let recency_factor = 1.0 / (1.0 + (now - entry.last_accessed) as f32 / 300.0);

            entry.priority = age_factor * frequency_factor * recency_factor;
        }

        Ok(())
    }
}

/// Result cache statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResultCacheStats {
    pub entry_count: usize,
    pub total_size_bytes: usize,
    pub hit_rate: f32,
    pub avg_entry_size: usize,
    pub oldest_entry_age: u64,
}
