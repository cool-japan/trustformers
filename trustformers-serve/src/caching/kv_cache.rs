//! KV Cache Implementation
//!
//! Manages key-value caches for transformer attention layers with sharing capabilities.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::config::KVCacheConfig;
use super::metrics::CacheStatsCollector;

/// KV cache slot identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KVCacheSlot {
    pub slot_id: usize,
    pub layer_id: usize,
    pub sequence_id: u64,
}

/// Key-Value cache entry for a single attention head
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheEntry {
    /// Key tensor data
    pub keys: Vec<f32>,

    /// Value tensor data
    pub values: Vec<f32>,

    /// Sequence length
    pub seq_len: usize,

    /// Head dimension
    pub head_dim: usize,

    /// Number of heads
    pub num_heads: usize,

    /// Last access timestamp
    pub last_accessed: u64,

    /// Access count
    pub access_count: u64,

    /// Reference count for sharing
    pub ref_count: u32,

    /// Entry size in bytes
    pub size_bytes: usize,
}

impl KVCacheEntry {
    pub fn new(
        keys: Vec<f32>,
        values: Vec<f32>,
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> Self {
        let size_bytes = (keys.len() + values.len()) * std::mem::size_of::<f32>();

        Self {
            keys,
            values,
            seq_len,
            head_dim,
            num_heads,
            last_accessed: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            access_count: 0,
            ref_count: 1,
            size_bytes,
        }
    }

    /// Mark entry as accessed
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.last_accessed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Increment reference count
    pub fn add_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Decrement reference count
    pub fn remove_ref(&mut self) -> bool {
        self.ref_count = self.ref_count.saturating_sub(1);
        self.ref_count == 0
    }

    /// Extend cache with new tokens
    pub fn extend(&mut self, new_keys: &[f32], new_values: &[f32], new_seq_len: usize) {
        self.keys.extend_from_slice(new_keys);
        self.values.extend_from_slice(new_values);
        self.seq_len = new_seq_len;
        self.size_bytes = (self.keys.len() + self.values.len()) * std::mem::size_of::<f32>();
        self.mark_accessed();
    }
}

/// Layer-specific cache
#[derive(Debug)]
pub struct LayerCache {
    layer_id: usize,
    entries: HashMap<u64, KVCacheEntry>, // sequence_id -> entry
    max_entries: usize,
    current_size_bytes: usize,
    max_size_bytes: usize,
}

impl LayerCache {
    pub fn new(layer_id: usize, max_entries: usize, max_size_bytes: usize) -> Self {
        Self {
            layer_id,
            entries: HashMap::new(),
            max_entries,
            current_size_bytes: 0,
            max_size_bytes,
        }
    }

    /// Get KV cache entry for sequence
    pub fn get(&mut self, sequence_id: u64) -> Option<&mut KVCacheEntry> {
        if let Some(entry) = self.entries.get_mut(&sequence_id) {
            entry.mark_accessed();
            Some(entry)
        } else {
            None
        }
    }

    /// Store KV cache entry
    pub fn put(&mut self, sequence_id: u64, entry: KVCacheEntry) -> Result<()> {
        // Check if we need to evict
        if self.entries.len() >= self.max_entries
            || self.current_size_bytes + entry.size_bytes > self.max_size_bytes
        {
            self.evict_entries(entry.size_bytes)?;
        }

        // Update size tracking
        if let Some(old_entry) = self.entries.insert(sequence_id, entry.clone()) {
            self.current_size_bytes = self.current_size_bytes.saturating_sub(old_entry.size_bytes);
        }
        self.current_size_bytes += entry.size_bytes;

        Ok(())
    }

    /// Remove entry
    pub fn remove(&mut self, sequence_id: u64) -> Option<KVCacheEntry> {
        if let Some(entry) = self.entries.remove(&sequence_id) {
            self.current_size_bytes = self.current_size_bytes.saturating_sub(entry.size_bytes);
            Some(entry)
        } else {
            None
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_size_bytes = 0;
    }

    /// Get statistics
    pub fn get_stats(&self) -> LayerCacheStats {
        LayerCacheStats {
            layer_id: self.layer_id,
            entry_count: self.entries.len(),
            size_bytes: self.current_size_bytes,
            avg_seq_len: if self.entries.is_empty() {
                0.0
            } else {
                self.entries.values().map(|e| e.seq_len as f32).sum::<f32>()
                    / self.entries.len() as f32
            },
        }
    }

    /// Evict entries to make space
    fn evict_entries(&mut self, needed_space: usize) -> Result<()> {
        let target_space = needed_space + (self.max_size_bytes / 10); // 10% buffer
        let mut freed_space = 0;

        // Sort by priority (LRU for now) and collect sequence IDs to evict
        let mut entries_to_evict: Vec<_> = self
            .entries
            .iter()
            .map(|(id, entry)| (*id, entry.last_accessed, entry.ref_count, entry.size_bytes))
            .collect();
        entries_to_evict
            .sort_by_key(|(_, last_accessed, ref_count, _)| (*last_accessed, *ref_count));

        for (sequence_id, _, ref_count, size) in entries_to_evict {
            if freed_space >= target_space {
                break;
            }

            // Don't evict entries with references
            if ref_count > 0 {
                continue;
            }

            if self.entries.remove(&sequence_id).is_some() {
                self.current_size_bytes = self.current_size_bytes.saturating_sub(size);
                freed_space += size;
            }
        }

        Ok(())
    }
}

/// Attention cache for transformer layers
pub struct AttentionCache {
    layers: Vec<LayerCache>,
    config: KVCacheConfig,
}

impl AttentionCache {
    pub fn new(config: KVCacheConfig) -> Self {
        let mut layers = Vec::with_capacity(config.max_layers);
        let layer_max_size = config.max_size_bytes / config.max_layers;
        let layer_max_entries = config.max_sequences / config.max_layers;

        for i in 0..config.max_layers {
            layers.push(LayerCache::new(i, layer_max_entries, layer_max_size));
        }

        Self { layers, config }
    }

    /// Get KV cache for specific layer and sequence
    pub fn get(&mut self, layer_id: usize, sequence_id: u64) -> Option<&mut KVCacheEntry> {
        if layer_id >= self.layers.len() {
            return None;
        }

        self.layers[layer_id].get(sequence_id)
    }

    /// Store KV cache for specific layer and sequence
    pub fn put(&mut self, layer_id: usize, sequence_id: u64, entry: KVCacheEntry) -> Result<()> {
        if layer_id >= self.layers.len() {
            return Err(anyhow::anyhow!("Layer ID {} out of bounds", layer_id));
        }

        self.layers[layer_id].put(sequence_id, entry)
    }

    /// Remove KV cache for specific layer and sequence
    pub fn remove(&mut self, layer_id: usize, sequence_id: u64) -> Option<KVCacheEntry> {
        if layer_id >= self.layers.len() {
            return None;
        }

        self.layers[layer_id].remove(sequence_id)
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Get statistics for all layers
    pub fn get_stats(&self) -> Vec<LayerCacheStats> {
        self.layers.iter().map(|layer| layer.get_stats()).collect()
    }
}

/// Shared KV cache with reference counting
pub struct SharedKVCache {
    cache: Arc<RwLock<AttentionCache>>,
    sequence_refs: Arc<RwLock<HashMap<u64, usize>>>, // sequence_id -> ref_count
    metrics: Arc<CacheStatsCollector>,
}

impl SharedKVCache {
    pub fn new(config: KVCacheConfig, metrics: Arc<CacheStatsCollector>) -> Self {
        Self {
            cache: Arc::new(RwLock::new(AttentionCache::new(config))),
            sequence_refs: Arc::new(RwLock::new(HashMap::new())),
            metrics,
        }
    }

    /// Get shared reference to KV cache
    pub async fn get_shared(&self, layer_id: usize, sequence_id: u64) -> Option<SharedKVCacheRef> {
        let mut cache = self.cache.write().await;

        if let Some(entry) = cache.get(layer_id, sequence_id) {
            entry.add_ref();

            // Update reference count
            let mut refs = self.sequence_refs.write().await;
            *refs.entry(sequence_id).or_insert(0) += 1;

            self.metrics.record_cache_hit("kv_cache").await;

            Some(SharedKVCacheRef {
                layer_id,
                sequence_id,
                cache: self.cache.clone(),
                sequence_refs: self.sequence_refs.clone(),
            })
        } else {
            self.metrics.record_cache_miss("kv_cache", "not_found").await;
            None
        }
    }

    /// Store KV cache with sharing enabled
    pub async fn put_shared(
        &self,
        layer_id: usize,
        sequence_id: u64,
        entry: KVCacheEntry,
    ) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.put(layer_id, sequence_id, entry)?;

        self.metrics.record_cache_put("kv_cache", 0).await; // Size tracked in AttentionCache

        Ok(())
    }

    /// Clear all caches
    pub async fn clear(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.clear();

        let mut refs = self.sequence_refs.write().await;
        refs.clear();

        Ok(())
    }
}

/// Shared reference to KV cache entry
pub struct SharedKVCacheRef {
    layer_id: usize,
    sequence_id: u64,
    cache: Arc<RwLock<AttentionCache>>,
    sequence_refs: Arc<RwLock<HashMap<u64, usize>>>,
}

impl Drop for SharedKVCacheRef {
    fn drop(&mut self) {
        // Async-safe cleanup using try_lock to avoid blocking
        // Try non-blocking cleanup first
        if let Ok(mut refs) = self.sequence_refs.try_write() {
            if let Some(count) = refs.get_mut(&self.sequence_id) {
                *count = count.saturating_sub(1);

                if *count == 0 {
                    refs.remove(&self.sequence_id);

                    // Try non-blocking cache cleanup
                    if let Ok(mut cache) = self.cache.try_write() {
                        if let Some(mut entry) = cache.remove(self.layer_id, self.sequence_id) {
                            if entry.remove_ref() {
                                // Entry can be safely removed
                            }
                        }
                    } else {
                        // If cache is locked, defer cleanup using runtime handle
                        let cache = self.cache.clone();
                        let layer_id = self.layer_id;
                        let sequence_id = self.sequence_id;

                        // Use Handle::try_current() for async-safe spawning
                        if let Ok(handle) = tokio::runtime::Handle::try_current() {
                            handle.spawn(async move {
                                let mut cache = cache.write().await;
                                if let Some(mut entry) = cache.remove(layer_id, sequence_id) {
                                    if entry.remove_ref() {
                                        // Entry can be safely removed
                                    }
                                }
                            });
                        }
                        // If no runtime available, cleanup will be deferred
                    }
                }
            }
        } else {
            // If sequence_refs is locked, spawn async cleanup as fallback
            let cache = self.cache.clone();
            let sequence_refs = self.sequence_refs.clone();
            let layer_id = self.layer_id;
            let sequence_id = self.sequence_id;

            // Use Handle::try_current() for async-safe spawning
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    let mut refs = sequence_refs.write().await;
                    if let Some(count) = refs.get_mut(&sequence_id) {
                        *count = count.saturating_sub(1);

                        if *count == 0 {
                            refs.remove(&sequence_id);

                            // Remove from cache if no references
                            let mut cache = cache.write().await;
                            if let Some(mut entry) = cache.remove(layer_id, sequence_id) {
                                if entry.remove_ref() {
                                    // Entry can be safely removed
                                }
                            }
                        }
                    }
                });
            }
            // If no runtime available, cleanup will be deferred
        }
    }
}

/// KV Cache manager
pub struct KVCacheManager {
    shared_cache: SharedKVCache,
    config: KVCacheConfig,
    metrics: Arc<CacheStatsCollector>,
}

impl KVCacheManager {
    pub fn new(config: KVCacheConfig, metrics: Arc<CacheStatsCollector>) -> Self {
        let shared_cache = SharedKVCache::new(config.clone(), metrics.clone());

        Self {
            shared_cache,
            config,
            metrics,
        }
    }

    /// Get KV cache entry
    pub async fn get(&self, layer_id: usize, sequence_id: u64) -> Option<SharedKVCacheRef> {
        self.shared_cache.get_shared(layer_id, sequence_id).await
    }

    /// Store KV cache entry
    pub async fn put(
        &self,
        layer_id: usize,
        sequence_id: u64,
        keys: Vec<f32>,
        values: Vec<f32>,
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> Result<()> {
        let entry = KVCacheEntry::new(keys, values, seq_len, head_dim, num_heads);
        self.shared_cache.put_shared(layer_id, sequence_id, entry).await
    }

    /// Clear all caches
    pub async fn clear(&self) -> Result<()> {
        self.shared_cache.clear().await
    }

    /// Update configuration
    pub async fn update_config(&self, config: KVCacheConfig) -> Result<()> {
        // Update the shared cache configuration
        {
            let mut cache = self.shared_cache.cache.write().await;
            cache.config = config.clone();

            // If the configuration changed significantly, we need to adjust layer caches
            let layer_max_size = config.max_size_bytes / config.max_layers;
            let layer_max_sequences = config.max_sequences / config.max_layers;

            // Update each layer's configuration
            for layer in &mut cache.layers {
                layer.max_size_bytes = layer_max_size;
                layer.max_entries = layer_max_sequences;

                // Trigger eviction if current usage exceeds new limits
                if layer.current_size_bytes > layer.max_size_bytes {
                    let excess_size = layer.current_size_bytes - layer.max_size_bytes;
                    let _ = layer.evict_entries(excess_size);
                }

                if layer.entries.len() > layer.max_entries {
                    let excess_entries = layer.entries.len() - layer.max_entries;
                    let _ =
                        layer.evict_entries(excess_entries * std::mem::size_of::<KVCacheEntry>());
                }
            }
        }

        // Note: Manager config is updated through the shared cache
        tracing::info!("KV cache configuration updated successfully");
        Ok(())
    }

    /// Run maintenance tasks
    pub async fn run_maintenance(&self) -> Result<()> {
        let mut total_entries_cleaned = 0;
        let mut total_bytes_freed = 0;
        let mut orphaned_refs_cleaned = 0;

        // Get mutable access to the cache and sequence refs
        {
            let mut cache = self.shared_cache.cache.write().await;
            let mut refs = self.shared_cache.sequence_refs.write().await;

            // Clean up each layer
            for layer in &mut cache.layers {
                let mut layer_entries_cleaned = 0;
                let mut layer_bytes_freed = 0;

                // Collect sequence IDs to remove (entries with zero ref count)
                let mut to_remove = Vec::new();
                for (sequence_id, entry) in &layer.entries {
                    if entry.ref_count == 0 {
                        to_remove.push(*sequence_id);
                    }
                }

                // Remove the identified entries
                for sequence_id in to_remove {
                    if let Some(removed_entry) = layer.remove(sequence_id) {
                        layer_bytes_freed += removed_entry.size_bytes;
                        layer_entries_cleaned += 1;

                        // Also remove from sequence references
                        refs.remove(&sequence_id);
                    }
                }

                // Trigger eviction if we're over size limits
                let current_size = layer.current_size_bytes;
                if current_size > layer.max_size_bytes {
                    let bytes_to_evict = current_size - (layer.max_size_bytes * 9 / 10);
                    let _ = layer.evict_entries(bytes_to_evict);
                }

                total_entries_cleaned += layer_entries_cleaned;
                total_bytes_freed += layer_bytes_freed;
            }

            // Clean up orphaned sequence references
            let mut to_remove_refs = Vec::new();
            for sequence_id in refs.keys() {
                let mut found = false;
                for layer in &cache.layers {
                    if layer.entries.contains_key(sequence_id) {
                        found = true;
                        break;
                    }
                }
                if !found {
                    to_remove_refs.push(*sequence_id);
                }
            }

            for sequence_id in to_remove_refs {
                refs.remove(&sequence_id);
                orphaned_refs_cleaned += 1;
            }
        }

        // Log maintenance results
        tracing::info!(
            entries_cleaned = total_entries_cleaned,
            bytes_freed = total_bytes_freed,
            orphaned_refs_cleaned = orphaned_refs_cleaned,
            "KV cache maintenance completed"
        );

        Ok(())

        /* Original implementation commented out due to AttentionCache API changes
        // Clean up unused entries and enforce eviction policies
        {
            let mut cache = self.shared_cache.cache.write().await;
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // Remove entries with zero reference count
            let mut to_remove = Vec::new();
            for ((layer_id, seq_id), entry) in &cache.entries {
                // Commented out due to API changes
            }
        }
        */
    }

    /// Calculate sharing ratio for cache efficiency
    async fn calculate_sharing_ratio(&self, cache: &AttentionCache) -> f32 {
        let mut total_entries = 0;
        let mut shared_entries = 0;

        // Count total entries and shared entries across all layers
        for layer in &cache.layers {
            for entry in layer.entries.values() {
                total_entries += 1;
                if entry.ref_count > 1 {
                    shared_entries += 1;
                }
            }
        }

        if total_entries > 0 {
            shared_entries as f32 / total_entries as f32
        } else {
            0.0
        }
    }

    /// Get KV cache statistics
    pub async fn get_stats(&self) -> KVCacheStats {
        let cache = self.shared_cache.cache.read().await;
        let layer_stats = cache.get_stats();

        let total_entries: usize = layer_stats.iter().map(|s| s.entry_count).sum();
        let total_size: usize = layer_stats.iter().map(|s| s.size_bytes).sum();

        KVCacheStats {
            layer_stats,
            total_entries,
            total_size_bytes: total_size,
            hit_rate: self.metrics.get_hit_rate("kv_cache").await,
            sharing_ratio: self.calculate_sharing_ratio(&cache).await,
        }
    }
}

/// Layer cache statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerCacheStats {
    pub layer_id: usize,
    pub entry_count: usize,
    pub size_bytes: usize,
    pub avg_seq_len: f32,
}

/// KV cache statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct KVCacheStats {
    pub layer_stats: Vec<LayerCacheStats>,
    pub total_entries: usize,
    pub total_size_bytes: usize,
    pub hit_rate: f32,
    pub sharing_ratio: f32,
}
