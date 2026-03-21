//! Edge Caching Strategies for Optimized Performance
//!
//! This module provides intelligent caching mechanisms for edge deployments
//! to improve performance, reduce latency, and optimize resource utilization
//! across global edge locations.

use crate::runtime::geo_distribution::GeoRegion;
use std::collections::BTreeMap;
use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::js_sys;

/// Cache entry types for different kinds of data
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CacheEntryType {
    /// Model weights and configuration
    Model,
    /// Inference results for common inputs
    InferenceResult,
    /// Tokenization results
    TokenizationResult,
    /// Preprocessed input data
    PreprocessedInput,
    /// Attention patterns
    AttentionPatterns,
    /// Embeddings
    Embeddings,
    /// Raw user data
    RawData,
}

/// Cache eviction policies
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based expiration
    TTL,
    /// Size-based (remove largest entries first)
    SizeBased,
    /// Adaptive (uses multiple factors)
    Adaptive,
}

/// Cache consistency levels
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsistencyLevel {
    /// Eventual consistency (best performance)
    Eventual,
    /// Strong consistency (guarantees correctness)
    Strong,
    /// Session consistency (consistent within user session)
    Session,
    /// Monotonic read consistency
    MonotonicRead,
}

/// Cache replication strategies
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReplicationStrategy {
    /// No replication (single copy)
    None,
    /// Replicate to nearest edge locations
    Nearest,
    /// Replicate to all edge locations in region
    Regional,
    /// Replicate to all edge locations globally
    Global,
    /// Adaptive replication based on access patterns
    Adaptive,
}

/// Cache entry metadata
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct CacheEntry {
    key: String,
    entry_type: CacheEntryType,
    data: Vec<u8>,
    size_bytes: usize,
    created_at: u64,
    last_accessed: u64,
    access_count: u32,
    ttl_ms: Option<u64>,
    region: GeoRegion,
    priority: f32,
    checksum: String,
    compression_ratio: f32,
}

#[wasm_bindgen]
impl CacheEntry {
    /// Create a new cache entry
    #[wasm_bindgen(constructor)]
    pub fn new(
        key: String,
        entry_type: CacheEntryType,
        data: Vec<u8>,
        ttl_ms: Option<u64>,
        region: GeoRegion,
        priority: f32,
    ) -> CacheEntry {
        let current_time = js_sys::Date::now() as u64;
        let size_bytes = data.len();
        let checksum = Self::calculate_checksum(&data);

        CacheEntry {
            key,
            entry_type,
            data,
            size_bytes,
            created_at: current_time,
            last_accessed: current_time,
            access_count: 0,
            ttl_ms,
            region,
            priority,
            checksum,
            compression_ratio: 1.0,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn key(&self) -> String {
        self.key.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn entry_type(&self) -> CacheEntryType {
        self.entry_type
    }

    #[wasm_bindgen(getter)]
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    #[wasm_bindgen(getter)]
    pub fn last_accessed(&self) -> u64 {
        self.last_accessed
    }

    #[wasm_bindgen(getter)]
    pub fn access_count(&self) -> u32 {
        self.access_count
    }

    #[wasm_bindgen(getter)]
    pub fn ttl_ms(&self) -> Option<u64> {
        self.ttl_ms
    }

    #[wasm_bindgen(getter)]
    pub fn region(&self) -> GeoRegion {
        self.region
    }

    #[wasm_bindgen(getter)]
    pub fn priority(&self) -> f32 {
        self.priority
    }

    #[wasm_bindgen(getter)]
    pub fn checksum(&self) -> String {
        self.checksum.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn compression_ratio(&self) -> f32 {
        self.compression_ratio
    }

    /// Get the cached data
    pub fn data(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Check if the entry has expired
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl_ms {
            let current_time = js_sys::Date::now() as u64;
            current_time > self.created_at + ttl
        } else {
            false
        }
    }

    /// Get the age of the entry in milliseconds
    pub fn age_ms(&self) -> u64 {
        let current_time = js_sys::Date::now() as u64;
        current_time - self.created_at
    }

    /// Get time since last access in milliseconds
    pub fn time_since_last_access_ms(&self) -> u64 {
        let current_time = js_sys::Date::now() as u64;
        current_time - self.last_accessed
    }

    /// Update access statistics
    pub fn update_access(&mut self) {
        self.last_accessed = js_sys::Date::now() as u64;
        self.access_count += 1;
    }

    /// Calculate access frequency (accesses per hour)
    pub fn access_frequency(&self) -> f32 {
        let age_hours = self.age_ms() as f32 / (1000.0 * 3600.0);
        if age_hours > 0.0 {
            self.access_count as f32 / age_hours
        } else {
            0.0
        }
    }

    /// Calculate cache efficiency score
    pub fn efficiency_score(&self) -> f32 {
        let age_factor = 1.0 - (self.age_ms() as f32 / (24.0 * 3600.0 * 1000.0)).min(1.0); // Favor newer entries
        let access_factor = (self.access_count as f32).ln().max(0.0); // Logarithmic access benefit
        let size_factor = 1.0 - (self.size_bytes as f32 / (100.0 * 1024.0 * 1024.0)).min(1.0); // Favor smaller entries
        let compression_factor = 1.0 / self.compression_ratio; // Favor compressed entries

        (age_factor + access_factor + size_factor + compression_factor) * self.priority
    }

    /// Verify data integrity
    pub fn verify_integrity(&self) -> bool {
        Self::calculate_checksum(&self.data) == self.checksum
    }

    /// Compress entry data
    pub fn compress(&mut self) -> Result<(), JsValue> {
        // Simple compression simulation (in real implementation, use actual compression)
        let original_size = self.data.len();

        // Simulate compression effect
        let compression_factor = match self.entry_type {
            CacheEntryType::Model => 0.6,              // Models compress well
            CacheEntryType::InferenceResult => 0.8,    // Results compress moderately
            CacheEntryType::TokenizationResult => 0.9, // Tokens don't compress much
            CacheEntryType::PreprocessedInput => 0.7,
            CacheEntryType::AttentionPatterns => 0.5, // Patterns compress very well
            CacheEntryType::Embeddings => 0.8,
            CacheEntryType::RawData => 0.9,
        };

        let compressed_size = (original_size as f32 * compression_factor) as usize;
        self.data.resize(compressed_size, 0);
        self.size_bytes = compressed_size;
        self.compression_ratio = original_size as f32 / compressed_size as f32;

        Ok(())
    }

    /// Decompress entry data
    pub fn decompress(&mut self) -> Result<(), JsValue> {
        // In real implementation, actually decompress the data
        let original_size = (self.size_bytes as f32 * self.compression_ratio) as usize;
        self.data.resize(original_size, 0);
        self.size_bytes = original_size;
        self.compression_ratio = 1.0;

        Ok(())
    }
}

impl CacheEntry {
    /// Calculate checksum for data integrity
    fn calculate_checksum(data: &[u8]) -> String {
        // Simple checksum calculation (in real implementation, use proper hashing)
        let mut checksum = 0u32;
        for byte in data {
            checksum = checksum.wrapping_add(*byte as u32);
        }
        format!("{:08x}", checksum)
    }
}

/// Cache statistics
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    total_entries: usize,
    total_size_bytes: usize,
    hit_count: u64,
    miss_count: u64,
    eviction_count: u64,
    compression_ratio: f32,
    average_age_ms: u64,
    memory_usage_bytes: usize,
    network_bytes_saved: usize,
    latency_improvement_ms: f32,
}

#[wasm_bindgen]
impl CacheStatistics {
    #[wasm_bindgen(getter)]
    pub fn total_entries(&self) -> usize {
        self.total_entries
    }

    #[wasm_bindgen(getter)]
    pub fn total_size_bytes(&self) -> usize {
        self.total_size_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn hit_count(&self) -> u64 {
        self.hit_count
    }

    #[wasm_bindgen(getter)]
    pub fn miss_count(&self) -> u64 {
        self.miss_count
    }

    #[wasm_bindgen(getter)]
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    #[wasm_bindgen(getter)]
    pub fn compression_ratio(&self) -> f32 {
        self.compression_ratio
    }

    #[wasm_bindgen(getter)]
    pub fn average_age_ms(&self) -> u64 {
        self.average_age_ms
    }

    #[wasm_bindgen(getter)]
    pub fn memory_usage_bytes(&self) -> usize {
        self.memory_usage_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn network_bytes_saved(&self) -> usize {
        self.network_bytes_saved
    }

    #[wasm_bindgen(getter)]
    pub fn latency_improvement_ms(&self) -> f32 {
        self.latency_improvement_ms
    }

    /// Calculate cache hit rate
    pub fn hit_rate(&self) -> f32 {
        let total_requests = self.hit_count + self.miss_count;
        if total_requests > 0 {
            self.hit_count as f32 / total_requests as f32
        } else {
            0.0
        }
    }

    /// Calculate cache efficiency
    pub fn efficiency(&self) -> f32 {
        let hit_rate = self.hit_rate();
        let compression_efficiency = self.compression_ratio;
        let memory_efficiency = if self.memory_usage_bytes > 0 {
            self.network_bytes_saved as f32 / self.memory_usage_bytes as f32
        } else {
            0.0
        };

        (hit_rate + compression_efficiency + memory_efficiency) / 3.0
    }
}

/// Cache configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub(crate) max_size_bytes: usize,
    pub(crate) max_entries: usize,
    pub(crate) eviction_policy: EvictionPolicy,
    pub(crate) consistency_level: ConsistencyLevel,
    pub(crate) replication_strategy: ReplicationStrategy,
    pub(crate) enable_compression: bool,
    pub(crate) enable_encryption: bool,
    pub(crate) default_ttl_ms: Option<u64>,
    pub(crate) prefetch_enabled: bool,
    pub(crate) prefetch_threshold: f32,
    pub(crate) background_cleanup_interval_ms: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl CacheConfig {
    /// Create a new cache configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> CacheConfig {
        CacheConfig {
            max_size_bytes: 100 * 1024 * 1024, // 100MB default
            max_entries: 1000,
            eviction_policy: EvictionPolicy::LRU,
            consistency_level: ConsistencyLevel::Eventual,
            replication_strategy: ReplicationStrategy::Nearest,
            enable_compression: true,
            enable_encryption: false,
            default_ttl_ms: Some(3600000), // 1 hour default
            prefetch_enabled: true,
            prefetch_threshold: 0.8,
            background_cleanup_interval_ms: 300000, // 5 minutes
        }
    }

    /// Create configuration optimized for performance
    pub fn for_performance() -> CacheConfig {
        CacheConfig {
            max_size_bytes: 500 * 1024 * 1024, // 500MB
            max_entries: 5000,
            eviction_policy: EvictionPolicy::Adaptive,
            consistency_level: ConsistencyLevel::Eventual,
            replication_strategy: ReplicationStrategy::Regional,
            enable_compression: true,
            enable_encryption: false,
            default_ttl_ms: Some(7200000), // 2 hours
            prefetch_enabled: true,
            prefetch_threshold: 0.9,
            background_cleanup_interval_ms: 180000, // 3 minutes
        }
    }

    /// Create configuration optimized for memory usage
    pub fn for_memory_efficiency() -> CacheConfig {
        CacheConfig {
            max_size_bytes: 50 * 1024 * 1024, // 50MB
            max_entries: 500,
            eviction_policy: EvictionPolicy::LRU,
            consistency_level: ConsistencyLevel::Eventual,
            replication_strategy: ReplicationStrategy::None,
            enable_compression: true,
            enable_encryption: false,
            default_ttl_ms: Some(1800000), // 30 minutes
            prefetch_enabled: false,
            prefetch_threshold: 0.7,
            background_cleanup_interval_ms: 600000, // 10 minutes
        }
    }

    /// Create configuration for edge computing
    pub fn for_edge_computing() -> CacheConfig {
        CacheConfig {
            max_size_bytes: 200 * 1024 * 1024, // 200MB
            max_entries: 2000,
            eviction_policy: EvictionPolicy::Adaptive,
            consistency_level: ConsistencyLevel::Session,
            replication_strategy: ReplicationStrategy::Adaptive,
            enable_compression: true,
            enable_encryption: true,
            default_ttl_ms: Some(3600000), // 1 hour
            prefetch_enabled: true,
            prefetch_threshold: 0.8,
            background_cleanup_interval_ms: 240000, // 4 minutes
        }
    }

    // Getters and setters
    #[wasm_bindgen(getter)]
    pub fn max_size_bytes(&self) -> usize {
        self.max_size_bytes
    }

    #[wasm_bindgen(setter)]
    pub fn set_max_size_bytes(&mut self, value: usize) {
        self.max_size_bytes = value;
    }

    #[wasm_bindgen(getter)]
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    #[wasm_bindgen(setter)]
    pub fn set_max_entries(&mut self, value: usize) {
        self.max_entries = value;
    }

    #[wasm_bindgen(getter)]
    pub fn eviction_policy(&self) -> EvictionPolicy {
        self.eviction_policy
    }

    #[wasm_bindgen(setter)]
    pub fn set_eviction_policy(&mut self, value: EvictionPolicy) {
        self.eviction_policy = value;
    }

    #[wasm_bindgen(getter)]
    pub fn consistency_level(&self) -> ConsistencyLevel {
        self.consistency_level
    }

    #[wasm_bindgen(setter)]
    pub fn set_consistency_level(&mut self, value: ConsistencyLevel) {
        self.consistency_level = value;
    }

    #[wasm_bindgen(getter)]
    pub fn replication_strategy(&self) -> ReplicationStrategy {
        self.replication_strategy
    }

    #[wasm_bindgen(setter)]
    pub fn set_replication_strategy(&mut self, value: ReplicationStrategy) {
        self.replication_strategy = value;
    }

    #[wasm_bindgen(getter)]
    pub fn enable_compression(&self) -> bool {
        self.enable_compression
    }

    #[wasm_bindgen(setter)]
    pub fn set_enable_compression(&mut self, value: bool) {
        self.enable_compression = value;
    }

    #[wasm_bindgen(getter)]
    pub fn enable_encryption(&self) -> bool {
        self.enable_encryption
    }

    #[wasm_bindgen(setter)]
    pub fn set_enable_encryption(&mut self, value: bool) {
        self.enable_encryption = value;
    }

    #[wasm_bindgen(getter)]
    pub fn default_ttl_ms(&self) -> Option<u64> {
        self.default_ttl_ms
    }

    #[wasm_bindgen(setter)]
    pub fn set_default_ttl_ms(&mut self, value: Option<u64>) {
        self.default_ttl_ms = value;
    }

    #[wasm_bindgen(getter)]
    pub fn prefetch_enabled(&self) -> bool {
        self.prefetch_enabled
    }

    #[wasm_bindgen(setter)]
    pub fn set_prefetch_enabled(&mut self, value: bool) {
        self.prefetch_enabled = value;
    }

    #[wasm_bindgen(getter)]
    pub fn prefetch_threshold(&self) -> f32 {
        self.prefetch_threshold
    }

    #[wasm_bindgen(setter)]
    pub fn set_prefetch_threshold(&mut self, value: f32) {
        self.prefetch_threshold = value.clamp(0.0, 1.0);
    }
}

/// Edge cache manager
#[wasm_bindgen]
pub struct EdgeCacheManager {
    config: CacheConfig,
    entries: BTreeMap<String, CacheEntry>,
    statistics: CacheStatistics,
    region: GeoRegion,
    last_cleanup: u64,
    prefetch_queue: Vec<String>,
    replication_peers: Vec<String>,
}

#[wasm_bindgen]
impl EdgeCacheManager {
    /// Create a new edge cache manager
    #[wasm_bindgen(constructor)]
    pub fn new(config: CacheConfig, region: GeoRegion) -> EdgeCacheManager {
        EdgeCacheManager {
            config,
            entries: BTreeMap::new(),
            statistics: CacheStatistics {
                total_entries: 0,
                total_size_bytes: 0,
                hit_count: 0,
                miss_count: 0,
                eviction_count: 0,
                compression_ratio: 1.0,
                average_age_ms: 0,
                memory_usage_bytes: 0,
                network_bytes_saved: 0,
                latency_improvement_ms: 0.0,
            },
            region,
            last_cleanup: js_sys::Date::now() as u64,
            prefetch_queue: Vec::new(),
            replication_peers: Vec::new(),
        }
    }

    /// Add or update a cache entry
    pub fn put(
        &mut self,
        key: &str,
        entry_type: CacheEntryType,
        data: Vec<u8>,
        ttl_ms: Option<u64>,
        priority: f32,
    ) -> Result<(), JsValue> {
        let mut entry = CacheEntry::new(
            key.to_string(),
            entry_type,
            data,
            ttl_ms.or(self.config.default_ttl_ms),
            self.region,
            priority,
        );

        // Apply compression if enabled
        if self.config.enable_compression {
            entry.compress()?;
        }

        // Check if we need to evict entries
        let entry_size = entry.size_bytes();
        while self.should_evict(entry_size) {
            self.evict_entry()?;
        }

        // Add or update entry
        if let Some(old_entry) = self.entries.insert(key.to_string(), entry) {
            self.statistics.total_size_bytes -= old_entry.size_bytes();
        } else {
            self.statistics.total_entries += 1;
        }

        self.statistics.total_size_bytes += entry_size;
        self.update_statistics();

        // Trigger replication if enabled
        if self.config.replication_strategy != ReplicationStrategy::None {
            self.schedule_replication(key)?;
        }

        Ok(())
    }

    /// Get a cache entry
    pub fn get(&mut self, key: &str) -> Option<Vec<u8>> {
        // First check if entry exists and is expired
        let should_remove =
            if let Some(entry) = self.entries.get(key) { entry.is_expired() } else { false };

        // Remove expired entry
        if should_remove {
            if let Some(entry) = self.entries.remove(key) {
                self.statistics.miss_count += 1;
                self.statistics.total_entries -= 1;
                self.statistics.total_size_bytes -= entry.size_bytes();
            }
            return None;
        }

        // Check if entry exists and collect data
        let entry_data = if let Some(entry) = self.entries.get_mut(key) {
            // Update access statistics
            entry.update_access();
            let entry_size = entry.size_bytes();
            let entry_clone = entry.clone();
            Some((entry_size, entry_clone))
        } else {
            None
        };

        // Process entry data without holding mutable borrow
        if let Some((entry_size, entry_clone)) = entry_data {
            self.statistics.hit_count += 1;
            let latency_improvement = self.estimate_latency_improvement(&entry_clone);
            self.statistics.network_bytes_saved += entry_size;
            self.statistics.latency_improvement_ms += latency_improvement;

            // Return decompressed data
            if entry_clone.compression_ratio > 1.0 {
                let mut decompressed_entry = entry_clone.clone();
                if decompressed_entry.decompress().is_ok() {
                    return Some(decompressed_entry.data());
                }
            }

            Some(entry_clone.data())
        } else {
            self.statistics.miss_count += 1;

            // Check if we should prefetch this entry
            if self.config.prefetch_enabled && self.should_prefetch(key) {
                self.prefetch_queue.push(key.to_string());
            }

            None
        }
    }

    /// Check if a key exists in the cache
    pub fn contains_key(&self, key: &str) -> bool {
        if let Some(entry) = self.entries.get(key) {
            !entry.is_expired()
        } else {
            false
        }
    }

    /// Remove a cache entry
    pub fn remove(&mut self, key: &str) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.statistics.total_entries -= 1;
            self.statistics.total_size_bytes -= entry.size_bytes();
            true
        } else {
            false
        }
    }

    /// Clear all cache entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.statistics.total_entries = 0;
        self.statistics.total_size_bytes = 0;
        self.statistics.memory_usage_bytes = 0;
    }

    /// Get cache statistics
    #[wasm_bindgen(getter)]
    pub fn statistics(&self) -> CacheStatistics {
        self.statistics.clone()
    }

    /// Get current cache size in bytes
    pub fn size_bytes(&self) -> usize {
        self.statistics.total_size_bytes
    }

    /// Get current number of entries
    pub fn entry_count(&self) -> usize {
        self.statistics.total_entries
    }

    /// Check if cache is near capacity
    pub fn is_near_capacity(&self) -> bool {
        let size_ratio =
            self.statistics.total_size_bytes as f32 / self.config.max_size_bytes as f32;
        let entry_ratio = self.statistics.total_entries as f32 / self.config.max_entries as f32;

        size_ratio > 0.9 || entry_ratio > 0.9
    }

    /// Perform cache cleanup
    pub fn cleanup(&mut self) -> Result<(), JsValue> {
        let current_time = js_sys::Date::now() as u64;

        // Only cleanup if enough time has passed
        if current_time - self.last_cleanup < self.config.background_cleanup_interval_ms {
            return Ok(());
        }

        // Remove expired entries
        let expired_keys: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            self.remove(&key);
        }

        // Perform background compression
        if self.config.enable_compression {
            self.compress_entries()?;
        }

        // Update cleanup timestamp
        self.last_cleanup = current_time;

        Ok(())
    }

    /// Prefetch entries based on access patterns
    pub async fn prefetch(&mut self) -> Result<(), JsValue> {
        if !self.config.prefetch_enabled || self.prefetch_queue.is_empty() {
            return Ok(());
        }

        // Process prefetch queue
        while let Some(key) = self.prefetch_queue.pop() {
            if !self.entries.contains_key(&key) {
                // In a real implementation, this would fetch from origin
                // For now, we'll simulate prefetching
                self.simulate_prefetch(&key).await?;
            }
        }

        Ok(())
    }

    /// Get cache health metrics
    pub fn get_health_metrics(&self) -> JsValue {
        let metrics = js_sys::Object::new();

        let hit_rate = self.statistics.hit_rate();
        let memory_usage_ratio =
            self.statistics.memory_usage_bytes as f32 / self.config.max_size_bytes as f32;
        let entry_count_ratio =
            self.statistics.total_entries as f32 / self.config.max_entries as f32;

        // Calculate health score (0-1)
        let health_score =
            (hit_rate + (1.0 - memory_usage_ratio) + (1.0 - entry_count_ratio)) / 3.0;

        js_sys::Reflect::set(
            &metrics,
            &JsValue::from_str("hit_rate"),
            &JsValue::from(hit_rate),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &metrics,
            &JsValue::from_str("memory_usage_ratio"),
            &JsValue::from(memory_usage_ratio),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &metrics,
            &JsValue::from_str("entry_count_ratio"),
            &JsValue::from(entry_count_ratio),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &metrics,
            &JsValue::from_str("health_score"),
            &JsValue::from(health_score),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &metrics,
            &JsValue::from_str("compression_ratio"),
            &JsValue::from(self.statistics.compression_ratio),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &metrics,
            &JsValue::from_str("network_bytes_saved"),
            &JsValue::from(self.statistics.network_bytes_saved),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &metrics,
            &JsValue::from_str("average_latency_improvement_ms"),
            &JsValue::from(self.statistics.latency_improvement_ms),
        )
        .expect("Failed to set property in JS object");

        metrics.into()
    }

    /// Get cache optimization recommendations
    pub fn get_optimization_recommendations(&self) -> js_sys::Array {
        let recommendations = js_sys::Array::new();

        let hit_rate = self.statistics.hit_rate();
        let memory_usage_ratio =
            self.statistics.memory_usage_bytes as f32 / self.config.max_size_bytes as f32;

        // Low hit rate recommendations
        if hit_rate < 0.5 {
            recommendations.push(&JsValue::from_str("Consider increasing cache size or TTL"));
            recommendations.push(&JsValue::from_str(
                "Review eviction policy - maybe switch to LFU",
            ));
        }

        // High memory usage recommendations
        if memory_usage_ratio > 0.8 {
            recommendations.push(&JsValue::from_str(
                "Enable compression to reduce memory usage",
            ));
            recommendations.push(&JsValue::from_str(
                "Consider more aggressive eviction policy",
            ));
        }

        // Low compression ratio recommendations
        if self.statistics.compression_ratio < 1.5 {
            recommendations.push(&JsValue::from_str(
                "Review compression settings or algorithms",
            ));
        }

        // High eviction rate recommendations
        if self.statistics.eviction_count > self.statistics.hit_count / 2 {
            recommendations.push(&JsValue::from_str(
                "Increase cache size to reduce evictions",
            ));
            recommendations.push(&JsValue::from_str("Review data access patterns"));
        }

        recommendations
    }

    /// Export cache configuration and statistics
    pub fn export_diagnostics(&self) -> String {
        let diagnostics = js_sys::Object::new();

        // Configuration
        let config = js_sys::Object::new();
        js_sys::Reflect::set(
            &config,
            &JsValue::from_str("max_size_mb"),
            &JsValue::from(self.config.max_size_bytes / (1024 * 1024)),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &config,
            &JsValue::from_str("max_entries"),
            &JsValue::from(self.config.max_entries),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &config,
            &JsValue::from_str("eviction_policy"),
            &JsValue::from(format!("{:?}", self.config.eviction_policy)),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &config,
            &JsValue::from_str("consistency_level"),
            &JsValue::from(format!("{:?}", self.config.consistency_level)),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &config,
            &JsValue::from_str("replication_strategy"),
            &JsValue::from(format!("{:?}", self.config.replication_strategy)),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(&diagnostics, &JsValue::from_str("config"), &config)
            .expect("Failed to set config in diagnostics object");

        // Statistics
        let stats = js_sys::Object::new();
        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("total_entries"),
            &JsValue::from(self.statistics.total_entries),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("total_size_mb"),
            &JsValue::from(self.statistics.total_size_bytes / (1024 * 1024)),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("hit_rate"),
            &JsValue::from(self.statistics.hit_rate()),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("hit_count"),
            &JsValue::from(self.statistics.hit_count),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("miss_count"),
            &JsValue::from(self.statistics.miss_count),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("eviction_count"),
            &JsValue::from(self.statistics.eviction_count),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &stats,
            &JsValue::from_str("compression_ratio"),
            &JsValue::from(self.statistics.compression_ratio),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(&diagnostics, &JsValue::from_str("statistics"), &stats)
            .expect("Failed to set statistics in diagnostics object");

        // Current state
        let state = js_sys::Object::new();
        js_sys::Reflect::set(
            &state,
            &JsValue::from_str("region"),
            &JsValue::from(format!("{:?}", self.region)),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &state,
            &JsValue::from_str("prefetch_queue_length"),
            &JsValue::from(self.prefetch_queue.len()),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(
            &state,
            &JsValue::from_str("replication_peers"),
            &JsValue::from(self.replication_peers.len()),
        )
        .expect("Failed to set property in JS object");
        js_sys::Reflect::set(&diagnostics, &JsValue::from_str("state"), &state)
            .expect("Failed to set state in diagnostics object");

        // Return as JSON string
        js_sys::JSON::stringify(&diagnostics)
            .expect("Failed to stringify diagnostics")
            .as_string()
            .expect("Stringified JSON should be a valid string")
    }

    /// Add a replication peer
    pub fn add_replication_peer(&mut self, peer_url: String) {
        if !self.replication_peers.contains(&peer_url) {
            self.replication_peers.push(peer_url);
        }
    }

    /// Remove a replication peer
    pub fn remove_replication_peer(&mut self, peer_url: &str) -> bool {
        if let Some(pos) = self.replication_peers.iter().position(|p| p == peer_url) {
            self.replication_peers.remove(pos);
            true
        } else {
            false
        }
    }

    /// Get replication peers
    pub fn get_replication_peers(&self) -> js_sys::Array {
        let peers = js_sys::Array::new();
        for peer in &self.replication_peers {
            peers.push(&JsValue::from_str(peer));
        }
        peers
    }
}

impl EdgeCacheManager {
    /// Check if we should evict entries before adding a new one
    fn should_evict(&self, new_entry_size: usize) -> bool {
        let size_limit_exceeded =
            self.statistics.total_size_bytes + new_entry_size > self.config.max_size_bytes;
        let entry_limit_exceeded = self.statistics.total_entries >= self.config.max_entries;

        size_limit_exceeded || entry_limit_exceeded
    }

    /// Evict an entry based on the configured eviction policy
    fn evict_entry(&mut self) -> Result<(), JsValue> {
        let key_to_evict = match self.config.eviction_policy {
            EvictionPolicy::LRU => self.find_lru_key(),
            EvictionPolicy::LFU => self.find_lfu_key(),
            EvictionPolicy::TTL => self.find_expiring_key(),
            EvictionPolicy::SizeBased => self.find_largest_key(),
            EvictionPolicy::Adaptive => self.find_adaptive_key(),
        };

        if let Some(key) = key_to_evict {
            self.remove(&key);
            self.statistics.eviction_count += 1;
        }

        Ok(())
    }

    /// Find the least recently used key
    fn find_lru_key(&self) -> Option<String> {
        self.entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone())
    }

    /// Find the least frequently used key
    fn find_lfu_key(&self) -> Option<String> {
        self.entries
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(key, _)| key.clone())
    }

    /// Find the key with the earliest expiration
    fn find_expiring_key(&self) -> Option<String> {
        self.entries
            .iter()
            .filter(|(_, entry)| entry.ttl_ms.is_some())
            .min_by_key(|(_, entry)| entry.created_at + entry.ttl_ms.unwrap_or(0))
            .map(|(key, _)| key.clone())
    }

    /// Find the key with the largest size
    fn find_largest_key(&self) -> Option<String> {
        self.entries
            .iter()
            .max_by_key(|(_, entry)| entry.size_bytes)
            .map(|(key, _)| key.clone())
    }

    /// Find the key with the lowest efficiency score
    fn find_adaptive_key(&self) -> Option<String> {
        self.entries
            .iter()
            .min_by(|(_, a), (_, b)| {
                a.efficiency_score()
                    .partial_cmp(&b.efficiency_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(key, _)| key.clone())
    }

    /// Check if we should prefetch a key
    fn should_prefetch(&self, key: &str) -> bool {
        let hit_rate = self.statistics.hit_rate();
        hit_rate > self.config.prefetch_threshold && !self.prefetch_queue.contains(&key.to_string())
    }

    /// Simulate prefetching (in real implementation, fetch from origin)
    async fn simulate_prefetch(&mut self, key: &str) -> Result<(), JsValue> {
        // Simulate network delay
        let delay = 50.0 + js_sys::Math::random() * 100.0;

        let promise = js_sys::Promise::resolve(&JsValue::from(delay));
        JsFuture::from(promise).await?;

        // Simulate prefetched data
        let prefetched_data = vec![0u8; 1024]; // 1KB of dummy data

        self.put(
            key,
            CacheEntryType::PreprocessedInput,
            prefetched_data,
            self.config.default_ttl_ms,
            0.5, // Medium priority for prefetched data
        )?;

        Ok(())
    }

    /// Schedule replication to peers
    fn schedule_replication(&self, key: &str) -> Result<(), JsValue> {
        // In a real implementation, this would trigger replication
        // For now, we'll just log the replication request
        web_sys::console::log_1(&format!("Scheduling replication for key: {}", key).into());
        Ok(())
    }

    /// Compress entries in the background
    fn compress_entries(&mut self) -> Result<(), JsValue> {
        let mut compressed_count = 0;

        for (_, entry) in self.entries.iter_mut() {
            if entry.compression_ratio <= 1.0 && entry.compress().is_ok() {
                compressed_count += 1;
            }
        }

        if compressed_count > 0 {
            self.update_statistics();
        }

        Ok(())
    }

    /// Update cache statistics
    fn update_statistics(&mut self) {
        self.statistics.total_entries = self.entries.len();
        self.statistics.total_size_bytes = self.entries.values().map(|e| e.size_bytes()).sum();
        self.statistics.memory_usage_bytes = self.statistics.total_size_bytes;

        // Calculate average compression ratio
        let total_compression_ratio: f32 = self.entries.values().map(|e| e.compression_ratio).sum();
        self.statistics.compression_ratio = if self.statistics.total_entries > 0 {
            total_compression_ratio / self.statistics.total_entries as f32
        } else {
            1.0
        };

        // Calculate average age
        let current_time = js_sys::Date::now() as u64;
        let total_age: u64 = self.entries.values().map(|e| current_time - e.created_at).sum();
        self.statistics.average_age_ms = if self.statistics.total_entries > 0 {
            total_age / self.statistics.total_entries as u64
        } else {
            0
        };
    }

    /// Estimate latency improvement for cache hit
    fn estimate_latency_improvement(&self, entry: &CacheEntry) -> f32 {
        // Base latency improvement based on entry type
        let base_improvement = match entry.entry_type {
            CacheEntryType::Model => 500.0, // Model loading is expensive
            CacheEntryType::InferenceResult => 100.0, // Inference is moderately expensive
            CacheEntryType::TokenizationResult => 50.0,
            CacheEntryType::PreprocessedInput => 30.0,
            CacheEntryType::AttentionPatterns => 200.0,
            CacheEntryType::Embeddings => 150.0,
            CacheEntryType::RawData => 20.0,
        };

        // Adjust based on data size (larger data = more latency saved)
        let size_factor = (entry.size_bytes as f32).ln().max(1.0);

        base_improvement * size_factor * 0.1
    }
}

/// Utility functions for edge caching
#[wasm_bindgen]
pub fn create_edge_cache_manager(config: CacheConfig, region: GeoRegion) -> EdgeCacheManager {
    EdgeCacheManager::new(config, region)
}

#[wasm_bindgen]
pub fn create_performance_cache_config() -> CacheConfig {
    CacheConfig::for_performance()
}

#[wasm_bindgen]
pub fn create_memory_efficient_cache_config() -> CacheConfig {
    CacheConfig::for_memory_efficiency()
}

#[wasm_bindgen]
pub fn create_edge_computing_cache_config() -> CacheConfig {
    CacheConfig::for_edge_computing()
}

#[wasm_bindgen]
pub fn estimate_cache_overhead(entry_count: usize, average_size_bytes: usize) -> usize {
    // Estimate overhead for metadata, indexing, etc.
    let metadata_overhead = entry_count * 200; // ~200 bytes per entry metadata
    let indexing_overhead = entry_count * 64; // ~64 bytes per entry in index
    let compression_overhead = (entry_count * average_size_bytes) / 10; // ~10% compression overhead

    metadata_overhead + indexing_overhead + compression_overhead
}
