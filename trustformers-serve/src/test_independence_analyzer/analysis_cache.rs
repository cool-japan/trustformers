//! Analysis Cache Implementation
//!
//! This module provides a sophisticated caching system for storing and retrieving
//! computed analysis results including dependency analysis, conflict detection,
//! and test grouping results.

use super::types::*;

use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

// Re-export types needed by other modules
pub use super::types::{
    CacheStatistics, CachedConflictAnalysis, CachedDependencyAnalysis, CachedGroupingAnalysis,
};

// ================================================================================================
// Analysis Cache Implementation
// ================================================================================================

/// Analysis cache for storing computed results
#[derive(Debug)]
pub struct AnalysisCache {
    /// Dependency analysis cache
    dependency_cache: Arc<RwLock<HashMap<String, CachedDependencyAnalysis>>>,

    /// Resource conflict cache
    conflict_cache: Arc<RwLock<HashMap<String, CachedConflictAnalysis>>>,

    /// Test grouping cache
    grouping_cache: Arc<RwLock<HashMap<String, CachedGroupingAnalysis>>>,

    /// Cache configuration
    config: Arc<RwLock<CacheConfig>>,

    /// Cache statistics
    statistics: Arc<Mutex<CacheStatistics>>,

    /// Cache eviction policy
    eviction_policy: Arc<EvictionPolicyImpl>,

    /// Cache serializer
    serializer: Arc<SerializerImpl>,
}

impl AnalysisCache {
    /// Create a new analysis cache with default configuration
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new analysis cache with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        let eviction_policy = Arc::new(match config.eviction_policy_type {
            EvictionPolicyType::LRU => EvictionPolicyImpl::Lru(LruEvictionPolicy::new()),
            EvictionPolicyType::LFU => EvictionPolicyImpl::Lfu(LfuEvictionPolicy::new()),
            EvictionPolicyType::TTL => {
                EvictionPolicyImpl::Ttl(TtlEvictionPolicy::new(config.default_ttl))
            },
            EvictionPolicyType::Custom(ref name) => {
                panic!("Custom eviction policy '{}' not implemented", name)
            },
        });

        let serializer = Arc::new(match config.serialization_format {
            SerializationFormat::Json => SerializerImpl::Json(JsonCacheSerializer),
            SerializationFormat::Bincode => SerializerImpl::Bincode(BincodeCacheSerializer),
            SerializationFormat::MessagePack => {
                SerializerImpl::MessagePack(MessagePackCacheSerializer)
            },
        });

        Self {
            dependency_cache: Arc::new(RwLock::new(HashMap::new())),
            conflict_cache: Arc::new(RwLock::new(HashMap::new())),
            grouping_cache: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(config)),
            statistics: Arc::new(Mutex::new(CacheStatistics::default())),
            eviction_policy,
            serializer,
        }
    }

    /// Store dependency analysis result
    pub fn store_dependency_analysis(
        &self,
        key: &str,
        analysis: CachedDependencyAnalysis,
    ) -> AnalysisResult<()> {
        let start_time = Instant::now();

        let mut cache = self.dependency_cache.write();
        let config = self.config.read();

        // Check cache size limits
        if cache.len() >= config.max_entries_per_type {
            let keys_to_evict = self.eviction_policy.select_for_eviction(&cache, 1)?;
            for key in keys_to_evict {
                cache.remove(&key);
                self.statistics.lock().evictions += 1;
            }
        }

        // Update cache metadata
        let mut analysis_with_metadata = analysis;
        analysis_with_metadata.metadata.last_accessed = Utc::now();
        analysis_with_metadata.metadata.cache_key = key.to_string();

        cache.insert(key.to_string(), analysis_with_metadata);

        // Update statistics
        let access_time = start_time.elapsed();
        let mut stats = self.statistics.lock();
        stats.update_after_access(false, access_time); // Store operation is a miss
        stats.memory_usage += self.estimate_entry_size(&cache.get(key).unwrap());
        *stats.entries_by_type.entry("dependency".to_string()).or_insert(0) += 1;

        debug!(
            key = %key,
            access_time_ms = %access_time.as_millis(),
            "Stored dependency analysis in cache"
        );

        Ok(())
    }

    /// Retrieve dependency analysis result
    pub fn get_dependency_analysis(
        &self,
        key: &str,
    ) -> AnalysisResult<Option<CachedDependencyAnalysis>> {
        let start_time = Instant::now();

        let mut cache = self.dependency_cache.write();
        let access_time = start_time.elapsed();

        match cache.get_mut(key) {
            Some(analysis) => {
                // Update access metadata
                analysis.metadata.last_accessed = Utc::now();
                analysis.metadata.access_count += 1;

                let result = analysis.clone();

                // Update statistics
                self.statistics.lock().update_after_access(true, access_time);

                debug!(
                    key = %key,
                    access_time_ms = %access_time.as_millis(),
                    "Cache hit for dependency analysis"
                );

                Ok(Some(result))
            },
            None => {
                // Update statistics
                self.statistics.lock().update_after_access(false, access_time);

                debug!(
                    key = %key,
                    access_time_ms = %access_time.as_millis(),
                    "Cache miss for dependency analysis"
                );

                Ok(None)
            },
        }
    }

    /// Store conflict analysis result
    pub fn store_conflict_analysis(
        &self,
        key: &str,
        analysis: CachedConflictAnalysis,
    ) -> AnalysisResult<()> {
        let start_time = Instant::now();

        let mut cache = self.conflict_cache.write();
        let config = self.config.read();

        // Check cache size limits
        if cache.len() >= config.max_entries_per_type {
            let keys_to_evict = self.eviction_policy.select_for_eviction(&cache, 1)?;
            for key in keys_to_evict {
                cache.remove(&key);
                self.statistics.lock().evictions += 1;
            }
        }

        // Update cache metadata
        let mut analysis_with_metadata = analysis;
        analysis_with_metadata.metadata.last_accessed = Utc::now();
        analysis_with_metadata.metadata.cache_key = key.to_string();

        cache.insert(key.to_string(), analysis_with_metadata);

        // Update statistics
        let access_time = start_time.elapsed();
        let mut stats = self.statistics.lock();
        stats.update_after_access(false, access_time);
        stats.memory_usage += self.estimate_entry_size(&cache.get(key).unwrap());
        *stats.entries_by_type.entry("conflict".to_string()).or_insert(0) += 1;

        debug!(
            key = %key,
            access_time_ms = %access_time.as_millis(),
            "Stored conflict analysis in cache"
        );

        Ok(())
    }

    /// Retrieve conflict analysis result
    pub fn get_conflict_analysis(
        &self,
        key: &str,
    ) -> AnalysisResult<Option<CachedConflictAnalysis>> {
        let start_time = Instant::now();

        let mut cache = self.conflict_cache.write();
        let access_time = start_time.elapsed();

        match cache.get_mut(key) {
            Some(analysis) => {
                // Update access metadata
                analysis.metadata.last_accessed = Utc::now();
                analysis.metadata.access_count += 1;

                let result = analysis.clone();

                // Update statistics
                self.statistics.lock().update_after_access(true, access_time);

                debug!(
                    key = %key,
                    access_time_ms = %access_time.as_millis(),
                    "Cache hit for conflict analysis"
                );

                Ok(Some(result))
            },
            None => {
                // Update statistics
                self.statistics.lock().update_after_access(false, access_time);

                debug!(
                    key = %key,
                    access_time_ms = %access_time.as_millis(),
                    "Cache miss for conflict analysis"
                );

                Ok(None)
            },
        }
    }

    /// Store grouping analysis result
    pub fn store_grouping_analysis(
        &self,
        key: &str,
        analysis: CachedGroupingAnalysis,
    ) -> AnalysisResult<()> {
        let start_time = Instant::now();

        let mut cache = self.grouping_cache.write();
        let config = self.config.read();

        // Check cache size limits
        if cache.len() >= config.max_entries_per_type {
            let keys_to_evict = self.eviction_policy.select_for_eviction(&cache, 1)?;
            for key in keys_to_evict {
                cache.remove(&key);
                self.statistics.lock().evictions += 1;
            }
        }

        // Update cache metadata
        let mut analysis_with_metadata = analysis;
        analysis_with_metadata.metadata.last_accessed = Utc::now();
        analysis_with_metadata.metadata.cache_key = key.to_string();

        cache.insert(key.to_string(), analysis_with_metadata);

        // Update statistics
        let access_time = start_time.elapsed();
        let mut stats = self.statistics.lock();
        stats.update_after_access(false, access_time);
        stats.memory_usage += self.estimate_entry_size(&cache.get(key).unwrap());
        *stats.entries_by_type.entry("grouping".to_string()).or_insert(0) += 1;

        debug!(
            key = %key,
            access_time_ms = %access_time.as_millis(),
            "Stored grouping analysis in cache"
        );

        Ok(())
    }

    /// Retrieve grouping analysis result
    pub fn get_grouping_analysis(
        &self,
        key: &str,
    ) -> AnalysisResult<Option<CachedGroupingAnalysis>> {
        let start_time = Instant::now();

        let mut cache = self.grouping_cache.write();
        let access_time = start_time.elapsed();

        match cache.get_mut(key) {
            Some(analysis) => {
                // Update access metadata
                analysis.metadata.last_accessed = Utc::now();
                analysis.metadata.access_count += 1;

                let result = analysis.clone();

                // Update statistics
                self.statistics.lock().update_after_access(true, access_time);

                debug!(
                    key = %key,
                    access_time_ms = %access_time.as_millis(),
                    "Cache hit for grouping analysis"
                );

                Ok(Some(result))
            },
            None => {
                // Update statistics
                self.statistics.lock().update_after_access(false, access_time);

                debug!(
                    key = %key,
                    access_time_ms = %access_time.as_millis(),
                    "Cache miss for grouping analysis"
                );

                Ok(None)
            },
        }
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        self.dependency_cache.write().clear();
        self.conflict_cache.write().clear();
        self.grouping_cache.write().clear();

        let mut stats = self.statistics.lock();
        stats.memory_usage = 0;
        stats.entries_by_type.clear();

        info!("Cleared all cache entries");
    }

    /// Clear specific cache type
    pub fn clear_cache_type(&self, cache_type: CacheType) {
        match cache_type {
            CacheType::Dependency => {
                let count = self.dependency_cache.write().len();
                self.dependency_cache.write().clear();
                let mut stats = self.statistics.lock();
                stats.entries_by_type.insert("dependency".to_string(), 0);
                info!(count = %count, "Cleared dependency cache");
            },
            CacheType::Conflict => {
                let count = self.conflict_cache.write().len();
                self.conflict_cache.write().clear();
                let mut stats = self.statistics.lock();
                stats.entries_by_type.insert("conflict".to_string(), 0);
                info!(count = %count, "Cleared conflict cache");
            },
            CacheType::Grouping => {
                let count = self.grouping_cache.write().len();
                self.grouping_cache.write().clear();
                let mut stats = self.statistics.lock();
                stats.entries_by_type.insert("grouping".to_string(), 0);
                info!(count = %count, "Cleared grouping cache");
            },
        }
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> CacheStatistics {
        (*self.statistics.lock()).clone()
    }

    /// Perform cache maintenance (cleanup expired entries, etc.)
    pub fn perform_maintenance(&self) -> AnalysisResult<MaintenanceResult> {
        let start_time = Instant::now();
        let mut result = MaintenanceResult::default();

        // Clean up expired entries
        result.expired_entries_removed = self.cleanup_expired_entries()?;

        // Perform eviction if over memory limit
        result.entries_evicted = self.evict_if_over_limit()?;

        // Compact cache if needed
        result.cache_compacted = self.compact_if_needed()?;

        // Update maintenance statistics
        let maintenance_time = start_time.elapsed();

        info!(
            maintenance_time_ms = %maintenance_time.as_millis(),
            expired_removed = %result.expired_entries_removed,
            entries_evicted = %result.entries_evicted,
            cache_compacted = %result.cache_compacted,
            "Cache maintenance completed"
        );

        Ok(result)
    }

    /// Export cache to persistent storage
    pub async fn export_to_storage(&self, path: &str) -> AnalysisResult<()> {
        let dependency_cache = (*self.dependency_cache.read()).clone();
        let conflict_cache = (*self.conflict_cache.read()).clone();
        let grouping_cache = (*self.grouping_cache.read()).clone();

        let export_data = CacheExportData {
            dependency_cache,
            conflict_cache,
            grouping_cache,
            exported_at: Utc::now(),
            version: "1.0".to_string(),
        };

        let serialized_data = self.serializer.serialize(&export_data)?;

        tokio::fs::write(path, serialized_data)
            .await
            .map_err(|e| AnalysisError::CacheError {
                message: format!("Failed to write cache export: {}", e),
            })?;

        info!(path = %path, "Cache exported to storage");
        Ok(())
    }

    /// Import cache from persistent storage
    pub async fn import_from_storage(&self, path: &str) -> AnalysisResult<()> {
        let data = tokio::fs::read(path).await.map_err(|e| AnalysisError::CacheError {
            message: format!("Failed to read cache import: {}", e),
        })?;

        let export_data: CacheExportData = self.serializer.deserialize(&data)?;

        // Replace current cache contents
        *self.dependency_cache.write() = export_data.dependency_cache;
        *self.conflict_cache.write() = export_data.conflict_cache;
        *self.grouping_cache.write() = export_data.grouping_cache;

        // Update statistics
        let mut stats = self.statistics.lock();
        stats.entries_by_type.insert(
            "dependency".to_string(),
            self.dependency_cache.read().len() as u64,
        );
        stats.entries_by_type.insert(
            "conflict".to_string(),
            self.conflict_cache.read().len() as u64,
        );
        stats.entries_by_type.insert(
            "grouping".to_string(),
            self.grouping_cache.read().len() as u64,
        );

        info!(path = %path, "Cache imported from storage");
        Ok(())
    }

    // ============================================================================================
    // Private Implementation Methods
    // ============================================================================================

    /// Estimate the memory size of a cache entry
    fn estimate_entry_size<T>(&self, _entry: &T) -> u64 {
        // Simplified estimation - in a real implementation, you'd use more sophisticated sizing
        std::mem::size_of::<T>() as u64
    }

    /// Clean up expired cache entries
    fn cleanup_expired_entries(&self) -> AnalysisResult<usize> {
        let config = self.config.read();
        let ttl = config.default_ttl;
        let now = Utc::now();
        let mut removed_count = 0;

        // Clean dependency cache
        {
            let mut cache = self.dependency_cache.write();
            let keys_to_remove: Vec<String> = cache
                .iter()
                .filter(|(_, analysis)| {
                    now.signed_duration_since(analysis.metadata.created_at)
                        > chrono::Duration::from_std(ttl).unwrap_or_default()
                })
                .map(|(key, _)| key.clone())
                .collect();

            for key in keys_to_remove {
                cache.remove(&key);
                removed_count += 1;
            }
        }

        // Clean conflict cache
        {
            let mut cache = self.conflict_cache.write();
            let keys_to_remove: Vec<String> = cache
                .iter()
                .filter(|(_, analysis)| {
                    now.signed_duration_since(analysis.metadata.created_at)
                        > chrono::Duration::from_std(ttl).unwrap_or_default()
                })
                .map(|(key, _)| key.clone())
                .collect();

            for key in keys_to_remove {
                cache.remove(&key);
                removed_count += 1;
            }
        }

        // Clean grouping cache
        {
            let mut cache = self.grouping_cache.write();
            let keys_to_remove: Vec<String> = cache
                .iter()
                .filter(|(_, analysis)| {
                    now.signed_duration_since(analysis.metadata.created_at)
                        > chrono::Duration::from_std(ttl).unwrap_or_default()
                })
                .map(|(key, _)| key.clone())
                .collect();

            for key in keys_to_remove {
                cache.remove(&key);
                removed_count += 1;
            }
        }

        if removed_count > 0 {
            debug!(removed_count = %removed_count, "Cleaned up expired cache entries");
        }

        Ok(removed_count)
    }

    /// Evict entries if over memory limit
    fn evict_if_over_limit(&self) -> AnalysisResult<usize> {
        let config = self.config.read();
        let current_memory = self.statistics.lock().memory_usage;

        if current_memory <= config.max_memory_usage {
            return Ok(0);
        }

        let target_memory = (config.max_memory_usage as f32 * 0.8) as u64; // Evict to 80% of limit
        let memory_to_free = current_memory - target_memory;
        let _freed_memory = 0u64;
        let evicted_count = 0;

        // Simple eviction strategy: evict oldest entries first
        // In a more sophisticated implementation, you'd use the eviction policy

        warn!(
            current_memory = %current_memory,
            max_memory = %config.max_memory_usage,
            memory_to_free = %memory_to_free,
            "Performing memory-based eviction"
        );

        // This is a simplified implementation - you'd want to implement proper LRU/LFU eviction
        // For now, just clear some entries if we're over the limit

        Ok(evicted_count)
    }

    /// Compact cache if needed
    fn compact_if_needed(&self) -> AnalysisResult<bool> {
        // Placeholder for cache compaction logic
        // In a real implementation, this might defragment internal storage,
        // reorganize data structures, etc.
        Ok(false)
    }
}

impl Default for AnalysisCache {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// Cache Configuration
// ================================================================================================

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum entries per cache type
    pub max_entries_per_type: usize,

    /// Maximum total memory usage (bytes)
    pub max_memory_usage: u64,

    /// Default TTL for cache entries
    pub default_ttl: Duration,

    /// Eviction policy type
    pub eviction_policy_type: EvictionPolicyType,

    /// Serialization format
    pub serialization_format: SerializationFormat,

    /// Enable cache persistence
    pub enable_persistence: bool,

    /// Persistence file path
    pub persistence_path: String,

    /// Maintenance interval
    pub maintenance_interval: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries_per_type: 10000,
            max_memory_usage: 100 * 1024 * 1024,    // 100MB
            default_ttl: Duration::from_secs(3600), // 1 hour
            eviction_policy_type: EvictionPolicyType::LRU,
            serialization_format: SerializationFormat::Json,
            enable_persistence: false,
            persistence_path: "/tmp/analysis_cache.bin".to_string(),
            maintenance_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Cache type enumeration
#[derive(Debug, Clone, Copy)]
pub enum CacheType {
    Dependency,
    Conflict,
    Grouping,
}

/// Eviction policy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicyType {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time To Live
    TTL,
    /// Custom eviction policy
    Custom(String),
}

/// Serialization format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializationFormat {
    /// JSON format
    Json,
    /// Binary format (bincode)
    Bincode,
    /// MessagePack format
    MessagePack,
}

/// Maintenance operation result
#[derive(Debug, Default)]
pub struct MaintenanceResult {
    /// Number of expired entries removed
    pub expired_entries_removed: usize,
    /// Number of entries evicted
    pub entries_evicted: usize,
    /// Whether cache was compacted
    pub cache_compacted: bool,
}

/// Cache export data structure
#[derive(Debug, Serialize, Deserialize)]
struct CacheExportData {
    dependency_cache: HashMap<String, CachedDependencyAnalysis>,
    conflict_cache: HashMap<String, CachedConflictAnalysis>,
    grouping_cache: HashMap<String, CachedGroupingAnalysis>,
    exported_at: DateTime<Utc>,
    version: String,
}

// ================================================================================================
// Cache Eviction Policies
// ================================================================================================

/// Trait for cache eviction policies
trait CacheEvictionPolicy: Send + Sync {
    /// Select entries for eviction
    fn select_for_eviction<T>(
        &self,
        cache: &HashMap<String, T>,
        count: usize,
    ) -> AnalysisResult<Vec<String>>;
}

/// LRU eviction policy
#[derive(Debug)]
struct LruEvictionPolicy;

impl LruEvictionPolicy {
    fn new() -> Self {
        Self
    }
}

impl CacheEvictionPolicy for LruEvictionPolicy {
    fn select_for_eviction<T>(
        &self,
        cache: &HashMap<String, T>,
        count: usize,
    ) -> AnalysisResult<Vec<String>> {
        // Simplified LRU implementation
        let keys: Vec<String> = cache.keys().take(count).cloned().collect();
        Ok(keys)
    }
}

/// LFU eviction policy
#[derive(Debug)]
struct LfuEvictionPolicy;

impl LfuEvictionPolicy {
    fn new() -> Self {
        Self
    }
}

impl CacheEvictionPolicy for LfuEvictionPolicy {
    fn select_for_eviction<T>(
        &self,
        cache: &HashMap<String, T>,
        count: usize,
    ) -> AnalysisResult<Vec<String>> {
        // Simplified LFU implementation
        let keys: Vec<String> = cache.keys().take(count).cloned().collect();
        Ok(keys)
    }
}

/// TTL eviction policy
#[derive(Debug)]
struct TtlEvictionPolicy {
    ttl: Duration,
}

impl TtlEvictionPolicy {
    fn new(ttl: Duration) -> Self {
        Self { ttl }
    }
}

impl CacheEvictionPolicy for TtlEvictionPolicy {
    fn select_for_eviction<T>(
        &self,
        cache: &HashMap<String, T>,
        count: usize,
    ) -> AnalysisResult<Vec<String>> {
        // TTL-based eviction would check creation times
        let keys: Vec<String> = cache.keys().take(count).cloned().collect();
        Ok(keys)
    }
}

/// Eviction policy implementation enum (for dyn compatibility)
#[derive(Debug)]
enum EvictionPolicyImpl {
    Lru(LruEvictionPolicy),
    Lfu(LfuEvictionPolicy),
    Ttl(TtlEvictionPolicy),
}

impl EvictionPolicyImpl {
    fn select_for_eviction<T>(
        &self,
        cache: &HashMap<String, T>,
        count: usize,
    ) -> AnalysisResult<Vec<String>> {
        match self {
            EvictionPolicyImpl::Lru(policy) => policy.select_for_eviction(cache, count),
            EvictionPolicyImpl::Lfu(policy) => policy.select_for_eviction(cache, count),
            EvictionPolicyImpl::Ttl(policy) => policy.select_for_eviction(cache, count),
        }
    }
}

// ================================================================================================
// Cache Serializers
// ================================================================================================

/// Trait for cache serialization
trait CacheSerializer: Send + Sync {
    /// Serialize data to bytes
    fn serialize<T: Serialize>(&self, data: &T) -> AnalysisResult<Vec<u8>>;

    /// Deserialize data from bytes
    fn deserialize<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> AnalysisResult<T>;
}

/// JSON cache serializer
#[derive(Debug)]
struct JsonCacheSerializer;

impl CacheSerializer for JsonCacheSerializer {
    fn serialize<T: Serialize>(&self, data: &T) -> AnalysisResult<Vec<u8>> {
        serde_json::to_vec(data).map_err(|e| AnalysisError::CacheError {
            message: format!("JSON serialization failed: {}", e),
        })
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> AnalysisResult<T> {
        serde_json::from_slice(data).map_err(|e| AnalysisError::CacheError {
            message: format!("JSON deserialization failed: {}", e),
        })
    }
}

/// Bincode cache serializer
#[derive(Debug)]
struct BincodeCacheSerializer;

impl CacheSerializer for BincodeCacheSerializer {
    fn serialize<T: Serialize>(&self, data: &T) -> AnalysisResult<Vec<u8>> {
        bincode::serialize(data).map_err(|e| AnalysisError::CacheError {
            message: format!("Bincode serialization failed: {}", e),
        })
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> AnalysisResult<T> {
        bincode::deserialize(data).map_err(|e| AnalysisError::CacheError {
            message: format!("Bincode deserialization failed: {}", e),
        })
    }
}

/// MessagePack cache serializer
#[derive(Debug)]
struct MessagePackCacheSerializer;

impl CacheSerializer for MessagePackCacheSerializer {
    fn serialize<T: Serialize>(&self, data: &T) -> AnalysisResult<Vec<u8>> {
        rmp_serde::to_vec(data).map_err(|e| AnalysisError::CacheError {
            message: format!("MessagePack serialization failed: {}", e),
        })
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> AnalysisResult<T> {
        rmp_serde::from_slice(data).map_err(|e| AnalysisError::CacheError {
            message: format!("MessagePack deserialization failed: {}", e),
        })
    }
}

/// Serializer implementation enum (for dyn compatibility)
#[derive(Debug)]
enum SerializerImpl {
    Json(JsonCacheSerializer),
    Bincode(BincodeCacheSerializer),
    MessagePack(MessagePackCacheSerializer),
}

impl SerializerImpl {
    fn serialize<T: Serialize>(&self, data: &T) -> AnalysisResult<Vec<u8>> {
        match self {
            SerializerImpl::Json(serializer) => serializer.serialize(data),
            SerializerImpl::Bincode(serializer) => serializer.serialize(data),
            SerializerImpl::MessagePack(serializer) => serializer.serialize(data),
        }
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> AnalysisResult<T> {
        match self {
            SerializerImpl::Json(serializer) => serializer.deserialize(data),
            SerializerImpl::Bincode(serializer) => serializer.deserialize(data),
            SerializerImpl::MessagePack(serializer) => serializer.deserialize(data),
        }
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Generate cache key for dependency analysis
pub fn generate_dependency_cache_key(test_id: &str, version: u64) -> String {
    format!("dep_{}_{}", test_id, version)
}

/// Generate cache key for conflict analysis
pub fn generate_conflict_cache_key(test1: &str, test2: &str, version: u64) -> String {
    let mut tests = [test1, test2];
    tests.sort(); // Ensure consistent ordering
    format!("conflict_{}_{}_v{}", tests[0], tests[1], version)
}

/// Generate cache key for grouping analysis
pub fn generate_grouping_cache_key(
    strategy: &GroupingStrategy,
    test_count: usize,
    version: u64,
) -> String {
    let strategy_str = match strategy {
        GroupingStrategy::ByCategory => "by_category",
        GroupingStrategy::ByResourceUsage => "by_resource_usage",
        GroupingStrategy::ByExecutionTime => "by_execution_time",
        GroupingStrategy::ByDependencies => "by_dependencies",
        GroupingStrategy::OptimalMl => "optimal_ml",
        GroupingStrategy::Custom(name) => name,
    };
    format!("grouping_{}_{}tests_v{}", strategy_str, test_count, version)
}
