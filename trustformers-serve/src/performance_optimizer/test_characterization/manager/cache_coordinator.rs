//! Cache Coordinator
//!
//! Coordinator for caching across all modules.

use super::super::types::*;

use anyhow::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex as TokioMutex, RwLock as TokioRwLock};
use tokio::task::{spawn, JoinHandle};
use tokio::time::interval;
use tracing::{error, info, instrument};

#[derive(Debug)]
pub struct CacheCoordinator {
    /// Main cache storage
    cache: Arc<TokioMutex<HashMap<String, CacheEntry>>>,
    /// Cache configuration
    config: Arc<TokioRwLock<CacheConfig>>,
    /// Cache statistics
    stats: Arc<CacheStatistics>,
    /// Cache maintenance task handle
    maintenance_task: Arc<TokioMutex<Option<JoinHandle<()>>>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Test characteristics data
    pub data: TestCharacteristics,
    /// Entry timestamp
    pub timestamp: SystemTime,
    /// Access count
    pub access_count: u64,
    /// Last access time
    pub last_access: SystemTime,
    /// Entry size in bytes
    pub size_bytes: usize,
    /// Compression enabled
    pub compressed: bool,
}

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: AtomicU64,
    /// Total cache misses
    pub misses: AtomicU64,
    /// Total cache size in bytes
    pub total_size_bytes: AtomicU64,
    /// Total entries
    pub total_entries: AtomicU64,
    /// Cache evictions
    pub evictions: AtomicU64,
    /// Cache warming operations
    pub warming_operations: AtomicU64,
}

impl CacheCoordinator {
    /// Create a new cache coordinator
    pub async fn new(config: CacheConfig) -> Result<Self> {
        Ok(Self {
            cache: Arc::new(TokioMutex::new(HashMap::new())),
            config: Arc::new(TokioRwLock::new(config)),
            stats: Arc::new(CacheStatistics::default()),
            maintenance_task: Arc::new(TokioMutex::new(None)),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Get test characteristics from cache
    #[instrument(skip(self))]
    pub async fn get_test_characteristics(
        &self,
        test_id: &str,
    ) -> Result<Option<TestCharacteristics>> {
        let mut cache = self.cache.lock().await;

        if let Some(entry) = cache.get_mut(test_id) {
            // Check if entry is still valid
            let now = SystemTime::now();
            let config = self.config.read().await;

            if let Ok(duration) = now.duration_since(entry.timestamp) {
                if duration.as_secs() <= config.cache_ttl_seconds {
                    // Update access metadata
                    entry.access_count += 1;
                    entry.last_access = now;

                    self.stats.hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(Some(entry.data.clone()));
                } else {
                    // Entry expired, remove it
                    cache.remove(test_id);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }

    /// Cache test characteristics
    #[instrument(skip(self, characteristics))]
    pub async fn cache_test_characteristics(
        &self,
        test_id: &str,
        characteristics: &TestCharacteristics,
    ) -> Result<()> {
        let mut cache = self.cache.lock().await;
        let config = self.config.read().await;

        // Check cache size limits
        if cache.len() >= config.max_cache_size {
            self.evict_entries(&mut cache, &config).await?;
        }

        let now = SystemTime::now();
        let entry_size = std::mem::size_of_val(characteristics);

        let entry = CacheEntry {
            data: characteristics.clone(),
            timestamp: now,
            access_count: 1,
            last_access: now,
            size_bytes: entry_size,
            compressed: config.cache_compression_enabled,
        };

        cache.insert(test_id.to_string(), entry);

        // Update statistics
        self.stats.total_entries.store(cache.len() as u64, Ordering::Relaxed);
        self.stats.total_size_bytes.fetch_add(entry_size as u64, Ordering::Relaxed);

        Ok(())
    }

    /// Evict cache entries based on policy
    async fn evict_entries(
        &self,
        cache: &mut HashMap<String, CacheEntry>,
        config: &CacheConfig,
    ) -> Result<()> {
        let eviction_count = cache.len() / 4; // Evict 25% of entries

        match config.eviction_policy.as_str() {
            "LRU" => {
                // Collect keys first to avoid borrow conflict
                let mut entries: Vec<_> =
                    cache.iter().map(|(k, v)| (k.clone(), v.last_access)).collect();
                entries.sort_by_key(|(_, last_access)| *last_access);

                let keys_to_remove: Vec<_> =
                    entries.iter().take(eviction_count).map(|(k, _)| k.clone()).collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            },
            "LFU" => {
                // Collect keys first to avoid borrow conflict
                let mut entries: Vec<_> =
                    cache.iter().map(|(k, v)| (k.clone(), v.access_count)).collect();
                entries.sort_by_key(|(_, access_count)| *access_count);

                let keys_to_remove: Vec<_> =
                    entries.iter().take(eviction_count).map(|(k, _)| k.clone()).collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            },
            "TTL" => {
                let now = SystemTime::now();
                let mut expired_keys = Vec::new();

                for (key, entry) in cache.iter() {
                    if let Ok(duration) = now.duration_since(entry.timestamp) {
                        if duration > config.ttl {
                            expired_keys.push(key.clone());
                        }
                    }
                }

                for key in expired_keys {
                    cache.remove(&key);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            },
            _ => {
                // Default to LRU - collect keys first to avoid borrow conflict
                let mut entries: Vec<_> =
                    cache.iter().map(|(k, v)| (k.clone(), v.last_access)).collect();
                entries.sort_by_key(|(_, last_access)| *last_access);

                let keys_to_remove: Vec<_> =
                    entries.iter().take(eviction_count).map(|(k, _)| k.clone()).collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            },
        }

        Ok(())
    }

    /// Start cache maintenance task
    pub async fn start_maintenance_task(&self) -> Result<()> {
        let cache = self.cache.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let shutdown = self.shutdown.clone();

        let task = spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Run every minute

            while !shutdown.load(Ordering::Acquire) {
                interval.tick().await;

                // Perform cache maintenance
                if let Err(e) = Self::perform_maintenance(&cache, &config, &stats).await {
                    error!("Cache maintenance failed: {}", e);
                }
            }
        });

        let mut maintenance_task = self.maintenance_task.lock().await;
        *maintenance_task = Some(task);

        Ok(())
    }

    /// Perform cache maintenance
    async fn perform_maintenance(
        cache: &Arc<TokioMutex<HashMap<String, CacheEntry>>>,
        config: &Arc<TokioRwLock<CacheConfig>>,
        stats: &Arc<CacheStatistics>,
    ) -> Result<()> {
        let mut cache_guard = cache.lock().await;
        let config_guard = config.read().await;

        let now = SystemTime::now();
        let mut expired_keys = Vec::new();
        let mut total_size = 0u64;

        // Find expired entries
        for (key, entry) in cache_guard.iter() {
            total_size += entry.size_bytes as u64;

            if let Ok(duration) = now.duration_since(entry.timestamp) {
                if duration.as_secs() > config_guard.cache_ttl_seconds {
                    expired_keys.push(key.clone());
                }
            }
        }

        // Remove expired entries
        for key in expired_keys {
            if let Some(entry) = cache_guard.remove(&key) {
                total_size -= entry.size_bytes as u64;
                stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Update statistics
        stats.total_entries.store(cache_guard.len() as u64, Ordering::Relaxed);
        stats.total_size_bytes.store(total_size, Ordering::Relaxed);

        Ok(())
    }

    /// Get cache statistics
    pub async fn get_statistics(&self) -> CacheStatistics {
        CacheStatistics {
            hits: AtomicU64::new(self.stats.hits.load(Ordering::Relaxed)),
            misses: AtomicU64::new(self.stats.misses.load(Ordering::Relaxed)),
            total_size_bytes: AtomicU64::new(self.stats.total_size_bytes.load(Ordering::Relaxed)),
            total_entries: AtomicU64::new(self.stats.total_entries.load(Ordering::Relaxed)),
            evictions: AtomicU64::new(self.stats.evictions.load(Ordering::Relaxed)),
            warming_operations: AtomicU64::new(
                self.stats.warming_operations.load(Ordering::Relaxed),
            ),
        }
    }

    /// Shutdown cache coordinator
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down CacheCoordinator");

        self.shutdown.store(true, Ordering::Release);

        // Cancel maintenance task
        let mut maintenance_task = self.maintenance_task.lock().await;
        if let Some(task) = maintenance_task.take() {
            task.abort();
        }

        // Clear cache
        let mut cache = self.cache.lock().await;
        cache.clear();

        info!("CacheCoordinator shutdown completed");
        Ok(())
    }
}
