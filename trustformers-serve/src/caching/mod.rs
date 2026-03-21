//! Caching Layer for TrustformeRS Inference Server
//!
//! Multi-tier caching system for optimization of inference performance:
//! - Result caching with TTL
//! - Embedding cache
//! - KV cache sharing
//! - Distributed caching support
//! - Cache warming

// Allow dead code for caching infrastructure under development
#![allow(dead_code)]

pub mod config;
pub mod distributed;
pub mod embedding_cache;
pub mod kv_cache;
pub mod metrics;
pub mod result_cache;
pub mod warming;

pub use result_cache::{
    CacheEntry, CacheHit, CacheKey, CacheMiss, CacheResult, ResultCacheService,
};

pub use embedding_cache::{EmbeddingCacheService, EmbeddingEntry, EmbeddingKey, VectorIndex};

pub use kv_cache::{
    AttentionCache, KVCacheEntry, KVCacheManager, KVCacheSlot, LayerCache, SharedKVCache,
};

pub use distributed::{
    CacheCluster, CacheNode, ConsistentHashing, DistributedCache, ReplicationStrategy,
};

pub use warming::{CacheWarmer, PreloadService, WarmingPolicy, WarmingScheduler};

pub use metrics::{
    CacheMetrics, CacheStatsCollector, EvictionTracker, HitRateTracker, PerformanceMonitor,
};

pub use config::{
    CacheConfig, CacheMode, ConsistencyLevel, EvictionPolicy, TierConfig, WarmingStrategy,
};

use anyhow::Result;
use std::sync::Arc;

/// Main caching service that orchestrates all cache types
#[derive(Clone)]
pub struct CachingService {
    result_cache: Arc<ResultCacheService>,
    embedding_cache: Arc<EmbeddingCacheService>,
    kv_cache: Arc<KVCacheManager>,
    distributed_cache: Option<Arc<DistributedCache>>,
    cache_warmer: Arc<CacheWarmer>,
    metrics: Arc<CacheStatsCollector>,
    config: CacheConfig,
}

impl CachingService {
    /// Create a new caching service
    pub fn new(config: CacheConfig) -> Self {
        let metrics = Arc::new(CacheStatsCollector::new());

        let result_cache = Arc::new(ResultCacheService::new(
            config.result_cache.clone(),
            metrics.clone(),
        ));

        let embedding_cache = Arc::new(EmbeddingCacheService::new(
            config.embedding_cache.clone(),
            metrics.clone(),
        ));

        let kv_cache = Arc::new(KVCacheManager::new(
            config.kv_cache.clone(),
            metrics.clone(),
        ));

        let distributed_cache = if config.enable_distributed {
            Some(Arc::new(DistributedCache::new(config.distributed.clone())))
        } else {
            None
        };

        let cache_warmer = Arc::new(CacheWarmer::new(
            config.warming.clone(),
            result_cache.clone(),
            embedding_cache.clone(),
        ));

        Self {
            result_cache,
            embedding_cache,
            kv_cache,
            distributed_cache,
            cache_warmer,
            metrics,
            config,
        }
    }

    /// Start the caching service
    pub async fn start(&self) -> Result<()> {
        // Start background tasks
        self.start_metrics_collection().await?;
        self.start_cache_maintenance().await?;
        self.start_warming_service().await?;

        if let Some(distributed) = &self.distributed_cache {
            distributed.start().await?;
        }

        Ok(())
    }

    /// Get result cache service
    pub fn result_cache(&self) -> &Arc<ResultCacheService> {
        &self.result_cache
    }

    /// Get embedding cache service
    pub fn embedding_cache(&self) -> &Arc<EmbeddingCacheService> {
        &self.embedding_cache
    }

    /// Get KV cache manager
    pub fn kv_cache(&self) -> &Arc<KVCacheManager> {
        &self.kv_cache
    }

    /// Get distributed cache if enabled
    pub fn distributed_cache(&self) -> Option<&Arc<DistributedCache>> {
        self.distributed_cache.as_ref()
    }

    /// Get cache warmer
    pub fn cache_warmer(&self) -> &Arc<CacheWarmer> {
        &self.cache_warmer
    }

    /// Get cache metrics
    pub async fn get_metrics(&self) -> CacheMetrics {
        self.metrics.get_metrics().await
    }

    /// Invalidate all caches
    pub async fn invalidate_all(&self) -> Result<()> {
        self.result_cache.clear().await?;
        self.embedding_cache.clear().await?;
        self.kv_cache.clear().await?;

        if let Some(distributed) = &self.distributed_cache {
            distributed.invalidate_all().await?;
        }

        Ok(())
    }

    /// Update caching configuration
    pub async fn update_config(&self, config: CacheConfig) -> Result<()> {
        self.result_cache.update_config(config.result_cache.clone()).await?;
        self.embedding_cache.update_config(config.embedding_cache.clone()).await?;
        self.kv_cache.update_config(config.kv_cache.clone()).await?;

        if let Some(distributed) = &self.distributed_cache {
            distributed.update_config(config.distributed.clone()).await?;
        }

        Ok(())
    }

    // Background tasks
    async fn start_metrics_collection(&self) -> Result<()> {
        let metrics = self.metrics.clone();

        tokio::spawn(async move {
            loop {
                metrics.collect_periodic_metrics().await;
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        });

        Ok(())
    }

    async fn start_cache_maintenance(&self) -> Result<()> {
        let result_cache = self.result_cache.clone();
        let embedding_cache = self.embedding_cache.clone();
        let kv_cache = self.kv_cache.clone();

        tokio::spawn(async move {
            loop {
                // Run maintenance tasks
                let _ = result_cache.run_maintenance().await;
                let _ = embedding_cache.run_maintenance().await;
                let _ = kv_cache.run_maintenance().await;

                tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
            }
        });

        Ok(())
    }

    async fn start_warming_service(&self) -> Result<()> {
        let warmer = self.cache_warmer.clone();

        tokio::spawn(async move {
            loop {
                let _ = warmer.run_warming_cycle().await;
                tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            }
        });

        Ok(())
    }

    /// Get comprehensive cache statistics
    pub async fn get_stats(&self) -> Result<CacheStats> {
        let result_cache_stats = self.result_cache.get_stats().await;
        let embedding_cache_stats = self.embedding_cache.get_stats().await;
        let kv_cache_stats = self.kv_cache.get_stats().await;
        let distributed_cache_stats = if let Some(distributed) = &self.distributed_cache {
            Some(distributed.get_stats().await?)
        } else {
            None
        };
        let warming_stats = self.cache_warmer.get_stats().await;

        // Calculate overall metrics from available data
        let total_entries = result_cache_stats.entry_count
            + embedding_cache_stats.entry_count
            + kv_cache_stats.total_entries;

        // Calculate weighted overall hit rate based on entry counts
        let overall_hit_rate = if total_entries > 0 {
            let result_weight = result_cache_stats.entry_count as f32 / total_entries as f32;
            let embedding_weight = embedding_cache_stats.entry_count as f32 / total_entries as f32;
            let kv_weight = kv_cache_stats.total_entries as f32 / total_entries as f32;

            result_cache_stats.hit_rate * result_weight
                + embedding_cache_stats.hit_rate * embedding_weight
                + kv_cache_stats.hit_rate * kv_weight
        } else {
            0.0
        };

        let memory_usage_bytes =
            result_cache_stats.total_size_bytes + kv_cache_stats.total_size_bytes;

        let cache_efficiency = overall_hit_rate * 100.0;

        Ok(CacheStats {
            result_cache_stats,
            embedding_cache_stats,
            kv_cache_stats,
            distributed_cache_stats,
            warming_stats,
            overall_hit_rate,
            memory_usage_bytes,
            cache_efficiency,
        })
    }
}

/// Overall cache statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheStats {
    pub result_cache_stats: result_cache::ResultCacheStats,
    pub embedding_cache_stats: embedding_cache::EmbeddingCacheStats,
    pub kv_cache_stats: kv_cache::KVCacheStats,
    pub distributed_cache_stats: Option<distributed::DistributedCacheStats>,
    pub warming_stats: warming::WarmingStats,
    pub overall_hit_rate: f32,
    pub memory_usage_bytes: usize,
    pub cache_efficiency: f32,
}
