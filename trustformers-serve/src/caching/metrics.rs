//! Cache Metrics Implementation
//!
//! Comprehensive metrics collection for cache performance monitoring.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Cache metrics for monitoring and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub hit_rates: HashMap<String, f32>,
    pub miss_rates: HashMap<String, f32>,
    pub eviction_rates: HashMap<String, f32>,
    pub average_latency_ms: HashMap<String, f32>,
    pub cache_sizes: HashMap<String, usize>,
    pub memory_usage_bytes: usize,
    pub total_requests: u64,
    pub total_hits: u64,
    pub total_misses: u64,
}

/// Hit rate tracker for individual cache tiers
pub struct HitRateTracker {
    hits: u64,
    misses: u64,
    window_size: usize,
    hit_history: Vec<bool>,
}

impl HitRateTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            hits: 0,
            misses: 0,
            window_size,
            hit_history: Vec::new(),
        }
    }

    /// Record a cache hit
    pub fn record_hit(&mut self) {
        self.hits += 1;
        self.hit_history.push(true);

        if self.hit_history.len() > self.window_size
            && !self.hit_history.remove(0) {
                self.misses = self.misses.saturating_sub(1);
            }
    }

    /// Record a cache miss
    pub fn record_miss(&mut self) {
        self.misses += 1;
        self.hit_history.push(false);

        if self.hit_history.len() > self.window_size
            && self.hit_history.remove(0) {
                self.hits = self.hits.saturating_sub(1);
            }
    }

    /// Get current hit rate
    pub fn get_hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }

    /// Get windowed hit rate
    pub fn get_windowed_hit_rate(&self) -> f32 {
        if self.hit_history.is_empty() {
            0.0
        } else {
            let hits = self.hit_history.iter().filter(|&&h| h).count();
            hits as f32 / self.hit_history.len() as f32
        }
    }
}

/// Eviction tracker for monitoring cache pressure
pub struct EvictionTracker {
    evictions_by_reason: HashMap<String, u64>,
    total_evictions: u64,
}

impl Default for EvictionTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionTracker {
    pub fn new() -> Self {
        Self {
            evictions_by_reason: HashMap::new(),
            total_evictions: 0,
        }
    }

    /// Record an eviction
    pub fn record_eviction(&mut self, reason: &str) {
        *self.evictions_by_reason.entry(reason.to_string()).or_insert(0) += 1;
        self.total_evictions += 1;
    }

    /// Get eviction rate for a specific reason
    pub fn get_eviction_rate(&self, reason: &str) -> f32 {
        if self.total_evictions == 0 {
            0.0
        } else {
            let evictions = self.evictions_by_reason.get(reason).unwrap_or(&0);
            *evictions as f32 / self.total_evictions as f32
        }
    }

    /// Get total eviction count
    pub fn get_total_evictions(&self) -> u64 {
        self.total_evictions
    }
}

/// Performance monitor for latency tracking
pub struct PerformanceMonitor {
    operation_times: HashMap<String, Vec<f32>>,
    max_samples: usize,
}

impl PerformanceMonitor {
    pub fn new(max_samples: usize) -> Self {
        Self {
            operation_times: HashMap::new(),
            max_samples,
        }
    }

    /// Record operation time
    pub fn record_operation_time(&mut self, operation: &str, time_ms: f32) {
        let times = self.operation_times.entry(operation.to_string()).or_default();
        times.push(time_ms);

        if times.len() > self.max_samples {
            times.remove(0);
        }
    }

    /// Get average latency for operation
    pub fn get_average_latency(&self, operation: &str) -> f32 {
        if let Some(times) = self.operation_times.get(operation) {
            if times.is_empty() {
                0.0
            } else {
                times.iter().sum::<f32>() / times.len() as f32
            }
        } else {
            0.0
        }
    }

    /// Get percentile latency
    pub fn get_percentile_latency(&self, operation: &str, percentile: f32) -> f32 {
        if let Some(times) = self.operation_times.get(operation) {
            if times.is_empty() {
                return 0.0;
            }

            let mut sorted_times = times.clone();
            sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let index = ((percentile / 100.0) * (sorted_times.len() - 1) as f32) as usize;
            sorted_times[index.min(sorted_times.len() - 1)]
        } else {
            0.0
        }
    }
}

/// Main cache statistics collector
pub struct CacheStatsCollector {
    hit_trackers: Arc<RwLock<HashMap<String, HitRateTracker>>>,
    eviction_trackers: Arc<RwLock<HashMap<String, EvictionTracker>>>,
    performance_monitor: Arc<RwLock<PerformanceMonitor>>,
    cache_sizes: Arc<RwLock<HashMap<String, usize>>>,
}

impl Default for CacheStatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheStatsCollector {
    pub fn new() -> Self {
        Self {
            hit_trackers: Arc::new(RwLock::new(HashMap::new())),
            eviction_trackers: Arc::new(RwLock::new(HashMap::new())),
            performance_monitor: Arc::new(RwLock::new(PerformanceMonitor::new(1000))),
            cache_sizes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record a cache hit
    pub async fn record_cache_hit(&self, cache_name: &str) {
        let mut trackers = self.hit_trackers.write().await;
        let tracker = trackers
            .entry(cache_name.to_string())
            .or_insert_with(|| HitRateTracker::new(1000));
        tracker.record_hit();
    }

    /// Record a cache miss
    pub async fn record_cache_miss(&self, cache_name: &str, _reason: &str) {
        let mut trackers = self.hit_trackers.write().await;
        let tracker = trackers
            .entry(cache_name.to_string())
            .or_insert_with(|| HitRateTracker::new(1000));
        tracker.record_miss();
    }

    /// Record a cache put operation
    pub async fn record_cache_put(&self, cache_name: &str, size_bytes: usize) {
        let mut sizes = self.cache_sizes.write().await;
        *sizes.entry(cache_name.to_string()).or_insert(0) += size_bytes;
    }

    /// Record a cache eviction
    pub async fn record_cache_eviction(&self, cache_name: &str, reason: &str) {
        let mut trackers = self.eviction_trackers.write().await;
        let tracker = trackers.entry(cache_name.to_string()).or_insert_with(EvictionTracker::new);
        tracker.record_eviction(reason);
    }

    /// Record cache invalidation
    pub async fn record_cache_invalidation(&self, cache_name: &str) {
        // Implement invalidation tracking
        let mut trackers = self.eviction_trackers.write().await;
        let tracker = trackers.entry(cache_name.to_string()).or_insert_with(EvictionTracker::new);
        tracker.record_eviction("invalidation");

        // Record invalidation as a special operation
        let mut monitor = self.performance_monitor.write().await;
        monitor.record_operation_time(&format!("{}_invalidation", cache_name), 0.1);

        tracing::debug!("Cache invalidation recorded for cache: {}", cache_name);
    }

    /// Record cache clear
    pub async fn record_cache_clear(&self, cache_name: &str, _entries_cleared: usize) {
        let mut sizes = self.cache_sizes.write().await;
        sizes.insert(cache_name.to_string(), 0);
    }

    /// Record operation time
    pub async fn record_operation_time(&self, operation: &str, time_ms: f32) {
        let mut monitor = self.performance_monitor.write().await;
        monitor.record_operation_time(operation, time_ms);
    }

    /// Get hit rate for a cache
    pub async fn get_hit_rate(&self, cache_name: &str) -> f32 {
        let trackers = self.hit_trackers.read().await;
        if let Some(tracker) = trackers.get(cache_name) {
            tracker.get_hit_rate()
        } else {
            0.0
        }
    }

    /// Get comprehensive metrics
    pub async fn get_metrics(&self) -> CacheMetrics {
        let trackers = self.hit_trackers.read().await;
        let eviction_trackers = self.eviction_trackers.read().await;
        let performance_monitor = self.performance_monitor.read().await;
        let cache_sizes = self.cache_sizes.read().await;

        let mut hit_rates = HashMap::new();
        let mut miss_rates = HashMap::new();
        let mut eviction_rates = HashMap::new();
        let mut average_latency_ms = HashMap::new();

        // Collect hit rates
        for (cache_name, tracker) in trackers.iter() {
            let hit_rate = tracker.get_hit_rate();
            hit_rates.insert(cache_name.clone(), hit_rate);
            miss_rates.insert(cache_name.clone(), 1.0 - hit_rate);
        }

        // Collect eviction rates
        for (cache_name, tracker) in eviction_trackers.iter() {
            let eviction_rate = tracker.get_eviction_rate("total");
            eviction_rates.insert(cache_name.clone(), eviction_rate);
        }

        // Collect average latencies
        for cache_name in hit_rates.keys() {
            let avg_latency =
                performance_monitor.get_average_latency(&format!("{}_get", cache_name));
            average_latency_ms.insert(cache_name.clone(), avg_latency);
        }

        let total_hits: u64 = trackers.values().map(|t| t.hits).sum();
        let total_misses: u64 = trackers.values().map(|t| t.misses).sum();
        let memory_usage_bytes: usize = cache_sizes.values().sum();

        CacheMetrics {
            hit_rates,
            miss_rates,
            eviction_rates,
            average_latency_ms,
            cache_sizes: cache_sizes.clone(),
            memory_usage_bytes,
            total_requests: total_hits + total_misses,
            total_hits,
            total_misses,
        }
    }

    /// Collect periodic metrics
    pub async fn collect_periodic_metrics(&self) {
        // Implement periodic metrics collection (e.g., send to monitoring system)
        let metrics = self.get_metrics().await;

        // Log comprehensive metrics for monitoring systems to pick up
        tracing::info!(
            "Cache metrics collected: total_requests={}, total_hits={}, total_misses={}, memory_usage_bytes={}",
            metrics.total_requests,
            metrics.total_hits,
            metrics.total_misses,
            metrics.memory_usage_bytes
        );

        // Log hit rates for each cache
        for (cache_name, hit_rate) in &metrics.hit_rates {
            tracing::info!(
                "Cache {} metrics: hit_rate={:.2}, miss_rate={:.2}, avg_latency_ms={:.2}",
                cache_name,
                hit_rate,
                metrics.miss_rates.get(cache_name).unwrap_or(&0.0),
                metrics.average_latency_ms.get(cache_name).unwrap_or(&0.0)
            );
        }

        // In a production environment, you would send these metrics to:
        // - Prometheus (via metrics crate)
        // - CloudWatch
        // - DataDog
        // - Grafana
        // - Custom monitoring endpoint

        // Example: Send to metrics endpoint (commented for now)
        // if let Err(e) = self.send_to_monitoring_system(&metrics).await {
        //     tracing::warn!("Failed to send metrics to monitoring system: {}", e);
        // }
    }

    /// Record request completion for result cache
    pub fn record_request_completion(&self, result: &super::result_cache::CacheResult) {
        // Implement request completion tracking
        let result_size = serde_json::to_vec(result).unwrap_or_default().len();

        // Use spawn to avoid blocking since this method is not async
        let performance_monitor = Arc::clone(&self.performance_monitor);
        tokio::spawn(async move {
            let mut monitor = performance_monitor.write().await;

            // Record completion timing (assuming processing time based on result size)
            let estimated_processing_time = (result_size as f32 / 1024.0).max(0.1); // Min 0.1ms
            monitor.record_operation_time("request_completion", estimated_processing_time);

            // Record result size for analytics
            monitor.record_operation_time("result_size_kb", result_size as f32 / 1024.0);
        });

        tracing::trace!(
            "Request completion recorded: result_size_bytes={}",
            result_size
        );
    }
}
