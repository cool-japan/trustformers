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

        if self.hit_history.len() > self.window_size && !self.hit_history.remove(0) {
            self.misses = self.misses.saturating_sub(1);
        }
    }

    /// Record a cache miss
    pub fn record_miss(&mut self) {
        self.misses += 1;
        self.hit_history.push(false);

        if self.hit_history.len() > self.window_size && self.hit_history.remove(0) {
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
            sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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

#[cfg(test)]
mod tests {
    use super::*;

    // --- HitRateTracker tests ---

    #[test]
    fn test_hit_rate_tracker_initial_rate_zero() {
        let tracker = HitRateTracker::new(100);
        assert!((tracker.get_hit_rate() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate_tracker_all_hits() {
        let mut tracker = HitRateTracker::new(100);
        for _ in 0..10 {
            tracker.record_hit();
        }
        assert!((tracker.get_hit_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate_tracker_all_misses() {
        let mut tracker = HitRateTracker::new(100);
        for _ in 0..10 {
            tracker.record_miss();
        }
        assert!((tracker.get_hit_rate() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate_tracker_mixed() {
        let mut tracker = HitRateTracker::new(200);
        for _ in 0..3 {
            tracker.record_hit();
        }
        for _ in 0..7 {
            tracker.record_miss();
        }
        let rate = tracker.get_hit_rate();
        assert!((rate - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate_tracker_windowed_rate() {
        let mut tracker = HitRateTracker::new(100);
        tracker.record_hit();
        tracker.record_hit();
        tracker.record_miss();
        let windowed = tracker.get_windowed_hit_rate();
        // 2 hits out of 3 total
        assert!((windowed - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_hit_rate_tracker_windowed_empty_returns_zero() {
        let tracker = HitRateTracker::new(100);
        assert!((tracker.get_windowed_hit_rate() - 0.0).abs() < 1e-6);
    }

    // --- EvictionTracker tests ---

    #[test]
    fn test_eviction_tracker_initial_zero() {
        let tracker = EvictionTracker::new();
        assert_eq!(tracker.get_total_evictions(), 0);
    }

    #[test]
    fn test_eviction_tracker_record_and_count() {
        let mut tracker = EvictionTracker::new();
        tracker.record_eviction("size_limit");
        tracker.record_eviction("size_limit");
        tracker.record_eviction("expired");
        assert_eq!(tracker.get_total_evictions(), 3);
    }

    #[test]
    fn test_eviction_tracker_rate_by_reason() {
        let mut tracker = EvictionTracker::new();
        tracker.record_eviction("expired");
        tracker.record_eviction("expired");
        tracker.record_eviction("size_limit");
        let rate = tracker.get_eviction_rate("expired");
        assert!((rate - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_eviction_tracker_unknown_reason_rate_zero() {
        let mut tracker = EvictionTracker::new();
        tracker.record_eviction("expired");
        let rate = tracker.get_eviction_rate("unknown_reason");
        assert!((rate - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_eviction_tracker_rate_when_empty_is_zero() {
        let tracker = EvictionTracker::new();
        assert!((tracker.get_eviction_rate("any") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_eviction_tracker_default_is_same_as_new() {
        let tracker = EvictionTracker::default();
        assert_eq!(tracker.get_total_evictions(), 0);
    }

    // --- PerformanceMonitor tests ---

    #[test]
    fn test_performance_monitor_initial_latency_zero() {
        let monitor = PerformanceMonitor::new(100);
        assert!((monitor.get_average_latency("get") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_performance_monitor_record_and_average() {
        let mut monitor = PerformanceMonitor::new(100);
        monitor.record_operation_time("get", 10.0);
        monitor.record_operation_time("get", 20.0);
        let avg = monitor.get_average_latency("get");
        assert!((avg - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_performance_monitor_percentile_latency() {
        let mut monitor = PerformanceMonitor::new(200);
        for i in 1..=10u32 {
            monitor.record_operation_time("put", i as f32 * 10.0);
        }
        let p50 = monitor.get_percentile_latency("put", 50.0);
        // sorted: [10,20,30,40,50,60,70,80,90,100], index=4 => 50.0
        assert!((p50 - 50.0).abs() < 1e-4);
    }

    #[test]
    fn test_performance_monitor_percentile_empty_returns_zero() {
        let monitor = PerformanceMonitor::new(100);
        assert!((monitor.get_percentile_latency("missing", 99.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_performance_monitor_window_caps_samples() {
        let mut monitor = PerformanceMonitor::new(5);
        for i in 0..10u32 {
            monitor.record_operation_time("op", i as f32);
        }
        // Only last 5 samples: 5,6,7,8,9 => avg = 7.0
        let avg = monitor.get_average_latency("op");
        assert!((avg - 7.0).abs() < 1e-4);
    }

    // --- CacheStatsCollector async tests ---

    #[tokio::test]
    async fn test_stats_collector_hit_rate_starts_zero() {
        let collector = CacheStatsCollector::new();
        let rate = collector.get_hit_rate("test_cache").await;
        assert!((rate - 0.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_stats_collector_record_hits_and_misses() {
        let collector = CacheStatsCollector::new();
        collector.record_cache_hit("my_cache").await;
        collector.record_cache_hit("my_cache").await;
        collector.record_cache_miss("my_cache", "not_found").await;
        let rate = collector.get_hit_rate("my_cache").await;
        assert!((rate - 2.0 / 3.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_stats_collector_get_metrics_totals() {
        let collector = CacheStatsCollector::new();
        collector.record_cache_hit("cache_a").await;
        collector.record_cache_miss("cache_a", "reason").await;
        collector.record_cache_hit("cache_b").await;
        let metrics = collector.get_metrics().await;
        assert_eq!(metrics.total_hits, 2);
        assert_eq!(metrics.total_misses, 1);
        assert_eq!(metrics.total_requests, 3);
    }

    #[tokio::test]
    async fn test_stats_collector_record_put_increases_size() {
        let collector = CacheStatsCollector::new();
        collector.record_cache_put("cache_x", 1024).await;
        collector.record_cache_put("cache_x", 512).await;
        let metrics = collector.get_metrics().await;
        let size = metrics.cache_sizes.get("cache_x").copied().unwrap_or(0);
        assert_eq!(size, 1536);
    }

    #[tokio::test]
    async fn test_stats_collector_clear_resets_size() {
        let collector = CacheStatsCollector::new();
        collector.record_cache_put("cache_y", 2048).await;
        collector.record_cache_clear("cache_y", 10).await;
        let metrics = collector.get_metrics().await;
        let size = metrics.cache_sizes.get("cache_y").copied().unwrap_or(999);
        assert_eq!(size, 0);
    }

    #[tokio::test]
    async fn test_stats_collector_eviction_tracking() {
        let collector = CacheStatsCollector::new();
        collector.record_cache_eviction("cache_z", "size_limit").await;
        collector.record_cache_eviction("cache_z", "expired").await;
        // Verify metrics doesn't panic and returns valid data
        let metrics = collector.get_metrics().await;
        assert!(metrics.eviction_rates.contains_key("cache_z"));
    }

    #[tokio::test]
    async fn test_stats_collector_operation_time_recording() {
        let collector = CacheStatsCollector::new();
        collector.record_operation_time("get_op", 5.5).await;
        collector.record_operation_time("get_op", 6.5).await;
        // Just verify it doesn't panic and returns valid metrics
        let _metrics = collector.get_metrics().await;
    }
}
