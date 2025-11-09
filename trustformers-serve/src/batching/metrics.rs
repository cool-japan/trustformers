//! Metrics Collection and Optimization for Dynamic Batching

use crate::batching::{
    aggregator::{OptimizationSuggestion, ProcessingResult, RequestBatch},
    config::OptimizationTarget,
};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Metrics collector for batching system
pub struct MetricsCollector {
    metrics: Arc<RwLock<BatchingMetrics>>,
    latency_tracker: Arc<LatencyTracker>,
    throughput_monitor: Arc<ThroughputMonitor>,
    batch_optimizer: Arc<BatchSizeOptimizer>,
    history_window: Duration,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(BatchingMetrics::default())),
            latency_tracker: Arc::new(LatencyTracker::new()),
            throughput_monitor: Arc::new(ThroughputMonitor::new()),
            batch_optimizer: Arc::new(BatchSizeOptimizer::new()),
            history_window: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Record batch formation
    pub fn record_batch_formed(&self, _batch: &RequestBatch) {
        // Would update metrics
    }

    /// Record request completion
    pub fn record_request_completion(&self, _result: &ProcessingResult) {
        // Would update metrics
    }

    /// Collect periodic metrics
    pub async fn collect_periodic_metrics(&self) {
        let mut metrics = self.metrics.write().await;

        // Update throughput
        metrics.current_throughput = self.throughput_monitor.get_current_rps().await;

        // Update latency percentiles
        metrics.latency_p50 = self.latency_tracker.get_percentile(0.5).await;
        metrics.latency_p95 = self.latency_tracker.get_percentile(0.95).await;
        metrics.latency_p99 = self.latency_tracker.get_percentile(0.99).await;

        // Clean old data
        metrics.clean_old_data(self.history_window);
    }

    /// Get optimization suggestions
    pub async fn get_optimization_suggestions(&self) -> Option<Vec<OptimizationSuggestion>> {
        let metrics = self.metrics.read().await;
        let mut suggestions = Vec::new();

        // Analyze batch size efficiency
        if let Some(optimal_size) = self.batch_optimizer.get_optimal_size(&metrics).await {
            if optimal_size > metrics.avg_batch_size * 1.2 {
                suggestions.push(OptimizationSuggestion::IncreaseBatchSize(
                    optimal_size as usize,
                ));
            } else if optimal_size < metrics.avg_batch_size * 0.8 {
                suggestions.push(OptimizationSuggestion::DecreaseBatchSize(
                    optimal_size as usize,
                ));
            }
        }

        // Analyze timeout efficiency
        if metrics.timeout_ratio > 0.3 {
            // Too many timeouts, reduce wait time
            let new_timeout = Duration::from_millis((metrics.avg_wait_time * 0.8) as u64);
            suggestions.push(OptimizationSuggestion::AdjustTimeout(new_timeout));
        }

        // Check if bucketing would help
        if metrics.padding_overhead > 0.2 {
            suggestions.push(OptimizationSuggestion::EnableBucketing);
        }

        if suggestions.is_empty() {
            None
        } else {
            Some(suggestions)
        }
    }

    /// Get metrics summary
    pub fn get_summary(&self) -> crate::batching::MetricsSummary {
        // Would need async access
        crate::batching::MetricsSummary {
            avg_batch_size: 0.0,
            avg_latency_ms: 0.0,
            throughput_rps: 0.0,
            queue_depth: 0,
            optimization_suggestions: vec![],
        }
    }
}

/// Core batching metrics
#[derive(Debug, Clone, Default)]
pub struct BatchingMetrics {
    // Batch statistics
    pub total_batches: usize,
    pub total_requests: usize,
    pub avg_batch_size: f32,
    pub max_batch_size: usize,
    pub min_batch_size: usize,

    // Timing metrics
    pub avg_wait_time: f32,
    pub max_wait_time: f32,
    pub avg_processing_time: f32,

    // Throughput metrics
    pub current_throughput: f32,
    pub peak_throughput: f32,
    pub requests_per_batch: f32,

    // Latency percentiles
    pub latency_p50: f32,
    pub latency_p95: f32,
    pub latency_p99: f32,

    // Efficiency metrics
    pub gpu_utilization: f32,
    pub memory_utilization: f32,
    pub padding_overhead: f32,
    pub timeout_ratio: f32,

    // Historical data
    pub batch_size_history: VecDeque<(Instant, usize)>,
    pub latency_history: VecDeque<(Instant, f32)>,
    pub throughput_history: VecDeque<(Instant, f32)>,
}

impl BatchingMetrics {
    /// Clean old historical data
    fn clean_old_data(&mut self, window: Duration) {
        let cutoff = Instant::now() - window;

        // Clean batch size history
        while let Some(&(time, _)) = self.batch_size_history.front() {
            if time < cutoff {
                self.batch_size_history.pop_front();
            } else {
                break;
            }
        }

        // Clean latency history
        while let Some(&(time, _)) = self.latency_history.front() {
            if time < cutoff {
                self.latency_history.pop_front();
            } else {
                break;
            }
        }

        // Clean throughput history
        while let Some(&(time, _)) = self.throughput_history.front() {
            if time < cutoff {
                self.throughput_history.pop_front();
            } else {
                break;
            }
        }
    }
}

/// Latency tracking
pub struct LatencyTracker {
    samples: Arc<RwLock<Vec<f32>>>,
    window_size: usize,
}

impl LatencyTracker {
    pub fn new() -> Self {
        Self {
            samples: Arc::new(RwLock::new(Vec::new())),
            window_size: 1000,
        }
    }

    /// Record a latency sample
    pub async fn record(&self, latency_ms: f32) {
        let mut samples = self.samples.write().await;
        samples.push(latency_ms);

        // Keep only recent samples
        if samples.len() > self.window_size {
            samples.remove(0);
        }
    }

    /// Get percentile value
    pub async fn get_percentile(&self, percentile: f32) -> f32 {
        let mut samples = self.samples.read().await.clone();
        if samples.is_empty() {
            return 0.0;
        }

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = ((samples.len() - 1) as f32 * percentile) as usize;
        samples[index]
    }

    /// Get average latency
    pub async fn get_average(&self) -> f32 {
        let samples = self.samples.read().await;
        if samples.is_empty() {
            return 0.0;
        }

        samples.iter().sum::<f32>() / samples.len() as f32
    }
}

/// Throughput monitoring
pub struct ThroughputMonitor {
    request_times: Arc<RwLock<VecDeque<Instant>>>,
    window: Duration,
}

impl ThroughputMonitor {
    pub fn new() -> Self {
        Self {
            request_times: Arc::new(RwLock::new(VecDeque::new())),
            window: Duration::from_secs(60),
        }
    }

    /// Record a request
    pub async fn record_request(&self) {
        let mut times = self.request_times.write().await;
        let now = Instant::now();
        times.push_back(now);

        // Remove old entries
        let cutoff = now - self.window;
        while let Some(&front_time) = times.front() {
            if front_time < cutoff {
                times.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get current requests per second
    pub async fn get_current_rps(&self) -> f32 {
        let times = self.request_times.read().await;
        if times.len() < 2 {
            return 0.0;
        }

        let duration = times.back().unwrap().duration_since(*times.front().unwrap());
        if duration.as_secs_f32() > 0.0 {
            times.len() as f32 / duration.as_secs_f32()
        } else {
            0.0
        }
    }

    /// Get requests in time window
    pub async fn get_requests_in_window(&self, window: Duration) -> usize {
        let times = self.request_times.read().await;
        let cutoff = Instant::now() - window;

        times.iter().filter(|&&t| t >= cutoff).count()
    }
}

/// Batch size optimizer
pub struct BatchSizeOptimizer {
    performance_history: Arc<RwLock<Vec<BatchPerformance>>>,
    optimization_target: OptimizationTarget,
}

#[derive(Debug, Clone)]
struct BatchPerformance {
    batch_size: usize,
    throughput: f32,
    avg_latency: f32,
    gpu_utilization: f32,
    timestamp: Instant,
}

impl BatchSizeOptimizer {
    pub fn new() -> Self {
        Self {
            performance_history: Arc::new(RwLock::new(Vec::new())),
            optimization_target: OptimizationTarget::Balanced,
        }
    }

    /// Record batch performance
    pub async fn record_performance(
        &self,
        batch_size: usize,
        throughput: f32,
        latency: f32,
        gpu_util: f32,
    ) {
        let mut history = self.performance_history.write().await;

        history.push(BatchPerformance {
            batch_size,
            throughput,
            avg_latency: latency,
            gpu_utilization: gpu_util,
            timestamp: Instant::now(),
        });

        // Keep only recent history
        if history.len() > 1000 {
            history.remove(0);
        }
    }

    /// Get optimal batch size based on history
    pub async fn get_optimal_size(&self, metrics: &BatchingMetrics) -> Option<f32> {
        let history = self.performance_history.read().await;
        if history.len() < 10 {
            return None;
        }

        // Group by batch size and calculate average performance
        let mut size_performance: HashMap<usize, (f32, f32, f32, usize)> = HashMap::new();

        for perf in history.iter() {
            let entry = size_performance.entry(perf.batch_size).or_insert((0.0, 0.0, 0.0, 0));
            entry.0 += perf.throughput;
            entry.1 += perf.avg_latency;
            entry.2 += perf.gpu_utilization;
            entry.3 += 1;
        }

        // Calculate scores for each batch size
        let mut best_size = metrics.avg_batch_size as usize;
        let mut best_score = 0.0;

        for (size, (total_throughput, total_latency, total_gpu, count)) in size_performance {
            let avg_throughput = total_throughput / count as f32;
            let avg_latency = total_latency / count as f32;
            let avg_gpu = total_gpu / count as f32;

            let score = match self.optimization_target {
                OptimizationTarget::Throughput => avg_throughput,
                OptimizationTarget::Latency => 1.0 / avg_latency,
                OptimizationTarget::Balanced => {
                    (avg_throughput / 100.0) * (1.0 / (avg_latency / 100.0)) * avg_gpu
                },
                OptimizationTarget::Cost => avg_throughput / (1.0 + (size as f32 / 32.0)),
            };

            if score > best_score {
                best_score = score;
                best_size = size;
            }
        }

        Some(best_size as f32)
    }
}

/// Memory utilization tracker
pub struct MemoryTracker {
    allocations: Arc<RwLock<HashMap<uuid::Uuid, usize>>>,
    peak_usage: Arc<RwLock<usize>>,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            allocations: Arc::new(RwLock::new(HashMap::new())),
            peak_usage: Arc::new(RwLock::new(0)),
        }
    }

    /// Track batch allocation
    pub async fn track_allocation(&self, batch_id: uuid::Uuid, size: usize) {
        let mut allocations = self.allocations.write().await;
        allocations.insert(batch_id, size);

        let total = allocations.values().sum::<usize>();
        let mut peak = self.peak_usage.write().await;
        if total > *peak {
            *peak = total;
        }
    }

    /// Track batch deallocation
    pub async fn track_deallocation(&self, batch_id: uuid::Uuid) {
        let mut allocations = self.allocations.write().await;
        allocations.remove(&batch_id);
    }

    /// Get current memory usage
    pub async fn get_current_usage(&self) -> usize {
        self.allocations.read().await.values().sum()
    }

    /// Get peak memory usage
    pub async fn get_peak_usage(&self) -> usize {
        *self.peak_usage.read().await
    }
}

use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_latency_tracker() {
        let tracker = LatencyTracker::new();

        // Record some samples
        for i in 1..=100 {
            tracker.record(i as f32).await;
        }

        let p50 = tracker.get_percentile(0.5).await;
        let p95 = tracker.get_percentile(0.95).await;

        assert!(p50 > 40.0 && p50 < 60.0);
        assert!(p95 > 90.0);
    }

    #[tokio::test]
    async fn test_throughput_monitor() {
        let monitor = ThroughputMonitor::new();

        // Record requests
        for _ in 0..10 {
            monitor.record_request().await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let rps = monitor.get_current_rps().await;
        assert!(rps > 0.0);
    }

    #[tokio::test]
    async fn test_memory_tracker() {
        let tracker = MemoryTracker::new();

        let batch_id = uuid::Uuid::new_v4();
        tracker.track_allocation(batch_id, 1000).await;

        assert_eq!(tracker.get_current_usage().await, 1000);

        tracker.track_deallocation(batch_id).await;
        assert_eq!(tracker.get_current_usage().await, 0);
    }
}
