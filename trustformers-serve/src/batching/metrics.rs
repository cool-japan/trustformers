//! Metrics Collection and Optimization for Dynamic Batching

use crate::batching::{
    aggregator::{OptimizationSuggestion, ProcessingOutput, ProcessingResult, RequestBatch},
    config::OptimizationTarget,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Counters that must be readable without `.await`.
///
/// [`MetricsCollector::get_summary`] is synchronous — it is called from
/// `get_stats`, which feeds `/admin/stats` — so the figures it reports cannot
/// live behind an async `RwLock`. They are kept here as atomics, updated on the
/// real batch-formation and request-completion paths.
///
/// Before 0.2.1 `record_batch_formed` and `record_request_completion` had empty
/// bodies marked "would update metrics", and `get_summary` returned a struct of
/// zeros commented "return mock values for testing". Those zeros were served
/// over `/admin/stats` as a measurement of a running system.
#[derive(Debug, Default)]
struct SyncCounters {
    /// Batches formed by the aggregator.
    total_batches: AtomicU64,
    /// Requests across all formed batches.
    total_requests_batched: AtomicU64,
    /// Requests that reached a completion result.
    total_completions: AtomicU64,
    /// Summed completion latency, in milliseconds.
    total_latency_ms: AtomicU64,
    /// Completions whose output was an error.
    total_errors: AtomicU64,
    /// Requests that have been batched but whose completion has not yet been
    /// recorded.
    ///
    /// Note this counts *batched* work, not queued-but-unbatched work: a
    /// request waiting in the aggregator for a batch to form is not included,
    /// because batch formation is the first event this collector observes.
    queue_depth: AtomicU64,
}

/// Metrics collector for batching system
pub struct MetricsCollector {
    metrics: Arc<RwLock<BatchingMetrics>>,
    latency_tracker: Arc<LatencyTracker>,
    throughput_monitor: Arc<ThroughputMonitor>,
    batch_optimizer: Arc<BatchSizeOptimizer>,
    history_window: Duration,
    /// Synchronously readable counters backing [`MetricsCollector::get_summary`].
    counters: Arc<SyncCounters>,
    /// When the collector started, for the throughput figure.
    started_at: Instant,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(BatchingMetrics::default())),
            latency_tracker: Arc::new(LatencyTracker::new()),
            throughput_monitor: Arc::new(ThroughputMonitor::new()),
            batch_optimizer: Arc::new(BatchSizeOptimizer::new()),
            history_window: Duration::from_secs(300), // 5 minutes
            counters: Arc::new(SyncCounters::default()),
            started_at: Instant::now(),
        }
    }

    /// Record that the aggregator formed `batch`.
    ///
    /// The requests it carries are added to the in-flight depth and released
    /// again as each completes, so the reported depth is batched work that has
    /// not yet finished.
    pub fn record_batch_formed(&self, batch: &RequestBatch) {
        let requests = batch.requests.len() as u64;
        self.counters.total_batches.fetch_add(1, AtomicOrdering::Relaxed);
        self.counters
            .total_requests_batched
            .fetch_add(requests, AtomicOrdering::Relaxed);
        self.counters.queue_depth.fetch_add(requests, AtomicOrdering::Relaxed);
    }

    /// Record that a request reached `result`.
    ///
    /// Latency is the figure the processor measured for that request; error
    /// outputs are counted separately so the summary cannot present failures as
    /// successful throughput.
    pub fn record_request_completion(&self, result: &ProcessingResult) {
        self.counters.total_completions.fetch_add(1, AtomicOrdering::Relaxed);
        self.counters
            .total_latency_ms
            .fetch_add(result.latency_ms, AtomicOrdering::Relaxed);
        if matches!(result.output, ProcessingOutput::Error(_)) {
            self.counters.total_errors.fetch_add(1, AtomicOrdering::Relaxed);
        }
        // Saturating: a completion for a batch formed before this collector
        // existed must not wrap the gauge to u64::MAX.
        let _ = self.counters.queue_depth.fetch_update(
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
            |depth| Some(depth.saturating_sub(1)),
        );
    }

    /// Requests that completed with an error output.
    pub fn total_errors(&self) -> u64 {
        self.counters.total_errors.load(AtomicOrdering::Relaxed)
    }

    /// Requests that reached a completion result.
    pub fn total_completions(&self) -> u64 {
        self.counters.total_completions.load(AtomicOrdering::Relaxed)
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
    /// A synchronously readable summary of what this collector has observed.
    ///
    /// Every field is derived from the counters the record methods maintain; a
    /// collector that has seen no traffic reports genuine zeros, which is not
    /// the same thing as the hardcoded zeros this used to return regardless of
    /// load.
    ///
    /// `optimization_suggestions` is empty here by construction: producing them
    /// requires the async analysis in
    /// [`MetricsCollector::get_optimization_suggestions`], and inventing a
    /// plausible suggestion synchronously would be worse than offering none.
    pub fn get_summary(&self) -> crate::batching::MetricsSummary {
        let total_batches = self.counters.total_batches.load(AtomicOrdering::Relaxed);
        let total_requests = self.counters.total_requests_batched.load(AtomicOrdering::Relaxed);
        let completions = self.counters.total_completions.load(AtomicOrdering::Relaxed);
        let latency_ms = self.counters.total_latency_ms.load(AtomicOrdering::Relaxed);

        let avg_batch_size = if total_batches > 0 {
            total_requests as f32 / total_batches as f32
        } else {
            0.0
        };
        let avg_latency_ms =
            if completions > 0 { latency_ms as f32 / completions as f32 } else { 0.0 };
        let elapsed = self.started_at.elapsed().as_secs_f32();
        let throughput_rps = if elapsed > 0.0 { completions as f32 / elapsed } else { 0.0 };

        crate::batching::MetricsSummary {
            total_batches: total_batches as usize,
            avg_batch_size,
            avg_latency_ms,
            throughput_rps,
            queue_depth: self.counters.queue_depth.load(AtomicOrdering::Relaxed) as usize,
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

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
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

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

impl Default for ThroughputMonitor {
    fn default() -> Self {
        Self::new()
    }
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

        let (Some(back), Some(front)) = (times.back(), times.front()) else {
            return 0.0;
        };
        let duration = back.duration_since(*front);
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
}

impl Default for BatchSizeOptimizer {
    fn default() -> Self {
        Self::new()
    }
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

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
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

    #[test]
    fn test_batching_metrics_default() {
        let m = BatchingMetrics::default();
        assert_eq!(m.total_batches, 0);
        assert_eq!(m.total_requests, 0);
        assert!((m.avg_batch_size - 0.0).abs() < 1e-9);
        assert_eq!(m.max_batch_size, 0);
        assert!(m.batch_size_history.is_empty());
    }

    #[test]
    fn test_batching_metrics_clean_old_data_empty() {
        let mut m = BatchingMetrics::default();
        m.clean_old_data(Duration::from_secs(60));
        assert!(m.batch_size_history.is_empty());
    }

    #[test]
    fn test_batching_metrics_clean_old_data_with_recent() {
        let mut m = BatchingMetrics::default();
        m.batch_size_history.push_back((Instant::now(), 32));
        m.latency_history.push_back((Instant::now(), 10.0));
        m.throughput_history.push_back((Instant::now(), 100.0));
        m.clean_old_data(Duration::from_secs(60));
        assert_eq!(m.batch_size_history.len(), 1);
        assert_eq!(m.latency_history.len(), 1);
        assert_eq!(m.throughput_history.len(), 1);
    }

    #[test]
    fn test_metrics_collector_new() {
        let collector = MetricsCollector::new();
        assert_eq!(collector.history_window, Duration::from_secs(300));
    }

    #[test]
    fn test_metrics_collector_default() {
        let collector = MetricsCollector::default();
        assert_eq!(collector.history_window, Duration::from_secs(300));
    }

    #[test]
    fn test_metrics_collector_get_summary() {
        let collector = MetricsCollector::new();
        let summary = collector.get_summary();
        assert_eq!(summary.total_batches, 0);
        assert!((summary.avg_batch_size - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_latency_tracker_new() {
        let tracker = LatencyTracker::new();
        assert_eq!(tracker.window_size, 1000);
    }

    #[test]
    fn test_latency_tracker_default() {
        let tracker = LatencyTracker::default();
        assert_eq!(tracker.window_size, 1000);
    }

    #[tokio::test]
    async fn test_latency_tracker_empty() {
        let tracker = LatencyTracker::new();
        let p50 = tracker.get_percentile(0.5).await;
        assert!((p50 - 0.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_latency_tracker_average_empty() {
        let tracker = LatencyTracker::new();
        let avg = tracker.get_average().await;
        assert!((avg - 0.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_latency_tracker_average() {
        let tracker = LatencyTracker::new();
        tracker.record(10.0).await;
        tracker.record(20.0).await;
        let avg = tracker.get_average().await;
        assert!((avg - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_throughput_monitor_new() {
        let monitor = ThroughputMonitor::new();
        assert_eq!(monitor.window, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_throughput_monitor_empty() {
        let monitor = ThroughputMonitor::new();
        let rps = monitor.get_current_rps().await;
        assert!((rps - 0.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_throughput_monitor_requests_in_window() {
        let monitor = ThroughputMonitor::new();
        monitor.record_request().await;
        let count = monitor.get_requests_in_window(Duration::from_secs(10)).await;
        assert_eq!(count, 1);
    }

    #[test]
    fn test_batch_size_optimizer_new() {
        let optimizer = BatchSizeOptimizer::new();
        assert!(matches!(
            optimizer.optimization_target,
            OptimizationTarget::Balanced
        ));
    }

    #[test]
    fn test_batch_size_optimizer_default() {
        let optimizer = BatchSizeOptimizer::default();
        assert!(matches!(
            optimizer.optimization_target,
            OptimizationTarget::Balanced
        ));
    }

    #[tokio::test]
    async fn test_batch_size_optimizer_insufficient_data() {
        let optimizer = BatchSizeOptimizer::new();
        let metrics = BatchingMetrics::default();
        let result = optimizer.get_optimal_size(&metrics).await;
        assert!(result.is_none());
    }

    #[test]
    fn test_memory_tracker_new() {
        let tracker = MemoryTracker::new();
        // Just verify construction
        assert!(matches!(tracker, MemoryTracker { .. }));
    }

    #[tokio::test]
    async fn test_memory_tracker_peak() {
        let tracker = MemoryTracker::new();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        tracker.track_allocation(id1, 500).await;
        tracker.track_allocation(id2, 500).await;
        assert_eq!(tracker.get_peak_usage().await, 1000);
        tracker.track_deallocation(id1).await;
        assert_eq!(tracker.get_peak_usage().await, 1000); // Peak stays
        assert_eq!(tracker.get_current_usage().await, 500);
    }

    // ── Regression tests: the summary used to be hardcoded zeros ──

    use crate::batching::aggregator::{Request, RequestId, RequestInput};
    use crate::batching::config::Priority;
    use std::collections::HashMap;

    fn test_batch(request_count: usize) -> RequestBatch {
        let requests: Vec<Request> = (0..request_count)
            .map(|i| Request {
                id: RequestId::new(),
                input: RequestInput::Text {
                    text: format!("request {i}"),
                    max_length: Some(8),
                },
                priority: Priority::Normal,
                submitted_at: Instant::now(),
                deadline: None,
                metadata: HashMap::new(),
            })
            .collect();
        RequestBatch {
            id: uuid::Uuid::new_v4(),
            requests,
            created_at: Instant::now(),
            total_memory: 0,
            max_sequence_length: 8,
            priority: Priority::Normal,
        }
    }

    fn completion(latency_ms: u64, output: ProcessingOutput) -> ProcessingResult {
        ProcessingResult {
            request_id: RequestId::new(),
            output,
            latency_ms,
            batch_id: uuid::Uuid::new_v4(),
        }
    }

    /// Regression: `record_batch_formed` and `record_request_completion` had
    /// empty bodies, and `get_summary` returned hardcoded zeros commented
    /// "return mock values for testing" — which `/admin/stats` served as a
    /// measurement of a running system. The summary must now reflect real
    /// observed traffic.
    #[test]
    fn summary_reflects_recorded_traffic_rather_than_hardcoded_zeros() {
        let collector = MetricsCollector::new();

        // A fresh collector genuinely has seen nothing.
        let empty = collector.get_summary();
        assert_eq!(empty.total_batches, 0);
        assert_eq!(empty.queue_depth, 0);

        collector.record_batch_formed(&test_batch(4));
        collector.record_batch_formed(&test_batch(2));

        let summary = collector.get_summary();
        assert_eq!(summary.total_batches, 2, "the old code always reported 0");
        assert!(
            (summary.avg_batch_size - 3.0).abs() < 1e-6,
            "six requests over two batches averages three, got {}",
            summary.avg_batch_size
        );
        assert_eq!(
            summary.queue_depth, 6,
            "all six requests are still in flight"
        );
    }

    /// Latency and throughput come from the recorded completions.
    #[test]
    fn summary_latency_is_the_mean_of_recorded_completions() {
        let collector = MetricsCollector::new();
        collector.record_batch_formed(&test_batch(2));

        collector.record_request_completion(&completion(100, ProcessingOutput::Text("a".into())));
        collector.record_request_completion(&completion(300, ProcessingOutput::Text("b".into())));

        let summary = collector.get_summary();
        assert!(
            (summary.avg_latency_ms - 200.0).abs() < 1e-6,
            "100ms and 300ms average to 200ms, got {}",
            summary.avg_latency_ms
        );
        assert!(
            summary.throughput_rps > 0.0,
            "two completions is not zero throughput"
        );
        assert_eq!(summary.queue_depth, 0, "both requests completed");
    }

    /// Errored completions are counted as errors, not as successful work.
    #[test]
    fn errored_completions_are_counted_separately() {
        let collector = MetricsCollector::new();
        collector.record_batch_formed(&test_batch(2));

        collector.record_request_completion(&completion(10, ProcessingOutput::Text("ok".into())));
        collector
            .record_request_completion(&completion(10, ProcessingOutput::Error("boom".into())));

        assert_eq!(collector.total_completions(), 2);
        assert_eq!(collector.total_errors(), 1);
    }

    /// The queue-depth gauge must not wrap when a completion arrives for work
    /// this collector never saw formed.
    #[test]
    fn queue_depth_does_not_underflow() {
        let collector = MetricsCollector::new();
        collector.record_request_completion(&completion(5, ProcessingOutput::Text("x".into())));
        assert_eq!(collector.get_summary().queue_depth, 0);
    }
}
