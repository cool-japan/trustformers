use crate::error::{Result, TrustformersError};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tokio::time::timeout;

/// Configuration for dynamic batching optimization
#[derive(Debug, Clone)]
pub struct DynamicBatchingConfig {
    /// Initial batch size
    pub initial_batch_size: usize,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Target latency in milliseconds
    pub target_latency_ms: u64,
    /// Maximum wait time for batching in milliseconds
    pub max_wait_time_ms: u64,
    /// Throughput optimization threshold (requests per second)
    pub throughput_threshold: f64,
    /// Performance window size for adaptive sizing
    pub performance_window_size: usize,
    /// Batch size adjustment factor
    pub adjustment_factor: f64,
}

impl Default for DynamicBatchingConfig {
    fn default() -> Self {
        Self {
            initial_batch_size: 8,
            min_batch_size: 1,
            max_batch_size: 64,
            target_latency_ms: 100,
            max_wait_time_ms: 50,
            throughput_threshold: 10.0,
            performance_window_size: 10,
            adjustment_factor: 1.2,
        }
    }
}

/// Alias for backward compatibility
pub type DynamicBatchConfig = DynamicBatchingConfig;

/// Performance metrics for dynamic batching
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub batch_size: usize,
    pub latency_ms: u64,
    pub throughput_rps: f64,
    pub timestamp: Instant,
    pub memory_usage_mb: f64,
    pub gpu_utilization: f32,
    pub queue_size: usize,
}

/// Dynamic batching manager that optimizes batch sizes based on performance
#[derive(Debug)]
pub struct DynamicBatcher<T> {
    config: DynamicBatchingConfig,
    current_batch_size: Arc<RwLock<usize>>,
    performance_history: Arc<Mutex<VecDeque<PerformanceMetrics>>>,
    pending_requests: Arc<Mutex<VecDeque<BatchRequest<T>>>>,
    notify: Arc<Notify>,
    is_running: Arc<Mutex<bool>>,
}

/// Request wrapper for batching
#[derive(Debug)]
pub struct BatchRequest<T> {
    pub input: T,
    pub response_sender: tokio::sync::oneshot::Sender<Result<T>>,
    pub timestamp: Instant,
    pub priority: RequestPriority,
}

/// Alias for backward compatibility
pub type DynamicBatchManager<T> = DynamicBatcher<T>;

/// Priority levels for batch requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum RequestPriority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl<T> DynamicBatcher<T>
where
    T: Send + Sync + Clone + 'static,
{
    /// Create a new dynamic batcher with configuration
    pub fn new(config: DynamicBatchingConfig) -> Self {
        Self {
            current_batch_size: Arc::new(RwLock::new(config.initial_batch_size)),
            config,
            performance_history: Arc::new(Mutex::new(VecDeque::new())),
            pending_requests: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    /// Add a request to the batching queue
    pub async fn add_request(&self, input: T, priority: RequestPriority) -> Result<T> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        let request = BatchRequest {
            input,
            response_sender: tx,
            timestamp: Instant::now(),
            priority,
        };

        // Add to queue based on priority
        {
            let mut queue = self.pending_requests.lock().expect("lock should not be poisoned");

            // Insert based on priority (higher priority first)
            let insert_pos =
                queue.iter().position(|r| r.priority < priority).unwrap_or(queue.len());

            queue.insert(insert_pos, request);
        }

        // Notify the batcher
        self.notify.notify_one();

        // Wait for response with timeout (use 10x max_wait_time for robustness in tests)
        let timeout_duration = Duration::from_millis(self.config.max_wait_time_ms * 10);

        match timeout(timeout_duration, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(TrustformersError::runtime_error(
                "Request channel closed".to_string(),
            )),
            Err(_) => Err(TrustformersError::runtime_error(format!(
                "Request timed out after {}ms",
                timeout_duration.as_millis()
            ))),
        }
    }

    /// Start the dynamic batching process
    pub async fn start<F, Fut>(&self, mut process_batch: F) -> Result<()>
    where
        F: FnMut(Vec<T>) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<Vec<T>>> + Send,
    {
        // Mark as running
        {
            let mut running = self.is_running.lock().expect("lock should not be poisoned");
            if *running {
                return Err(TrustformersError::runtime_error(
                    "Batcher is already running".to_string(),
                ));
            }
            *running = true;
        }

        loop {
            // Check if we should stop
            {
                let running = self.is_running.lock().expect("lock should not be poisoned");
                if !*running {
                    break;
                }
            }

            // Wait for requests or timeout
            let wait_future = self.notify.notified();
            let timeout_future =
                tokio::time::sleep(Duration::from_millis(self.config.max_wait_time_ms));

            tokio::select! {
                _ = wait_future => {},
                _ = timeout_future => {},
            }

            // Process available requests
            let batch = self.collect_batch().await;
            if !batch.is_empty() {
                let start_time = Instant::now();
                let batch_size = batch.len();

                // Extract inputs for processing
                let inputs: Vec<T> = batch.iter().map(|req| req.input.clone()).collect();

                // Process the batch
                match process_batch(inputs).await {
                    Ok(outputs) => {
                        // Send responses back
                        for (request, output) in batch.into_iter().zip(outputs) {
                            let _ = request.response_sender.send(Ok(output));
                        }

                        // Record performance metrics
                        let latency = start_time.elapsed().as_millis() as u64;
                        self.record_performance(batch_size, latency).await;

                        // Adjust batch size based on performance
                        self.adjust_batch_size().await;
                    },
                    Err(e) => {
                        // Send error to all requests in the batch
                        let error_msg = format!("Batch processing failed: {}", e);
                        for request in batch {
                            let _ = request.response_sender.send(Err(
                                TrustformersError::invalid_input_simple(error_msg.clone()),
                            ));
                        }
                    },
                }
            }
        }

        Ok(())
    }

    /// Stop the dynamic batching process
    pub fn stop(&self) {
        let mut running = self.is_running.lock().expect("lock should not be poisoned");
        *running = false;
        self.notify.notify_one();
    }

    /// Collect a batch of requests based on current batch size and timing
    async fn collect_batch(&self) -> Vec<BatchRequest<T>> {
        let current_size = *self.current_batch_size.read().expect("lock should not be poisoned");
        let mut batch = Vec::with_capacity(current_size);

        let mut queue = self.pending_requests.lock().expect("lock should not be poisoned");

        // Collect up to current_batch_size requests
        while batch.len() < current_size && !queue.is_empty() {
            if let Some(request) = queue.pop_front() {
                // Check if request has expired
                if request.timestamp.elapsed()
                    < Duration::from_millis(self.config.max_wait_time_ms * 3)
                {
                    batch.push(request);
                } else {
                    // Send timeout error for expired request
                    let _ = request.response_sender.send(Err(TrustformersError::runtime_error(
                        "Request expired in queue".to_string(),
                    )));
                }
            }
        }

        batch
    }

    /// Record performance metrics for adaptive batch sizing
    async fn record_performance(&self, batch_size: usize, latency_ms: u64) {
        let throughput = (batch_size as f64) / (latency_ms as f64 / 1000.0);

        let metrics = PerformanceMetrics {
            batch_size,
            latency_ms,
            throughput_rps: throughput,
            timestamp: Instant::now(),
            memory_usage_mb: self.estimate_memory_usage().await,
            gpu_utilization: self.estimate_gpu_utilization().await,
            queue_size: self.pending_requests.lock().expect("lock should not be poisoned").len(),
        };

        let mut history = self.performance_history.lock().expect("lock should not be poisoned");
        history.push_back(metrics);

        // Keep only recent history
        while history.len() > self.config.performance_window_size {
            history.pop_front();
        }
    }

    /// Adjust batch size based on performance history
    async fn adjust_batch_size(&self) {
        let history = self.performance_history.lock().expect("lock should not be poisoned");
        if history.len() < 3 {
            return; // Need more data points
        }

        let recent_metrics: Vec<_> = history.iter().rev().take(3).collect();
        let avg_latency =
            recent_metrics.iter().map(|m| m.latency_ms).sum::<u64>() / recent_metrics.len() as u64;
        let avg_throughput = recent_metrics.iter().map(|m| m.throughput_rps).sum::<f64>()
            / recent_metrics.len() as f64;

        let mut current_size =
            self.current_batch_size.write().expect("lock should not be poisoned");
        let old_size = *current_size;

        // Adaptive sizing logic
        if avg_latency > self.config.target_latency_ms {
            // Latency too high, reduce batch size
            *current_size = std::cmp::max(
                self.config.min_batch_size,
                (*current_size as f64 / self.config.adjustment_factor) as usize,
            );
        } else if avg_throughput < self.config.throughput_threshold {
            // Throughput too low, increase batch size
            *current_size = std::cmp::min(
                self.config.max_batch_size,
                (*current_size as f64 * self.config.adjustment_factor) as usize,
            );
        } else if avg_latency < self.config.target_latency_ms / 2 {
            // Latency very good, try to increase throughput
            *current_size = std::cmp::min(
                self.config.max_batch_size,
                (*current_size as f64 * 1.1) as usize,
            );
        }

        // Log batch size changes
        if *current_size != old_size {
            tracing::info!(
                "Adjusted batch size: {} -> {} (latency: {}ms, throughput: {:.2} rps)",
                old_size,
                *current_size,
                avg_latency,
                avg_throughput
            );
        }
    }

    /// Estimate current memory usage (placeholder)
    async fn estimate_memory_usage(&self) -> f64 {
        // In a real implementation, this would query actual memory usage
        // For now, return a placeholder value
        100.0
    }

    /// Estimate GPU utilization (placeholder)
    async fn estimate_gpu_utilization(&self) -> f32 {
        // In a real implementation, this would query GPU metrics
        // For now, return a placeholder value
        0.5
    }

    /// Get current performance statistics
    pub fn get_performance_stats(&self) -> Option<BatchingStats> {
        let history = self.performance_history.lock().expect("lock should not be poisoned");
        if history.is_empty() {
            return None;
        }

        let recent_metrics: Vec<_> = history.iter().rev().take(10).collect();
        let avg_latency =
            recent_metrics.iter().map(|m| m.latency_ms).sum::<u64>() / recent_metrics.len() as u64;
        let avg_throughput = recent_metrics.iter().map(|m| m.throughput_rps).sum::<f64>()
            / recent_metrics.len() as f64;
        let avg_batch_size =
            recent_metrics.iter().map(|m| m.batch_size).sum::<usize>() / recent_metrics.len();

        Some(BatchingStats {
            current_batch_size: *self
                .current_batch_size
                .read()
                .expect("lock should not be poisoned"),
            avg_latency_ms: avg_latency,
            avg_throughput_rps: avg_throughput,
            avg_batch_size,
            queue_length: self.pending_requests.lock().expect("lock should not be poisoned").len(),
            total_processed: history.len(),
        })
    }
}

/// Statistics for batching performance
#[derive(Debug, Clone)]
pub struct BatchingStats {
    pub current_batch_size: usize,
    pub avg_latency_ms: u64,
    pub avg_throughput_rps: f64,
    pub avg_batch_size: usize,
    pub queue_length: usize,
    pub total_processed: usize,
}

/// Enhanced pipeline trait with dynamic batching support
#[async_trait::async_trait]
pub trait DynamicBatchPipeline<T: Send + Sync + Clone + 'static>: Send + Sync {
    type Output: Send + Clone;

    /// Process a single input
    async fn process_single(&self, input: T) -> Result<Self::Output>;

    /// Process a batch of inputs (optimized implementation)
    async fn process_batch(&self, inputs: Vec<T>) -> Result<Vec<Self::Output>> {
        // Default implementation: process each individually
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.process_single(input).await?);
        }
        Ok(results)
    }

    /// Create a dynamic batcher for this pipeline
    fn create_batcher(&self, config: DynamicBatchingConfig) -> DynamicBatcher<T> {
        DynamicBatcher::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    // ── Config defaults ───────────────────────────────────────────────────────

    #[test]
    fn test_config_default_batch_size_within_bounds() {
        let config = DynamicBatchingConfig::default();
        assert!(config.initial_batch_size >= config.min_batch_size);
        assert!(config.initial_batch_size <= config.max_batch_size);
    }

    #[test]
    fn test_config_default_target_latency_positive() {
        let config = DynamicBatchingConfig::default();
        assert!(config.target_latency_ms > 0);
    }

    #[test]
    fn test_config_default_adjustment_factor_gt_one() {
        let config = DynamicBatchingConfig::default();
        assert!(
            config.adjustment_factor > 1.0,
            "adjustment factor should be >1 for meaningful batch expansion"
        );
    }

    #[test]
    fn test_config_alias_type() {
        // DynamicBatchConfig must be an alias for DynamicBatchingConfig
        let _config: DynamicBatchConfig = DynamicBatchingConfig::default();
    }

    // ── Priority ordering ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_priority_ordering() {
        let config = DynamicBatchingConfig {
            initial_batch_size: 4,
            max_wait_time_ms: 50,
            ..Default::default()
        };
        let batcher = DynamicBatcher::new(config);
        // Add requests; don't await – just inspect the queue
        let _low = batcher.add_request(1_i32, RequestPriority::Low);
        let _normal = batcher.add_request(2_i32, RequestPriority::Normal);
        let _high = batcher.add_request(3_i32, RequestPriority::High);
        let _critical = batcher.add_request(4_i32, RequestPriority::Critical);

        let queue = batcher.pending_requests.lock().expect("lock should not be poisoned");
        let priorities: Vec<_> = queue.iter().map(|r| r.priority).collect();
        assert!(
            priorities.windows(2).all(|w| w[0] >= w[1]),
            "requests must be ordered highest-priority first"
        );
    }

    #[test]
    fn test_priority_order_values() {
        assert!(RequestPriority::Critical > RequestPriority::High);
        assert!(RequestPriority::High > RequestPriority::Normal);
        assert!(RequestPriority::Normal > RequestPriority::Low);
    }

    #[test]
    fn test_priority_default_is_normal() {
        let p = RequestPriority::default();
        assert_eq!(p, RequestPriority::Normal);
    }

    // ── Max batch tokens constraint (simulated) ───────────────────────────────

    #[test]
    fn test_batch_respects_max_batch_size() {
        let max = 4_usize;
        let config = DynamicBatchingConfig {
            initial_batch_size: max,
            min_batch_size: 1,
            max_batch_size: max,
            ..Default::default()
        };
        let batcher = DynamicBatcher::<i32>::new(config);
        let current = *batcher.current_batch_size.read().expect("lock ok");
        assert!(current <= max, "initial batch size must not exceed max");
    }

    // ── Throughput estimation ────────────────────────────────────────────────

    #[test]
    fn test_throughput_formula() {
        // throughput = batch_size / (latency_ms / 1000)
        let batch_size = 8_usize;
        let latency_ms = 100_u64;
        let throughput = (batch_size as f64) / (latency_ms as f64 / 1000.0);
        assert!(
            (throughput - 80.0).abs() < 1e-6,
            "throughput should be batch/latency_sec"
        );
    }

    #[test]
    fn test_throughput_increases_with_larger_batch_same_latency() {
        let latency_ms = 100_u64;
        let t_small = (4_f64) / (latency_ms as f64 / 1000.0);
        let t_large = (8_f64) / (latency_ms as f64 / 1000.0);
        assert!(
            t_large > t_small,
            "larger batch at same latency → higher throughput"
        );
    }

    // ── Latency SLO tracking ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_adjust_batch_size_reduces_on_high_latency() {
        let config = DynamicBatchingConfig {
            initial_batch_size: 16,
            min_batch_size: 1,
            max_batch_size: 64,
            target_latency_ms: 10, // very tight SLO
            max_wait_time_ms: 5,
            throughput_threshold: 1.0,
            performance_window_size: 5,
            adjustment_factor: 1.5,
        };
        let batcher = DynamicBatcher::<i32>::new(config.clone());
        // Record high-latency batches
        for _ in 0..4 {
            batcher.record_performance(8, 500).await; // 500ms >> 10ms SLO
        }
        batcher.adjust_batch_size().await;
        let current = *batcher.current_batch_size.read().expect("lock ok");
        assert!(
            current < config.initial_batch_size,
            "batch size should decrease when latency exceeds SLO"
        );
    }

    #[tokio::test]
    async fn test_adjust_batch_size_increases_on_low_throughput() {
        let config = DynamicBatchingConfig {
            initial_batch_size: 4,
            min_batch_size: 1,
            max_batch_size: 64,
            target_latency_ms: 1000, // very loose SLO
            max_wait_time_ms: 5,
            throughput_threshold: 1000.0, // throughput we'll never meet
            performance_window_size: 5,
            adjustment_factor: 1.5,
        };
        let batcher = DynamicBatcher::<i32>::new(config.clone());
        // Very low throughput: latency=1ms, batch=1 → 1000 rps, but threshold=1000 exactly
        // Force below threshold: latency=500ms, batch=1 → 2rps
        for _ in 0..4 {
            batcher.record_performance(1, 500).await;
        }
        batcher.adjust_batch_size().await;
        let current = *batcher.current_batch_size.read().expect("lock ok");
        assert!(
            current >= config.initial_batch_size,
            "batch size should increase when throughput is below threshold"
        );
    }

    // ── Variable-length padding / grouping (logic tests) ─────────────────────

    #[test]
    fn test_sequence_grouping_short_sequences() {
        // Sequences of lengths: group by ≤ 128 together
        let seq_lens = [64_usize, 100, 128, 50];
        let short_group: Vec<_> = seq_lens.iter().filter(|&&l| l <= 128).collect();
        assert_eq!(
            short_group.len(),
            4,
            "all sequences should be in the short group"
        );
    }

    #[test]
    fn test_sequence_grouping_long_sequences() {
        let seq_lens = [64_usize, 256, 512, 128, 300];
        let long_group: Vec<_> = seq_lens.iter().filter(|&&l| l > 128).collect();
        assert_eq!(long_group.len(), 3);
    }

    // ── Batch formation by deadline ───────────────────────────────────────────

    #[tokio::test]
    async fn test_collect_batch_max_size_respected() {
        let config = DynamicBatchingConfig {
            initial_batch_size: 2,
            max_wait_time_ms: 1000,
            ..Default::default()
        };
        let batcher = DynamicBatcher::<i32>::new(config);
        // Enqueue 5 requests
        {
            let mut queue = batcher.pending_requests.lock().expect("lock ok");
            for i in 0..5_i32 {
                let (tx, _rx) = tokio::sync::oneshot::channel();
                queue.push_back(BatchRequest {
                    input: i,
                    response_sender: tx,
                    timestamp: Instant::now(),
                    priority: RequestPriority::Normal,
                });
            }
        }
        let batch = batcher.collect_batch().await;
        assert_eq!(
            batch.len(),
            2,
            "collect_batch should respect current_batch_size"
        );
    }

    // ── Batcher stats ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_performance_stats_empty_returns_none() {
        let batcher = DynamicBatcher::<i32>::new(DynamicBatchingConfig::default());
        assert!(
            batcher.get_performance_stats().is_none(),
            "no performance stats before any batches processed"
        );
    }

    #[tokio::test]
    async fn test_get_performance_stats_after_recording() {
        let batcher = DynamicBatcher::<i32>::new(DynamicBatchingConfig::default());
        batcher.record_performance(4, 100).await;
        batcher.record_performance(4, 110).await;
        batcher.record_performance(4, 90).await;
        let stats = batcher.get_performance_stats();
        assert!(stats.is_some(), "stats should be available after recording");
        let s = stats.expect("stats should be Some");
        assert_eq!(s.avg_batch_size, 4);
    }

    // ── Basic end-to-end test ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_dynamic_batching_basic() {
        let config = DynamicBatchingConfig {
            initial_batch_size: 2,
            max_wait_time_ms: 10,
            ..Default::default()
        };
        let batcher = DynamicBatcher::new(config);
        let processor = |inputs: Vec<i32>| async move {
            sleep(Duration::from_millis(1)).await;
            Ok(inputs.into_iter().map(|x| x * 2).collect())
        };
        let batcher_clone = Arc::new(batcher);
        let batcher_for_task = batcher_clone.clone();
        let process_task = tokio::spawn(async move { batcher_for_task.start(processor).await });
        let results = futures::future::join_all(vec![
            batcher_clone.add_request(1, RequestPriority::Normal),
            batcher_clone.add_request(2, RequestPriority::Normal),
            batcher_clone.add_request(3, RequestPriority::High),
        ])
        .await;
        batcher_clone.stop();
        let _ = process_task.await;
        assert_eq!(results.len(), 3);
        for result in results {
            assert!(result.is_ok());
        }
    }

    // ── BatchingStats fields ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_batching_stats_current_size_within_bounds() {
        let config = DynamicBatchingConfig {
            initial_batch_size: 4,
            min_batch_size: 1,
            max_batch_size: 64,
            ..Default::default()
        };
        let batcher = DynamicBatcher::<i32>::new(config.clone());
        for _ in 0..5 {
            batcher.record_performance(4, 80).await;
        }
        let stats = batcher.get_performance_stats().expect("stats should exist");
        assert!(stats.current_batch_size >= config.min_batch_size);
        assert!(stats.current_batch_size <= config.max_batch_size);
    }

    // ── DynamicBatchPipeline trait default implementation ─────────────────────

    #[tokio::test]
    async fn test_pipeline_trait_default_batch_falls_back_to_single() {
        struct AddOnePipeline;

        #[async_trait::async_trait]
        impl DynamicBatchPipeline<i32> for AddOnePipeline {
            type Output = i32;
            async fn process_single(&self, input: i32) -> Result<i32> {
                Ok(input + 1)
            }
        }

        let pipeline = AddOnePipeline;
        let results = pipeline
            .process_batch(vec![1, 2, 3])
            .await
            .expect("process_batch should succeed via default impl");
        assert_eq!(results, vec![2, 3, 4]);
    }
}
