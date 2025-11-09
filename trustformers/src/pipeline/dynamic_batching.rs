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
            let mut queue = self.pending_requests.lock().unwrap();

            // Insert based on priority (higher priority first)
            let insert_pos =
                queue.iter().position(|r| r.priority < priority).unwrap_or(queue.len());

            queue.insert(insert_pos, request);
        }

        // Notify the batcher
        self.notify.notify_one();

        // Wait for response with timeout
        let timeout_duration = Duration::from_millis(self.config.max_wait_time_ms * 2);

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
            let mut running = self.is_running.lock().unwrap();
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
                let running = self.is_running.lock().unwrap();
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
                        for (request, output) in batch.into_iter().zip(outputs.into_iter()) {
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
        let mut running = self.is_running.lock().unwrap();
        *running = false;
        self.notify.notify_one();
    }

    /// Collect a batch of requests based on current batch size and timing
    async fn collect_batch(&self) -> Vec<BatchRequest<T>> {
        let current_size = *self.current_batch_size.read().unwrap();
        let mut batch = Vec::with_capacity(current_size);

        let mut queue = self.pending_requests.lock().unwrap();

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
            queue_size: self.pending_requests.lock().unwrap().len(),
        };

        let mut history = self.performance_history.lock().unwrap();
        history.push_back(metrics);

        // Keep only recent history
        while history.len() > self.config.performance_window_size {
            history.pop_front();
        }
    }

    /// Adjust batch size based on performance history
    async fn adjust_batch_size(&self) {
        let history = self.performance_history.lock().unwrap();
        if history.len() < 3 {
            return; // Need more data points
        }

        let recent_metrics: Vec<_> = history.iter().rev().take(3).collect();
        let avg_latency =
            recent_metrics.iter().map(|m| m.latency_ms).sum::<u64>() / recent_metrics.len() as u64;
        let avg_throughput = recent_metrics.iter().map(|m| m.throughput_rps).sum::<f64>()
            / recent_metrics.len() as f64;

        let mut current_size = self.current_batch_size.write().unwrap();
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
        let history = self.performance_history.lock().unwrap();
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
            current_batch_size: *self.current_batch_size.read().unwrap(),
            avg_latency_ms: avg_latency,
            avg_throughput_rps: avg_throughput,
            avg_batch_size,
            queue_length: self.pending_requests.lock().unwrap().len(),
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

    #[tokio::test]
    async fn test_dynamic_batching_basic() {
        let config = DynamicBatchingConfig {
            initial_batch_size: 2,
            max_wait_time_ms: 10,
            ..Default::default()
        };

        let batcher = DynamicBatcher::new(config);

        // Mock processor that doubles the input
        let processor = |inputs: Vec<i32>| async move {
            sleep(Duration::from_millis(1)).await;
            Ok(inputs.into_iter().map(|x| x * 2).collect())
        };

        // Start batcher in background
        let batcher_clone = Arc::new(batcher);
        let batcher_for_task = batcher_clone.clone();

        let process_task = tokio::spawn(async move { batcher_for_task.start(processor).await });

        // Send some requests
        let results = futures::future::join_all(vec![
            batcher_clone.add_request(1, RequestPriority::Normal),
            batcher_clone.add_request(2, RequestPriority::Normal),
            batcher_clone.add_request(3, RequestPriority::High),
        ])
        .await;

        // Stop the batcher
        batcher_clone.stop();
        let _ = process_task.await;

        // Check results
        assert_eq!(results.len(), 3);
        for result in results {
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let config = DynamicBatchingConfig {
            initial_batch_size: 3,
            max_wait_time_ms: 50,
            ..Default::default()
        };

        let batcher = DynamicBatcher::new(config);

        // Add requests with different priorities
        let _low = batcher.add_request(1, RequestPriority::Low);
        let _normal = batcher.add_request(2, RequestPriority::Normal);
        let _high = batcher.add_request(3, RequestPriority::High);
        let _critical = batcher.add_request(4, RequestPriority::Critical);

        // Check the order in queue
        let queue = batcher.pending_requests.lock().unwrap();
        let priorities: Vec<_> = queue.iter().map(|r| r.priority).collect();

        // Should be ordered by priority (highest first)
        assert!(priorities.windows(2).all(|w| w[0] >= w[1]));
    }
}
