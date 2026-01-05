//! Dynamic Batching System for TrustformeRS Inference Server
//!
//! This module implements intelligent request batching to maximize throughput
//! while maintaining low latency for inference requests.

// Allow dead code for batching infrastructure under development
#![allow(dead_code)]

pub mod aggregator;
pub mod config;
pub mod metrics;
pub mod processor;
pub mod scheduler;

pub use aggregator::{
    AdaptiveBatchingStrategy, AggregatorStats, BatchAggregator, BatchingStrategy,
    CoalescingStrategy, ContinuousBatchingStrategy, FCFSStrategy, LoadAwareBatchingStrategy,
    PredictiveBatchingStrategy, PriorityStrategy, ProcessingResult, Request, RequestBatch,
    RequestId, SequencePackingStrategy, SequenceState,
};

pub use processor::{BatchExecutor, BatchProcessor, ProcessingError, ProcessingStats};

pub use scheduler::{
    BatchScheduler, PriorityQueue, SchedulerStats, SchedulingPolicy, TimeoutPolicy,
};

pub use metrics::{
    BatchSizeOptimizer, BatchingMetrics, LatencyTracker, MetricsCollector, ThroughputMonitor,
};

pub use config::{
    AdaptiveConfig, BatchingConfig, BatchingMode, DynamicBatchConfig, OptimizationTarget,
};

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main dynamic batching service
#[derive(Clone)]
pub struct DynamicBatchingService {
    aggregator: Arc<RwLock<BatchAggregator>>,
    processor: Arc<BatchProcessor>,
    scheduler: Arc<BatchScheduler>,
    metrics: Arc<MetricsCollector>,
    config: BatchingConfig,
}

impl DynamicBatchingService {
    /// Create a new dynamic batching service
    pub fn new(config: BatchingConfig) -> Self {
        let metrics = Arc::new(MetricsCollector::new());

        Self {
            aggregator: Arc::new(RwLock::new(BatchAggregator::new(
                config.clone(),
                metrics.clone(),
            ))),
            processor: Arc::new(BatchProcessor::new(config.clone())),
            scheduler: Arc::new(BatchScheduler::new(config.clone())),
            metrics,
            config,
        }
    }

    /// Start the batching service
    pub async fn start(&self) -> Result<()> {
        // Start background tasks
        self.start_batch_collector().await?;
        self.start_batch_timeout_checker().await?;
        self.start_metrics_collector().await?;
        self.start_optimizer().await?;

        Ok(())
    }

    /// Submit a request for batched processing
    pub async fn submit_request(&self, request: Request) -> Result<ProcessingResult> {
        // Add to aggregator
        let batch_future = self.aggregator.write().await.add_request(request).await?;

        // Wait for processing with a reasonable timeout
        let timeout_duration = self.config.max_wait_time * 10 + std::time::Duration::from_secs(5);
        let result = tokio::time::timeout(timeout_duration, batch_future)
            .await
            .map_err(|_| anyhow::anyhow!("Request processing timed out"))?
            .map_err(|e| anyhow::anyhow!("Request processing failed: {}", e))?;

        // Update metrics
        self.metrics.record_request_completion(&result);

        Ok(result)
    }

    /// Get current batching statistics
    pub async fn get_stats(&self) -> BatchingStats {
        BatchingStats {
            aggregator_stats: self.aggregator.read().await.get_stats(),
            processor_stats: self.processor.get_stats(),
            scheduler_stats: self.scheduler.get_stats(),
            metrics_summary: self.metrics.get_summary(),
        }
    }

    /// Update batching configuration dynamically
    pub async fn update_config(&self, config: BatchingConfig) -> Result<()> {
        self.aggregator.write().await.update_config(config.clone()).await?;
        self.processor.update_config(config.clone())?;
        self.scheduler.update_config(config)?;
        Ok(())
    }

    // Background tasks
    async fn start_batch_collector(&self) -> Result<()> {
        // Get the batch receiver and response channels ONCE at startup (briefly hold lock)
        // This avoids holding the RwLock during the main processing loop
        let batch_rx = self.aggregator.read().await.batch_receiver();
        let response_channels = self.aggregator.read().await.response_channels();
        let processor = self.processor.clone();

        tokio::spawn(async move {
            loop {
                // Wait for next batch using the direct receiver (no RwLock held)
                let batch = {
                    tokio::time::timeout(
                        tokio::time::Duration::from_millis(50),
                        batch_rx.lock().await.recv(),
                    )
                    .await
                    .ok()
                    .flatten()
                };

                if let Some(batch) = batch {
                    let batch_id = batch.id;
                    // Process batch
                    if let Ok(results) = processor.process_batch(batch).await {
                        // Send results back to waiting callers (using direct channel access)
                        let mut channels = response_channels.lock().await;
                        for (request_id, result) in results {
                            if let Some(tx) = channels.remove(&request_id) {
                                let _ = tx.send(result);
                            }
                        }
                        drop(channels);
                        tracing::debug!("Processed batch {}", batch_id);
                    }
                }

                tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
            }
        });

        Ok(())
    }

    async fn start_batch_timeout_checker(&self) -> Result<()> {
        let aggregator = self.aggregator.clone();
        let max_wait_time = self.config.max_wait_time;

        tokio::spawn(async move {
            // Check interval should be half of max_wait_time for responsiveness
            let check_interval = max_wait_time / 2;
            let check_interval =
                if check_interval < tokio::time::Duration::from_millis(10) {
                    tokio::time::Duration::from_millis(10)
                } else {
                    check_interval
                };

            loop {
                tokio::time::sleep(check_interval).await;

                // Briefly acquire write lock to check for timed-out requests and form batches
                let mut agg = aggregator.write().await;
                if let Err(e) = agg.try_form_batch_on_timeout().await {
                    tracing::debug!("Timeout batch formation error: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn start_metrics_collector(&self) -> Result<()> {
        let metrics = self.metrics.clone();

        tokio::spawn(async move {
            loop {
                metrics.collect_periodic_metrics().await;
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        });

        Ok(())
    }

    async fn start_optimizer(&self) -> Result<()> {
        let aggregator = self.aggregator.clone();
        let metrics = self.metrics.clone();
        let config = self.config.clone();

        if !config.enable_adaptive_batching {
            return Ok(());
        }

        tokio::spawn(async move {
            loop {
                // Get optimization suggestions
                if let Some(suggestions) = metrics.get_optimization_suggestions().await {
                    // Apply optimizations
                    let mut agg = aggregator.write().await;
                    agg.apply_optimizations(suggestions).await;
                }

                tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            }
        });

        Ok(())
    }
}

/// Batching statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchingStats {
    pub aggregator_stats: AggregatorStats,
    pub processor_stats: ProcessingStats,
    pub scheduler_stats: SchedulerStats,
    pub metrics_summary: MetricsSummary,
}

/// Metrics summary
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSummary {
    pub avg_batch_size: f32,
    pub avg_latency_ms: f32,
    pub throughput_rps: f32,
    pub queue_depth: usize,
    pub optimization_suggestions: Vec<String>,
}
