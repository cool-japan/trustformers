//! Batch Processor for Dynamic Batching

use crate::batching::{
    aggregator::{ProcessingOutput, ProcessingResult, RequestBatch, RequestId},
    config::BatchingConfig,
};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, Semaphore};
use trustformers_core::{tensor::Tensor, traits::Model};

/// Batch processor that executes batched inference
pub struct BatchProcessor {
    config: BatchingConfig,
    executor: Arc<dyn BatchExecutor>,
    processing_semaphore: Arc<Semaphore>,
    stats: Arc<RwLock<ProcessingStats>>,
}

impl BatchProcessor {
    pub fn new(config: BatchingConfig) -> Self {
        let max_concurrent = config.max_batch_size / config.min_batch_size;

        Self {
            config: config.clone(),
            executor: Arc::new(DefaultBatchExecutor::new()),
            processing_semaphore: Arc::new(Semaphore::new(max_concurrent)),
            stats: Arc::new(RwLock::new(ProcessingStats::default())),
        }
    }

    /// Process a batch of requests
    pub async fn process_batch(
        &self,
        batch: RequestBatch,
    ) -> Result<HashMap<RequestId, ProcessingResult>> {
        let _permit = self.processing_semaphore.acquire().await?;
        let start_time = Instant::now();

        // Update stats
        self.stats.write().await.record_batch_start(&batch);

        // Execute batch
        let results = self.executor.execute_batch(&batch, &self.config).await?;

        // Update stats
        let processing_time = start_time.elapsed();
        self.stats.write().await.record_batch_completion(&batch, processing_time);

        // Convert to processing results
        let mut output = HashMap::new();
        for (request_id, result) in results {
            output.insert(
                request_id.clone(),
                ProcessingResult {
                    request_id,
                    output: result,
                    latency_ms: processing_time.as_millis() as u64,
                    batch_id: batch.id,
                },
            );
        }

        Ok(output)
    }

    /// Update processor configuration
    pub fn update_config(&self, _config: BatchingConfig) -> Result<()> {
        // In practice, would update internal config
        Ok(())
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> ProcessingStats {
        // Would need async access
        ProcessingStats::default()
    }
}

/// Batch executor trait
#[async_trait::async_trait]
pub trait BatchExecutor: Send + Sync {
    /// Execute a batch and return results
    async fn execute_batch(
        &self,
        batch: &RequestBatch,
        config: &BatchingConfig,
    ) -> Result<HashMap<RequestId, ProcessingOutput>>;

    /// Check if executor can handle the batch
    async fn can_handle_batch(&self, batch: &RequestBatch) -> bool;

    /// Get executor capabilities
    fn capabilities(&self) -> ExecutorCapabilities;
}

/// Common model trait alias for batching
/// We use a generic approach instead of trait objects to avoid Config dyn compatibility issues
pub trait BatchModel: Send + Sync {
    /// Perform forward pass
    fn forward(&self, input: Tensor) -> Result<Tensor>;
}

/// Default batch executor implementation
pub struct DefaultBatchExecutor {
    model: Option<Arc<dyn BatchModel>>,
}

impl Default for DefaultBatchExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultBatchExecutor {
    pub fn new() -> Self {
        Self { model: None }
    }

    pub fn with_model(model: Arc<dyn BatchModel>) -> Self {
        Self { model: Some(model) }
    }
}

#[async_trait::async_trait]
impl BatchExecutor for DefaultBatchExecutor {
    async fn execute_batch(
        &self,
        batch: &RequestBatch,
        _config: &BatchingConfig,
    ) -> Result<HashMap<RequestId, ProcessingOutput>> {
        // This is a simplified implementation
        // In practice, would:
        // 1. Convert requests to tensors with proper padding
        // 2. Run model inference
        // 3. Post-process outputs
        // 4. Map back to individual requests

        let mut results = HashMap::new();

        // Simulate batch processing
        for request in &batch.requests {
            let output = match &request.input {
                crate::batching::aggregator::RequestInput::Text { text, .. } => {
                    // Simulate text generation
                    ProcessingOutput::Text(format!("Processed: {}", text))
                },
                crate::batching::aggregator::RequestInput::TokenIds { ids, .. } => {
                    // Simulate token processing
                    ProcessingOutput::Tokens(ids.iter().map(|&id| id + 1).collect())
                },
                crate::batching::aggregator::RequestInput::Image { .. } => {
                    // Simulate image classification
                    ProcessingOutput::Classification(vec![
                        ("cat".to_string(), 0.8),
                        ("dog".to_string(), 0.2),
                    ])
                },
                crate::batching::aggregator::RequestInput::Multimodal { .. } => {
                    ProcessingOutput::Text("Multimodal output".to_string())
                },
            };

            results.insert(request.id.clone(), output);
        }

        Ok(results)
    }

    async fn can_handle_batch(&self, _batch: &RequestBatch) -> bool {
        // Check if we can handle this batch size and type
        true
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 128,
            supported_input_types: vec![
                InputType::Text,
                InputType::Tokens,
                InputType::Image,
                InputType::Multimodal,
            ],
            supports_dynamic_batching: true,
            supports_continuous_batching: false,
        }
    }
}

/// Model-specific batch executor
pub struct ModelBatchExecutor<M: Model> {
    model: Arc<M>,
    tokenizer: Option<Arc<dyn Tokenizer>>,
}

impl<M: Model> ModelBatchExecutor<M> {
    pub fn new(model: Arc<M>) -> Self {
        Self {
            model,
            tokenizer: None,
        }
    }

    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }
}

/// Tokenizer trait (simplified)
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
    fn pad(&self, sequences: Vec<Vec<u32>>, max_length: usize) -> (Tensor, Tensor);
}

/// Processing error types
#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {
    #[error("Batch too large: {0} > {1}")]
    BatchTooLarge(usize, usize),

    #[error("Unsupported input type")]
    UnsupportedInputType,

    #[error("Model execution failed: {0}")]
    ModelError(String),

    #[error("Out of memory")]
    OutOfMemory,

    #[error("Timeout")]
    Timeout,
}

/// Executor capabilities
#[derive(Debug, Clone)]
pub struct ExecutorCapabilities {
    pub max_batch_size: usize,
    pub supported_input_types: Vec<InputType>,
    pub supports_dynamic_batching: bool,
    pub supports_continuous_batching: bool,
}

/// Supported input types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputType {
    Text,
    Tokens,
    Image,
    Audio,
    Multimodal,
}

/// Processing statistics
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ProcessingStats {
    pub total_batches: usize,
    pub total_requests: usize,
    pub avg_batch_size: f32,
    pub avg_processing_time_ms: f32,
    pub success_rate: f32,
    pub current_processing: usize,
}

impl ProcessingStats {
    fn record_batch_start(&mut self, _batch: &RequestBatch) {
        self.current_processing += 1;
    }

    fn record_batch_completion(&mut self, batch: &RequestBatch, duration: std::time::Duration) {
        self.total_batches += 1;
        self.total_requests += batch.requests.len();
        self.current_processing -= 1;

        // Update averages
        let batch_size = batch.requests.len() as f32;
        self.avg_batch_size = (self.avg_batch_size * (self.total_batches - 1) as f32 + batch_size)
            / self.total_batches as f32;

        let processing_ms = duration.as_millis() as f32;
        self.avg_processing_time_ms =
            (self.avg_processing_time_ms * (self.total_batches - 1) as f32 + processing_ms)
                / self.total_batches as f32;
    }
}

/// Continuous batching support for LLMs
pub struct ContinuousBatchingExecutor {
    max_sequences: usize,
    active_sequences: Arc<RwLock<HashMap<RequestId, SequenceState>>>,
}

/// Sequence state for continuous batching
#[derive(Debug, Clone)]
struct SequenceState {
    tokens: Vec<u32>,
    position: usize,
    kv_cache_slot: Option<usize>,
    finished: bool,
}

impl ContinuousBatchingExecutor {
    pub fn new(max_sequences: usize) -> Self {
        Self {
            max_sequences,
            active_sequences: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add new sequences to active batch
    pub async fn add_sequences(&self, requests: Vec<(RequestId, Vec<u32>)>) -> Result<()> {
        let mut sequences = self.active_sequences.write().await;

        for (id, tokens) in requests {
            if sequences.len() >= self.max_sequences {
                return Err(anyhow!("Maximum sequences reached"));
            }

            sequences.insert(
                id,
                SequenceState {
                    tokens,
                    position: 0,
                    kv_cache_slot: None,
                    finished: false,
                },
            );
        }

        Ok(())
    }

    /// Process one step of generation
    pub async fn process_step(&self) -> Result<HashMap<RequestId, u32>> {
        // In practice, would:
        // 1. Gather active sequences
        // 2. Create batch with current tokens
        // 3. Run model forward pass
        // 4. Update KV cache
        // 5. Return next tokens

        Ok(HashMap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_default_executor() {
        let executor = DefaultBatchExecutor::new();
        let capabilities = executor.capabilities();

        assert_eq!(capabilities.max_batch_size, 128);
        assert!(capabilities.supports_dynamic_batching);
    }

    #[test]
    fn test_processing_stats() {
        let mut stats = ProcessingStats::default();

        let batch = RequestBatch {
            id: uuid::Uuid::new_v4(),
            requests: vec![],
            created_at: Instant::now(),
            total_memory: 0,
            max_sequence_length: 0,
            priority: crate::batching::config::Priority::Normal,
        };

        stats.record_batch_start(&batch);
        assert_eq!(stats.current_processing, 1);

        stats.record_batch_completion(&batch, std::time::Duration::from_millis(100));
        assert_eq!(stats.current_processing, 0);
        assert_eq!(stats.total_batches, 1);
    }
}
