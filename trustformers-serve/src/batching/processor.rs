//! Batch Processor for Dynamic Batching
//!
//! The processor owns the [`BatchExecutor`] that turns a [`RequestBatch`] into
//! real model outputs. No executor in this module fabricates inference results:
//! when no model is wired in, every request in the batch is answered with an
//! explicit [`ProcessingOutput::Error`] so callers see the configuration problem
//! instead of a plausible-looking echo.

use crate::batching::{
    aggregator::{ProcessingOutput, ProcessingResult, RequestBatch, RequestId, RequestInput},
    config::BatchingConfig,
};
use anyhow::{anyhow, Result};
use parking_lot::RwLock as SyncRwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, Semaphore};
use trustformers_core::tensor::Tensor;

/// Message returned for every request when the batching stack has no model.
pub const NO_MODEL_CONFIGURED: &str =
    "no model configured for the batch executor: install one with \
     BatchProcessor::with_executor / DynamicBatchingService::with_executor";

/// Decode budget used when a request does not carry its own `max_length`.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 32;

/// Compute the number of batches that may be executed concurrently.
fn max_concurrent_for(config: &BatchingConfig) -> usize {
    (config.max_batch_size / config.min_batch_size.max(1)).max(1)
}

/// Batch processor that executes batched inference
pub struct BatchProcessor {
    /// Live configuration; `update_config` genuinely replaces this value.
    config: SyncRwLock<BatchingConfig>,
    executor: Arc<dyn BatchExecutor>,
    processing_semaphore: Arc<Semaphore>,
    /// Permits currently granted to `processing_semaphore`, tracked so that a
    /// configuration update can grow or shrink the concurrency window.
    granted_permits: AtomicUsize,
    /// Live counters, behind a synchronous lock so `get_stats` stays callable
    /// from non-async code while still returning the real numbers.
    stats: Arc<SyncRwLock<ProcessingStats>>,
}

impl BatchProcessor {
    /// Create a processor with the default (model-less) executor.
    ///
    /// The resulting processor answers every request with
    /// [`ProcessingOutput::Error`]; use [`BatchProcessor::with_executor`] to
    /// install a real model-backed executor.
    pub fn new(config: BatchingConfig) -> Self {
        Self::with_executor(config, Arc::new(DefaultBatchExecutor::new()))
    }

    /// Create a processor backed by a caller-supplied executor.
    pub fn with_executor(config: BatchingConfig, executor: Arc<dyn BatchExecutor>) -> Self {
        let max_concurrent = max_concurrent_for(&config);

        Self {
            config: SyncRwLock::new(config),
            executor,
            processing_semaphore: Arc::new(Semaphore::new(max_concurrent)),
            granted_permits: AtomicUsize::new(max_concurrent),
            stats: Arc::new(SyncRwLock::new(ProcessingStats::default())),
        }
    }

    /// The executor currently installed on this processor.
    pub fn executor(&self) -> &Arc<dyn BatchExecutor> {
        &self.executor
    }

    /// Snapshot of the active configuration.
    pub fn config(&self) -> BatchingConfig {
        self.config.read().clone()
    }

    /// Process a batch of requests
    pub async fn process_batch(
        &self,
        batch: RequestBatch,
    ) -> Result<HashMap<RequestId, ProcessingResult>> {
        let _permit = self.processing_semaphore.acquire().await?;
        let start_time = Instant::now();

        // Update stats
        self.stats.write().record_batch_start(&batch);

        // Execute batch against the active configuration snapshot.
        let config = self.config();
        let results = match self.executor.execute_batch(&batch, &config).await {
            Ok(results) => results,
            Err(e) => {
                // Record the failed batch so success_rate reflects reality, then
                // propagate. `process_batch` callers must still answer every
                // pending request; see `DynamicBatchingService::start_batch_collector`.
                let processing_time = start_time.elapsed();
                self.stats.write().record_batch_failure(&batch, processing_time);
                return Err(e);
            },
        };

        // Update stats
        let processing_time = start_time.elapsed();
        self.stats.write().record_batch_completion(&batch, processing_time);

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

    /// Update processor configuration.
    ///
    /// The configuration is stored and used by every subsequent `process_batch`
    /// call, and the concurrency window is resized to match the new
    /// `max_batch_size` / `min_batch_size` ratio.
    pub fn update_config(&self, config: BatchingConfig) -> Result<()> {
        let new_concurrency = max_concurrent_for(&config);
        *self.config.write() = config;

        let previous = self.granted_permits.swap(new_concurrency, AtomicOrdering::SeqCst);
        match new_concurrency.cmp(&previous) {
            std::cmp::Ordering::Greater => {
                self.processing_semaphore.add_permits(new_concurrency - previous);
            },
            std::cmp::Ordering::Less => {
                // `forget_permits` returns how many it could actually remove; the
                // remainder is reclaimed as in-flight batches release their permits.
                let removed = self.processing_semaphore.forget_permits(previous - new_concurrency);
                if removed < previous - new_concurrency {
                    tracing::debug!(
                        "batch concurrency shrink is partially deferred: {} of {} permits removed",
                        removed,
                        previous - new_concurrency
                    );
                }
            },
            std::cmp::Ordering::Equal => {},
        }

        Ok(())
    }

    /// Get processing statistics.
    ///
    /// Returns the real accumulated counters maintained by `process_batch`.
    pub fn get_stats(&self) -> ProcessingStats {
        self.stats.read().clone()
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

    /// Whether this executor is backed by a real model.
    ///
    /// Serving-state probes (HTTP `/health/detailed`, gRPC `HealthCheck`) use
    /// this to report `NOT_SERVING` instead of pretending to serve.
    fn has_model(&self) -> bool {
        true
    }
}

/// A model that can run a real forward pass over a padded batch of token ids.
///
/// Input is an `[batch, seq_len]` tensor of token ids stored as `f32`; output is
/// an `[batch, seq_len, vocab]` logits tensor.
pub trait BatchModel: Send + Sync {
    /// Perform forward pass
    fn forward(&self, input: Tensor) -> Result<Tensor>;

    /// Vocabulary size of the model's output distribution.
    fn vocab_size(&self) -> usize;

    /// Token id used for right padding of ragged batches.
    fn pad_token_id(&self) -> u32 {
        0
    }

    /// Token id that terminates generation, when the model defines one.
    fn eos_token_id(&self) -> Option<u32> {
        None
    }

    /// Maximum number of positions the model can attend over, when it has a
    /// fixed context window.
    fn max_context_tokens(&self) -> Option<usize> {
        None
    }
}

/// A model that can turn a token sequence into a single dense representation.
///
/// Implementations must pool the model's own hidden states. Returning a zero
/// vector, a hash of the input, or any other stand-in is never acceptable: the
/// caller cannot tell such a vector from a real embedding. A model that has no
/// hidden states to pool must not implement this trait at all, so the serving
/// layer can answer "no embedding model available" honestly.
pub trait EmbeddingModel: Send + Sync {
    /// Pool the model's hidden states for `ids` into one vector.
    ///
    /// # Errors
    ///
    /// Fails when `ids` is empty, when the forward pass fails, or when the model
    /// returns a hidden-state layout this implementation cannot pool.
    fn embed_tokens(&self, ids: &[u32]) -> Result<Vec<f32>>;

    /// Dimensionality of the vectors [`Self::embed_tokens`] produces.
    fn embedding_dim(&self) -> usize;
}

/// Message prefix used when a prompt does not fit the model's context window.
pub const CONTEXT_WINDOW_EXCEEDED: &str = "context window exceeded";

/// Tokenizer trait used to bridge text requests to a [`BatchModel`].
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
}

/// Default batch executor.
///
/// Without a model it is deliberately inert: `execute_batch` answers every
/// request with [`ProcessingOutput::Error`] rather than echoing the input.
pub struct DefaultBatchExecutor {
    model: Option<Arc<dyn BatchModel>>,
    tokenizer: Option<Arc<dyn Tokenizer>>,
}

impl Default for DefaultBatchExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultBatchExecutor {
    pub fn new() -> Self {
        Self {
            model: None,
            tokenizer: None,
        }
    }

    pub fn with_model(model: Arc<dyn BatchModel>) -> Self {
        Self {
            model: Some(model),
            tokenizer: None,
        }
    }

    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }
}

#[async_trait::async_trait]
impl BatchExecutor for DefaultBatchExecutor {
    async fn execute_batch(
        &self,
        batch: &RequestBatch,
        config: &BatchingConfig,
    ) -> Result<HashMap<RequestId, ProcessingOutput>> {
        match &self.model {
            Some(model) => {
                let mut inner = ModelBatchExecutor::new(Arc::clone(model));
                if let Some(tokenizer) = &self.tokenizer {
                    inner = inner.with_tokenizer(Arc::clone(tokenizer));
                }
                inner.execute_batch(batch, config).await
            },
            None => {
                let mut results = HashMap::new();
                for request in &batch.requests {
                    results.insert(
                        request.id.clone(),
                        ProcessingOutput::Error(NO_MODEL_CONFIGURED.to_string()),
                    );
                }
                Ok(results)
            },
        }
    }

    async fn can_handle_batch(&self, batch: &RequestBatch) -> bool {
        self.model.is_some() && batch.requests.len() <= self.capabilities().max_batch_size
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 128,
            supported_input_types: vec![InputType::Text, InputType::Tokens],
            supports_dynamic_batching: true,
            supports_continuous_batching: false,
        }
    }

    fn has_model(&self) -> bool {
        self.model.is_some()
    }
}

/// Model-backed batch executor.
///
/// Pads the batch to a rectangular `[batch, seq]` tensor, runs the real forward
/// pass, and reads the logits at each sequence's own final position. Because the
/// backing decoders are causal, right padding cannot influence the position that
/// is read, so unpadding is exact rather than approximate.
pub struct ModelBatchExecutor {
    model: Arc<dyn BatchModel>,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    /// Decode budget for requests that do not carry their own `max_length`.
    default_max_new_tokens: usize,
}

impl ModelBatchExecutor {
    pub fn new(model: Arc<dyn BatchModel>) -> Self {
        Self {
            model,
            tokenizer: None,
            default_max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }

    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Override the decode budget used when a request has no `max_length`.
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.default_max_new_tokens = max_new_tokens.max(1);
        self
    }

    /// Run `max_new_tokens` greedy decoding steps over a ragged batch of prompts.
    ///
    /// Returns, for each input row, the newly generated token ids (never the prompt).
    fn generate(&self, prompts: &[Vec<u32>], max_new_tokens: usize) -> Result<Vec<Vec<u32>>> {
        let vocab = self.model.vocab_size();
        if vocab == 0 {
            return Err(anyhow!("model reports a zero-sized vocabulary"));
        }
        let pad_id = self.model.pad_token_id();
        let eos_id = self.model.eos_token_id();

        // A prompt that cannot fit the model's context window is refused with a
        // clear message rather than being silently truncated or handed to the
        // model to fail on an out-of-range position embedding.
        if let Some(context) = self.model.max_context_tokens() {
            if let Some(longest) = prompts.iter().map(|p| p.len()).max() {
                if longest + max_new_tokens > context {
                    return Err(anyhow!(
                        "{}: a prompt of {} token(s) plus {} new token(s) exceeds the model's \
                         context window of {} token(s)",
                        CONTEXT_WINDOW_EXCEEDED,
                        longest,
                        max_new_tokens,
                        context
                    ));
                }
            }
        }

        let mut sequences: Vec<Vec<u32>> = prompts.to_vec();
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
        let mut finished: Vec<bool> = prompts.iter().map(|p| p.is_empty()).collect();

        for _ in 0..max_new_tokens {
            if finished.iter().all(|f| *f) {
                break;
            }

            let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
            if max_len == 0 {
                break;
            }

            let mut flat = Vec::with_capacity(sequences.len() * max_len);
            for seq in &sequences {
                flat.extend(seq.iter().map(|&id| id as f32));
                flat.extend(std::iter::repeat_n(pad_id as f32, max_len - seq.len()));
            }

            let input = Tensor::from_vec(flat, &[sequences.len(), max_len])
                .map_err(|e| anyhow!("failed to build batched input tensor: {}", e))?;
            let logits = self
                .model
                .forward(input)
                .map_err(|e| anyhow!("model forward pass failed: {}", e))?;

            let shape = logits.shape();
            let values = logits
                .data()
                .map_err(|e| anyhow!("failed to read logits from model output: {}", e))?;
            let expected = sequences.len() * max_len * vocab;
            if values.len() != expected {
                return Err(anyhow!(
                    "model returned {} logits for shape {:?}; expected {} ([batch, seq, vocab] = [{}, {}, {}])",
                    values.len(),
                    shape,
                    expected,
                    sequences.len(),
                    max_len,
                    vocab
                ));
            }

            for (row, seq) in sequences.iter_mut().enumerate() {
                if finished[row] {
                    continue;
                }
                let last = seq.len() - 1;
                let offset = (row * max_len + last) * vocab;
                let slice = &values[offset..offset + vocab];
                let next = argmax(slice);
                seq.push(next);
                generated[row].push(next);
                if Some(next) == eos_id {
                    finished[row] = true;
                }
            }
        }

        Ok(generated)
    }
}

/// Index of the largest value in `values` (first index wins on ties).
fn argmax(values: &[f32]) -> u32 {
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    best_index as u32
}

#[async_trait::async_trait]
impl BatchExecutor for ModelBatchExecutor {
    async fn execute_batch(
        &self,
        batch: &RequestBatch,
        config: &BatchingConfig,
    ) -> Result<HashMap<RequestId, ProcessingOutput>> {
        let mut results: HashMap<RequestId, ProcessingOutput> = HashMap::new();

        // Rows that can actually be fed to the model, and how to render them back.
        enum Render {
            Text,
            Tokens,
        }
        let mut rows: Vec<Vec<u32>> = Vec::new();
        let mut row_ids: Vec<RequestId> = Vec::new();
        let mut row_render: Vec<Render> = Vec::new();
        let mut row_budget: Vec<usize> = Vec::new();

        let _ = config;
        let default_budget = self.default_max_new_tokens;

        for request in &batch.requests {
            match &request.input {
                RequestInput::Text { text, max_length } => match &self.tokenizer {
                    Some(tokenizer) => {
                        let ids = tokenizer.encode(text);
                        if ids.is_empty() {
                            results.insert(
                                request.id.clone(),
                                ProcessingOutput::Error(
                                    "prompt tokenized to zero tokens".to_string(),
                                ),
                            );
                            continue;
                        }
                        rows.push(ids);
                        row_ids.push(request.id.clone());
                        row_render.push(Render::Text);
                        row_budget.push(max_length.unwrap_or(default_budget).max(1));
                    },
                    None => {
                        results.insert(
                            request.id.clone(),
                            ProcessingOutput::Error(
                                "text input requires a tokenizer; this executor was built without one"
                                    .to_string(),
                            ),
                        );
                    },
                },
                RequestInput::TokenIds { ids, .. } => {
                    if ids.is_empty() {
                        results.insert(
                            request.id.clone(),
                            ProcessingOutput::Error("empty token id sequence".to_string()),
                        );
                        continue;
                    }
                    rows.push(ids.clone());
                    row_ids.push(request.id.clone());
                    row_render.push(Render::Tokens);
                    row_budget.push(default_budget);
                },
                RequestInput::Image { .. } => {
                    results.insert(
                        request.id.clone(),
                        ProcessingOutput::Error(
                            "image inputs are not supported by the text batch executor".to_string(),
                        ),
                    );
                },
                RequestInput::Multimodal { .. } => {
                    results.insert(
                        request.id.clone(),
                        ProcessingOutput::Error(
                            "multimodal inputs are not supported by the text batch executor"
                                .to_string(),
                        ),
                    );
                },
            }
        }

        if rows.is_empty() {
            return Ok(results);
        }

        // A single decode budget is used for the whole batch: the batch advances
        // in lock-step, and per-row budgets are applied when slicing the output.
        let steps = row_budget.iter().copied().max().unwrap_or(1);
        let generated = match self.generate(&rows, steps) {
            Ok(generated) => generated,
            Err(e) => {
                // Answer every row with the real reason instead of failing the
                // whole batch and leaving callers to time out.
                let message = e.to_string();
                for request_id in row_ids {
                    results.insert(request_id, ProcessingOutput::Error(message.clone()));
                }
                return Ok(results);
            },
        };

        for (index, request_id) in row_ids.into_iter().enumerate() {
            let budget = row_budget[index];
            let tokens: Vec<u32> =
                generated[index].iter().copied().take(budget).collect::<Vec<_>>();

            let output = match row_render[index] {
                Render::Tokens => ProcessingOutput::Tokens(tokens),
                Render::Text => match &self.tokenizer {
                    Some(tokenizer) => ProcessingOutput::Text(tokenizer.decode(&tokens)),
                    None => ProcessingOutput::Error(
                        "text input requires a tokenizer; this executor was built without one"
                            .to_string(),
                    ),
                },
            };
            results.insert(request_id, output);
        }

        Ok(results)
    }

    async fn can_handle_batch(&self, batch: &RequestBatch) -> bool {
        batch.requests.len() <= self.capabilities().max_batch_size
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 128,
            supported_input_types: vec![InputType::Text, InputType::Tokens],
            supports_dynamic_batching: true,
            supports_continuous_batching: false,
        }
    }

    fn has_model(&self) -> bool {
        true
    }
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
    /// Batches whose executor returned an error.
    pub failed_batches: usize,
}

impl ProcessingStats {
    fn record_batch_start(&mut self, _batch: &RequestBatch) {
        self.current_processing += 1;
    }

    fn record_batch_completion(&mut self, batch: &RequestBatch, duration: std::time::Duration) {
        self.total_batches += 1;
        self.total_requests += batch.requests.len();
        self.current_processing = self.current_processing.saturating_sub(1);

        // Update averages
        let batch_size = batch.requests.len() as f32;
        self.avg_batch_size = (self.avg_batch_size * (self.total_batches - 1) as f32 + batch_size)
            / self.total_batches as f32;

        let processing_ms = duration.as_millis() as f32;
        self.avg_processing_time_ms =
            (self.avg_processing_time_ms * (self.total_batches - 1) as f32 + processing_ms)
                / self.total_batches as f32;

        self.recompute_success_rate();
    }

    fn record_batch_failure(&mut self, batch: &RequestBatch, duration: std::time::Duration) {
        self.total_batches += 1;
        self.total_requests += batch.requests.len();
        self.failed_batches += 1;
        self.current_processing = self.current_processing.saturating_sub(1);

        let batch_size = batch.requests.len() as f32;
        self.avg_batch_size = (self.avg_batch_size * (self.total_batches - 1) as f32 + batch_size)
            / self.total_batches as f32;

        let processing_ms = duration.as_millis() as f32;
        self.avg_processing_time_ms =
            (self.avg_processing_time_ms * (self.total_batches - 1) as f32 + processing_ms)
                / self.total_batches as f32;

        self.recompute_success_rate();
    }

    fn recompute_success_rate(&mut self) {
        if self.total_batches == 0 {
            self.success_rate = 0.0;
        } else {
            let succeeded = self.total_batches - self.failed_batches;
            self.success_rate = succeeded as f32 / self.total_batches as f32;
        }
    }
}

/// Continuous batching support for LLMs.
///
/// This type only tracks sequence admission. Token generation lives in
/// [`crate::continuous_batching`], which owns the model and the KV cache;
/// `process_step` therefore reports an explicit error rather than silently
/// returning "no progress" forever.
pub struct ContinuousBatchingExecutor {
    max_sequences: usize,
    active_sequences: Arc<RwLock<HashMap<RequestId, SequenceState>>>,
}

/// Sequence state for continuous batching
#[derive(Debug, Clone)]
pub struct SequenceState {
    pub tokens: Vec<u32>,
    pub position: usize,
    pub kv_cache_slot: Option<usize>,
    pub finished: bool,
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

            let position = tokens.len();
            sequences.insert(
                id,
                SequenceState {
                    tokens,
                    position,
                    kv_cache_slot: None,
                    finished: false,
                },
            );
        }

        Ok(())
    }

    /// Snapshot of the currently admitted sequences.
    pub async fn active_sequences(&self) -> HashMap<RequestId, SequenceState> {
        self.active_sequences.read().await.clone()
    }

    /// Process one step of generation.
    ///
    /// # Errors
    ///
    /// Always returns an error: this admission-only type owns neither a model nor
    /// a KV cache, so it cannot produce a token. Use
    /// [`crate::continuous_batching::ContinuousBatchingScheduler`] for real
    /// continuous decoding.
    pub async fn process_step(&self) -> Result<HashMap<RequestId, u32>> {
        Err(anyhow!(
            "ContinuousBatchingExecutor tracks sequence admission only and cannot generate \
             tokens; drive decoding through crate::continuous_batching, which owns the model \
             and the KV cache"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batching::aggregator::Request;
    use crate::batching::config::Priority;

    /// A genuinely computed stand-in model: it runs a real (if tiny) linear map
    /// over the input ids, so its outputs depend on the actual input rather than
    /// on a canned string.
    struct CountingModel {
        vocab: usize,
    }

    impl BatchModel for CountingModel {
        fn forward(&self, input: Tensor) -> Result<Tensor> {
            let shape = input.shape();
            let (batch, seq) = (shape[0], shape[1]);
            let ids = input.data()?;
            let mut logits = vec![0.0f32; batch * seq * self.vocab];
            for row in 0..batch {
                for position in 0..seq {
                    let id = ids[row * seq + position] as usize;
                    // Argmax at each position is (id + 1) mod vocab.
                    let target = (id + 1) % self.vocab;
                    logits[(row * seq + position) * self.vocab + target] = 10.0;
                }
            }
            Ok(Tensor::from_vec(logits, &[batch, seq, self.vocab])?)
        }

        fn vocab_size(&self) -> usize {
            self.vocab
        }
    }

    fn token_request(ids: Vec<u32>) -> Request {
        Request {
            id: RequestId::new(),
            input: RequestInput::TokenIds {
                ids,
                attention_mask: None,
            },
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            deadline: None,
            metadata: HashMap::new(),
        }
    }

    fn batch_of(requests: Vec<Request>) -> RequestBatch {
        RequestBatch {
            id: uuid::Uuid::new_v4(),
            requests,
            created_at: Instant::now(),
            total_memory: 0,
            max_sequence_length: 0,
            priority: Priority::Normal,
        }
    }

    #[tokio::test]
    async fn test_default_executor() {
        let executor = DefaultBatchExecutor::new();
        let capabilities = executor.capabilities();

        assert_eq!(capabilities.max_batch_size, 128);
        assert!(capabilities.supports_dynamic_batching);
    }

    /// Regression: the model-less executor must never echo its input back as a
    /// successful inference result.
    #[tokio::test]
    async fn model_less_executor_reports_error_not_echo() {
        let executor = DefaultBatchExecutor::new();
        assert!(!executor.has_model());
        let request = Request {
            id: RequestId::new(),
            input: RequestInput::Text {
                text: "Hello, world!".to_string(),
                max_length: Some(8),
            },
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            deadline: None,
            metadata: HashMap::new(),
        };
        let id = request.id.clone();
        let batch = batch_of(vec![request]);

        let results = executor
            .execute_batch(&batch, &BatchingConfig::default())
            .await
            .expect("model-less executor must answer, not fail");

        match results.get(&id) {
            Some(ProcessingOutput::Error(message)) => {
                assert!(
                    message.contains("no model configured"),
                    "unexpected message: {message}"
                );
            },
            other => panic!("expected an explicit error output, got {other:?}"),
        }
        assert!(!executor.can_handle_batch(&batch).await);
    }

    /// Regression: a model-backed executor must return tokens derived from the
    /// real forward pass, not `id + 1` echoes of the request.
    #[tokio::test]
    async fn model_backed_executor_runs_real_forward() {
        let model = Arc::new(CountingModel { vocab: 16 });
        let executor = ModelBatchExecutor::new(model).with_max_new_tokens(4);

        let request = token_request(vec![3, 4, 5]);
        let id = request.id.clone();
        let batch = batch_of(vec![request]);

        let results = executor
            .execute_batch(&batch, &BatchingConfig::default())
            .await
            .expect("real executor must succeed");

        match results.get(&id) {
            Some(ProcessingOutput::Tokens(tokens)) => {
                // Greedy decode from [3,4,5]: 6, then 7, then 8, then 9.
                assert_eq!(tokens, &vec![6, 7, 8, 9]);
            },
            other => panic!("expected token output, got {other:?}"),
        }
    }

    /// Regression: ragged batches must not leak padding into the read-out
    /// position; the short row must continue from *its own* last token.
    #[tokio::test]
    async fn ragged_batch_unpads_exactly() {
        let model = Arc::new(CountingModel { vocab: 32 });
        let executor = ModelBatchExecutor::new(model).with_max_new_tokens(2);

        let short = token_request(vec![1]);
        let long = token_request(vec![10, 11, 12, 13]);
        let (short_id, long_id) = (short.id.clone(), long.id.clone());
        let batch = batch_of(vec![short, long]);

        let results = executor
            .execute_batch(&batch, &BatchingConfig::default())
            .await
            .expect("ragged batch must succeed");

        match results.get(&short_id) {
            Some(ProcessingOutput::Tokens(tokens)) => assert_eq!(tokens, &vec![2, 3]),
            other => panic!("expected token output, got {other:?}"),
        }
        match results.get(&long_id) {
            Some(ProcessingOutput::Tokens(tokens)) => assert_eq!(tokens, &vec![14, 15]),
            other => panic!("expected token output, got {other:?}"),
        }
    }

    /// Regression: `get_stats` must return the counters `process_batch` records,
    /// not `ProcessingStats::default()`.
    #[tokio::test]
    async fn processor_get_stats_returns_real_counters() {
        let processor = BatchProcessor::new(BatchingConfig::default());
        assert_eq!(processor.get_stats().total_batches, 0);

        let batch = batch_of(vec![token_request(vec![1, 2]), token_request(vec![3])]);
        processor.process_batch(batch).await.expect("model-less batch still resolves");

        let stats = processor.get_stats();
        assert_eq!(stats.total_batches, 1);
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.current_processing, 0);
        assert!(stats.avg_batch_size > 1.9);
    }

    /// Regression: `update_config` must actually store the new configuration.
    #[test]
    fn processor_update_config_is_applied() {
        let processor = BatchProcessor::new(BatchingConfig::default());
        let original = processor.config().max_batch_size;

        let mut updated = BatchingConfig::default();
        updated.max_batch_size = original + 17;
        processor.update_config(updated).expect("update must succeed");

        assert_eq!(processor.config().max_batch_size, original + 17);
    }

    /// Regression: an injected executor must actually be used.
    #[tokio::test]
    async fn processor_with_executor_uses_injected_executor() {
        let model = Arc::new(CountingModel { vocab: 16 });
        let processor = BatchProcessor::with_executor(
            BatchingConfig::default(),
            Arc::new(ModelBatchExecutor::new(model).with_max_new_tokens(1)),
        );

        let request = token_request(vec![7]);
        let id = request.id.clone();
        let results = processor
            .process_batch(batch_of(vec![request]))
            .await
            .expect("batch must succeed");

        match &results.get(&id).expect("result present").output {
            ProcessingOutput::Tokens(tokens) => assert_eq!(tokens, &vec![8]),
            other => panic!("expected token output, got {other:?}"),
        }
    }

    #[test]
    fn test_processing_stats() {
        let mut stats = ProcessingStats::default();

        let batch = batch_of(vec![]);

        stats.record_batch_start(&batch);
        assert_eq!(stats.current_processing, 1);

        stats.record_batch_completion(&batch, std::time::Duration::from_millis(100));
        assert_eq!(stats.current_processing, 0);
        assert_eq!(stats.total_batches, 1);
        assert!((stats.success_rate - 1.0).abs() < f32::EPSILON);
    }

    /// Regression: the admission-only continuous executor must report an error
    /// instead of returning an empty map that callers would spin on forever.
    #[tokio::test]
    async fn continuous_executor_process_step_is_explicit() {
        let executor = ContinuousBatchingExecutor::new(4);
        executor
            .add_sequences(vec![(RequestId::new(), vec![1, 2, 3])])
            .await
            .expect("admission must succeed");
        assert_eq!(executor.active_sequences().await.len(), 1);

        let error = executor.process_step().await.expect_err("must not claim progress");
        assert!(error.to_string().contains("continuous_batching"));
    }
}
