// SPDX-License-Identifier: Apache-2.0

//! # Batch Inference Utilities for Trustformers Models
//!
//! This module provides efficient batch processing capabilities for transformer models,
//! enabling high-throughput inference with automatic optimization and memory management.
//!
//! ## Features
//!
//! - **Dynamic batching**: Automatically group requests into optimal batch sizes
//! - **Smart padding**: Efficient padding strategies to minimize computation waste
//! - **Memory management**: Automatic batch size adjustment based on available memory
//! - **Parallel processing**: Leverage multi-core CPUs for batch preprocessing
//! - **Adaptive batching**: Adjust batch size based on sequence lengths
//! - **Batch queue management**: Priority-based request scheduling
//!
//! ## Usage Example
//!
//! ```rust
//! use trustformers_models::batch_inference::{BatchProcessor, BatchConfig};
//!
//! // Configure batch processing
//! let config = BatchConfig {
//!     max_batch_size: 32,
//!     timeout_ms: 100,
//!     padding_strategy: PaddingStrategy::Longest,
//!     ..Default::default()
//! };
//!
//! let processor = BatchProcessor::new(config);
//!
//! // Process requests in batches
//! // let results = processor.process_batch(requests)?;
//! ```

use scirs2_core::ndarray::Array2;
use scirs2_core::parallel_ops::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;

/// Padding strategy for batch processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// Pad all sequences to the longest sequence in the batch
    Longest,

    /// Pad all sequences to a fixed length
    Fixed(usize),

    /// Pad to multiples of a specific value (e.g., 8 for better GPU utilization)
    Multiple(usize),

    /// No padding (requires all sequences to be same length)
    None,
}

impl Default for PaddingStrategy {
    fn default() -> Self {
        Self::Longest
    }
}

/// Truncation strategy for sequences that exceed max length
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Truncate from the end (keep beginning)
    End,

    /// Truncate from the beginning (keep end)
    Beginning,

    /// Error if sequence exceeds max length
    Error,

    /// No truncation
    None,
}

impl Default for TruncationStrategy {
    fn default() -> Self {
        Self::End
    }
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,

    /// Minimum batch size before processing (for efficiency)
    pub min_batch_size: usize,

    /// Maximum wait time before processing incomplete batch (milliseconds)
    pub timeout_ms: u64,

    /// Padding strategy
    pub padding_strategy: PaddingStrategy,

    /// Truncation strategy
    pub truncation_strategy: TruncationStrategy,

    /// Maximum sequence length (for truncation/padding)
    pub max_sequence_length: Option<usize>,

    /// Padding token ID
    pub pad_token_id: u32,

    /// Enable dynamic batch size adjustment based on sequence lengths
    pub dynamic_batching: bool,

    /// Target memory usage in MB (for automatic batch size adjustment)
    pub target_memory_mb: Option<usize>,

    /// Enable parallel preprocessing
    pub parallel_preprocessing: bool,

    /// Sort sequences by length for more efficient batching
    pub sort_by_length: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            min_batch_size: 1,
            timeout_ms: 100,
            padding_strategy: PaddingStrategy::default(),
            truncation_strategy: TruncationStrategy::default(),
            max_sequence_length: None,
            pad_token_id: 0,
            dynamic_batching: true,
            target_memory_mb: None,
            parallel_preprocessing: true,
            sort_by_length: true,
        }
    }
}

impl BatchConfig {
    /// Create a configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            max_batch_size: 8,
            min_batch_size: 1,
            timeout_ms: 10,
            dynamic_batching: false,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            max_batch_size: 128,
            min_batch_size: 32,
            timeout_ms: 500,
            dynamic_batching: true,
            sort_by_length: true,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_batch_size == 0 {
            return Err(TrustformersError::invalid_config(
                "max_batch_size must be positive".to_string(),
            ));
        }

        if self.min_batch_size > self.max_batch_size {
            return Err(TrustformersError::invalid_config(
                "min_batch_size cannot exceed max_batch_size".to_string(),
            ));
        }

        if let PaddingStrategy::Fixed(len) = self.padding_strategy {
            if len == 0 {
                return Err(TrustformersError::invalid_config(
                    "fixed padding length must be positive".to_string(),
                ));
            }
        }

        if let PaddingStrategy::Multiple(multiple) = self.padding_strategy {
            if multiple == 0 {
                return Err(TrustformersError::invalid_config(
                    "padding multiple must be positive".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// A batch of input sequences ready for model inference
#[derive(Debug, Clone)]
pub struct BatchedInput {
    /// Input IDs tensor (batch_size, max_seq_len)
    pub input_ids: Tensor,

    /// Attention mask (batch_size, max_seq_len) - 1 for real tokens, 0 for padding
    pub attention_mask: Tensor,

    /// Original sequence lengths before padding
    pub sequence_lengths: Vec<usize>,

    /// Batch size
    pub batch_size: usize,

    /// Maximum sequence length in this batch
    pub max_seq_length: usize,
}

/// A single inference request
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Unique request ID
    pub id: String,

    /// Input token IDs
    pub input_ids: Vec<u32>,

    /// Optional priority (higher = more urgent)
    pub priority: i32,

    /// Request timestamp
    pub timestamp: Instant,
}

impl InferenceRequest {
    /// Create a new inference request
    pub fn new(id: String, input_ids: Vec<u32>) -> Self {
        Self {
            id,
            input_ids,
            priority: 0,
            timestamp: Instant::now(),
        }
    }

    /// Create a request with priority
    pub fn with_priority(id: String, input_ids: Vec<u32>, priority: i32) -> Self {
        Self {
            id,
            input_ids,
            priority,
            timestamp: Instant::now(),
        }
    }
}

/// Batch processor for efficient inference
pub struct BatchProcessor {
    config: BatchConfig,
    request_queue: Arc<Mutex<VecDeque<InferenceRequest>>>,
    batch_start_time: Arc<Mutex<Option<Instant>>>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: BatchConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            batch_start_time: Arc::new(Mutex::new(None)),
        })
    }

    /// Add a request to the queue
    pub fn enqueue(&self, request: InferenceRequest) -> Result<()> {
        let mut queue = self.request_queue.lock().map_err(|e| {
            TrustformersError::lock_error(format!("Failed to lock request queue: {}", e))
        })?;

        // Insert based on priority (higher priority first)
        let insert_pos =
            queue.iter().position(|r| r.priority < request.priority).unwrap_or(queue.len());

        queue.insert(insert_pos, request);

        // Set batch start time if this is the first request
        let mut start_time = self.batch_start_time.lock().map_err(|e| {
            TrustformersError::lock_error(format!("Failed to lock batch start time: {}", e))
        })?;

        if start_time.is_none() {
            *start_time = Some(Instant::now());
        }

        Ok(())
    }

    /// Check if batch should be processed
    pub fn should_process_batch(&self) -> Result<bool> {
        let queue = self.request_queue.lock().map_err(|e| {
            TrustformersError::lock_error(format!("Failed to lock request queue: {}", e))
        })?;

        if queue.is_empty() {
            return Ok(false);
        }

        // Process if we have enough requests
        if queue.len() >= self.config.min_batch_size {
            return Ok(true);
        }

        // Process if timeout exceeded
        let start_time = self.batch_start_time.lock().map_err(|e| {
            TrustformersError::lock_error(format!("Failed to lock batch start time: {}", e))
        })?;

        if let Some(start) = *start_time {
            let elapsed = start.elapsed();
            if elapsed >= Duration::from_millis(self.config.timeout_ms) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get next batch of requests from queue
    pub fn get_next_batch(&self) -> Result<Vec<InferenceRequest>> {
        let mut queue = self.request_queue.lock().map_err(|e| {
            TrustformersError::lock_error(format!("Failed to lock request queue: {}", e))
        })?;

        let batch_size = self.config.max_batch_size.min(queue.len());
        let mut batch: Vec<InferenceRequest> = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            if let Some(request) = queue.pop_front() {
                batch.push(request);
            }
        }

        // Reset batch start time if queue is empty
        if queue.is_empty() {
            let mut start_time = self.batch_start_time.lock().map_err(|e| {
                TrustformersError::lock_error(format!("Failed to lock batch start time: {}", e))
            })?;
            *start_time = None;
        }

        Ok(batch)
    }

    /// Prepare a batch for inference
    pub fn prepare_batch(&self, requests: Vec<InferenceRequest>) -> Result<BatchedInput> {
        if requests.is_empty() {
            return Err(TrustformersError::invalid_input(
                "Cannot prepare batch from empty requests".to_string(),
            ));
        }

        let mut sequences: Vec<Vec<u32>> = requests.into_iter().map(|r| r.input_ids).collect();

        // Sort by length if enabled (for more efficient batching)
        if self.config.sort_by_length {
            sequences.sort_by_key(|s| std::cmp::Reverse(s.len()));
        }

        // Apply truncation
        if let Some(max_len) = self.config.max_sequence_length {
            sequences = self.apply_truncation(sequences, max_len)?;
        }

        // Determine target length for padding
        let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let target_len = self.compute_target_length(max_len)?;

        // Apply padding
        let (padded_ids, attention_masks, seq_lengths) =
            self.apply_padding(&sequences, target_len)?;

        let batch_size = sequences.len();

        Ok(BatchedInput {
            input_ids: padded_ids,
            attention_mask: attention_masks,
            sequence_lengths: seq_lengths,
            batch_size,
            max_seq_length: target_len,
        })
    }

    /// Apply truncation to sequences
    fn apply_truncation(&self, sequences: Vec<Vec<u32>>, max_len: usize) -> Result<Vec<Vec<u32>>> {
        let process_seq = |seq: Vec<u32>| -> Result<Vec<u32>> {
            if seq.len() <= max_len {
                return Ok(seq);
            }

            match self.config.truncation_strategy {
                TruncationStrategy::End => Ok(seq[..max_len].to_vec()),
                TruncationStrategy::Beginning => Ok(seq[seq.len() - max_len..].to_vec()),
                TruncationStrategy::Error => Err(TrustformersError::invalid_input(format!(
                    "Sequence length {} exceeds max length {}",
                    seq.len(),
                    max_len
                ))),
                TruncationStrategy::None => Ok(seq),
            }
        };

        if self.config.parallel_preprocessing && sequences.len() > 4 {
            sequences.into_par_iter().map(process_seq).collect::<Result<Vec<_>>>()
        } else {
            sequences.into_iter().map(process_seq).collect::<Result<Vec<_>>>()
        }
    }

    /// Compute target length for padding
    fn compute_target_length(&self, max_len: usize) -> Result<usize> {
        match self.config.padding_strategy {
            PaddingStrategy::Longest => Ok(max_len),
            PaddingStrategy::Fixed(len) => Ok(len),
            PaddingStrategy::Multiple(multiple) => {
                // Round up to nearest multiple
                Ok(max_len.div_ceil(multiple) * multiple)
            },
            PaddingStrategy::None => Ok(max_len),
        }
    }

    /// Apply padding to sequences
    fn apply_padding(
        &self,
        sequences: &[Vec<u32>],
        target_len: usize,
    ) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let batch_size = sequences.len();
        let pad_token = self.config.pad_token_id;

        // Create padded input IDs and attention masks
        let mut input_ids = Array2::<u32>::from_elem((batch_size, target_len), pad_token);
        let mut attention_mask = Array2::<i32>::zeros((batch_size, target_len));
        let mut seq_lengths = Vec::with_capacity(batch_size);

        for (i, seq) in sequences.iter().enumerate() {
            let seq_len = seq.len();
            seq_lengths.push(seq_len);

            // Copy sequence
            for (j, &token) in seq.iter().enumerate() {
                input_ids[[i, j]] = token;
                attention_mask[[i, j]] = 1;
            }
        }

        let input_ids_tensor = Tensor::F32(input_ids.mapv(|x| x as f32).into_dyn());

        let attention_mask_tensor = Tensor::F32(attention_mask.mapv(|x| x as f32).into_dyn());

        Ok((input_ids_tensor, attention_mask_tensor, seq_lengths))
    }

    /// Estimate memory usage for a batch
    pub fn estimate_memory_mb(&self, batch_size: usize, seq_length: usize) -> usize {
        // Rough estimate: 4 bytes per token (float32)
        // Account for input_ids, attention_mask, and intermediate tensors
        let tokens = batch_size * seq_length;
        let bytes_per_token = 4;
        let overhead_multiplier = 10; // Account for intermediate activations

        let memory_bytes = tokens * bytes_per_token * overhead_multiplier;
        let memory_mb = memory_bytes / (1024 * 1024);

        // Ensure at least 1 MB for small batches
        memory_mb.max(1)
    }

    /// Adjust batch size based on memory constraints
    pub fn adjust_batch_size(&self, sequences: &[Vec<u32>]) -> usize {
        if let Some(target_mb) = self.config.target_memory_mb {
            let max_seq_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);

            // Binary search for optimal batch size
            let mut left = 1;
            let mut right = self.config.max_batch_size;
            let mut best_size = 1;

            while left <= right {
                let mid = (left + right) / 2;
                let estimated_mb = self.estimate_memory_mb(mid, max_seq_len);

                if estimated_mb <= target_mb {
                    best_size = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }

            best_size.min(sequences.len())
        } else {
            self.config.max_batch_size.min(sequences.len())
        }
    }

    /// Get current queue size
    pub fn queue_size(&self) -> Result<usize> {
        let queue = self.request_queue.lock().map_err(|e| {
            TrustformersError::lock_error(format!("Failed to lock request queue: {}", e))
        })?;
        Ok(queue.len())
    }
}

/// Batch statistics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct BatchStatistics {
    /// Average batch size
    pub avg_batch_size: f32,

    /// Average sequence length
    pub avg_sequence_length: f32,

    /// Average wait time in milliseconds
    pub avg_wait_time_ms: f32,

    /// Total number of batches processed
    pub total_batches: usize,

    /// Total number of requests processed
    pub total_requests: usize,

    /// Average padding ratio (padded tokens / total tokens)
    pub avg_padding_ratio: f32,
}

impl BatchStatistics {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self {
            avg_batch_size: 0.0,
            avg_sequence_length: 0.0,
            avg_wait_time_ms: 0.0,
            total_batches: 0,
            total_requests: 0,
            avg_padding_ratio: 0.0,
        }
    }

    /// Update statistics with new batch
    pub fn update(&mut self, batch: &BatchedInput, wait_time_ms: f32) {
        self.total_batches += 1;
        self.total_requests += batch.batch_size;

        // Update running averages
        let n = self.total_batches as f32;
        self.avg_batch_size = (self.avg_batch_size * (n - 1.0) + batch.batch_size as f32) / n;

        let actual_tokens: usize = batch.sequence_lengths.iter().sum();
        let total_tokens = batch.batch_size * batch.max_seq_length;
        let padding_ratio = 1.0 - (actual_tokens as f32 / total_tokens as f32);

        self.avg_padding_ratio = (self.avg_padding_ratio * (n - 1.0) + padding_ratio) / n;
        self.avg_wait_time_ms = (self.avg_wait_time_ms * (n - 1.0) + wait_time_ms) / n;

        let avg_seq_len: f32 =
            batch.sequence_lengths.iter().sum::<usize>() as f32 / batch.batch_size as f32;
        self.avg_sequence_length = (self.avg_sequence_length * (n - 1.0) + avg_seq_len) / n;
    }
}

impl Default for BatchStatistics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_validation() {
        let valid_config = BatchConfig::default();
        assert!(valid_config.validate().is_ok());

        let mut invalid_config = BatchConfig::default();
        invalid_config.max_batch_size = 0;
        assert!(invalid_config.validate().is_err());

        let mut invalid_config2 = BatchConfig::default();
        invalid_config2.min_batch_size = 100;
        invalid_config2.max_batch_size = 10;
        assert!(invalid_config2.validate().is_err());
    }

    #[test]
    fn test_batch_config_presets() {
        let low_latency = BatchConfig::low_latency();
        assert_eq!(low_latency.max_batch_size, 8);
        assert_eq!(low_latency.timeout_ms, 10);

        let high_throughput = BatchConfig::high_throughput();
        assert_eq!(high_throughput.max_batch_size, 128);
        assert!(high_throughput.sort_by_length);
    }

    #[test]
    fn test_padding_strategy() {
        let longest = PaddingStrategy::Longest;
        let fixed = PaddingStrategy::Fixed(128);
        let multiple = PaddingStrategy::Multiple(8);

        assert!(matches!(longest, PaddingStrategy::Longest));
        assert!(matches!(fixed, PaddingStrategy::Fixed(128)));
        assert!(matches!(multiple, PaddingStrategy::Multiple(8)));
    }

    #[test]
    fn test_batch_processor_creation() -> Result<()> {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config)?;
        assert_eq!(processor.queue_size()?, 0);
        Ok(())
    }

    #[test]
    fn test_enqueue_requests() -> Result<()> {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config)?;

        let request1 = InferenceRequest::new("req1".to_string(), vec![1, 2, 3]);
        let request2 = InferenceRequest::new("req2".to_string(), vec![4, 5, 6, 7]);

        processor.enqueue(request1)?;
        processor.enqueue(request2)?;

        assert_eq!(processor.queue_size()?, 2);
        Ok(())
    }

    #[test]
    fn test_priority_ordering() -> Result<()> {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config)?;

        let low_priority = InferenceRequest::with_priority("low".to_string(), vec![1, 2], 1);
        let high_priority = InferenceRequest::with_priority("high".to_string(), vec![3, 4], 10);

        processor.enqueue(low_priority)?;
        processor.enqueue(high_priority)?;

        let batch = processor.get_next_batch()?;
        assert_eq!(batch[0].id, "high"); // High priority should be first
        Ok(())
    }

    #[test]
    fn test_truncation() -> Result<()> {
        let mut config = BatchConfig::default();
        config.max_sequence_length = Some(5);
        config.truncation_strategy = TruncationStrategy::End;

        let processor = BatchProcessor::new(config)?;

        let sequences = vec![vec![1, 2, 3, 4, 5, 6, 7, 8], vec![9, 10, 11]];

        let truncated = processor.apply_truncation(sequences, 5)?;
        assert_eq!(truncated[0].len(), 5);
        assert_eq!(truncated[0], vec![1, 2, 3, 4, 5]);
        assert_eq!(truncated[1], vec![9, 10, 11]);

        Ok(())
    }

    #[test]
    fn test_padding_computation() -> Result<()> {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config)?;

        // Test longest padding
        let target = processor.compute_target_length(10)?;
        assert_eq!(target, 10);

        // Test multiple padding
        let mut config2 = BatchConfig::default();
        config2.padding_strategy = PaddingStrategy::Multiple(8);
        let processor2 = BatchProcessor::new(config2)?;

        let target2 = processor2.compute_target_length(10)?;
        assert_eq!(target2, 16); // Rounded up to multiple of 8

        Ok(())
    }

    #[test]
    fn test_memory_estimation() {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config).unwrap();

        // Test with larger sequences to see actual memory differences
        let mem_mb_small = processor.estimate_memory_mb(32, 2048);
        assert!(mem_mb_small > 1);

        // Larger batch should use more memory
        let mem_mb_large = processor.estimate_memory_mb(64, 2048);
        assert!(mem_mb_large > mem_mb_small);

        // Very small batch should still return at least 1 MB
        let mem_mb_tiny = processor.estimate_memory_mb(1, 10);
        assert_eq!(mem_mb_tiny, 1);
    }

    #[test]
    fn test_batch_statistics() {
        let mut stats = BatchStatistics::new();
        assert_eq!(stats.total_batches, 0);

        let batch = BatchedInput {
            input_ids: Tensor::F32(Array2::zeros((2, 10)).into_dyn()),
            attention_mask: Tensor::F32(Array2::zeros((2, 10)).into_dyn()),
            sequence_lengths: vec![8, 6],
            batch_size: 2,
            max_seq_length: 10,
        };

        stats.update(&batch, 50.0);
        assert_eq!(stats.total_batches, 1);
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.avg_batch_size, 2.0);
    }

    #[test]
    fn test_prepare_batch() -> Result<()> {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config)?;

        let requests = vec![
            InferenceRequest::new("1".to_string(), vec![1, 2, 3]),
            InferenceRequest::new("2".to_string(), vec![4, 5]),
        ];

        let batched = processor.prepare_batch(requests)?;
        assert_eq!(batched.batch_size, 2);
        assert_eq!(batched.max_seq_length, 3);
        assert_eq!(batched.sequence_lengths, vec![3, 2]);

        Ok(())
    }
}
