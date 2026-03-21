//! Batch processing support for efficient inference

use crate::core::tensor::WasmTensor;
use js_sys::{Date, Promise};
use serde::{Deserialize, Serialize};
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// Batching strategies for different use cases
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchingStrategy {
    /// Process immediately without batching
    Immediate,
    /// Fixed-size batches
    FixedSize,
    /// Dynamic batching based on timing
    Dynamic,
    /// Adaptive batching based on load and performance
    Adaptive,
}

/// Batch processing configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct BatchConfig {
    strategy: BatchingStrategy,
    max_batch_size: usize,
    timeout_ms: u32,
    target_latency_ms: u32,
    memory_limit_mb: f32,
    enable_prioritization: bool,
    enable_preemption: bool,
}

#[wasm_bindgen]
impl BatchConfig {
    /// Create a new batch configuration
    #[wasm_bindgen(constructor)]
    pub fn new(strategy: BatchingStrategy, max_batch_size: usize) -> Self {
        Self {
            strategy,
            max_batch_size,
            timeout_ms: 100,       // 100ms default timeout
            target_latency_ms: 50, // Target 50ms latency
            memory_limit_mb: 100.0,
            enable_prioritization: false,
            enable_preemption: false,
        }
    }

    /// Create configuration optimized for real-time applications
    pub fn real_time() -> Self {
        Self {
            strategy: BatchingStrategy::Dynamic,
            max_batch_size: 4,
            timeout_ms: 10,
            target_latency_ms: 20,
            memory_limit_mb: 50.0,
            enable_prioritization: true,
            enable_preemption: true,
        }
    }

    /// Create configuration optimized for throughput
    pub fn throughput() -> Self {
        Self {
            strategy: BatchingStrategy::FixedSize,
            max_batch_size: 32,
            timeout_ms: 500,
            target_latency_ms: 200,
            memory_limit_mb: 500.0,
            enable_prioritization: false,
            enable_preemption: false,
        }
    }

    /// Create configuration optimized for mobile devices
    pub fn mobile() -> Self {
        Self {
            strategy: BatchingStrategy::Adaptive,
            max_batch_size: 2,
            timeout_ms: 50,
            target_latency_ms: 100,
            memory_limit_mb: 20.0,
            enable_prioritization: true,
            enable_preemption: false,
        }
    }

    /// Set timeout for batch completion
    pub fn set_timeout_ms(&mut self, timeout_ms: u32) {
        self.timeout_ms = timeout_ms;
    }

    /// Set target latency
    pub fn set_target_latency_ms(&mut self, latency_ms: u32) {
        self.target_latency_ms = latency_ms;
    }

    /// Set memory limit
    pub fn set_memory_limit_mb(&mut self, limit_mb: f32) {
        self.memory_limit_mb = limit_mb;
    }

    /// Enable request prioritization
    pub fn enable_prioritization(&mut self) {
        self.enable_prioritization = true;
    }

    /// Enable batch preemption for high-priority requests
    pub fn enable_preemption(&mut self) {
        self.enable_preemption = true;
    }
}

/// Priority levels for batch requests
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Batch request with metadata
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub id: String,
    pub input: WasmTensor,
    pub priority: Priority,
    pub timestamp: f64,
    pub timeout_ms: Option<u32>,
    pub callback: Option<js_sys::Function>,
}

/// Batch response with results
#[wasm_bindgen]
pub struct BatchResponse {
    request_id: String,
    result: Option<WasmTensor>,
    error: Option<String>,
    processing_time_ms: f64,
    queue_time_ms: f64,
    batch_size: usize,
}

#[wasm_bindgen]
impl BatchResponse {
    /// Get the request ID
    #[wasm_bindgen(getter)]
    pub fn request_id(&self) -> String {
        self.request_id.clone()
    }

    /// Get the result tensor
    pub fn result(&self) -> Option<WasmTensor> {
        self.result.clone()
    }

    /// Get error message if any
    #[wasm_bindgen(getter)]
    pub fn error(&self) -> Option<String> {
        self.error.clone()
    }

    /// Get processing time in milliseconds
    #[wasm_bindgen(getter)]
    pub fn processing_time_ms(&self) -> f64 {
        self.processing_time_ms
    }

    /// Get queue time in milliseconds
    #[wasm_bindgen(getter)]
    pub fn queue_time_ms(&self) -> f64 {
        self.queue_time_ms
    }

    /// Get the batch size this request was processed in
    #[wasm_bindgen(getter)]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Check if the request was successful
    #[wasm_bindgen(getter)]
    pub fn is_success(&self) -> bool {
        self.error.is_none() && self.result.is_some()
    }
}

/// Batch statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    pub total_requests: usize,
    pub completed_requests: usize,
    pub failed_requests: usize,
    pub average_batch_size: f32,
    pub average_processing_time_ms: f32,
    pub average_queue_time_ms: f32,
    pub throughput_requests_per_second: f32,
    pub memory_usage_mb: f32,
}

/// Batch processor for efficient inference
#[wasm_bindgen]
pub struct BatchProcessor {
    config: BatchConfig,
    pending_requests: Vec<BatchRequest>,
    #[allow(dead_code)]
    active_batch: Option<Vec<BatchRequest>>,
    stats: BatchStats,
    last_batch_time: f64,
    adaptive_batch_size: usize,
    request_counter: usize,
}

#[wasm_bindgen]
impl BatchProcessor {
    /// Create a new batch processor
    #[wasm_bindgen(constructor)]
    pub fn new(config: BatchConfig) -> Self {
        let adaptive_batch_size = config.max_batch_size.min(4); // Start with smaller batches

        Self {
            config,
            pending_requests: Vec::new(),
            active_batch: None,
            stats: BatchStats {
                total_requests: 0,
                completed_requests: 0,
                failed_requests: 0,
                average_batch_size: 0.0,
                average_processing_time_ms: 0.0,
                average_queue_time_ms: 0.0,
                throughput_requests_per_second: 0.0,
                memory_usage_mb: 0.0,
            },
            last_batch_time: Date::now(),
            adaptive_batch_size,
            request_counter: 0,
        }
    }

    /// Add a request to the batch queue
    pub fn add_request(
        &mut self,
        input: WasmTensor,
        priority: Priority,
        timeout_ms: Option<u32>,
    ) -> String {
        self.request_counter += 1;
        let request_id = format!("req_{counter}", counter = self.request_counter);

        let request = BatchRequest {
            id: request_id.clone(),
            input,
            priority,
            timestamp: Date::now(),
            timeout_ms,
            callback: None,
        };

        // Insert request based on priority
        if self.config.enable_prioritization {
            let insert_pos = self
                .pending_requests
                .iter()
                .position(|r| r.priority < priority)
                .unwrap_or(self.pending_requests.len());
            self.pending_requests.insert(insert_pos, request);
        } else {
            self.pending_requests.push(request);
        }

        self.stats.total_requests += 1;

        web_sys::console::log_1(
            &format!(
                "Added request {} to batch queue (priority: {:?})",
                request_id, priority
            )
            .into(),
        );

        request_id
    }

    /// Process pending requests based on batching strategy
    pub async fn process_batch(&mut self) -> Result<Vec<BatchResponse>, JsValue> {
        if self.pending_requests.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = self.determine_batch_size();
        let batch_requests = self.extract_batch(batch_size);

        if batch_requests.is_empty() {
            return Ok(Vec::new());
        }

        let batch_start_time = Date::now();

        web_sys::console::log_1(
            &format!(
                "Processing batch of {len} requests",
                len = batch_requests.len()
            )
            .into(),
        );

        // Combine inputs into a single batch tensor
        let batch_inputs = self.combine_inputs(&batch_requests)?;

        // Process the batch (this would call the actual inference engine)
        let batch_results = self.process_batch_inference(&batch_inputs).await?;

        let processing_time = Date::now() - batch_start_time;

        // Split results back to individual responses
        let responses = self.create_responses(
            batch_requests,
            batch_results,
            processing_time,
            batch_start_time,
        );

        // Update statistics
        self.update_stats(&responses, processing_time);

        // Update adaptive batch size
        if self.config.strategy == BatchingStrategy::Adaptive {
            self.update_adaptive_batch_size(processing_time, responses.len());
        }

        self.last_batch_time = Date::now();

        Ok(responses)
    }

    /// Check if a batch is ready to process
    pub fn is_batch_ready(&self) -> bool {
        if self.pending_requests.is_empty() {
            return false;
        }

        match self.config.strategy {
            BatchingStrategy::Immediate => true,
            BatchingStrategy::FixedSize => {
                self.pending_requests.len() >= self.config.max_batch_size
            },
            BatchingStrategy::Dynamic => {
                let elapsed = Date::now() - self.last_batch_time;
                elapsed >= self.config.timeout_ms as f64
                    || self.pending_requests.len() >= self.config.max_batch_size
            },
            BatchingStrategy::Adaptive => {
                let elapsed = Date::now() - self.last_batch_time;
                elapsed >= self.config.timeout_ms as f64
                    || self.pending_requests.len() >= self.adaptive_batch_size
            },
        }
    }

    /// Get current queue length
    #[wasm_bindgen(getter)]
    pub fn queue_length(&self) -> usize {
        self.pending_requests.len()
    }

    /// Get batch statistics
    pub fn get_stats(&self) -> String {
        format!(
            "Batch Stats: {} total, {} completed, {} failed, avg batch size: {:.1}, avg processing: {:.1}ms, throughput: {:.1} req/s",
            self.stats.total_requests,
            self.stats.completed_requests,
            self.stats.failed_requests,
            self.stats.average_batch_size,
            self.stats.average_processing_time_ms,
            self.stats.throughput_requests_per_second
        )
    }

    /// Clear all pending requests
    pub fn clear_queue(&mut self) {
        self.pending_requests.clear();
        web_sys::console::log_1(&"Batch queue cleared".into());
    }

    /// Update configuration
    pub fn update_config(&mut self, config: BatchConfig) {
        self.config = config;
        self.adaptive_batch_size = self.config.max_batch_size.min(4);
        web_sys::console::log_1(&"Batch configuration updated".into());
    }

    // Private helper methods

    fn determine_batch_size(&self) -> usize {
        match self.config.strategy {
            BatchingStrategy::Immediate => 1,
            BatchingStrategy::FixedSize => {
                self.config.max_batch_size.min(self.pending_requests.len())
            },
            BatchingStrategy::Dynamic => {
                let elapsed = Date::now() - self.last_batch_time;
                if elapsed >= self.config.timeout_ms as f64 {
                    self.pending_requests.len().min(self.config.max_batch_size)
                } else {
                    self.config.max_batch_size.min(self.pending_requests.len())
                }
            },
            BatchingStrategy::Adaptive => self.adaptive_batch_size.min(self.pending_requests.len()),
        }
    }

    fn extract_batch(&mut self, batch_size: usize) -> Vec<BatchRequest> {
        let actual_size = batch_size.min(self.pending_requests.len());
        self.pending_requests.drain(0..actual_size).collect()
    }

    fn combine_inputs(&self, requests: &[BatchRequest]) -> Result<WasmTensor, JsValue> {
        if requests.is_empty() {
            return Err("No requests to process".into());
        }

        if requests.len() == 1 {
            return Ok(requests[0].input.clone());
        }

        // Get the shape of the first tensor to validate compatibility
        let first_shape = requests[0].input.shape();
        let batch_size = requests.len();

        // Validate that all tensors have compatible shapes for batching
        for (i, request) in requests.iter().enumerate().skip(1) {
            let current_shape = request.input.shape();
            if current_shape.len() != first_shape.len() {
                return Err(format!(
                    "Tensor {} has incompatible rank: {} vs {}",
                    i,
                    current_shape.len(),
                    first_shape.len()
                )
                .into());
            }

            // Check that all dimensions except the first (batch dimension) match
            for (dim_idx, (&current_dim, &first_dim)) in
                current_shape[1..].iter().zip(first_shape[1..].iter()).enumerate()
            {
                if current_dim != first_dim {
                    return Err(format!(
                        "Tensor {} has incompatible shape at dimension {}: {} vs {}",
                        i,
                        dim_idx + 1,
                        current_dim,
                        first_dim
                    )
                    .into());
                }
            }
        }

        // Create new shape with batch dimension
        let mut batched_shape = first_shape.clone();
        batched_shape[0] = batch_size;

        // Calculate total size for the batched tensor
        let total_elements = batched_shape.iter().product::<usize>();
        let mut batched_data = vec![0.0f32; total_elements];

        // Copy data from each tensor into the batched tensor
        let elements_per_batch = first_shape.iter().product::<usize>();

        for (batch_idx, request) in requests.iter().enumerate() {
            let tensor_data = request.input.data();
            let start_idx = batch_idx * elements_per_batch;
            let end_idx = start_idx + elements_per_batch.min(tensor_data.len());

            if end_idx <= batched_data.len() {
                batched_data[start_idx..end_idx]
                    .copy_from_slice(&tensor_data[..elements_per_batch.min(tensor_data.len())]);
            }
        }

        // Create the batched tensor
        WasmTensor::new(batched_data, batched_shape)
    }

    async fn process_batch_inference(
        &self,
        batch_input: &WasmTensor,
    ) -> Result<Vec<WasmTensor>, JsValue> {
        // Simulate inference processing time
        let processing_delay = 10.0 + (self.pending_requests.len() as f64 * 2.0);

        // Simulate inference delay (in a real implementation, this would be actual model inference)
        let delay_promise = Promise::new(&mut |resolve, _| {
            let _timeout_id = web_sys::window()
                .expect("window should be available in browser context")
                .set_timeout_with_callback_and_timeout_and_arguments_0(
                    &resolve,
                    processing_delay as i32,
                )
                .expect("set_timeout should succeed with valid callback");
            // Note: In a real app, you'd want to track timeout_id for cleanup
        });

        JsFuture::from(delay_promise).await?;

        // Perform actual inference on the batched input
        let batch_shape = batch_input.shape();
        let batch_size = batch_shape[0];

        // For demonstration, perform a simple transformation
        // In a real implementation, this would use the model's forward pass
        let batch_output = match batch_shape.len() {
            2 => {
                // For 2D tensors (batch_size, features), return logits
                let output_features = 10; // Assuming classification with 10 classes
                batch_input.matmul(&WasmTensor::randn(vec![batch_shape[1], output_features])?)?
            },
            3 => {
                // For 3D tensors (batch_size, seq_len, features), return sequence output
                let _output_features = batch_shape[2]; // Same feature size
                batch_input.relu() // Simple activation for demonstration
            },
            _ => {
                // For other shapes, apply element-wise transformation
                batch_input.relu()
            },
        };

        // Split the batched output back into individual results
        let output_shape = batch_output.shape();
        let elements_per_batch = output_shape[1..].iter().product::<usize>();
        let output_data = batch_output.data();

        let mut results = Vec::new();
        for batch_idx in 0..batch_size {
            let start_idx = batch_idx * elements_per_batch;
            let end_idx = start_idx + elements_per_batch;

            if end_idx <= output_data.len() {
                let batch_data = output_data[start_idx..end_idx].to_vec();
                let mut individual_shape = output_shape[1..].to_vec();
                individual_shape.insert(0, 1); // Add batch dimension of 1

                results.push(WasmTensor::new(batch_data, individual_shape)?);
            }
        }

        if results.len() != batch_size {
            return Err(format!(
                "Expected {batch_size} results but got {len}",
                len = results.len()
            )
            .into());
        }

        Ok(results)
    }

    fn create_responses(
        &self,
        requests: Vec<BatchRequest>,
        results: Vec<WasmTensor>,
        processing_time: f64,
        batch_start_time: f64,
    ) -> Vec<BatchResponse> {
        let batch_size = requests.len();
        requests
            .into_iter()
            .zip(results)
            .map(|(request, result)| {
                let queue_time = batch_start_time - request.timestamp;

                BatchResponse {
                    request_id: request.id,
                    result: Some(result),
                    error: None,
                    processing_time_ms: processing_time,
                    queue_time_ms: queue_time,
                    batch_size,
                }
            })
            .collect()
    }

    fn update_stats(&mut self, responses: &[BatchResponse], processing_time: f64) {
        let successful = responses.iter().filter(|r| r.is_success()).count();
        let failed = responses.len() - successful;

        self.stats.completed_requests += successful;
        self.stats.failed_requests += failed;

        // Update running averages
        let total_completed = self.stats.completed_requests as f32;
        if total_completed > 0.0 {
            self.stats.average_batch_size = (self.stats.average_batch_size
                * (total_completed - successful as f32)
                + responses.len() as f32)
                / total_completed;

            self.stats.average_processing_time_ms = (self.stats.average_processing_time_ms
                * (total_completed - successful as f32)
                + processing_time as f32)
                / total_completed;

            if let Some(first_response) = responses.first() {
                self.stats.average_queue_time_ms = (self.stats.average_queue_time_ms
                    * (total_completed - 1.0)
                    + first_response.queue_time_ms as f32)
                    / total_completed;
            }
        }

        // Calculate throughput (requests per second)
        if processing_time > 0.0 {
            self.stats.throughput_requests_per_second =
                (responses.len() as f32) / (processing_time / 1000.0) as f32;
        }
    }

    fn update_adaptive_batch_size(&mut self, processing_time: f64, batch_size: usize) {
        let target_latency = self.config.target_latency_ms as f64;

        if processing_time > target_latency * 1.5 {
            // Decrease batch size if we're too slow
            self.adaptive_batch_size = (self.adaptive_batch_size - 1).max(1);
        } else if processing_time < target_latency * 0.7 && batch_size == self.adaptive_batch_size {
            // Increase batch size if we're fast and the batch was full
            self.adaptive_batch_size =
                (self.adaptive_batch_size + 1).min(self.config.max_batch_size);
        }

        web_sys::console::log_1(
            &format!(
                "Adaptive batch size updated to {}",
                self.adaptive_batch_size
            )
            .into(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config() {
        let config = BatchConfig::real_time();
        assert_eq!(config.strategy, BatchingStrategy::Dynamic);
        assert!(config.enable_prioritization);

        let throughput_config = BatchConfig::throughput();
        assert_eq!(throughput_config.max_batch_size, 32);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }
}
