//! Batch processing for efficient model serving
//!
//! This module provides advanced batch processing capabilities for TrustformeRS model serving,
//! including dynamic batching, request queuing, and performance optimization.

use anyhow::anyhow;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{TrustformersError, TrustformersResult};
use crate::{c_str_to_string, string_to_c_str};

/// Global batch processor registry
static BATCH_PROCESSOR_REGISTRY: Lazy<Mutex<HashMap<String, Arc<BatchProcessor>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessorConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum wait time before processing batch (ms)
    pub max_wait_time_ms: u64,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Number of worker threads
    pub num_workers: usize,
    /// Enable dynamic batching
    pub enable_dynamic_batching: bool,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Enable request prioritization
    pub enable_prioritization: bool,
    /// Memory limit for batches (MB)
    pub memory_limit_mb: usize,
}

impl Default for BatchProcessorConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_wait_time_ms: 100,
            max_queue_size: 1000,
            num_workers: 4,
            enable_dynamic_batching: true,
            batch_timeout_ms: 5000,
            enable_prioritization: true,
            memory_limit_mb: 2048,
        }
    }
}

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Batch processing request
#[derive(Debug)]
pub struct BatchRequest {
    pub id: String,
    pub input_data: Vec<f32>,
    pub input_shape: Vec<usize>,
    pub priority: RequestPriority,
    pub timeout_ms: u64,
    pub created_at: Instant,
    pub response_sender: tokio::sync::oneshot::Sender<BatchResponse>,
}

/// Batch processing response
#[derive(Debug, Clone)]
pub struct BatchResponse {
    pub id: String,
    pub output_data: Vec<f32>,
    pub output_shape: Vec<usize>,
    pub processing_time_ms: f64,
    pub status: BatchStatus,
    pub error_message: Option<String>,
}

/// Batch processing status
#[derive(Debug, Clone, PartialEq)]
pub enum BatchStatus {
    Success,
    Timeout,
    Error,
    QueueFull,
    MemoryExhausted,
}

/// Batch of requests to process together
#[derive(Debug)]
struct ProcessingBatch {
    requests: Vec<BatchRequest>,
    created_at: Instant,
    estimated_memory_mb: usize,
}

/// Batch processor statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct BatchProcessorStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub timeout_requests: u64,
    pub queue_full_requests: u64,
    pub current_queue_size: usize,
    pub peak_queue_size: usize,
    pub average_batch_size: f64,
    pub average_processing_time_ms: f64,
    pub average_wait_time_ms: f64,
    pub total_batches_processed: u64,
    pub memory_usage_mb: usize,
    pub peak_memory_usage_mb: usize,
}

/// Main batch processor
pub struct BatchProcessor {
    config: BatchProcessorConfig,
    model_handle: usize,
    request_queue: Arc<Mutex<VecDeque<BatchRequest>>>,
    queue_condvar: Arc<Condvar>,
    stats: Arc<Mutex<BatchProcessorStats>>,
    shutdown_signal: Arc<Mutex<bool>>,
    worker_handles: Vec<thread::JoinHandle<()>>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: BatchProcessorConfig, model_handle: usize) -> TrustformersResult<Self> {
        let request_queue = Arc::new(Mutex::new(VecDeque::new()));
        let queue_condvar = Arc::new(Condvar::new());
        let stats = Arc::new(Mutex::new(BatchProcessorStats::default()));
        let shutdown_signal = Arc::new(Mutex::new(false));

        let mut processor = Self {
            config: config.clone(),
            model_handle,
            request_queue: request_queue.clone(),
            queue_condvar: queue_condvar.clone(),
            stats: stats.clone(),
            shutdown_signal: shutdown_signal.clone(),
            worker_handles: Vec::new(),
        };

        // Start worker threads
        processor.start_workers()?;

        Ok(processor)
    }

    /// Start worker threads for batch processing
    fn start_workers(&mut self) -> TrustformersResult<()> {
        for worker_id in 0..self.config.num_workers {
            let request_queue = self.request_queue.clone();
            let queue_condvar = self.queue_condvar.clone();
            let stats = self.stats.clone();
            let shutdown_signal = self.shutdown_signal.clone();
            let config = self.config.clone();
            let model_handle = self.model_handle;

            let handle = thread::spawn(move || {
                Self::worker_loop(
                    worker_id,
                    request_queue,
                    queue_condvar,
                    stats,
                    shutdown_signal,
                    config,
                    model_handle,
                );
            });

            self.worker_handles.push(handle);
        }

        Ok(())
    }

    /// Worker thread main loop
    fn worker_loop(
        worker_id: usize,
        request_queue: Arc<Mutex<VecDeque<BatchRequest>>>,
        queue_condvar: Arc<Condvar>,
        stats: Arc<Mutex<BatchProcessorStats>>,
        shutdown_signal: Arc<Mutex<bool>>,
        config: BatchProcessorConfig,
        model_handle: usize,
    ) {
        eprintln!("Batch processor worker {} started", worker_id);

        loop {
            // Check shutdown signal
            if let Ok(shutdown) = shutdown_signal.lock() {
                if *shutdown {
                    break;
                }
            }

            // Get batch of requests
            let batch = Self::get_batch_for_processing(&request_queue, &queue_condvar, &config);

            if let Some(batch) = batch {
                // Process the batch
                Self::process_batch(batch, &stats, model_handle);
            } else {
                // No requests available, wait briefly
                thread::sleep(Duration::from_millis(10));
            }
        }

        eprintln!("Batch processor worker {} stopped", worker_id);
    }

    /// Get a batch of requests for processing
    fn get_batch_for_processing(
        request_queue: &Arc<Mutex<VecDeque<BatchRequest>>>,
        queue_condvar: &Arc<Condvar>,
        config: &BatchProcessorConfig,
    ) -> Option<ProcessingBatch> {
        let mut queue = request_queue.lock().ok()?;

        // Wait for requests or timeout
        let wait_result = queue_condvar
            .wait_timeout(queue, Duration::from_millis(config.max_wait_time_ms))
            .ok()?;

        queue = wait_result.0;

        if queue.is_empty() {
            return None;
        }

        // Build batch dynamically
        let mut batch_requests = Vec::new();
        let mut estimated_memory = 0;
        let batch_start = Instant::now();

        while !queue.is_empty() && batch_requests.len() < config.max_batch_size {
            // Check memory limit
            if estimated_memory > config.memory_limit_mb * 1024 * 1024 {
                break;
            }

            // Check timeout
            if batch_start.elapsed().as_millis() > config.max_wait_time_ms as u128 {
                break;
            }

            if let Some(request) = queue.pop_front() {
                // Estimate memory usage for this request
                let request_memory = request.input_data.len() * std::mem::size_of::<f32>();
                estimated_memory += request_memory;

                batch_requests.push(request);

                // If dynamic batching is disabled, take only one request
                if !config.enable_dynamic_batching {
                    break;
                }
            }
        }

        if batch_requests.is_empty() {
            None
        } else {
            Some(ProcessingBatch {
                requests: batch_requests,
                created_at: batch_start,
                estimated_memory_mb: estimated_memory / (1024 * 1024),
            })
        }
    }

    /// Process a batch of requests
    fn process_batch(
        batch: ProcessingBatch,
        stats: &Arc<Mutex<BatchProcessorStats>>,
        model_handle: usize,
    ) {
        let processing_start = Instant::now();
        let batch_size = batch.requests.len();

        // Update stats
        if let Ok(mut stats_guard) = stats.lock() {
            stats_guard.total_batches_processed += 1;
            stats_guard.memory_usage_mb = batch.estimated_memory_mb;
            stats_guard.peak_memory_usage_mb =
                stats_guard.peak_memory_usage_mb.max(batch.estimated_memory_mb);
        }

        // Process each request in the batch
        for request in batch.requests {
            let request_start = Instant::now();

            // Check request timeout
            if request.created_at.elapsed().as_millis() > request.timeout_ms as u128 {
                let timeout_response = BatchResponse {
                    id: request.id.clone(),
                    output_data: Vec::new(),
                    output_shape: Vec::new(),
                    processing_time_ms: request_start.elapsed().as_millis() as f64,
                    status: BatchStatus::Timeout,
                    error_message: Some("Request timeout".to_string()),
                };

                let _ = request.response_sender.send(timeout_response);

                // Update timeout stats
                if let Ok(mut stats_guard) = stats.lock() {
                    stats_guard.timeout_requests += 1;
                    stats_guard.total_requests += 1;
                }
                continue;
            }

            // Simulate model inference (replace with actual model call)
            let inference_result = Self::simulate_model_inference(
                model_handle,
                &request.input_data,
                &request.input_shape,
            );

            let processing_time = request_start.elapsed().as_millis() as f64;

            let response = match inference_result {
                Ok((output_data, output_shape)) => BatchResponse {
                    id: request.id.clone(),
                    output_data,
                    output_shape,
                    processing_time_ms: processing_time,
                    status: BatchStatus::Success,
                    error_message: None,
                },
                Err(e) => BatchResponse {
                    id: request.id.clone(),
                    output_data: Vec::new(),
                    output_shape: Vec::new(),
                    processing_time_ms: processing_time,
                    status: BatchStatus::Error,
                    error_message: Some(e.to_string()),
                },
            };

            // Send response
            let _ = request.response_sender.send(response.clone());

            // Update stats
            if let Ok(mut stats_guard) = stats.lock() {
                stats_guard.total_requests += 1;

                match response.status {
                    BatchStatus::Success => stats_guard.successful_requests += 1,
                    BatchStatus::Error => stats_guard.failed_requests += 1,
                    _ => {},
                }

                // Update averages
                let total = stats_guard.total_requests as f64;
                stats_guard.average_processing_time_ms =
                    (stats_guard.average_processing_time_ms * (total - 1.0) + processing_time)
                        / total;

                let wait_time = request_start.duration_since(request.created_at).as_millis() as f64;
                stats_guard.average_wait_time_ms =
                    (stats_guard.average_wait_time_ms * (total - 1.0) + wait_time) / total;

                let batch_count = stats_guard.total_batches_processed as f64;
                stats_guard.average_batch_size =
                    (stats_guard.average_batch_size * (batch_count - 1.0) + batch_size as f64)
                        / batch_count;
            }
        }
    }

    /// Simulate model inference (replace with actual model call)
    fn simulate_model_inference(
        _model_handle: usize,
        input_data: &[f32],
        input_shape: &[usize],
    ) -> TrustformersResult<(Vec<f32>, Vec<usize>)> {
        // Simulate processing time based on input size
        let input_size = input_data.len();
        let processing_time_ms = (input_size as f64 * 0.001).max(1.0).min(100.0);
        thread::sleep(Duration::from_millis(processing_time_ms as u64));

        // Simulate output (in real implementation, call actual model)
        let output_size = input_size / 2; // Example: reduce dimensionality
        let mut output_data = Vec::with_capacity(output_size);

        for i in 0..output_size {
            // Simple transformation for simulation
            let val =
                input_data.get(i * 2).unwrap_or(&0.0) + input_data.get(i * 2 + 1).unwrap_or(&0.0);
            output_data.push(val * 0.5);
        }

        // Calculate output shape
        let mut output_shape = input_shape.to_vec();
        if !output_shape.is_empty() {
            let last_idx = output_shape.len() - 1;
            output_shape[last_idx] =
                output_size / input_shape[..input_shape.len() - 1].iter().product::<usize>().max(1);
        }

        Ok((output_data, output_shape))
    }

    /// Submit a request for batch processing
    pub async fn submit_request(
        &self,
        id: String,
        input_data: Vec<f32>,
        input_shape: Vec<usize>,
        priority: RequestPriority,
        timeout_ms: u64,
    ) -> TrustformersResult<BatchResponse> {
        // Check queue size
        let queue_size = {
            let queue = self.request_queue.lock().map_err(|_| anyhow!("Lock error"))?;
            queue.len()
        };

        if queue_size >= self.config.max_queue_size {
            // Update queue full stats
            if let Ok(mut stats) = self.stats.lock() {
                stats.queue_full_requests += 1;
                stats.total_requests += 1;
            }

            return Ok(BatchResponse {
                id,
                output_data: Vec::new(),
                output_shape: Vec::new(),
                processing_time_ms: 0.0,
                status: BatchStatus::QueueFull,
                error_message: Some("Request queue is full".to_string()),
            });
        }

        // Create response channel
        let (response_sender, response_receiver) = tokio::sync::oneshot::channel();

        // Create request
        let request = BatchRequest {
            id: id.clone(),
            input_data,
            input_shape,
            priority,
            timeout_ms,
            created_at: Instant::now(),
            response_sender,
        };

        // Add request to queue
        {
            let mut queue = self.request_queue.lock().map_err(|_| anyhow!("Lock error"))?;

            if self.config.enable_prioritization {
                // Insert based on priority
                let insert_pos =
                    queue.iter().position(|req| req.priority < priority).unwrap_or(queue.len());
                queue.insert(insert_pos, request);
            } else {
                // Simple FIFO
                queue.push_back(request);
            }

            // Update queue stats
            if let Ok(mut stats) = self.stats.lock() {
                stats.current_queue_size = queue.len();
                stats.peak_queue_size = stats.peak_queue_size.max(queue.len());
            }
        }

        // Notify workers
        self.queue_condvar.notify_all();

        // Wait for response
        match response_receiver.await {
            Ok(response) => Ok(response),
            Err(_) => Ok(BatchResponse {
                id,
                output_data: Vec::new(),
                output_shape: Vec::new(),
                processing_time_ms: 0.0,
                status: BatchStatus::Error,
                error_message: Some("Internal communication error".to_string()),
            }),
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> BatchProcessorStats {
        self.stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }

    /// Shutdown the batch processor
    pub fn shutdown(&mut self) -> TrustformersResult<()> {
        // Signal shutdown
        if let Ok(mut shutdown) = self.shutdown_signal.lock() {
            *shutdown = true;
        }

        // Notify all workers
        self.queue_condvar.notify_all();

        // Wait for workers to finish
        while let Some(handle) = self.worker_handles.pop() {
            handle.join().map_err(|_| anyhow!("Failed to join worker thread"))?;
        }

        Ok(())
    }
}

// C API exports for batch processing

/// Create batch processor
#[no_mangle]
pub extern "C" fn trustformers_batch_processor_create(
    config_json: *const c_char,
    model_handle: usize,
    processor_id: *mut *mut c_char,
) -> TrustformersError {
    if config_json.is_null() || processor_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let config_str = match c_str_to_string(config_json) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let config: BatchProcessorConfig = match serde_json::from_str(&config_str) {
        Ok(config) => config,
        Err(_) => return TrustformersError::SerializationError,
    };

    let processor = match BatchProcessor::new(config, model_handle) {
        Ok(processor) => processor,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let id = format!("batch_processor_{}", model_handle);

    // Store processor in registry
    match BATCH_PROCESSOR_REGISTRY.lock() {
        Ok(mut registry) => {
            registry.insert(id.clone(), Arc::new(processor));
        },
        Err(_) => return TrustformersError::RuntimeError,
    }

    unsafe {
        *processor_id = string_to_c_str(id);
    }

    TrustformersError::Success
}

/// Get batch processor statistics
#[no_mangle]
pub extern "C" fn trustformers_batch_processor_get_stats(
    processor_id: *const c_char,
    stats_json: *mut *mut c_char,
) -> TrustformersError {
    if processor_id.is_null() || stats_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let id = match c_str_to_string(processor_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let registry = match BATCH_PROCESSOR_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let processor = match registry.get(&id) {
        Some(processor) => processor,
        None => return TrustformersError::InvalidParameter,
    };

    let stats = processor.get_stats();
    let stats_json_str = match serde_json::to_string_pretty(&stats) {
        Ok(json) => json,
        Err(_) => return TrustformersError::SerializationError,
    };

    unsafe {
        *stats_json = string_to_c_str(stats_json_str);
    }

    TrustformersError::Success
}

/// Destroy batch processor
#[no_mangle]
pub extern "C" fn trustformers_batch_processor_destroy(
    processor_id: *const c_char,
) -> TrustformersError {
    if processor_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let id = match c_str_to_string(processor_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let mut registry = match BATCH_PROCESSOR_REGISTRY.lock() {
        Ok(registry) => registry,
        Err(_) => return TrustformersError::RuntimeError,
    };

    if registry.remove(&id).is_some() {
        TrustformersError::Success
    } else {
        TrustformersError::InvalidParameter
    }
}
