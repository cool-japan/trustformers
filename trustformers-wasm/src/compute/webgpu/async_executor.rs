//! Asynchronous execution engine for GPU operations
//!
//! This module provides efficient asynchronous execution of GPU operations
//! with proper queuing, scheduling, and progress tracking.

#![allow(dead_code)]

use js_sys::Function;
use std::collections::{BTreeMap, VecDeque};
use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

// Use shared WebGPU types
use super::types::GpuDevice;

/// Priority levels for async operations
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Status of async operations
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Async operation descriptor
#[derive(Debug, Clone)]
pub struct AsyncOperation {
    pub id: usize,
    pub priority: Priority,
    pub operation_type: String,
    pub estimated_duration_ms: f64,
    pub status: ExecutionStatus,
    pub progress: f32,
    pub dependencies: Vec<usize>,
    pub callback: Option<Function>,
    pub cancelled: bool,
}

/// Result of an async operation
#[wasm_bindgen]
#[derive(Clone)]
pub struct OperationResult {
    operation_id: usize,
    status: ExecutionStatus,
    execution_time_ms: f64,
    error_message: Option<String>,
}

#[wasm_bindgen]
impl OperationResult {
    #[wasm_bindgen(getter)]
    pub fn operation_id(&self) -> usize {
        self.operation_id
    }

    #[wasm_bindgen(getter)]
    pub fn status(&self) -> ExecutionStatus {
        self.status
    }

    #[wasm_bindgen(getter)]
    pub fn execution_time_ms(&self) -> f64 {
        self.execution_time_ms
    }

    #[wasm_bindgen(getter)]
    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }
}

/// Asynchronous execution engine
#[wasm_bindgen]
pub struct AsyncExecutor {
    device: GpuDevice,
    operation_queue: VecDeque<AsyncOperation>,
    running_operations: BTreeMap<usize, AsyncOperation>,
    completed_operations: Vec<OperationResult>,
    next_operation_id: usize,
    max_concurrent_operations: usize,
    total_operations_executed: usize,
    average_execution_time: f64,
    performance_metrics: ExecutionMetrics,
}

/// Performance metrics for the async executor
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    total_operations: usize,
    successful_operations: usize,
    failed_operations: usize,
    average_queue_time_ms: f64,
    average_execution_time_ms: f64,
    peak_queue_size: usize,
    current_queue_size: usize,
}

#[wasm_bindgen]
impl ExecutionMetrics {
    #[wasm_bindgen(getter)]
    pub fn total_operations(&self) -> usize {
        self.total_operations
    }

    #[wasm_bindgen(getter)]
    pub fn successful_operations(&self) -> usize {
        self.successful_operations
    }

    #[wasm_bindgen(getter)]
    pub fn failed_operations(&self) -> usize {
        self.failed_operations
    }

    #[wasm_bindgen(getter)]
    pub fn average_execution_time_ms(&self) -> f64 {
        self.average_execution_time_ms
    }

    #[wasm_bindgen(getter)]
    pub fn current_queue_size(&self) -> usize {
        self.current_queue_size
    }
}

#[wasm_bindgen]
impl AsyncExecutor {
    /// Create a new async executor
    pub fn new(device: GpuDevice, max_concurrent: Option<usize>) -> AsyncExecutor {
        let max_concurrent_operations = max_concurrent.unwrap_or(4);

        AsyncExecutor {
            device,
            operation_queue: VecDeque::new(),
            running_operations: BTreeMap::new(),
            completed_operations: Vec::new(),
            next_operation_id: 1,
            max_concurrent_operations,
            total_operations_executed: 0,
            average_execution_time: 0.0,
            performance_metrics: ExecutionMetrics {
                total_operations: 0,
                successful_operations: 0,
                failed_operations: 0,
                average_queue_time_ms: 0.0,
                average_execution_time_ms: 0.0,
                peak_queue_size: 0,
                current_queue_size: 0,
            },
        }
    }

    /// Queue a new async operation
    pub fn queue_operation(
        &mut self,
        operation_type: String,
        priority: Priority,
        estimated_duration_ms: Option<f64>,
        dependencies: Option<Vec<usize>>,
    ) -> usize {
        let operation_id = self.next_operation_id;
        self.next_operation_id += 1;

        let operation = AsyncOperation {
            id: operation_id,
            priority,
            operation_type,
            estimated_duration_ms: estimated_duration_ms.unwrap_or(10.0),
            status: ExecutionStatus::Queued,
            progress: 0.0,
            dependencies: dependencies.unwrap_or_default(),
            callback: None,
            cancelled: false,
        };

        // Insert operation maintaining priority order
        self.insert_operation_by_priority(operation);

        // Update metrics
        self.performance_metrics.total_operations += 1;
        self.performance_metrics.current_queue_size = self.operation_queue.len();
        self.performance_metrics.peak_queue_size = self
            .performance_metrics
            .peak_queue_size
            .max(self.performance_metrics.current_queue_size);

        operation_id
    }

    /// Execute operations asynchronously
    pub async fn execute_next_operations(&mut self) -> Result<Vec<OperationResult>, JsValue> {
        let mut results = Vec::new();

        // Start new operations up to the concurrent limit
        while self.running_operations.len() < self.max_concurrent_operations
            && !self.operation_queue.is_empty()
        {
            if let Some(operation) = self.get_next_ready_operation() {
                let _operation_id = operation.id;
                self.start_operation(operation).await?;
            } else {
                break; // No ready operations (waiting for dependencies)
            }
        }

        // Check for completed operations
        let completed_ids: Vec<usize> = self.running_operations.keys().cloned().collect();
        for id in completed_ids {
            if let Some(result) = self.check_operation_completion(id).await? {
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Get current execution status
    #[wasm_bindgen(getter)]
    pub fn metrics(&self) -> ExecutionMetrics {
        self.performance_metrics.clone()
    }

    /// Get queue status
    pub fn get_queue_status(&self) -> js_sys::Object {
        let status = js_sys::Object::new();
        js_sys::Reflect::set(
            &status,
            &"queue_length".into(),
            &self.operation_queue.len().into(),
        )
        .expect("Failed to set executor status property");
        js_sys::Reflect::set(
            &status,
            &"running_operations".into(),
            &self.running_operations.len().into(),
        )
        .expect("Failed to set executor status property");
        js_sys::Reflect::set(
            &status,
            &"completed_operations".into(),
            &self.completed_operations.len().into(),
        )
        .expect("Failed to set executor status property");
        status
    }

    /// Cancel a queued or running operation
    pub fn cancel_operation(&mut self, operation_id: usize) -> bool {
        // Remove from queue if present
        let original_len = self.operation_queue.len();
        self.operation_queue.retain(|op| op.id != operation_id);

        if self.operation_queue.len() < original_len {
            self.performance_metrics.current_queue_size = self.operation_queue.len();
            return true;
        }

        // Handle cancellation of running operations
        if let Some(operation) = self.running_operations.get_mut(&operation_id) {
            operation.cancelled = true;
            operation.status = ExecutionStatus::Cancelled;

            web_sys::console::log_1(&format!("⚠️ Operation {} cancelled", operation_id).into());

            return true;
        }

        false
    }

    /// Cancel all pending and running operations
    pub fn cancel_all_operations(&mut self) -> usize {
        let mut cancelled_count = 0;

        // Cancel all queued operations
        cancelled_count += self.operation_queue.len();
        self.operation_queue.clear();

        // Cancel all running operations
        for (_, operation) in self.running_operations.iter_mut() {
            if !operation.cancelled {
                operation.cancelled = true;
                operation.status = ExecutionStatus::Cancelled;
                cancelled_count += 1;
            }
        }

        // Update metrics
        self.performance_metrics.current_queue_size = 0;

        web_sys::console::log_1(&format!("⚠️ Cancelled {} operations", cancelled_count).into());

        cancelled_count
    }

    /// Clear completed operations from history
    pub fn clear_completed(&mut self) {
        self.completed_operations.clear();
    }

    /// Wait for all operations to complete
    pub async fn wait_for_completion(&mut self) -> Result<Vec<OperationResult>, JsValue> {
        let mut all_results = Vec::new();

        while !self.operation_queue.is_empty() || !self.running_operations.is_empty() {
            let batch_results = self.execute_next_operations().await?;
            all_results.extend(batch_results);

            // Small delay to prevent busy waiting
            let promise = js_sys::Promise::resolve(&JsValue::from(1));
            JsFuture::from(promise).await?;
        }

        Ok(all_results)
    }
}

// Private implementation methods
impl AsyncExecutor {
    /// Insert operation maintaining priority order
    fn insert_operation_by_priority(&mut self, operation: AsyncOperation) {
        let mut insert_index = self.operation_queue.len();

        // Find insertion point to maintain priority order
        for (i, existing) in self.operation_queue.iter().enumerate() {
            if operation.priority > existing.priority {
                insert_index = i;
                break;
            }
        }

        self.operation_queue.insert(insert_index, operation);
    }

    /// Get next operation that's ready to execute (dependencies satisfied)
    fn get_next_ready_operation(&mut self) -> Option<AsyncOperation> {
        for i in 0..self.operation_queue.len() {
            let operation = &self.operation_queue[i];
            if self.are_dependencies_satisfied(&operation.dependencies) {
                return self.operation_queue.remove(i);
            }
        }
        None
    }

    /// Check if operation dependencies are satisfied
    fn are_dependencies_satisfied(&self, dependencies: &[usize]) -> bool {
        for &dep_id in dependencies {
            // Check if dependency is completed
            let completed = self.completed_operations.iter().any(|result| {
                result.operation_id == dep_id && result.status == ExecutionStatus::Completed
            });

            if !completed {
                return false;
            }
        }
        true
    }

    /// Start executing an operation
    async fn start_operation(&mut self, mut operation: AsyncOperation) -> Result<(), JsValue> {
        operation.status = ExecutionStatus::Running;
        let operation_id = operation.id;

        // Record start time for metrics
        let _start_time = js_sys::Date::now();

        // Add to running operations
        self.running_operations.insert(operation_id, operation);

        // Update queue metrics
        self.performance_metrics.current_queue_size = self.operation_queue.len();

        Ok(())
    }

    /// Check if an operation has completed
    async fn check_operation_completion(
        &mut self,
        operation_id: usize,
    ) -> Result<Option<OperationResult>, JsValue> {
        let start_time = js_sys::Date::now();

        if let Some(operation) = self.running_operations.get_mut(&operation_id) {
            // Check if operation has been cancelled
            if operation.cancelled {
                let execution_time = start_time - operation.estimated_duration_ms;

                let result = OperationResult {
                    operation_id,
                    status: ExecutionStatus::Cancelled,
                    execution_time_ms: execution_time,
                    error_message: Some("Operation was cancelled".to_string()),
                };

                // Remove from running operations
                self.running_operations.remove(&operation_id);

                // Update metrics (cancelled operations don't count as successful)
                self.performance_metrics.failed_operations += 1;
                self.update_average_execution_time(execution_time);

                // Add to completed operations
                self.completed_operations.push(result.clone());

                return Ok(Some(result));
            }

            // Check if operation has been running long enough
            let elapsed_time = start_time - operation.estimated_duration_ms;

            // For GPU operations, we'd check command buffer completion
            // For now, use a more realistic completion check
            let completion_probability = (elapsed_time / operation.estimated_duration_ms).min(1.0);

            // Update progress based on time
            operation.progress = completion_probability as f32;

            if completion_probability >= 1.0 {
                // Operation completed
                let execution_time = elapsed_time;

                let result = OperationResult {
                    operation_id,
                    status: ExecutionStatus::Completed,
                    execution_time_ms: execution_time,
                    error_message: None,
                };

                // Remove from running operations
                self.running_operations.remove(&operation_id);

                // Update metrics
                self.performance_metrics.successful_operations += 1;
                self.update_average_execution_time(execution_time);

                // Add to completed operations
                self.completed_operations.push(result.clone());

                return Ok(Some(result));
            }
        }

        Ok(None)
    }

    /// Update average execution time metric
    fn update_average_execution_time(&mut self, execution_time: f64) {
        let total_ops = self.total_operations_executed as f64;
        self.average_execution_time =
            (self.average_execution_time * total_ops + execution_time) / (total_ops + 1.0);
        self.total_operations_executed += 1;
        self.performance_metrics.average_execution_time_ms = self.average_execution_time;
    }
}

/// Utility function to create async executor
#[wasm_bindgen]
pub fn create_async_executor(device: GpuDevice, max_concurrent: Option<usize>) -> AsyncExecutor {
    AsyncExecutor::new(device, max_concurrent)
}

/// Promise-based operation execution
#[wasm_bindgen]
pub async fn execute_operation_async(
    executor: &mut AsyncExecutor,
    operation_type: String,
    priority: Priority,
) -> Result<OperationResult, JsValue> {
    let operation_id = executor.queue_operation(operation_type, priority, None, None);

    // Wait for this specific operation to complete
    loop {
        let results = executor.execute_next_operations().await?;

        for result in results {
            if result.operation_id == operation_id {
                return Ok(result);
            }
        }

        // Small delay before checking again
        let promise = js_sys::Promise::resolve(&JsValue::from(1));
        JsFuture::from(promise).await?;
    }
}

/// Progress tracking for long operations
#[wasm_bindgen]
pub struct ProgressTracker {
    operation_id: usize,
    executor: *mut AsyncExecutor, // Raw pointer for interior mutability
}

#[wasm_bindgen]
impl ProgressTracker {
    /// Get current progress (0.0 to 1.0)
    pub fn get_progress(&self) -> f32 {
        // In a real implementation, this would safely access the executor
        // For now, return a simulated progress
        0.5
    }

    /// Check if operation is completed
    pub fn is_completed(&self) -> bool {
        self.get_progress() >= 1.0
    }

    /// Get estimated remaining time in milliseconds
    pub fn estimated_remaining_ms(&self) -> f64 {
        let progress = self.get_progress();
        if progress <= 0.0 {
            return 1000.0; // Default estimate
        }

        let elapsed_estimate = 100.0; // Simulated elapsed time
        (elapsed_estimate / progress as f64) * (1.0 - progress as f64)
    }
}
