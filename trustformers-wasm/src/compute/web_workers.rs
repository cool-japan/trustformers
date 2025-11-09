//! Web Workers integration for background tensor processing
//!
//! This module provides seamless integration with Web Workers to offload
//! heavy tensor computations to background threads, keeping the main thread responsive.

#![allow(dead_code)]

use crate::core::tensor::WasmTensor;
use js_sys::{Function, Object, Promise};
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use std::boxed::Box;
use std::collections::BTreeMap;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{BroadcastChannel, MessageEvent, Worker, WorkerOptions, WorkerType};

/// Types of tasks that can be offloaded to Web Workers
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WorkerTaskType {
    /// Matrix multiplication
    MatrixMultiplication,
    /// Tensor element-wise operations
    ElementWise,
    /// Convolution operations
    Convolution,
    /// Attention mechanism
    Attention,
    /// Tokenization
    Tokenization,
    /// Model inference
    Inference,
}

/// Priority levels for worker tasks
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum WorkerPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Worker task configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerTask {
    pub id: usize,
    pub task_type: WorkerTaskType,
    pub priority: WorkerPriority,
    pub data: WorkerTaskData,
    pub timeout_ms: Option<u32>,
    pub retry_count: u32,
}

/// Data payload for worker tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerTaskData {
    MatMul {
        a_data: Vec<f32>,
        a_shape: Vec<usize>,
        b_data: Vec<f32>,
        b_shape: Vec<usize>,
    },
    ElementWise {
        operation: String,
        inputs: Vec<TensorData>,
        params: BTreeMap<String, f32>,
    },
    Inference {
        model_data: Vec<u8>,
        input_data: Vec<f32>,
        input_shape: Vec<usize>,
    },
    Tokenization {
        text: String,
        vocab_data: Vec<u8>,
        max_length: Option<usize>,
    },
}

/// Simplified tensor data for worker communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Result of a worker task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerTaskResult {
    pub task_id: usize,
    pub success: bool,
    pub result_data: Option<TensorData>,
    pub error_message: Option<String>,
    pub execution_time_ms: f64,
}

/// Worker pool manager
#[wasm_bindgen]
pub struct WorkerPool {
    workers: Vec<WorkerInstance>,
    task_queue: Vec<WorkerTask>,
    pending_tasks: BTreeMap<usize, WorkerTask>,
    next_task_id: usize,
    max_workers: usize,
    worker_script_url: String,
}

/// Individual worker instance
struct WorkerInstance {
    worker: Worker,
    id: usize,
    is_busy: bool,
    current_task_id: Option<usize>,
    total_tasks_completed: usize,
    average_task_time_ms: f64,
}

#[wasm_bindgen]
impl WorkerPool {
    /// Create a new worker pool
    #[wasm_bindgen(constructor)]
    pub fn new(max_workers: usize, worker_script_url: String) -> Result<WorkerPool, JsValue> {
        if max_workers == 0 {
            return Err(JsValue::from_str("Worker pool must have at least 1 worker"));
        }

        let mut pool = WorkerPool {
            workers: Vec::new(),
            task_queue: Vec::new(),
            pending_tasks: BTreeMap::new(),
            next_task_id: 1,
            max_workers,
            worker_script_url,
        };

        // Initialize workers
        for i in 0..max_workers {
            pool.create_worker(i)?;
        }

        Ok(pool)
    }

    /// Submit a task to the worker pool
    pub fn submit_task(
        &mut self,
        task_type: WorkerTaskType,
        priority: WorkerPriority,
        data: JsValue,
        timeout_ms: Option<u32>,
    ) -> Result<usize, JsValue> {
        let task_id = self.next_task_id;
        self.next_task_id += 1;

        // Convert JsValue to WorkerTaskData
        let task_data: WorkerTaskData = from_value(data)?;

        let task = WorkerTask {
            id: task_id,
            task_type,
            priority,
            data: task_data,
            timeout_ms,
            retry_count: 0,
        };

        // Insert task maintaining priority order
        self.insert_task_by_priority(task);

        Ok(task_id)
    }

    /// Process pending tasks and assign to available workers
    pub async fn process_tasks(&mut self) -> Result<js_sys::Array, JsValue> {
        let mut results = Vec::new();

        // Assign tasks to available workers
        while let Some(available_worker_id) = self.find_available_worker() {
            if let Some(task) = self.get_next_task() {
                self.assign_task_to_worker(available_worker_id, task).await?;
            } else {
                break; // No more tasks
            }
        }

        // Check for completed tasks
        results.extend(self.collect_completed_tasks().await?);

        // Convert Vec to js_sys::Array
        let js_results = js_sys::Array::new();
        for result in results {
            let js_result = to_value(&result)?;
            js_results.push(&js_result);
        }
        Ok(js_results)
    }

    /// Wait for a specific task to complete
    pub async fn wait_for_task(&mut self, task_id: usize) -> Result<JsValue, JsValue> {
        #[allow(clippy::excessive_nesting)]
        loop {
            let results_array = self.process_tasks().await?;

            // Convert js_sys::Array back to iterate
            for i in 0..results_array.length() {
                let result_js = results_array.get(i);
                if let Ok(result) = from_value::<WorkerTaskResult>(result_js) {
                    if result.task_id == task_id {
                        return Ok(to_value(&result)?);
                    }
                }
            }

            // Check if task is still pending or in queue
            if !self.pending_tasks.contains_key(&task_id)
                && !self.task_queue.iter().any(|t| t.id == task_id)
            {
                return Err(JsValue::from_str("Task not found"));
            }

            // Small delay before checking again
            let promise = Promise::resolve(&JsValue::from(10));
            JsFuture::from(promise).await?;
        }
    }

    /// Get worker pool statistics
    pub fn get_stats(&self) -> js_sys::Object {
        let stats = Object::new();

        js_sys::Reflect::set(&stats, &"total_workers".into(), &self.workers.len().into()).unwrap();
        js_sys::Reflect::set(
            &stats,
            &"busy_workers".into(),
            &self.count_busy_workers().into(),
        )
        .unwrap();
        js_sys::Reflect::set(
            &stats,
            &"queued_tasks".into(),
            &self.task_queue.len().into(),
        )
        .unwrap();
        js_sys::Reflect::set(
            &stats,
            &"pending_tasks".into(),
            &self.pending_tasks.len().into(),
        )
        .unwrap();

        let total_completed: usize = self.workers.iter().map(|w| w.total_tasks_completed).sum();
        js_sys::Reflect::set(
            &stats,
            &"total_completed_tasks".into(),
            &total_completed.into(),
        )
        .unwrap();

        stats
    }

    /// Terminate all workers and clean up
    pub fn terminate(&mut self) {
        for worker_instance in &self.workers {
            worker_instance.worker.terminate();
        }
        self.workers.clear();
        self.task_queue.clear();
        self.pending_tasks.clear();
    }
}

// Private implementation methods
impl WorkerPool {
    /// Create a new worker instance
    fn create_worker(&mut self, worker_id: usize) -> Result<(), JsValue> {
        let worker_options = WorkerOptions::new();
        worker_options.set_type(WorkerType::Module);

        let worker = Worker::new_with_options(&self.worker_script_url, &worker_options)?;

        // Set up message handler
        let _worker_clone = worker.clone();
        let closure = Closure::wrap(Box::new(move |event: MessageEvent| {
            // Handle worker messages
            web_sys::console::log_2(&"Worker message:".into(), &event.data());
        }) as Box<dyn FnMut(_)>);

        worker.set_onmessage(Some(closure.as_ref().unchecked_ref()));
        closure.forget(); // Prevent cleanup

        let worker_instance = WorkerInstance {
            worker,
            id: worker_id,
            is_busy: false,
            current_task_id: None,
            total_tasks_completed: 0,
            average_task_time_ms: 0.0,
        };

        self.workers.push(worker_instance);
        Ok(())
    }

    /// Insert task maintaining priority order
    fn insert_task_by_priority(&mut self, task: WorkerTask) {
        let mut insert_index = self.task_queue.len();

        for (i, existing) in self.task_queue.iter().enumerate() {
            if task.priority > existing.priority {
                insert_index = i;
                break;
            }
        }

        self.task_queue.insert(insert_index, task);
    }

    /// Find an available worker
    fn find_available_worker(&self) -> Option<usize> {
        self.workers
            .iter()
            .enumerate()
            .find(|(_, worker)| !worker.is_busy)
            .map(|(i, _)| i)
    }

    /// Get the next task from the queue
    fn get_next_task(&mut self) -> Option<WorkerTask> {
        self.task_queue.pop()
    }

    /// Assign a task to a specific worker
    async fn assign_task_to_worker(
        &mut self,
        worker_id: usize,
        task: WorkerTask,
    ) -> Result<(), JsValue> {
        if worker_id >= self.workers.len() {
            return Err(JsValue::from_str("Invalid worker ID"));
        }

        let task_id = task.id;

        // Mark worker as busy
        self.workers[worker_id].is_busy = true;
        self.workers[worker_id].current_task_id = Some(task_id);

        // Add to pending tasks
        self.pending_tasks.insert(task_id, task.clone());

        // Send task to worker
        let task_message = to_value(&task)?;
        self.workers[worker_id].worker.post_message(&task_message)?;

        Ok(())
    }

    /// Collect completed tasks from workers
    async fn collect_completed_tasks(&mut self) -> Result<Vec<WorkerTaskResult>, JsValue> {
        let mut results = Vec::new();

        // Check for messages from workers
        #[allow(clippy::excessive_nesting)]
        for worker_instance in &mut self.workers {
            if let Some(task_id) = worker_instance.current_task_id {
                // Check if worker has completed the task
                if let Some(_task) = self.pending_tasks.get(&task_id) {
                    // Create a promise to check for worker message
                    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
                        let worker_clone = worker_instance.worker.clone();
                        let closure = Closure::wrap(Box::new(move |event: MessageEvent| {
                            resolve.call1(&JsValue::UNDEFINED, &event.data()).unwrap();
                        })
                            as Box<dyn FnMut(_)>);

                        worker_clone.set_onmessage(Some(closure.as_ref().unchecked_ref()));
                        closure.forget();
                    });

                    // Try to get result without blocking
                    let result_data = match wasm_bindgen_futures::JsFuture::from(promise).await {
                        Ok(data) => data,
                        Err(_) => JsValue::NULL,
                    };

                    {
                        if !result_data.is_null() {
                            // Parse the result
                            if let Ok(worker_result) = from_value::<WorkerTaskResult>(result_data) {
                                // Remove from pending tasks
                                self.pending_tasks.remove(&task_id);

                                // Update worker status
                                worker_instance.is_busy = false;
                                worker_instance.current_task_id = None;
                                worker_instance.total_tasks_completed += 1;

                                // Update average execution time
                                let current_avg = worker_instance.average_task_time_ms;
                                let completed_tasks = worker_instance.total_tasks_completed;
                                worker_instance.average_task_time_ms = (current_avg
                                    * (completed_tasks - 1) as f64
                                    + worker_result.execution_time_ms)
                                    / completed_tasks as f64;

                                results.push(worker_result);
                            }
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Count busy workers
    fn count_busy_workers(&self) -> usize {
        self.workers.iter().filter(|w| w.is_busy).count()
    }
}

/// Utility functions for creating worker tasks
/// Create a matrix multiplication task
#[wasm_bindgen]
pub fn create_matmul_task(a: &WasmTensor, b: &WasmTensor, _priority: WorkerPriority) -> JsValue {
    let task_data = WorkerTaskData::MatMul {
        a_data: a.data().clone(),
        a_shape: a.shape().clone(),
        b_data: b.data().clone(),
        b_shape: b.shape().clone(),
    };

    to_value(&task_data).unwrap_or(JsValue::NULL)
}

/// Create an element-wise operation task
#[wasm_bindgen]
pub fn create_elementwise_task(
    operation: String,
    tensor_data: &js_sys::Array,
    _priority: WorkerPriority,
) -> JsValue {
    let mut tensor_inputs = Vec::new();

    // Expect the input to be an array of serialized tensor data
    for i in 0..tensor_data.length() {
        let tensor_js = tensor_data.get(i);
        if let Ok(tensor_data_obj) = from_value::<TensorData>(tensor_js) {
            tensor_inputs.push(tensor_data_obj);
        }
    }

    let task_data = WorkerTaskData::ElementWise {
        operation,
        inputs: tensor_inputs,
        params: BTreeMap::new(),
    };

    to_value(&task_data).unwrap_or(JsValue::NULL)
}

/// Broadcast channel for worker coordination
#[wasm_bindgen]
pub struct WorkerCoordinator {
    channel: BroadcastChannel,
    worker_count: usize,
}

#[wasm_bindgen]
impl WorkerCoordinator {
    /// Create a new worker coordinator
    #[wasm_bindgen(constructor)]
    pub fn new(channel_name: String) -> Result<WorkerCoordinator, JsValue> {
        let channel = BroadcastChannel::new(&channel_name)?;

        Ok(WorkerCoordinator {
            channel,
            worker_count: 0,
        })
    }

    /// Broadcast a message to all workers
    pub fn broadcast(&self, message: &JsValue) -> Result<(), JsValue> {
        self.channel.post_message(message)
    }

    /// Set up message handler
    pub fn set_message_handler(&self, handler: &Function) {
        self.channel.set_onmessage(Some(handler));
    }

    /// Close the coordination channel
    pub fn close(&self) {
        self.channel.close();
    }
}

/// Check if Web Workers are supported
#[wasm_bindgen]
pub fn is_web_workers_supported() -> bool {
    web_sys::window()
        .and_then(|w| js_sys::Reflect::get(&w, &JsValue::from_str("Worker")).ok())
        .map(|v| !v.is_undefined())
        .unwrap_or(false)
}

/// Get optimal number of workers based on hardware
#[wasm_bindgen]
pub fn get_optimal_worker_count() -> usize {
    // Try to get navigator.hardwareConcurrency
    let worker_count = web_sys::window()
        .map(|w| w.navigator().hardware_concurrency() as usize)
        .unwrap_or(4);

    // Cap at reasonable limits
    worker_count.clamp(1, 8)
}

/// Initialize web workers subsystem
pub fn initialize() -> Result<(), crate::compute::ComputeError> {
    // Placeholder - actual worker pool is created when needed
    Ok(())
}
