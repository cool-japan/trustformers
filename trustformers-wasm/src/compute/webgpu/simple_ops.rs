//! Enhanced WebGPU operations with shared memory optimization

#![allow(dead_code)]

use super::types::{GpuComputePipeline, GpuDevice, GpuDeviceExt};
use crate::core::tensor::WasmTensor;
use crate::webgpu::shaders::*;
use std::collections::BTreeMap;
use std::string::String;
use wasm_bindgen::prelude::*;

/// Enhanced WebGPU tensor operations with shared memory optimization
pub struct SimpleGpuOps {
    device: GpuDevice,
    pipelines: BTreeMap<String, GpuComputePipeline>,
    shared_memory_threshold: usize,
}

impl SimpleGpuOps {
    pub fn new(device: GpuDevice) -> Self {
        SimpleGpuOps {
            device,
            pipelines: BTreeMap::new(),
            shared_memory_threshold: 1024, // Elements threshold for using shared memory
        }
    }

    /// Create a compute shader pipeline
    fn create_pipeline(
        &self,
        shader_code: &str,
        label: &str,
    ) -> Result<GpuComputePipeline, JsValue> {
        let shader_desc = super::types::create_shader_module_descriptor(shader_code, Some(label))?;

        let shader_module = self.device.create_shader_module(&shader_desc);

        let compute_stage = super::types::create_programmable_stage(&shader_module, "main")?;
        let pipeline_desc =
            super::types::create_compute_pipeline_descriptor(&compute_stage, Some(label))?;

        Ok(self.device.create_compute_pipeline(&pipeline_desc))
    }

    /// Get or create a pipeline for the given operation
    fn get_pipeline(
        &mut self,
        operation: &str,
        use_shared_memory: bool,
    ) -> Result<&GpuComputePipeline, JsValue> {
        let key = if use_shared_memory {
            format!("{}_shared", operation)
        } else {
            operation.to_string()
        };

        if !self.pipelines.contains_key(&key) {
            let shader_code = match operation {
                "matmul" => {
                    if use_shared_memory {
                        MATMUL_SHARED_SHADER
                    } else {
                        MATMUL_SHADER
                    }
                },
                "softmax" => {
                    if use_shared_memory {
                        SOFTMAX_SHARED_SHADER
                    } else {
                        SOFTMAX_SHADER
                    }
                },
                "layer_norm" => {
                    if use_shared_memory {
                        LAYER_NORM_SHARED_SHADER
                    } else {
                        LAYER_NORM_SHADER
                    }
                },
                "attention" => {
                    if use_shared_memory {
                        ATTENTION_SHARED_SHADER
                    } else {
                        ATTENTION_SHADER
                    }
                },
                "add" => ADD_SHADER,
                "mul" => MUL_SHADER,
                "relu" => RELU_SHADER,
                "gelu" => GELU_SHADER,
                _ => {
                    return Err(JsValue::from_str(&format!(
                        "Unknown operation: {}",
                        operation
                    )))
                },
            };

            let pipeline = self.create_pipeline(shader_code, &key)?;
            self.pipelines.insert(key.clone(), pipeline);
        }

        Ok(self.pipelines.get(&key).expect("pipeline just inserted with key"))
    }

    /// Determine if shared memory should be used based on tensor dimensions
    fn should_use_shared_memory(&self, dimensions: &[usize]) -> bool {
        let total_elements: usize = dimensions.iter().product();
        total_elements >= self.shared_memory_threshold
    }

    /// Enhanced matrix multiplication with automatic shared memory selection
    pub async fn matmul(&mut self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(JsValue::from_str(
                "Matrix multiplication requires 2D tensors",
            ));
        }

        let use_shared = self.should_use_shared_memory(&[a_shape[0], a_shape[1], b_shape[1]]);

        // For now, fall back to CPU implementation but track the decision
        if use_shared {
            web_sys::console::log_1(&JsValue::from_str("Would use shared memory matmul"));
        } else {
            web_sys::console::log_1(&JsValue::from_str("Would use basic matmul"));
        }

        a.matmul(b)
    }

    /// Enhanced softmax with shared memory optimization
    pub async fn softmax(&mut self, input: &WasmTensor, dim: usize) -> Result<WasmTensor, JsValue> {
        let shape = input.shape();
        let use_shared = self.should_use_shared_memory(&shape);

        if use_shared {
            web_sys::console::log_1(&JsValue::from_str("Would use shared memory softmax"));
        } else {
            web_sys::console::log_1(&JsValue::from_str("Would use basic softmax"));
        }

        // Fall back to CPU implementation for now
        input.softmax(dim as i32)
    }

    /// Enhanced layer normalization with shared memory optimization
    pub async fn layer_norm(
        &mut self,
        input: &WasmTensor,
        normalized_shape: &[usize],
        eps: f32,
    ) -> Result<WasmTensor, JsValue> {
        let shape = input.shape();
        let use_shared = self.should_use_shared_memory(&shape);

        if use_shared {
            web_sys::console::log_1(&JsValue::from_str("Would use shared memory layer_norm"));
        } else {
            web_sys::console::log_1(&JsValue::from_str("Would use basic layer_norm"));
        }

        // Fall back to CPU implementation for now
        input.layer_norm(normalized_shape, eps)
    }

    /// Enhanced attention mechanism with shared memory optimization
    pub async fn attention(
        &mut self,
        query: &WasmTensor,
        key: &WasmTensor,
        value: &WasmTensor,
    ) -> Result<WasmTensor, JsValue> {
        let q_shape = query.shape();
        let use_shared = self.should_use_shared_memory(&q_shape);

        if use_shared {
            web_sys::console::log_1(&JsValue::from_str("Would use shared memory attention"));
        } else {
            web_sys::console::log_1(&JsValue::from_str("Would use basic attention"));
        }

        // Fall back to CPU implementation for now
        query.scaled_dot_product_attention(key, value, None)
    }

    /// Element-wise addition (no shared memory needed)
    pub async fn add(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, JsValue> {
        a.add(b)
    }

    /// ReLU activation (no shared memory needed)
    pub async fn relu(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        Ok(input.relu())
    }

    /// Get device info with shared memory capabilities
    pub fn device_info(&self) -> String {
        format!(
            "WebGPU Device with {} cached pipelines, shared memory threshold: {} elements",
            self.pipelines.len(),
            self.shared_memory_threshold
        )
    }

    /// Set the threshold for when to use shared memory optimizations
    pub fn set_shared_memory_threshold(&mut self, threshold: usize) {
        self.shared_memory_threshold = threshold;
    }

    /// Get the current shared memory threshold
    pub fn get_shared_memory_threshold(&self) -> usize {
        self.shared_memory_threshold
    }

    /// Clear cached pipelines
    pub fn clear_pipeline_cache(&mut self) {
        self.pipelines.clear();
    }
}
