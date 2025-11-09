//! WebGPU-accelerated tensor operations

#![allow(dead_code)]

use super::shaders;
use super::types::{
    buffer_usage, GpuBindGroup, GpuBuffer, GpuCommandEncoderExt, GpuComputePassEncoderExt,
    GpuComputePipeline, GpuComputePipelineExt, GpuDevice, GpuDeviceExt, GpuQueueExt,
};
use super::WebGPUBackend;
use crate::core::tensor::WasmTensor;
use std::vec;
use wasm_bindgen::prelude::*;

/// GPU-accelerated tensor operations
pub struct TensorOps {
    device: GpuDevice,
    matmul_pipeline: Option<GpuComputePipeline>,
    add_pipeline: Option<GpuComputePipeline>,
    mul_pipeline: Option<GpuComputePipeline>,
    relu_pipeline: Option<GpuComputePipeline>,
    gelu_pipeline: Option<GpuComputePipeline>,
    softmax_pipeline: Option<GpuComputePipeline>,
    layer_norm_pipeline: Option<GpuComputePipeline>,
    attention_pipeline: Option<GpuComputePipeline>,
}

impl TensorOps {
    /// Create new tensor operations handler
    pub fn new(device: GpuDevice) -> Self {
        TensorOps {
            device,
            matmul_pipeline: None,
            add_pipeline: None,
            mul_pipeline: None,
            relu_pipeline: None,
            gelu_pipeline: None,
            softmax_pipeline: None,
            layer_norm_pipeline: None,
            attention_pipeline: None,
        }
    }

    /// Initialize compute pipelines
    pub fn init_pipelines(&mut self) -> Result<(), JsValue> {
        self.matmul_pipeline =
            Some(self.create_compute_pipeline(shaders::MATMUL_SHADER, "matmul")?);

        self.add_pipeline = Some(self.create_compute_pipeline(shaders::ADD_SHADER, "add")?);

        self.mul_pipeline = Some(self.create_compute_pipeline(shaders::MUL_SHADER, "mul")?);

        self.relu_pipeline = Some(self.create_compute_pipeline(shaders::RELU_SHADER, "relu")?);

        self.gelu_pipeline = Some(self.create_compute_pipeline(shaders::GELU_SHADER, "gelu")?);

        self.softmax_pipeline =
            Some(self.create_compute_pipeline(shaders::SOFTMAX_SHADER, "softmax")?);

        self.layer_norm_pipeline =
            Some(self.create_compute_pipeline(shaders::LAYER_NORM_SHADER, "layer_norm")?);

        self.attention_pipeline =
            Some(self.create_compute_pipeline(shaders::ATTENTION_SHADER, "attention")?);

        Ok(())
    }

    /// Create a compute pipeline from shader code
    fn create_compute_pipeline(
        &self,
        shader_code: &str,
        label: &str,
    ) -> Result<GpuComputePipeline, JsValue> {
        // Create shader module
        let shader_descriptor =
            super::types::create_shader_module_descriptor(shader_code, Some(label))?;

        let shader_module = self.device.create_shader_module(&shader_descriptor);

        // Create compute stage and pipeline using helper functions
        let compute_stage = super::types::create_programmable_stage(&shader_module, "main")?;
        let pipeline_descriptor =
            super::types::create_compute_pipeline_descriptor(&compute_stage, Some(label))?;

        Ok(self.device.create_compute_pipeline(&pipeline_descriptor))
    }

    /// Matrix multiplication on GPU
    pub async fn matmul_gpu(
        &self,
        backend: &mut WebGPUBackend,
        a: &WasmTensor,
        b: &WasmTensor,
    ) -> Result<WasmTensor, JsValue> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(JsValue::from_str(
                "Matrix multiplication requires 2D tensors",
            ));
        }

        if a_shape[1] != b_shape[0] {
            return Err(JsValue::from_str(
                "Invalid dimensions for matrix multiplication",
            ));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        // Create GPU buffers
        let a_buffer =
            backend.create_buffer(&a.data(), buffer_usage::STORAGE | buffer_usage::COPY_DST)?;

        let b_buffer =
            backend.create_buffer(&b.data(), buffer_usage::STORAGE | buffer_usage::COPY_DST)?;

        let result_size = m * n;
        let result_buffer = backend.create_buffer(
            &vec![0.0f32; result_size],
            buffer_usage::STORAGE | buffer_usage::COPY_SRC,
        )?;

        // Create uniform buffer for dimensions
        let dims = vec![m as f32, n as f32, k as f32];
        let dims_buffer =
            backend.create_buffer(&dims, buffer_usage::UNIFORM | buffer_usage::COPY_DST)?;

        // Create command encoder and compute pass
        let encoder = self.device.create_command_encoder();
        let pass = encoder.begin_compute_pass();

        if let Some(pipeline) = &self.matmul_pipeline {
            pass.set_pipeline(pipeline);

            // Create bind group using pipeline layout
            let bind_group = self.create_matmul_bind_group(
                pipeline,
                &a_buffer,
                &b_buffer,
                &result_buffer,
                &dims_buffer,
            )?;
            pass.set_bind_group(0, &bind_group);

            // Dispatch compute shader
            let workgroup_x = (m + 7) / 8;
            let workgroup_y = (n + 7) / 8;
            pass.dispatch_workgroups(workgroup_x as u32, workgroup_y as u32, 1);
        }

        pass.end();

        // Submit commands
        let command_buffer = encoder.finish();
        self.device.queue().submit(&js_sys::Array::of1(&command_buffer));

        // Read back results
        let result_data = backend.read_buffer(&result_buffer, result_size).await?;

        WasmTensor::new(result_data, vec![m, n])
    }

    /// Element-wise addition on GPU
    pub async fn add_gpu(
        &self,
        backend: &mut WebGPUBackend,
        a: &WasmTensor,
        b: &WasmTensor,
    ) -> Result<WasmTensor, JsValue> {
        if a.shape() != b.shape() {
            return Err(JsValue::from_str("Shape mismatch for addition"));
        }

        let size = a.data().len();

        // Create GPU buffers
        let a_buffer =
            backend.create_buffer(&a.data(), buffer_usage::STORAGE | buffer_usage::COPY_DST)?;

        let b_buffer =
            backend.create_buffer(&b.data(), buffer_usage::STORAGE | buffer_usage::COPY_DST)?;

        let result_buffer = backend.create_buffer(
            &vec![0.0f32; size],
            buffer_usage::STORAGE | buffer_usage::COPY_SRC,
        )?;

        let size_buffer = backend.create_buffer(
            &[size as f32],
            buffer_usage::UNIFORM | buffer_usage::COPY_DST,
        )?;

        // Create bind group and run compute shader
        if let Some(pipeline) = &self.add_pipeline {
            let bind_group = self.create_binary_op_bind_group(
                pipeline,
                &a_buffer,
                &b_buffer,
                &result_buffer,
                &size_buffer,
            )?;

            let encoder = self.device.create_command_encoder();
            let pass = encoder.begin_compute_pass();
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group);
            pass.dispatch_workgroups(((size + 255) / 256) as u32, 1, 1);
            pass.end();

            let command_buffer = encoder.finish();
            self.device.queue().submit(&js_sys::Array::of1(&command_buffer));
        }

        // Read back results
        let result_data = backend.read_buffer(&result_buffer, size).await?;

        WasmTensor::new(result_data, a.shape())
    }

    /// ReLU activation on GPU
    pub async fn relu_gpu(
        &self,
        backend: &mut WebGPUBackend,
        input: &WasmTensor,
    ) -> Result<WasmTensor, JsValue> {
        let size = input.data().len();

        // Create GPU buffers
        let input_buffer = backend.create_buffer(
            &input.data(),
            buffer_usage::STORAGE | buffer_usage::COPY_DST,
        )?;

        let output_buffer = backend.create_buffer(
            &vec![0.0f32; size],
            buffer_usage::STORAGE | buffer_usage::COPY_SRC,
        )?;

        let size_buffer = backend.create_buffer(
            &[size as f32],
            buffer_usage::UNIFORM | buffer_usage::COPY_DST,
        )?;

        // Create bind group and run compute shader
        if let Some(pipeline) = &self.relu_pipeline {
            let bind_group = self.create_activation_bind_group(
                pipeline,
                &input_buffer,
                &output_buffer,
                &size_buffer,
            )?;

            let encoder = self.device.create_command_encoder();
            let pass = encoder.begin_compute_pass();
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group);
            pass.dispatch_workgroups(((size + 255) / 256) as u32, 1, 1);
            pass.end();

            let command_buffer = encoder.finish();
            self.device.queue().submit(&js_sys::Array::of1(&command_buffer));
        }

        // Read back results
        let result_data = backend.read_buffer(&output_buffer, size).await?;

        WasmTensor::new(result_data, input.shape())
    }

    /// GELU activation on GPU
    pub async fn gelu_gpu(
        &self,
        backend: &mut WebGPUBackend,
        input: &WasmTensor,
    ) -> Result<WasmTensor, JsValue> {
        let size = input.data().len();

        let input_buffer = backend.create_buffer(
            &input.data(),
            buffer_usage::STORAGE | buffer_usage::COPY_DST,
        )?;

        let output_buffer = backend.create_buffer(
            &vec![0.0f32; size],
            buffer_usage::STORAGE | buffer_usage::COPY_SRC,
        )?;

        let size_buffer = backend.create_buffer(
            &[size as f32],
            buffer_usage::UNIFORM | buffer_usage::COPY_DST,
        )?;

        // Create bind group and run compute shader
        if let Some(pipeline) = &self.gelu_pipeline {
            let bind_group = self.create_activation_bind_group(
                pipeline,
                &input_buffer,
                &output_buffer,
                &size_buffer,
            )?;

            let encoder = self.device.create_command_encoder();
            let pass = encoder.begin_compute_pass();
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group);
            pass.dispatch_workgroups(((size + 255) / 256) as u32, 1, 1);
            pass.end();

            let command_buffer = encoder.finish();
            self.device.queue().submit(&js_sys::Array::of1(&command_buffer));
        }

        let result_data = backend.read_buffer(&output_buffer, size).await?;
        WasmTensor::new(result_data, input.shape())
    }

    /// Softmax on GPU
    pub async fn softmax_gpu(
        &self,
        backend: &mut WebGPUBackend,
        input: &WasmTensor,
    ) -> Result<WasmTensor, JsValue> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(JsValue::from_str("Softmax requires 2D tensor"));
        }

        let batch_size = shape[0];
        let feature_size = shape[1];
        let total_size = batch_size * feature_size;

        let input_buffer = backend.create_buffer(
            &input.data(),
            buffer_usage::STORAGE | buffer_usage::COPY_DST,
        )?;

        let output_buffer = backend.create_buffer(
            &vec![0.0f32; total_size],
            buffer_usage::STORAGE | buffer_usage::COPY_SRC,
        )?;

        let dims_buffer = backend.create_buffer(
            &[batch_size as f32, feature_size as f32],
            buffer_usage::UNIFORM | buffer_usage::COPY_DST,
        )?;

        // Create bind group and run compute shader
        if let Some(pipeline) = &self.softmax_pipeline {
            let bind_group = self.create_softmax_bind_group(
                pipeline,
                &input_buffer,
                &output_buffer,
                &dims_buffer,
            )?;

            let encoder = self.device.create_command_encoder();
            let pass = encoder.begin_compute_pass();
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group);
            pass.dispatch_workgroups(batch_size as u32, 1, 1);
            pass.end();

            let command_buffer = encoder.finish();
            self.device.queue().submit(&js_sys::Array::of1(&command_buffer));
        }

        let result_data = backend.read_buffer(&output_buffer, total_size).await?;
        WasmTensor::new(result_data, shape.clone())
    }

    /// Layer normalization on GPU
    pub async fn layer_norm_gpu(
        &self,
        backend: &mut WebGPUBackend,
        input: &WasmTensor,
        gamma: &WasmTensor,
        beta: &WasmTensor,
    ) -> Result<WasmTensor, JsValue> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(JsValue::from_str("Layer norm requires 2D tensor"));
        }

        let batch_size = shape[0];
        let feature_size = shape[1];
        let total_size = batch_size * feature_size;

        if gamma.data().len() != feature_size || beta.data().len() != feature_size {
            return Err(JsValue::from_str("Gamma and beta must match feature size"));
        }

        let input_buffer = backend.create_buffer(
            &input.data(),
            buffer_usage::STORAGE | buffer_usage::COPY_DST,
        )?;

        let gamma_buffer = backend.create_buffer(
            &gamma.data(),
            buffer_usage::STORAGE | buffer_usage::COPY_DST,
        )?;

        let beta_buffer =
            backend.create_buffer(&beta.data(), buffer_usage::STORAGE | buffer_usage::COPY_DST)?;

        let output_buffer = backend.create_buffer(
            &vec![0.0f32; total_size],
            buffer_usage::STORAGE | buffer_usage::COPY_SRC,
        )?;

        let dims_buffer = backend.create_buffer(
            &[batch_size as f32, feature_size as f32],
            buffer_usage::UNIFORM | buffer_usage::COPY_DST,
        )?;

        // Create bind group and run compute shader
        if let Some(pipeline) = &self.layer_norm_pipeline {
            let bind_group = self.create_layer_norm_bind_group(
                pipeline,
                &input_buffer,
                &gamma_buffer,
                &beta_buffer,
                &output_buffer,
                &dims_buffer,
            )?;

            let encoder = self.device.create_command_encoder();
            let pass = encoder.begin_compute_pass();
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group);
            pass.dispatch_workgroups(batch_size as u32, 1, 1);
            pass.end();

            let command_buffer = encoder.finish();
            self.device.queue().submit(&js_sys::Array::of1(&command_buffer));
        }

        let result_data = backend.read_buffer(&output_buffer, total_size).await?;
        WasmTensor::new(result_data, shape.clone())
    }

    /// Helper to create bind group for matrix multiplication
    fn create_matmul_bind_group(
        &self,
        pipeline: &GpuComputePipeline,
        a_buffer: &GpuBuffer,
        b_buffer: &GpuBuffer,
        result_buffer: &GpuBuffer,
        dims_buffer: &GpuBuffer,
    ) -> Result<GpuBindGroup, JsValue> {
        // Get bind group layout from pipeline
        let layout = pipeline.get_bind_group_layout(0);

        let entries = js_sys::Array::new();

        let entry0 = super::types::create_bind_group_entry(0, &a_buffer.into())?;
        entries.push(&entry0);

        let entry1 = super::types::create_bind_group_entry(1, &b_buffer.into())?;
        entries.push(&entry1);

        let entry2 = super::types::create_bind_group_entry(2, &result_buffer.into())?;
        entries.push(&entry2);

        let entry3 = super::types::create_bind_group_entry(3, &dims_buffer.into())?;
        entries.push(&entry3);

        let descriptor = super::types::create_bind_group_descriptor(&layout, &entries)?;
        Ok(self.device.create_bind_group(&descriptor))
    }

    /// Helper to create bind group for binary operations
    fn create_binary_op_bind_group(
        &self,
        pipeline: &GpuComputePipeline,
        a_buffer: &GpuBuffer,
        b_buffer: &GpuBuffer,
        result_buffer: &GpuBuffer,
        size_buffer: &GpuBuffer,
    ) -> Result<GpuBindGroup, JsValue> {
        // Get bind group layout from pipeline
        let layout = pipeline.get_bind_group_layout(0);

        let entries = js_sys::Array::new();

        let entry0 = super::types::create_bind_group_entry(0, &a_buffer.into())?;
        entries.push(&entry0);

        let entry1 = super::types::create_bind_group_entry(1, &b_buffer.into())?;
        entries.push(&entry1);

        let entry2 = super::types::create_bind_group_entry(2, &result_buffer.into())?;
        entries.push(&entry2);

        let entry3 = super::types::create_bind_group_entry(3, &size_buffer.into())?;
        entries.push(&entry3);

        let descriptor = super::types::create_bind_group_descriptor(&layout, &entries)?;
        Ok(self.device.create_bind_group(&descriptor))
    }

    /// Helper to create bind group for activation functions
    fn create_activation_bind_group(
        &self,
        pipeline: &GpuComputePipeline,
        input_buffer: &GpuBuffer,
        output_buffer: &GpuBuffer,
        size_buffer: &GpuBuffer,
    ) -> Result<GpuBindGroup, JsValue> {
        // Get bind group layout from pipeline
        let layout = pipeline.get_bind_group_layout(0);

        let entries = js_sys::Array::new();

        let entry0 = super::types::create_bind_group_entry(0, &input_buffer.into())?;
        entries.push(&entry0);

        let entry1 = super::types::create_bind_group_entry(1, &output_buffer.into())?;
        entries.push(&entry1);

        let entry2 = super::types::create_bind_group_entry(2, &size_buffer.into())?;
        entries.push(&entry2);

        let descriptor = super::types::create_bind_group_descriptor(&layout, &entries)?;
        Ok(self.device.create_bind_group(&descriptor))
    }

    /// Helper to dispatch compute shader
    fn dispatch_compute(
        &self,
        pipeline: &Option<GpuComputePipeline>,
        bind_group: &GpuBindGroup,
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<(), JsValue> {
        let encoder = self.device.create_command_encoder();
        let pass = encoder.begin_compute_pass();

        if let Some(pipeline) = pipeline {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group);
            pass.dispatch_workgroups(x, y, z);
        }

        pass.end();

        let command_buffer = encoder.finish();
        self.device.queue().submit(&js_sys::Array::of1(&command_buffer));

        Ok(())
    }

    /// Helper to create bind group for softmax operation
    fn create_softmax_bind_group(
        &self,
        pipeline: &GpuComputePipeline,
        input_buffer: &GpuBuffer,
        output_buffer: &GpuBuffer,
        dims_buffer: &GpuBuffer,
    ) -> Result<GpuBindGroup, JsValue> {
        // Get bind group layout from pipeline
        let layout = pipeline.get_bind_group_layout(0);

        let entries = js_sys::Array::new();

        let entry0 = super::types::create_bind_group_entry(0, &input_buffer.into())?;
        entries.push(&entry0);

        let entry1 = super::types::create_bind_group_entry(1, &output_buffer.into())?;
        entries.push(&entry1);

        let entry2 = super::types::create_bind_group_entry(2, &dims_buffer.into())?;
        entries.push(&entry2);

        let descriptor = super::types::create_bind_group_descriptor(&layout, &entries)?;
        Ok(self.device.create_bind_group(&descriptor))
    }

    /// Helper to create bind group for layer normalization
    fn create_layer_norm_bind_group(
        &self,
        pipeline: &GpuComputePipeline,
        input_buffer: &GpuBuffer,
        gamma_buffer: &GpuBuffer,
        beta_buffer: &GpuBuffer,
        output_buffer: &GpuBuffer,
        dims_buffer: &GpuBuffer,
    ) -> Result<GpuBindGroup, JsValue> {
        // Get bind group layout from pipeline
        let layout = pipeline.get_bind_group_layout(0);

        let entries = js_sys::Array::new();

        let entry0 = super::types::create_bind_group_entry(0, &input_buffer.into())?;
        entries.push(&entry0);

        let entry1 = super::types::create_bind_group_entry(1, &gamma_buffer.into())?;
        entries.push(&entry1);

        let entry2 = super::types::create_bind_group_entry(2, &beta_buffer.into())?;
        entries.push(&entry2);

        let entry3 = super::types::create_bind_group_entry(3, &output_buffer.into())?;
        entries.push(&entry3);

        let entry4 = super::types::create_bind_group_entry(4, &dims_buffer.into())?;
        entries.push(&entry4);

        let descriptor = super::types::create_bind_group_descriptor(&layout, &entries)?;
        Ok(self.device.create_bind_group(&descriptor))
    }
}
