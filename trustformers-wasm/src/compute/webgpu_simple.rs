//! Simplified WebGPU support module
//! This provides a basic structure for WebGPU support that can be
//! expanded when web-sys WebGPU bindings are more complete.

use super::webgpu::types::{
    buffer_usage, create_buffer_descriptor, Gpu, GpuAdapter, GpuAdapterExt, GpuBindGroup,
    GpuBuffer, GpuBufferExt, GpuCommandEncoder, GpuCommandEncoderExt, GpuComputePassEncoderExt,
    GpuComputePipeline, GpuComputePipelineExt, GpuDevice, GpuDeviceExt, GpuExt, GpuQueue,
    GpuQueueExt,
};
use crate::core::tensor::WasmTensor;
use std::string::String;
use wasm_bindgen::prelude::*;

/// Check if WebGPU is available
#[wasm_bindgen]
pub fn is_webgpu_available() -> bool {
    web_sys::window()
        .and_then(|w| {
            js_sys::Reflect::get(&w.navigator(), &JsValue::from_str("gpu"))
                .ok()
                .filter(|v| !v.is_undefined())
        })
        .is_some()
}

/// WebGPU status information
#[wasm_bindgen]
pub fn get_webgpu_status() -> String {
    if is_webgpu_available() {
        String::from("WebGPU is available (full implementation pending)")
    } else {
        String::from("WebGPU is not available in this browser")
    }
}

/// WebGPU-accelerated operations
#[wasm_bindgen]
pub struct WebGPUOps {
    device: Option<GpuDevice>,
    queue: Option<GpuQueue>,
    matmul_pipeline: Option<GpuComputePipeline>,
    add_pipeline: Option<GpuComputePipeline>,
    relu_pipeline: Option<GpuComputePipeline>,
    sigmoid_pipeline: Option<GpuComputePipeline>,
    tanh_pipeline: Option<GpuComputePipeline>,
    gelu_pipeline: Option<GpuComputePipeline>,
    softmax_pipeline: Option<GpuComputePipeline>,
}

impl Default for WebGPUOps {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WebGPUOps {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        WebGPUOps {
            device: None,
            queue: None,
            matmul_pipeline: None,
            add_pipeline: None,
            relu_pipeline: None,
            sigmoid_pipeline: None,
            tanh_pipeline: None,
            gelu_pipeline: None,
            softmax_pipeline: None,
        }
    }

    /// Initialize WebGPU device and pipelines
    pub async fn initialize(&mut self) -> Result<(), JsValue> {
        if !is_webgpu_available() {
            return Err(JsValue::from_str("WebGPU is not available"));
        }

        // Get WebGPU adapter and device
        let navigator = web_sys::window().ok_or("No window")?.navigator();

        let gpu = js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu"))?;
        let gpu: Gpu = gpu.dyn_into()?;

        let adapter_promise = gpu.request_adapter();
        let adapter = wasm_bindgen_futures::JsFuture::from(adapter_promise).await?;
        let adapter: GpuAdapter = adapter.dyn_into()?;

        let device_promise = adapter.request_device();
        let device = wasm_bindgen_futures::JsFuture::from(device_promise).await?;
        let device: GpuDevice = device.dyn_into()?;

        let queue = device.queue();

        // Create compute pipelines
        self.matmul_pipeline = Some(self.create_matmul_pipeline(&device)?);
        self.add_pipeline = Some(self.create_add_pipeline(&device)?);
        self.relu_pipeline = Some(self.create_relu_pipeline(&device)?);
        self.sigmoid_pipeline = Some(self.create_sigmoid_pipeline(&device)?);
        self.tanh_pipeline = Some(self.create_tanh_pipeline(&device)?);
        self.gelu_pipeline = Some(self.create_gelu_pipeline(&device)?);
        self.softmax_pipeline = Some(self.create_softmax_pipeline(&device)?);

        self.device = Some(device);
        self.queue = Some(queue);

        Ok(())
    }

    /// Create matrix multiplication compute pipeline
    fn create_matmul_pipeline(&self, device: &GpuDevice) -> Result<GpuComputePipeline, JsValue> {
        let shader_code = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>; // M, N, K

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let M = dims.x;
    let N = dims.y;
    let K = dims.z;

    if (row >= M || col >= N) {
        return;
    }

    var sum = 0.0;
    for (var k = 0u; k < K; k = k + 1u) {
        let a_idx = row * K + k;
        let b_idx = k * N + col;
        sum = sum + a[a_idx] * b[b_idx];
    }

    let result_idx = row * N + col;
    result[result_idx] = sum;
}
"#;

        self.create_compute_pipeline(device, shader_code, "matmul")
    }

    /// Create element-wise addition compute pipeline
    fn create_add_pipeline(&self, device: &GpuDevice) -> Result<GpuComputePipeline, JsValue> {
        let shader_code = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&a)) {
        return;
    }
    result[idx] = a[idx] + b[idx];
}
"#;

        self.create_compute_pipeline(device, shader_code, "add")
    }

    /// Create ReLU activation compute pipeline
    fn create_relu_pipeline(&self, device: &GpuDevice) -> Result<GpuComputePipeline, JsValue> {
        let shader_code = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input)) {
        return;
    }
    result[idx] = max(0.0, input[idx]);
}
"#;

        self.create_compute_pipeline(device, shader_code, "relu")
    }

    /// Create Sigmoid activation compute pipeline
    fn create_sigmoid_pipeline(&self, device: &GpuDevice) -> Result<GpuComputePipeline, JsValue> {
        let shader_code = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input)) {
        return;
    }
    result[idx] = 1.0 / (1.0 + exp(-input[idx]));
}
"#;

        self.create_compute_pipeline(device, shader_code, "sigmoid")
    }

    /// Create Tanh activation compute pipeline
    fn create_tanh_pipeline(&self, device: &GpuDevice) -> Result<GpuComputePipeline, JsValue> {
        let shader_code = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input)) {
        return;
    }
    let e2x = exp(2.0 * input[idx]);
    result[idx] = (e2x - 1.0) / (e2x + 1.0);
}
"#;

        self.create_compute_pipeline(device, shader_code, "tanh")
    }

    /// Create GELU activation compute pipeline
    fn create_gelu_pipeline(&self, device: &GpuDevice) -> Result<GpuComputePipeline, JsValue> {
        let shader_code = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input)) {
        return;
    }
    let x = input[idx];
    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = 0.7978845608028654;
    let x_cubed = x * x * x;
    let tanh_input = sqrt_2_over_pi * (x + 0.044715 * x_cubed);
    let e2x = exp(2.0 * tanh_input);
    let tanh_val = (e2x - 1.0) / (e2x + 1.0);
    result[idx] = x * 0.5 * (1.0 + tanh_val);
}
"#;

        self.create_compute_pipeline(device, shader_code, "gelu")
    }

    /// Create Softmax compute pipeline
    fn create_softmax_pipeline(&self, device: &GpuDevice) -> Result<GpuComputePipeline, JsValue> {
        let shader_code = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>; // batch_size, seq_length

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let seq_length = dims.y;

    if (batch_idx >= dims.x) {
        return;
    }

    let start_idx = batch_idx * seq_length;

    // Find max value for numerical stability
    var max_val = input[start_idx];
    for (var i = 1u; i < seq_length; i = i + 1u) {
        max_val = max(max_val, input[start_idx + i]);
    }

    // Compute sum of exponentials
    var sum_exp = 0.0;
    for (var i = 0u; i < seq_length; i = i + 1u) {
        let exp_val = exp(input[start_idx + i] - max_val);
        result[start_idx + i] = exp_val;
        sum_exp = sum_exp + exp_val;
    }

    // Normalize
    for (var i = 0u; i < seq_length; i = i + 1u) {
        result[start_idx + i] = result[start_idx + i] / sum_exp;
    }
}
"#;

        self.create_compute_pipeline(device, shader_code, "softmax")
    }

    /// Helper to create a compute pipeline
    fn create_compute_pipeline(
        &self,
        device: &GpuDevice,
        code: &str,
        label: &str,
    ) -> Result<GpuComputePipeline, JsValue> {
        let shader_desc = super::webgpu::types::create_shader_module_descriptor(code, Some(label))?;
        let shader_module = device.create_shader_module(&shader_desc);

        let compute_stage =
            super::webgpu::types::create_programmable_stage(&shader_module, "main")?;
        let pipeline_desc =
            super::webgpu::types::create_compute_pipeline_descriptor(&compute_stage, Some(label))?;

        Ok(device.create_compute_pipeline(&pipeline_desc))
    }

    /// GPU-accelerated matrix multiplication
    pub async fn matmul(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let device = match &self.device {
            Some(d) => d,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU not initialized, falling back to CPU",
                ));
                return a.matmul(b);
            },
        };

        let pipeline = match &self.matmul_pipeline {
            Some(p) => p,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU matmul pipeline not ready, falling back to CPU",
                ));
                return a.matmul(b);
            },
        };

        let queue = match &self.queue {
            Some(q) => q,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU queue not ready, falling back to CPU",
                ));
                return a.matmul(b);
            },
        };

        // Get tensor data and shapes
        let a_data = a.data();
        let b_data = b.data();
        let a_shape = a.shape();
        let b_shape = b.shape();

        // Validate matrix multiplication dimensions
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(JsValue::from_str(
                "Matrix multiplication requires 2D tensors",
            ));
        }
        if a_shape[1] != b_shape[0] {
            return Err(JsValue::from_str(
                "Matrix dimensions incompatible for multiplication",
            ));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        let result_size = m * n;

        // Create GPU buffers
        let a_buffer = self.create_buffer(device, &a_data, "Matrix A")?;
        let b_buffer = self.create_buffer(device, &b_data, "Matrix B")?;
        let result_buffer = self.create_result_buffer(device, result_size, "Matrix Result")?;

        // Create uniform buffer for dimensions
        let dims = [m as u32, n as u32, k as u32, 0u32]; // Pad to 16 bytes
        let dims_bytes: &[u8] = bytemuck::cast_slice(&dims);
        let dims_buffer_desc = create_buffer_descriptor(
            dims_bytes.len() as f64,
            buffer_usage::UNIFORM,
            Some("Dimensions"),
            true,
        )?;
        let dims_buffer = device.create_buffer(&dims_buffer_desc);

        let dims_array_buffer = dims_buffer.get_mapped_range();
        let dims_uint8_array = js_sys::Uint8Array::new(&dims_array_buffer);
        dims_uint8_array.copy_from(dims_bytes);
        dims_buffer.unmap();

        // Create bind group
        let bind_group = self.create_matmul_bind_group(
            device,
            pipeline,
            &a_buffer,
            &b_buffer,
            &result_buffer,
            &dims_buffer,
        )?;

        // Create command encoder and dispatch compute
        // GpuCommandEncoderDescriptor not available in web-sys 0.3.81 - using default
        let encoder = device.create_command_encoder();

        // GpuComputePassDescriptor not available in web-sys 0.3.81 - using default
        let compute_pass = encoder.begin_compute_pass();

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group);

        // Dispatch with appropriate workgroup counts
        let workgroup_x = m.div_ceil(8); // Round up to cover all rows
        let workgroup_y = n.div_ceil(8); // Round up to cover all columns
        compute_pass.dispatch_workgroups(workgroup_x as u32, workgroup_y as u32, 1);
        compute_pass.end();

        // Read back results
        let result_data =
            self.read_buffer(device, queue, &encoder, &result_buffer, result_size).await?;

        // Create result tensor
        WasmTensor::new(result_data, vec![m, n])
    }

    /// GPU-accelerated element-wise addition
    pub async fn add(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let device = match &self.device {
            Some(d) => d,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU not initialized, falling back to CPU",
                ));
                return a.add(b);
            },
        };

        let pipeline = match &self.add_pipeline {
            Some(p) => p,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU add pipeline not ready, falling back to CPU",
                ));
                return a.add(b);
            },
        };

        let queue = match &self.queue {
            Some(q) => q,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU queue not ready, falling back to CPU",
                ));
                return a.add(b);
            },
        };

        // Get tensor data and shapes
        let a_data = a.data();
        let b_data = b.data();
        let a_shape = a.shape();
        let b_shape = b.shape();

        // Validate tensor shapes are compatible
        if a_shape != b_shape {
            return Err(JsValue::from_str(
                "Tensor shapes must match for element-wise addition",
            ));
        }

        let size = a_data.len();

        // Create GPU buffers
        let a_buffer = self.create_buffer(device, &a_data, "Tensor A")?;
        let b_buffer = self.create_buffer(device, &b_data, "Tensor B")?;
        let result_buffer = self.create_result_buffer(device, size, "Addition Result")?;

        // Create bind group
        let bind_group = self.create_elementwise_bind_group(
            device,
            pipeline,
            &a_buffer,
            Some(&b_buffer),
            &result_buffer,
        )?;

        // Create command encoder and dispatch compute
        // GpuCommandEncoderDescriptor not available in web-sys 0.3.81 - using default
        let encoder = device.create_command_encoder();

        // GpuComputePassDescriptor not available in web-sys 0.3.81 - using default
        let compute_pass = encoder.begin_compute_pass();

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group);

        // Dispatch with appropriate workgroup count
        let workgroup_count = size.div_ceil(64); // Round up to cover all elements
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        compute_pass.end();

        // Read back results
        let result_data = self.read_buffer(device, queue, &encoder, &result_buffer, size).await?;

        // Create result tensor
        WasmTensor::new(result_data, a_shape)
    }

    /// GPU-accelerated ReLU activation
    pub async fn relu(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let device = match &self.device {
            Some(d) => d,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU not initialized, falling back to CPU",
                ));
                return Ok(input.relu());
            },
        };

        let pipeline = match &self.relu_pipeline {
            Some(p) => p,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU ReLU pipeline not ready, falling back to CPU",
                ));
                return Ok(input.relu());
            },
        };

        let queue = match &self.queue {
            Some(q) => q,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU queue not ready, falling back to CPU",
                ));
                return Ok(input.relu());
            },
        };

        // Get tensor data and shape
        let input_data = input.data();
        let shape = input.shape();
        let size = input_data.len();

        // Create GPU buffers
        let input_buffer = self.create_buffer(device, &input_data, "ReLU Input")?;
        let result_buffer = self.create_result_buffer(device, size, "ReLU Result")?;

        // Create bind group (no second input for unary operation)
        let bind_group = self.create_elementwise_bind_group(
            device,
            pipeline,
            &input_buffer,
            None,
            &result_buffer,
        )?;

        // Create command encoder and dispatch compute
        // GpuCommandEncoderDescriptor not available in web-sys 0.3.81 - using default
        let encoder = device.create_command_encoder();

        // GpuComputePassDescriptor not available in web-sys 0.3.81 - using default
        let compute_pass = encoder.begin_compute_pass();

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group);

        // Dispatch with appropriate workgroup count
        let workgroup_count = size.div_ceil(64); // Round up to cover all elements
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        compute_pass.end();

        // Read back results
        let result_data = self.read_buffer(device, queue, &encoder, &result_buffer, size).await?;

        // Create result tensor
        WasmTensor::new(result_data, shape)
    }

    /// GPU-accelerated Sigmoid activation
    pub async fn sigmoid(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let device = match &self.device {
            Some(d) => d,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU not initialized, falling back to CPU",
                ));
                return Ok(input.sigmoid());
            },
        };

        let pipeline = match &self.sigmoid_pipeline {
            Some(p) => p,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU Sigmoid pipeline not ready, falling back to CPU",
                ));
                return Ok(input.sigmoid());
            },
        };

        let queue = match &self.queue {
            Some(q) => q,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU queue not ready, falling back to CPU",
                ));
                return Ok(input.sigmoid());
            },
        };

        self.apply_unary_activation(device, pipeline, queue, input, "Sigmoid").await
    }

    /// GPU-accelerated Tanh activation
    pub async fn tanh(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let device = match &self.device {
            Some(d) => d,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU not initialized, falling back to CPU",
                ));
                return Ok(input.tanh());
            },
        };

        let pipeline = match &self.tanh_pipeline {
            Some(p) => p,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU Tanh pipeline not ready, falling back to CPU",
                ));
                return Ok(input.tanh());
            },
        };

        let queue = match &self.queue {
            Some(q) => q,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU queue not ready, falling back to CPU",
                ));
                return Ok(input.tanh());
            },
        };

        self.apply_unary_activation(device, pipeline, queue, input, "Tanh").await
    }

    /// GPU-accelerated GELU activation
    pub async fn gelu(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let device = match &self.device {
            Some(d) => d,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU not initialized, falling back to CPU",
                ));
                return Ok(input.gelu());
            },
        };

        let pipeline = match &self.gelu_pipeline {
            Some(p) => p,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU GELU pipeline not ready, falling back to CPU",
                ));
                return Ok(input.gelu());
            },
        };

        let queue = match &self.queue {
            Some(q) => q,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU queue not ready, falling back to CPU",
                ));
                return Ok(input.gelu());
            },
        };

        self.apply_unary_activation(device, pipeline, queue, input, "GELU").await
    }

    /// GPU-accelerated Softmax activation
    pub async fn softmax(&self, input: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let device = match &self.device {
            Some(d) => d,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU not initialized, falling back to CPU",
                ));
                return input.softmax(-1);
            },
        };

        let pipeline = match &self.softmax_pipeline {
            Some(p) => p,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU Softmax pipeline not ready, falling back to CPU",
                ));
                return input.softmax(-1);
            },
        };

        let queue = match &self.queue {
            Some(q) => q,
            None => {
                web_sys::console::log_1(&JsValue::from_str(
                    "WebGPU queue not ready, falling back to CPU",
                ));
                return input.softmax(-1);
            },
        };

        // Get tensor data and shape
        let input_data = input.data();
        let shape = input.shape();

        // For softmax, we assume last dimension is the one to normalize over
        if shape.len() < 2 {
            return Err(JsValue::from_str("Softmax requires at least 2D tensors"));
        }

        let batch_size = shape.iter().take(shape.len() - 1).product::<usize>();
        let seq_length = shape[shape.len() - 1];
        let total_size = input_data.len();

        // Create GPU buffers
        let input_buffer = self.create_buffer(device, &input_data, "Softmax Input")?;
        let result_buffer = self.create_result_buffer(device, total_size, "Softmax Result")?;

        // Create uniform buffer for dimensions
        let dims = [batch_size as u32, seq_length as u32];
        let dims_bytes: &[u8] = bytemuck::cast_slice(&dims);
        let dims_buffer_desc = create_buffer_descriptor(
            dims_bytes.len() as f64,
            buffer_usage::UNIFORM,
            Some("Softmax Dimensions"),
            true,
        )?;
        let dims_buffer = device.create_buffer(&dims_buffer_desc);

        let dims_array_buffer = dims_buffer.get_mapped_range();
        let dims_uint8_array = js_sys::Uint8Array::new(&dims_array_buffer);
        dims_uint8_array.copy_from(dims_bytes);
        dims_buffer.unmap();

        // Create bind group for softmax
        let bind_group = self.create_softmax_bind_group(
            device,
            pipeline,
            &input_buffer,
            &result_buffer,
            &dims_buffer,
        )?;

        // Create command encoder and dispatch compute
        // GpuCommandEncoderDescriptor not available in web-sys 0.3.81 - using default
        let encoder = device.create_command_encoder();

        // GpuComputePassDescriptor not available in web-sys 0.3.81 - using default
        let compute_pass = encoder.begin_compute_pass();

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group);

        // Dispatch with appropriate workgroup count (one workgroup per batch)
        compute_pass.dispatch_workgroups(batch_size as u32, 1, 1);
        compute_pass.end();

        // Read back results
        let result_data =
            self.read_buffer(device, queue, &encoder, &result_buffer, total_size).await?;

        // Create result tensor
        WasmTensor::new(result_data, shape)
    }

    /// Helper method for unary activation functions
    async fn apply_unary_activation(
        &self,
        device: &GpuDevice,
        pipeline: &GpuComputePipeline,
        queue: &GpuQueue,
        input: &WasmTensor,
        operation_name: &str,
    ) -> Result<WasmTensor, JsValue> {
        // Get tensor data and shape
        let input_data = input.data();
        let shape = input.shape();
        let size = input_data.len();

        // Create GPU buffers
        let input_buffer =
            self.create_buffer(device, &input_data, &format!("{} Input", operation_name))?;
        let result_buffer =
            self.create_result_buffer(device, size, &format!("{} Result", operation_name))?;

        // Create bind group (no second input for unary operation)
        let bind_group = self.create_elementwise_bind_group(
            device,
            pipeline,
            &input_buffer,
            None,
            &result_buffer,
        )?;

        // Create command encoder and dispatch compute
        // GpuCommandEncoderDescriptor not available in web-sys 0.3.81 - using default
        let encoder = device.create_command_encoder();

        // GpuComputePassDescriptor not available in web-sys 0.3.81 - using default
        let compute_pass = encoder.begin_compute_pass();

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group);

        // Dispatch with appropriate workgroup count
        let workgroup_count = size.div_ceil(64); // Round up to cover all elements
        compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        compute_pass.end();

        // Read back results
        let result_data = self.read_buffer(device, queue, &encoder, &result_buffer, size).await?;

        // Create result tensor
        WasmTensor::new(result_data, shape)
    }

    /// Get implementation info
    pub fn info(&self) -> String {
        if self.device.is_some() {
            String::from("WebGPU operations (GPU compute pipelines initialized)")
        } else {
            String::from("WebGPU operations (not initialized, using CPU fallback)")
        }
    }

    /// Check if WebGPU is properly initialized
    pub fn is_initialized(&self) -> bool {
        self.device.is_some() && self.queue.is_some()
    }

    /// Create a GPU buffer with data
    fn create_buffer(
        &self,
        device: &GpuDevice,
        data: &[f32],
        label: &str,
    ) -> Result<GpuBuffer, JsValue> {
        let data_bytes: &[u8] = bytemuck::cast_slice(data);
        let buffer_desc = create_buffer_descriptor(
            data_bytes.len() as f64,
            buffer_usage::STORAGE | buffer_usage::COPY_DST,
            Some(label),
            true,
        )?;

        let buffer = device.create_buffer(&buffer_desc);
        let array_buffer = buffer.get_mapped_range();
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        uint8_array.copy_from(data_bytes);
        buffer.unmap();

        Ok(buffer)
    }

    /// Create a result buffer for output
    fn create_result_buffer(
        &self,
        device: &GpuDevice,
        size: usize,
        label: &str,
    ) -> Result<GpuBuffer, JsValue> {
        let byte_size = size * 4; // f32 = 4 bytes
        let buffer_desc = create_buffer_descriptor(
            byte_size as f64,
            buffer_usage::STORAGE | buffer_usage::COPY_SRC,
            Some(label),
            false,
        )?;

        Ok(device.create_buffer(&buffer_desc))
    }

    /// Create bind group for matrix multiplication
    fn create_matmul_bind_group(
        &self,
        device: &GpuDevice,
        pipeline: &GpuComputePipeline,
        a_buffer: &GpuBuffer,
        b_buffer: &GpuBuffer,
        result_buffer: &GpuBuffer,
        dims_buffer: &GpuBuffer,
    ) -> Result<GpuBindGroup, JsValue> {
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let entries = js_sys::Array::new();

        // Binding 0: Matrix A
        let entry0 = super::webgpu::types::create_bind_group_entry(0, a_buffer)?;
        entries.push(&entry0);

        // Binding 1: Matrix B
        let entry1 = super::webgpu::types::create_bind_group_entry(1, b_buffer)?;
        entries.push(&entry1);

        // Binding 2: Result
        let entry2 = super::webgpu::types::create_bind_group_entry(2, result_buffer)?;
        entries.push(&entry2);

        // Binding 3: Dimensions uniform
        let entry3 = super::webgpu::types::create_bind_group_entry(3, dims_buffer)?;
        entries.push(&entry3);

        let bind_group_desc =
            super::webgpu::types::create_bind_group_descriptor(&bind_group_layout, &entries)?;

        Ok(device.create_bind_group(&bind_group_desc))
    }

    /// Create bind group for element-wise operations
    fn create_elementwise_bind_group(
        &self,
        device: &GpuDevice,
        pipeline: &GpuComputePipeline,
        a_buffer: &GpuBuffer,
        b_buffer: Option<&GpuBuffer>,
        result_buffer: &GpuBuffer,
    ) -> Result<GpuBindGroup, JsValue> {
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let entries = js_sys::Array::new();

        // Binding 0: Input A
        let entry0 = super::webgpu::types::create_bind_group_entry(0, a_buffer)?;
        entries.push(&entry0);

        // Binding 1: Input B (if present) or Result (for unary ops)
        if let Some(b_buf) = b_buffer {
            let entry1 = super::webgpu::types::create_bind_group_entry(1, b_buf)?;
            entries.push(&entry1);

            // Binding 2: Result (for binary ops)
            let entry2 = super::webgpu::types::create_bind_group_entry(2, result_buffer)?;
            entries.push(&entry2);
        } else {
            // For unary operations like ReLU
            let entry1 = super::webgpu::types::create_bind_group_entry(1, result_buffer)?;
            entries.push(&entry1);
        }

        let bind_group_desc =
            super::webgpu::types::create_bind_group_descriptor(&bind_group_layout, &entries)?;

        Ok(device.create_bind_group(&bind_group_desc))
    }

    /// Create bind group for softmax operation
    fn create_softmax_bind_group(
        &self,
        device: &GpuDevice,
        pipeline: &GpuComputePipeline,
        input_buffer: &GpuBuffer,
        result_buffer: &GpuBuffer,
        dims_buffer: &GpuBuffer,
    ) -> Result<GpuBindGroup, JsValue> {
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let entries = js_sys::Array::new();

        // Binding 0: Input
        let entry0 = super::webgpu::types::create_bind_group_entry(0, input_buffer)?;
        entries.push(&entry0);

        // Binding 1: Result
        let entry1 = super::webgpu::types::create_bind_group_entry(1, result_buffer)?;
        entries.push(&entry1);

        // Binding 2: Dimensions uniform
        let entry2 = super::webgpu::types::create_bind_group_entry(2, dims_buffer)?;
        entries.push(&entry2);

        let bind_group_desc =
            super::webgpu::types::create_bind_group_descriptor(&bind_group_layout, &entries)?;

        Ok(device.create_bind_group(&bind_group_desc))
    }

    /// Read buffer data back from GPU
    async fn read_buffer(
        &self,
        device: &GpuDevice,
        queue: &GpuQueue,
        encoder: &GpuCommandEncoder,
        buffer: &GpuBuffer,
        size: usize,
    ) -> Result<Vec<f32>, JsValue> {
        let byte_size = size * 4; // f32 = 4 bytes

        // Create staging buffer for reading
        let staging_desc = create_buffer_descriptor(
            byte_size as f64,
            buffer_usage::COPY_DST | buffer_usage::MAP_READ,
            Some("Staging Buffer"),
            false,
        )?;
        let staging_buffer = device.create_buffer(&staging_desc);

        // Copy from result buffer to staging buffer
        encoder.copy_buffer_to_buffer(buffer, 0.0, &staging_buffer, 0.0, byte_size as f64);

        // Submit commands
        let commands = js_sys::Array::new();
        commands.push(&encoder.finish());
        queue.submit(&commands);

        // Map and read the staging buffer
        let map_promise = staging_buffer.map_async(
            1u32, // READ = 1, GpuMapMode not available in web-sys 0.3.81
            0.0,
            byte_size as f64,
        );
        wasm_bindgen_futures::JsFuture::from(map_promise).await?;

        // Call getMappedRange(0, byte_size) using Reflect
        let get_mapped_range_fn =
            js_sys::Reflect::get(&staging_buffer, &JsValue::from_str("getMappedRange"))?;
        let get_mapped_range_fn: &js_sys::Function = get_mapped_range_fn.unchecked_ref();
        let args = js_sys::Array::new();
        args.push(&JsValue::from_f64(0.0));
        args.push(&JsValue::from_f64(byte_size as f64));
        let array_buffer = get_mapped_range_fn.apply(&staging_buffer, &args)?;

        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let mut bytes = vec![0u8; byte_size];
        uint8_array.copy_to(&mut bytes);

        staging_buffer.unmap();

        // Convert bytes back to f32
        let float_data: &[f32] = bytemuck::cast_slice(&bytes);
        Ok(float_data.to_vec())
    }
}
