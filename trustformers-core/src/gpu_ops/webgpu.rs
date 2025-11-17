//! WebGPU backend for tensor operations
//!
//! This module provides WebGPU acceleration for tensor operations.
//! WebGPU is cross-platform and works in browsers, desktop, and mobile.
//! Uses wgpu-rs for WebGPU API access.
//!
//! Features:
//! - Cross-platform GPU support (Windows/Linux/macOS/Web)
//! - Automatic backend selection (Vulkan/Metal/DX12/OpenGL)
//! - Browser/WASM compatible
//! - Persistent buffer caching
//! - GPU-to-GPU operations

use crate::device::Device;
use crate::errors::Result;
use crate::tensor::Tensor;

#[cfg(feature = "wgpu_backend")]
use crate::errors::TrustformersError;
#[cfg(feature = "wgpu_backend")]
use std::collections::HashMap;
#[cfg(feature = "wgpu_backend")]
use std::sync::Arc;
#[cfg(feature = "wgpu_backend")]
use wgpu::util::DeviceExt;

/// Buffer ID for persistent GPU buffers
#[cfg(feature = "wgpu_backend")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

#[cfg(feature = "wgpu_backend")]
impl BufferId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        BufferId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

#[cfg(feature = "wgpu_backend")]
impl Default for BufferId {
    fn default() -> Self {
        Self::new()
    }
}

/// Persistent buffer cache for WebGPU
#[cfg(feature = "wgpu_backend")]
struct BufferCache {
    buffers: HashMap<BufferId, Arc<wgpu::Buffer>>,
}

#[cfg(feature = "wgpu_backend")]
impl BufferCache {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    fn insert(&mut self, id: BufferId, buffer: Arc<wgpu::Buffer>) {
        self.buffers.insert(id, buffer);
    }

    fn get(&self, id: &BufferId) -> Option<Arc<wgpu::Buffer>> {
        self.buffers.get(id).cloned()
    }

    fn remove(&mut self, id: &BufferId) -> Option<Arc<wgpu::Buffer>> {
        self.buffers.remove(id)
    }

    fn clear(&mut self) {
        self.buffers.clear();
    }

    fn len(&self) -> usize {
        self.buffers.len()
    }
}

/// WebGPU backend for GPU-accelerated operations
#[cfg(feature = "wgpu_backend")]
pub struct WebGpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffer_cache: Arc<std::sync::Mutex<BufferCache>>,
}

#[cfg(feature = "wgpu_backend")]
impl WebGpuBackend {
    /// Create a new WebGPU backend
    pub fn new() -> Result<Self> {
        use pollster::FutureExt;

        // Request adapter
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("No suitable WebGPU adapter found: {:?}", e),
                    "WebGpuBackend::new",
                )
            })?;

        // Request device and queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("TrustformeRS WebGPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .block_on()
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create WebGPU device: {}", e),
                    "WebGpuBackend::new",
                )
            })?;

        println!("✓ WebGPU backend initialized");
        println!("  Backend: {:?}", adapter.get_info().backend);
        println!("  Device: {}", adapter.get_info().name);

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            buffer_cache: Arc::new(std::sync::Mutex::new(BufferCache::new())),
        })
    }

    /// Create a persistent GPU buffer and return its ID
    pub fn create_persistent_buffer(&self, data: &[f32]) -> Result<BufferId> {
        let buffer = Arc::new(
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Persistent Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            }),
        );

        let buffer_id = BufferId::new();

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "create_persistent_buffer",
            )
        })?;

        cache.insert(buffer_id, buffer);
        Ok(buffer_id)
    }

    /// Get a persistent buffer by ID
    pub fn get_persistent_buffer(&self, id: &BufferId) -> Result<Arc<wgpu::Buffer>> {
        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "get_persistent_buffer",
            )
        })?;

        cache.get(id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Buffer {:?} not found in cache", id),
                "get_persistent_buffer",
            )
        })
    }

    /// Remove a persistent buffer from cache
    pub fn remove_persistent_buffer(&self, id: &BufferId) -> Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "remove_persistent_buffer",
            )
        })?;

        cache.remove(id);
        Ok(())
    }

    /// Clear all persistent buffers
    pub fn clear_buffer_cache(&self) -> Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "clear_buffer_cache")
        })?;

        cache.clear();
        Ok(())
    }

    /// Get the number of cached buffers
    pub fn buffer_cache_size(&self) -> Result<usize> {
        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "buffer_cache_size")
        })?;

        Ok(cache.len())
    }

    /// Perform matrix multiplication on WebGPU
    /// C = A @ B where A is [m, k], B is [k, n], C is [m, n]
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // WebGPU shader in WGSL (WebGPU Shading Language)
        const SHADER_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>;  // M, N, K

@compute @workgroup_size(16, 16)
fn matmul_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let M = dims.x;
    let N = dims.y;
    let K = dims.z;

    if (row >= M || col >= N) {
        return;
    }

    var sum = 0.0;
    for (var i = 0u; i < K; i = i + 1u) {
        sum += a[row * K + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}
"#;

        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Multiply Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        // Create buffers
        let a_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A Buffer"),
            contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B Buffer"),
            contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let result_size = m * n;
        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C Buffer"),
            size: (result_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for dimensions
        let dims_data = [m as u32, n as u32, k as u32, 0u32]; // Padding for alignment
        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dimensions Buffer"),
            contents: bytemuck::cast_slice(&dims_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (result_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Matmul Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matmul Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matmul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("matmul_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create command encoder and dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Matmul Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matmul Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (16x16 workgroup size)
            let workgroups_x = (n as u32 + 15) / 16;
            let workgroups_y = (m as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(
            &c_buffer,
            0,
            &staging_buffer,
            0,
            (result_size * std::mem::size_of::<f32>()) as u64,
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read result from staging buffer
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // In wgpu 27.0, polling is automatic - removed manual poll call
        receiver.recv().unwrap().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to map buffer: {:?}", e),
                "matmul_f32",
            )
        })?;

        // Copy data
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Execute GELU activation on GPU
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    pub fn gelu_f32(&self, input: &[f32]) -> Result<Vec<f32>> {
        const SHADER_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn gelu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }

    let x = input[idx];

    // Clamp extreme values to prevent NaN
    if (x > 10.0) {
        output[idx] = x;
        return;
    } else if (x < -10.0) {
        output[idx] = 0.0;
        return;
    }

    let x_cubed = x * x * x;
    // sqrt(2/π) ≈ 0.7978845608
    var inner = 0.7978845608 * (x + 0.044715 * x_cubed);

    // Clamp inner to prevent tanh overflow
    inner = clamp(inner, -20.0, 20.0);

    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}
"#;

        let size = input.len();

        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GELU Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        // Create buffers
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let size_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Size Buffer"),
            contents: bytemuck::cast_slice(&[size as u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout and bind group
        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GELU Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GELU Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: size_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GELU Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GELU Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("gelu_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create command encoder and dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GELU Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GELU Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (size as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (size * std::mem::size_of::<f32>()) as u64,
        );

        // Submit and read result
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // In wgpu 27.0, polling is automatic - removed manual poll call
        receiver.recv().unwrap().map_err(|e| {
            TrustformersError::hardware_error(&format!("Failed to map buffer: {:?}", e), "gelu_f32")
        })?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Execute LayerNorm on GPU
    /// LayerNorm: output = (x - mean) / sqrt(var + eps) * weight + bias
    pub fn layernorm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        const SHADER_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>;  // seq_len, hidden_size, eps (as bits), padding

@compute @workgroup_size(64)
fn layernorm_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = global_id.x;
    let seq_len = params.x;
    let hidden_size = params.y;
    let eps_bits = params.z;
    let eps = bitcast<f32>(eps_bits);

    if (pos >= seq_len) {
        return;
    }

    let offset = pos * hidden_size;

    // Compute mean
    var sum = 0.0;
    for (var i = 0u; i < hidden_size; i = i + 1u) {
        sum += input[offset + i];
    }
    let mean = sum / f32(hidden_size);

    // Compute variance
    var var_sum = 0.0;
    for (var i = 0u; i < hidden_size; i = i + 1u) {
        let diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    let variance = var_sum / f32(hidden_size);
    let std_dev = sqrt(variance + eps);

    // Normalize and apply affine transform
    for (var i = 0u; i < hidden_size; i = i + 1u) {
        let normalized = (input[offset + i] - mean) / std_dev;
        output[offset + i] = normalized * weight[i] + bias[i];
    }
}
"#;

        let total_size = seq_len * hidden_size;

        if input.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input size {} doesn't match seq_len {} * hidden_size {}",
                input.len(),
                seq_len,
                hidden_size
            )));
        }

        if weight.len() != hidden_size || bias.len() != hidden_size {
            return Err(TrustformersError::shape_error(
                "Weight/bias size must match hidden_size".to_string(),
            ));
        }

        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LayerNorm Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        // Create buffers
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let weight_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weight Buffer"),
            contents: bytemuck::cast_slice(weight),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bias_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bias Buffer"),
            contents: bytemuck::cast_slice(bias),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (total_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer (eps needs to be passed as bits for uniform buffer)
        let eps_bits = eps.to_bits();
        let params_data = [seq_len as u32, hidden_size as u32, eps_bits, 0u32];
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (total_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("LayerNorm Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LayerNorm Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LayerNorm Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LayerNorm Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("layernorm_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Execute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LayerNorm Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LayerNorm Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (seq_len as u32 + 63) / 64;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (total_size * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read result
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // In wgpu 27.0, polling is automatic - removed manual poll call
        receiver.recv().unwrap().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to map buffer: {:?}", e),
                "layernorm_f32",
            )
        })?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Copy GPU buffer data back to CPU
    pub fn buffer_to_cpu(&self, buffer_id: &BufferId, size: usize) -> Result<Vec<f32>> {
        let buffer = self.get_persistent_buffer(buffer_id)?;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &buffer,
            0,
            &staging_buffer,
            0,
            (size * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // In wgpu 27.0, polling is automatic - removed manual poll call
        receiver.recv().unwrap().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to map buffer: {:?}", e),
                "buffer_to_cpu",
            )
        })?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Get device information
    pub fn device_info(&self) -> String {
        "WebGPU Device".to_string()
    }
}

/// Global WebGPU backend cache
#[cfg(feature = "wgpu_backend")]
static WEBGPU_BACKEND: once_cell::sync::Lazy<std::sync::Mutex<Option<Arc<WebGpuBackend>>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(None));

/// Get or create WebGPU backend instance
#[cfg(feature = "wgpu_backend")]
pub fn get_webgpu_backend() -> Result<Arc<WebGpuBackend>> {
    let mut cache = WEBGPU_BACKEND.lock().map_err(|_| {
        TrustformersError::hardware_error(
            "Failed to lock WebGPU backend cache",
            "get_webgpu_backend",
        )
    })?;

    if cache.is_none() {
        *cache = Some(Arc::new(WebGpuBackend::new()?));
    }

    cache.clone().ok_or_else(|| {
        TrustformersError::hardware_error("WebGPU backend not initialized", "get_webgpu_backend")
    })
}

/// Dispatch matrix multiplication to WebGPU backend
#[allow(unused_variables)]
pub fn dispatch_webgpu_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "wgpu_backend")]
    {
        match (a, b) {
            (Tensor::F32(a_arr), Tensor::F32(b_arr)) => {
                // Convert to 2D arrays
                if a_arr.ndim() != 2 || b_arr.ndim() != 2 {
                    return Err(TrustformersError::shape_error(
                        "WebGPU dispatch currently only supports 2D tensors".to_string(),
                    ));
                }

                let a_2d = a_arr
                    .clone()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .map_err(|e| {
                        TrustformersError::shape_error(format!("Failed to convert to 2D: {}", e))
                    })?;
                let b_2d = b_arr
                    .clone()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .map_err(|e| {
                        TrustformersError::shape_error(format!("Failed to convert to 2D: {}", e))
                    })?;

                let (m, k) = a_2d.dim();
                let (k2, n) = b_2d.dim();

                if k != k2 {
                    return Err(TrustformersError::shape_error(format!(
                        "Matrix dimension mismatch: {}×{} vs {}×{}",
                        m, k, k2, n
                    )));
                }

                // Get WebGPU backend
                let backend = get_webgpu_backend()?;

                // Convert to contiguous slices
                let a_data: Vec<f32> = a_2d.iter().copied().collect();
                let b_data: Vec<f32> = b_2d.iter().copied().collect();

                // Execute WebGPU matmul
                let result_data = backend.matmul_f32(&a_data, &b_data, m, k, n)?;

                // Convert back to tensor
                let result_2d = scirs2_core::ndarray::Array2::from_shape_vec((m, n), result_data)
                    .map_err(|e| {
                    TrustformersError::shape_error(format!("Failed to reshape result: {}", e))
                })?;

                let result_dyn = result_2d.into_dyn();
                return Ok(Tensor::F32(result_dyn));
            },
            _ => {
                // Fallback to CPU matmul for non-F32 tensors
                return a.matmul(b);
            },
        }
    }

    #[cfg(not(feature = "wgpu_backend"))]
    {
        // No WebGPU support, fallback to CPU
        a.matmul(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "wgpu_backend")]
    fn test_webgpu_backend_creation() -> Result<()> {
        match WebGpuBackend::new() {
            Ok(backend) => {
                println!("WebGPU backend created: {}", backend.device_info());
                Ok(())
            },
            Err(e) => {
                eprintln!("Skipping WebGPU test: {}", e);
                Ok(())
            },
        }
    }

    #[test]
    #[cfg(feature = "wgpu_backend")]
    fn test_webgpu_matmul() -> Result<()> {
        let backend = match WebGpuBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping WebGPU test: no adapter available");
                return Ok(());
            },
        };

        // Simple 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]

        let result = backend.matmul_f32(&a, &b, 2, 2, 2)?;

        // Expected: [[19, 22], [43, 50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-4,
                "Mismatch at index {}: {} vs {}",
                i,
                res,
                exp
            );
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "wgpu_backend")]
    fn test_webgpu_gelu() -> Result<()> {
        let backend = match WebGpuBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping WebGPU test: no adapter available");
                return Ok(());
            },
        };

        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = backend.gelu_f32(&input)?;

        // GELU should be monotonic increasing
        for i in 0..result.len() - 1 {
            assert!(result[i] <= result[i + 1], "GELU should be monotonic");
        }

        // GELU(0) ≈ 0
        assert!((result[2]).abs() < 0.01, "GELU(0) should be ~0");

        Ok(())
    }

    #[test]
    #[cfg(feature = "wgpu_backend")]
    fn test_webgpu_layernorm() -> Result<()> {
        let backend = match WebGpuBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping WebGPU test: no adapter available");
                return Ok(());
            },
        };

        // Simple test: 2 sequences, 4 features each
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 0.0];

        let result = backend.layernorm_f32(&input, &weight, &bias, 2, 4, 1e-5)?;

        assert_eq!(result.len(), 8);

        // Each row should have mean ≈ 0
        for row in 0..2 {
            let row_data = &result[row * 4..(row + 1) * 4];
            let mean: f32 = row_data.iter().sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-4, "Mean should be ~0, got {}", mean);
        }

        Ok(())
    }
}
