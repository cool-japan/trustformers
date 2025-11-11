//! Metal GPU backend for matrix operations
//!
//! This module provides Metal GPU acceleration for tensor operations on Apple Silicon.
//! Uses metal-rs for direct Metal API access.

use crate::device::Device;
use crate::errors::Result;
use crate::tensor::Tensor;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::errors::TrustformersError;
#[cfg(all(target_os = "macos", feature = "metal"))]
use metal::{Buffer, CommandQueue, CompileOptions, Device as MetalDevice, MTLResourceOptions};
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::collections::HashMap;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::mem;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::Arc;

/// Buffer ID for persistent GPU buffers
#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

#[cfg(feature = "metal")]
impl BufferId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        BufferId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

#[cfg(feature = "metal")]
impl Default for BufferId {
    fn default() -> Self {
        Self::new()
    }
}

/// Persistent buffer cache for Metal GPU
#[cfg(all(target_os = "macos", feature = "metal"))]
struct BufferCache {
    buffers: HashMap<BufferId, Arc<Buffer>>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl BufferCache {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    fn insert(&mut self, id: BufferId, buffer: Arc<Buffer>) {
        self.buffers.insert(id, buffer);
    }

    fn get(&self, id: &BufferId) -> Option<Arc<Buffer>> {
        self.buffers.get(id).cloned()
    }

    fn remove(&mut self, id: &BufferId) -> Option<Arc<Buffer>> {
        self.buffers.remove(id)
    }

    fn clear(&mut self) {
        self.buffers.clear();
    }

    fn len(&self) -> usize {
        self.buffers.len()
    }
}

/// Metal GPU backend for matrix multiplication and element-wise operations
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct MetalBackend {
    device: MetalDevice,
    command_queue: CommandQueue,
    buffer_cache: Arc<std::sync::Mutex<BufferCache>>,
    // Cached compiled pipelines (avoid recompilation overhead)
    matmul_pipeline: Arc<metal::ComputePipelineState>,
    gelu_pipeline: Arc<metal::ComputePipelineState>,
    layernorm_pipeline: Arc<metal::ComputePipelineState>,
    rope_pipeline: Arc<metal::ComputePipelineState>,
    softmax_causal_pipeline: Arc<metal::ComputePipelineState>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Create a new Metal backend
    pub fn new() -> Result<Self> {
        let device = MetalDevice::system_default().ok_or_else(|| {
            TrustformersError::hardware_error("No Metal device found", "MetalBackend::new")
        })?;

        let command_queue = device.new_command_queue();

        // Compile shaders ONCE during initialization (critical optimization!)
        let shader_source = r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void matmul(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint row = gid.y;
                uint col = gid.x;

                if (row >= M || col >= N) return;

                float sum = 0.0;
                for (uint i = 0; i < K; ++i) {
                    sum += a[row * K + i] * b[i * N + col];
                }
                c[row * N + col] = sum;
            }

            // GELU activation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            // With NaN guarding for extreme values
            kernel void gelu(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& size [[buffer(2)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= size) return;
                float x = input[gid];

                // Clamp extreme values to prevent NaN
                if (x > 10.0f) {
                    output[gid] = x;  // GELU(x) ≈ x for large positive x
                    return;
                } else if (x < -10.0f) {
                    output[gid] = 0.0f;  // GELU(x) ≈ 0 for large negative x
                    return;
                }

                float x_cubed = x * x * x;
                // sqrt(2/π) ≈ 0.7978845608
                float inner = 0.7978845608f * (x + 0.044715f * x_cubed);

                // Clamp inner to prevent tanh overflow
                inner = clamp(inner, -20.0f, 20.0f);

                output[gid] = 0.5f * x * (1.0f + tanh(inner));
            }

            // RoPE (Rotary Position Embedding)
            // Rotates pairs (i, i+D/2) for i in 0..D/2 where D is rotary_ndims
            // Input/output: [seq_len, num_heads, head_dim]
            kernel void rope(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& seq_len [[buffer(2)]],
                constant uint& num_heads [[buffer(3)]],
                constant uint& head_dim [[buffer(4)]],
                constant uint& rotary_ndims [[buffer(5)]],
                constant float& base [[buffer(6)]],
                uint3 gid [[thread_position_in_grid]]
            ) {
                uint pos = gid.z;
                uint h = gid.y;
                uint i = gid.x;

                if (pos >= seq_len || h >= num_heads || i >= (rotary_ndims / 2)) return;

                uint j = i + (rotary_ndims / 2);

                // Calculate rotation angle
                float freq = 1.0f / pow(base, 2.0f * float(i) / float(rotary_ndims));
                float angle = float(pos) * freq;
                float cos_val = cos(angle);
                float sin_val = sin(angle);

                // Get input indices
                uint idx_i = pos * (num_heads * head_dim) + h * head_dim + i;
                uint idx_j = pos * (num_heads * head_dim) + h * head_dim + j;

                // Apply rotation: (x_i, x_j) → (x_i*cos - x_j*sin, x_i*sin + x_j*cos)
                float x_i = input[idx_i];
                float x_j = input[idx_j];

                output[idx_i] = x_i * cos_val - x_j * sin_val;
                output[idx_j] = x_i * sin_val + x_j * cos_val;

                // Copy non-rotated dimensions
                if (i == 0) {
                    for (uint k = rotary_ndims; k < head_dim; ++k) {
                        uint idx_k = pos * (num_heads * head_dim) + h * head_dim + k;
                        output[idx_k] = input[idx_k];
                    }
                }
            }

            // Softmax with causal mask for attention
            // Input: [seq_len, seq_len] attention scores
            // Output: [seq_len, seq_len] attention weights
            // Applies causal mask: position i can only attend to j <= i
            kernel void softmax_causal(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& seq_len [[buffer(2)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= seq_len) return;

                uint row = gid;
                uint offset = row * seq_len;

                // Find max for numerical stability (only consider j <= row for causal mask)
                float max_val = -3.402823466e+38f;  // -FLT_MAX
                for (uint j = 0; j <= row; ++j) {
                    max_val = max(max_val, input[offset + j]);
                }

                // Handle edge case: if all values are -inf (shouldn't happen but be safe)
                if (max_val < -1e38f) {
                    for (uint j = 0; j < seq_len; ++j) {
                        output[offset + j] = (j == 0) ? 1.0f : 0.0f;
                    }
                    return;
                }

                // Compute exp and sum
                float sum = 0.0f;
                for (uint j = 0; j <= row; ++j) {
                    sum += exp(input[offset + j] - max_val);
                }

                // Handle degenerate case
                if (sum < 1e-10f) {
                    for (uint j = 0; j < seq_len; ++j) {
                        output[offset + j] = (j == 0) ? 1.0f : 0.0f;
                    }
                    return;
                }

                // Normalize and apply causal mask
                for (uint j = 0; j < seq_len; ++j) {
                    if (j <= row) {
                        output[offset + j] = exp(input[offset + j] - max_val) / sum;
                    } else {
                        output[offset + j] = 0.0f;  // Causal mask
                    }
                }
            }

            // LayerNorm: output = (x - mean) / sqrt(var + eps) * weight + bias
            // Optimized for transformer layers: normalize over last dimension (hidden_size)
            kernel void layernorm(
                device const float* input [[buffer(0)]],
                device const float* weight [[buffer(1)]],
                device const float* bias [[buffer(2)]],
                device float* output [[buffer(3)]],
                constant uint& seq_len [[buffer(4)]],
                constant uint& hidden_size [[buffer(5)]],
                constant float& eps [[buffer(6)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= seq_len) return;

                // Each thread processes one sequence position
                uint offset = gid * hidden_size;

                // Compute mean
                float sum = 0.0f;
                for (uint i = 0; i < hidden_size; ++i) {
                    sum += input[offset + i];
                }
                float mean = sum / float(hidden_size);

                // Compute variance
                float var_sum = 0.0f;
                for (uint i = 0; i < hidden_size; ++i) {
                    float diff = input[offset + i] - mean;
                    var_sum += diff * diff;
                }
                float variance = var_sum / float(hidden_size);
                float std_dev = sqrt(variance + eps);

                // Normalize and apply affine transform
                for (uint i = 0; i < hidden_size; ++i) {
                    float normalized = (input[offset + i] - mean) / std_dev;
                    output[offset + i] = normalized * weight[i] + bias[i];
                }
            }
        "#;

        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to compile Metal shader: {}", e),
                    "MetalBackend::new",
                )
            })?;

        let kernel = library.get_function("matmul", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;

        let matmul_pipeline =
            device.new_compute_pipeline_state_with_function(&kernel).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create matmul pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;

        // Compile GELU kernel
        let gelu_kernel = library.get_function("gelu", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get gelu kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;

        let gelu_pipeline =
            device.new_compute_pipeline_state_with_function(&gelu_kernel).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create gelu pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;

        // Compile LayerNorm kernel
        let layernorm_kernel = library.get_function("layernorm", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get layernorm kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;

        let layernorm_pipeline = device
            .new_compute_pipeline_state_with_function(&layernorm_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create layernorm pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;

        // Compile RoPE kernel
        let rope_kernel = library.get_function("rope", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get rope kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;

        let rope_pipeline =
            device.new_compute_pipeline_state_with_function(&rope_kernel).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create rope pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;

        // Compile Softmax (causal) kernel
        let softmax_causal_kernel = library.get_function("softmax_causal", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get softmax_causal kernel function: {}", e),
                "MetalBackend::new",
            )
        })?;

        let softmax_causal_pipeline = device
            .new_compute_pipeline_state_with_function(&softmax_causal_kernel)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create softmax_causal pipeline: {}", e),
                    "MetalBackend::new",
                )
            })?;

        Ok(Self {
            device,
            command_queue,
            buffer_cache: Arc::new(std::sync::Mutex::new(BufferCache::new())),
            matmul_pipeline: Arc::new(matmul_pipeline),
            gelu_pipeline: Arc::new(gelu_pipeline),
            layernorm_pipeline: Arc::new(layernorm_pipeline),
            rope_pipeline: Arc::new(rope_pipeline),
            softmax_causal_pipeline: Arc::new(softmax_causal_pipeline),
        })
    }

    /// Create a persistent GPU buffer and return its ID
    pub fn create_persistent_buffer(&self, data: &[f32]) -> Result<BufferId> {
        let buffer = Arc::new(self.create_buffer(data)?);
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
    pub fn get_persistent_buffer(&self, id: &BufferId) -> Result<Arc<Buffer>> {
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

    /// Perform matrix multiplication on Metal GPU
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Create Metal buffers
        let a_buffer = self.create_buffer(a)?;
        let b_buffer = self.create_buffer(b)?;

        let result_size = m * n;
        let c_buffer = self.device.new_buffer(
            (result_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Use pre-compiled pipeline (no shader compilation overhead!)

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&*self.matmul_pipeline);
        encoder.set_buffer(0, Some(&a_buffer), 0);
        encoder.set_buffer(1, Some(&b_buffer), 0);
        encoder.set_buffer(2, Some(&c_buffer), 0);

        // Set dimensions as constants
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &m_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &n_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const _,
        );

        // Dispatch threads
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };

        let threadgroups = metal::MTLSize {
            width: (n as u64 + 15) / 16,
            height: (m as u64 + 15) / 16,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        // Execute
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to CPU
        let result_ptr = c_buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, result_size) }.to_vec();

        Ok(result)
    }

    /// Perform matrix multiplication with cached weight buffer
    /// This avoids transferring weight data on each forward pass
    pub fn matmul_with_cached_weight(
        &self,
        a: &[f32],
        weight_buffer_id: &BufferId,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Get cached weight buffer
        let b_buffer = self.get_persistent_buffer(weight_buffer_id)?;

        // Create buffer for input (activations change on each forward pass)
        let a_buffer = self.create_buffer(a)?;

        let result_size = m * n;
        let c_buffer = self.device.new_buffer(
            (result_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Use pre-compiled pipeline (no shader compilation overhead!)

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&*self.matmul_pipeline);
        encoder.set_buffer(0, Some(&a_buffer), 0);
        encoder.set_buffer(1, Some(&*b_buffer), 0); // Use cached weight buffer
        encoder.set_buffer(2, Some(&c_buffer), 0);

        // Set dimensions as constants
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &m_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &n_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const _,
        );

        // Dispatch threads
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };

        let threadgroups = metal::MTLSize {
            width: (n as u64 + 15) / 16,
            height: (m as u64 + 15) / 16,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        // Execute
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to CPU
        let result_ptr = c_buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, result_size) }.to_vec();

        Ok(result)
    }

    /// Execute GELU activation on GPU
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    pub fn gelu_f32(&self, input: &[f32]) -> Result<Vec<f32>> {
        let size = input.len();

        // Create Metal buffers
        let input_buffer = self.create_buffer(input)?;
        let output_buffer = self.device.new_buffer(
            (size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Use pre-compiled pipeline
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&*self.gelu_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        // Set size parameter
        let size_u32 = size as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &size_u32 as *const u32 as *const _,
        );

        // Dispatch threads (1D grid for element-wise operation)
        let threadgroup_size = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let threadgroups = metal::MTLSize {
            width: (size as u64 + 255) / 256,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        // Execute
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to CPU
        let result_ptr = output_buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, size) }.to_vec();

        Ok(result)
    }

    /// Execute GELU on GPU buffer → GPU buffer (ZERO CPU TRANSFERS!)
    /// Input and output stay on GPU
    pub fn gelu_gpu_to_gpu(&self, input_buffer_id: &BufferId, size: usize) -> Result<BufferId> {
        // Get input buffer
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;

        // Create output buffer
        let output_buffer = self.device.new_buffer(
            (size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Use pre-compiled pipeline
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&*self.gelu_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        let size_u32 = size as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &size_u32 as *const u32 as *const _,
        );

        // Dispatch threads
        let threadgroup_size = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (size as u64 + 255) / 256,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Store output buffer and return ID
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "gelu_gpu_to_gpu")
        })?;
        cache.insert(output_id, output_buffer_arc);

        Ok(output_id)
    }

    /// Execute LayerNorm GPU-to-GPU (ZERO CPU transfers!)
    /// Input, weight, bias, and output all stay on GPU
    /// LayerNorm: output = (x - mean) / sqrt(var + eps) * weight + bias
    pub fn layernorm_gpu_to_gpu(
        &self,
        input_buffer_id: &BufferId,
        weight_buffer_id: &BufferId,
        bias_buffer_id: &BufferId,
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<BufferId> {
        let total_size = seq_len * hidden_size;

        // Get input buffers
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let weight_buffer = self.get_persistent_buffer(weight_buffer_id)?;
        let bias_buffer = self.get_persistent_buffer(bias_buffer_id)?;

        // Create output buffer
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Use pre-compiled pipeline
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&*self.layernorm_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&*weight_buffer), 0);
        encoder.set_buffer(2, Some(&*bias_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        // Set parameters
        let seq_len_u32 = seq_len as u32;
        let hidden_size_u32 = hidden_size as u32;
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &seq_len_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &hidden_size_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<f32>() as u64,
            &eps as *const f32 as *const _,
        );

        // Dispatch threads (one thread per sequence position)
        let threadgroup_size = metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (seq_len as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Store output buffer and return ID
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "layernorm_gpu_to_gpu")
        })?;
        cache.insert(output_id, output_buffer_arc);

        Ok(output_id)
    }

    /// Execute matrix multiplication GPU-to-GPU (ZERO CPU transfers!)
    /// Input, weight, and output all stay on GPU
    /// Performs C = A × B where:
    /// - A has shape [m, k] (input activations)
    /// - B has shape [k, n] (weight matrix, already transposed and cached)
    /// - C has shape [m, n] (output)
    pub fn matmul_gpu_to_gpu(
        &self,
        input_buffer_id: &BufferId,
        weight_buffer_id: &BufferId,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<BufferId> {
        // Get input buffers
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let weight_buffer = self.get_persistent_buffer(weight_buffer_id)?;

        // Create output buffer
        let result_size = m * n;
        let output_buffer = self.device.new_buffer(
            (result_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Use pre-compiled matmul pipeline
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&*self.matmul_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&*weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);

        // Set dimensions as constants
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &m_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &n_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const _,
        );

        // Dispatch threads - 2D grid for matrix multiplication
        // Each thread computes one element of output matrix
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (n as u64 + 15) / 16,
            height: (m as u64 + 15) / 16,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Store output buffer and return ID
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "matmul_gpu_to_gpu")
        })?;
        cache.insert(output_id, output_buffer_arc);

        Ok(output_id)
    }

    /// Execute LayerNorm on GPU
    /// LayerNorm: output = (x - mean) / sqrt(var + eps) * weight + bias
    /// Optimized for transformer models (normalize over hidden dimension)
    pub fn layernorm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
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

        // Create Metal buffers
        let input_buffer = self.create_buffer(input)?;
        let weight_buffer = self.create_buffer(weight)?;
        let bias_buffer = self.create_buffer(bias)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Use pre-compiled pipeline
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&*self.layernorm_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&bias_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        // Set parameters
        let seq_len_u32 = seq_len as u32;
        let hidden_size_u32 = hidden_size as u32;
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &seq_len_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &hidden_size_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<f32>() as u64,
            &eps as *const f32 as *const _,
        );

        // Dispatch threads (one thread per sequence position)
        let threadgroup_size = metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };

        let threadgroups = metal::MTLSize {
            width: (seq_len as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        // Execute
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to CPU
        let result_ptr = output_buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, total_size) }.to_vec();

        Ok(result)
    }

    /// Execute RoPE (Rotary Position Embedding) on GPU
    /// Rotates Q and K tensors for position encoding
    /// Input/output shape: [seq_len, num_heads, head_dim]
    pub fn rope_f32(
        &self,
        input: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_ndims: usize,
        base: f32,
    ) -> Result<Vec<f32>> {
        let total_size = seq_len * num_heads * head_dim;

        if input.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input size {} doesn't match seq_len {} * num_heads {} * head_dim {}",
                input.len(),
                seq_len,
                num_heads,
                head_dim
            )));
        }

        // Create Metal buffers
        let input_buffer = self.create_buffer(input)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Use pre-compiled pipeline
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&*self.rope_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        // Set parameters
        let seq_len_u32 = seq_len as u32;
        let num_heads_u32 = num_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let rotary_ndims_u32 = rotary_ndims as u32;

        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &seq_len_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &num_heads_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &head_dim_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &rotary_ndims_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<f32>() as u64,
            &base as *const f32 as *const _,
        );

        // Dispatch 3D grid: (rotary_ndims/2, num_heads, seq_len)
        let threadgroup_size = metal::MTLSize {
            width: 8,  // rotary_ndims/2 dimension
            height: 4, // num_heads dimension
            depth: 4,  // seq_len dimension
        };

        let threadgroups = metal::MTLSize {
            width: ((rotary_ndims / 2) as u64 + 7) / 8,
            height: (num_heads as u64 + 3) / 4,
            depth: (seq_len as u64 + 3) / 4,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        // Execute
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to CPU
        let result_ptr = output_buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, total_size) }.to_vec();

        Ok(result)
    }

    /// Execute Softmax with causal mask on GPU
    /// Applies causal mask: position i can only attend to j <= i
    /// Input/output shape: [seq_len, seq_len]
    pub fn softmax_causal_f32(&self, input: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let total_size = seq_len * seq_len;

        if input.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input size {} doesn't match seq_len^2 {}",
                input.len(),
                total_size
            )));
        }

        // Create Metal buffers
        let input_buffer = self.create_buffer(input)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Use pre-compiled pipeline
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&*self.softmax_causal_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        let seq_len_u32 = seq_len as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &seq_len_u32 as *const u32 as *const _,
        );

        // Dispatch threads (one thread per row)
        let threadgroup_size = metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };

        let threadgroups = metal::MTLSize {
            width: (seq_len as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        // Execute
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back to CPU
        let result_ptr = output_buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, total_size) }.to_vec();

        Ok(result)
    }

    fn create_buffer(&self, data: &[f32]) -> Result<Buffer> {
        let byte_size = std::mem::size_of_val(data) as u64;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_size,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }
}

/// Global Metal backend cache
#[cfg(all(target_os = "macos", feature = "metal"))]
static METAL_BACKEND: once_cell::sync::Lazy<std::sync::Mutex<Option<MetalBackend>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(None));

/// Get or create Metal backend instance
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn get_metal_backend() -> Result<MetalBackend> {
    let mut cache = METAL_BACKEND.lock().map_err(|_| {
        TrustformersError::hardware_error("Failed to lock Metal backend cache", "get_metal_backend")
    })?;

    if cache.is_none() {
        *cache = Some(MetalBackend::new()?);
    }

    // Clone the cached backend (device and command queue are Arc-based internally)
    cache
        .as_ref()
        .ok_or_else(|| {
            TrustformersError::hardware_error("Metal backend not initialized", "get_metal_backend")
        })
        .and_then(|backend| {
            Ok(MetalBackend {
                device: backend.device.clone(),
                command_queue: backend.command_queue.clone(),
                buffer_cache: Arc::clone(&backend.buffer_cache),
                matmul_pipeline: Arc::clone(&backend.matmul_pipeline),
                gelu_pipeline: Arc::clone(&backend.gelu_pipeline),
                layernorm_pipeline: Arc::clone(&backend.layernorm_pipeline),
                rope_pipeline: Arc::clone(&backend.rope_pipeline),
                softmax_causal_pipeline: Arc::clone(&backend.softmax_causal_pipeline),
            })
        })
}

/// Dispatch matrix multiplication to appropriate backend based on device
#[allow(unused_variables)]
pub fn dispatch_matmul(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Device::Metal(_device_id) = device {
            match (a, b) {
                (Tensor::F32(a_arr), Tensor::F32(b_arr)) => {
                    // Convert to 2D arrays
                    if a_arr.ndim() != 2 || b_arr.ndim() != 2 {
                        return Err(TrustformersError::shape_error(
                            "Metal dispatch currently only supports 2D tensors".to_string(),
                        ));
                    }

                    let a_2d = a_arr
                        .clone()
                        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to convert to 2D: {}",
                                e
                            ))
                        })?;
                    let b_2d = b_arr
                        .clone()
                        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to convert to 2D: {}",
                                e
                            ))
                        })?;

                    let (m, k) = a_2d.dim();
                    let (k2, n) = b_2d.dim();

                    if k != k2 {
                        return Err(TrustformersError::shape_error(format!(
                            "Matrix dimension mismatch: {}×{} vs {}×{}",
                            m, k, k2, n
                        )));
                    }

                    // Get Metal backend
                    let backend = get_metal_backend()?;

                    // Convert to contiguous slices
                    let a_data: Vec<f32> = a_2d.iter().copied().collect();
                    let b_data: Vec<f32> = b_2d.iter().copied().collect();

                    // Execute Metal matmul
                    let result_data = backend.matmul_f32(&a_data, &b_data, m, k, n)?;

                    // Convert back to tensor
                    let result_2d = scirs2_core::ndarray::Array2::from_shape_vec(
                        (m, n),
                        result_data,
                    )
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
    }

    // Default: CPU matmul (or if Metal not available/configured)
    a.matmul(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_matmul_cpu() -> Result<()> {
        let a = Tensor::randn(&[2, 3])?;
        let b = Tensor::randn(&[3, 4])?;

        let c = dispatch_matmul(&a, &b, &Device::CPU)?;

        assert_eq!(c.shape(), &[2, 4]);
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_dispatch_matmul_metal() -> Result<()> {
        let a = Tensor::randn(&[2, 3])?;
        let b = Tensor::randn(&[3, 4])?;

        let c = dispatch_matmul(&a, &b, &Device::Metal(0))?;

        assert_eq!(c.shape(), &[2, 4]);
        Ok(())
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_metal_backend_correctness() -> Result<()> {
        // Test that Metal matmul produces correct results
        let backend = MetalBackend::new()?;

        // Simple 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]

        let result = backend.matmul_f32(&a, &b, 2, 2, 2)?;

        // Expected: [[19, 22], [43, 50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                res,
                exp
            );
        }

        Ok(())
    }
}
