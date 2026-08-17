use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;

/// Standalone Metal compute kernels for a handful of tensor primitives.
///
/// # Scope, honestly stated
///
/// This is a small, self-contained set of hand-written MSL kernels compiled at
/// runtime. It is **not** the crate's main GPU path: `gpu_ops::metal::MetalBackend`
/// is, and it is what `Device::Metal` dispatches to. Keep that in mind before
/// reaching for this type.
///
/// What it actually provides:
/// * `matrix_multiply` - a single 2-D `f32` GEMM (batched inputs are rejected, not
///   silently flattened).
/// * `add_tensors` - elementwise addition.
/// * `gelu` - elementwise tanh-approximation GELU.
/// * `flash_attention` - a fused single-pass attention kernel (see its own docs for
///   what it does and does not do).
///
/// What it does **not** provide, despite what these docs used to claim:
/// * Metal Performance Shaders. Nothing here uses MPS or MPSGraph. An `MPSGraph` was
///   constructed in the constructor and immediately dropped; it is gone.
/// * "Native integration with Core ML and Apple's ML frameworks". There is none.
///
/// # Requirements
/// * macOS with the `metal` cargo feature (the whole module is gated on both).
pub struct MetalImpl {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    device: metal::Device,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    command_queue: metal::CommandQueue,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    library: metal::Library,
}

impl MetalImpl {
    /// Create new Metal implementation
    pub fn new() -> Result<Self> {
        Self::new_with_metal()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn new_with_metal() -> Result<Self> {
        // Create Metal device (automatically selects the best available GPU)
        let device = metal::Device::system_default().ok_or_else(|| {
            TrustformersError::hardware_error("No Metal-compatible device found", "MetalImpl::new")
        })?;

        // Verify Metal Performance Shaders support
        if !Self::supports_mps(&device) {
            return Err(TrustformersError::hardware_error(
                "Metal Performance Shaders not supported on this device",
                "MetalImpl::new",
            ));
        }

        // Create command queue for submitting GPU work
        let command_queue = device.new_command_queue();

        // Create compute library with custom kernels
        let library = Self::create_kernel_library(&device)?;

        log::info!(
            "Metal backend initialized successfully on device: {}",
            device.name()
        );
        log::info!(
            "Unified memory size: {} GB",
            device.recommended_max_working_set_size() / (1024 * 1024 * 1024)
        );

        Ok(Self {
            device,
            command_queue,
            library,
        })
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[allow(deprecated)]
    fn supports_mps(device: &metal::Device) -> bool {
        // Check for Metal Performance Shaders support
        // Available on macOS 10.15+, iOS 13+, and Apple Silicon
        //
        // Note: MTLFeatureSet is deprecated in favor of MTLGPUFamily API in Metal 3.0+
        // However, MTLGPUFamily requires Metal 3.0 (macOS 13+, iOS 16+) which would
        // limit compatibility. We use the deprecated API to support a broader range of devices.
        //
        // For modern devices (macOS 13+), consider using:
        //   device.supports_family(metal::MTLGPUFamily::Apple7) || // M1/M2
        //   device.supports_family(metal::MTLGPUFamily::Apple8) || // M3
        //   device.supports_family(metal::MTLGPUFamily::Mac2)      // Intel
        //
        // Current implementation maintains compatibility with macOS 10.15+
        device.supports_feature_set(metal::MTLFeatureSet::macOS_GPUFamily2_v1)
            || device.supports_feature_set(metal::MTLFeatureSet::iOS_GPUFamily4_v1)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn create_kernel_library(device: &metal::Device) -> Result<metal::Library> {
        // Metal Shading Language source for custom kernels
        let kernel_source = r#"
            #include <metal_stdlib>
            using namespace metal;

            // Fast matrix multiplication kernel optimized for Apple Silicon
            kernel void matrix_multiply_f32(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* result [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                uint2 thread_position [[thread_position_in_grid]]
            ) {
                uint row = thread_position.y;
                uint col = thread_position.x;

                if (row >= M || col >= N) return;

                float sum = 0.0;
                for (uint i = 0; i < K; ++i) {
                    sum += a[row * K + i] * b[i * N + col];
                }
                result[row * N + col] = sum;
            }

            // Element-wise addition with broadcasting support
            kernel void add_tensors_f32(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* result [[buffer(2)]],
                constant uint& size [[buffer(3)]],
                uint index [[thread_position_in_grid]]
            ) {
                if (index >= size) return;
                result[index] = a[index] + b[index];
            }

            // ReLU activation function
            kernel void relu_f32(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& size [[buffer(2)]],
                uint index [[thread_position_in_grid]]
            ) {
                if (index >= size) return;
                output[index] = max(0.0f, input[index]);
            }

            // GELU activation function
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            kernel void gelu_f32(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& size [[buffer(2)]],
                uint index [[thread_position_in_grid]]
            ) {
                if (index >= size) return;
                float x = input[index];
                float x_cubed = x * x * x;
                // sqrt(2/π) ≈ 0.7978845608
                float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
                output[index] = 0.5f * x * (1.0f + tanh(inner));
            }

            // Optimized softmax for attention mechanisms
            kernel void softmax_f32(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& batch_size [[buffer(2)]],
                constant uint& seq_len [[buffer(3)]],
                uint2 thread_position [[thread_position_in_grid]]
            ) {
                uint batch = thread_position.y;
                uint seq = thread_position.x;

                if (batch >= batch_size || seq >= seq_len) return;

                uint offset = batch * seq_len;

                // Find maximum value for numerical stability
                float max_val = input[offset];
                for (uint i = 0; i < seq_len; ++i) {
                    max_val = max(max_val, input[offset + i]);
                }

                // Compute exponentials and sum
                float sum = 0.0;
                for (uint i = 0; i < seq_len; ++i) {
                    sum += exp(input[offset + i] - max_val);
                }

                // Normalize
                output[offset + seq] = exp(input[offset + seq] - max_val) / sum;
            }

            // Optimized Flash Attention kernel for transformer models
            // Implements: output = softmax(Q @ K^T / sqrt(d_k)) @ V
            // Optimized for memory efficiency with tiling and fused operations
            kernel void flash_attention_f32(
                device const float* query [[buffer(0)]],     // [batch, seq_q, dim]
                device const float* key [[buffer(1)]],       // [batch, seq_k, dim]
                device const float* value [[buffer(2)]],     // [batch, seq_k, dim_v]
                device float* output [[buffer(3)]],          // [batch, seq_q, dim_v]
                constant uint& batch_size [[buffer(4)]],
                constant uint& seq_q [[buffer(5)]],
                constant uint& seq_k [[buffer(6)]],
                constant uint& dim [[buffer(7)]],
                constant uint& dim_v [[buffer(8)]],
                constant float& scale [[buffer(9)]],
                uint3 thread_position [[thread_position_in_grid]]
            ) {
                uint b = thread_position.z;  // batch
                uint q_idx = thread_position.y;  // query position
                uint v_idx = thread_position.x;  // value dimension

                if (b >= batch_size || q_idx >= seq_q || v_idx >= dim_v) return;

                // Calculate attention scores: Q[q_idx] @ K^T
                // Q[q_idx] shape: [dim], K^T shape: [dim, seq_k]
                float max_score = -INFINITY;
                float scores[1024];  // Assuming seq_k <= 1024, adjust if needed

                uint q_offset = (b * seq_q + q_idx) * dim;

                for (uint k_idx = 0; k_idx < seq_k; ++k_idx) {
                    uint k_offset = (b * seq_k + k_idx) * dim;

                    // Compute dot product Q[q_idx] · K[k_idx]
                    float score = 0.0;
                    for (uint d = 0; d < dim; ++d) {
                        score += query[q_offset + d] * key[k_offset + d];
                    }

                    // Scale by sqrt(d_k)
                    score *= scale;
                    scores[k_idx] = score;

                    // Track maximum for numerical stability
                    max_score = max(max_score, score);
                }

                // Compute softmax: exp(score - max_score) / sum
                float sum_exp = 0.0;
                for (uint k_idx = 0; k_idx < seq_k; ++k_idx) {
                    scores[k_idx] = exp(scores[k_idx] - max_score);
                    sum_exp += scores[k_idx];
                }

                // Normalize attention weights
                for (uint k_idx = 0; k_idx < seq_k; ++k_idx) {
                    scores[k_idx] /= sum_exp;
                }

                // Apply attention to values: attention_weights @ V
                // Result: output[q_idx, v_idx] = sum_k(attention[k] * V[k, v_idx])
                float result = 0.0;
                for (uint k_idx = 0; k_idx < seq_k; ++k_idx) {
                    uint v_offset = (b * seq_k + k_idx) * dim_v + v_idx;
                    result += scores[k_idx] * value[v_offset];
                }

                // Write output
                uint out_offset = (b * seq_q + q_idx) * dim_v + v_idx;
                output[out_offset] = result;
            }
        "#;

        let compile_options = metal::CompileOptions::new();
        device.new_library_with_source(kernel_source, &compile_options).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compile Metal kernels: {}", e),
                "MetalImpl::create_kernel_library",
            )
        })
    }

    /// Execute matrix multiplication using custom Metal kernel
    ///
    /// Implements optimized matrix multiplication for Apple Silicon using custom Metal compute shaders.
    /// This implementation uses the pre-compiled matrix_multiply_f32 kernel which is optimized for
    /// Apple's unified memory architecture.
    ///
    /// Algorithm: result\[i,j\] = sum_k(a\[i,k\] * b\[k,j\]) for all i in \[0,M), j in \[0,N)
    ///
    /// # Performance
    /// - Utilizes GPU parallelization with 2D thread grid
    /// - Memory-efficient with unified memory on Apple Silicon
    /// - Batching handled via thread groups for optimal throughput
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn matrix_multiply(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Validate input tensors
        let a_shape = a.shape();
        let b_shape = b.shape();

        // The `matrix_multiply_f32` kernel indexes `a[row * K + i]` over a flat buffer
        // and the result is declared `[a_rows, b_cols]`, so it implements a *single*
        // 2-D GEMM. The previous check accepted `len() >= 2`, took the trailing two
        // dimensions and silently dropped every batch dimension - a 3-D input produced
        // numerically wrong output with a 2-D shape and no error at all. Reject
        // anything the kernel cannot actually compute.
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "MetalImpl::matrix_multiply implements a single 2-D GEMM; got shapes \
                     {a_shape:?} @ {b_shape:?}. Use `gpu_ops::metal::MetalBackend::\
                     batched_matmul_gpu_to_gpu` for batched inputs."
                ),
                "MetalImpl::matrix_multiply",
            ));
        }

        let a_rows = a_shape[0];
        let a_cols = a_shape[1];
        let b_rows = b_shape[0];
        let b_cols = b_shape[1];

        if a_cols != b_rows {
            return Err(
                crate::errors::shape_mismatch(vec![a_rows, a_cols], vec![b_rows, b_cols])
                    .with_operation("MetalImpl::matrix_multiply"),
            );
        }

        // Create Metal buffers from tensors
        let buffer_a = a.to_metal_buffer(&self.device)?;
        let buffer_b = b.to_metal_buffer(&self.device)?;

        // Allocate result buffer
        let result_size = a_rows * b_cols;
        let result_buffer = self.device.new_buffer(
            (result_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get matrix multiplication kernel function
        let function = self.library.get_function("matrix_multiply_f32", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get matrix_multiply_f32 kernel: {}", e),
                "MetalImpl::matrix_multiply",
            )
        })?;

        // Create compute pipeline
        let pipeline =
            self.device.new_compute_pipeline_state_with_function(&function).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create compute pipeline: {}", e),
                    "MetalImpl::matrix_multiply",
                )
            })?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline and buffers
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0); // Input matrix A
        encoder.set_buffer(1, Some(&buffer_b), 0); // Input matrix B
        encoder.set_buffer(2, Some(&result_buffer), 0); // Output result

        // Set matrix dimensions as parameters
        let m = a_rows as u32;
        let n = b_cols as u32;
        let k = a_cols as u32;
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const std::ffi::c_void,
        );

        // Configure thread execution
        // Use 2D grid: (N columns, M rows) for output matrix
        let thread_group_size = metal::MTLSize::new(16, 16, 1); // 16x16 thread block
        let thread_groups = metal::MTLSize::new(
            b_cols.div_ceil(16) as u64, // Number of column blocks
            a_rows.div_ceil(16) as u64, // Number of row blocks
            1,
        );

        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        // Execute and wait for completion
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Convert result buffer back to Tensor
        let result_shape = vec![a_rows, b_cols];
        Tensor::from_metal_buffer(&result_buffer, &result_shape)
    }

    /// Execute element-wise addition using custom Metal kernel
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn add_tensors(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            return Err(TrustformersError::tensor_op_error(
                "Tensor shapes must match for addition",
                "MetalImpl::add_tensors",
            ));
        }

        let size = a.len();
        let result_shape = a.shape().to_vec();

        // Create Metal buffers
        let buffer_a = a.to_metal_buffer(&self.device)?;
        let buffer_b = b.to_metal_buffer(&self.device)?;
        let result_buffer = self.device.new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get compute function
        let function = self.library.get_function("add_tensors_f32", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get add_tensors kernel: {}", e),
                "MetalImpl::add_tensors",
            )
        })?;

        let pipeline =
            self.device.new_compute_pipeline_state_with_function(&function).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create compute pipeline: {}", e),
                    "MetalImpl::add_tensors",
                )
            })?;

        // Execute kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0);
        let size_u32 = size as u32;
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &size_u32 as *const u32 as *const std::ffi::c_void,
        );

        let thread_group_size = metal::MTLSize::new(256, 1, 1);
        let thread_groups = metal::MTLSize::new(size.div_ceil(256) as u64, 1, 1);

        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Convert result back to Tensor
        Tensor::from_metal_buffer(&result_buffer, &result_shape)
    }

    /// Execute GELU activation function using Metal kernel
    ///
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    ///
    /// This is a GPU-accelerated elementwise operation optimized for Apple Silicon.
    pub fn gelu(&self, input: &Tensor) -> Result<Tensor> {
        let size = input.len();
        let result_shape = input.shape().to_vec();

        // Create Metal buffers
        let input_buffer = input.to_metal_buffer(&self.device)?;
        let output_buffer = self.device.new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get compute function
        let function = self.library.get_function("gelu_f32", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get gelu kernel: {}", e),
                "MetalImpl::gelu",
            )
        })?;

        let pipeline =
            self.device.new_compute_pipeline_state_with_function(&function).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create compute pipeline: {}", e),
                    "MetalImpl::gelu",
                )
            })?;

        // Execute kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let size_u32 = size as u32;
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &size_u32 as *const u32 as *const std::ffi::c_void,
        );

        let thread_group_size = metal::MTLSize::new(256, 1, 1);
        let thread_groups = metal::MTLSize::new(size.div_ceil(256) as u64, 1, 1);

        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Convert result back to Tensor
        Tensor::from_metal_buffer(&output_buffer, &result_shape)
    }

    /// Execute Flash Attention using custom Metal kernel
    ///
    /// Implements memory-efficient flash attention optimized for Apple Silicon.
    /// This fused kernel implementation reduces memory bandwidth requirements and
    /// improves performance compared to separate matrix multiplication operations.
    ///
    /// Algorithm: output = softmax(Q @ K^T / sqrt(d_k)) @ V
    ///
    /// # What this kernel is
    ///
    /// A *fused, non-tiled* attention kernel: one GPU thread per `(query, dim_v)`
    /// pair recomputes the full score row for its query and applies a numerically
    /// stable (max-subtracted) softmax. It avoids materialising the attention matrix
    /// in global memory, which is where the memory saving comes from.
    ///
    /// It is **not** FlashAttention in the tiled, IO-aware sense despite the name:
    /// there is no KV blocking and the score row is recomputed once per `dim_v`
    /// thread, so compute scales with `seq_k * dim_v` rather than `seq_k`. The
    /// tiled implementation lives in
    /// `gpu_ops::metal::MetalBackend::flash_attention_with_cache`.
    ///
    /// The kernel keeps a `float scores[1024]` thread-private array and refuses any
    /// `seq_k > 1024` rather than overrunning it.
    ///
    /// # Arguments
    /// - `query`: Query tensor [batch, seq_q, dim]
    /// - `key`: Key tensor [batch, seq_k, dim]
    /// - `value`: Value tensor [batch, seq_k, dim_v]
    /// - `output`: Output tensor (modified in-place) [batch, seq_q, dim_v]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn flash_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
    ) -> Result<()> {
        // Validate input tensors
        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        if q_shape.len() < 3 || k_shape.len() < 3 || v_shape.len() < 3 {
            return Err(TrustformersError::tensor_op_error(
                "Flash attention requires at least 3D tensors [batch, seq_len, dim]",
                "MetalImpl::flash_attention",
            ));
        }

        let batch_size = q_shape[0];
        let seq_len_q = q_shape[1];
        let dim_q = q_shape[2];
        let seq_len_k = k_shape[1];
        let dim_k = k_shape[2];
        let dim_v = v_shape[2];

        if dim_q != dim_k {
            return Err(crate::errors::shape_mismatch(
                vec![batch_size, seq_len_q, dim_q],
                vec![batch_size, seq_len_k, dim_k],
            )
            .with_operation("MetalImpl::flash_attention"));
        }

        // Validate sequence length doesn't exceed kernel limits
        if seq_len_k > 1024 {
            return Err(TrustformersError::tensor_op_error(
                "Flash attention kernel supports maximum sequence length of 1024. For longer sequences, use chunked processing",
                "MetalImpl::flash_attention",
            ));
        }

        // Create Metal buffers from input tensors
        let buffer_q = query.to_metal_buffer(&self.device)?;
        let buffer_k = key.to_metal_buffer(&self.device)?;
        let buffer_v = value.to_metal_buffer(&self.device)?;

        // Allocate output buffer
        let output_size = batch_size * seq_len_q * dim_v;
        let output_buffer = self.device.new_buffer(
            (output_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get flash attention kernel function
        let function = self.library.get_function("flash_attention_f32", None).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get flash_attention_f32 kernel: {}", e),
                "MetalImpl::flash_attention",
            )
        })?;

        // Create compute pipeline
        let pipeline =
            self.device.new_compute_pipeline_state_with_function(&function).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create compute pipeline: {}", e),
                    "MetalImpl::flash_attention",
                )
            })?;

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline and buffers
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer_q), 0); // Query
        encoder.set_buffer(1, Some(&buffer_k), 0); // Key
        encoder.set_buffer(2, Some(&buffer_v), 0); // Value
        encoder.set_buffer(3, Some(&output_buffer), 0); // Output

        // Set dimensions as parameters
        let batch_u32 = batch_size as u32;
        let seq_q_u32 = seq_len_q as u32;
        let seq_k_u32 = seq_len_k as u32;
        let dim_u32 = dim_q as u32;
        let dim_v_u32 = dim_v as u32;
        let scale = 1.0_f32 / (dim_q as f32).sqrt();

        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &batch_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &seq_q_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &seq_k_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &dim_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            8,
            std::mem::size_of::<u32>() as u64,
            &dim_v_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            9,
            std::mem::size_of::<f32>() as u64,
            &scale as *const f32 as *const std::ffi::c_void,
        );

        // Configure thread execution
        // 3D grid: (dim_v, seq_q, batch)
        let thread_group_size = metal::MTLSize::new(8, 8, 1); // 8x8x1 thread block
        let thread_groups = metal::MTLSize::new(
            dim_v.div_ceil(8) as u64,     // dim_v blocks
            seq_len_q.div_ceil(8) as u64, // seq_q blocks
            batch_size as u64,            // batch blocks
        );

        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        // Execute and wait for completion
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Convert result buffer to Tensor and copy to output
        let result_shape = vec![batch_size, seq_len_q, dim_v];
        let result_tensor = Tensor::from_metal_buffer(&output_buffer, &result_shape)?;
        *output = result_tensor;

        Ok(())
    }

    /// Human-readable summary of the live Metal device.
    pub fn device_info(&self) -> Result<String> {
        Ok(format!(
            "Metal Device: {}\nRecommended working set: {} GB\nMax threads per threadgroup: {}",
            self.device.name(),
            self.device.recommended_max_working_set_size() / (1024 * 1024 * 1024),
            self.device.max_threads_per_threadgroup().width
        ))
    }
}

/// Extensions to Tensor for Metal integration
trait TensorMetalExt {
    fn to_metal_buffer(&self, device: &metal::Device) -> Result<metal::Buffer>;
    fn from_metal_buffer(buffer: &metal::Buffer, shape: &[usize]) -> Result<Tensor>;
}

impl TensorMetalExt for Tensor {
    fn to_metal_buffer(&self, device: &metal::Device) -> Result<metal::Buffer> {
        let data = self.data()?;
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            (data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    fn from_metal_buffer(buffer: &metal::Buffer, shape: &[usize]) -> Result<Tensor> {
        let data_ptr = buffer.contents() as *const f32;
        let len = shape.iter().product::<usize>();
        let data = unsafe { std::slice::from_raw_parts(data_ptr, len) };
        Tensor::from_slice(data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_metal_impl_creation() {
        // Test that MetalImpl can be created (will use mock on non-Metal systems)
        let result = MetalImpl::new();
        assert!(result.is_ok());
    }

    #[test]
    fn test_device_info() {
        let metal_impl = MetalImpl::new().expect("operation failed in test");
        let info = metal_impl.device_info().expect("operation failed in test");
        assert!(!info.is_empty());
    }

    /// Regression: this asserted only the output *shape*, so a kernel returning
    /// arbitrary numbers passed. Check the values against the closed-form product.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_matrix_multiply() {
        let metal_impl = MetalImpl::new().expect("operation failed in test");

        // A = [[1,2],[3,4]], B = [[5,6],[7,8]] -> A@B = [[19,22],[43,50]]
        let a =
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor operation failed");
        let b =
            Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("tensor operation failed");

        let result_tensor = metal_impl.matrix_multiply(&a, &b).expect("matmul failed");
        assert_eq!(result_tensor.shape(), &[2, 2]);
        let actual = result_tensor.data_f32().expect("tensor operation failed");
        for (got, want) in actual.iter().zip([19.0f32, 22.0, 43.0, 50.0].iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "GPU matmul {actual:?} != [19, 22, 43, 50]"
            );
        }
    }

    /// Regression: `matrix_multiply` accepted any tensor with `shape().len() >= 2`,
    /// took the trailing two dims and dropped the batch dimensions - a 3-D input got
    /// a numerically wrong 2-D answer and no error. It must refuse instead.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn matrix_multiply_rejects_batched_inputs_instead_of_flattening() {
        let metal_impl = MetalImpl::new().expect("operation failed in test");
        let a = Tensor::from_slice(&[1.0; 8], &[2, 2, 2]).expect("tensor operation failed");
        let b = Tensor::from_slice(&[1.0; 4], &[2, 2]).expect("tensor operation failed");
        let error = metal_impl
            .matrix_multiply(&a, &b)
            .expect_err("a 3-D operand must be rejected, not silently flattened");
        let rendered = format!("{error}");
        assert!(
            rendered.contains("single 2-D GEMM"),
            "error must explain the kernel's real limitation, got: {rendered}"
        );
    }

    /// GELU had no correctness test at all; assert against the same tanh
    /// approximation the kernel implements.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn gelu_matches_the_tanh_approximation() {
        fn reference(x: f32) -> f32 {
            let inner = 0.7978845608_f32 * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }

        let metal_impl = MetalImpl::new().expect("operation failed in test");
        let inputs: Vec<f32> = vec![-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0];
        let tensor = Tensor::from_slice(&inputs, &[inputs.len()]).expect("tensor operation failed");
        let out = metal_impl.gelu(&tensor).expect("gelu failed");
        assert_eq!(out.shape(), &[inputs.len()]);
        let actual = out.data_f32().expect("tensor operation failed");
        for (index, (got, x)) in actual.iter().zip(inputs.iter()).enumerate() {
            let want = reference(*x);
            assert!(
                (got - want).abs() < 1e-5,
                "gelu[{index}] (x={x}): GPU {got} vs reference {want}"
            );
        }
        // Shape properties a constant or pass-through kernel would violate:
        // GELU(0) == 0 exactly, GELU is strictly increasing for x >= 0, and it dips
        // negative on (-inf, 0) with a minimum near x = -0.75 (it is NOT monotone).
        assert!(
            actual[3].abs() < 1e-6,
            "GELU(0) must be 0, got {}",
            actual[3]
        );
        assert!(
            actual[4..].windows(2).all(|pair| pair[1] > pair[0]),
            "GELU must be strictly increasing for x >= 0, got {:?}",
            &actual[3..]
        );
        assert!(
            actual[1] < 0.0 && actual[2] < 0.0,
            "GELU(-1) and GELU(-0.5) must be negative, got {} and {}",
            actual[1],
            actual[2]
        );
        assert!(
            actual[1] < actual[2],
            "GELU's minimum lies near x = -0.75, so GELU(-1) < GELU(-0.5); got {} vs {}",
            actual[1],
            actual[2]
        );
    }

    /// `flash_attention` had no correctness test. Compare it against a CPU
    /// reference implementation of softmax(QK^T/sqrt(d)) @ V.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn flash_attention_matches_cpu_reference() {
        let metal_impl = MetalImpl::new().expect("operation failed in test");
        let (batch, seq, dim) = (1usize, 3usize, 2usize);
        let q: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
        let k: Vec<f32> = vec![0.7, 0.1, -0.3, 0.2, 0.5, -0.4];
        let v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let q_t = Tensor::from_slice(&q, &[batch, seq, dim]).expect("tensor");
        let k_t = Tensor::from_slice(&k, &[batch, seq, dim]).expect("tensor");
        let v_t = Tensor::from_slice(&v, &[batch, seq, dim]).expect("tensor");
        let mut out = Tensor::zeros(&[batch, seq, dim]).expect("tensor");
        metal_impl
            .flash_attention(&q_t, &k_t, &v_t, &mut out)
            .expect("flash attention failed");
        let actual = out.data_f32().expect("tensor operation failed");

        // CPU reference.
        let scale = 1.0f32 / (dim as f32).sqrt();
        let mut expected = vec![0.0f32; seq * dim];
        for row in 0..seq {
            let mut scores = vec![0.0f32; seq];
            let mut max_score = f32::NEG_INFINITY;
            for (col, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += q[row * dim + d] * k[col * dim + d];
                }
                *score = dot * scale;
                max_score = max_score.max(*score);
            }
            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for d in 0..dim {
                let mut acc = 0.0f32;
                for (col, score) in scores.iter().enumerate() {
                    acc += (score / sum) * v[col * dim + d];
                }
                expected[row * dim + d] = acc;
            }
        }

        for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "flash_attention[{index}]: GPU {got} vs CPU reference {want} \
                 (full GPU {actual:?} vs reference {expected:?})"
            );
        }
    }

    /// The kernel's `float scores[1024]` array bounds the sequence length; anything
    /// longer must error rather than overrun it.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn flash_attention_rejects_sequences_past_the_kernel_limit() {
        let metal_impl = MetalImpl::new().expect("operation failed in test");
        let (batch, seq, dim) = (1usize, 1025usize, 2usize);
        let q = Tensor::zeros(&[batch, 1, dim]).expect("tensor");
        let k = Tensor::zeros(&[batch, seq, dim]).expect("tensor");
        let v = Tensor::zeros(&[batch, seq, dim]).expect("tensor");
        let mut out = Tensor::zeros(&[batch, 1, dim]).expect("tensor");
        let error = metal_impl
            .flash_attention(&q, &k, &v, &mut out)
            .expect_err("seq_k = 1025 exceeds the kernel's scores[1024] array");
        assert!(format!("{error}").contains("1024"));
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_add_tensors() {
        let metal_impl = MetalImpl::new().expect("operation failed in test");

        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).expect("tensor operation failed");
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[4]).expect("tensor operation failed");

        let result = metal_impl.add_tensors(&a, &b);
        assert!(result.is_ok());

        let result_tensor = result.expect("tensor operation failed");
        assert_eq!(result_tensor.shape(), &[4]);

        let expected = [6.0, 8.0, 10.0, 12.0];
        let actual = result_tensor.data_f32().expect("tensor operation failed");
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6);
        }
    }
}
