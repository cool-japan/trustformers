//! # MetalBackend - initialize_mps_group Methods
//!
//! This module contains method implementations for `MetalBackend`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::common::*;

use super::metalbackend_type::MetalBackend;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(unused_imports)]
use super::types::{BufferCache, BufferId};

/// # Buffer storage mode
///
/// Every GPU-to-GPU op in this module allocates its output `StorageModeShared`, not
/// `StorageModePrivate`. Private allocations have no CPU mapping, and on this
/// platform `MTLBuffer::contents` hands back a **non-null pointer that reads as
/// zeroes** rather than the documented null - so `download_buffer_to_vec` on a
/// Private intermediate silently produced an all-zero tensor instead of failing.
/// Live model code does exactly that (`generation/core.rs`, `gpt2/model_core.rs`),
/// so this was a real source of silently wrong output.
///
/// Shared costs nothing here: Apple Silicon has unified memory, so a Shared buffer is
/// just as GPU-resident as a Private one. `download_buffer_to_vec` additionally
/// rejects any buffer that is still Private or Memoryless, and
/// `download_buffer_via_staging` blits such buffers through a Shared staging
/// allocation.
#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalBackend {
    /// Initialize the Pure-Rust oxicuda-metal compute backend.
    pub(crate) fn initialize_mps(
        _device: &MetalDevice,
        _command_queue: &CommandQueue,
    ) -> Option<oxicuda_metal::MetalBackend> {
        use oxicuda_backend::ComputeBackend;
        let mut backend = oxicuda_metal::MetalBackend::new();
        match backend.init() {
            Ok(()) => {
                tracing::info!("oxicuda-metal compute backend initialized");
                Some(backend)
            },
            Err(error) => {
                // Never swallow this silently: without the oxicuda backend every
                // `matmul_gpu_to_gpu_mps` call fails with "MPS not initialized", and a
                // missing log turns that into an unexplained runtime error.
                tracing::warn!(
                    %error,
                    "oxicuda-metal compute backend failed to initialize; GPU-to-GPU \
                     matmul will be unavailable"
                );
                None
            },
        }
    }
    /// Create a persistent GPU buffer and return its ID.
    ///
    /// The entry is **pinned**: this constructor is how long-lived model weights get
    /// uploaded (see `Linear::ensure_weight_on_gpu`), so it must survive LRU pressure
    /// from the transient intermediates the `*_gpu_to_gpu` ops mint. Release it with
    /// [`remove_persistent_buffer`](Self::remove_persistent_buffer) or
    /// [`clear_buffer_cache`](Self::clear_buffer_cache), or unpin it with
    /// [`set_buffer_pinned`](Self::set_buffer_pinned) to make it reclaimable.
    pub fn create_persistent_buffer(&self, data: &[f32]) -> Result<BufferId> {
        let buffer = Arc::new(self.create_buffer(data)?);
        let buffer_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "create_persistent_buffer",
            )
        })?;
        cache.insert_pinned(buffer_id, buffer);
        Ok(buffer_id)
    }

    /// Create a transient GPU buffer: cached, but LRU-evictable under memory pressure.
    ///
    /// Use this for scratch inputs that do not need to outlive the op that consumes
    /// them; use [`create_persistent_buffer`](Self::create_persistent_buffer) for
    /// weights.
    pub fn create_transient_buffer(&self, data: &[f32]) -> Result<BufferId> {
        let buffer = Arc::new(self.create_buffer(data)?);
        let buffer_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "create_transient_buffer",
            )
        })?;
        cache.insert(buffer_id, buffer);
        Ok(buffer_id)
    }
    /// Create a GPU buffer the cache is allowed to reclaim under memory pressure.
    ///
    /// **Only use this when the caller can regenerate the contents.** Entries in this
    /// tier are LRU-evicted once the cache exceeds its byte cap, after which
    /// `get_persistent_buffer` on the returned id fails with a structured error. Op
    /// results and weights must not go here - see `types.rs` for why evicting data the
    /// caller can still name is a correctness bug, not an optimisation.
    pub fn create_evictable_buffer(&self, data: &[f32]) -> Result<BufferId> {
        let buffer = Arc::new(self.create_buffer(data)?);
        let buffer_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "create_evictable_buffer",
            )
        })?;
        cache.insert_evictable(buffer_id, buffer);
        Ok(buffer_id)
    }
    /// Perform GPU-to-GPU matrix multiplication using MPS (100-500x faster than naive kernel)
    ///
    /// This method operates entirely on GPU without CPU transfers, using Metal Performance Shaders
    /// for highly optimized matrix multiplication.
    ///
    /// # Arguments
    /// * `a_buffer_id` - Left matrix buffer ID (M x K) already on GPU
    /// * `b_buffer_id` - Right matrix buffer ID (K x N) already on GPU
    /// * `m` - Rows in A and result
    /// * `k` - Columns in A, rows in B
    /// * `n` - Columns in B and result
    ///
    /// # Returns
    /// BufferId of result matrix (M x N) on GPU
    pub fn matmul_gpu_to_gpu_mps(
        &self,
        a_buffer_id: &BufferId,
        b_buffer_id: &BufferId,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<BufferId> {
        let oxi = self.mps_ops.as_ref().as_ref().ok_or_else(|| {
            TrustformersError::hardware_error(
                "MPS not initialized - GPU-to-GPU matmul unavailable",
                "matmul_gpu_to_gpu_mps",
            )
        })?;
        let a_buffer = self.get_persistent_buffer(a_buffer_id)?;
        let b_buffer = self.get_persistent_buffer(b_buffer_id)?;
        // oxicuda-metal owns a SEPARATE MTLCommandQueue (oxicuda-metal/src/device.rs),
        // and Metal orders nothing across queues: drain our own queue before handing
        // these buffers over, or the GEMM can read half-written inputs.
        self.flush()?;
        // Resident output buffer (Shared so callers can read it back directly).
        let c_buffer = Arc::new(self.device.new_buffer(
            (m * n * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        // Zero-copy: GEMM runs directly into the resident buffers, no host round-trip.
        oxi_resident_gemm(
            oxi,
            &a_buffer,
            &b_buffer,
            &c_buffer,
            m,
            k,
            n,
            1.0_f64,
            "matmul_gpu_to_gpu_mps",
        )?;
        let result_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "matmul_gpu_to_gpu_mps",
            )
        })?;
        cache.insert(result_id, c_buffer);
        Ok(result_id)
    }
    /// GPU-to-GPU scaled matmul using MPS (FUSED scale+matmul for 1.5-2x additional speedup!)
    /// Critical: This fuses the scaling operation into matmul, eliminating a separate kernel dispatch.
    ///
    /// Computes: C = alpha * (A @ B), where A: [M x K], B: [K x N] → C: [M x N]
    ///
    /// # Arguments
    /// * `a_buffer_id` - Left matrix buffer ID (M x K)
    /// * `b_buffer_id` - Right matrix buffer ID (K x N)
    /// * `m` - Number of rows in A and C
    /// * `k` - Number of columns in A and rows in B
    /// * `n` - Number of columns in B and C
    /// * `alpha` - Scaling factor (e.g., 1/sqrt(head_dim) for attention scores)
    ///
    /// # Returns
    /// BufferId of result matrix (M x N) on GPU
    pub fn matmul_gpu_to_gpu_mps_scaled(
        &self,
        a_buffer_id: &BufferId,
        b_buffer_id: &BufferId,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
    ) -> Result<BufferId> {
        let oxi = self.mps_ops.as_ref().as_ref().ok_or_else(|| {
            TrustformersError::hardware_error(
                "MPS not initialized - GPU-to-GPU scaled matmul unavailable",
                "matmul_gpu_to_gpu_mps_scaled",
            )
        })?;
        let a_buffer = self.get_persistent_buffer(a_buffer_id)?;
        let b_buffer = self.get_persistent_buffer(b_buffer_id)?;
        // Cross-queue hand-off to oxicuda-metal: drain our queue first (see
        // `matmul_gpu_to_gpu_mps` for the full rationale).
        self.flush()?;
        // Resident output buffer (Shared so oxicuda's resident GEMM can import it via
        // `register_external`, which requires a CPU-accessible buffer; on Apple Silicon
        // unified memory Shared is still GPU-resident, so there is no readback penalty).
        let c_out = Arc::new(self.device.new_buffer(
            (m * n * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        // Zero-copy: scaled GEMM (alpha) runs directly into the resident buffers.
        oxi_resident_gemm(
            oxi,
            &a_buffer,
            &b_buffer,
            &c_out,
            m,
            k,
            n,
            alpha as f64,
            "matmul_gpu_to_gpu_mps_scaled",
        )?;
        let result_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "matmul_gpu_to_gpu_mps_scaled",
            )
        })?;
        cache.insert(result_id, c_out);
        Ok(result_id)
    }
    /// Execute GELU on GPU buffer → GPU buffer (ZERO CPU TRANSFERS!)
    /// Input and output stay on GPU
    pub fn gelu_gpu_to_gpu(&self, input_buffer_id: &BufferId, size: usize) -> Result<BufferId> {
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let output_buffer = self.device.new_buffer(
            (size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gelu_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let size_u32 = size as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &size_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (size as u64).div_ceil(256),
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "gelu_gpu_to_gpu")
        })?;
        cache.insert(output_id, output_buffer_arc);
        Ok(output_id)
    }
    /// Execute element-wise addition GPU-to-GPU (ZERO CPU TRANSFERS!)
    /// Critical for residual connections in transformers
    /// Input buffers and output stay on GPU
    pub fn add_gpu_to_gpu(
        &self,
        a_buffer_id: &BufferId,
        b_buffer_id: &BufferId,
        size: usize,
    ) -> Result<BufferId> {
        let a_buffer = self.get_persistent_buffer(a_buffer_id)?;
        let b_buffer = self.get_persistent_buffer(b_buffer_id)?;
        let output_buffer = self.device.new_buffer(
            (size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.elementwise_add_pipeline);
        encoder.set_buffer(0, Some(&*a_buffer), 0);
        encoder.set_buffer(1, Some(&*b_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        let size_u32 = size as u32;
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &size_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (size as u64).div_ceil(256),
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "add_gpu_to_gpu")
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
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let weight_buffer = self.get_persistent_buffer(weight_buffer_id)?;
        let bias_buffer = self.get_persistent_buffer(bias_buffer_id)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.layernorm_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&*weight_buffer), 0);
        encoder.set_buffer(2, Some(&*bias_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
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
        let threadgroup_size = metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (seq_len as u64).div_ceil(64),
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
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
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let weight_buffer = self.get_persistent_buffer(weight_buffer_id)?;
        let result_size = m * n;
        let output_buffer = self.device.new_buffer(
            (result_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.matmul_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&*weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
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
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (n as u64).div_ceil(16),
            height: (m as u64).div_ceil(16),
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "matmul_gpu_to_gpu")
        })?;
        cache.insert(output_id, output_buffer_arc);
        Ok(output_id)
    }
    /// Add bias to matrix GPU-to-GPU (ZERO CPU transfers!)
    /// Input: \[m, n\], Bias: \[n\] → Output: \[m, n\]
    pub fn add_bias_gpu_to_gpu(
        &self,
        input_buffer_id: &BufferId,
        bias_buffer_id: &BufferId,
        m: usize,
        n: usize,
    ) -> Result<BufferId> {
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let bias_buffer = self.get_persistent_buffer(bias_buffer_id)?;
        let total_size = m * n;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.add_bias_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&*bias_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        let m_u32 = m as u32;
        let n_u32 = n as u32;
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
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (n as u64).div_ceil(16),
            height: (m as u64).div_ceil(16),
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "add_bias_gpu_to_gpu")
        })?;
        cache.insert(output_id, output_buffer_arc);
        Ok(output_id)
    }
    /// Stack multiple GPU buffers along a new batch dimension
    /// Input: Vec of buffer IDs, each with shape [seq_len, hidden]
    /// Output: Single buffer with shape [batch_size, seq_len, hidden]
    pub fn stack_gpu_buffers(
        &self,
        input_buffer_ids: &[BufferId],
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<BufferId> {
        let batch_size = input_buffer_ids.len();
        let elements_per_tensor = seq_len * hidden_size;
        let total_elements = batch_size * elements_per_tensor;
        let output_buffer = self.device.new_buffer(
            (total_elements * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.copy_with_offset_pipeline);
        for (batch_idx, buffer_id) in input_buffer_ids.iter().enumerate() {
            let input_buffer = self.get_persistent_buffer(buffer_id)?;
            let output_offset = (batch_idx * elements_per_tensor) as u32;
            let num_elements = elements_per_tensor as u32;
            encoder.set_buffer(0, Some(&*input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_bytes(
                2,
                mem::size_of::<u32>() as u64,
                &output_offset as *const u32 as *const _,
            );
            encoder.set_bytes(
                3,
                mem::size_of::<u32>() as u64,
                &num_elements as *const u32 as *const _,
            );
            let threadgroup_size = metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            let threadgroups = metal::MTLSize {
                width: (elements_per_tensor as u64).div_ceil(256),
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        }
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "stack_gpu_buffers")
        })?;
        cache.insert(output_id, output_buffer_arc.clone());
        // let ptr = output_buffer_arc.contents() as *const f32;
        // let output_slice = unsafe { std::slice::from_raw_parts(ptr, total_elements) };
        Ok(output_id)
    }
    /// Split QKV tensor on GPU (eliminates CPU transfer for attention)
    /// Input: qkv buffer [batch, seq_len, 3*hidden_size]
    /// Outputs: 3 separate buffers Q, K, V each [batch, seq_len, hidden_size]
    pub fn split_qkv_gpu(
        &self,
        qkv_buffer_id: &BufferId,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<(BufferId, BufferId, BufferId)> {
        let qkv_buffer = self.get_persistent_buffer(qkv_buffer_id)?;
        let elements_per_output = batch_size * seq_len * hidden_size;
        let bytes_per_output = (elements_per_output * mem::size_of::<f32>()) as u64;
        let q_buffer =
            self.device.new_buffer(bytes_per_output, MTLResourceOptions::StorageModeShared);
        let k_buffer =
            self.device.new_buffer(bytes_per_output, MTLResourceOptions::StorageModeShared);
        let v_buffer =
            self.device.new_buffer(bytes_per_output, MTLResourceOptions::StorageModeShared);
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.split_qkv_pipeline);
        encoder.set_buffer(0, Some(&*qkv_buffer), 0);
        encoder.set_buffer(1, Some(&q_buffer), 0);
        encoder.set_buffer(2, Some(&k_buffer), 0);
        encoder.set_buffer(3, Some(&v_buffer), 0);
        let batch_u32 = batch_size as u32;
        let seq_u32 = seq_len as u32;
        let hidden_u32 = hidden_size as u32;
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &batch_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &seq_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<u32>() as u64,
            &hidden_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 8,
            height: 8,
            depth: 8,
        };
        let threadgroups = metal::MTLSize {
            width: (batch_size as u64).div_ceil(8),
            height: (seq_len as u64).div_ceil(8),
            depth: (hidden_size as u64).div_ceil(8),
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let q_id = BufferId::new();
        let k_id = BufferId::new();
        let v_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "split_qkv_gpu")
        })?;
        cache.insert(q_id, Arc::new(q_buffer));
        cache.insert(k_id, Arc::new(k_buffer));
        cache.insert(v_id, Arc::new(v_buffer));
        Ok((q_id, k_id, v_id))
    }
    /// Execute softmax with causal mask on GPU-to-GPU (ZERO CPU transfers!)
    /// Input: [seq_len, seq_len] attention scores buffer
    /// Output: [seq_len, seq_len] attention weights buffer
    /// Applies causal mask: position i can only attend to j <= i
    pub fn softmax_causal_gpu_to_gpu(
        &self,
        input_buffer_id: &BufferId,
        seq_len: usize,
    ) -> Result<BufferId> {
        let total_size = seq_len * seq_len;
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.softmax_causal_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let seq_len_u32 = seq_len as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &seq_len_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (seq_len as u64).div_ceil(64),
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        // {
        //     let ptr = output_buffer.contents() as *const f32;
        //     let output_slice = unsafe { std::slice::from_raw_parts(ptr, total_size) };
        //     if seq_len <= 15 {
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "softmax_causal_gpu_to_gpu",
            )
        })?;
        cache.insert(output_id, output_buffer_arc);
        Ok(output_id)
    }
    /// Scale buffer elements by a scalar: output\[i\] = input\[i\] * scale
    /// Used for attention score scaling: scores *= 1/sqrt(head_dim)
    pub fn scale_buffer_gpu_to_gpu(
        &self,
        input_buffer_id: &BufferId,
        scale: f32,
        size: usize,
    ) -> Result<BufferId> {
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let output_buffer = self.device.new_buffer(
            (size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.scale_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            mem::size_of::<f32>() as u64,
            &scale as *const f32 as *const _,
        );
        let size_u32 = size as u32;
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &size_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (size as u64).div_ceil(256),
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "scale_buffer_gpu_to_gpu",
            )
        })?;
        cache.insert(output_id, output_buffer_arc);
        Ok(output_id)
    }
    /// Reshape for multi-head attention: [seq_len, hidden_size] → [num_heads, seq_len, head_dim]
    /// Used to split Q, K, V into separate heads for multi-head attention
    pub fn reshape_to_heads_gpu(
        &self,
        input_buffer_id: &BufferId,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<BufferId> {
        let hidden_size = num_heads * head_dim;
        let total_size = seq_len * hidden_size;
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.reshape_to_heads_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let seq_len_u32 = seq_len as u32;
        let num_heads_u32 = num_heads as u32;
        let head_dim_u32 = head_dim as u32;
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
        let threadgroup_size = metal::MTLSize {
            width: 8,
            height: 8,
            depth: 8,
        };
        let threadgroups = metal::MTLSize {
            width: (num_heads as u64).div_ceil(8),
            height: (seq_len as u64).div_ceil(8),
            depth: (head_dim as u64).div_ceil(8),
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "reshape_to_heads_gpu")
        })?;
        cache.insert(output_id, output_buffer_arc);
        Ok(output_id)
    }
    /// Reshape from multi-head attention: [num_heads, seq_len, head_dim] → [seq_len, hidden_size]
    /// Used to concatenate head outputs back to flat representation
    pub fn reshape_from_heads_gpu(
        &self,
        input_buffer_id: &BufferId,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<BufferId> {
        let hidden_size = num_heads * head_dim;
        let total_size = seq_len * hidden_size;
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.reshape_from_heads_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let seq_len_u32 = seq_len as u32;
        let num_heads_u32 = num_heads as u32;
        let head_dim_u32 = head_dim as u32;
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
        let threadgroup_size = metal::MTLSize {
            width: 8,
            height: 8,
            depth: 8,
        };
        let threadgroups = metal::MTLSize {
            width: (num_heads as u64).div_ceil(8),
            height: (seq_len as u64).div_ceil(8),
            depth: (head_dim as u64).div_ceil(8),
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "reshape_from_heads_gpu",
            )
        })?;
        cache.insert(output_id, output_buffer_arc);
        Ok(output_id)
    }
    /// Extract a single head from reshaped buffer: [num_heads, seq_len, head_dim] → [seq_len, head_dim]
    /// Input is at [head_idx, :, :], output is [:, :]
    pub fn extract_head_gpu(
        &self,
        heads_buffer_id: &BufferId,
        head_idx: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<BufferId> {
        let head_size = seq_len * head_dim;
        let offset_elements = head_idx * head_size;
        let offset_bytes = offset_elements * mem::size_of::<f32>();
        let src_buffer = self.get_persistent_buffer(heads_buffer_id)?;
        let dst_buffer = self.device.new_buffer(
            (head_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let blit_encoder = command_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(
            &src_buffer,
            offset_bytes as u64,
            &dst_buffer,
            0,
            (head_size * mem::size_of::<f32>()) as u64,
        );
        blit_encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_buffer_arc = Arc::new(dst_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "extract_head_gpu")
        })?;
        cache.insert(output_id, output_buffer_arc);
        Ok(output_id)
    }
    /// Transpose 2D matrix on GPU: output[j, i] = input[i, j]
    /// Input: [rows, cols], Output: [cols, rows]
    /// Critical for attention: K^T in Q @ K^T
    pub fn transpose_gpu_to_gpu(
        &self,
        input_buffer_id: &BufferId,
        rows: usize,
        cols: usize,
    ) -> Result<BufferId> {
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let output_buffer = Arc::new(self.device.new_buffer(
            (rows * cols * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.transpose_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&*output_buffer), 0);
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &rows_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &cols_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (cols as u64).div_ceil(16),
            height: (rows as u64).div_ceil(16),
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "transpose_gpu_to_gpu")
        })?;
        cache.insert(output_id, output_buffer);
        Ok(output_id)
    }
    /// Batched transpose for multi-head attention: Transpose all heads in parallel
    /// Input: [num_heads, rows, cols], Output: [num_heads, cols, rows]
    /// Critical optimization: All heads transposed in single GPU dispatch (8-12x faster than sequential)
    /// Used for K^T in attention: [num_heads, seq_len, head_dim] → [num_heads, head_dim, seq_len]
    pub fn batched_transpose_gpu_to_gpu(
        &self,
        input_buffer_id: &BufferId,
        num_heads: usize,
        rows: usize,
        cols: usize,
    ) -> Result<BufferId> {
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let output_buffer = Arc::new(self.device.new_buffer(
            (num_heads * rows * cols * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.batched_transpose_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&*output_buffer), 0);
        let num_heads_u32 = num_heads as u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &num_heads_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &rows_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &cols_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (cols as u64).div_ceil(16),
            height: (rows as u64).div_ceil(16),
            depth: num_heads as u64,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "batched_transpose_gpu_to_gpu",
            )
        })?;
        cache.insert(output_id, output_buffer);
        Ok(output_id)
    }
    /// Batched softmax with causal mask: Process all heads in parallel
    /// Input: [num_heads, seq_len, seq_len] attention scores
    /// Output: [num_heads, seq_len, seq_len] attention weights with causal masking
    /// Critical optimization: All heads processed in single GPU dispatch (8-12x faster than sequential)
    /// Causal mask ensures position i can only attend to j <= i (autoregressive generation)
    pub fn batched_softmax_causal_gpu_to_gpu(
        &self,
        input_buffer_id: &BufferId,
        num_heads: usize,
        seq_len: usize,
    ) -> Result<BufferId> {
        let total_size = num_heads * seq_len * seq_len;
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let output_buffer = self.device.new_buffer(
            (total_size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.batched_softmax_causal_pipeline);
        encoder.set_buffer(0, Some(&*input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let num_heads_u32 = num_heads as u32;
        let seq_len_u32 = seq_len as u32;
        encoder.set_bytes(
            2,
            mem::size_of::<u32>() as u64,
            &num_heads_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &seq_len_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (seq_len as u64).div_ceil(64),
            height: num_heads as u64,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_buffer_arc = Arc::new(output_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "batched_softmax_causal_gpu_to_gpu",
            )
        })?;
        cache.insert(output_id, output_buffer_arc);
        Ok(output_id)
    }
    /// Batched matmul for multi-head attention: Multiply all heads in parallel
    /// A: [num_heads, M, K], B: [num_heads, K, N] → C: [num_heads, M, N]
    /// Critical optimization: All heads processed in single GPU dispatch (8-12x faster than sequential)
    /// Example: Attention weights @ V for all heads simultaneously
    pub fn batched_matmul_gpu_to_gpu(
        &self,
        a_buffer_id: &BufferId,
        b_buffer_id: &BufferId,
        num_heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<BufferId> {
        let a_buffer = self.get_persistent_buffer(a_buffer_id)?;
        let b_buffer = self.get_persistent_buffer(b_buffer_id)?;
        let output_buffer = Arc::new(self.device.new_buffer(
            (num_heads * m * n * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.batched_matmul_pipeline);
        encoder.set_buffer(0, Some(&*a_buffer), 0);
        encoder.set_buffer(1, Some(&*b_buffer), 0);
        encoder.set_buffer(2, Some(&*output_buffer), 0);
        let num_heads_u32 = num_heads as u32;
        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &num_heads_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &m_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<u32>() as u64,
            &n_u32 as *const u32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (n as u64).div_ceil(16),
            height: (m as u64).div_ceil(16),
            depth: num_heads as u64,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed(); // Wait for GPU to complete
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "batched_matmul_gpu_to_gpu",
            )
        })?;
        cache.insert(output_id, output_buffer);
        Ok(output_id)
    }
    /// Batched scaled matmul for multi-head attention: Multiply all heads in parallel with scaling
    /// A: [num_heads, M, K], B: [num_heads, K, N] → C: [num_heads, M, N]
    /// Computes C = alpha * (A @ B) for all heads simultaneously
    /// Critical optimization: Fuses scaling into matmul for all heads (1.5-2x faster than separate ops)
    /// Example: Q @ K^T / sqrt(d_k) for all heads simultaneously
    pub fn batched_matmul_scaled_gpu_to_gpu(
        &self,
        a_buffer_id: &BufferId,
        b_buffer_id: &BufferId,
        num_heads: usize,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
    ) -> Result<BufferId> {
        let a_buffer = self.get_persistent_buffer(a_buffer_id)?;
        let b_buffer = self.get_persistent_buffer(b_buffer_id)?;
        let output_buffer = Arc::new(self.device.new_buffer(
            (num_heads * m * n * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.batched_matmul_scaled_pipeline);
        encoder.set_buffer(0, Some(&*a_buffer), 0);
        encoder.set_buffer(1, Some(&*b_buffer), 0);
        encoder.set_buffer(2, Some(&*output_buffer), 0);
        let num_heads_u32 = num_heads as u32;
        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &num_heads_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &m_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &k_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<u32>() as u64,
            &n_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            mem::size_of::<u32>() as u64,
            &alpha as *const f32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (n as u64).div_ceil(16),
            height: (m as u64).div_ceil(16),
            depth: num_heads as u64,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "batched_matmul_scaled_gpu_to_gpu",
            )
        })?;
        cache.insert(output_id, output_buffer);
        Ok(output_id)
    }
    /// Fused batched scaled matmul + softmax with causal mask: Process all heads in parallel
    /// Q: [num_heads, seq_len, head_dim], K^T: [num_heads, head_dim, seq_len]
    /// Output: [num_heads, seq_len, seq_len] attention weights
    /// Critical optimization: Fuses Q @ K^T scaling and softmax into single kernel (1.5-2x faster)
    /// Eliminates intermediate scaled_scores buffer and reduces GPU dispatches from 4 → 3
    pub fn batched_scaled_matmul_softmax_causal_gpu_to_gpu(
        &self,
        q_buffer_id: &BufferId,
        k_t_buffer_id: &BufferId,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        alpha: f32,
    ) -> Result<BufferId> {
        let q_buffer = self.get_persistent_buffer(q_buffer_id)?;
        let k_t_buffer = self.get_persistent_buffer(k_t_buffer_id)?;
        // Shape validation. The kernel is unbounded in `seq_len` (online softmax, no
        // per-thread score array), but it still indexes Q/K^T by the declared shape, so
        // a mismatched buffer would read past the allocation. Reject that here with a
        // structured error instead of dispatching a corrupting kernel.
        Self::validate_fused_attention_shapes(
            &q_buffer,
            &k_t_buffer,
            num_heads,
            seq_len,
            seq_len,
            head_dim,
            "batched_scaled_matmul_softmax_causal_gpu_to_gpu",
        )?;
        // StorageModeShared, not Private: the kernel parks raw scores in this buffer and
        // the result is read back by tests and by CPU fallbacks. `MTLBuffer::contents`
        // is null for Private allocations, so a Private output made every readback a
        // null dereference. On Apple Silicon's unified memory Shared is still fully
        // GPU-resident, so there is no bandwidth penalty.
        let output_bytes = num_heads * seq_len * seq_len * mem::size_of::<f32>();
        self.validate_allocation_bytes(
            output_bytes,
            "batched_scaled_matmul_softmax_causal_gpu_to_gpu",
        )?;
        let output_buffer = Arc::new(
            self.device
                .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared),
        );
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.batched_scaled_matmul_softmax_causal_pipeline);
        encoder.set_buffer(0, Some(&*q_buffer), 0);
        encoder.set_buffer(1, Some(&*k_t_buffer), 0);
        encoder.set_buffer(2, Some(&*output_buffer), 0);
        let num_heads_u32 = num_heads as u32;
        let seq_len_u32 = seq_len as u32;
        let head_dim_u32 = head_dim as u32;
        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &num_heads_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &seq_len_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &head_dim_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<u32>() as u64,
            &alpha as *const f32 as *const _,
        );
        let threadgroup_size = metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (seq_len as u64).div_ceil(64),
            height: num_heads as u64,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "batched_scaled_matmul_softmax_causal_gpu_to_gpu",
            )
        })?;
        cache.insert(output_id, output_buffer);
        Ok(output_id)
    }

    /// Fused scaled matmul + softmax for generation (q_seq != kv_seq)
    ///
    /// Optimized for autoregressive generation where q_seq=1, kv_seq=cached_length.
    /// Fuses Q @ K^T, scaling, and softmax into a single GPU kernel dispatch.
    ///
    /// # Performance
    ///
    /// Eliminates separate softmax kernel dispatch and reduces memory bandwidth by ~1.5-2x
    /// compared to separate matmul+scale and softmax operations. Critical for generation performance.
    ///
    /// # Arguments
    ///
    /// * `q_buffer_id` - Query tensor: [num_heads, q_seq_len, head_dim]
    /// * `k_t_buffer_id` - Transposed key tensor: [num_heads, head_dim, kv_seq_len]
    /// * `num_heads` - Number of attention heads
    /// * `q_seq_len` - Query sequence length (typically 1 during generation)
    /// * `kv_seq_len` - Key/Value sequence length (all cached tokens)
    /// * `head_dim` - Dimension per head
    /// * `alpha` - Scaling factor (1/sqrt(head_dim))
    ///
    /// # Returns
    ///
    /// Buffer ID containing attention weights: [num_heads, q_seq_len, kv_seq_len]
    pub fn batched_scaled_matmul_softmax_gen_gpu_to_gpu(
        &self,
        q_buffer_id: &BufferId,
        k_t_buffer_id: &BufferId,
        num_heads: usize,
        q_seq_len: usize,
        kv_seq_len: usize,
        head_dim: usize,
        alpha: f32,
    ) -> Result<BufferId> {
        let q_buffer = self.get_persistent_buffer(q_buffer_id)?;
        let k_t_buffer = self.get_persistent_buffer(k_t_buffer_id)?;
        // Same contract as the causal variant: the kernel is unbounded in `kv_seq_len`,
        // but the declared shape must match the operand buffers.
        Self::validate_fused_attention_shapes(
            &q_buffer,
            &k_t_buffer,
            num_heads,
            q_seq_len,
            kv_seq_len,
            head_dim,
            "batched_scaled_matmul_softmax_gen_gpu_to_gpu",
        )?;

        // Shared, not Private - see the causal variant for why.
        let output_bytes = num_heads * q_seq_len * kv_seq_len * mem::size_of::<f32>();
        self.validate_allocation_bytes(
            output_bytes,
            "batched_scaled_matmul_softmax_gen_gpu_to_gpu",
        )?;
        let output_buffer = Arc::new(
            self.device
                .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared),
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.batched_scaled_matmul_softmax_gen_pipeline);
        encoder.set_buffer(0, Some(&*q_buffer), 0);
        encoder.set_buffer(1, Some(&*k_t_buffer), 0);
        encoder.set_buffer(2, Some(&*output_buffer), 0);

        let num_heads_u32 = num_heads as u32;
        let q_seq_len_u32 = q_seq_len as u32;
        let kv_seq_len_u32 = kv_seq_len as u32;
        let head_dim_u32 = head_dim as u32;

        encoder.set_bytes(
            3,
            mem::size_of::<u32>() as u64,
            &num_heads_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            mem::size_of::<u32>() as u64,
            &q_seq_len_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            mem::size_of::<u32>() as u64,
            &kv_seq_len_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            mem::size_of::<u32>() as u64,
            &head_dim_u32 as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            mem::size_of::<u32>() as u64,
            &alpha as *const f32 as *const _,
        );

        // Dispatch: one thread per (q_row, head) pair
        let threadgroup_size = metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let threadgroups = metal::MTLSize {
            width: (q_seq_len as u64).div_ceil(64),
            height: num_heads as u64,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        self.commit_async(command_buffer);

        let output_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "batched_scaled_matmul_softmax_gen_gpu_to_gpu",
            )
        })?;
        cache.insert(output_id, output_buffer);

        Ok(output_id)
    }

    /// Concatenate cached K/V with new K/V for KV-cache (GPU-aware, ZERO CPU transfers!)
    ///
    /// # Arguments
    ///
    /// * `cached_buffer_id` - Optional cached tensor [batch, num_heads, cached_seq_len, head_dim]
    /// * `new_buffer_id` - New tensor [batch, num_heads, new_seq_len, head_dim]
    /// * `batch_size` - Batch size
    /// * `num_heads` - Number of attention heads
    /// * `cached_seq_len` - Sequence length of cached tensor (0 if no cache)
    /// * `new_seq_len` - Sequence length of new tensor
    /// * `head_dim` - Head dimension
    ///
    /// # Returns
    ///
    /// Buffer ID containing concatenated tensor [batch, num_heads, cached_seq_len+new_seq_len, head_dim]
    pub fn concat_kv_cache(
        &self,
        cached_buffer_id: Option<&BufferId>,
        new_buffer_id: &BufferId,
        batch_size: usize,
        num_heads: usize,
        cached_seq_len: usize,
        new_seq_len: usize,
        head_dim: usize,
    ) -> Result<BufferId> {
        let cached_buffer_id = match cached_buffer_id {
            Some(id) if cached_seq_len > 0 => *id,
            _ => return Ok(*new_buffer_id),
        };
        let total_seq_len = cached_seq_len + new_seq_len;
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "concat_kv_cache")
        })?;
        let cached_buffer = cache.get(&cached_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error("Cached buffer not found", "concat_kv_cache")
        })?;
        let new_buffer = cache.get(new_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error("New buffer not found", "concat_kv_cache")
        })?;
        let output_size = batch_size * num_heads * total_seq_len * head_dim;
        let output_buffer = Arc::new(self.device.new_buffer(
            (output_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.concat_seq_dim_pipeline);
        encoder.set_buffer(0, Some(&**cached_buffer), 0);
        encoder.set_buffer(1, Some(&**new_buffer), 0);
        encoder.set_buffer(2, Some(&*output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &(batch_size as u32) as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &(num_heads as u32) as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &(cached_seq_len as u32) as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &(new_seq_len as u32) as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &(head_dim as u32) as *const u32 as *const _,
        );
        let threads_per_threadgroup =
            metal::MTLSize::new((head_dim as u64).min(256), (num_heads as u64).min(4), 1);
        let threadgroups = metal::MTLSize::new(
            head_dim.div_ceil(threads_per_threadgroup.width as usize) as u64,
            num_heads.div_ceil(threads_per_threadgroup.height as usize) as u64,
            batch_size as u64,
        );
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();
        self.commit_async(command_buffer);
        let output_id = BufferId::new();
        cache.insert(output_id, output_buffer);
        Ok(output_id)
    }
    /// Execute full multi-head attention on GPU with OPTIMIZED SYNCHRONIZATION (Phase 3)
    /// Uses single command buffer for all batched operations to eliminate intermediate waits
    /// Expected 2-3x speedup from reduced CPU-GPU synchronization overhead
    pub fn attention_gpu_to_gpu_optimized(
        &self,
        q_buffer_id: &BufferId,
        k_buffer_id: &BufferId,
        v_buffer_id: &BufferId,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<BufferId> {
        let _hidden_size = num_heads * head_dim;
        if batch_size != 1 {
            return Err(TrustformersError::tensor_op_error(
                "GPU attention currently only supports batch_size=1",
                "attention_gpu_to_gpu_optimized",
            ));
        }
        let q_heads = self.reshape_to_heads_gpu(q_buffer_id, seq_len, num_heads, head_dim)?;
        let k_heads = self.reshape_to_heads_gpu(k_buffer_id, seq_len, num_heads, head_dim)?;
        let v_heads = self.reshape_to_heads_gpu(v_buffer_id, seq_len, num_heads, head_dim)?;
        let command_buffer = self.command_queue.new_command_buffer();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q_heads_buffer = self.get_persistent_buffer(&q_heads)?;
        let k_heads_buffer = self.get_persistent_buffer(&k_heads)?;
        let v_heads_buffer = self.get_persistent_buffer(&v_heads)?;
        let k_heads_t_buffer = Arc::new(self.device.new_buffer(
            (num_heads * seq_len * head_dim * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let attn_weights_buffer = Arc::new(self.device.new_buffer(
            (num_heads * seq_len * seq_len * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let output_heads_buffer = Arc::new(self.device.new_buffer(
            (num_heads * seq_len * head_dim * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.batched_transpose_pipeline);
            encoder.set_buffer(0, Some(&*k_heads_buffer), 0);
            encoder.set_buffer(1, Some(&*k_heads_t_buffer), 0);
            let num_heads_u32 = num_heads as u32;
            let rows_u32 = seq_len as u32;
            let cols_u32 = head_dim as u32;
            encoder.set_bytes(
                2,
                mem::size_of::<u32>() as u64,
                &num_heads_u32 as *const u32 as *const _,
            );
            encoder.set_bytes(
                3,
                mem::size_of::<u32>() as u64,
                &rows_u32 as *const u32 as *const _,
            );
            encoder.set_bytes(
                4,
                mem::size_of::<u32>() as u64,
                &cols_u32 as *const u32 as *const _,
            );
            let threadgroup_size = metal::MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };
            let threadgroups = metal::MTLSize {
                width: (head_dim as u64).div_ceil(16),
                height: (seq_len as u64).div_ceil(16),
                depth: num_heads as u64,
            };
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            encoder.end_encoding();
        }
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.batched_scaled_matmul_softmax_causal_pipeline);
            encoder.set_buffer(0, Some(&*q_heads_buffer), 0);
            encoder.set_buffer(1, Some(&*k_heads_t_buffer), 0);
            encoder.set_buffer(2, Some(&*attn_weights_buffer), 0);
            let num_heads_u32 = num_heads as u32;
            let seq_len_u32 = seq_len as u32;
            let head_dim_u32 = head_dim as u32;
            encoder.set_bytes(
                3,
                mem::size_of::<u32>() as u64,
                &num_heads_u32 as *const u32 as *const _,
            );
            encoder.set_bytes(
                4,
                mem::size_of::<u32>() as u64,
                &seq_len_u32 as *const u32 as *const _,
            );
            encoder.set_bytes(
                5,
                mem::size_of::<u32>() as u64,
                &head_dim_u32 as *const u32 as *const _,
            );
            encoder.set_bytes(
                6,
                mem::size_of::<u32>() as u64,
                &scale as *const f32 as *const _,
            );
            let threadgroup_size = metal::MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            };
            let threadgroups = metal::MTLSize {
                width: (seq_len as u64).div_ceil(64),
                height: num_heads as u64,
                depth: 1,
            };
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            encoder.end_encoding();
        }
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.batched_matmul_pipeline);
            encoder.set_buffer(0, Some(&*attn_weights_buffer), 0);
            encoder.set_buffer(1, Some(&*v_heads_buffer), 0);
            encoder.set_buffer(2, Some(&*output_heads_buffer), 0);
            let num_heads_u32 = num_heads as u32;
            let m_u32 = seq_len as u32;
            let k_u32 = seq_len as u32;
            let n_u32 = head_dim as u32;
            encoder.set_bytes(
                3,
                mem::size_of::<u32>() as u64,
                &num_heads_u32 as *const u32 as *const _,
            );
            encoder.set_bytes(
                4,
                mem::size_of::<u32>() as u64,
                &m_u32 as *const u32 as *const _,
            );
            encoder.set_bytes(
                5,
                mem::size_of::<u32>() as u64,
                &k_u32 as *const u32 as *const _,
            );
            encoder.set_bytes(
                6,
                mem::size_of::<u32>() as u64,
                &n_u32 as *const u32 as *const _,
            );
            let threadgroup_size = metal::MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };
            let threadgroups = metal::MTLSize {
                width: (head_dim as u64).div_ceil(16),
                height: (seq_len as u64).div_ceil(16),
                depth: num_heads as u64,
            };
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            encoder.end_encoding();
        }
        self.commit_async(command_buffer);
        let output_heads_id = BufferId::new();
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "attention_gpu_to_gpu_optimized",
            )
        })?;
        cache.insert(output_heads_id, output_heads_buffer);
        let final_output =
            self.reshape_from_heads_gpu(&output_heads_id, seq_len, num_heads, head_dim)?;
        Ok(final_output)
    }
}

/// Zero-copy resident GEMM through the oxicuda-metal backend.
///
/// Registers the three already-resident Metal buffers (`A`, `B`, `C`) as
/// external imports on the oxicuda backend (each `register_external` takes its
/// own retain and borrows the buffer — it never frees the caller's buffer),
/// then runs `C = alpha * (A @ B)` directly into the resident output buffer and
/// releases the three import handles. No host round-trip: the result stays
/// GPU-resident in `c_buffer`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn oxi_resident_gemm(
    oxi: &oxicuda_metal::MetalBackend,
    a_buffer: &Arc<Buffer>,
    b_buffer: &Arc<Buffer>,
    c_buffer: &Arc<Buffer>,
    m: usize,
    k: usize,
    n: usize,
    alpha: f64,
    op: &'static str,
) -> Result<()> {
    use oxicuda_backend::{BackendTranspose, ComputeBackend};
    let elem = mem::size_of::<f32>();
    let a_h = oxi.register_external(a_buffer, m * k * elem).map_err(|e| {
        TrustformersError::hardware_error(&format!("oxicuda-metal register a: {e}"), op)
    })?;
    let b_h = oxi.register_external(b_buffer, k * n * elem).map_err(|e| {
        TrustformersError::hardware_error(&format!("oxicuda-metal register b: {e}"), op)
    })?;
    let c_h = oxi.register_external(c_buffer, m * n * elem).map_err(|e| {
        TrustformersError::hardware_error(&format!("oxicuda-metal register c: {e}"), op)
    })?;
    let gemm_res = oxi.gemm(
        BackendTranspose::NoTrans,
        BackendTranspose::NoTrans,
        m,
        n,
        k,
        alpha,
        a_h,
        k,
        b_h,
        n,
        0.0_f64,
        c_h,
        n,
    );
    // Always release the three import handles, even if the GEMM failed, so the
    // oxicuda-side retains never leak. Freeing an import only drops oxicuda's
    // own retain; the caller's `Arc<Buffer>` remains valid.
    let free_a = oxi.free(a_h);
    let free_b = oxi.free(b_h);
    let free_c = oxi.free(c_h);
    gemm_res
        .map_err(|e| TrustformersError::hardware_error(&format!("oxicuda-metal gemm: {e}"), op))?;
    free_a.map_err(|e| {
        TrustformersError::hardware_error(&format!("oxicuda-metal free a: {e}"), op)
    })?;
    free_b.map_err(|e| {
        TrustformersError::hardware_error(&format!("oxicuda-metal free b: {e}"), op)
    })?;
    free_c.map_err(|e| {
        TrustformersError::hardware_error(&format!("oxicuda-metal free c: {e}"), op)
    })?;
    Ok(())
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;

    // Read back a resident result buffer by blitting it into a CPU-mappable staging
    // buffer. The scaled matmul output is StorageModeShared, but exercising the blit
    // readback path here keeps coverage of that mechanism.
    fn read_private_result(
        backend: &MetalBackend,
        id: &BufferId,
        n_elems: usize,
    ) -> Result<Vec<f32>> {
        let src = backend.get_persistent_buffer(id)?;
        let bytes = (n_elems * mem::size_of::<f32>()) as u64;
        let staging = backend.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let cb = backend.command_queue.new_command_buffer();
        let blit = cb.new_blit_command_encoder();
        blit.copy_from_buffer(&src, 0, &staging, 0, bytes);
        blit.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let ptr = staging.contents() as *const f32;
        // SAFETY: staging is Shared/CPU-mappable and holds n_elems f32 after the blit.
        let slice = unsafe { std::slice::from_raw_parts(ptr, n_elems) };
        Ok(slice.to_vec())
    }

    // Naive CPU triple-loop reference: C(m×n) = scale * (A(m×k) @ B(k×n)), row-major.
    fn cpu_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, scale: f32) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = scale * acc;
            }
        }
        c
    }

    #[test]
    fn mps_oxicuda_matmul_parity() -> Result<()> {
        let backend = MetalBackend::new()?;

        // Non-trivial shape; deterministic row-major fills.
        let (m, k, n) = (12usize, 9usize, 7usize);
        let mut a = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                a[i * k + j] = ((i * k + j) % 11) as f32 * 0.3 - 1.5;
            }
        }
        let mut b = vec![0.0f32; k * n];
        for p in 0..k {
            for q in 0..n {
                b[p * n + q] = ((p * n + q) % 13) as f32 * 0.2 - 1.0;
            }
        }

        let a_id = backend.create_persistent_buffer(&a)?;
        let b_id = backend.create_persistent_buffer(&b)?;

        // 1) Unscaled path (Shared output → download_buffer_to_vec).
        let c_id = backend.matmul_gpu_to_gpu_mps(&a_id, &b_id, m, k, n)?;
        let got_unscaled = backend.download_buffer_to_vec(&c_id)?;
        let ref_unscaled = cpu_ref(&a, &b, m, k, n, 1.0);
        assert_eq!(got_unscaled.len(), m * n, "unscaled result length");
        for idx in 0..(m * n) {
            assert!(
                (got_unscaled[idx] - ref_unscaled[idx]).abs() < 1e-3,
                "unscaled mismatch at {idx}: got {} want {}",
                got_unscaled[idx],
                ref_unscaled[idx]
            );
        }

        // 2) Scaled path (Shared resident output → blit-readback). Use an attention-like scale.
        let alpha = 1.0f32 / (k as f32).sqrt();
        let c_scaled_id = backend.matmul_gpu_to_gpu_mps_scaled(&a_id, &b_id, m, k, n, alpha)?;
        let got_scaled = read_private_result(&backend, &c_scaled_id, m * n)?;
        let ref_scaled = cpu_ref(&a, &b, m, k, n, alpha);
        assert_eq!(got_scaled.len(), m * n, "scaled result length");
        for idx in 0..(m * n) {
            assert!(
                (got_scaled[idx] - ref_scaled[idx]).abs() < 1e-3,
                "scaled mismatch at {idx}: got {} want {}",
                got_scaled[idx],
                ref_scaled[idx]
            );
        }

        println!(
            "mps_oxicuda_matmul_parity PASS (unscaled + scaled, shape {m}x{k}x{n}, alpha={alpha})"
        );
        Ok(())
    }
}
