//! CudaBackend struct and core implementation

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use crate::errors::Result;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use crate::errors::TrustformersError;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::nvrtc::compile_ptx;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use std::collections::HashMap;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use std::sync::Arc;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use super::cuda_types::*;

/// CUDA GPU backend for matrix multiplication and element-wise operations
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub struct CudaBackend {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) buffer_cache: Arc<std::sync::Mutex<BufferCache>>,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaBackend {
    /// Create a new CUDA backend
    pub fn new(device_id: usize) -> Result<Self> {
        let context = CudaContext::new(device_id).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create CUDA context: {}", e),
                "CudaBackend::new",
            )
        })?;

        let stream = context.default_stream();

        println!("✓ CUDA backend initialized on device {}", device_id);

        Ok(Self {
            context,
            stream,
            buffer_cache: Arc::new(std::sync::Mutex::new(BufferCache::new())),
        })
    }

    /// Create a persistent GPU buffer and return its ID
    pub fn create_persistent_buffer(&self, data: &[f32]) -> Result<BufferId> {
        let buffer = Arc::new(self.stream.clone_htod(data).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy data to device: {}", e),
                "create_persistent_buffer",
            )
        })?);

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
    pub fn get_persistent_buffer(&self, id: &BufferId) -> Result<Arc<CudaSlice<f32>>> {
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

    /// Download data from GPU buffer to CPU
    pub fn download_buffer(&self, buffer_id: &BufferId) -> Result<Vec<f32>> {
        let buffer = self.get_persistent_buffer(buffer_id)?;
        let data_vec = self.stream.clone_dtoh(&*buffer).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy data from device: {}", e),
                "download_buffer",
            )
        })?;
        Ok(data_vec)
    }

    /// Perform matrix multiplication on CUDA GPU
    /// C = A @ B where A is [m, k], B is [k, n], C is [m, n]
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // CUDA kernel for optimized matrix multiplication
        const KERNEL_SRC: &str = r#"
extern "C" __global__ void matmul_kernel(
    const float* a,
    const float* b,
    float* c,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (unsigned int i = 0; i < K; ++i) {
        sum += a[row * K + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}
"#;

        // Compile PTX
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compile CUDA kernel: {}", e),
                "matmul_f32",
            )
        })?;

        // Load kernel module
        let module = self.context.load_module(ptx).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load CUDA module: {}", e),
                "matmul_f32",
            )
        })?;

        let kernel = module.load_function("matmul_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load kernel function: {}", e),
                "matmul_f32",
            )
        })?;

        // Allocate device memory and copy data
        let a_dev = self.stream.clone_htod(a).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy A to device: {}", e),
                "matmul_f32",
            )
        })?;

        let b_dev = self.stream.clone_htod(b).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy B to device: {}", e),
                "matmul_f32",
            )
        })?;

        let result_size = m * n;
        let mut c_dev = self.stream.alloc_zeros::<f32>(result_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate result buffer: {}", e),
                "matmul_f32",
            )
        })?;

        // Launch kernel using builder pattern
        let cfg = LaunchConfig {
            grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&m_u32)
                .arg(&n_u32)
                .arg(&k_u32)
                .launch(cfg)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch kernel: {}", e),
                        "matmul_f32",
                    )
                })?;
        }

        // Copy result back to CPU
        let result = self.stream.clone_dtoh(&c_dev).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result to host: {}", e),
                "matmul_f32",
            )
        })?;

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
        // CUDA kernel for matrix multiplication
        const KERNEL_SRC: &str = r#"
extern "C" __global__ void matmul_kernel(
    const float* a,
    const float* b,
    float* c,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (unsigned int i = 0; i < K; ++i) {
        sum += a[row * K + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}
"#;

        // Compile and load kernel
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compile CUDA kernel: {}", e),
                "matmul_with_cached_weight",
            )
        })?;

        let module = self.context.load_module(ptx).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load CUDA module: {}", e),
                "matmul_with_cached_weight",
            )
        })?;

        let kernel = module.load_function("matmul_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load kernel function: {}", e),
                "matmul_with_cached_weight",
            )
        })?;

        // Get cached weight buffer
        let b_dev = self.get_persistent_buffer(weight_buffer_id)?;

        // Create buffer for input (activations change on each forward pass)
        let a_dev = self.stream.clone_htod(a).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy A to device: {}", e),
                "matmul_with_cached_weight",
            )
        })?;

        let result_size = m * n;
        let mut c_dev = self.stream.alloc_zeros::<f32>(result_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate result buffer: {}", e),
                "matmul_with_cached_weight",
            )
        })?;

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&a_dev)
                .arg(&*b_dev)
                .arg(&mut c_dev)
                .arg(&m_u32)
                .arg(&n_u32)
                .arg(&k_u32)
                .launch(cfg)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch kernel: {}", e),
                        "matmul_with_cached_weight",
                    )
                })?;
        }

        // Copy result back to CPU
        let result = self.stream.clone_dtoh(&c_dev).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result to host: {}", e),
                "matmul_with_cached_weight",
            )
        })?;

        Ok(result)
    }

    /// Execute GELU activation on GPU
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    /// With NaN guarding for numerical stability
    pub fn gelu_f32(&self, input: &[f32]) -> Result<Vec<f32>> {
        const KERNEL_SRC: &str = r#"
extern "C" __global__ void gelu_kernel(
    const float* input,
    float* output,
    unsigned int size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];

    // Clamp extreme values to prevent NaN
    if (x > 10.0f) {
        output[idx] = x;  // GELU(x) ≈ x for large positive x
        return;
    } else if (x < -10.0f) {
        output[idx] = 0.0f;  // GELU(x) ≈ 0 for large negative x
        return;
    }

    float x_cubed = x * x * x;
    // sqrt(2/π) ≈ 0.7978845608
    float inner = 0.7978845608f * (x + 0.044715f * x_cubed);

    // Clamp inner to prevent tanh overflow
    inner = fminf(fmaxf(inner, -20.0f), 20.0f);

    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
}
"#;

        let size = input.len();

        // Compile and load kernel
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compile GELU kernel: {}", e),
                "gelu_f32",
            )
        })?;

        let module = self.context.load_module(ptx).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load GELU module: {}", e),
                "gelu_f32",
            )
        })?;

        let kernel = module.load_function("gelu_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load GELU kernel function: {}", e),
                "gelu_f32",
            )
        })?;

        // Allocate device memory
        let input_dev = self.stream.clone_htod(input).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy input to device: {}", e),
                "gelu_f32",
            )
        })?;

        let mut output_dev = self.stream.alloc_zeros::<f32>(size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer: {}", e),
                "gelu_f32",
            )
        })?;

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: ((size as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let size_u32 = size as u32;

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&input_dev)
                .arg(&mut output_dev)
                .arg(&size_u32)
                .launch(cfg)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch GELU kernel: {}", e),
                        "gelu_f32",
                    )
                })?;
        }

        // Copy result back to CPU
        let result = self.stream.clone_dtoh(&output_dev).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result to host: {}", e),
                "gelu_f32",
            )
        })?;

        Ok(result)
    }

    /// Execute GELU on GPU buffer → GPU buffer (ZERO CPU TRANSFERS!)
    /// Input and output stay on GPU
    pub fn gelu_gpu_to_gpu(&self, input_buffer_id: &BufferId, size: usize) -> Result<BufferId> {
        const KERNEL_SRC: &str = r#"
extern "C" __global__ void gelu_kernel(
    const float* input,
    float* output,
    unsigned int size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];

    // Clamp extreme values to prevent NaN
    if (x > 10.0f) {
        output[idx] = x;
        return;
    } else if (x < -10.0f) {
        output[idx] = 0.0f;
        return;
    }

    float x_cubed = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
    inner = fminf(fmaxf(inner, -20.0f), 20.0f);

    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
}
"#;

        // Compile and load kernel
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compile GELU kernel: {}", e),
                "gelu_gpu_to_gpu",
            )
        })?;

        let module = self.context.load_module(ptx).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load GELU module: {}", e),
                "gelu_gpu_to_gpu",
            )
        })?;

        let kernel = module.load_function("gelu_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load GELU kernel function: {}", e),
                "gelu_gpu_to_gpu",
            )
        })?;

        // Get input buffer
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;

        // Create output buffer
        let mut output_dev = self.stream.alloc_zeros::<f32>(size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer: {}", e),
                "gelu_gpu_to_gpu",
            )
        })?;

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: ((size as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let size_u32 = size as u32;

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&*input_buffer)
                .arg(&mut output_dev)
                .arg(&size_u32)
                .launch(cfg)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch GELU kernel: {}", e),
                        "gelu_gpu_to_gpu",
                    )
                })?;
        }

        // Store output buffer and return ID
        let output_buffer_arc = Arc::new(output_dev);
        let output_id = BufferId::new();

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "gelu_gpu_to_gpu")
        })?;
        cache.insert(output_id, output_buffer_arc);

        Ok(output_id)
    }

    /// Add bias to matrix GPU-to-GPU (ZERO CPU transfers!)
    /// Input: [m, n], Bias: [n] → Output: [m, n]
    pub fn add_bias_gpu_to_gpu(
        &self,
        input_buffer_id: &BufferId,
        bias_buffer_id: &BufferId,
        m: usize,
        n: usize,
    ) -> Result<BufferId> {
        const KERNEL_SRC: &str = r#"
extern "C" __global__ void add_bias_kernel(
    const float* input,
    const float* bias,
    float* output,
    unsigned int m,
    unsigned int n
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) return;

    unsigned int idx = row * n + col;
    output[idx] = input[idx] + bias[col];
}
"#;

        // Compile and load kernel
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compile add_bias kernel: {}", e),
                "add_bias_gpu_to_gpu",
            )
        })?;

        let module = self.context.load_module(ptx).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load add_bias module: {}", e),
                "add_bias_gpu_to_gpu",
            )
        })?;

        let kernel = module.load_function("add_bias_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load add_bias kernel function: {}", e),
                "add_bias_gpu_to_gpu",
            )
        })?;

        // Get input buffers
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let bias_buffer = self.get_persistent_buffer(bias_buffer_id)?;

        // Create output buffer
        let total_size = m * n;
        let mut output_dev = self.stream.alloc_zeros::<f32>(total_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer: {}", e),
                "add_bias_gpu_to_gpu",
            )
        })?;

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let m_u32 = m as u32;
        let n_u32 = n as u32;

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&*input_buffer)
                .arg(&*bias_buffer)
                .arg(&mut output_dev)
                .arg(&m_u32)
                .arg(&n_u32)
                .launch(cfg)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch add_bias kernel: {}", e),
                        "add_bias_gpu_to_gpu",
                    )
                })?;
        }

        // Store output buffer and return ID
        let output_buffer_arc = Arc::new(output_dev);
        let output_id = BufferId::new();

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "add_bias_gpu_to_gpu")
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
        const KERNEL_SRC: &str = r#"
extern "C" __global__ void layernorm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    unsigned int seq_len,
    unsigned int hidden_size,
    float eps
) {
    unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= seq_len) return;

    // Each thread processes one sequence position
    unsigned int offset = pos * hidden_size;

    // Compute mean
    float sum = 0.0f;
    for (unsigned int i = 0; i < hidden_size; ++i) {
        sum += input[offset + i];
    }
    float mean = sum / (float)hidden_size;

    // Compute variance
    float var_sum = 0.0f;
    for (unsigned int i = 0; i < hidden_size; ++i) {
        float diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / (float)hidden_size;
    float std_dev = sqrtf(variance + eps);

    // Normalize and apply affine transform
    for (unsigned int i = 0; i < hidden_size; ++i) {
        float normalized = (input[offset + i] - mean) / std_dev;
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

        // Compile and load kernel
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compile LayerNorm kernel: {}", e),
                "layernorm_f32",
            )
        })?;

        let module = self.context.load_module(ptx).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load LayerNorm module: {}", e),
                "layernorm_f32",
            )
        })?;

        let kernel = module.load_function("layernorm_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load LayerNorm kernel function: {}", e),
                "layernorm_f32",
            )
        })?;

        // Allocate device memory
        let input_dev = self.stream.clone_htod(input).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy input to device: {}", e),
                "layernorm_f32",
            )
        })?;

        let weight_dev = self.stream.clone_htod(weight).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy weight to device: {}", e),
                "layernorm_f32",
            )
        })?;

        let bias_dev = self.stream.clone_htod(bias).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy bias to device: {}", e),
                "layernorm_f32",
            )
        })?;

        let mut output_dev = self.stream.alloc_zeros::<f32>(total_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer: {}", e),
                "layernorm_f32",
            )
        })?;

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: ((seq_len as u32 + 63) / 64, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };

        let seq_len_u32 = seq_len as u32;
        let hidden_size_u32 = hidden_size as u32;

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&input_dev)
                .arg(&weight_dev)
                .arg(&bias_dev)
                .arg(&mut output_dev)
                .arg(&seq_len_u32)
                .arg(&hidden_size_u32)
                .arg(&eps)
                .launch(cfg)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch LayerNorm kernel: {}", e),
                        "layernorm_f32",
                    )
                })?;
        }

        // Copy result back to CPU
        let result = self.stream.clone_dtoh(&output_dev).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result to host: {}", e),
                "layernorm_f32",
            )
        })?;

        Ok(result)
    }

    /// Execute LayerNorm GPU-to-GPU (ZERO CPU transfers!)
    /// Input, weight, bias, and output all stay on GPU
    pub fn layernorm_gpu_to_gpu(
        &self,
        input_buffer_id: &BufferId,
        weight_buffer_id: &BufferId,
        bias_buffer_id: &BufferId,
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<BufferId> {
        const KERNEL_SRC: &str = r#"
extern "C" __global__ void layernorm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    unsigned int seq_len,
    unsigned int hidden_size,
    float eps
) {
    unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= seq_len) return;

    unsigned int offset = pos * hidden_size;

    // Compute mean
    float sum = 0.0f;
    for (unsigned int i = 0; i < hidden_size; ++i) {
        sum += input[offset + i];
    }
    float mean = sum / (float)hidden_size;

    // Compute variance
    float var_sum = 0.0f;
    for (unsigned int i = 0; i < hidden_size; ++i) {
        float diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / (float)hidden_size;
    float std_dev = sqrtf(variance + eps);

    // Normalize and apply affine transform
    for (unsigned int i = 0; i < hidden_size; ++i) {
        float normalized = (input[offset + i] - mean) / std_dev;
        output[offset + i] = normalized * weight[i] + bias[i];
    }
}
"#;

        let total_size = seq_len * hidden_size;

        // Compile and load kernel
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compile LayerNorm kernel: {}", e),
                "layernorm_gpu_to_gpu",
            )
        })?;

        let module = self.context.load_module(ptx).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load LayerNorm module: {}", e),
                "layernorm_gpu_to_gpu",
            )
        })?;

        let kernel = module.load_function("layernorm_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load LayerNorm kernel function: {}", e),
                "layernorm_gpu_to_gpu",
            )
        })?;

        // Get input buffers
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let weight_buffer = self.get_persistent_buffer(weight_buffer_id)?;
        let bias_buffer = self.get_persistent_buffer(bias_buffer_id)?;

        // Create output buffer
        let mut output_dev = self.stream.alloc_zeros::<f32>(total_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer: {}", e),
                "layernorm_gpu_to_gpu",
            )
        })?;

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: ((seq_len as u32 + 63) / 64, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };

        let seq_len_u32 = seq_len as u32;
        let hidden_size_u32 = hidden_size as u32;

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&*input_buffer)
                .arg(&*weight_buffer)
                .arg(&*bias_buffer)
                .arg(&mut output_dev)
                .arg(&seq_len_u32)
                .arg(&hidden_size_u32)
                .arg(&eps)
                .launch(cfg)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch LayerNorm kernel: {}", e),
                        "layernorm_gpu_to_gpu",
                    )
                })?;
        }

        // Store output buffer and return ID
        let output_buffer_arc = Arc::new(output_dev);
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
        const KERNEL_SRC: &str = r#"
extern "C" __global__ void matmul_kernel(
    const float* a,
    const float* b,
    float* c,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (unsigned int i = 0; i < K; ++i) {
        sum += a[row * K + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}
"#;

        // Compile and load kernel
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compile matmul kernel: {}", e),
                "matmul_gpu_to_gpu",
            )
        })?;

        let module = self.context.load_module(ptx).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load matmul module: {}", e),
                "matmul_gpu_to_gpu",
            )
        })?;

        let kernel = module.load_function("matmul_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load matmul kernel function: {}", e),
                "matmul_gpu_to_gpu",
            )
        })?;

        // Get input buffers
        let input_buffer = self.get_persistent_buffer(input_buffer_id)?;
        let weight_buffer = self.get_persistent_buffer(weight_buffer_id)?;

        // Create output buffer
        let result_size = m * n;
        let mut output_dev = self.stream.alloc_zeros::<f32>(result_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer: {}", e),
                "matmul_gpu_to_gpu",
            )
        })?;

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&*input_buffer)
                .arg(&*weight_buffer)
                .arg(&mut output_dev)
                .arg(&m_u32)
                .arg(&n_u32)
                .arg(&k_u32)
                .launch(cfg)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch matmul kernel: {}", e),
                        "matmul_gpu_to_gpu",
                    )
                })?;
        }

        // Store output buffer and return ID
        let output_buffer_arc = Arc::new(output_dev);
        let output_id = BufferId::new();

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "matmul_gpu_to_gpu")
        })?;
        cache.insert(output_id, output_buffer_arc);

        Ok(output_id)
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
        const KERNEL_SRC: &str = r#"
extern "C" __global__ void rope_kernel(
    const float* input,
    float* output,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int rotary_ndims,
    float base
) {
    unsigned int pos = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int h = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= seq_len || h >= num_heads || i >= (rotary_ndims / 2)) return;

    unsigned int j = i + (rotary_ndims / 2);

    // Calculate rotation angle
    float freq = 1.0f / powf(base, 2.0f * (float)i / (float)rotary_ndims);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Get input indices
    unsigned int idx_i = pos * (num_heads * head_dim) + h * head_dim + i;
    unsigned int idx_j = pos * (num_heads * head_dim) + h * head_dim + j;

    // Apply rotation: (x_i, x_j) → (x_i*cos - x_j*sin, x_i*sin + x_j*cos)
    float x_i = input[idx_i];
    float x_j = input[idx_j];

    output[idx_i] = x_i * cos_val - x_j * sin_val;
    output[idx_j] = x_i * sin_val + x_j * cos_val;

    // Copy non-rotated dimensions (first thread in i-dimension handles this)
    if (i == 0) {
        for (unsigned int k = rotary_ndims; k < head_dim; ++k) {
            unsigned int idx_k = pos * (num_heads * head_dim) + h * head_dim + k;
            output[idx_k] = input[idx_k];
        }
    }
}
"#;

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

        // Compile and load kernel
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compile RoPE kernel: {}", e),
                "rope_f32",
            )
        })?;

        let module = self.context.load_module(ptx).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load RoPE module: {}", e),
                "rope_f32",
            )
        })?;

        let kernel = module.load_function("rope_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load RoPE kernel function: {}", e),
                "rope_f32",
            )
        })?;

        // Allocate device memory
        let input_dev = self.stream.clone_htod(input).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy input to device: {}", e),
                "rope_f32",
            )
        })?;

        let mut output_dev = self.stream.alloc_zeros::<f32>(total_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer: {}", e),
                "rope_f32",
            )
        })?;

        // Launch kernel with 3D grid
        let cfg = LaunchConfig {
            grid_dim: (
                ((rotary_ndims / 2) as u32 + 7) / 8,
                (num_heads as u32 + 3) / 4,
                (seq_len as u32 + 3) / 4,
            ),
            block_dim: (8, 4, 4),
            shared_mem_bytes: 0,
        };

        let seq_len_u32 = seq_len as u32;
        let num_heads_u32 = num_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let rotary_ndims_u32 = rotary_ndims as u32;

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&input_dev)
                .arg(&mut output_dev)
                .arg(&seq_len_u32)
                .arg(&num_heads_u32)
                .arg(&head_dim_u32)
                .arg(&rotary_ndims_u32)
                .arg(&base)
                .launch(cfg)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch RoPE kernel: {}", e),
                        "rope_f32",
                    )
                })?;
        }

        // Copy result back to CPU
        let result = self.stream.clone_dtoh(&output_dev).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result to host: {}", e),
                "rope_f32",
            )
        })?;

        Ok(result)
    }

    /// Execute Softmax with causal mask on GPU
    /// Applies causal mask: position i can only attend to j <= i
    /// Input/output shape: [seq_len, seq_len]
    pub fn softmax_causal_f32(&self, input: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        const KERNEL_SRC: &str = r#"
extern "C" __global__ void softmax_causal_kernel(
    const float* input,
    float* output,
    unsigned int seq_len
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;

    unsigned int offset = row * seq_len;

    // Find max for numerical stability (only consider j <= row for causal mask)
    float max_val = -3.402823466e+38f;  // -FLT_MAX
    for (unsigned int j = 0; j <= row; ++j) {
        max_val = fmaxf(max_val, input[offset + j]);
    }

    // Handle edge case: if all values are -inf
    if (max_val < -1e38f) {
        for (unsigned int j = 0; j < seq_len; ++j) {
            output[offset + j] = (j == 0) ? 1.0f : 0.0f;
        }
        return;
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (unsigned int j = 0; j <= row; ++j) {
        sum += expf(input[offset + j] - max_val);
    }

    // Handle degenerate case
    if (sum < 1e-10f) {
        for (unsigned int j = 0; j < seq_len; ++j) {
            output[offset + j] = (j == 0) ? 1.0f : 0.0f;
        }
        return;
    }

    // Normalize and apply causal mask
    for (unsigned int j = 0; j < seq_len; ++j) {
        if (j <= row) {
            output[offset + j] = expf(input[offset + j] - max_val) / sum;
        } else {
            output[offset + j] = 0.0f;  // Causal mask
        }
    }
}
"#;

        let total_size = seq_len * seq_len;

        if input.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input size {} doesn't match seq_len^2 {}",
                input.len(),
                total_size
            )));
        }

        // Compile and load kernel
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compile softmax_causal kernel: {}", e),
                "softmax_causal_f32",
            )
        })?;

        let module = self.context.load_module(ptx).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load softmax_causal module: {}", e),
                "softmax_causal_f32",
            )
        })?;

        let kernel = module.load_function("softmax_causal_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load softmax_causal kernel function: {}", e),
                "softmax_causal_f32",
            )
        })?;

        // Allocate device memory
        let input_dev = self.stream.clone_htod(input).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy input to device: {}", e),
                "softmax_causal_f32",
            )
        })?;

        let mut output_dev = self.stream.alloc_zeros::<f32>(total_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer: {}", e),
                "softmax_causal_f32",
            )
        })?;

        // Launch kernel (one thread per row)
        let cfg = LaunchConfig {
            grid_dim: ((seq_len as u32 + 63) / 64, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };

        let seq_len_u32 = seq_len as u32;

        unsafe {
            self.stream
                .launch_builder(&kernel)
                .arg(&input_dev)
                .arg(&mut output_dev)
                .arg(&seq_len_u32)
                .launch(cfg)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch softmax_causal kernel: {}", e),
                        "softmax_causal_f32",
                    )
                })?;
        }

        // Copy result back to CPU
        let result = self.stream.clone_dtoh(&output_dev).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result to host: {}", e),
                "softmax_causal_f32",
            )
        })?;

        Ok(result)
    }

    /// Copy GPU buffer data back to CPU
    pub fn buffer_to_cpu(&self, buffer_id: &BufferId, _size: usize) -> Result<Vec<f32>> {
        let buffer = self.get_persistent_buffer(buffer_id)?;

        self.stream.clone_dtoh(&*buffer).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy buffer to CPU: {}", e),
                "buffer_to_cpu",
            )
        })
    }

    /// Get device information
    pub fn device_info(&self) -> String {
        format!("CUDA Device (ordinal: {})", self.context.ordinal())
    }
}

/// Global CUDA backend cache (one per device)
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub(crate) static CUDA_BACKENDS: once_cell::sync::Lazy<
    std::sync::Mutex<HashMap<usize, Arc<CudaBackend>>>,
> = once_cell::sync::Lazy::new(|| std::sync::Mutex::new(HashMap::new()));
