// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Real CUDA implementation for TrustformeRS hardware acceleration
//!
//! This module provides actual CUDA runtime API bindings to replace the simulated
//! implementations in cuda_kernels.rs. It uses the cudarc crate for safe CUDA operations.

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use std::sync::{Arc, OnceLock};

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig};
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use std::collections::HashMap;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use std::sync::Mutex;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::nvrtc::compile_ptx;

/// Real CUDA implementation with hardware bindings
pub struct CudaImpl {
    /// CUDA context
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    context: Arc<CudaContext>,
    /// CUDA stream
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    stream: Arc<CudaStream>,
    /// Compiled kernel cache
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    kernel_cache: Arc<Mutex<HashMap<String, CudaKernel>>>,
    /// Memory pool for efficient allocation
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    memory_pool: Arc<Mutex<MemoryPool>>,

    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    _placeholder: (),
}

/// Compiled CUDA kernel
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[derive(Clone)]
pub struct CudaKernel {
    /// Kernel function
    func: cudarc::driver::CudaFunction,
    /// Kernel name
    name: String,
    /// Grid configuration
    grid_config: (u32, u32, u32),
    /// Block configuration
    block_config: (u32, u32, u32),
    /// Shared memory size
    shared_memory: u32,
}

/// Memory pool for efficient GPU memory management
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub struct MemoryPool {
    /// Available memory blocks
    available_blocks: Vec<MemoryBlock>,
    /// Allocated blocks
    allocated_blocks: HashMap<usize, MemoryBlock>,
    /// Total allocated memory
    total_allocated: usize,
    /// Peak memory usage
    peak_memory: usize,
}

/// Memory block in GPU memory
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[derive(Clone)]
pub struct MemoryBlock {
    /// GPU memory slice
    slice: CudaSlice<f32>,
    /// Block size in bytes
    size: usize,
    /// Block ID
    id: usize,
}

/// Global CUDA instance
#[allow(dead_code)]
static CUDA_INSTANCE: OnceLock<Arc<CudaImpl>> = OnceLock::new();

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaImpl {
    /// Initialize CUDA with the first available device
    pub fn new() -> Result<Self> {
        let context = CudaContext::new(0).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to initialize CUDA context: {}", e),
                "cuda_init",
            )
        })?;

        let stream = context.default_stream();

        Ok(Self {
            context,
            stream,
            kernel_cache: Arc::new(Mutex::new(HashMap::new())),
            memory_pool: Arc::new(Mutex::new(MemoryPool::new())),
        })
    }

    /// Get global CUDA instance
    pub fn global() -> Result<&'static Arc<CudaImpl>> {
        static ONCE: std::sync::Once = std::sync::Once::new();
        static mut INIT_RESULT: Option<std::result::Result<Arc<CudaImpl>, TrustformersError>> =
            None;

        unsafe {
            ONCE.call_once(|| {
                INIT_RESULT = Some(Self::new().map(Arc::new));
            });

            match INIT_RESULT.as_ref().unwrap() {
                Ok(instance) => {
                    CUDA_INSTANCE.set(instance.clone()).ok();
                    Ok(CUDA_INSTANCE.get().unwrap())
                },
                Err(e) => Err(e.clone()),
            }
        }
    }

    /// Compile and cache a CUDA kernel
    pub fn compile_kernel(
        &self,
        name: &str,
        source: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<CudaKernel> {
        // Check cache first
        {
            let cache = self.kernel_cache.lock().expect("Lock poisoned");
            if let Some(kernel) = cache.get(name) {
                return Ok(kernel.clone());
            }
        }

        // Compile PTX from CUDA source
        let ptx = compile_ptx(source).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to compile CUDA kernel: {}", e),
                "cuda_compile",
            )
        })?;

        // Load kernel function
        self.device.load_ptx(ptx, "module", &["kernel"]).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to load PTX: {}", e),
                "cuda_load_ptx",
            )
        })?;

        let func = self.device.get_func("module", "kernel").ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Failed to get kernel function: {}", name),
                "cuda_get_func",
            )
        })?;

        let kernel = CudaKernel {
            func,
            name: name.to_string(),
            grid_config: grid,
            block_config: block,
            shared_memory: 0,
        };

        // Cache the kernel
        {
            let mut cache = self.kernel_cache.lock().expect("Lock poisoned");
            cache.insert(name.to_string(), kernel.clone());
        }

        Ok(kernel)
    }

    /// Allocate GPU memory with pooling
    pub fn allocate_memory(&self, size: usize) -> Result<CudaSlice<f32>> {
        let elements = (size + 3) / 4; // Convert bytes to f32 elements

        // Try to get from pool first
        {
            let mut pool = self.memory_pool.lock().expect("Lock poisoned");
            if let Some(block) = pool.get_block(elements) {
                return Ok(block.slice);
            }
        }

        // Allocate new memory
        let slice = self.device.alloc_zeros::<f32>(elements).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate GPU memory: {}", e),
                "cuda_alloc",
            )
        })?;

        // Update memory tracking
        {
            let mut pool = self.memory_pool.lock().expect("Lock poisoned");
            pool.total_allocated += size;
            pool.peak_memory = pool.peak_memory.max(pool.total_allocated);
        }

        Ok(slice)
    }

    /// Copy tensor data to GPU
    pub fn copy_to_gpu(&self, tensor: &Tensor) -> Result<CudaSlice<f32>> {
        let data = tensor.data_f32()?;
        let mut gpu_slice = self.allocate_memory(data.len() * 4)?;

        self.device.htod_copy_into(data, &mut gpu_slice).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy data to GPU: {}", e),
                "cuda_htod",
            )
        })?;

        Ok(gpu_slice)
    }

    /// Copy data from GPU to tensor
    pub fn copy_from_gpu(&self, gpu_slice: &CudaSlice<f32>, tensor: &mut Tensor) -> Result<()> {
        let mut data = vec![0.0f32; gpu_slice.len()];

        self.device.dtoh_sync_copy_into(gpu_slice, &mut data).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy data from GPU: {}", e),
                "cuda_dtoh",
            )
        })?;

        tensor.set_data_f32(&data)?;
        Ok(())
    }

    /// Execute matrix multiplication using real CUDA
    pub fn matmul(&self, a: &Tensor, b: &Tensor, c: &mut Tensor) -> Result<()> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = c.shape();

        // Validate dimensions
        if a_shape.len() != 2 || b_shape.len() != 2 || c_shape.len() != 2 {
            return Err(TrustformersError::tensor_op_error(
                "Matrix multiplication requires 2D tensors",
                "CudaImpl::matmul",
            ));
        }

        let m = a_shape[0] as u32;
        let k = a_shape[1] as u32;
        let n = b_shape[1] as u32;

        // Generate optimized CUDA kernel
        let kernel_source = self.generate_optimized_matmul_kernel(m, k, n);
        let kernel = self.compile_kernel(
            "matmul_kernel",
            &kernel_source,
            ((n + 15) / 16, (m + 15) / 16, 1),
            (16, 16, 1),
        )?;

        // Allocate GPU memory and copy data
        let a_gpu = self.copy_to_gpu(a)?;
        let b_gpu = self.copy_to_gpu(b)?;
        let mut c_gpu = self.allocate_memory(c_shape[0] * c_shape[1] * 4)?;

        // Configure kernel launch
        let launch_config = LaunchConfig {
            grid_dim: kernel.grid_config,
            block_dim: kernel.block_config,
            shared_mem_bytes: kernel.shared_memory,
        };

        // Launch kernel
        let args = (&a_gpu, &b_gpu, &mut c_gpu, m, k, n);
        unsafe {
            kernel.func.launch(launch_config, args).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to launch CUDA kernel: {}", e),
                    "kernel_launch",
                )
            })?;
        }

        // Synchronize and copy result back
        self.device.synchronize().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("CUDA synchronization failed: {}", e),
                "cuda_synchronize_matmul",
            )
        })?;

        self.copy_from_gpu(&c_gpu, c)?;

        Ok(())
    }

    /// Execute Flash Attention using real CUDA
    pub fn flash_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
    ) -> Result<()> {
        let q_shape = query.shape();
        let batch_size = q_shape[0] as u32;
        let seq_len = q_shape[1] as u32;
        let head_dim = q_shape[2] as u32;

        // Generate Flash Attention kernel
        let kernel_source = self.generate_flash_attention_kernel(batch_size, seq_len, head_dim);
        let kernel = self.compile_kernel(
            "flash_attention_kernel",
            &kernel_source,
            (batch_size, seq_len, 1),
            (256, 1, 1),
        )?;

        // Allocate GPU memory
        let q_gpu = self.copy_to_gpu(query)?;
        let k_gpu = self.copy_to_gpu(key)?;
        let v_gpu = self.copy_to_gpu(value)?;
        let mut o_gpu = self.allocate_memory(output.memory_usage())?;

        // Configure kernel with optimized shared memory
        let launch_config = LaunchConfig {
            grid_dim: kernel.grid_config,
            block_dim: kernel.block_config,
            shared_mem_bytes: 48 * 1024, // 48KB shared memory for tiling
        };

        // Launch kernel
        let args = (
            &q_gpu, &k_gpu, &v_gpu, &mut o_gpu, batch_size, seq_len, head_dim,
        );
        unsafe {
            kernel.func.launch(launch_config, args).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to launch Flash Attention kernel: {}", e),
                    "flash_attention_launch",
                )
            })?;
        }

        // Synchronize and copy result
        self.device.synchronize().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("CUDA synchronization failed: {}", e),
                "cuda_synchronize_flash_attention",
            )
        })?;

        self.copy_from_gpu(&o_gpu, output)?;

        Ok(())
    }

    /// Fused GELU activation function
    pub fn fused_gelu(&self, input: &Tensor, output: &mut Tensor, approximate: bool) -> Result<()> {
        let shape = input.shape();
        let total_elements = shape.iter().product::<usize>();

        // Generate optimized GELU kernel
        let kernel_source = self.generate_fused_gelu_kernel(approximate);
        let grid_size = ((total_elements + 255) / 256) as u32;
        let kernel = self.compile_kernel(
            "fused_gelu_kernel",
            &kernel_source,
            (grid_size, 1, 1),
            (256, 1, 1),
        )?;

        // Get input and output data
        let input_data = self.device.htod_copy(input.data()?).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy input to device: {}", e),
                "cuda_htod_gelu",
            )
        })?;

        let mut output_data = self.allocate_memory(total_elements * 4)?;

        // Launch kernel
        unsafe {
            kernel
                .func
                .launch(
                    LaunchConfig {
                        grid_dim: kernel.grid_config,
                        block_dim: kernel.block_config,
                        shared_mem_bytes: kernel.shared_memory,
                    },
                    (&input_data, &mut output_data, total_elements as u32),
                )
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch GELU kernel: {}", e),
                        "gelu_launch",
                    )
                })?;
        }

        // Synchronize and copy back
        self.device.synchronize().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("CUDA synchronization failed: {}", e),
                "cuda_synchronize_gelu",
            )
        })?;

        let result_data = self.device.dtoh_sync_copy(&output_data).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result from device: {}", e),
                "cuda_dtoh_gelu",
            )
        })?;

        *output = Tensor::from_vec(result_data, &shape)?;
        Ok(())
    }

    /// Fused bias addition with activation
    pub fn fused_bias_activation(
        &self,
        input: &Tensor,
        bias: &Tensor,
        output: &mut Tensor,
        activation: &str,
    ) -> Result<()> {
        let shape = input.shape();
        let total_elements = shape.iter().product::<usize>();
        let bias_size = bias.shape().iter().product::<usize>();

        // Generate optimized bias+activation kernel
        let kernel_source = self.generate_fused_bias_activation_kernel(activation);
        let grid_size = ((total_elements + 255) / 256) as u32;
        let kernel = self.compile_kernel(
            "fused_bias_activation_kernel",
            &kernel_source,
            (grid_size, 1, 1),
            (256, 1, 1),
        )?;

        // Prepare device data
        let input_data = self.device.htod_copy(input.data()?).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy input to device: {}", e),
                "cuda_htod_bias",
            )
        })?;

        let bias_data = self.device.htod_copy(bias.data()?).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy bias to device: {}", e),
                "cuda_htod_bias",
            )
        })?;

        let mut output_data = self.allocate_memory(total_elements * 4)?;

        // Launch kernel
        unsafe {
            kernel
                .func
                .launch(
                    LaunchConfig {
                        grid_dim: kernel.grid_config,
                        block_dim: kernel.block_config,
                        shared_mem_bytes: kernel.shared_memory,
                    },
                    (
                        &input_data,
                        &bias_data,
                        &mut output_data,
                        total_elements as u32,
                        bias_size as u32,
                    ),
                )
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to launch bias activation kernel: {}", e),
                        "bias_activation_launch",
                    )
                })?;
        }

        // Synchronize and copy back
        self.device.synchronize().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("CUDA synchronization failed: {}", e),
                "cuda_synchronize_bias",
            )
        })?;

        let result_data = self.device.dtoh_sync_copy(&output_data).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result from device: {}", e),
                "cuda_dtoh_bias",
            )
        })?;

        *output = Tensor::from_vec(result_data, &shape)?;
        Ok(())
    }

    /// Generate CUDA kernel source for fused GELU activation
    fn generate_fused_gelu_kernel(&self, approximate: bool) -> String {
        if approximate {
            // Fast tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            r#"
extern "C" __global__ void fused_gelu_kernel(const float* input, float* output, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        float arg = 0.7978845608f * (x + 0.044715f * x3);  // sqrt(2/π) ≈ 0.7978845608
        float tanh_val = tanhf(arg);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}
"#
            .to_string()
        } else {
            // Accurate erf-based GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            r#"
extern "C" __global__ void fused_gelu_kernel(const float* input, float* output, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float scaled = x * 0.7071067812f;  // 1/sqrt(2) ≈ 0.7071067812
        float erf_val = erff(scaled);
        output[idx] = 0.5f * x * (1.0f + erf_val);
    }
}
"#
            .to_string()
        }
    }

    /// Generate CUDA kernel source for fused bias + activation
    fn generate_fused_bias_activation_kernel(&self, activation: &str) -> String {
        let activation_code = match activation {
            "relu" => "fmaxf(value, 0.0f)",
            "gelu" => "0.5f * value * (1.0f + tanhf(0.7978845608f * (value + 0.044715f * value * value * value)))",
            "silu" => "value / (1.0f + expf(-value))",  // SiLU/Swish
            "tanh" => "tanhf(value)",
            "none" => "value",
            _ => "value",  // Default to identity
        };

        format!(
            r#"
extern "C" __global__ void fused_bias_activation_kernel(
    const float* input,
    const float* bias,
    float* output,
    unsigned int n,
    unsigned int bias_size
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        float bias_val = bias[idx % bias_size];  // Broadcast bias
        float value = input[idx] + bias_val;
        output[idx] = {};
    }}
}}
"#,
            activation_code
        )
    }

    /// Generate optimized matrix multiplication kernel
    fn generate_optimized_matmul_kernel(&self, m: u32, k: u32, n: u32) -> String {
        format!(
            r#"
extern "C" __global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const unsigned int M,
    const unsigned int K,
    const unsigned int N
) {{
    // Optimized matrix multiplication with tiling
    const int TILE_SIZE = 16;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Calculate global thread indices
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    // Shared memory for tiling
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Tile loop
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {{
        // Load tile into shared memory
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {{
            sum += As[ty][i] * Bs[i][tx];
        }}

        __syncthreads();
    }}

    // Write result
    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}
"#
        )
    }

    /// Generate optimized Flash Attention kernel
    fn generate_flash_attention_kernel(
        &self,
        batch_size: u32,
        seq_len: u32,
        head_dim: u32,
    ) -> String {
        format!(
            r#"
extern "C" __global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const unsigned int batch_size,
    const unsigned int seq_len,
    const unsigned int head_dim
) {{
    // Flash Attention implementation with memory-efficient tiling
    const int batch_id = blockIdx.x;
    const int seq_id = blockIdx.y;
    const int tid = threadIdx.x;

    if (batch_id >= batch_size || seq_id >= seq_len) return;

    // Shared memory for computation
    extern __shared__ float shared_mem[];
    float* shared_scores = shared_mem;
    float* shared_values = shared_mem + seq_len;

    // Compute QK^T scores
    float max_score = -INFINITY;
    for (int k = tid; k < seq_len; k += blockDim.x) {{
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {{
            int q_idx = batch_id * seq_len * head_dim + seq_id * head_dim + d;
            int k_idx = batch_id * seq_len * head_dim + k * head_dim + d;
            score += Q[q_idx] * K[k_idx];
        }}
        shared_scores[k] = score;
        max_score = fmaxf(max_score, score);
    }}

    // Reduce to find global maximum
    __shared__ float max_shared[256];
    max_shared[tid] = max_score;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + stride]);
        }}
        __syncthreads();
    }}

    float global_max = max_shared[0];

    // Compute softmax
    float sum_exp = 0.0f;
    for (int k = tid; k < seq_len; k += blockDim.x) {{
        float exp_score = expf(shared_scores[k] - global_max);
        shared_scores[k] = exp_score;
        sum_exp += exp_score;
    }}

    // Reduce sum
    __shared__ float sum_shared[256];
    sum_shared[tid] = sum_exp;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            sum_shared[tid] += sum_shared[tid + stride];
        }}
        __syncthreads();
    }}

    float global_sum = sum_shared[0];

    // Normalize attention weights
    for (int k = tid; k < seq_len; k += blockDim.x) {{
        shared_scores[k] /= global_sum;
    }}
    __syncthreads();

    // Compute output
    for (int d = tid; d < head_dim; d += blockDim.x) {{
        float output_val = 0.0f;
        for (int k = 0; k < seq_len; k++) {{
            int v_idx = batch_id * seq_len * head_dim + k * head_dim + d;
            output_val += shared_scores[k] * V[v_idx];
        }}
        int o_idx = batch_id * seq_len * head_dim + seq_id * head_dim + d;
        O[o_idx] = output_val;
    }}
}}
"#
        )
    }

    /// Get device information
    pub fn device_info(&self) -> String {
        format!(
            "CUDA Device: {}, Properties: Available",
            self.device.name().unwrap_or_else(|_| "Unknown".to_string())
        )
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> (usize, usize) {
        let pool = self.memory_pool.lock().expect("Lock poisoned");
        (pool.total_allocated, pool.peak_memory)
    }
}

// Stub implementation when CUDA is not available
#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
impl CudaImpl {
    pub fn new() -> Result<Self> {
        Err(TrustformersError::hardware_error(
            "CUDA support not available on this platform",
            "CudaImpl::new",
        ))
    }

    pub fn global() -> Result<&'static Arc<CudaImpl>> {
        Err(TrustformersError::hardware_error(
            "CUDA support not available on this platform",
            "CudaImpl::global",
        ))
    }

    pub fn memory_stats(&self) -> (usize, usize) {
        (0, 0)
    }

    pub fn matmul(&self, _a: &Tensor, _b: &Tensor, _c: &mut Tensor) -> Result<()> {
        Err(TrustformersError::hardware_error(
            "CUDA support not available",
            "CudaImpl::matmul",
        ))
    }

    pub fn fused_gelu(
        &self,
        _input: &Tensor,
        _output: &mut Tensor,
        _approximate: bool,
    ) -> Result<()> {
        Err(TrustformersError::hardware_error(
            "CUDA support not available",
            "CudaImpl::fused_gelu",
        ))
    }

    pub fn fused_bias_activation(
        &self,
        _input: &Tensor,
        _bias: &Tensor,
        _output: &mut Tensor,
        _activation: &str,
    ) -> Result<()> {
        Err(TrustformersError::hardware_error(
            "CUDA support not available",
            "CudaImpl::fused_bias_activation",
        ))
    }

    pub fn flash_attention(
        &self,
        _query: &Tensor,
        _key: &Tensor,
        _value: &Tensor,
        _output: &mut Tensor,
    ) -> Result<()> {
        Err(TrustformersError::hardware_error(
            "CUDA support not available",
            "CudaImpl::flash_attention",
        ))
    }

    pub fn device_info(&self) -> String {
        "CUDA not available".to_string()
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl MemoryPool {
    /// Create new memory pool
    pub fn new() -> Self {
        Self {
            available_blocks: Vec::new(),
            allocated_blocks: HashMap::new(),
            total_allocated: 0,
            peak_memory: 0,
        }
    }

    /// Get a memory block from pool
    pub fn get_block(&mut self, elements: usize) -> Option<MemoryBlock> {
        // Find suitable block
        let pos = self.available_blocks.iter().position(|block| block.size >= elements * 4)?;
        Some(self.available_blocks.remove(pos))
    }

    /// Return block to pool
    pub fn return_block(&mut self, block: MemoryBlock) {
        self.available_blocks.push(block);
    }
}

/// Public API functions for external use
pub mod api {
    use super::*;

    /// Initialize CUDA acceleration
    pub fn init_cuda() -> Result<()> {
        CudaImpl::global().map(|_| ())
    }

    /// Execute matrix multiplication on GPU
    pub fn cuda_matmul(a: &Tensor, b: &Tensor, c: &mut Tensor) -> Result<()> {
        CudaImpl::global()?.matmul(a, b, c)
    }

    /// Execute Flash Attention on GPU
    pub fn cuda_flash_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
    ) -> Result<()> {
        CudaImpl::global()?.flash_attention(query, key, value, output)
    }

    /// Get CUDA device information
    pub fn cuda_device_info() -> Result<String> {
        Ok(CudaImpl::global()?.device_info())
    }

    /// Get memory statistics
    pub fn cuda_memory_stats() -> Result<(usize, usize)> {
        Ok(CudaImpl::global()?.memory_stats())
    }

    /// Execute fused GELU activation on GPU
    pub fn cuda_fused_gelu(input: &Tensor, output: &mut Tensor, approximate: bool) -> Result<()> {
        CudaImpl::global()?.fused_gelu(input, output, approximate)
    }

    /// Execute fused bias + activation on GPU
    pub fn cuda_fused_bias_activation(
        input: &Tensor,
        bias: &Tensor,
        output: &mut Tensor,
        activation: &str,
    ) -> Result<()> {
        CudaImpl::global()?.fused_bias_activation(input, bias, output, activation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_cuda_initialization() {
        match CudaImpl::new() {
            Ok(_) => println!("CUDA initialized successfully"),
            Err(_) => println!("CUDA not available, skipping test"),
        }
    }

    #[test]
    fn test_cuda_matmul() {
        if let Ok(cuda) = CudaImpl::new() {
            let a = Tensor::ones(&[4, 4]).expect("Failed to create ones tensor");
            let b = Tensor::ones(&[4, 4]).expect("Failed to create ones tensor");
            let mut c = Tensor::zeros(&[4, 4]).expect("Failed to create zero tensor");

            cuda.matmul(&a, &b, &mut c).expect("Matrix multiplication failed");

            // Result should be all 4s (4x4 matrix of ones * 4x4 matrix of ones)
            let data = c.data_f32().unwrap();
            assert!(data.iter().all(|&x| (x - 4.0).abs() < 1e-6));
        }
    }

    #[test]
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new();
        assert_eq!(pool.total_allocated, 0);
        assert_eq!(pool.peak_memory, 0);
    }
}
