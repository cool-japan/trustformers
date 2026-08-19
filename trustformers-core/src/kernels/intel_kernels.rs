//! Intel oneAPI GPU kernel implementations.
//!
//! This module provides GPU-accelerated operations using Intel oneAPI/DPC++.
//! It supports Intel Arc GPUs, Intel Xe integrated graphics, and Intel Data Center GPU Max Series.

use crate::errors::{hardware_error, Result};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Intel oneAPI kernel configuration
#[derive(Debug, Clone)]
pub struct IntelKernelConfig {
    pub device_id: usize,
    pub workgroup_size: usize,
    pub preferred_workgroup_size_multiple: usize,
    pub max_workgroup_size: usize,
    pub local_memory_size: usize,
    pub global_memory_size: usize,
    pub compute_units: usize,
    pub max_clock_frequency: u32,
    pub sub_group_size: usize,
    pub enable_profiling: bool,
    pub enable_fp16: bool,
    pub enable_dpas: bool, // Intel Xe Matrix Extensions (XMX)
}

impl Default for IntelKernelConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            workgroup_size: 256,
            preferred_workgroup_size_multiple: 32,
            max_workgroup_size: 1024,
            local_memory_size: 65536,
            global_memory_size: 16 * 1024 * 1024 * 1024, // 16GB
            compute_units: 96,
            max_clock_frequency: 2200,
            sub_group_size: 16,
            enable_profiling: false,
            enable_fp16: true,
            enable_dpas: true,
        }
    }
}

/// Intel oneAPI GPU kernel manager
pub struct IntelKernel {
    config: IntelKernelConfig,
    device: IntelDevice,
    #[allow(dead_code)]
    context: IntelContext,
    #[allow(dead_code)]
    command_queue: IntelCommandQueue,
    compiled_kernels: HashMap<String, IntelCompiledKernel>,
    memory_pool: Arc<Mutex<IntelMemoryPool>>,
    #[allow(dead_code)]
    profiling_enabled: bool,
}

/// Intel GPU device information
#[derive(Debug, Clone)]
pub struct IntelDevice {
    pub id: usize,
    pub name: String,
    pub vendor: String,
    pub driver_version: String,
    pub device_type: IntelDeviceType,
    pub compute_units: usize,
    pub max_clock_frequency: u32,
    pub local_memory_size: usize,
    pub global_memory_size: usize,
    pub max_workgroup_size: usize,
    pub sub_group_sizes: Vec<usize>,
    pub extensions: Vec<String>,
    pub supports_fp16: bool,
    pub supports_dpas: bool,
    pub supports_systolic_arrays: bool,
}

/// Intel GPU device types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntelDeviceType {
    /// Intel Arc discrete GPU (e.g., A770, A750, A380)
    Arc,
    /// Intel Xe integrated GPU (e.g., Xe-LP, Xe-HPG)
    Xe,
    /// Intel Data Center GPU Max Series (e.g., Max 1550, Max 1100)
    DataCenterMax,
    /// Intel Iris Xe integrated GPU
    IrisXe,
    /// Intel UHD integrated GPU
    UHD,
    /// Unknown Intel GPU
    Unknown,
}

/// Intel oneAPI context
pub struct IntelContext {
    // Internal oneAPI context handle
    handle: Option<usize>, // Use usize instead of raw pointer for thread safety
    #[allow(dead_code)]
    device_id: usize,
}

/// Intel oneAPI command queue
pub struct IntelCommandQueue {
    // Internal oneAPI command queue handle
    #[allow(dead_code)]
    handle: Option<usize>, // Use usize instead of raw pointer for thread safety
    #[allow(dead_code)]
    context: Option<usize>, // Use usize instead of raw pointer for thread safety
}

/// Compiled Intel oneAPI kernel
pub struct IntelCompiledKernel {
    // Internal kernel handle
    #[allow(dead_code)]
    handle: Option<usize>, // Use usize instead of raw pointer for thread safety
    #[allow(dead_code)]
    name: String,
    source_hash: u64,
    #[allow(dead_code)]
    workgroup_size: usize,
    #[allow(dead_code)]
    local_memory_size: usize,
    #[allow(dead_code)]
    compilation_time: std::time::Duration,
}

/// Intel oneAPI memory pool for efficient memory management
pub struct IntelMemoryPool {
    #[allow(dead_code)]
    allocations: HashMap<usize, IntelMemoryAllocation>,
    total_allocated: usize,
    peak_allocated: usize,
    allocation_count: usize,
    free_list: Vec<(usize, usize)>, // (size, address)
}

/// Intel oneAPI memory allocation
pub struct IntelMemoryAllocation {
    #[allow(dead_code)]
    ptr: Option<usize>, // Use usize instead of raw pointer for thread safety
    #[allow(dead_code)]
    size: usize,
    #[allow(dead_code)]
    alignment: usize,
    #[allow(dead_code)]
    allocated_at: std::time::Instant,
}

/// Intel GPU precision types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntelPrecision {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
}

impl IntelKernel {
    /// Create a new Intel oneAPI kernel manager
    pub fn new(config: IntelKernelConfig) -> Result<Self> {
        let device = Self::detect_device(config.device_id)?;
        let context = Self::create_context(&device)?;
        let command_queue = Self::create_command_queue(&context, &device)?;
        let memory_pool = Arc::new(Mutex::new(IntelMemoryPool::new()));

        Ok(Self {
            config,
            device,
            context,
            command_queue,
            compiled_kernels: HashMap::new(),
            memory_pool,
            profiling_enabled: false,
        })
    }

    /// Detect Intel GPU device `device_id` by looking it up in
    /// `IntelUtils::detect_devices`. There is no real oneAPI/Level-Zero/SYCL
    /// binding in this build (the `intel` feature pulls in no FFI
    /// dependency), so there is no runtime this can genuinely query, and
    /// `detect_devices` therefore never reports one: fabricating a fixed
    /// "Intel Arc A770" identity regardless of what hardware (if any) is
    /// actually attached would misrepresent every machine that lacks one.
    fn detect_device(device_id: usize) -> Result<IntelDevice> {
        IntelUtils::detect_devices()?
            .into_iter()
            .find(|d| d.id == device_id)
            .ok_or_else(|| {
                hardware_error(
                    format!("intel device {device_id}"),
                    "no real Intel oneAPI/Level-Zero runtime is available in this build to detect \
                 a device (the `intel` feature provides no FFI binding)",
                )
            })
    }

    /// Create oneAPI context
    fn create_context(device: &IntelDevice) -> Result<IntelContext> {
        // In a real implementation, this would create a oneAPI/SYCL context
        Ok(IntelContext {
            handle: None,
            device_id: device.id,
        })
    }

    /// Create command queue
    fn create_command_queue(
        context: &IntelContext,
        device: &IntelDevice,
    ) -> Result<IntelCommandQueue> {
        // A command queue only makes sense bound to the device its context was
        // created for; catching a mismatch here is the one real check this
        // simulated (no FFI binding) constructor can still make.
        if device.id != context.device_id {
            return Err(hardware_error(
                format!("intel device {}", device.id),
                format!(
                    "command queue device does not match the context's device {}",
                    context.device_id
                ),
            ));
        }
        // In a real implementation, this would create a oneAPI command queue
        Ok(IntelCommandQueue {
            handle: None,
            context: context.handle,
        })
    }

    /// Compile kernel from DPC++ source
    pub fn compile_kernel(&mut self, name: &str, source: &str) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Calculate source hash for caching
        let source_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            source.hash(&mut hasher);
            hasher.finish()
        };

        // Check if already compiled
        if let Some(cached) = self.compiled_kernels.get(name) {
            if cached.source_hash == source_hash {
                return Ok(());
            }
        }

        // Compile kernel (simulated)
        let compiled_kernel = IntelCompiledKernel {
            handle: None,
            name: name.to_string(),
            source_hash,
            workgroup_size: self.config.workgroup_size,
            local_memory_size: self.config.local_memory_size,
            compilation_time: start_time.elapsed(),
        };

        self.compiled_kernels.insert(name.to_string(), compiled_kernel);
        Ok(())
    }

    /// Execute matrix multiplication kernel
    pub fn gemm(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        c: &mut Tensor,
        alpha: f32,
        beta: f32,
        precision: IntelPrecision,
    ) -> Result<()> {
        let kernel_name = format!("gemm_{:?}", precision);

        // Compile kernel if not already compiled
        if !self.compiled_kernels.contains_key(&kernel_name) {
            let source = self.generate_gemm_kernel(precision)?;
            self.compile_kernel(&kernel_name, &source)?;
        }

        // Execute kernel (simulated)
        // In a real implementation, this would:
        // 1. Allocate GPU memory for tensors
        // 2. Copy data to GPU
        // 3. Set kernel arguments
        // 4. Launch kernel with appropriate workgroup size
        // 5. Copy results back to CPU

        // For now, fall back to CPU implementation
        Self::gemm_cpu_fallback(a, b, c, alpha, beta)
    }

    /// Generate optimized GEMM kernel source code
    fn generate_gemm_kernel(&self, precision: IntelPrecision) -> Result<String> {
        let data_type = match precision {
            IntelPrecision::FP32 => "float",
            IntelPrecision::FP16 => "half",
            IntelPrecision::BF16 => "bfloat16",
            IntelPrecision::INT8 => "int8_t",
            IntelPrecision::INT4 => "int4_t",
        };

        let kernel_source = format!(
            r#"
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

using namespace sycl;

// Optimized GEMM kernel for Intel GPUs
void gemm_kernel(
    queue& q,
    const {data_type}* a,
    const {data_type}* b,
    {data_type}* c,
    int m, int n, int k,
    {data_type} alpha,
    {data_type} beta
) {{
    // Use Intel XMX instructions for matrix multiplication if available
    #ifdef INTEL_XMX_AVAILABLE
    // Use DPAS (Dot Product Accumulate Systolic) instructions
    q.submit([&](handler& h) {{
        h.parallel_for<class gemm_dpas>(
            nd_range<2>({{m, n}}, {{16, 16}}),
            [=](nd_item<2> item) {{
                int i = item.get_global_id(0);
                int j = item.get_global_id(1);

                if (i < m && j < n) {{
                    {data_type} sum = 0;

                    // Use subgroup matrix multiply-accumulate
                    auto sg = item.get_sub_group();

                    // Tile the computation for better cache locality
                    for (int tile = 0; tile < k; tile += 16) {{
                        // Load tiles into subgroup local memory
                        // Use DPAS instructions for 4x4 matrix multiplication
                        sum += intel_sub_group_f16_f16_matrix_mad_k16(
                            a + i * k + tile,
                            b + tile * n + j,
                            sum
                        );
                    }}

                    c[i * n + j] = alpha * sum + beta * c[i * n + j];
                }}
            }}
        );
    }});
    #else
    // Standard implementation for non-XMX GPUs
    q.submit([&](handler& h) {{
        h.parallel_for<class gemm_standard>(
            nd_range<2>({{m, n}}, {{16, 16}}),
            [=](nd_item<2> item) {{
                int i = item.get_global_id(0);
                int j = item.get_global_id(1);

                if (i < m && j < n) {{
                    {data_type} sum = 0;
                    for (int l = 0; l < k; l++) {{
                        sum += a[i * k + l] * b[l * n + j];
                    }}
                    c[i * n + j] = alpha * sum + beta * c[i * n + j];
                }}
            }}
        );
    }});
    #endif
}}
"#,
            data_type = data_type
        );

        Ok(kernel_source)
    }

    /// Execute layer normalization kernel
    pub fn layer_norm(
        &mut self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        output: &mut Tensor,
        eps: f32,
        precision: IntelPrecision,
    ) -> Result<()> {
        let kernel_name = format!("layer_norm_{:?}", precision);

        if !self.compiled_kernels.contains_key(&kernel_name) {
            let source = self.generate_layer_norm_kernel(precision)?;
            self.compile_kernel(&kernel_name, &source)?;
        }

        // Execute kernel (simulated)
        // For now, fall back to CPU implementation
        Self::layer_norm_cpu_fallback(input, weight, bias, output, eps)
    }

    /// Generate optimized layer normalization kernel
    fn generate_layer_norm_kernel(&self, precision: IntelPrecision) -> Result<String> {
        let data_type = match precision {
            IntelPrecision::FP32 => "float",
            IntelPrecision::FP16 => "half",
            IntelPrecision::BF16 => "bfloat16",
            IntelPrecision::INT8 => "int8_t",
            IntelPrecision::INT4 => "int4_t",
        };

        let kernel_source = format!(
            r#"
#include <sycl/sycl.hpp>

using namespace sycl;

// Optimized Layer Normalization kernel for Intel GPUs
void layer_norm_kernel(
    queue& q,
    const {data_type}* input,
    const {data_type}* weight,
    const {data_type}* bias,
    {data_type}* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    {data_type} eps
) {{
    q.submit([&](handler& h) {{
        // Use local memory for reduction
        auto local_mem = local_accessor<{data_type}>(256, h);

        h.parallel_for<class layer_norm>(
            nd_range<2>({{batch_size * seq_len, 256}}, {{1, 256}}),
            [=](nd_item<2> item) {{
                int batch_seq = item.get_global_id(0);
                int tid = item.get_local_id(1);
                int local_size = item.get_local_range(1);

                if (batch_seq >= batch_size * seq_len) return;

                const {data_type}* input_row = input + batch_seq * hidden_size;
                {data_type}* output_row = output + batch_seq * hidden_size;

                // Compute mean using subgroup reduction
                {data_type} sum = 0;
                for (int i = tid; i < hidden_size; i += local_size) {{
                    sum += input_row[i];
                }}

                // Reduce within subgroup
                auto sg = item.get_sub_group();
                sum = reduce_over_group(sg, sum, plus<{data_type}>());

                // Reduce across subgroups
                if (sg.get_local_id()[0] == 0) {{
                    local_mem[sg.get_group_id()[0]] = sum;
                }}

                item.barrier(access::fence_space::local_space);

                if (tid == 0) {{
                    {data_type} mean = 0;
                    for (int i = 0; i < local_size / sg.get_local_range()[0]; i++) {{
                        mean += local_mem[i];
                    }}
                    mean /= hidden_size;
                    local_mem[0] = mean;
                }}

                item.barrier(access::fence_space::local_space);
                {data_type} mean = local_mem[0];

                // Compute variance
                {data_type} var_sum = 0;
                for (int i = tid; i < hidden_size; i += local_size) {{
                    {data_type} diff = input_row[i] - mean;
                    var_sum += diff * diff;
                }}

                // Reduce variance
                var_sum = reduce_over_group(sg, var_sum, plus<{data_type}>());

                if (sg.get_local_id()[0] == 0) {{
                    local_mem[sg.get_group_id()[0]] = var_sum;
                }}

                item.barrier(access::fence_space::local_space);

                if (tid == 0) {{
                    {data_type} variance = 0;
                    for (int i = 0; i < local_size / sg.get_local_range()[0]; i++) {{
                        variance += local_mem[i];
                    }}
                    variance /= hidden_size;
                    local_mem[0] = variance;
                }}

                item.barrier(access::fence_space::local_space);
                {data_type} variance = local_mem[0];

                // Normalize and scale
                {data_type} inv_std = rsqrt(variance + eps);
                for (int i = tid; i < hidden_size; i += local_size) {{
                    {data_type} normalized = (input_row[i] - mean) * inv_std;
                    output_row[i] = normalized * weight[i];
                    if (bias) {{
                        output_row[i] += bias[i];
                    }}
                }}
            }}
        );
    }});
}}
"#,
            data_type = data_type
        );

        Ok(kernel_source)
    }

    /// Execute attention kernel
    pub fn attention(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
        scale: f32,
        precision: IntelPrecision,
    ) -> Result<()> {
        let kernel_name = format!("attention_{:?}", precision);

        if !self.compiled_kernels.contains_key(&kernel_name) {
            let source = self.generate_attention_kernel(precision)?;
            self.compile_kernel(&kernel_name, &source)?;
        }

        // Execute kernel (simulated)
        // For now, fall back to CPU implementation
        Self::attention_cpu_fallback(query, key, value, output, scale)
    }

    /// Generate optimized attention kernel
    fn generate_attention_kernel(&self, precision: IntelPrecision) -> Result<String> {
        let data_type = match precision {
            IntelPrecision::FP32 => "float",
            IntelPrecision::FP16 => "half",
            IntelPrecision::BF16 => "bfloat16",
            IntelPrecision::INT8 => "int8_t",
            IntelPrecision::INT4 => "int4_t",
        };

        let kernel_source = format!(
            r#"
#include <sycl/sycl.hpp>

using namespace sycl;

// Optimized Flash Attention kernel for Intel GPUs
void flash_attention_kernel(
    queue& q,
    const {data_type}* query,
    const {data_type}* key,
    const {data_type}* value,
    {data_type}* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    {data_type} scale
) {{
    // Use tiled attention to reduce memory usage
    const int TILE_SIZE = 64;

    q.submit([&](handler& h) {{
        // Allocate local memory for tiles
        auto q_tile = local_accessor<{data_type}>(TILE_SIZE * head_dim, h);
        auto k_tile = local_accessor<{data_type}>(TILE_SIZE * head_dim, h);
        auto v_tile = local_accessor<{data_type}>(TILE_SIZE * head_dim, h);
        auto scores_tile = local_accessor<{data_type}>(TILE_SIZE * TILE_SIZE, h);

        h.parallel_for<class flash_attention>(
            nd_range<3>({{batch_size, num_heads, seq_len}}, {{1, 1, TILE_SIZE}}),
            [=](nd_item<3> item) {{
                int batch = item.get_global_id(0);
                int head = item.get_global_id(1);
                int q_idx = item.get_global_id(2);
                int tid = item.get_local_id(2);

                if (batch >= batch_size || head >= num_heads || q_idx >= seq_len) return;

                // Load query vector
                const {data_type}* q_ptr = query + (batch * num_heads + head) * seq_len * head_dim + q_idx * head_dim;

                {data_type} max_score = -INFINITY;
                {data_type} sum_exp = 0;
                {data_type} output_acc[head_dim];

                // Initialize accumulator
                for (int d = 0; d < head_dim; d++) {{
                    output_acc[d] = 0;
                }}

                // Process key-value pairs in tiles
                for (int k_start = 0; k_start < seq_len; k_start += TILE_SIZE) {{
                    int k_end = min(k_start + TILE_SIZE, seq_len);
                    int tile_size = k_end - k_start;

                    // Load key tile
                    const {data_type}* k_ptr = key + (batch * num_heads + head) * seq_len * head_dim + k_start * head_dim;
                    for (int k = tid; k < tile_size * head_dim; k += TILE_SIZE) {{
                        k_tile[k] = k_ptr[k];
                    }}

                    // Load value tile
                    const {data_type}* v_ptr = value + (batch * num_heads + head) * seq_len * head_dim + k_start * head_dim;
                    for (int v = tid; v < tile_size * head_dim; v += TILE_SIZE) {{
                        v_tile[v] = v_ptr[v];
                    }}

                    item.barrier(access::fence_space::local_space);

                    // Compute attention scores for this tile
                    {data_type} tile_max = -INFINITY;
                    for (int k = 0; k < tile_size; k++) {{
                        {data_type} score = 0;
                        for (int d = 0; d < head_dim; d++) {{
                            score += q_ptr[d] * k_tile[k * head_dim + d];
                        }}
                        score *= scale;
                        scores_tile[k] = score;
                        tile_max = max(tile_max, score);
                    }}

                    // Update global maximum
                    {data_type} new_max = max(max_score, tile_max);
                    {data_type} old_scale = exp(max_score - new_max);
                    {data_type} tile_scale = exp(tile_max - new_max);

                    // Rescale previous accumulator
                    for (int d = 0; d < head_dim; d++) {{
                        output_acc[d] *= old_scale;
                    }}
                    sum_exp *= old_scale;

                    // Compute softmax and accumulate
                    {data_type} tile_sum = 0;
                    for (int k = 0; k < tile_size; k++) {{
                        {data_type} prob = exp(scores_tile[k] - new_max);
                        tile_sum += prob;

                        // Accumulate weighted values
                        for (int d = 0; d < head_dim; d++) {{
                            output_acc[d] += prob * v_tile[k * head_dim + d];
                        }}
                    }}

                    sum_exp += tile_sum;
                    max_score = new_max;

                    item.barrier(access::fence_space::local_space);
                }}

                // Normalize output
                {data_type}* out_ptr = output + (batch * num_heads + head) * seq_len * head_dim + q_idx * head_dim;
                for (int d = 0; d < head_dim; d++) {{
                    out_ptr[d] = output_acc[d] / sum_exp;
                }}
            }}
        );
    }});
}}
"#,
            data_type = data_type
        );

        Ok(kernel_source)
    }

    /// Get device information
    pub fn device_info(&self) -> &IntelDevice {
        &self.device
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> Result<IntelMemoryStats> {
        let pool = self.memory_pool.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(IntelMemoryStats {
            total_allocated: pool.total_allocated,
            peak_allocated: pool.peak_allocated,
            allocation_count: pool.allocation_count,
            fragmentation_ratio: pool.fragmentation_ratio(),
        })
    }

    /// CPU fallback implementations.
    ///
    /// These are reached on every call: `gemm`/`layer_norm`/`attention` above
    /// only ever "compile" a kernel source string and then immediately call
    /// straight through to these, since the `intel` feature has no real
    /// oneAPI/Level-Zero FFI binding to dispatch to instead (see
    /// [`Self::detect_device`]). They used to be empty `Ok(())` bodies that
    /// left `c`/`output` completely untouched while reporting success - every
    /// caller silently got back whatever garbage was already in its output
    /// tensor. `alpha`/`beta`/`eps`/`scale` were computed and threaded all the
    /// way down here and then never read. Not "optimized" (no BLAS, no SIMD
    /// intrinsics), but a real, correct computation on the CPU, which is what
    /// a function named `_cpu_fallback` promises.
    fn gemm_cpu_fallback(
        a: &Tensor,
        b: &Tensor,
        c: &mut Tensor,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        // C = alpha * (A @ B) + beta * C
        let product = a.matmul(b)?;
        let scaled_product = product.mul_scalar(alpha)?;
        *c = if beta == 0.0 {
            scaled_product
        } else {
            scaled_product.add(&c.mul_scalar(beta)?)?
        };
        Ok(())
    }

    fn layer_norm_cpu_fallback(
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        output: &mut Tensor,
        eps: f32,
    ) -> Result<()> {
        // Normalize over the last (feature) axis, then apply the learned
        // affine transform: out = normalize(input) * weight + bias.
        let normalized = input.layer_norm(-1, eps)?;
        let scaled = normalized.mul(weight)?;
        *output = match bias {
            Some(b) => scaled.add(b)?,
            None => scaled,
        };
        Ok(())
    }

    fn attention_cpu_fallback(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
        scale: f32,
    ) -> Result<()> {
        // softmax(scale * Q K^T) V over [batch, seq, head_dim] tensors; `scale`
        // is computed by the caller (see `IntelImpl::flash_attention`) rather
        // than re-derived here.
        let q_shape = query.shape();
        if q_shape.len() != 3 {
            return Err(hardware_error(
                "intel attention",
                format!("expected a 3-D [batch, seq, head_dim] query, got {q_shape:?}"),
            ));
        }
        for (name, tensor) in [("key", key), ("value", value)] {
            if tensor.shape() != q_shape {
                return Err(hardware_error(
                    "intel attention",
                    format!(
                        "{name} shape {:?} must match query shape {q_shape:?}",
                        tensor.shape()
                    ),
                ));
            }
        }

        let key_transposed = key.transpose(1, 2)?;
        let scores = query.matmul(&key_transposed)?;
        let scaled_scores = scores.mul_scalar(scale)?;
        let attention_weights = scaled_scores.softmax(2)?;
        *output = attention_weights.matmul(value)?;
        Ok(())
    }
}

/// Intel GPU memory statistics
#[derive(Debug, Clone)]
pub struct IntelMemoryStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub allocation_count: usize,
    pub fragmentation_ratio: f32,
}

impl IntelMemoryPool {
    fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            total_allocated: 0,
            peak_allocated: 0,
            allocation_count: 0,
            free_list: Vec::new(),
        }
    }

    fn fragmentation_ratio(&self) -> f32 {
        if self.total_allocated == 0 {
            0.0
        } else {
            let free_space: usize = self.free_list.iter().map(|(size, _)| size).sum();
            free_space as f32 / self.total_allocated as f32
        }
    }
}

/// Intel oneAPI utilities
pub struct IntelUtils;

impl IntelUtils {
    /// Detect available Intel GPU devices.
    ///
    /// There is no real oneAPI/Level-Zero/SYCL binding in this build (the
    /// `intel` feature pulls in no FFI dependency - see the workspace
    /// `Cargo.toml`), so there is no runtime this can genuinely enumerate.
    /// This used to unconditionally fabricate a fixed "Intel Arc A770"
    /// entry, so every machine - including ones with no Intel GPU at all -
    /// saw one reported as present. Honestly report zero devices instead;
    /// callers that need a device (`IntelKernel::new`) then fail clearly
    /// rather than silently operating against invented hardware.
    pub fn detect_devices() -> Result<Vec<IntelDevice>> {
        Ok(vec![])
    }

    /// Get optimal workgroup size for a given problem size
    pub fn get_optimal_workgroup_size(problem_size: usize, max_workgroup_size: usize) -> usize {
        // Simple heuristic for workgroup size selection - prefer larger sizes for better performance
        let candidates = vec![1024, 512, 256, 128, 64, 32];

        for &size in &candidates {
            if size <= max_workgroup_size && problem_size.is_multiple_of(size) {
                return size;
            }
        }

        // Fall back to a reasonable default
        256.min(max_workgroup_size)
    }

    /// Check if Intel XMX (Xe Matrix Extensions) is available
    pub fn has_xmx_support(device: &IntelDevice) -> bool {
        device.supports_dpas && device.supports_systolic_arrays
    }

    /// Get recommended precision for a given device
    pub fn get_recommended_precision(device: &IntelDevice) -> IntelPrecision {
        if device.supports_fp16 {
            IntelPrecision::FP16
        } else {
            IntelPrecision::FP32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `detect_devices` used to unconditionally fabricate
    /// an "Intel Arc A770" entry. With no real oneAPI/Level-Zero runtime
    /// wired up, honest detection must report zero devices.
    #[test]
    fn test_intel_device_detection_reports_no_phantom_devices() {
        let devices = IntelUtils::detect_devices().expect("operation failed in test");
        assert!(devices.is_empty());
    }

    /// Regression test: `IntelKernel::new` used to always succeed (via the
    /// fabricated device from `detect_device`) even with no real GPU
    /// present. It must now honestly fail instead of reporting a phantom
    /// "Intel Arc A770" kernel manager as ready for use.
    #[test]
    fn test_intel_kernel_creation_errors_without_real_hardware() {
        let config = IntelKernelConfig::default();
        let result = IntelKernel::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_workgroup_size_selection() {
        let optimal_size = IntelUtils::get_optimal_workgroup_size(1024, 256);
        assert_eq!(optimal_size, 256);

        let optimal_size = IntelUtils::get_optimal_workgroup_size(512, 1024);
        assert_eq!(optimal_size, 512);
    }

    #[test]
    fn test_xmx_support_detection() {
        let device = IntelDevice {
            id: 0,
            name: "Intel Arc A770".to_string(),
            vendor: "Intel Corporation".to_string(),
            driver_version: "31.0.101.4146".to_string(),
            device_type: IntelDeviceType::Arc,
            compute_units: 32,
            max_clock_frequency: 2400,
            local_memory_size: 65536,
            global_memory_size: 16 * 1024 * 1024 * 1024,
            max_workgroup_size: 1024,
            sub_group_sizes: vec![8, 16, 32],
            extensions: vec![],
            supports_fp16: true,
            supports_dpas: true,
            supports_systolic_arrays: true,
        };

        assert!(IntelUtils::has_xmx_support(&device));
    }

    #[test]
    fn test_precision_recommendation() {
        let device = IntelDevice {
            id: 0,
            name: "Intel Arc A770".to_string(),
            vendor: "Intel Corporation".to_string(),
            driver_version: "31.0.101.4146".to_string(),
            device_type: IntelDeviceType::Arc,
            compute_units: 32,
            max_clock_frequency: 2400,
            local_memory_size: 65536,
            global_memory_size: 16 * 1024 * 1024 * 1024,
            max_workgroup_size: 1024,
            sub_group_sizes: vec![8, 16, 32],
            extensions: vec![],
            supports_fp16: true,
            supports_dpas: true,
            supports_systolic_arrays: true,
        };

        assert_eq!(
            IntelUtils::get_recommended_precision(&device),
            IntelPrecision::FP16
        );
    }

    /// Regression test: `gemm_cpu_fallback` used to be an empty `Ok(())` body
    /// under the file's now-removed blanket `#![allow(unused_variables)]`, so
    /// it left `c` completely untouched. `c` starts at a garbage sentinel
    /// value that is not the correct product, so the old code (which returns
    /// `c` unchanged) would have failed this assertion.
    #[test]
    fn gemm_cpu_fallback_actually_computes_the_product() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
        let mut c = Tensor::from_vec(vec![99.0, 99.0, 99.0, 99.0], &[2, 2])?;

        IntelKernel::gemm_cpu_fallback(&a, &b, &mut c, 1.0, 0.0)?;
        assert_eq!(c.data()?, vec![19.0, 22.0, 43.0, 50.0]);

        // beta != 0 must blend in the previous contents of `c`, not just
        // overwrite them again with the same product.
        let mut c_with_beta = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2])?;
        IntelKernel::gemm_cpu_fallback(&a, &b, &mut c_with_beta, 2.0, 1.0)?;
        assert_eq!(c_with_beta.data()?, vec![39.0, 45.0, 87.0, 101.0]);
        Ok(())
    }

    /// Regression test: `layer_norm_cpu_fallback` used to be an empty
    /// `Ok(())` body, leaving `output` at its garbage sentinel value instead
    /// of the normalized-and-scaled result.
    #[test]
    fn layer_norm_cpu_fallback_actually_normalizes() -> Result<()> {
        // All-zero input normalizes to all-zero (mean subtracted from itself),
        // so the affine transform's output is exactly `bias` regardless of
        // `weight` - a numerically exact expectation with no floating-point
        // approximation to tolerate.
        let input = Tensor::from_vec(vec![0.0; 4], &[1, 4])?;
        let weight = Tensor::from_vec(vec![3.0; 4], &[4])?;
        let bias = Tensor::from_vec(vec![5.0; 4], &[4])?;
        let mut output = Tensor::from_vec(vec![99.0; 4], &[1, 4])?;

        IntelKernel::layer_norm_cpu_fallback(&input, &weight, Some(&bias), &mut output, 1e-5)?;
        assert_eq!(output.data()?, vec![5.0, 5.0, 5.0, 5.0]);
        Ok(())
    }

    /// Regression test: `attention_cpu_fallback` used to be an empty
    /// `Ok(())` body, leaving `output` at its garbage sentinel value.
    /// `value` is constant across the sequence, so any convex combination of
    /// its rows (i.e. any valid softmax-weighted sum) reproduces that same
    /// row exactly - an exact expectation that does not depend on the actual
    /// attention weights `query`/`key` produce.
    #[test]
    fn attention_cpu_fallback_actually_computes_attention() -> Result<()> {
        let query = Tensor::from_vec(vec![0.3, -0.1, 0.7, 0.2], &[1, 2, 2])?;
        let key = Tensor::from_vec(vec![-0.4, 0.5, 0.1, -0.2], &[1, 2, 2])?;
        let value = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 2, 2])?;
        let mut output = Tensor::from_vec(vec![99.0; 4], &[1, 2, 2])?;

        IntelKernel::attention_cpu_fallback(&query, &key, &value, &mut output, 0.5)?;
        let data = output.data()?;
        for v in data {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
        Ok(())
    }

    /// `create_command_queue` used to ignore `device` entirely; it must now
    /// reject a queue request for a device that does not match the context
    /// it was created for.
    #[test]
    fn create_command_queue_rejects_a_device_context_mismatch() {
        let context = IntelContext {
            handle: None,
            device_id: 0,
        };
        let mismatched_device = IntelDevice {
            id: 1,
            name: "Intel Arc A770".to_string(),
            vendor: "Intel Corporation".to_string(),
            driver_version: "31.0.101.4146".to_string(),
            device_type: IntelDeviceType::Arc,
            compute_units: 32,
            max_clock_frequency: 2400,
            local_memory_size: 65536,
            global_memory_size: 16 * 1024 * 1024 * 1024,
            max_workgroup_size: 1024,
            sub_group_sizes: vec![8, 16, 32],
            extensions: vec![],
            supports_fp16: true,
            supports_dpas: true,
            supports_systolic_arrays: true,
        };

        assert!(IntelKernel::create_command_queue(&context, &mismatched_device).is_err());
    }
}
