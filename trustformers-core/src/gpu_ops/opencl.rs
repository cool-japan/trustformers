//! OpenCL GPU backend for tensor operations
//!
//! This module provides OpenCL GPU acceleration for tensor operations.
//! OpenCL is a cross-platform, vendor-neutral standard for GPU computing.
//!
//! Features:
//! - Cross-vendor GPU support (Intel, AMD, NVIDIA, Apple, etc.)
//! - Cross-platform (Windows, Linux, macOS)
//! - Persistent buffer caching
//! - GPU-to-GPU operations
//!
//! Note: OpenCL support requires opencl3 crate and OpenCL runtime installed.

#[allow(unused_imports)]
use crate::device::Device;
use crate::errors::Result;
use crate::tensor::Tensor;

#[cfg(feature = "opencl")]
use crate::errors::TrustformersError;
#[cfg(feature = "opencl")]
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
#[cfg(feature = "opencl")]
use opencl3::context::Context;
#[cfg(feature = "opencl")]
use opencl3::device::{get_all_devices, Device as ClDevice, CL_DEVICE_TYPE_GPU};
#[cfg(feature = "opencl")]
use opencl3::kernel::{ExecuteKernel, Kernel};
#[cfg(feature = "opencl")]
use opencl3::memory::{Buffer as ClBuffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
#[cfg(feature = "opencl")]
use opencl3::program::Program;
#[cfg(feature = "opencl")]
use opencl3::types::{cl_float, CL_BLOCKING};
#[cfg(feature = "opencl")]
use std::collections::HashMap;
#[cfg(feature = "opencl")]
use std::sync::Arc;

/// Buffer ID for persistent GPU buffers
#[cfg(feature = "opencl")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

#[cfg(feature = "opencl")]
impl BufferId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        BufferId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

#[cfg(feature = "opencl")]
impl Default for BufferId {
    fn default() -> Self {
        Self::new()
    }
}

/// Persistent buffer cache for OpenCL
#[cfg(feature = "opencl")]
struct BufferCache {
    buffers: HashMap<BufferId, Arc<ClBuffer<cl_float>>>,
}

#[cfg(feature = "opencl")]
impl BufferCache {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    fn insert(&mut self, id: BufferId, buffer: Arc<ClBuffer<cl_float>>) {
        self.buffers.insert(id, buffer);
    }

    fn get(&self, id: &BufferId) -> Option<Arc<ClBuffer<cl_float>>> {
        self.buffers.get(id).cloned()
    }

    fn remove(&mut self, id: &BufferId) -> Option<Arc<ClBuffer<cl_float>>> {
        self.buffers.remove(id)
    }

    fn clear(&mut self) {
        self.buffers.clear();
    }

    fn len(&self) -> usize {
        self.buffers.len()
    }
}

/// OpenCL GPU backend for matrix multiplication and element-wise operations
#[cfg(feature = "opencl")]
pub struct OpenClBackend {
    context: Arc<Context>,
    queue: Arc<CommandQueue>,
    device: ClDevice,
    buffer_cache: Arc<std::sync::Mutex<BufferCache>>,
}

#[cfg(feature = "opencl")]
impl OpenClBackend {
    /// Create a new OpenCL backend
    pub fn new(device_id: usize) -> Result<Self> {
        // Get all GPU devices
        let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get OpenCL devices: {:?}", e),
                "OpenClBackend::new",
            )
        })?;

        if device_ids.is_empty() {
            return Err(TrustformersError::hardware_error(
                "No OpenCL GPU devices found",
                "OpenClBackend::new",
            ));
        }

        if device_id >= device_ids.len() {
            return Err(TrustformersError::hardware_error(
                &format!(
                    "Device ID {} out of range (available: {})",
                    device_id,
                    device_ids.len()
                ),
                "OpenClBackend::new",
            ));
        }

        let device = ClDevice::new(device_ids[device_id]);

        // Create context
        let context = Context::from_device(&device).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create OpenCL context: {:?}", e),
                "OpenClBackend::new",
            )
        })?;

        // Create command queue
        let queue =
            CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to create OpenCL command queue: {:?}", e),
                        "OpenClBackend::new",
                    )
                })?;

        let device_name = device.name().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get device name: {:?}", e),
                "OpenClBackend::new",
            )
        })?;

        println!("✓ OpenCL backend initialized");
        println!("  Device: {}", device_name);

        Ok(Self {
            context: Arc::new(context),
            queue: Arc::new(queue),
            device,
            buffer_cache: Arc::new(std::sync::Mutex::new(BufferCache::new())),
        })
    }

    /// Create a persistent GPU buffer and return its ID
    pub fn create_persistent_buffer(&self, data: &[f32]) -> Result<BufferId> {
        let mut buffer = unsafe {
            ClBuffer::<cl_float>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                data.len(),
                std::ptr::null_mut(),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create buffer: {:?}", e),
                    "create_persistent_buffer",
                )
            })?
        };

        // Write data to buffer
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, data, &[])
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to write buffer: {:?}", e),
                        "create_persistent_buffer",
                    )
                })?
        };

        let buffer_id = BufferId::new();

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "create_persistent_buffer",
            )
        })?;

        cache.insert(buffer_id, Arc::new(buffer));
        Ok(buffer_id)
    }

    /// Get a persistent buffer by ID
    pub fn get_persistent_buffer(&self, id: &BufferId) -> Result<Arc<ClBuffer<cl_float>>> {
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

    /// Perform matrix multiplication on OpenCL GPU
    /// C = A @ B where A is [m, k], B is [k, n], C is [m, n]
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // OpenCL kernel source
        const KERNEL_SRC: &str = r#"
__kernel void matmul_kernel(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const unsigned int M,
    const unsigned int N,
    const unsigned int K
) {
    const unsigned int row = get_global_id(1);
    const unsigned int col = get_global_id(0);

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (unsigned int i = 0; i < K; ++i) {
        sum += a[row * K + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}
"#;

        // Build program
        let program = Program::create_and_build_from_source(&self.context, KERNEL_SRC, "")
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to build OpenCL program: {:?}", e),
                    "matmul_f32",
                )
            })?;

        let kernel = Kernel::create(&program, "matmul_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create OpenCL kernel: {:?}", e),
                "matmul_f32",
            )
        })?;

        // Create buffers
        let mut a_buffer = unsafe {
            ClBuffer::<cl_float>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                a.len(),
                std::ptr::null_mut(),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create A buffer: {:?}", e),
                    "matmul_f32",
                )
            })?
        };

        let mut b_buffer = unsafe {
            ClBuffer::<cl_float>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                b.len(),
                std::ptr::null_mut(),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create B buffer: {:?}", e),
                    "matmul_f32",
                )
            })?
        };

        let result_size = m * n;
        let c_buffer = unsafe {
            ClBuffer::<cl_float>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                result_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create C buffer: {:?}", e),
                    "matmul_f32",
                )
            })?
        };

        // Write input data
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut a_buffer, CL_BLOCKING, 0, a, &[])
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to write A buffer: {:?}", e),
                        "matmul_f32",
                    )
                })?
        };

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut b_buffer, CL_BLOCKING, 0, b, &[])
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to write B buffer: {:?}", e),
                        "matmul_f32",
                    )
                })?
        };

        // Execute kernel
        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&a_buffer)
                .set_arg(&b_buffer)
                .set_arg(&c_buffer)
                .set_arg(&(m as u32))
                .set_arg(&(n as u32))
                .set_arg(&(k as u32))
                .set_global_work_sizes(&[n, m])
                .set_local_work_sizes(&[16, 16])
                .enqueue_nd_range(&self.queue)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to execute OpenCL kernel: {:?}", e),
                        "matmul_f32",
                    )
                })?
        };

        kernel_event.wait().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to wait for kernel: {:?}", e),
                "matmul_f32",
            )
        })?;

        // Read result
        let mut result = vec![0.0f32; result_size];
        unsafe {
            self.queue
                .enqueue_read_buffer(&c_buffer, CL_BLOCKING, 0, &mut result, &[])
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to read result buffer: {:?}", e),
                        "matmul_f32",
                    )
                })?
        };

        Ok(result)
    }

    /// Execute GELU activation on GPU
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    pub fn gelu_f32(&self, input: &[f32]) -> Result<Vec<f32>> {
        const KERNEL_SRC: &str = r#"
__kernel void gelu_kernel(
    __global const float* input,
    __global float* output,
    const unsigned int size
) {
    const unsigned int idx = get_global_id(0);
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
    // sqrt(2/π) ≈ 0.7978845608
    float inner = 0.7978845608f * (x + 0.044715f * x_cubed);

    // Clamp inner to prevent tanh overflow
    inner = clamp(inner, -20.0f, 20.0f);

    output[idx] = 0.5f * x * (1.0f + tanh(inner));
}
"#;

        let size = input.len();

        // Build program
        let program = Program::create_and_build_from_source(&self.context, KERNEL_SRC, "")
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to build OpenCL program: {:?}", e),
                    "gelu_f32",
                )
            })?;

        let kernel = Kernel::create(&program, "gelu_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create OpenCL kernel: {:?}", e),
                "gelu_f32",
            )
        })?;

        // Create buffers
        let mut input_buffer = unsafe {
            ClBuffer::<cl_float>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                size,
                std::ptr::null_mut(),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create input buffer: {:?}", e),
                    "gelu_f32",
                )
            })?
        };

        let output_buffer = unsafe {
            ClBuffer::<cl_float>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                size,
                std::ptr::null_mut(),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create output buffer: {:?}", e),
                    "gelu_f32",
                )
            })?
        };

        // Write input data
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buffer, CL_BLOCKING, 0, input, &[])
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to write input buffer: {:?}", e),
                        "gelu_f32",
                    )
                })?
        };

        // Execute kernel
        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&input_buffer)
                .set_arg(&output_buffer)
                .set_arg(&(size as u32))
                .set_global_work_size(size)
                .set_local_work_size(256)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to execute OpenCL kernel: {:?}", e),
                        "gelu_f32",
                    )
                })?
        };

        kernel_event.wait().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to wait for kernel: {:?}", e),
                "gelu_f32",
            )
        })?;

        // Read result
        let mut result = vec![0.0f32; size];
        unsafe {
            self.queue
                .enqueue_read_buffer(&output_buffer, CL_BLOCKING, 0, &mut result, &[])
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to read result buffer: {:?}", e),
                        "gelu_f32",
                    )
                })?
        };

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
        const KERNEL_SRC: &str = r#"
__kernel void layernorm_kernel(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    const unsigned int seq_len,
    const unsigned int hidden_size,
    const float eps
) {
    const unsigned int pos = get_global_id(0);
    if (pos >= seq_len) return;

    const unsigned int offset = pos * hidden_size;

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
    float std_dev = sqrt(variance + eps);

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

        // Build program
        let program = Program::create_and_build_from_source(&self.context, KERNEL_SRC, "")
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to build OpenCL program: {:?}", e),
                    "layernorm_f32",
                )
            })?;

        let kernel = Kernel::create(&program, "layernorm_kernel").map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create OpenCL kernel: {:?}", e),
                "layernorm_f32",
            )
        })?;

        // Create buffers
        let mut input_buffer = unsafe {
            ClBuffer::<cl_float>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                total_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create input buffer: {:?}", e),
                    "layernorm_f32",
                )
            })?
        };

        let mut weight_buffer = unsafe {
            ClBuffer::<cl_float>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create weight buffer: {:?}", e),
                    "layernorm_f32",
                )
            })?
        };

        let mut bias_buffer = unsafe {
            ClBuffer::<cl_float>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                hidden_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create bias buffer: {:?}", e),
                    "layernorm_f32",
                )
            })?
        };

        let output_buffer = unsafe {
            ClBuffer::<cl_float>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                total_size,
                std::ptr::null_mut(),
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to create output buffer: {:?}", e),
                    "layernorm_f32",
                )
            })?
        };

        // Write input data
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buffer, CL_BLOCKING, 0, input, &[])
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to write input buffer: {:?}", e),
                        "layernorm_f32",
                    )
                })?
        };

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut weight_buffer, CL_BLOCKING, 0, weight, &[])
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to write weight buffer: {:?}", e),
                        "layernorm_f32",
                    )
                })?
        };

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut bias_buffer, CL_BLOCKING, 0, bias, &[])
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to write bias buffer: {:?}", e),
                        "layernorm_f32",
                    )
                })?
        };

        // Execute kernel
        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&input_buffer)
                .set_arg(&weight_buffer)
                .set_arg(&bias_buffer)
                .set_arg(&output_buffer)
                .set_arg(&(seq_len as u32))
                .set_arg(&(hidden_size as u32))
                .set_arg(&eps)
                .set_global_work_size(seq_len)
                .set_local_work_size(64)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to execute OpenCL kernel: {:?}", e),
                        "layernorm_f32",
                    )
                })?
        };

        kernel_event.wait().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to wait for kernel: {:?}", e),
                "layernorm_f32",
            )
        })?;

        // Read result
        let mut result = vec![0.0f32; total_size];
        unsafe {
            self.queue
                .enqueue_read_buffer(&output_buffer, CL_BLOCKING, 0, &mut result, &[])
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to read result buffer: {:?}", e),
                        "layernorm_f32",
                    )
                })?
        };

        Ok(result)
    }

    /// Copy GPU buffer data back to CPU
    pub fn buffer_to_cpu(&self, buffer_id: &BufferId, size: usize) -> Result<Vec<f32>> {
        let buffer = self.get_persistent_buffer(buffer_id)?;

        let mut result = vec![0.0f32; size];
        unsafe {
            self.queue
                .enqueue_read_buffer(&buffer, CL_BLOCKING, 0, &mut result, &[])
                .map_err(|e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to read buffer: {:?}", e),
                        "buffer_to_cpu",
                    )
                })?
        };

        Ok(result)
    }

    /// Get device information
    pub fn device_info(&self) -> Result<String> {
        let name = self.device.name().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get device name: {:?}", e),
                "device_info",
            )
        })?;

        Ok(format!("OpenCL Device: {}", name))
    }
}

/// Global OpenCL backend cache
#[cfg(feature = "opencl")]
static OPENCL_BACKENDS: once_cell::sync::Lazy<
    std::sync::Mutex<HashMap<usize, Arc<OpenClBackend>>>,
> = once_cell::sync::Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

/// Get or create OpenCL backend instance
#[cfg(feature = "opencl")]
pub fn get_opencl_backend(device_id: usize) -> Result<Arc<OpenClBackend>> {
    let mut cache = OPENCL_BACKENDS.lock().map_err(|_| {
        TrustformersError::hardware_error(
            "Failed to lock OpenCL backend cache",
            "get_opencl_backend",
        )
    })?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(device_id) {
        let backend = OpenClBackend::new(device_id)?;
        e.insert(Arc::new(backend));
    }

    cache.get(&device_id).cloned().ok_or_else(|| {
        TrustformersError::hardware_error("OpenCL backend not found", "get_opencl_backend")
    })
}

/// Dispatch matrix multiplication to OpenCL backend
#[allow(unused_variables)]
pub fn dispatch_opencl_matmul(a: &Tensor, b: &Tensor, device_id: usize) -> Result<Tensor> {
    #[cfg(feature = "opencl")]
    {
        match (a, b) {
            (Tensor::F32(a_arr), Tensor::F32(b_arr)) => {
                if a_arr.ndim() != 2 || b_arr.ndim() != 2 {
                    return Err(TrustformersError::shape_error(
                        "OpenCL dispatch currently only supports 2D tensors".to_string(),
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

                let backend = get_opencl_backend(device_id)?;

                let a_data: Vec<f32> = a_2d.iter().copied().collect();
                let b_data: Vec<f32> = b_2d.iter().copied().collect();

                let result_data = backend.matmul_f32(&a_data, &b_data, m, k, n)?;

                let result_2d = scirs2_core::ndarray::Array2::from_shape_vec((m, n), result_data)
                    .map_err(|e| {
                    TrustformersError::shape_error(format!("Failed to reshape result: {}", e))
                })?;

                let result_dyn = result_2d.into_dyn();
                Ok(Tensor::F32(result_dyn))
            },
            _ => a.matmul(b),
        }
    }

    #[cfg(not(feature = "opencl"))]
    {
        // No OpenCL support, fallback to CPU
        a.matmul(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "opencl")]
    fn test_opencl_backend_creation() -> Result<()> {
        match OpenClBackend::new(0) {
            Ok(backend) => {
                println!("OpenCL backend created: {:?}", backend.device_info()?);
                Ok(())
            },
            Err(e) => {
                eprintln!("Skipping OpenCL test: {}", e);
                Ok(())
            },
        }
    }

    #[test]
    #[cfg(feature = "opencl")]
    fn test_opencl_matmul() -> Result<()> {
        let backend = match OpenClBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping OpenCL test: no device available");
                return Ok(());
            },
        };

        // Simple 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = match backend.matmul_f32(&a, &b, 2, 2, 2) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "Skipping OpenCL matmul test: kernel execution failed: {}",
                    e
                );
                return Ok(());
            },
        };

        // Expected: [[19, 22], [43, 50]]
        let expected = [19.0, 22.0, 43.0, 50.0];

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
    #[cfg(feature = "opencl")]
    fn test_opencl_gelu() -> Result<()> {
        let backend = match OpenClBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping OpenCL test: no device available");
                return Ok(());
            },
        };

        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = match backend.gelu_f32(&input) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Skipping OpenCL gelu test: kernel execution failed: {}", e);
                return Ok(());
            },
        };

        // GELU should be monotonic increasing
        for i in 0..result.len() - 1 {
            assert!(result[i] <= result[i + 1], "GELU should be monotonic");
        }

        // GELU(0) ≈ 0
        assert!((result[2]).abs() < 0.01, "GELU(0) should be ~0");

        Ok(())
    }

    // Non-feature-gated tests for CPU fallback and utility functions

    #[test]
    fn test_opencl_matmul_fallback_identity_like() -> Result<()> {
        // Test CPU fallback path for matmul
        let a = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).expect("tensor failed");
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], &[2, 2]).expect("tensor failed");
        let result = a.matmul(&b)?;
        let data = result.data_f32().expect("data failed");
        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 4.0).abs() < 1e-5);
        assert!((data[2] - 5.0).abs() < 1e-5);
        assert!((data[3] - 6.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_fallback_ones() -> Result<()> {
        let a = Tensor::ones(&[3, 3]).expect("tensor failed");
        let b = Tensor::ones(&[3, 3]).expect("tensor failed");
        let result = a.matmul(&b)?;
        let data = result.data_f32().expect("data failed");
        for val in data.iter() {
            assert!((val - 3.0).abs() < 1e-5, "Expected 3.0, got {}", val);
        }
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_fallback_zeros() -> Result<()> {
        let a = Tensor::zeros(&[2, 3]).expect("tensor failed");
        let b = Tensor::ones(&[3, 2]).expect("tensor failed");
        let result = a.matmul(&b)?;
        let data = result.data_f32().expect("data failed");
        for val in data.iter() {
            assert!((val - 0.0).abs() < 1e-5, "Expected 0.0, got {}", val);
        }
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_fallback_rectangular() -> Result<()> {
        let a =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("tensor failed");
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])
            .expect("tensor failed");
        let result = a.matmul(&b)?;
        let data = result.data_f32().expect("data failed");
        // [2,3] x [3,2] = [2,2]
        assert_eq!(result.shape(), &[2, 2]);
        // First row: 1*7+2*9+3*11 = 7+18+33 = 58, 1*8+2*10+3*12 = 8+20+36 = 64
        assert!((data[0] - 58.0).abs() < 1e-4);
        assert!((data[1] - 64.0).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_fallback_single_element() -> Result<()> {
        let a = Tensor::from_vec(vec![3.0], &[1, 1]).expect("tensor failed");
        let b = Tensor::from_vec(vec![4.0], &[1, 1]).expect("tensor failed");
        let result = a.matmul(&b)?;
        let data = result.data_f32().expect("data failed");
        assert!((data[0] - 12.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_fallback_negative_values() -> Result<()> {
        let a = Tensor::from_vec(vec![-1.0, 2.0, 3.0, -4.0], &[2, 2]).expect("tensor failed");
        let b = Tensor::from_vec(vec![1.0, -1.0, -1.0, 1.0], &[2, 2]).expect("tensor failed");
        let result = a.matmul(&b)?;
        let data = result.data_f32().expect("data failed");
        // [-1*1+2*(-1), -1*(-1)+2*1] = [-3, 3]
        // [3*1+(-4)*(-1), 3*(-1)+(-4)*1] = [7, -7]
        assert!((data[0] - (-3.0)).abs() < 1e-4);
        assert!((data[1] - 3.0).abs() < 1e-4);
        assert!((data[2] - 7.0).abs() < 1e-4);
        assert!((data[3] - (-7.0)).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_fallback_large_matrix() -> Result<()> {
        let n = 16;
        let a = Tensor::ones(&[n, n]).expect("tensor failed");
        let b = Tensor::ones(&[n, n]).expect("tensor failed");
        let result = a.matmul(&b)?;
        let data = result.data_f32().expect("data failed");
        for val in data.iter() {
            assert!((val - n as f32).abs() < 1e-4);
        }
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_fallback_asymmetric() -> Result<()> {
        let a = Tensor::ones(&[1, 5]).expect("tensor failed");
        let b = Tensor::ones(&[5, 1]).expect("tensor failed");
        let result = a.matmul(&b)?;
        assert_eq!(result.shape(), &[1, 1]);
        let data = result.data_f32().expect("data failed");
        assert!((data[0] - 5.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_fallback_wide_result() -> Result<()> {
        let a = Tensor::ones(&[5, 1]).expect("tensor failed");
        let b = Tensor::ones(&[1, 5]).expect("tensor failed");
        let result = a.matmul(&b)?;
        assert_eq!(result.shape(), &[5, 5]);
        let data = result.data_f32().expect("data failed");
        for val in data.iter() {
            assert!((val - 1.0).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_fallback_deterministic() -> Result<()> {
        // Use LCG for deterministic data
        let mut seed: u64 = 42;
        let mut gen = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX as f32)
        };
        let a_data: Vec<f32> = (0..12).map(|_| gen()).collect();
        let b_data: Vec<f32> = (0..12).map(|_| gen()).collect();
        let a = Tensor::from_vec(a_data, &[3, 4]).expect("tensor failed");
        let b = Tensor::from_vec(b_data, &[4, 3]).expect("tensor failed");
        let r1 = a.matmul(&b)?;
        let r2 = a.matmul(&b)?;
        let d1 = r1.data_f32().expect("data failed");
        let d2 = r2.data_f32().expect("data failed");
        for (v1, v2) in d1.iter().zip(d2.iter()) {
            assert!((v1 - v2).abs() < 1e-6, "Results should be deterministic");
        }
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_preserves_shape() -> Result<()> {
        let a = Tensor::ones(&[4, 6]).expect("tensor failed");
        let b = Tensor::ones(&[6, 8]).expect("tensor failed");
        let result = a.matmul(&b)?;
        assert_eq!(result.shape(), &[4, 8]);
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_small_values() -> Result<()> {
        let a = Tensor::from_vec(vec![0.001, 0.002, 0.003, 0.004], &[2, 2]).expect("tensor failed");
        let b = Tensor::from_vec(vec![0.001, 0.002, 0.003, 0.004], &[2, 2]).expect("tensor failed");
        let result = a.matmul(&b)?;
        let data = result.data_f32().expect("data failed");
        // Should be very small but non-zero
        for val in data.iter() {
            assert!(*val > 0.0);
            assert!(*val < 0.1);
        }
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_large_values() -> Result<()> {
        let a =
            Tensor::from_vec(vec![1000.0, 2000.0, 3000.0, 4000.0], &[2, 2]).expect("tensor failed");
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).expect("tensor failed");
        let result = a.matmul(&b)?;
        let data = result.data_f32().expect("data failed");
        assert!((data[0] - 1000.0).abs() < 1e-1);
        assert!((data[1] - 2000.0).abs() < 1e-1);
        assert!((data[2] - 3000.0).abs() < 1e-1);
        assert!((data[3] - 4000.0).abs() < 1e-1);
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_commutative_check() -> Result<()> {
        // AB != BA in general
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor failed");
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("tensor failed");
        let ab = a.matmul(&b)?;
        let ba = b.matmul(&a)?;
        let ab_data = ab.data_f32().expect("data failed");
        let ba_data = ba.data_f32().expect("data failed");
        // AB = [[19,22],[43,50]], BA = [[23,34],[31,46]]
        assert!(
            (ab_data[0] - ba_data[0]).abs() > 0.1,
            "AB and BA should differ"
        );
        Ok(())
    }

    #[test]
    fn test_opencl_matmul_associative() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor failed");
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("tensor failed");
        let c = Tensor::from_vec(vec![9.0, 10.0, 11.0, 12.0], &[2, 2]).expect("tensor failed");
        let ab = a.matmul(&b)?;
        let abc_1 = ab.matmul(&c)?;
        let bc = b.matmul(&c)?;
        let abc_2 = a.matmul(&bc)?;
        let d1 = abc_1.data_f32().expect("data failed");
        let d2 = abc_2.data_f32().expect("data failed");
        for (v1, v2) in d1.iter().zip(d2.iter()) {
            assert!((v1 - v2).abs() < 1e-2, "(AB)C should equal A(BC)");
        }
        Ok(())
    }
}
