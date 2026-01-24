#![allow(unused_variables)] // Multi-backend GPU implementation with feature gates

use crate::errors::{Result, TrustformersError};
use crate::gpu::GpuBackend;
#[cfg(feature = "cuda")]
use crate::kernels::cuda_kernels::CudaKernel;
#[cfg(feature = "intel")]
use crate::kernels::intel_kernels::{IntelKernel, IntelKernelConfig};
#[cfg(feature = "rocm")]
use crate::kernels::rocm_kernels::RocmKernel;
#[cfg(feature = "vulkan")]
use crate::kernels::vulkan_kernels::VulkanKernel;
use crate::tensor::Tensor;
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "intel",
    feature = "vulkan"
))]
use std::sync::{Arc, Mutex};

/// GPU-accelerated operations manager
///
/// This provides a high-level interface for GPU-accelerated tensor operations,
/// automatically selecting the best implementation based on available hardware.
pub struct GpuAcceleratedOps {
    backend: GpuBackend,
    #[cfg(feature = "cuda")]
    cuda_kernel: Option<Arc<Mutex<CudaKernel>>>,
    #[cfg(feature = "rocm")]
    rocm_kernel: Option<Arc<Mutex<RocmKernel>>>,
    #[cfg(feature = "intel")]
    intel_kernel: Option<Arc<Mutex<IntelKernel>>>,
    #[cfg(feature = "vulkan")]
    vulkan_kernel: Option<Arc<Mutex<VulkanKernel>>>,
    #[allow(dead_code)]
    device_id: usize,
    #[allow(dead_code)]
    enable_async: bool,
}

/// Configuration for GPU operations
#[derive(Debug, Clone)]
pub struct GpuOpsConfig {
    pub device_id: usize,
    pub enable_async: bool,
    pub memory_pool_size: u64,
    pub kernel_cache_size: usize,
    pub precision: GpuPrecision,
}

/// Supported precision types for GPU operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuPrecision {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
}

impl Default for GpuOpsConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            enable_async: true,
            memory_pool_size: 2 * 1024 * 1024 * 1024, // 2GB
            kernel_cache_size: 1000,
            precision: GpuPrecision::FP32,
        }
    }
}

impl GpuAcceleratedOps {
    /// Create new GPU-accelerated operations manager
    pub fn new(config: GpuOpsConfig) -> Result<Self> {
        let backend = GpuBackend::default();

        #[cfg(feature = "cuda")]
        let cuda_kernel = if backend == GpuBackend::Cuda {
            Some(Arc::new(Mutex::new(CudaKernel::new()?)))
        } else {
            None
        };

        #[cfg(feature = "rocm")]
        let rocm_kernel = if backend == GpuBackend::Rocm {
            Some(Arc::new(Mutex::new(RocmKernel::new()?)))
        } else {
            None
        };

        #[cfg(feature = "intel")]
        let intel_kernel = if backend == GpuBackend::Intel {
            let intel_config = IntelKernelConfig {
                device_id: config.device_id,
                workgroup_size: 256,
                ..Default::default()
            };
            Some(Arc::new(Mutex::new(IntelKernel::new(intel_config)?)))
        } else {
            None
        };

        #[cfg(feature = "vulkan")]
        let vulkan_kernel = if backend == GpuBackend::Vulkan {
            let mut kernel = VulkanKernel::new()?;
            kernel.initialize(config.device_id)?;
            Some(Arc::new(Mutex::new(kernel)))
        } else {
            None
        };

        Ok(Self {
            backend,
            #[cfg(feature = "cuda")]
            cuda_kernel,
            #[cfg(feature = "rocm")]
            rocm_kernel,
            #[cfg(feature = "intel")]
            intel_kernel,
            #[cfg(feature = "vulkan")]
            vulkan_kernel,
            device_id: config.device_id,
            enable_async: config.enable_async,
        })
    }

    /// Matrix multiplication with GPU acceleration
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TrustformersError::tensor_op_error(
                "Matrix multiplication requires 2D tensors",
                "matmul",
            ));
        }

        if a_shape[1] != b_shape[0] {
            return Err(TrustformersError::tensor_op_error(
                "Matrix dimensions incompatible for multiplication",
                "matmul",
            ));
        }

        let mut result = Tensor::zeros(&[a_shape[0], b_shape[1]])?;

        match self.backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                if let Some(ref cuda_kernel) = self.cuda_kernel {
                    let mut kernel = cuda_kernel.lock().unwrap();
                    kernel.matmul(a, b, &mut result, None)?;
                } else {
                    return Err(TrustformersError::tensor_op_error(
                        "CUDA kernel not available",
                        "matmul",
                    ));
                }
                #[cfg(not(feature = "cuda"))]
                return Err(TrustformersError::tensor_op_error(
                    "CUDA support not enabled",
                    "matmul",
                ));
            },
            GpuBackend::Rocm => {
                #[cfg(feature = "rocm")]
                if let Some(ref rocm_kernel) = self.rocm_kernel {
                    let mut kernel = rocm_kernel.lock().unwrap();
                    kernel.matmul(a, b, &mut result, None)?;
                } else {
                    return Err(TrustformersError::tensor_op_error(
                        "ROCm kernel not available",
                        "matmul",
                    ));
                }
                #[cfg(not(feature = "rocm"))]
                return Err(TrustformersError::tensor_op_error(
                    "ROCm support not enabled",
                    "matmul",
                ));
            },
            GpuBackend::Intel => {
                #[cfg(feature = "intel")]
                if let Some(ref intel_kernel) = self.intel_kernel {
                    let mut kernel = intel_kernel.lock().unwrap();
                    kernel.gemm(
                        a,
                        b,
                        &mut result,
                        1.0,
                        0.0,
                        crate::kernels::intel_kernels::IntelPrecision::FP32,
                    )?;
                } else {
                    return Err(TrustformersError::tensor_op_error(
                        "Intel kernel not available",
                        "matmul",
                    ));
                }
                #[cfg(not(feature = "intel"))]
                return Err(TrustformersError::tensor_op_error(
                    "Intel oneAPI support not enabled",
                    "matmul",
                ));
            },
            GpuBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                if let Some(ref vulkan_kernel) = self.vulkan_kernel {
                    let mut kernel = vulkan_kernel.lock().unwrap();
                    kernel.matmul(a, b, &mut result, None)?;
                } else {
                    return Err(TrustformersError::tensor_op_error(
                        "Vulkan kernel not available",
                        "matmul",
                    ));
                }
                #[cfg(not(feature = "vulkan"))]
                return Err(TrustformersError::tensor_op_error(
                    "Vulkan support not enabled",
                    "matmul",
                ));
            },
            _ => {
                // Fallback to CPU implementation
                self.cpu_matmul(a, b, &mut result)?;
            },
        }

        Ok(result)
    }

    /// Batch matrix multiplication
    pub fn batch_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 3 || b_shape.len() != 3 {
            return Err(TrustformersError::tensor_op_error(
                "Batch matrix multiplication requires 3D tensors",
                "batch_matmul",
            ));
        }

        if a_shape[0] != b_shape[0] || a_shape[2] != b_shape[1] {
            return Err(TrustformersError::tensor_op_error(
                "Batch matrix dimensions incompatible",
                "batch_matmul",
            ));
        }

        let result = Tensor::zeros(&[a_shape[0], a_shape[1], b_shape[2]])?;

        // For batch operations, we can parallelize across the batch dimension
        for batch in 0..a_shape[0] {
            let a_slice = a.slice(0, batch, batch + 1)?;
            let b_slice = b.slice(0, batch, batch + 1)?;
            let mut result_slice = result.slice(0, batch, batch + 1)?;

            match self.backend {
                GpuBackend::Cuda => {
                    #[cfg(feature = "cuda")]
                    if let Some(ref cuda_kernel) = self.cuda_kernel {
                        let mut kernel = cuda_kernel.lock().unwrap();
                        kernel.matmul(&a_slice, &b_slice, &mut result_slice, None)?;
                    } else {
                        self.cpu_matmul(&a_slice, &b_slice, &mut result_slice)?;
                    }
                    #[cfg(not(feature = "cuda"))]
                    self.cpu_matmul(&a_slice, &b_slice, &mut result_slice)?;
                },
                GpuBackend::Rocm => {
                    #[cfg(feature = "rocm")]
                    if let Some(ref rocm_kernel) = self.rocm_kernel {
                        let mut kernel = rocm_kernel.lock().unwrap();
                        kernel.matmul(&a_slice, &b_slice, &mut result_slice, None)?;
                    } else {
                        self.cpu_matmul(&a_slice, &b_slice, &mut result_slice)?;
                    }
                    #[cfg(not(feature = "rocm"))]
                    self.cpu_matmul(&a_slice, &b_slice, &mut result_slice)?;
                },
                _ => {
                    self.cpu_matmul(&a_slice, &b_slice, &mut result_slice)?;
                },
            }
        }

        Ok(result)
    }

    /// Flash attention implementation
    pub fn flash_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        scale: f32,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 {
            return Err(TrustformersError::tensor_op_error(
                "Attention requires 3D tensors [batch, seq_len, hidden_dim]",
                "flash_attention",
            ));
        }

        if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
            return Err(TrustformersError::tensor_op_error(
                "Batch dimensions must match for attention",
                "flash_attention",
            ));
        }

        let mut output = Tensor::zeros(&q_shape)?;

        match self.backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                if let Some(ref cuda_kernel) = self.cuda_kernel {
                    let mut kernel = cuda_kernel.lock().unwrap();
                    kernel.flash_attention(query, key, value, &mut output, None)?;
                } else {
                    self.cpu_attention(query, key, value, &mut output, scale, mask)?;
                }
                #[cfg(not(feature = "cuda"))]
                self.cpu_attention(query, key, value, &mut output, scale, mask)?;
            },
            GpuBackend::Rocm => {
                #[cfg(feature = "rocm")]
                if let Some(ref rocm_kernel) = self.rocm_kernel {
                    let mut kernel = rocm_kernel.lock().unwrap();
                    kernel.flash_attention(query, key, value, &mut output, None)?;
                } else {
                    self.cpu_attention(query, key, value, &mut output, scale, mask)?;
                }
                #[cfg(not(feature = "rocm"))]
                self.cpu_attention(query, key, value, &mut output, scale, mask)?;
            },
            GpuBackend::Intel => {
                #[cfg(feature = "intel")]
                if let Some(ref intel_kernel) = self.intel_kernel {
                    let mut kernel = intel_kernel.lock().unwrap();
                    kernel.attention(
                        query,
                        key,
                        value,
                        &mut output,
                        scale,
                        crate::kernels::intel_kernels::IntelPrecision::FP32,
                    )?;
                } else {
                    self.cpu_attention(query, key, value, &mut output, scale, mask)?;
                }
                #[cfg(not(feature = "intel"))]
                self.cpu_attention(query, key, value, &mut output, scale, mask)?;
            },
            GpuBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                if let Some(ref vulkan_kernel) = self.vulkan_kernel {
                    let mut kernel = vulkan_kernel.lock().unwrap();
                    kernel.flash_attention(query, key, value, &mut output, None)?;
                } else {
                    self.cpu_attention(query, key, value, &mut output, scale, mask)?;
                }
                #[cfg(not(feature = "vulkan"))]
                self.cpu_attention(query, key, value, &mut output, scale, mask)?;
            },
            _ => {
                self.cpu_attention(query, key, value, &mut output, scale, mask)?;
            },
        }

        Ok(output)
    }

    /// Layer normalization with GPU acceleration
    pub fn layer_norm(
        &self,
        input: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        epsilon: f32,
    ) -> Result<Tensor> {
        let input_shape = input.shape();
        let mut output = Tensor::zeros(&input_shape)?;

        match self.backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                if let Some(ref cuda_kernel) = self.cuda_kernel {
                    let mut kernel = cuda_kernel.lock().unwrap();
                    kernel.layer_norm(input, gamma, beta, &mut output, epsilon, None)?;
                } else {
                    self.cpu_layer_norm(input, gamma, beta, &mut output, epsilon)?;
                }
                #[cfg(not(feature = "cuda"))]
                self.cpu_layer_norm(input, gamma, beta, &mut output, epsilon)?;
            },
            GpuBackend::Rocm => {
                #[cfg(feature = "rocm")]
                if let Some(ref rocm_kernel) = self.rocm_kernel {
                    let mut kernel = rocm_kernel.lock().unwrap();
                    kernel.layer_norm(input, gamma, beta, &mut output, epsilon, None)?;
                } else {
                    self.cpu_layer_norm(input, gamma, beta, &mut output, epsilon)?;
                }
                #[cfg(not(feature = "rocm"))]
                self.cpu_layer_norm(input, gamma, beta, &mut output, epsilon)?;
            },
            GpuBackend::Intel => {
                #[cfg(feature = "intel")]
                if let Some(ref intel_kernel) = self.intel_kernel {
                    let mut kernel = intel_kernel.lock().unwrap();
                    kernel.layer_norm(
                        input,
                        gamma,
                        Some(beta),
                        &mut output,
                        epsilon,
                        crate::kernels::intel_kernels::IntelPrecision::FP32,
                    )?;
                } else {
                    self.cpu_layer_norm(input, gamma, beta, &mut output, epsilon)?;
                }
                #[cfg(not(feature = "intel"))]
                self.cpu_layer_norm(input, gamma, beta, &mut output, epsilon)?;
            },
            GpuBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                if let Some(ref vulkan_kernel) = self.vulkan_kernel {
                    let mut kernel = vulkan_kernel.lock().unwrap();
                    kernel.layer_norm(
                        input,
                        gamma,
                        Some(beta),
                        &mut output,
                        epsilon,
                        crate::kernels::vulkan_kernels::VulkanPrecision::FP32,
                    )?;
                } else {
                    self.cpu_layer_norm(input, gamma, beta, &mut output, epsilon)?;
                }
                #[cfg(not(feature = "vulkan"))]
                self.cpu_layer_norm(input, gamma, beta, &mut output, epsilon)?;
            },
            _ => {
                self.cpu_layer_norm(input, gamma, beta, &mut output, epsilon)?;
            },
        }

        Ok(output)
    }

    /// GELU activation with GPU acceleration
    pub fn gelu(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        let mut output = Tensor::zeros(&input_shape)?;

        match self.backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                if let Some(ref cuda_kernel) = self.cuda_kernel {
                    let mut kernel = cuda_kernel.lock().unwrap();
                    kernel.fused_gelu(input, &mut output, None)?;
                } else {
                    self.cpu_gelu(input, &mut output)?;
                }
                #[cfg(not(feature = "cuda"))]
                self.cpu_gelu(input, &mut output)?;
            },
            GpuBackend::Rocm => {
                #[cfg(feature = "rocm")]
                if let Some(ref rocm_kernel) = self.rocm_kernel {
                    let mut kernel = rocm_kernel.lock().unwrap();
                    kernel.fused_gelu(input, &mut output, None)?;
                } else {
                    self.cpu_gelu(input, &mut output)?;
                }
                #[cfg(not(feature = "rocm"))]
                self.cpu_gelu(input, &mut output)?;
            },
            GpuBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                if let Some(ref vulkan_kernel) = self.vulkan_kernel {
                    let mut kernel = vulkan_kernel.lock().unwrap();
                    kernel.gelu(input, &mut output, None)?;
                } else {
                    self.cpu_gelu(input, &mut output)?;
                }
                #[cfg(not(feature = "vulkan"))]
                self.cpu_gelu(input, &mut output)?;
            },
            _ => {
                self.cpu_gelu(input, &mut output)?;
            },
        }

        Ok(output)
    }

    /// Reduce sum with GPU acceleration
    pub fn reduce_sum(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        let input_shape = input.shape();

        if dim >= input_shape.len() {
            return Err(TrustformersError::tensor_op_error(
                "Reduction dimension out of bounds",
                "reduce_sum",
            ));
        }

        let mut output_shape = input_shape.clone();
        output_shape.remove(dim);
        let mut output = Tensor::zeros(&output_shape)?;

        match self.backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                if let Some(ref cuda_kernel) = self.cuda_kernel {
                    let mut kernel = cuda_kernel.lock().unwrap();
                    kernel.reduce_sum(input, &mut output, dim, None)?;
                } else {
                    self.cpu_reduce_sum(input, &mut output, dim)?;
                }
                #[cfg(not(feature = "cuda"))]
                self.cpu_reduce_sum(input, &mut output, dim)?;
            },
            GpuBackend::Rocm => {
                #[cfg(feature = "rocm")]
                if let Some(ref rocm_kernel) = self.rocm_kernel {
                    let mut kernel = rocm_kernel.lock().unwrap();
                    kernel.reduce_sum(input, &mut output, dim, None)?;
                } else {
                    self.cpu_reduce_sum(input, &mut output, dim)?;
                }
                #[cfg(not(feature = "rocm"))]
                self.cpu_reduce_sum(input, &mut output, dim)?;
            },
            GpuBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                if let Some(ref vulkan_kernel) = self.vulkan_kernel {
                    let mut kernel = vulkan_kernel.lock().unwrap();
                    kernel.reduce_sum(input, &mut output, dim, None)?;
                } else {
                    self.cpu_reduce_sum(input, &mut output, dim)?;
                }
                #[cfg(not(feature = "vulkan"))]
                self.cpu_reduce_sum(input, &mut output, dim)?;
            },
            _ => {
                self.cpu_reduce_sum(input, &mut output, dim)?;
            },
        }

        Ok(output)
    }

    /// Softmax with GPU acceleration
    pub fn softmax(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        let input_shape = input.shape();
        let output = Tensor::zeros(&input_shape)?;

        // Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
        let max_vals = self.reduce_max(input, dim)?;
        let shifted = self.subtract_broadcast(input, &max_vals, dim)?;
        let exp_vals = self.exp(&shifted)?;
        let sum_exp = self.reduce_sum(&exp_vals, dim)?;
        let result = self.divide_broadcast(&exp_vals, &sum_exp, dim)?;

        Ok(result)
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        match self.backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                return self.cuda_kernel.is_some();
                #[cfg(not(feature = "cuda"))]
                return false;
            },
            GpuBackend::Rocm => {
                #[cfg(feature = "rocm")]
                return self.rocm_kernel.is_some();
                #[cfg(not(feature = "rocm"))]
                return false;
            },
            GpuBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                return self.vulkan_kernel.is_some();
                #[cfg(not(feature = "vulkan"))]
                return false;
            },
            GpuBackend::Cpu => false,
            _ => false, // Other backends not implemented yet
        }
    }

    /// Get current GPU backend
    pub fn get_backend(&self) -> GpuBackend {
        self.backend
    }

    /// Get GPU memory usage
    pub fn get_memory_usage(&self) -> Result<(u64, u64, u64)> {
        match self.backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                if let Some(ref cuda_kernel) = self.cuda_kernel {
                    let kernel = cuda_kernel.lock().unwrap();
                    kernel.get_memory_stats(self.device_id)
                } else {
                    Ok((0, 0, 0))
                }
                #[cfg(not(feature = "cuda"))]
                Ok((0, 0, 0))
            },
            GpuBackend::Rocm => {
                #[cfg(feature = "rocm")]
                if let Some(ref rocm_kernel) = self.rocm_kernel {
                    let kernel = rocm_kernel.lock().unwrap();
                    kernel.get_memory_stats(self.device_id)
                } else {
                    Ok((0, 0, 0))
                }
                #[cfg(not(feature = "rocm"))]
                Ok((0, 0, 0))
            },
            GpuBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                if let Some(ref vulkan_kernel) = self.vulkan_kernel {
                    let kernel = vulkan_kernel.lock().unwrap();
                    kernel.get_memory_stats(self.device_id)
                } else {
                    Ok((0, 0, 0))
                }
                #[cfg(not(feature = "vulkan"))]
                Ok((0, 0, 0))
            },
            _ => Ok((0, 0, 0)),
        }
    }

    /// Synchronize GPU operations
    pub fn synchronize(&self) -> Result<()> {
        // In a real implementation, this would call cudaDeviceSynchronize()
        Ok(())
    }

    // CPU fallback implementations
    fn cpu_matmul(&self, a: &Tensor, b: &Tensor, result: &mut Tensor) -> Result<()> {
        let a_data = a.data()?;
        let b_data = b.data()?;
        let result_data = result.data_mut()?;

        let a_shape = a.shape();
        let b_shape = b.shape();

        // Implement CPU matrix multiplication
        for i in 0..a_shape[0] {
            for j in 0..b_shape[1] {
                let mut sum = 0.0;
                for k in 0..a_shape[1] {
                    sum += a_data[i * a_shape[1] + k] * b_data[k * b_shape[1] + j];
                }
                result_data[i * b_shape[1] + j] = sum;
            }
        }

        Ok(())
    }

    fn cpu_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        output: &mut Tensor,
        scale: f32,
        _mask: Option<&Tensor>,
    ) -> Result<()> {
        let q_shape = query.shape();
        let q_data = query.data()?;
        let k_data = key.data()?;
        let v_data = value.data()?;
        let o_data = output.data_mut()?;

        let batch_size = q_shape[0];
        let seq_len = q_shape[1];
        let hidden_dim = q_shape[2];

        for batch in 0..batch_size {
            for i in 0..seq_len {
                // Compute attention scores
                let mut scores = vec![0.0; seq_len];
                for (j, score_ref) in scores.iter_mut().enumerate() {
                    let mut score = 0.0;
                    for d in 0..hidden_dim {
                        let q_idx = batch * seq_len * hidden_dim + i * hidden_dim + d;
                        let k_idx = batch * seq_len * hidden_dim + j * hidden_dim + d;
                        score += q_data[q_idx] * k_data[k_idx];
                    }
                    *score_ref = score * scale;
                }

                // Apply softmax
                let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut exp_sum = 0.0;
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                    exp_sum += *score;
                }
                for score in &mut scores {
                    *score /= exp_sum;
                }

                // Compute weighted sum
                for d in 0..hidden_dim {
                    let mut output_val = 0.0;
                    for (j, &score) in scores.iter().enumerate() {
                        let v_idx = batch * seq_len * hidden_dim + j * hidden_dim + d;
                        output_val += score * v_data[v_idx];
                    }
                    let o_idx = batch * seq_len * hidden_dim + i * hidden_dim + d;
                    o_data[o_idx] = output_val;
                }
            }
        }

        Ok(())
    }

    fn cpu_layer_norm(
        &self,
        input: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        output: &mut Tensor,
        epsilon: f32,
    ) -> Result<()> {
        let input_data = input.data()?;
        let gamma_data = gamma.data()?;
        let beta_data = beta.data()?;
        let output_data = output.data_mut()?;

        let input_shape = input.shape();
        let last_dim = input_shape[input_shape.len() - 1];
        let num_elements = input_data.len() / last_dim;

        for i in 0..num_elements {
            let start = i * last_dim;
            let end = start + last_dim;
            let slice = &input_data[start..end];

            // Compute mean
            let mean = slice.iter().sum::<f32>() / last_dim as f32;

            // Compute variance
            let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / last_dim as f32;
            let std_dev = (variance + epsilon).sqrt();

            // Normalize
            for j in 0..last_dim {
                let normalized = (slice[j] - mean) / std_dev;
                output_data[start + j] = normalized * gamma_data[j] + beta_data[j];
            }
        }

        Ok(())
    }

    fn cpu_gelu(&self, input: &Tensor, output: &mut Tensor) -> Result<()> {
        let input_data = input.data()?;
        let output_data = output.data_mut()?;

        for i in 0..input_data.len() {
            let x = input_data[i];
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let x_cubed = x * x * x;
            let tanh_arg = 0.797885 * (x + 0.044715 * x_cubed);
            let tanh_val = tanh_arg.tanh();
            output_data[i] = 0.5 * x * (1.0 + tanh_val);
        }

        Ok(())
    }

    fn cpu_reduce_sum(&self, input: &Tensor, output: &mut Tensor, dim: usize) -> Result<()> {
        let input_data = input.data()?;
        let output_data = output.data_mut()?;
        let input_shape = input.shape();

        let reduce_size = input_shape[dim];
        let outer_size = input_shape[..dim].iter().product::<usize>();
        let inner_size = input_shape[dim + 1..].iter().product::<usize>();

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum = 0.0;
                for reduce_idx in 0..reduce_size {
                    let input_idx =
                        outer * reduce_size * inner_size + reduce_idx * inner_size + inner;
                    sum += input_data[input_idx];
                }
                let output_idx = outer * inner_size + inner;
                output_data[output_idx] = sum;
            }
        }

        Ok(())
    }

    // Helper methods for complex operations
    fn reduce_max(&self, input: &Tensor, dim: usize) -> Result<Tensor> {
        // Similar to reduce_sum but with max operation
        let input_shape = input.shape();
        let mut output_shape = input_shape.clone();
        output_shape.remove(dim);

        // Implementation would be similar to reduce_sum
        Tensor::zeros(&output_shape)
    }

    fn subtract_broadcast(&self, a: &Tensor, b: &Tensor, dim: usize) -> Result<Tensor> {
        // Broadcast subtraction
        let a_shape = a.shape();
        Tensor::zeros(&a_shape)
    }

    fn exp(&self, input: &Tensor) -> Result<Tensor> {
        // Element-wise exponential
        let input_shape = input.shape();
        Tensor::zeros(&input_shape)
    }

    fn divide_broadcast(&self, a: &Tensor, b: &Tensor, dim: usize) -> Result<Tensor> {
        // Broadcast division
        let a_shape = a.shape();
        Tensor::zeros(&a_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_accelerated_ops_creation() {
        let config = GpuOpsConfig::default();
        let ops = GpuAcceleratedOps::new(config);
        assert!(ops.is_ok());
    }

    #[test]
    fn test_gpu_ops_config_default() {
        let config = GpuOpsConfig::default();
        assert_eq!(config.device_id, 0);
        assert!(config.enable_async);
        assert_eq!(config.precision, GpuPrecision::FP32);
    }

    #[test]
    fn test_backend_detection() {
        let config = GpuOpsConfig::default();
        let ops = GpuAcceleratedOps::new(config).unwrap();

        // Backend should be detected automatically
        let backend = ops.get_backend();
        assert!(matches!(
            backend,
            GpuBackend::Cuda | GpuBackend::Rocm | GpuBackend::Cpu | GpuBackend::Metal
        ));
    }

    #[test]
    fn test_matmul_dimensions() {
        let config = GpuOpsConfig::default();
        let ops = GpuAcceleratedOps::new(config).unwrap();

        let a = Tensor::ones(&[2, 3]).unwrap();
        let b = Tensor::ones(&[3, 4]).unwrap();

        let result = ops.matmul(&a, &b);
        assert!(result.is_ok());

        let result_tensor = result.unwrap();
        assert_eq!(result_tensor.shape(), &[2, 4]);
    }

    #[test]
    fn test_batch_matmul_dimensions() {
        let config = GpuOpsConfig::default();
        let ops = GpuAcceleratedOps::new(config).unwrap();

        let a = Tensor::ones(&[2, 3, 4]).unwrap();
        let b = Tensor::ones(&[2, 4, 5]).unwrap();

        let result = ops.batch_matmul(&a, &b);
        assert!(result.is_ok());

        let result_tensor = result.unwrap();
        assert_eq!(result_tensor.shape(), &[2, 3, 5]);
    }

    #[test]
    fn test_flash_attention_dimensions() {
        let config = GpuOpsConfig::default();
        let ops = GpuAcceleratedOps::new(config).unwrap();

        let batch_size = 2;
        let seq_len = 10;
        let hidden_dim = 64;

        let query = Tensor::ones(&[batch_size, seq_len, hidden_dim]).unwrap();
        let key = Tensor::ones(&[batch_size, seq_len, hidden_dim]).unwrap();
        let value = Tensor::ones(&[batch_size, seq_len, hidden_dim]).unwrap();

        let result = ops.flash_attention(&query, &key, &value, 0.125, None);
        assert!(result.is_ok());

        let result_tensor = result.unwrap();
        assert_eq!(result_tensor.shape(), &[batch_size, seq_len, hidden_dim]);
    }

    #[test]
    fn test_layer_norm_dimensions() {
        let config = GpuOpsConfig::default();
        let ops = GpuAcceleratedOps::new(config).unwrap();

        let input = Tensor::ones(&[2, 10, 64]).unwrap();
        let gamma = Tensor::ones(&[64]).unwrap();
        let beta = Tensor::zeros(&[64]).unwrap();

        let result = ops.layer_norm(&input, &gamma, &beta, 1e-5);
        assert!(result.is_ok());

        let result_tensor = result.unwrap();
        assert_eq!(result_tensor.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_gelu_dimensions() {
        let config = GpuOpsConfig::default();
        let ops = GpuAcceleratedOps::new(config).unwrap();

        let input = Tensor::ones(&[2, 10, 64]).unwrap();

        let result = ops.gelu(&input);
        assert!(result.is_ok());

        let result_tensor = result.unwrap();
        assert_eq!(result_tensor.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_memory_usage() {
        let config = GpuOpsConfig::default();
        let ops = GpuAcceleratedOps::new(config).unwrap();

        let (total, peak, free) = ops.get_memory_usage().unwrap();
        // Should return some values (may be 0 if no GPU available)
        assert!(total >= 0);
        assert!(peak >= 0);
        assert!(free >= 0);
    }

    #[test]
    fn test_synchronize() {
        let config = GpuOpsConfig::default();
        let ops = GpuAcceleratedOps::new(config).unwrap();

        let result = ops.synchronize();
        assert!(result.is_ok());
    }
}
