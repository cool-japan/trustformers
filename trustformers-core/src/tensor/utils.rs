//! Tensor utility functions.
//!
//! This module contains utility functions for working with tensors.

use super::{DType, Tensor};
use crate::errors::{Result, TrustformersError};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

/// Global counter for unique tensor IDs
#[allow(dead_code)] // Reserved for future tensor tracking features
static TENSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

lazy_static::lazy_static! {
    /// Global gradient registry for tracking tensor gradients
    static ref GRADIENT_REGISTRY: Arc<RwLock<HashMap<u64, Tensor>>> = Arc::new(RwLock::new(HashMap::new()));
}

thread_local! {
    /// Thread-local gradient mode flag
    static GRADIENT_MODE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Enable gradient tracking for current thread
pub fn enable_grad() {
    GRADIENT_MODE.with(|mode| mode.set(true));
}

/// Disable gradient tracking for current thread
pub fn disable_grad() {
    GRADIENT_MODE.with(|mode| mode.set(false));
}

/// Check if gradient tracking is enabled for current thread
pub fn is_grad_enabled() -> bool {
    GRADIENT_MODE.with(|mode| mode.get())
}

/// Clear all gradients from the registry
pub fn clear_gradients() {
    if let Ok(mut registry) = GRADIENT_REGISTRY.write() {
        registry.clear();
    }
}

impl Tensor {
    /// Get a unique identifier for this tensor instance
    fn tensor_id(&self) -> u64 {
        // Generate a hash-based ID from tensor data and shape
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.shape().hash(&mut hasher);

        match self {
            Tensor::F32(arr) => {
                arr.as_ptr().hash(&mut hasher);
                arr.len().hash(&mut hasher);
            },
            Tensor::F64(arr) => {
                arr.as_ptr().hash(&mut hasher);
                arr.len().hash(&mut hasher);
            },
            Tensor::I64(arr) => {
                arr.as_ptr().hash(&mut hasher);
                arr.len().hash(&mut hasher);
            },
            #[cfg(feature = "metal")]
            Tensor::Metal(data) => {
                // Use buffer_id as a unique identifier for Metal tensors
                data.buffer_id.hash(&mut hasher);
                self.len().hash(&mut hasher);
            },
            #[cfg(feature = "cuda")]
            Tensor::CUDA(data) => {
                // Use buffer_id as a unique identifier for CUDA tensors
                data.buffer_id.hash(&mut hasher);
                self.len().hash(&mut hasher);
            },
            _ => {
                // For other tensor types, use a simpler approach
                self.len().hash(&mut hasher);
            },
        }

        hasher.finish()
    }
    /// Get the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A vector containing the dimensions of the tensor.
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Tensor::F32(a) => a.shape().to_vec(),
            Tensor::F64(a) => a.shape().to_vec(),
            Tensor::F16(a) => a.shape().to_vec(),
            Tensor::BF16(a) => a.shape().to_vec(),
            Tensor::I64(a) => a.shape().to_vec(),
            Tensor::C32(a) => a.shape().to_vec(),
            Tensor::C64(a) => a.shape().to_vec(),
            Tensor::CF16(a) => a.shape().to_vec(),
            Tensor::CBF16(a) => a.shape().to_vec(),
            Tensor::Sparse(s) => s.shape().to_vec(),
            #[cfg(feature = "torch")]
            Tensor::Torch(t) => t.size().iter().map(|&d| d as usize).collect(),
            #[cfg(feature = "candle")]
            Tensor::Candle(t) => t.shape().dims().to_vec(),
            #[cfg(feature = "metal")]
            Tensor::Metal(data) => data.shape.clone(),
            #[cfg(feature = "cuda")]
            Tensor::CUDA(data) => data.shape.clone(),
        }
    }

    /// Get the number of elements in the tensor.
    ///
    /// # Returns
    ///
    /// The total number of elements in the tensor.
    pub fn len(&self) -> usize {
        match self {
            Tensor::F32(a) => a.len(),
            Tensor::F64(a) => a.len(),
            Tensor::F16(a) => a.len(),
            Tensor::BF16(a) => a.len(),
            Tensor::I64(a) => a.len(),
            Tensor::C32(a) => a.len(),
            Tensor::C64(a) => a.len(),
            Tensor::CF16(a) => a.len(),
            Tensor::CBF16(a) => a.len(),
            Tensor::Sparse(s) => s.nnz(), // Non-zero elements for sparse tensors
            #[cfg(feature = "torch")]
            Tensor::Torch(t) => t.numel(),
            #[cfg(feature = "candle")]
            Tensor::Candle(t) => t.elem_count(),
            #[cfg(feature = "metal")]
            Tensor::Metal(data) => data.shape.iter().product(),
            #[cfg(feature = "cuda")]
            Tensor::CUDA(data) => data.shape.iter().product(),
        }
    }

    /// Check if the tensor is empty.
    ///
    /// # Returns
    ///
    /// True if the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of dimensions in the tensor.
    ///
    /// # Returns
    ///
    /// The number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Get the size in bytes of the tensor.
    ///
    /// # Returns
    ///
    /// The size in bytes of the tensor data.
    pub fn size_bytes(&self) -> usize {
        match self {
            Tensor::F32(a) => a.len() * std::mem::size_of::<f32>(),
            Tensor::F64(a) => a.len() * std::mem::size_of::<f64>(),
            Tensor::F16(a) => a.len() * std::mem::size_of::<half::f16>(),
            Tensor::BF16(a) => a.len() * std::mem::size_of::<half::bf16>(),
            Tensor::I64(a) => a.len() * std::mem::size_of::<i64>(),
            Tensor::C32(a) => a.len() * std::mem::size_of::<scirs2_core::Complex32>(),
            Tensor::C64(a) => a.len() * std::mem::size_of::<scirs2_core::Complex64>(),
            Tensor::CF16(a) => a.len() * std::mem::size_of::<scirs2_core::Complex<half::f16>>(),
            Tensor::CBF16(a) => a.len() * std::mem::size_of::<scirs2_core::Complex<half::bf16>>(),
            Tensor::Sparse(s) => s.nnz() * std::mem::size_of::<f32>(), // Simplified estimate
            #[cfg(feature = "torch")]
            Tensor::Torch(t) => t.numel() * std::mem::size_of::<f32>(), // Simplified
            #[cfg(feature = "candle")]
            Tensor::Candle(t) => t.elem_count() * std::mem::size_of::<f32>(), // Simplified
            #[cfg(feature = "metal")]
            Tensor::Metal(data) => {
                let num_elements: usize = data.shape.iter().product();
                num_elements * data.dtype.size_in_bytes()
            },
            #[cfg(feature = "cuda")]
            Tensor::CUDA(data) => {
                let num_elements: usize = data.shape.iter().product();
                num_elements * data.dtype.size_in_bytes()
            },
        }
    }

    /// Transfer tensor to specified device.
    ///
    /// # Arguments
    ///
    /// * `device` - Device identifier (e.g., "cpu", "cuda:0", "mps", "tpu:0")
    ///
    /// # Returns
    ///
    /// A tensor on the specified device (currently CPU-only with validation).
    pub fn to_device(&self, device: &str) -> Result<Tensor> {
        // Validate device string format and provide helpful error messages
        let device_lower = device.to_lowercase();

        // Parse device components
        let (device_type, device_index) = if device_lower.contains(':') {
            let parts: Vec<&str> = device_lower.split(':').collect();
            if parts.len() != 2 {
                return Err(TrustformersError::tensor_op_error(
                    &format!("Invalid device format '{}'. Expected format: 'device_type' or 'device_type:index'", device),
                    "to_device"
                ));
            }

            let index = parts[1].parse::<usize>().map_err(|_| {
                TrustformersError::tensor_op_error(
                    &format!(
                        "Invalid device index '{}'. Expected a non-negative integer",
                        parts[1]
                    ),
                    "to_device",
                )
            })?;

            (parts[0], Some(index))
        } else {
            (device_lower.as_str(), None)
        };

        // Validate supported device types
        match device_type {
            "cpu" => {
                // CPU is always supported
                if let Some(index) = device_index {
                    if index > 0 {
                        return Err(TrustformersError::tensor_op_error(
                            &format!("CPU device index {} not supported. CPU only supports index 0 or no index", index),
                            "to_device"
                        ));
                    }
                }
                // Return a clone for CPU (no actual transfer needed)
                Ok(self.clone())
            },
            "cuda" => {
                // CUDA support would require additional backend integration
                if let Some(index) = device_index {
                    Err(TrustformersError::tensor_op_error(
                        &format!("CUDA device cuda:{} not available. This build doesn't support CUDA. Consider using CPU instead with device='cpu'", index),
                        "to_device"
                    ))
                } else {
                    Err(TrustformersError::tensor_op_error(
                        "CUDA devices not available. This build doesn't support CUDA. Consider using CPU instead with device='cpu'",
                        "to_device"
                    ))
                }
            },
            "mps" => {
                // Metal Performance Shaders (Apple Silicon)
                Err(TrustformersError::tensor_op_error(
                    "MPS device not available. This build doesn't support Metal Performance Shaders. Consider using CPU instead with device='cpu'",
                    "to_device"
                ))
            },
            "tpu" => {
                // Tensor Processing Unit
                Err(TrustformersError::tensor_op_error(
                    "TPU devices not available. This build doesn't support TPU. Consider using CPU instead with device='cpu'",
                    "to_device"
                ))
            },
            "xpu" | "intel" => {
                // Intel XPU (Intel GPU/AI accelerators)
                Err(TrustformersError::tensor_op_error(
                    "Intel XPU devices not available. This build doesn't support Intel XPU. Consider using CPU instead with device='cpu'",
                    "to_device"
                ))
            },
            "npu" => {
                // Neural Processing Unit
                Err(TrustformersError::tensor_op_error(
                    "NPU devices not available. This build doesn't support NPU. Consider using CPU instead with device='cpu'",
                    "to_device"
                ))
            },
            _ => {
                Err(TrustformersError::tensor_op_error(
                    &format!("Unknown device type '{}'. Supported device types: cpu, cuda, mps, tpu, xpu, npu. For this build, only 'cpu' is supported", device_type),
                    "to_device"
                ))
            },
        }
    }

    /// Transfer tensor to specified device using Device enum.
    ///
    /// This is the preferred method for device transfers in modern code.
    /// It supports Metal GPU acceleration and provides better type safety.
    ///
    /// # Arguments
    ///
    /// * `device` - Device enum (Device::CPU, Device::Metal(0), etc.)
    ///
    /// # Returns
    ///
    /// A tensor on the specified device.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    /// use trustformers_core::device::Device;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let cpu_tensor = Tensor::randn(&[2, 3])?;
    ///
    /// // Transfer to Metal GPU
    /// let gpu_tensor = cpu_tensor.to_device_enum(&Device::Metal(0))?;
    ///
    /// // Transfer back to CPU
    /// let result = gpu_tensor.to_device_enum(&Device::CPU)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_device_enum(&self, device: &crate::device::Device) -> Result<Tensor> {
        match (self, device) {
            // F32 → Metal
            #[cfg(feature = "metal")]
            (Tensor::F32(arr), crate::device::Device::Metal(_)) => {
                use crate::gpu_ops::metal::get_metal_backend;
                let backend = get_metal_backend()?;
                let data_vec: Vec<f32> = arr.iter().copied().collect();

                #[cfg(debug_assertions)]
                {
                    // Debug: Verify data_vec before GPU upload (only in debug builds)
                    eprintln!(
                        "🔍 to_device_enum(F32→Metal): data_vec.len()={}",
                        data_vec.len()
                    );
                    if !data_vec.is_empty() {
                        eprintln!(
                            "🔍 to_device_enum: first 10 values: {:?}",
                            &data_vec[..10.min(data_vec.len())]
                        );
                        eprintln!(
                            "🔍 to_device_enum: stats - min={:.4}, max={:.4}, mean={:.4}",
                            data_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                            data_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                            data_vec.iter().sum::<f32>() / data_vec.len() as f32
                        );
                    }
                }

                let buffer_id = backend.create_persistent_buffer(&data_vec)?;

                #[cfg(debug_assertions)]
                {
                    eprintln!("🔍 to_device_enum: Created buffer_id={:?}", buffer_id);

                    // Verify by immediately downloading (GPU→CPU transfer - expensive!)
                    let verify_data = backend.download_buffer_to_vec(&buffer_id)?;
                    eprintln!(
                        "🔍 to_device_enum: Verification download - len={}, first 10: {:?}",
                        verify_data.len(),
                        &verify_data[..10.min(verify_data.len())]
                    );
                }

                Ok(Tensor::Metal(super::MetalTensorData {
                    buffer_id,
                    shape: arr.shape().to_vec(),
                    dtype: DType::F32,
                }))
            },

            // F64 → Metal (convert to F32 first)
            #[cfg(feature = "metal")]
            (Tensor::F64(arr), crate::device::Device::Metal(_)) => {
                use crate::gpu_ops::metal::get_metal_backend;
                let backend = get_metal_backend()?;
                let data_vec: Vec<f32> = arr.iter().map(|&x| x as f32).collect();
                let buffer_id = backend.create_persistent_buffer(&data_vec)?;
                Ok(Tensor::Metal(super::MetalTensorData {
                    buffer_id,
                    shape: arr.shape().to_vec(),
                    dtype: DType::F32,
                }))
            },

            // Metal → F32
            #[cfg(feature = "metal")]
            (Tensor::Metal(metal_data), crate::device::Device::CPU) => {
                use crate::gpu_ops::metal::get_metal_backend;
                let backend = get_metal_backend()?;
                let buffer = backend.get_persistent_buffer(&metal_data.buffer_id)?;

                // Download from GPU
                let size: usize = metal_data.shape.iter().product();

                // Handle different dtypes
                match metal_data.dtype {
                    DType::F32 => {
                        let ptr = buffer.contents() as *const f32;
                        let data_vec = unsafe { std::slice::from_raw_parts(ptr, size) }.to_vec();

                        // Convert to ArrayD
                        use scirs2_core::ndarray::ArrayD;
                        let arr = ArrayD::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(&metal_data.shape),
                            data_vec,
                        )
                        .map_err(|e| {
                            TrustformersError::tensor_op_error(
                                &format!("Failed to create array from shape: {}", e),
                                "to_device_enum",
                            )
                        })?;
                        Ok(Tensor::F32(arr))
                    },
                    _ => Err(TrustformersError::tensor_op_error(
                        &format!("Unsupported Metal tensor dtype: {:?}", metal_data.dtype),
                        "to_device_enum",
                    )),
                }
            },

            // Metal → Metal (different device, currently just clone)
            #[cfg(feature = "metal")]
            (Tensor::Metal(metal_data), crate::device::Device::Metal(_)) => {
                // For now, just return a clone (buffer is reference counted)
                // TODO: Implement actual device-to-device transfer if needed
                Ok(Tensor::Metal(metal_data.clone()))
            },

            // Already on correct device - no-op
            (Tensor::F32(_), crate::device::Device::CPU) => Ok(self.clone()),
            (Tensor::F64(_), crate::device::Device::CPU) => Ok(self.clone()),
            (Tensor::F16(_), crate::device::Device::CPU) => Ok(self.clone()),
            (Tensor::BF16(_), crate::device::Device::CPU) => Ok(self.clone()),
            (Tensor::I64(_), crate::device::Device::CPU) => Ok(self.clone()),
            (Tensor::C32(_), crate::device::Device::CPU) => Ok(self.clone()),
            (Tensor::C64(_), crate::device::Device::CPU) => Ok(self.clone()),
            (Tensor::CF16(_), crate::device::Device::CPU) => Ok(self.clone()),
            (Tensor::CBF16(_), crate::device::Device::CPU) => Ok(self.clone()),
            (Tensor::Sparse(_), crate::device::Device::CPU) => Ok(self.clone()),

            // Metal not available in this build
            #[cfg(not(feature = "metal"))]
            (_, crate::device::Device::Metal(_)) => Err(TrustformersError::hardware_error(
                "Metal not available. Compile with --features metal",
                "to_device_enum",
            )),

            // F32 → CUDA
            #[cfg(feature = "cuda")]
            (Tensor::F32(_arr), crate::device::Device::CUDA(_device_id)) => {
                #[cfg(any(target_os = "linux", target_os = "windows"))]
                {
                    use crate::gpu_ops::cuda::get_cuda_backend;
                    let backend = get_cuda_backend(*device_id)?;
                    let data_vec: Vec<f32> = arr.iter().copied().collect();
                    let buffer_id = backend.create_persistent_buffer(&data_vec)?;
                    Ok(Tensor::CUDA(super::CudaTensorData {
                        buffer_id,
                        shape: arr.shape().to_vec(),
                        dtype: DType::F32,
                    }))
                }
                #[cfg(not(any(target_os = "linux", target_os = "windows")))]
                {
                    Err(TrustformersError::hardware_error(
                        "CUDA is only supported on Linux and Windows",
                        "to_device_enum",
                    ))
                }
            },

            // F64 → CUDA (convert to F32 first)
            #[cfg(feature = "cuda")]
            (Tensor::F64(_arr), crate::device::Device::CUDA(_device_id)) => {
                #[cfg(any(target_os = "linux", target_os = "windows"))]
                {
                    use crate::gpu_ops::cuda::get_cuda_backend;
                    let backend = get_cuda_backend(*device_id)?;
                    let data_vec: Vec<f32> = arr.iter().map(|&x| x as f32).collect();
                    let buffer_id = backend.create_persistent_buffer(&data_vec)?;
                    Ok(Tensor::CUDA(super::CudaTensorData {
                        buffer_id,
                        shape: arr.shape().to_vec(),
                        dtype: DType::F32,
                    }))
                }
                #[cfg(not(any(target_os = "linux", target_os = "windows")))]
                {
                    Err(TrustformersError::hardware_error(
                        "CUDA is only supported on Linux and Windows",
                        "to_device_enum",
                    ))
                }
            },

            // CUDA → F32
            #[cfg(feature = "cuda")]
            (Tensor::CUDA(_cuda_data), crate::device::Device::CPU) => {
                #[cfg(any(target_os = "linux", target_os = "windows"))]
                {
                    use crate::gpu_ops::cuda::get_cuda_backend;
                    // Use device 0 by default for downloading
                    let backend = get_cuda_backend(0)?;

                    // Handle different dtypes
                    match cuda_data.dtype {
                        DType::F32 => {
                            // Download from GPU to CPU
                            let data_vec = backend.download_buffer(&cuda_data.buffer_id)?;

                            // Convert to ArrayD
                            use scirs2_core::ndarray::ArrayD;
                            let arr = ArrayD::from_shape_vec(
                                scirs2_core::ndarray::IxDyn(&cuda_data.shape),
                                data_vec,
                            )
                            .map_err(|e| {
                                TrustformersError::tensor_op_error(
                                    &format!("Failed to create array from shape: {}", e),
                                    "to_device_enum",
                                )
                            })?;
                            Ok(Tensor::F32(arr))
                        },
                        _ => Err(TrustformersError::tensor_op_error(
                            &format!("Unsupported CUDA tensor dtype: {:?}", cuda_data.dtype),
                            "to_device_enum",
                        )),
                    }
                }
                #[cfg(not(any(target_os = "linux", target_os = "windows")))]
                {
                    Err(TrustformersError::hardware_error(
                        "CUDA is only supported on Linux and Windows",
                        "to_device_enum",
                    ))
                }
            },

            // CUDA → CUDA (different device, currently just clone)
            #[cfg(feature = "cuda")]
            (Tensor::CUDA(cuda_data), crate::device::Device::CUDA(_)) => {
                // For now, just return a clone (buffer is reference counted)
                // TODO: Implement actual device-to-device transfer if needed
                Ok(Tensor::CUDA(cuda_data.clone()))
            },

            // CUDA not available in this build
            #[cfg(not(feature = "cuda"))]
            (_, crate::device::Device::CUDA(_)) => Err(TrustformersError::hardware_error(
                "CUDA not available. Compile with --features cuda",
                "to_device_enum",
            )),

            // ROCm transfers (placeholder for future implementation)
            (_, crate::device::Device::ROCm(_)) => Err(TrustformersError::hardware_error(
                "ROCm transfer not implemented yet",
                "to_device_enum",
            )),

            // WebGPU transfers (placeholder for future implementation)
            (_, crate::device::Device::WebGPU) => Err(TrustformersError::hardware_error(
                "WebGPU transfer not implemented yet",
                "to_device_enum",
            )),

            // Fallback for unsupported combinations
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::tensor_op_error(
                &format!(
                    "Unsupported device transfer from {:?} to {:?}",
                    self.dtype(),
                    device
                ),
                "to_device_enum",
            )),
        }
    }

    /// Get gradient tensor.
    ///
    /// # Returns
    ///
    /// Returns the gradient tensor associated with this tensor, or None if no gradient exists.
    /// Gradients are only tracked when gradient mode is enabled via `enable_grad()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::{Tensor, utils::{enable_grad, disable_grad}};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// enable_grad();
    /// let x = Tensor::randn(&[2, 3])?;
    /// // After some computation that requires gradients...
    /// if let Ok(grad_tensor) = x.grad() {
    ///     println!("Gradient: {:?}", grad_tensor.shape());
    /// }
    /// disable_grad();
    /// # Ok(())
    /// # }
    /// ```
    pub fn grad(&self) -> Result<Tensor> {
        if !is_grad_enabled() {
            return Err(TrustformersError::tensor_op_error(
                "Gradient tracking is not enabled. Use enable_grad() to enable gradient tracking.",
                "grad",
            ));
        }

        let tensor_id = self.tensor_id();

        if let Ok(registry) = GRADIENT_REGISTRY.read() {
            if let Some(grad_tensor) = registry.get(&tensor_id) {
                Ok(grad_tensor.clone())
            } else {
                Err(TrustformersError::tensor_op_error(
                    "No gradient found for this tensor. Gradients are set during backward pass.",
                    "grad",
                ))
            }
        } else {
            Err(TrustformersError::tensor_op_error(
                "Failed to access gradient registry.",
                "grad",
            ))
        }
    }

    /// Set gradient tensor.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor to set for this tensor
    ///
    /// # Returns
    ///
    /// Returns Ok(()) if the gradient was successfully set, or an error if gradient tracking
    /// is not enabled or if the gradient shape doesn't match the tensor shape.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::{Tensor, utils::enable_grad};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// enable_grad();
    /// let mut x = Tensor::randn(&[2, 3])?;
    /// let grad = Tensor::ones(&[2, 3])?;
    /// x.set_grad(grad)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_grad(&mut self, grad: Tensor) -> Result<()> {
        if !is_grad_enabled() {
            return Err(TrustformersError::tensor_op_error(
                "Gradient tracking is not enabled. Use enable_grad() to enable gradient tracking.",
                "set_grad",
            ));
        }

        // Validate gradient shape matches tensor shape
        if self.shape() != grad.shape() {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Gradient shape {:?} doesn't match tensor shape {:?}",
                    grad.shape(),
                    self.shape()
                ),
                "set_grad",
            ));
        }

        let tensor_id = self.tensor_id();

        if let Ok(mut registry) = GRADIENT_REGISTRY.write() {
            registry.insert(tensor_id, grad);
            Ok(())
        } else {
            Err(TrustformersError::tensor_op_error(
                "Failed to access gradient registry.",
                "set_grad",
            ))
        }
    }

    /// Get tensor data as a vector (for F32 tensors).
    ///
    /// # Returns
    ///
    /// A Result containing a vector with the tensor data.
    pub fn data(&self) -> Result<Vec<f32>> {
        match self {
            Tensor::F32(a) => Ok(a.iter().cloned().collect()),
            Tensor::F64(a) => Ok(a.iter().map(|&x| x as f32).collect()),
            Tensor::I64(a) => Ok(a.iter().map(|&x| x as f32).collect()),
            #[cfg(feature = "metal")]
            Tensor::Metal(_) => {
                // Convert to CPU first, then get data
                let cpu_tensor = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_tensor.data()
            },
            #[cfg(feature = "cuda")]
            Tensor::CUDA(_) => {
                // Convert to CPU first, then get data
                let cpu_tensor = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_tensor.data()
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for data conversion",
                "data_conversion",
            )),
        }
    }

    /// Get tensor data as F32 vector (alias for data() method).
    ///
    /// # Returns
    ///
    /// A Result containing a vector with the tensor data as f32.
    pub fn data_f32(&self) -> Result<Vec<f32>> {
        self.data()
    }

    /// Set tensor data from F32 vector.
    ///
    /// # Arguments
    ///
    /// * `data` - Vector of f32 values to set as tensor data
    ///
    /// # Returns
    ///
    /// A Result indicating success or failure.
    pub fn set_data_f32(&mut self, data: &[f32]) -> Result<()> {
        match self {
            Tensor::F32(a) => {
                let shape = a.shape().to_vec();
                let expected_len: usize = shape.iter().product();
                if data.len() != expected_len {
                    return Err(TrustformersError::tensor_op_error(
                        &format!(
                            "Data length {} does not match tensor size {}",
                            data.len(),
                            expected_len
                        ),
                        "set_data_f32",
                    ));
                }
                *a = ArrayD::from_shape_vec(IxDyn(&shape), data.to_vec()).map_err(|e| {
                    TrustformersError::tensor_op_error(&e.to_string(), "set_data_f32")
                })?;
                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "set_data_f32 only supported for F32 tensors",
                "set_data_f32",
            )),
        }
    }

    /// Get mutable reference to tensor data as a slice (for F32 tensors).
    ///
    /// # Returns
    ///
    /// A Result containing a mutable slice of the tensor data.
    pub fn data_mut(&mut self) -> Result<&mut [f32]> {
        match self {
            Tensor::F32(a) => Ok(a.as_slice_mut().unwrap()),
            _ => Err(TrustformersError::tensor_op_error(
                "Mutable data access only supported for F32 tensors",
                "data_mut",
            )),
        }
    }

    /// Modify tensor data in-place with a closure.
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that takes a mutable slice of the tensor data
    ///
    /// # Returns
    ///
    /// A Result indicating success or failure.
    pub fn modify_data<F>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(&mut [f32]),
    {
        match self {
            Tensor::F32(a) => {
                if let Some(slice) = a.as_slice_mut() {
                    f(slice);
                    Ok(())
                } else {
                    Err(TrustformersError::tensor_op_error(
                        "Cannot get mutable slice",
                        "modify_data",
                    ))
                }
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Modify data only supported for F32 tensors",
                "modify_data",
            )),
        }
    }

    /// Get the device where the tensor is stored.
    ///
    /// # Returns
    ///
    /// A string representing the device.
    pub fn device(&self) -> String {
        match self {
            Tensor::F32(_)
            | Tensor::F64(_)
            | Tensor::F16(_)
            | Tensor::BF16(_)
            | Tensor::I64(_)
            | Tensor::C32(_)
            | Tensor::C64(_)
            | Tensor::CF16(_)
            | Tensor::CBF16(_) => "cpu".to_string(),
            Tensor::Sparse(_) => "cpu".to_string(),
            #[cfg(feature = "torch")]
            Tensor::Torch(t) => format!("{:?}", t.device()),
            #[cfg(feature = "candle")]
            Tensor::Candle(t) => format!("{:?}", t.device()),
            #[cfg(feature = "metal")]
            Tensor::Metal(_) => "metal".to_string(),
            #[cfg(feature = "cuda")]
            Tensor::CUDA(_) => "cuda".to_string(),
        }
    }

    /// Get the number of elements in the tensor.
    ///
    /// # Returns
    ///
    /// The total number of elements.
    pub fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Get memory usage in bytes.
    ///
    /// # Returns
    ///
    /// Memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        match self {
            Tensor::F32(a) => a.len() * std::mem::size_of::<f32>(),
            Tensor::F64(a) => a.len() * std::mem::size_of::<f64>(),
            Tensor::F16(a) => a.len() * std::mem::size_of::<half::f16>(),
            Tensor::BF16(a) => a.len() * std::mem::size_of::<half::bf16>(),
            Tensor::I64(a) => a.len() * std::mem::size_of::<i64>(),
            Tensor::C32(a) => a.len() * std::mem::size_of::<scirs2_core::Complex32>(),
            Tensor::C64(a) => a.len() * std::mem::size_of::<scirs2_core::Complex64>(),
            Tensor::CF16(a) => a.len() * std::mem::size_of::<scirs2_core::Complex<half::f16>>(),
            Tensor::CBF16(a) => a.len() * std::mem::size_of::<scirs2_core::Complex<half::bf16>>(),
            Tensor::Sparse(s) => s.memory_usage(),
            #[cfg(feature = "torch")]
            Tensor::Torch(t) => t.numel() * 4, // Approximate
            #[cfg(feature = "candle")]
            Tensor::Candle(t) => t.elem_count() * 4, // Approximate
            #[cfg(feature = "metal")]
            Tensor::Metal(m) => m.shape.iter().product::<usize>() * 4, // Approximate as f32
            #[cfg(feature = "cuda")]
            Tensor::CUDA(c) => c.shape.iter().product::<usize>() * 4, // Approximate as f32
        }
    }

    /// Get the data type of the tensor.
    ///
    /// # Returns
    ///
    /// The data type.
    pub fn dtype(&self) -> DType {
        match self {
            Tensor::F32(_) => DType::F32,
            Tensor::F64(_) => DType::F64,
            Tensor::F16(_) => DType::F16,
            Tensor::BF16(_) => DType::BF16,
            Tensor::I64(_) => DType::I64,
            Tensor::C32(_) => DType::C32,
            Tensor::C64(_) => DType::C64,
            Tensor::CF16(_) => DType::CF16,
            Tensor::CBF16(_) => DType::CBF16,
            Tensor::Sparse(_) => DType::F32, // Sparse tensors use f32 by default
            #[cfg(feature = "torch")]
            Tensor::Torch(_) => DType::F32, // Default assumption
            #[cfg(feature = "candle")]
            Tensor::Candle(_) => DType::F32, // Default assumption
            #[cfg(feature = "metal")]
            Tensor::Metal(data) => data.dtype,
            #[cfg(feature = "cuda")]
            Tensor::CUDA(data) => data.dtype,
        }
    }

    /// Get the data type (alias for dtype).
    pub fn get_dtype(&self) -> DType {
        self.dtype()
    }

    /// Get a float value at a specific index.
    ///
    /// # Arguments
    ///
    /// * `index` - The linear index
    ///
    /// # Returns
    ///
    /// The float value at the index.
    pub fn get_float(&self, index: usize) -> Result<f32> {
        match self {
            Tensor::F32(a) => {
                if index >= a.len() {
                    return Err(TrustformersError::tensor_op_error(
                        &format!(
                            "Index {} out of bounds for tensor of size {}",
                            index,
                            a.len()
                        ),
                        "get_float",
                    ));
                }
                Ok(a.iter().nth(index).copied().unwrap_or(0.0))
            },
            Tensor::F64(a) => {
                if index >= a.len() {
                    return Err(TrustformersError::tensor_op_error(
                        &format!(
                            "Index {} out of bounds for tensor of size {}",
                            index,
                            a.len()
                        ),
                        "get_float",
                    ));
                }
                Ok(a.iter().nth(index).copied().unwrap_or(0.0) as f32)
            },
            #[cfg(feature = "metal")]
            Tensor::Metal(_) => {
                // Convert to CPU first, then get float
                let cpu_tensor = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_tensor.get_float(index)
            },
            #[cfg(feature = "cuda")]
            Tensor::CUDA(_) => {
                // Convert to CPU first, then get float
                let cpu_tensor = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_tensor.get_float(index)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Get float not supported for this tensor type",
                "get_float",
            )),
        }
    }

    /// Get a scalar value from a 0-dimensional or 1-element tensor.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type to convert to (i32, i64, f32, f64)
    ///
    /// # Returns
    ///
    /// The scalar value.
    pub fn item<T>(&self) -> Result<T>
    where
        T: num_traits::NumCast,
    {
        if self.len() != 1 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "item() requires a single-element tensor, but got {} elements",
                    self.len()
                ),
                "item",
            ));
        }

        match self {
            Tensor::F32(a) => {
                let val = a.iter().next().copied().unwrap_or(0.0);
                T::from(val).ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to convert f32 to target type",
                        "item",
                    )
                })
            },
            Tensor::F64(a) => {
                let val = a.iter().next().copied().unwrap_or(0.0);
                T::from(val).ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to convert f64 to target type",
                        "item",
                    )
                })
            },
            Tensor::I64(a) => {
                let val = a.iter().next().copied().unwrap_or(0);
                T::from(val).ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to convert i64 to target type",
                        "item",
                    )
                })
            },
            #[cfg(feature = "metal")]
            Tensor::Metal(_) => {
                // Convert to CPU first, then get item
                let cpu_tensor = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_tensor.item::<T>()
            },
            #[cfg(feature = "cuda")]
            Tensor::CUDA(_) => {
                // Convert to CPU first, then get item
                let cpu_tensor = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_tensor.item::<T>()
            },
            _ => Err(TrustformersError::tensor_op_error(
                "item() not supported for this tensor type",
                "item",
            )),
        }
    }

    /// Get an i64 scalar value from a tensor.
    ///
    /// # Returns
    ///
    /// The i64 scalar value.
    pub fn get_scalar_i64(&self) -> Result<i64> {
        self.item::<i64>()
    }

    /// Compare tensor elements with a scalar value.
    /// TEMPORARY: Uses ndarray. Will be replaced with SciRS2-Core.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to compare against
    ///
    /// # Returns
    ///
    /// A boolean tensor where True indicates elements equal to the scalar.
    pub fn eq_scalar(&self, scalar: f64) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let scalar_f32 = scalar as f32;
                let result =
                    a.mapv(|x| if (x - scalar_f32).abs() < 1e-6 { 1.0f32 } else { 0.0f32 });
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| if (x - scalar).abs() < 1e-9 { 1.0f64 } else { 0.0f64 });
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let scalar_i64 = scalar as i64;
                let result = a.mapv(|x| if x == scalar_i64 { 1i64 } else { 0i64 });
                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "eq_scalar not supported for this tensor type",
                "eq_scalar",
            )),
        }
    }

    /// Split tensor into batches along the first dimension
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Size of each batch
    ///
    /// # Returns
    ///
    /// Vector of tensors, each representing a batch. The last batch may be smaller
    /// if the tensor size is not evenly divisible by batch_size.
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_core::tensor::Tensor;
    ///
    /// let tensor = Tensor::ones(&[10, 4]).unwrap();
    /// let batches = tensor.batch_split(3).unwrap();
    /// assert_eq!(batches.len(), 4); // [3, 3, 3, 1]
    /// assert_eq!(batches[0].shape(), &[3, 4]);
    /// assert_eq!(batches[3].shape(), &[1, 4]);
    /// ```
    pub fn batch_split(&self, batch_size: usize) -> Result<Vec<Tensor>> {
        if batch_size == 0 {
            return Err(TrustformersError::tensor_op_error(
                "Batch size must be greater than 0",
                "batch_split",
            ));
        }

        let shape = self.shape();
        if shape.is_empty() {
            return Err(TrustformersError::tensor_op_error(
                "Cannot batch split a scalar tensor",
                "batch_split",
            ));
        }

        let total_size = shape[0];
        let mut batches = Vec::new();

        for start in (0..total_size).step_by(batch_size) {
            let end = std::cmp::min(start + batch_size, total_size);
            let batch = self.slice(0, start, end)?;
            batches.push(batch);
        }

        Ok(batches)
    }

    /// Batch tensors together along a new first dimension
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensors to batch together. All tensors must have the same shape.
    ///
    /// # Returns
    ///
    /// A new tensor with shape [batch_size, original_shape...]
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_core::tensor::Tensor;
    ///
    /// let t1 = Tensor::ones(&[3, 4]).unwrap();
    /// let t2 = Tensor::zeros(&[3, 4]).unwrap();
    /// let t3 = Tensor::ones(&[3, 4]).unwrap();
    ///
    /// let batched = Tensor::batch_stack(&[&t1, &t2, &t3]).unwrap();
    /// assert_eq!(batched.shape(), &[3, 3, 4]);
    /// ```
    pub fn batch_stack(tensors: &[&Tensor]) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(TrustformersError::tensor_op_error(
                "Cannot stack empty tensor list",
                "batch_stack",
            ));
        }

        // Verify all tensors have the same shape
        let reference_shape = tensors[0].shape();
        for (i, tensor) in tensors.iter().enumerate() {
            if tensor.shape() != reference_shape {
                return Err(TrustformersError::tensor_op_error(
                    &format!(
                        "Tensor {} has shape {:?}, expected {:?}",
                        i,
                        tensor.shape(),
                        reference_shape
                    ),
                    "batch_stack",
                ));
            }
        }

        // Create new shape with batch dimension
        let mut new_shape = vec![tensors.len()];
        new_shape.extend_from_slice(&reference_shape);

        match tensors[0] {
            Tensor::F32(_) => {
                let mut result_data = Vec::new();
                for tensor in tensors {
                    if let Tensor::F32(arr) = tensor {
                        result_data.extend(arr.iter().copied());
                    }
                }
                Tensor::from_vec(result_data, &new_shape)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Batch stacking currently only implemented for F32 tensors",
                "batch_stack",
            )),
        }
    }

    /// Unbatch a tensor by removing the first dimension
    ///
    /// # Returns
    ///
    /// Vector of tensors, each representing an item from the batch
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_core::tensor::Tensor;
    ///
    /// let batched = Tensor::ones(&[3, 4, 5]).unwrap();
    /// let unbatched = batched.unbatch().unwrap();
    /// assert_eq!(unbatched.len(), 3);
    /// assert_eq!(unbatched[0].shape(), &[4, 5]);
    /// ```
    pub fn unbatch(&self) -> Result<Vec<Tensor>> {
        let shape = self.shape();
        if shape.is_empty() {
            return Err(TrustformersError::tensor_op_error(
                "Cannot unbatch a scalar tensor",
                "unbatch",
            ));
        }

        let batch_size = shape[0];
        let mut items = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let item = self.slice(0, i, i + 1)?;
            // Remove the batch dimension by squeezing the first axis
            let squeezed = item.squeeze(0)?;
            items.push(squeezed);
        }

        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_tracking_basic() {
        // Test basic gradient functionality
        enable_grad();

        let mut x = Tensor::ones(&[2, 3]).unwrap();
        let grad = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Set gradient
        assert!(x.set_grad(grad.clone()).is_ok());

        // Get gradient
        let retrieved_grad = x.grad().unwrap();
        assert_eq!(retrieved_grad.shape(), vec![2, 3]);

        disable_grad();
    }

    #[test]
    fn test_gradient_tracking_disabled() {
        // Test that gradients fail when tracking is disabled
        disable_grad();

        let mut x = Tensor::ones(&[2, 3]).unwrap();
        let grad = Tensor::ones(&[2, 3]).unwrap();

        // Should fail when gradient tracking is disabled
        assert!(x.set_grad(grad).is_err());
        assert!(x.grad().is_err());
    }

    #[test]
    fn test_gradient_shape_validation() {
        enable_grad();

        let mut x = Tensor::ones(&[2, 3]).unwrap();
        let wrong_shape_grad = Tensor::ones(&[3, 2]).unwrap();

        // Should fail when gradient shape doesn't match tensor shape
        assert!(x.set_grad(wrong_shape_grad).is_err());

        disable_grad();
    }

    #[test]
    fn test_clear_gradients() {
        enable_grad();

        let mut x = Tensor::ones(&[2, 3]).unwrap();
        let grad = Tensor::ones(&[2, 3]).unwrap();

        // Set gradient
        x.set_grad(grad).unwrap();

        // Verify gradient exists
        assert!(x.grad().is_ok());

        // Clear all gradients
        clear_gradients();

        // Gradient should no longer exist
        assert!(x.grad().is_err());

        disable_grad();
    }

    #[test]
    fn test_gradient_mode_functions() {
        // Test gradient mode control functions
        disable_grad();
        assert!(!is_grad_enabled());

        enable_grad();
        assert!(is_grad_enabled());

        disable_grad();
        assert!(!is_grad_enabled());
    }
}
