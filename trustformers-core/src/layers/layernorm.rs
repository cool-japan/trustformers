//! Layer Normalization implementations
//!
//! This module provides LayerNorm and RMSNorm layers with device support.

use crate::device::Device;
use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use crate::traits::Layer;
use scirs2_core::ndarray::{Axis, IxDyn};

/// Layer Normalization
///
/// Normalizes activations across the feature dimension, providing more stable training
/// and faster convergence. Used extensively in transformer architectures.
///
/// # Parameters
///
/// - `weight`: Learnable affine transform weight (gamma)
/// - `bias`: Learnable affine transform bias (beta)
/// - `eps`: Small constant added to variance for numerical stability
///
/// # Example
///
/// ```no_run
/// use trustformers_core::layers::LayerNorm;
/// use trustformers_core::tensor::Tensor;
/// use trustformers_core::traits::Layer;
/// use trustformers_core::device::Device;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create LayerNorm for hidden size 768
/// let layer_norm = LayerNorm::new_with_device(vec![768], 1e-5, Device::CPU)?;
///
/// // Apply normalization
/// let input = Tensor::randn(&[4, 128, 768])?;
/// let normalized = layer_norm.forward(input)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    device: Device,
}

impl LayerNorm {
    /// Creates a new LayerNorm layer on CPU
    pub fn new(normalized_shape: Vec<usize>, eps: f32) -> Result<Self> {
        Self::new_with_device(normalized_shape, eps, Device::CPU)
    }

    /// Creates a new LayerNorm layer on specified device
    pub fn new_with_device(normalized_shape: Vec<usize>, eps: f32, device: Device) -> Result<Self> {
        let weight = Tensor::ones(&normalized_shape)?;
        let bias = Tensor::zeros(&normalized_shape)?;

        Ok(Self {
            normalized_shape,
            weight,
            bias,
            eps,
            device,
        })
    }

    /// Creates a simple 1D LayerNorm on CPU
    pub fn new_simple(normalized_shape: usize, eps: f32) -> Self {
        Self::new(vec![normalized_shape], eps).unwrap()
    }

    /// Sets the weight tensor
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    /// Sets the bias tensor
    pub fn set_bias(&mut self, bias: Tensor) -> Result<()> {
        self.bias = bias;
        Ok(())
    }

    /// Returns the device this layer uses for computations
    pub fn device(&self) -> Device {
        self.device
    }

    /// Moves this layer to a different device
    pub fn to_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Returns the total number of learnable parameters in this layer
    pub fn parameter_count(&self) -> usize {
        let weight_count = self.weight.len();
        let bias_count = self.bias.len();
        weight_count + bias_count
    }

    /// Pre-upload layer parameters to GPU for zero-transfer pipeline
    /// This converts weight and bias to Metal tensors for GPU-resident computation
    #[cfg(feature = "metal")]
    pub fn weights_to_gpu(&mut self, device: &crate::device::Device) -> Result<()> {
        use crate::device::Device;

        if !matches!(device, Device::Metal(_)) {
            return Ok(());
        }

        // Update device setting
        self.device = *device;

        // Convert weight and bias to Metal tensors
        // LayerNorm has a GPU-to-GPU kernel that needs Metal weight/bias
        self.weight = self.weight.to_device_enum(device)?;
        self.bias = self.bias.to_device_enum(device)?;

        Ok(())
    }

    /// Pre-upload layer parameters to CUDA GPU for zero-transfer pipeline
    /// This converts weight and bias to CUDA tensors for GPU-resident computation
    #[cfg(feature = "cuda")]
    pub fn weights_to_gpu_cuda(&mut self, device: &crate::device::Device) -> Result<()> {
        use crate::device::Device;

        if !matches!(device, Device::CUDA(_)) {
            return Ok(());
        }

        // Update device setting
        self.device = *device;

        // Convert weight and bias to CUDA tensors
        // LayerNorm has a GPU-to-GPU kernel that needs CUDA weight/bias
        self.weight = self.weight.to_device_enum(device)?;
        self.bias = self.bias.to_device_enum(device)?;

        Ok(())
    }
}

impl Layer for LayerNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            // GPU-resident Metal tensor - process on GPU
            #[cfg(feature = "metal")]
            Tensor::Metal(metal_data) => {
                use crate::gpu_ops::metal::get_metal_backend;
                use crate::tensor::MetalTensorData;

                // Check if we can use GPU kernel (2D or 3D with matching last dimension)
                let can_use_gpu = (metal_data.shape.len() == 2 || metal_data.shape.len() == 3)
                    && self.normalized_shape.len() == 1
                    && metal_data.shape[metal_data.shape.len() - 1] == self.normalized_shape[0];

                if can_use_gpu {
                    let backend = get_metal_backend()?;
                    let hidden_size = self.normalized_shape[0];

                    // Get weight and bias buffer IDs
                    match (&self.weight, &self.bias) {
                        (Tensor::Metal(w_data), Tensor::Metal(b_data)) => {
                            // eprintln!(
                            //     "✅ LayerNorm: GPU-to-GPU path (Metal→Metal, shape: {:?})",
                            //     metal_data.shape
                            // );

                            if metal_data.shape.len() == 2 {
                                // 2D case: (seq_len, hidden_size)
                                let seq_len = metal_data.shape[0];

                                let output_buffer_id = backend.layernorm_gpu_to_gpu(
                                    &metal_data.buffer_id,
                                    &w_data.buffer_id,
                                    &b_data.buffer_id,
                                    seq_len,
                                    hidden_size,
                                    self.eps,
                                )?;

                                return Ok(Tensor::Metal(MetalTensorData {
                                    buffer_id: output_buffer_id,
                                    shape: metal_data.shape.clone(),
                                    dtype: metal_data.dtype,
                                }));
                            } else if metal_data.shape.len() == 3 {
                                // 3D case: (batch, seq_len, hidden_size)
                                // Flatten to 2D: (batch * seq_len, hidden_size)
                                let batch = metal_data.shape[0];
                                let seq_len = metal_data.shape[1];
                                let flattened_seq_len = batch * seq_len;

                                // eprintln!(
                                //     "   Reshaping 3D→2D: {:?} → [{}, {}]",
                                //     metal_data.shape, flattened_seq_len, hidden_size
                                // );

                                // Run GPU kernel on flattened 2D tensor
                                let output_buffer_id = backend.layernorm_gpu_to_gpu(
                                    &metal_data.buffer_id,
                                    &w_data.buffer_id,
                                    &b_data.buffer_id,
                                    flattened_seq_len,
                                    hidden_size,
                                    self.eps,
                                )?;

                                // eprintln!(
                                //     "   Reshaping 2D→3D: [{}, {}] → {:?}",
                                //     flattened_seq_len, hidden_size, metal_data.shape
                                // );

                                // Return with original 3D shape
                                return Ok(Tensor::Metal(MetalTensorData {
                                    buffer_id: output_buffer_id,
                                    shape: metal_data.shape.clone(),
                                    dtype: metal_data.dtype,
                                }));
                            }
                        },
                        _ => {
                            // eprintln!("⚠️  LayerNorm: Weight/bias not on GPU, falling back to CPU");
                        },
                    }
                } else {
                    // eprintln!(
                    //     "⚠️  LayerNorm: Unsupported shape {:?}, falling back to CPU",
                    //     metal_data.shape
                    // );
                }

                // Fallback: convert to CPU and process (avoid recursion)
                let cpu_input = input.to_device_enum(&crate::device::Device::CPU)?;
                let cpu_weight = self.weight.to_device_enum(&crate::device::Device::CPU)?;
                let cpu_bias = self.bias.to_device_enum(&crate::device::Device::CPU)?;

                // Extract F32 arrays
                let input_arr = match cpu_input {
                    Tensor::F32(arr) => arr,
                    _ => {
                        return Err(TrustformersError::tensor_op_error(
                            "Failed to convert input to F32",
                            "LayerNorm::forward",
                        ))
                    },
                };
                let weight_arr = match cpu_weight {
                    Tensor::F32(arr) => arr,
                    _ => {
                        return Err(TrustformersError::tensor_op_error(
                            "Failed to convert weight to F32",
                            "LayerNorm::forward",
                        ))
                    },
                };
                let bias_arr = match cpu_bias {
                    Tensor::F32(arr) => arr,
                    _ => {
                        return Err(TrustformersError::tensor_op_error(
                            "Failed to convert bias to F32",
                            "LayerNorm::forward",
                        ))
                    },
                };

                // Process on CPU directly (inline to avoid recursion)
                let ndim = input_arr.ndim();
                let norm_ndim = self.normalized_shape.len();
                let axes: Vec<usize> = ((ndim - norm_ndim)..ndim).collect();

                // Compute mean
                let mut mean = input_arr.clone();
                for &axis in axes.iter().rev() {
                    mean = mean.mean_axis(Axis(axis)).unwrap().insert_axis(Axis(axis));
                }

                // Compute variance
                let diff = &input_arr - &mean;
                let mut var = (&diff * &diff).to_owned();
                for &axis in axes.iter().rev() {
                    var = var.mean_axis(Axis(axis)).unwrap().insert_axis(Axis(axis));
                }

                // Normalize
                let normalized = &diff / (var + self.eps).mapv(f32::sqrt);

                // Broadcast weight and bias
                let mut broadcast_shape = vec![1; ndim];
                for (i, &dim) in self.normalized_shape.iter().enumerate() {
                    broadcast_shape[ndim - norm_ndim + i] = dim;
                }

                let w_broadcast = weight_arr
                    .view()
                    .into_shape_with_order(IxDyn(&broadcast_shape))
                    .map_err(|e| {
                        TrustformersError::shape_error(format!("Failed to broadcast weight: {}", e))
                    })?;
                let b_broadcast = bias_arr
                    .view()
                    .into_shape_with_order(IxDyn(&broadcast_shape))
                    .map_err(|e| {
                        TrustformersError::shape_error(format!("Failed to broadcast bias: {}", e))
                    })?;

                let output = &normalized * &w_broadcast + &b_broadcast;
                return Ok(Tensor::F32(output));
            },

            // GPU-resident CUDA tensor - process on GPU
            #[cfg(feature = "cuda")]
            Tensor::CUDA(_cuda_data) => {
                #[allow(unused_imports)]
                use crate::tensor::CudaTensorData;

                #[cfg(any(target_os = "linux", target_os = "windows"))]
                {
                    use crate::gpu_ops::cuda::get_cuda_backend;

                    if cuda_data.shape.len() == 2 && self.normalized_shape.len() == 1 {
                        let device_id = if let Device::CUDA(id) = self.device { id } else { 0 };
                        let backend = get_cuda_backend(device_id)?;
                        let shape = &cuda_data.shape;
                        let seq_len = shape[0];
                        let hidden_size = shape[1];

                        if hidden_size == self.normalized_shape[0] {
                            // Get weight and bias buffer IDs
                            match (&self.weight, &self.bias) {
                                (Tensor::CUDA(w_data), Tensor::CUDA(b_data)) => {
                                    // All on GPU - zero transfers!
                                    let output_buffer_id = backend.layernorm_gpu_to_gpu(
                                        &cuda_data.buffer_id,
                                        &w_data.buffer_id,
                                        &b_data.buffer_id,
                                        seq_len,
                                        hidden_size,
                                        self.eps,
                                    )?;

                                    return Ok(Tensor::CUDA(CudaTensorData {
                                        buffer_id: output_buffer_id,
                                        shape: cuda_data.shape.clone(),
                                        dtype: cuda_data.dtype,
                                    }));
                                },
                                _ => {
                                    // Weight/bias not on GPU - fallback to CPU
                                },
                            }
                        }
                    }
                }

                // Fallback: convert to CPU and process (avoid recursion)
                // This handles 3D CUDA tensors that can't use 2D GPU kernel
                let cpu_input = input.to_device_enum(&crate::device::Device::CPU)?;
                let cpu_weight = self.weight.to_device_enum(&crate::device::Device::CPU)?;
                let cpu_bias = self.bias.to_device_enum(&crate::device::Device::CPU)?;

                // Extract F32 arrays
                let input_arr = match cpu_input {
                    Tensor::F32(arr) => arr,
                    _ => {
                        return Err(TrustformersError::tensor_op_error(
                            "Failed to convert input to F32",
                            "LayerNorm::forward",
                        ))
                    },
                };
                let weight_arr = match cpu_weight {
                    Tensor::F32(arr) => arr,
                    _ => {
                        return Err(TrustformersError::tensor_op_error(
                            "Failed to convert weight to F32",
                            "LayerNorm::forward",
                        ))
                    },
                };
                let bias_arr = match cpu_bias {
                    Tensor::F32(arr) => arr,
                    _ => {
                        return Err(TrustformersError::tensor_op_error(
                            "Failed to convert bias to F32",
                            "LayerNorm::forward",
                        ))
                    },
                };

                // Process on CPU directly (inline to avoid recursion)
                let ndim = input_arr.ndim();
                let norm_ndim = self.normalized_shape.len();
                let axes: Vec<usize> = ((ndim - norm_ndim)..ndim).collect();

                // Compute mean
                let mut mean = input_arr.clone();
                for &axis in axes.iter().rev() {
                    mean = mean.mean_axis(Axis(axis)).unwrap().insert_axis(Axis(axis));
                }

                // Compute variance
                let diff = &input_arr - &mean;
                let mut var = (&diff * &diff).to_owned();
                for &axis in axes.iter().rev() {
                    var = var.mean_axis(Axis(axis)).unwrap().insert_axis(Axis(axis));
                }

                // Normalize
                let normalized = &diff / (var + self.eps).mapv(f32::sqrt);

                // Broadcast weight and bias
                let mut broadcast_shape = vec![1; ndim];
                for (i, &dim) in self.normalized_shape.iter().enumerate() {
                    broadcast_shape[ndim - norm_ndim + i] = dim;
                }

                let w_broadcast = weight_arr
                    .view()
                    .into_shape_with_order(IxDyn(&broadcast_shape))
                    .map_err(|e| {
                        TrustformersError::shape_error(format!("Failed to broadcast weight: {}", e))
                    })?;
                let b_broadcast = bias_arr
                    .view()
                    .into_shape_with_order(IxDyn(&broadcast_shape))
                    .map_err(|e| {
                        TrustformersError::shape_error(format!("Failed to broadcast bias: {}", e))
                    })?;

                let output = &normalized * &w_broadcast + &b_broadcast;
                return Ok(Tensor::F32(output));
            },

            Tensor::F32(arr) => {
                // Try Metal GPU acceleration for 2D tensors (seq_len, hidden_size)
                #[cfg(feature = "metal")]
                {
                    use crate::gpu_ops::metal::get_metal_backend;
                    if arr.ndim() == 2 && self.normalized_shape.len() == 1 {
                        if let Ok(backend) = get_metal_backend() {
                            let shape = arr.shape();
                            let seq_len = shape[0];
                            let hidden_size = shape[1];

                            if hidden_size == self.normalized_shape[0] {
                                match (&self.weight, &self.bias) {
                                    // Case 1: Weight/bias on GPU - upload input and use GPU-to-GPU
                                    #[allow(unreachable_patterns)]
                                    (Tensor::Metal(w_data), Tensor::Metal(b_data)) => {
                                        use crate::tensor::MetalTensorData;

                                        // Upload input to GPU
                                        let input_vec: Vec<f32> = arr.iter().copied().collect();
                                        let input_buffer_id =
                                            backend.create_persistent_buffer(&input_vec)?;

                                        // Execute GPU-to-GPU
                                        let output_buffer_id = backend.layernorm_gpu_to_gpu(
                                            &input_buffer_id,
                                            &w_data.buffer_id,
                                            &b_data.buffer_id,
                                            seq_len,
                                            hidden_size,
                                            self.eps,
                                        )?;

                                        // Return Metal tensor
                                        return Ok(Tensor::Metal(MetalTensorData {
                                            buffer_id: output_buffer_id,
                                            shape: arr.shape().to_vec(),
                                            dtype: crate::tensor::DType::F32,
                                        }));
                                    },

                                    // Case 2: Weight/bias on CPU - standard path
                                    (Tensor::F32(w_arr), Tensor::F32(b_arr)) => {
                                        let input_vec: Vec<f32> = arr.iter().copied().collect();
                                        let weight_vec: Vec<f32> = w_arr.iter().copied().collect();
                                        let bias_vec: Vec<f32> = b_arr.iter().copied().collect();

                                        // Execute on GPU
                                        if let Ok(output_vec) = backend.layernorm_f32(
                                            &input_vec,
                                            &weight_vec,
                                            &bias_vec,
                                            seq_len,
                                            hidden_size,
                                            self.eps,
                                        ) {
                                            // Convert back to tensor
                                            use scirs2_core::ndarray::ArrayD;
                                            let output_arr =
                                                ArrayD::from_shape_vec(arr.raw_dim(), output_vec)
                                                    .map_err(|e| {
                                                    TrustformersError::tensor_op_error(
                                                        &format!(
                                                        "Failed to reshape LayerNorm result: {}",
                                                        e
                                                    ),
                                                        "LayerNorm::forward",
                                                    )
                                                })?;
                                            return Ok(Tensor::F32(output_arr));
                                        }
                                    },
                                    _ => {},
                                }
                            }
                        }
                    }
                }

                // Fallback to CPU implementation
                let ndim = arr.ndim();
                let norm_ndim = self.normalized_shape.len();

                // For LayerNorm, we normalize over the last norm_ndim dimensions
                // Calculate mean and variance
                let axes: Vec<usize> = ((ndim - norm_ndim)..ndim).collect();

                // Compute mean across the normalized dimensions
                let mut mean = arr.clone();
                for &axis in axes.iter().rev() {
                    mean = mean.mean_axis(Axis(axis)).unwrap().insert_axis(Axis(axis));
                }

                // Compute variance
                let diff = arr - &mean;
                let mut var = (&diff * &diff).to_owned();
                for &axis in axes.iter().rev() {
                    var = var.mean_axis(Axis(axis)).unwrap().insert_axis(Axis(axis));
                }

                // Normalize
                let normalized = &diff / (var + self.eps).mapv(f32::sqrt);

                // Convert weight/bias to F32 if needed
                let weight_f32 = match &self.weight {
                    Tensor::F32(w) => w.clone(),
                    #[cfg(feature = "metal")]
                    Tensor::Metal(_) => {
                        let cpu_weight = self.weight.to_device_enum(&crate::device::Device::CPU)?;
                        match cpu_weight {
                            Tensor::F32(w) => w,
                            _ => {
                                return Err(TrustformersError::tensor_op_error(
                                    "Failed to convert weight to F32",
                                    "LayerNorm::forward",
                                ))
                            },
                        }
                    },
                    #[cfg(feature = "cuda")]
                    Tensor::CUDA(_) => {
                        let cpu_weight = self.weight.to_device_enum(&crate::device::Device::CPU)?;
                        match cpu_weight {
                            Tensor::F32(w) => w,
                            _ => {
                                return Err(TrustformersError::tensor_op_error(
                                    "Failed to convert CUDA weight to F32",
                                    "LayerNorm::forward",
                                ))
                            },
                        }
                    },
                    _ => {
                        return Err(TrustformersError::tensor_op_error(
                            "Unsupported weight type",
                            "LayerNorm::forward",
                        ))
                    },
                };

                let bias_f32 = match &self.bias {
                    Tensor::F32(b) => b.clone(),
                    #[cfg(feature = "metal")]
                    Tensor::Metal(_) => {
                        let cpu_bias = self.bias.to_device_enum(&crate::device::Device::CPU)?;
                        match cpu_bias {
                            Tensor::F32(b) => b,
                            _ => {
                                return Err(TrustformersError::tensor_op_error(
                                    "Failed to convert bias to F32",
                                    "LayerNorm::forward",
                                ))
                            },
                        }
                    },
                    #[cfg(feature = "cuda")]
                    Tensor::CUDA(_) => {
                        let cpu_bias = self.bias.to_device_enum(&crate::device::Device::CPU)?;
                        match cpu_bias {
                            Tensor::F32(b) => b,
                            _ => {
                                return Err(TrustformersError::tensor_op_error(
                                    "Failed to convert CUDA bias to F32",
                                    "LayerNorm::forward",
                                ))
                            },
                        }
                    },
                    _ => {
                        return Err(TrustformersError::tensor_op_error(
                            "Unsupported bias type",
                            "LayerNorm::forward",
                        ))
                    },
                };

                // Handle broadcasting for weight and bias
                let mut broadcast_shape = vec![1; ndim];
                for (i, &dim) in self.normalized_shape.iter().enumerate() {
                    broadcast_shape[ndim - norm_ndim + i] = dim;
                }

                let w_broadcast = weight_f32
                    .view()
                    .into_shape_with_order(IxDyn(&broadcast_shape))
                    .map_err(|e| {
                        TrustformersError::shape_error(format!("Failed to broadcast weight: {}", e))
                    })?;
                let b_broadcast = bias_f32
                    .view()
                    .into_shape_with_order(IxDyn(&broadcast_shape))
                    .map_err(|e| {
                        TrustformersError::shape_error(format!("Failed to broadcast bias: {}", e))
                    })?;

                let output = &normalized * &w_broadcast + &b_broadcast;
                Ok(Tensor::F32(output))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for LayerNorm",
                "LayerNorm::forward",
            )),
        }
    }
}

/// Root Mean Square Layer Normalization
///
/// RMSNorm normalizes the input using only the root mean square (RMS) of the input,
/// without centering by subtracting the mean. This is computationally more efficient
/// than standard LayerNorm and is used in many modern architectures like LLaMA.
///
/// # Example
///
/// ```no_run
/// use trustformers_core::layers::RMSNorm;
/// use trustformers_core::tensor::Tensor;
/// use trustformers_core::traits::Layer;
/// use trustformers_core::device::Device;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create RMSNorm for hidden size 768
/// let rms_norm = RMSNorm::new_with_device(768, 1e-5, Device::CPU)?;
///
/// // Apply normalization
/// let input = Tensor::randn(&[4, 128, 768])?;
/// let normalized = rms_norm.forward(input)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
    device: Device,
}

impl RMSNorm {
    /// Creates a new RMSNorm layer on CPU
    pub fn new(hidden_size: usize, eps: f32) -> Result<Self> {
        Self::new_with_device(hidden_size, eps, Device::CPU)
    }

    /// Creates a new RMSNorm layer on specified device
    pub fn new_with_device(hidden_size: usize, eps: f32, device: Device) -> Result<Self> {
        let weight = Tensor::ones(&[hidden_size])?;
        Ok(Self {
            weight,
            eps,
            device,
        })
    }

    /// Sets the weight tensor
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    /// Returns the device this layer uses for computations
    pub fn device(&self) -> Device {
        self.device
    }

    /// Moves this layer to a different device
    pub fn to_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Returns the total number of learnable parameters in this layer
    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

impl Layer for RMSNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                let ndim = arr.ndim();
                let last_dim = ndim - 1;

                // Compute RMS: sqrt(mean(x^2))
                let squares = arr.mapv(|x| x * x);
                let mean_squares =
                    squares.mean_axis(Axis(last_dim)).unwrap().insert_axis(Axis(last_dim));
                let rms = mean_squares.mapv(|x| (x + self.eps).sqrt());

                // Normalize: x / rms
                let normalized = arr / &rms;

                // Apply weight
                match &self.weight {
                    Tensor::F32(w) => {
                        let mut broadcast_shape = vec![1; ndim];
                        broadcast_shape[last_dim] = w.len();

                        let w_broadcast = w
                            .view()
                            .into_shape_with_order(IxDyn(&broadcast_shape))
                            .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to broadcast weight: {}",
                                e
                            ))
                        })?;

                        let output = &normalized * &w_broadcast;
                        Ok(Tensor::F32(output))
                    },
                    _ => Err(TrustformersError::tensor_op_error(
                        "RMSNorm weight type mismatch",
                        "RMSNorm::forward",
                    )),
                }
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for RMSNorm",
                "RMSNorm::forward",
            )),
        }
    }
}
