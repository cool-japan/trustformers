//! Linear (fully connected) layer implementation.
//!
//! This module provides the `Linear` layer, which performs affine transformations
//! of the form: `y = xW^T + b`, where `W` is the weight matrix and `b` is the
//! optional bias vector.

use crate::device::Device;
use crate::errors::{Result, TrustformersError};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::gpu_ops::dispatch_matmul;
use crate::tensor::Tensor;
use crate::traits::Layer;
use scirs2_core::ndarray::{Array2, Ix2, IxDyn};
use scirs2_core::simd_ops::SimdUnifiedOps;

/// A linear transformation layer (fully connected layer).
///
/// The `Linear` layer applies a linear transformation to the incoming data:
/// `y = xW^T + b`. This is one of the most fundamental building blocks in
/// neural networks.
///
/// # Parameters
///
/// - `weight`: Learnable weight matrix of shape `[out_features, in_features]`
/// - `bias`: Optional learnable bias vector of shape `[out_features]`
///
/// # Input/Output Shapes
///
/// - Input: `[..., in_features]` - Can be 2D or 3D
/// - Output: `[..., out_features]` - Same number of dimensions as input
///
/// # Example
///
/// ```no_run
/// use trustformers_core::layers::Linear;
/// use trustformers_core::tensor::Tensor;
/// use trustformers_core::traits::Layer;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a linear layer: 768 → 3072
/// let linear = Linear::new(768, 3072, true)?;
///
/// // Apply to 2D input: [seq_len, in_features]
/// let input_2d = Tensor::randn(&[128, 768])?;
/// let output_2d = linear.forward(input_2d)?;  // Shape: [128, 3072]
///
/// // Apply to 3D input: [batch, seq_len, in_features]
/// let input_3d = Tensor::randn(&[4, 128, 768])?;
/// let output_3d = linear.forward(input_3d)?;  // Shape: [4, 128, 3072]
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    device: Device,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    weight_buffer_id: std::sync::Arc<std::sync::RwLock<Option<crate::gpu_ops::BufferId>>>,
    #[cfg(feature = "cuda")]
    weight_buffer_id_cuda:
        std::sync::Arc<std::sync::RwLock<Option<crate::gpu_ops::cuda::BufferId>>>,
}

impl Linear {
    /// Creates a new linear layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Size of each input sample
    /// * `out_features` - Size of each output sample
    /// * `bias` - Whether to include a learnable bias
    ///
    /// # Returns
    ///
    /// A new `Linear` layer with randomly initialized weights using a normal
    /// distribution, and bias initialized to zeros if enabled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::layers::Linear;
    ///
    /// // Linear layer without bias
    /// let linear1 = Linear::new(512, 1024, false);
    ///
    /// // Linear layer with bias
    /// let linear2 = Linear::new(512, 1024, true);
    /// ```
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self::new_with_device(in_features, out_features, bias, Device::CPU)
    }

    /// Creates a new linear layer with specified device.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Size of each input sample
    /// * `out_features` - Size of each output sample
    /// * `bias` - Whether to include a learnable bias
    /// * `device` - Device to use for computations (CPU, Metal, CUDA, etc.)
    ///
    /// # Returns
    ///
    /// A new `Linear` layer with randomly initialized weights.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::layers::Linear;
    /// use trustformers_core::Device;
    ///
    /// // Create a linear layer on Metal GPU
    /// let linear = Linear::new_with_device(768, 3072, true, Device::metal_if_available(0));
    /// ```
    pub fn new_with_device(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: Device,
    ) -> Self {
        let weight = Tensor::randn(&[out_features, in_features]).unwrap();
        let bias = if bias { Some(Tensor::zeros(&[out_features]).unwrap()) } else { None };

        Self {
            weight,
            bias,
            device,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            weight_buffer_id: std::sync::Arc::new(std::sync::RwLock::new(None)),
            #[cfg(feature = "cuda")]
            weight_buffer_id_cuda: std::sync::Arc::new(std::sync::RwLock::new(None)),
        }
    }

    /// Returns the device this layer uses for computations.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Move this layer to a different device.
    ///
    /// # Arguments
    ///
    /// * `device` - Target device
    ///
    /// # Returns
    ///
    /// Self with updated device.
    pub fn to_device(mut self, device: Device) -> Self {
        self.device = device;
        // Clear cached buffer when changing device
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Ok(mut buffer_id) = self.weight_buffer_id.write() {
                *buffer_id = None;
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let Ok(mut buffer_id) = self.weight_buffer_id_cuda.write() {
                *buffer_id = None;
            }
        }
        self
    }

    /// Sets the weight matrix for this layer.
    ///
    /// # Arguments
    ///
    /// * `weight` - The new weight tensor, must have shape `[out_features, in_features]`
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful.
    ///
    /// # Note
    ///
    /// This method is typically used when loading pretrained weights.
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.weight = weight;
        // Clear cached buffer when weights are updated
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Ok(mut buffer_id) = self.weight_buffer_id.write() {
                *buffer_id = None;
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let Ok(mut buffer_id) = self.weight_buffer_id_cuda.write() {
                *buffer_id = None;
            }
        }
        Ok(())
    }

    /// Sets the bias vector for this layer.
    ///
    /// # Arguments
    ///
    /// * `bias` - The new bias tensor, must have shape `[out_features]`
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful.
    ///
    /// # Note
    ///
    /// This will enable bias even if the layer was created without bias.
    pub fn set_bias(&mut self, bias: Tensor) -> Result<()> {
        self.bias = Some(bias);
        Ok(())
    }

    /// Returns a reference to the weight matrix.
    ///
    /// # Returns
    ///
    /// A reference to the weight tensor of shape `[out_features, in_features]`.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns a reference to the bias vector if present.
    ///
    /// # Returns
    ///
    /// `Some(&bias)` if bias is enabled, `None` otherwise.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Returns the total number of learnable parameters in this layer.
    ///
    /// # Returns
    ///
    /// The total parameter count including weights and bias (if present).
    pub fn parameter_count(&self) -> usize {
        let weight_count = self.weight.len();
        let bias_count = self.bias.as_ref().map_or(0, |b| b.len());
        weight_count + bias_count
    }

    /// Initialize persistent GPU weight buffer for Metal device
    /// This is called automatically on first forward pass with Metal device
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn ensure_weight_buffer_cached(&self) -> Result<()> {
        use crate::gpu_ops::metal::get_metal_backend;

        // Check if already cached (read lock is cheaper)
        if let Ok(buffer_id) = self.weight_buffer_id.read() {
            if buffer_id.is_some() {
                return Ok(()); // Already cached
            }
        }

        // Only cache if using Metal
        if matches!(self.device, Device::Metal(_)) {
            // Get write lock to cache the buffer
            let mut buffer_id_guard = self.weight_buffer_id.write().map_err(|_| {
                TrustformersError::hardware_error(
                    "Failed to acquire write lock on buffer cache",
                    "ensure_weight_buffer_cached",
                )
            })?;

            // Double-check after acquiring write lock (another thread might have cached it)
            if buffer_id_guard.is_some() {
                return Ok(());
            }

            // Get weight data as f32 slice
            // CRITICAL FIX: Cache the TRANSPOSED weight, not the original!
            // The Metal shader expects weight in [in_features, out_features] layout
            // but self.weight is stored as [out_features, in_features]
            let weight_t = self.weight.transpose(0, 1)?;
            match &weight_t {
                Tensor::F32(arr) => {
                    if arr.ndim() != 2 {
                        return Err(TrustformersError::shape_error(
                            "Weight tensor must be 2D for Metal caching".to_string(),
                        ));
                    }

                    // Convert to contiguous vec for GPU upload
                    // Using as_standard_layout() ensures proper row-major order
                    let contiguous_arr = arr.as_standard_layout();
                    let weight_data: Vec<f32> = contiguous_arr.iter().copied().collect();

                    // Get Metal backend and cache the buffer
                    let backend = get_metal_backend()?;
                    let new_buffer_id = backend.create_persistent_buffer(&weight_data)?;
                    *buffer_id_guard = Some(new_buffer_id);
                },
                _ => {
                    return Err(TrustformersError::tensor_op_error(
                        "Only F32 tensors supported for Metal caching",
                        "ensure_weight_buffer_cached",
                    ));
                },
            }
        }
        Ok(())
    }

    /// Pre-cache layer weights on GPU for zero-transfer pipeline
    /// This uploads weights to GPU memory in advance to avoid transfers during forward pass
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn weights_to_gpu(&mut self, device: &crate::device::Device) -> Result<()> {
        use crate::device::Device;

        if !matches!(device, Device::Metal(_)) {
            return Ok(()); // Nothing to do for non-Metal devices
        }

        // Update device setting
        self.device = *device;

        // Pre-cache weight buffer on GPU (keeps weight as F32 CPU tensor)
        // The caching mechanism handles the GPU upload internally
        self.ensure_weight_buffer_cached()?;

        // Upload bias to GPU if present (for GPU bias addition kernel)
        if let Some(ref bias) = self.bias {
            self.bias = Some(bias.to_device_enum(device)?);
        }

        Ok(())
    }

    /// Initialize persistent GPU weight buffer for CUDA device
    /// This is called automatically on first forward pass with CUDA device
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn ensure_weight_buffer_cached_cuda(&self) -> Result<()> {
        use crate::gpu_ops::cuda::get_cuda_backend;

        // Check if already cached (read lock is cheaper)
        if let Ok(buffer_id) = self.weight_buffer_id_cuda.read() {
            if buffer_id.is_some() {
                return Ok(()); // Already cached
            }
        }

        // Only cache if using CUDA
        if matches!(self.device, Device::CUDA(_)) {
            // Get write lock to cache the buffer
            let mut buffer_id_guard = self.weight_buffer_id_cuda.write().map_err(|_| {
                TrustformersError::hardware_error(
                    "Failed to acquire write lock on CUDA buffer cache",
                    "ensure_weight_buffer_cached_cuda",
                )
            })?;

            // Double-check after acquiring write lock (another thread might have cached it)
            if buffer_id_guard.is_some() {
                return Ok(());
            }

            // Get weight data as f32 slice
            // CRITICAL FIX: Cache the TRANSPOSED weight, not the original!
            // The CUDA kernel expects weight in [in_features, out_features] layout
            // but self.weight is stored as [out_features, in_features]
            let weight_t = self.weight.transpose(0, 1)?;
            match &weight_t {
                Tensor::F32(arr) => {
                    if arr.ndim() != 2 {
                        return Err(TrustformersError::shape_error(
                            "Weight tensor must be 2D for CUDA caching".to_string(),
                        ));
                    }

                    // Convert to contiguous vec for GPU upload
                    // Using as_standard_layout() ensures proper row-major order
                    let contiguous_arr = arr.as_standard_layout();
                    let weight_data: Vec<f32> = contiguous_arr.iter().copied().collect();

                    // Get CUDA backend and cache the buffer
                    let device_id = if let Device::CUDA(id) = self.device {
                        id
                    } else {
                        0 // Default to device 0
                    };
                    let backend = get_cuda_backend(device_id)?;
                    let new_buffer_id = backend.create_persistent_buffer(&weight_data)?;
                    *buffer_id_guard = Some(new_buffer_id);
                },
                _ => {
                    return Err(TrustformersError::tensor_op_error(
                        "Only F32 tensors supported for CUDA caching",
                        "ensure_weight_buffer_cached_cuda",
                    ));
                },
            }
        }
        Ok(())
    }

    /// Pre-cache layer weights on GPU for zero-transfer pipeline (CUDA)
    /// This uploads weights to GPU memory in advance to avoid transfers during forward pass
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    pub fn weights_to_gpu_cuda(&mut self, device: &crate::device::Device) -> Result<()> {
        use crate::device::Device;

        if !matches!(device, Device::CUDA(_)) {
            return Ok(()); // Nothing to do for non-CUDA devices
        }

        // Update device setting
        self.device = *device;

        // Pre-cache weight buffer on GPU (keeps weight as F32 CPU tensor)
        // The caching mechanism handles the GPU upload internally
        self.ensure_weight_buffer_cached_cuda()?;

        // Upload bias to GPU if present (for GPU bias addition kernel)
        if let Some(ref bias) = self.bias {
            self.bias = Some(bias.to_device_enum(device)?);
        }

        Ok(())
    }
}

impl Layer for Linear {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // =====================================================================
        // GPU-TO-GPU PATH: Tensor::Metal (ZERO CPU TRANSFERS!)
        // =====================================================================
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Tensor::Metal(ref input_metal) = input {
            use crate::gpu_ops::metal::get_metal_backend;
            use crate::tensor::MetalTensorData;

            // eprintln!("🎯 Linear::forward - GPU-to-GPU path triggered (Tensor::Metal input)");

            // Ensure weight buffer is cached on GPU
            self.ensure_weight_buffer_cached()?;

            // Get cached weight buffer ID
            let weight_buffer_id = {
                let buffer_id_guard = self.weight_buffer_id.read().map_err(|_| {
                    TrustformersError::hardware_error(
                        "Failed to acquire read lock on buffer cache",
                        "Linear::forward",
                    )
                })?;

                if let Some(id) = *buffer_id_guard {
                    id
                } else {
                    // Weight not cached - fallback to CPU
                    let cpu_input = input.to_device_enum(&crate::device::Device::CPU)?;
                    return self.forward(cpu_input);
                }
            };

            // Get Metal backend
            let backend = get_metal_backend()?;

            // Extract input shape and calculate matmul dimensions
            let shape = &input_metal.shape;
            let weight_shape = self.weight.shape();
            let in_features = shape[shape.len() - 1];

            // Check shape compatibility
            if in_features != weight_shape[1] {
                return Err(TrustformersError::shape_error(format!(
                    "Linear layer input features {} doesn't match weight shape {:?}",
                    in_features, weight_shape
                )));
            }

            // Flatten input to [batch, in_features] for matmul
            // Works for both 2D [seq_len, in_features] and 3D [batch, seq_len, in_features]
            let batch_dims: usize = shape[..shape.len() - 1].iter().product();
            let m = batch_dims; // number of rows in output
            let k = in_features; // shared dimension
            let n = self.weight.shape()[0]; // out_features

            // Perform GPU-to-GPU matmul using MPS (100-500x faster!)
            // Try MPS first, fallback to naive kernel if MPS unavailable
            let output_buffer_id = backend
                .matmul_gpu_to_gpu_mps(&input_metal.buffer_id, &weight_buffer_id, m, k, n)
                .or_else(|_e| {
                    // eprintln!(
                    //     "⚠️  MPS matmul failed: {:?}, falling back to naive Metal kernel",
                    //     e
                    // );
                    // Fallback to naive Metal kernel if MPS fails
                    backend.matmul_gpu_to_gpu(&input_metal.buffer_id, &weight_buffer_id, m, k, n)
                })?;

            // Calculate output shape (preserve batch dimensions, change last dim)
            let mut output_shape = shape[..shape.len() - 1].to_vec();
            output_shape.push(n);

            // Create output Metal tensor
            let mut output = Tensor::Metal(MetalTensorData {
                buffer_id: output_buffer_id,
                shape: output_shape.clone(),
                dtype: input_metal.dtype,
            });

            // Handle bias if present
            if let Some(ref bias) = self.bias {
                // eprintln!("🔍 Linear: Has bias, checking type...");
                // Try GPU-to-GPU bias addition if bias is on GPU
                match bias {
                    #[cfg(all(target_os = "macos", feature = "metal"))]
                    Tensor::Metal(bias_data) => {
                        // eprintln!("🔍 Linear: Bias is Metal, using GPU-to-GPU bias addition");
                        // Both output and bias are Metal tensors - use GPU kernel!
                        if let Tensor::Metal(output_data) = &output {
                            // eprintln!("🔍 Linear: Output is Metal, calling add_bias_gpu_to_gpu");
                            let output_buffer_id = backend.add_bias_gpu_to_gpu(
                                &output_data.buffer_id,
                                &bias_data.buffer_id,
                                batch_dims,
                                n,
                            )?;
                            // eprintln!(
                            //     "🔍 Linear: add_bias_gpu_to_gpu succeeded, returning Metal tensor"
                            // );

                            return Ok(Tensor::Metal(MetalTensorData {
                                buffer_id: output_buffer_id,
                                shape: output_shape.clone(),
                                dtype: output_data.dtype,
                            }));
                        }
                        // eprintln!("🔍 Linear: Output is NOT Metal, falling back to CPU");
                    },
                    _ => {
                        // eprintln!(
                        //     "🔍 Linear: Bias is NOT Metal (type={:?}), falling back to CPU",
                        //     std::mem::discriminant(bias)
                        // );
                    },
                }

                // Fallback: CPU bias addition
                // eprintln!("🔍 Linear: Using CPU bias fallback");
                output = output.to_device_enum(&crate::device::Device::CPU)?;
                // eprintln!("🔍 Linear: Converted output to CPU");
                output = output.add(bias)?;
                // eprintln!("🔍 Linear: Added bias on CPU");

                // Convert back to Metal tensor if needed
                if matches!(self.device, crate::device::Device::Metal(_)) {
                    // eprintln!("🔍 Linear: Converting back to Metal device");
                    output = output.to_device_enum(&self.device)?;
                    // eprintln!(
                    //     "🔍 Linear: Converted back to Metal, type={:?}",
                    //     std::mem::discriminant(&output)
                    // );
                }
            } else {
                // eprintln!("🔍 Linear: No bias");
            }

            // eprintln!(
            //     "🔍 Linear: Returning output, type={:?}",
            //     std::mem::discriminant(&output)
            // );
            return Ok(output);
        }

        // =====================================================================
        // GPU-TO-GPU PATH: Tensor::CUDA (ZERO CPU TRANSFERS!)
        // =====================================================================
        #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
        if let Tensor::CUDA(ref input_cuda) = input {
            use crate::gpu_ops::cuda::get_cuda_backend;
            use crate::tensor::CudaTensorData;

            // Ensure weight buffer is cached on GPU
            self.ensure_weight_buffer_cached_cuda()?;

            // Get cached weight buffer ID
            let weight_buffer_id = {
                let buffer_id_guard = self.weight_buffer_id_cuda.read().map_err(|_| {
                    TrustformersError::hardware_error(
                        "Failed to acquire read lock on CUDA buffer cache",
                        "Linear::forward",
                    )
                })?;

                if let Some(id) = *buffer_id_guard {
                    id
                } else {
                    // Weight not cached - fallback to CPU
                    let cpu_input = input.to_device_enum(&crate::device::Device::CPU)?;
                    return self.forward(cpu_input);
                }
            };

            // Get CUDA backend
            let device_id = if let Device::CUDA(id) = self.device {
                id
            } else {
                0 // Default to device 0
            };
            let backend = get_cuda_backend(device_id)?;

            // Extract input shape and calculate matmul dimensions
            let shape = &input_cuda.shape;
            let weight_shape = self.weight.shape();
            let in_features = shape[shape.len() - 1];

            // Check shape compatibility
            if in_features != weight_shape[1] {
                return Err(TrustformersError::shape_error(format!(
                    "Linear layer input features {} doesn't match weight shape {:?}",
                    in_features, weight_shape
                )));
            }

            // Flatten input to [batch, in_features] for matmul
            // Works for both 2D [seq_len, in_features] and 3D [batch, seq_len, in_features]
            let batch_dims: usize = shape[..shape.len() - 1].iter().product();
            let m = batch_dims; // number of rows in output
            let k = in_features; // shared dimension
            let n = self.weight.shape()[0]; // out_features

            // Perform GPU-to-GPU matmul (ZERO CPU TRANSFERS!)
            let output_buffer_id =
                backend.matmul_gpu_to_gpu(&input_cuda.buffer_id, &weight_buffer_id, m, k, n)?;

            // Calculate output shape (preserve batch dimensions, change last dim)
            let mut output_shape = shape[..shape.len() - 1].to_vec();
            output_shape.push(n);

            // Create output CUDA tensor
            let mut output = Tensor::CUDA(CudaTensorData {
                buffer_id: output_buffer_id,
                shape: output_shape.clone(),
                dtype: input_cuda.dtype,
            });

            // Handle bias if present
            if let Some(ref bias) = self.bias {
                // Try GPU-to-GPU bias addition if bias is on GPU
                match bias {
                    #[cfg(feature = "cuda")]
                    Tensor::CUDA(bias_data) => {
                        // Both output and bias are CUDA tensors - use GPU kernel!
                        if let Tensor::CUDA(output_data) = &output {
                            let output_buffer_id = backend.add_bias_gpu_to_gpu(
                                &output_data.buffer_id,
                                &bias_data.buffer_id,
                                batch_dims,
                                n,
                            )?;

                            return Ok(Tensor::CUDA(CudaTensorData {
                                buffer_id: output_buffer_id,
                                shape: output_shape.clone(),
                                dtype: output_data.dtype,
                            }));
                        }
                    },
                    _ => {},
                }

                // Fallback: CPU bias addition
                output = output.to_device_enum(&crate::device::Device::CPU)?;
                output = output.add(bias)?;

                // Convert back to CUDA tensor if needed
                if matches!(self.device, crate::device::Device::CUDA(_)) {
                    output = output.to_device_enum(&self.device)?;
                }
            }

            return Ok(output);
        }

        // =====================================================================
        // CPU/F32 PATH (existing implementation)
        // =====================================================================
        // Handle different input shapes for matmul
        let input_shape = input.shape();
        let weight_t = self.weight.transpose(0, 1)?;

        let output = if input_shape.len() == 2 {
            // Standard 2D input: [seq_len, hidden_size] x [hidden_size, out_features]

            // Try to use cached Metal buffer if available (ZERO-COPY OPTIMIZATION)
            #[cfg(all(target_os = "macos", feature = "metal"))]
            if matches!(self.device, Device::Metal(_)) {
                // Ensure buffer is cached
                self.ensure_weight_buffer_cached()?;

                // Try to use cached buffer
                if let Ok(buffer_id_guard) = self.weight_buffer_id.read() {
                    if let Some(buffer_id) = *buffer_id_guard {
                        // We have a cached buffer! Use it for ZERO-COPY matmul
                        use crate::gpu_ops::metal::get_metal_backend;

                        if let (Tensor::F32(inp), Tensor::F32(w_t)) = (&input, &weight_t) {
                            if inp.ndim() == 2 && w_t.ndim() == 2 {
                                let inp_shape = inp.shape();
                                let w_shape = w_t.shape();
                                let m = inp_shape[0];
                                let k = inp_shape[1];
                                let k2 = w_shape[0];
                                let n = w_shape[1];

                                if k == k2 {
                                    // Get Metal backend
                                    if let Ok(backend) = get_metal_backend() {
                                        // Convert input to contiguous
                                        let input_data: Vec<f32> = inp.iter().copied().collect();

                                        // Call matmul with CACHED weight buffer (no weight transfer!)
                                        if let Ok(result) = backend.matmul_with_cached_weight(
                                            &input_data,
                                            &buffer_id,
                                            m,
                                            k,
                                            n,
                                        ) {
                                            let result_arr =
                                                scirs2_core::ndarray::Array2::from_shape_vec(
                                                    (m, n),
                                                    result,
                                                )
                                                .map_err(|e| {
                                                    TrustformersError::shape_error(format!(
                                                        "Result reshape failed: {}",
                                                        e
                                                    ))
                                                })?;

                                            // Add bias if present
                                            let mut output = Tensor::F32(result_arr.into_dyn());
                                            if let Some(ref bias) = self.bias {
                                                output = output.add(bias)?;
                                            }
                                            return Ok(output);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Fallback: Use standard dispatch (for non-Metal or if cached path failed)
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                if self.device.is_gpu() {
                    dispatch_matmul(&input, &weight_t, &self.device)?
                } else {
                    input.matmul(&weight_t)?
                }
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                input.matmul(&weight_t)?
            }
        } else if input_shape.len() == 3 {
            // Batched 3D input: [batch, seq_len, hidden_size] x [hidden_size, out_features]
            // Handle manually since tensor.matmul doesn't support 3D x 2D
            match (&input, &weight_t) {
                (Tensor::F32(inp), Tensor::F32(w)) => {
                    let batch = input_shape[0];
                    let seq_len = input_shape[1];
                    let hidden = input_shape[2];
                    let out_features = w.shape()[1];

                    // Try to use cached Metal buffer for 3D case (CRITICAL OPTIMIZATION)
                    #[cfg(all(target_os = "macos", feature = "metal"))]
                    if matches!(self.device, Device::Metal(_)) {
                        // Ensure buffer is cached
                        self.ensure_weight_buffer_cached()?;

                        if let Ok(buffer_id_guard) = self.weight_buffer_id.read() {
                            if let Some(buffer_id) = *buffer_id_guard {
                                // Reshape 3D input to 2D for matmul
                                let m = batch * seq_len;
                                let k = hidden;
                                let n = out_features;

                                // Get Metal backend
                                use crate::gpu_ops::metal::get_metal_backend;
                                if let Ok(backend) = get_metal_backend() {
                                    // Convert input to contiguous 2D data
                                    let input_data: Vec<f32> = inp.iter().copied().collect();

                                    // Call matmul with CACHED weight buffer (no weight transfer!)
                                    if let Ok(result) = backend.matmul_with_cached_weight(
                                        &input_data,
                                        &buffer_id,
                                        m,
                                        k,
                                        n,
                                    ) {
                                        // Reshape result from 2D [m, n] back to 3D [batch, seq_len, out_features]
                                        let result_arr =
                                            scirs2_core::ndarray::Array2::from_shape_vec(
                                                (m, n),
                                                result,
                                            )
                                            .map_err(
                                                |e| {
                                                    TrustformersError::shape_error(format!(
                                                        "Result reshape failed: {}",
                                                        e
                                                    ))
                                                },
                                            )?;

                                        let result_3d = result_arr
                                            .into_shape_with_order(IxDyn(&[
                                                batch,
                                                seq_len,
                                                out_features,
                                            ]))
                                            .map_err(|e| {
                                                TrustformersError::shape_error(format!(
                                                    "3D reshape failed: {}",
                                                    e
                                                ))
                                            })?;

                                        // Add bias if present
                                        let mut output = Tensor::F32(result_3d);
                                        if let Some(ref bias) = self.bias {
                                            output = output.add(bias)?;
                                        }
                                        return Ok(output);
                                    }
                                }
                            }
                        }
                    }

                    // Fallback: CPU path with ndarray

                    // Ensure contiguous layout before reshaping input to 2D for dot product
                    let inp_contiguous = inp.to_owned();
                    let inp_2d = inp_contiguous
                        .into_shape_with_order([batch * seq_len, hidden])
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to reshape input: {}",
                                e
                            ))
                        })?;

                    // Ensure contiguous layout for weight and convert to 2D for GEMM
                    let w_contiguous = w.to_owned();
                    let w_2d = w_contiguous.into_dimensionality::<Ix2>().map_err(|e| {
                        TrustformersError::shape_error(format!(
                            "Failed to convert weight to 2D: {}",
                            e
                        ))
                    })?;

                    // Use BLAS-accelerated GEMM via scirs2-core for larger matrices
                    let m = inp_2d.nrows();
                    let n = w_2d.ncols();
                    let k = inp_2d.ncols();
                    // Note: scirs2-core has bugs with matrices of size <64, use 64 as threshold
                    const MIN_SIZE_FOR_SIMD_GEMM: usize = 64;
                    let out_2d = if m < MIN_SIZE_FOR_SIMD_GEMM
                        || n < MIN_SIZE_FOR_SIMD_GEMM
                        || k < MIN_SIZE_FOR_SIMD_GEMM
                    {
                        inp_2d.dot(&w_2d)
                    } else {
                        let mut res = Array2::<f32>::zeros((m, n));
                        f32::simd_gemm(1.0, &inp_2d.view(), &w_2d.view(), 0.0, &mut res);
                        res
                    };

                    // Reshape back to 3D
                    let out_3d = out_2d
                        .into_shape_with_order(IxDyn(&[batch, seq_len, out_features]))
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to reshape output: {}",
                                e
                            ))
                        })?;

                    Tensor::F32(out_3d)
                },
                (Tensor::F64(inp), Tensor::F64(w)) => {
                    let batch = input_shape[0];
                    let seq_len = input_shape[1];
                    let hidden = input_shape[2];
                    let out_features = w.shape()[1];

                    // Ensure contiguous layout before reshaping
                    let inp_contiguous = inp.to_owned();
                    let inp_2d = inp_contiguous
                        .into_shape_with_order([batch * seq_len, hidden])
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to reshape input: {}",
                                e
                            ))
                        })?;

                    // Ensure contiguous layout for weight and convert to 2D for GEMM
                    let w_contiguous = w.to_owned();
                    let w_2d = w_contiguous.into_dimensionality::<Ix2>().map_err(|e| {
                        TrustformersError::shape_error(format!(
                            "Failed to convert weight to 2D: {}",
                            e
                        ))
                    })?;

                    // Use BLAS-accelerated GEMM via scirs2-core for larger matrices
                    let m = inp_2d.nrows();
                    let n = w_2d.ncols();
                    let k = inp_2d.ncols();
                    // Note: scirs2-core has bugs with matrices of size <64, use 64 as threshold
                    const MIN_SIZE_FOR_SIMD_GEMM: usize = 64;
                    let out_2d = if m < MIN_SIZE_FOR_SIMD_GEMM
                        || n < MIN_SIZE_FOR_SIMD_GEMM
                        || k < MIN_SIZE_FOR_SIMD_GEMM
                    {
                        inp_2d.dot(&w_2d)
                    } else {
                        let mut res = Array2::<f64>::zeros((m, n));
                        f64::simd_gemm(1.0, &inp_2d.view(), &w_2d.view(), 0.0, &mut res);
                        res
                    };

                    let out_3d = out_2d
                        .into_shape_with_order(IxDyn(&[batch, seq_len, out_features]))
                        .map_err(|e| {
                            TrustformersError::shape_error(format!(
                                "Failed to reshape output: {}",
                                e
                            ))
                        })?;

                    Tensor::F64(out_3d)
                },
                _ => {
                    return Err(TrustformersError::tensor_op_error(
                        "Unsupported tensor types for 3D linear layer",
                        "Linear::forward",
                    ))
                },
            }
        } else {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Linear layer doesn't support input with {} dimensions",
                    input_shape.len()
                ),
                "Linear::forward",
            ));
        };

        if let Some(ref bias) = self.bias {
            // Handle broadcasting for bias addition
            match (&output, bias) {
                (Tensor::F32(out_arr), Tensor::F32(bias_arr)) => {
                    // Broadcast bias to match output shape
                    let result = out_arr + bias_arr;
                    Ok(Tensor::F32(result))
                },
                _ => output.add(bias),
            }
        } else {
            Ok(output)
        }
    }
}
