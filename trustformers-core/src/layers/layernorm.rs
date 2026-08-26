//! Layer Normalization implementations
//!
//! This module provides LayerNorm and RMSNorm layers with device support.

use crate::device::Device;
use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use crate::traits::Layer;
use scirs2_core::ndarray::{Array2, ArrayD, ArrayView1, Axis, IxDyn};
use scirs2_core::simd::reductions::{simd_mean_f32, simd_variance_f32};
use scirs2_core::simd_ops::SimdUnifiedOps;

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
        // reason: `Self::new` only fails for an empty normalized-shape, which a
        // single-dimension `vec![normalized_shape]` can never be; this public
        // `-> Self` constructor has no fallible alternative without an API break.
        #[allow(clippy::expect_used)]
        Self::new(vec![normalized_shape], eps)
            .expect("LayerNorm::new should not fail with valid shape")
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

    /// Returns a reference to the elementwise scale (`gamma`).
    ///
    /// Spelled `weight` to match the HuggingFace / PyTorch checkpoint key
    /// `*.LayerNorm.weight`, so it can be exposed directly through
    /// [`Model::named_tensors`](crate::traits::Model::named_tensors).
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns a reference to the elementwise shift (`beta`).
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// Returns a mutable reference to the elementwise scale.
    ///
    /// `LayerNorm` caches nothing derived from its parameters, so no
    /// invalidation is needed here.
    pub fn weight_mut(&mut self) -> &mut Tensor {
        &mut self.weight
    }

    /// Returns a mutable reference to the elementwise shift.
    pub fn bias_mut(&mut self) -> &mut Tensor {
        &mut self.bias
    }

    /// Append this layer's parameters to `into` under `<prefix>.weight` /
    /// `<prefix>.bias` — the names PyTorch's `nn.LayerNorm` uses.
    ///
    /// See [`crate::layers::Linear::collect_named_parameters`] for the rationale.
    pub fn collect_named_parameters<'a>(
        &'a self,
        prefix: &str,
        into: &mut Vec<(String, &'a Tensor)>,
    ) {
        into.push((format!("{prefix}.weight"), &self.weight));
        into.push((format!("{prefix}.bias"), &self.bias));
    }

    /// Mutable counterpart of [`LayerNorm::collect_named_parameters`].
    pub fn collect_named_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        let (weight, bias) = self.parameters_mut();
        into.push((format!("{prefix}.weight"), weight));
        into.push((format!("{prefix}.bias"), bias));
    }

    /// Borrow the scale **and** the shift mutably at the same time.
    ///
    /// Two successive `*_mut()` calls each borrow all of `*self`, so only a
    /// disjoint-field borrow inside this impl can hand out both at once — which
    /// is what building a `Vec<(String, &mut Tensor)>` for
    /// [`Model::named_tensors_mut`](crate::traits::Model::named_tensors_mut)
    /// requires.
    pub fn parameters_mut(&mut self) -> (&mut Tensor, &mut Tensor) {
        (&mut self.weight, &mut self.bias)
    }

    /// The shape the trailing dimensions are normalised over.
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// The epsilon added to the variance for numerical stability.
    pub fn eps(&self) -> f32 {
        self.eps
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
    #[cfg(all(target_os = "macos", feature = "metal"))]
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
            #[cfg(all(target_os = "macos", feature = "metal"))]
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
                            tracing::trace!(
                                shape = ?metal_data.shape,
                                "LayerNorm: GPU-to-GPU path (Metal input, Metal parameters)"
                            );

                            if metal_data.shape.len() == 2 {
                                // 2D case: (seq_len, hidden_size)
                                let seq_len = metal_data.shape[0];

                                let output_buffer_id = backend.layernorm_gpu_to_gpu(
                                    &metal_data.buffer_id(),
                                    &w_data.buffer_id(),
                                    &b_data.buffer_id(),
                                    seq_len,
                                    hidden_size,
                                    self.eps,
                                )?;

                                return Ok(Tensor::Metal(MetalTensorData::new(
                                    &backend,
                                    output_buffer_id,
                                    metal_data.shape.clone(),
                                    metal_data.dtype,
                                )?));
                            } else if metal_data.shape.len() == 3 {
                                // 3D case: (batch, seq_len, hidden_size)
                                // Flatten to 2D: (batch * seq_len, hidden_size)
                                let batch = metal_data.shape[0];
                                let seq_len = metal_data.shape[1];
                                let flattened_seq_len = batch * seq_len;

                                tracing::trace!(
                                    from = ?metal_data.shape,
                                    rows = flattened_seq_len,
                                    hidden_size,
                                    "LayerNorm: flattening 3-D input to 2-D for the GPU kernel"
                                );

                                // Run GPU kernel on flattened 2D tensor
                                let output_buffer_id = backend.layernorm_gpu_to_gpu(
                                    &metal_data.buffer_id(),
                                    &w_data.buffer_id(),
                                    &b_data.buffer_id(),
                                    flattened_seq_len,
                                    hidden_size,
                                    self.eps,
                                )?;

                                // Return with original 3D shape
                                return Ok(Tensor::Metal(MetalTensorData::new(
                                    &backend,
                                    output_buffer_id,
                                    metal_data.shape.clone(),
                                    metal_data.dtype,
                                )?));
                            }
                        },
                        _ => {
                            tracing::trace!(
                                "LayerNorm: parameters are not GPU-resident, \
                                 falling back to the CPU kernel"
                            );
                        },
                    }
                } else {
                    tracing::trace!(
                        shape = ?metal_data.shape,
                        normalized_shape = ?self.normalized_shape,
                        "LayerNorm: shape has no GPU kernel, falling back to the CPU kernel"
                    );
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

                // Process on CPU directly (inline to avoid recursion).
                // Fused single-pass normalisation (see `layer_norm_f32_cpu`).
                let output = layer_norm_f32_cpu(
                    &input_arr,
                    &weight_arr,
                    &bias_arr,
                    &self.normalized_shape,
                    self.eps,
                )?;
                Ok(Tensor::F32(output))
            },

            // GPU-resident CUDA tensor - process on GPU
            #[cfg(feature = "cuda")]
            #[allow(unused_variables)]
            Tensor::CUDA(cuda_data) => {
                #[allow(unused_imports)]
                use crate::tensor::CudaTensorData;

                #[cfg(any(target_os = "linux", target_os = "windows"))]
                {
                    use crate::gpu_ops::cuda::get_cuda_backend;

                    if cuda_data.shape.len() == 2 && self.normalized_shape.len() == 1 {
                        // Address the device the resident input actually lives on (the
                        // buffer handle carries the ordinal) instead of the layer's
                        // configured device.
                        let device_id = cuda_data.device_id();
                        let backend = get_cuda_backend(device_id)?;
                        let shape = &cuda_data.shape;
                        let seq_len = shape[0];
                        let hidden_size = shape[1];

                        if hidden_size == self.normalized_shape[0] {
                            // Get weight and bias buffer IDs; they must be resident on
                            // the *same* device as the input for a GPU-to-GPU kernel.
                            match (&self.weight, &self.bias) {
                                (Tensor::CUDA(w_data), Tensor::CUDA(b_data))
                                    if w_data.device_id() == device_id
                                        && b_data.device_id() == device_id =>
                                {
                                    // All on the same GPU - zero transfers!
                                    let output_buffer_id = backend.layernorm_gpu_to_gpu(
                                        &cuda_data.buffer_id(),
                                        &w_data.buffer_id(),
                                        &b_data.buffer_id(),
                                        seq_len,
                                        hidden_size,
                                        self.eps,
                                    )?;

                                    return Ok(Tensor::CUDA(CudaTensorData::new(
                                        output_buffer_id,
                                        device_id,
                                        cuda_data.shape.clone(),
                                        cuda_data.dtype,
                                    )));
                                },
                                _ => {
                                    // Weight/bias not resident on this GPU - fallback to CPU
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

                // Process on CPU directly (inline to avoid recursion).
                // Fused single-pass normalisation (see `layer_norm_f32_cpu`).
                let output = layer_norm_f32_cpu(
                    &input_arr,
                    &weight_arr,
                    &bias_arr,
                    &self.normalized_shape,
                    self.eps,
                )?;
                Ok(Tensor::F32(output))
            },

            Tensor::F32(arr) => {
                // Try Metal GPU acceleration for 2D tensors (seq_len, hidden_size)
                #[cfg(all(target_os = "macos", feature = "metal"))]
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

                                        // Upload input to GPU. This buffer is
                                        // upload-only scratch: nothing else ever
                                        // references it, so it must be released once
                                        // `layernorm_gpu_to_gpu` has consumed it below
                                        // or it leaks as an unreleased `Pinned` cache
                                        // entry for the rest of the process.
                                        let input_vec: Vec<f32> = arr.iter().copied().collect();
                                        let input_buffer_id =
                                            backend.create_persistent_buffer(&input_vec)?;

                                        // Execute GPU-to-GPU. Capture the result rather
                                        // than propagating with `?` immediately: the
                                        // upload above must be released regardless of
                                        // whether this call succeeds or fails, mirroring
                                        // how `attention_gpu_to_gpu` /
                                        // `attention_with_cache_gpu_to_gpu` release their
                                        // own scratch "regardless of where it failed".
                                        let layernorm_result = backend.layernorm_gpu_to_gpu(
                                            &input_buffer_id,
                                            &w_data.buffer_id(),
                                            &b_data.buffer_id(),
                                            seq_len,
                                            hidden_size,
                                            self.eps,
                                        );
                                        backend.release_buffers(&[input_buffer_id])?;
                                        let output_buffer_id = layernorm_result?;

                                        // Return Metal tensor
                                        return Ok(Tensor::Metal(MetalTensorData::new(
                                            &backend,
                                            output_buffer_id,
                                            arr.shape().to_vec(),
                                            crate::tensor::DType::F32,
                                        )?));
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

                // Every CPU shape is normalised by `layer_norm_f32_cpu`, which
                // derives the normalised axes from `self.normalized_shape` itself
                // and vectorises its own reductions. There is deliberately no
                // separate "SIMD fast path" here any more: the one that used to
                // live at this spot called `scirs2_core::simd::normalization::
                // simd_layer_norm_f32`, whose variance is the *sample* (n-1)
                // estimator, while this function - and the Metal kernels - use the
                // *population* (n) estimator that `torch.nn.LayerNorm` defines. A
                // tensor therefore normalised differently depending on its element
                // count (>= 64 took the SIMD branch) and, on `metal` builds, on its
                // rank (2-D was diverted to the GPU above, 3-D was not).

                // Convert weight/bias to F32 if needed
                let weight_f32 = match &self.weight {
                    Tensor::F32(w) => w.clone(),
                    #[cfg(all(target_os = "macos", feature = "metal"))]
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
                    #[cfg(all(target_os = "macos", feature = "metal"))]
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

                // Fused single-pass normalisation (see `layer_norm_f32_cpu`).
                let output = layer_norm_f32_cpu(
                    arr,
                    &weight_f32,
                    &bias_f32,
                    &self.normalized_shape,
                    self.eps,
                )?;
                Ok(Tensor::F32(output))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for LayerNorm",
                "LayerNorm::forward",
            )),
        }
    }
}

/// Reduction length at which the vectorised mean/variance start paying for
/// their call overhead.
///
/// The threshold is on the length of **one normalised group**
/// (`normalized_shape.iter().product()`), not on the whole tensor: only the
/// per-group reduction is vectorised, so a `[32, 8]` tensor (256 elements, but
/// groups of 8) has nothing to gain. The predecessor of this code gated on the
/// total element count instead, which is how `[32, 8]` ended up on a different
/// code path - and, back when the two paths disagreed about the estimator, a
/// different answer - from `[2, 8]`.
const MIN_NORM_LEN_FOR_SIMD: usize = 64;

/// Fused CPU LayerNorm over the trailing `normalized_shape` axes.
///
/// Computes `y = gamma * (x - mean) / sqrt(var + eps) + beta` in a **single**
/// streaming pass per normalized group, writing into one output buffer.
///
/// The three call sites in `LayerNorm::forward` previously each ran the same
/// five-array recipe: `input.clone()` purely to seed a reduction accumulator
/// (its contents discarded by the very next statement), `&input - &mean`,
/// `(&diff * &diff).to_owned()` (an extra copy of an already-owned product),
/// `&diff / (var + eps).sqrt()`, and finally the broadcast scale/shift. For a
/// `[1, 2048, 4096]` hidden state that is ~160 MiB of temporaries per call.
///
/// # Variance estimator
///
/// `var` is the **population** variance - the sum of squared deviations divided
/// by `n`, not by `n - 1`. That is what `torch.nn.LayerNorm` (and therefore every
/// HuggingFace checkpoint) is trained against, and what the Metal `layernorm`
/// kernel computes, so it is the single estimator used on every device and every
/// shape. Reductions are vectorised through `scirs2-core` once a normalised group
/// is long enough to pay for the call (see [`MIN_NORM_LEN_FOR_SIMD`]); the
/// estimator does not change with that choice.
fn layer_norm_f32_cpu(
    input: &ArrayD<f32>,
    weight: &ArrayD<f32>,
    bias: &ArrayD<f32>,
    normalized_shape: &[usize],
    eps: f32,
) -> Result<ArrayD<f32>> {
    let input_shape = input.shape();
    let norm_ndim = normalized_shape.len();
    if norm_ndim == 0 || norm_ndim > input_shape.len() {
        return Err(TrustformersError::shape_error(format!(
            "LayerNorm normalized_shape {:?} is not a suffix of input shape {:?}",
            normalized_shape, input_shape
        )));
    }
    let split = input_shape.len() - norm_ndim;
    if input_shape[split..] != *normalized_shape {
        return Err(TrustformersError::shape_error(format!(
            "LayerNorm normalized_shape {:?} is not a suffix of input shape {:?}",
            normalized_shape, input_shape
        )));
    }

    let norm_len: usize = normalized_shape.iter().product();
    if norm_len == 0 {
        return Err(TrustformersError::shape_error(
            "LayerNorm normalized_shape must not contain a zero dimension".to_string(),
        ));
    }
    if weight.len() != norm_len || bias.len() != norm_len {
        return Err(TrustformersError::shape_error(format!(
            "LayerNorm weight ({}) and bias ({}) must both hold {} values",
            weight.len(),
            bias.len(),
            norm_len
        )));
    }

    let weight_contiguous = weight.as_standard_layout();
    let bias_contiguous = bias.as_standard_layout();
    let weight_slice = weight_contiguous.as_slice().ok_or_else(|| {
        TrustformersError::tensor_op_error(
            "LayerNorm weight is not contiguous",
            "LayerNorm::forward",
        )
    })?;
    let bias_slice = bias_contiguous.as_slice().ok_or_else(|| {
        TrustformersError::tensor_op_error("LayerNorm bias is not contiguous", "LayerNorm::forward")
    })?;

    let mut output = input.as_standard_layout().into_owned();
    {
        let values = output.as_slice_mut().ok_or_else(|| {
            TrustformersError::tensor_op_error(
                "LayerNorm output buffer is not contiguous",
                "LayerNorm::forward",
            )
        })?;

        let inverse_count = 1.0f32 / norm_len as f32;
        // `simd_variance_f32` is Bessel-corrected (divides by n-1); the population
        // variance this layer must use is that value scaled by (n-1)/n. Hoisted out
        // of the loop because it depends only on `norm_len`.
        let sample_to_population = (norm_len as f32 - 1.0) * inverse_count;
        let vectorise = norm_len >= MIN_NORM_LEN_FOR_SIMD;

        for group in values.chunks_mut(norm_len) {
            let (mean, variance) = if vectorise {
                // Borrowed immutably only for the two reductions; the view is dead
                // before `group` is written below.
                let view = ArrayView1::from(&group[..]);
                let mean = simd_mean_f32(&view);
                // `simd_variance_f32` re-derives its centre with the very same
                // `simd_mean_f32` call, so `mean` here *is* the point the squared
                // deviations were taken about - the two stay consistent.
                // `norm_len >= MIN_NORM_LEN_FOR_SIMD >= 2`, so its `len < 2 => NaN`
                // branch is unreachable.
                (mean, simd_variance_f32(&view) * sample_to_population)
            } else {
                let mean = group.iter().sum::<f32>() * inverse_count;
                let variance =
                    group.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() * inverse_count;
                (mean, variance)
            };
            let inverse_std = 1.0 / (variance + eps).sqrt();

            for ((value, &gamma), &beta) in
                group.iter_mut().zip(weight_slice.iter()).zip(bias_slice.iter())
            {
                *value = (*value - mean) * inverse_std * gamma + beta;
            }
        }
    }

    Ok(output)
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

    /// Returns a reference to the elementwise scale.
    ///
    /// RMSNorm has no shift term, so there is deliberately no `bias()`.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns a mutable reference to the elementwise scale.
    pub fn weight_mut(&mut self) -> &mut Tensor {
        &mut self.weight
    }

    /// Append this layer's single parameter to `into` under `<prefix>.weight`.
    ///
    /// RMSNorm has no shift term, so exactly one entry is produced.
    pub fn collect_named_parameters<'a>(
        &'a self,
        prefix: &str,
        into: &mut Vec<(String, &'a Tensor)>,
    ) {
        into.push((format!("{prefix}.weight"), &self.weight));
    }

    /// Mutable counterpart of [`RMSNorm::collect_named_parameters`].
    pub fn collect_named_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        into.push((format!("{prefix}.weight"), &mut self.weight));
    }

    /// The epsilon added inside the reciprocal square root.
    pub fn eps(&self) -> f32 {
        self.eps
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
                let hidden_size = arr.shape()[last_dim];

                // Minimum size threshold for SIMD (avoid overhead on small tensors)
                const MIN_SIZE_FOR_SIMD_RMS_NORM: usize = 64;

                // SIMD-accelerated path for 2D/3D tensors
                if (ndim == 2 || ndim == 3) && arr.len() >= MIN_SIZE_FOR_SIMD_RMS_NORM {
                    // Extract weight as 1D array (owned copy for proper lifetime)
                    let weight_1d = match &self.weight {
                        Tensor::F32(w) => {
                            if w.len() == hidden_size {
                                use scirs2_core::ndarray::Array1;
                                let data: Vec<f32> = w.iter().copied().collect();
                                Array1::from_vec(data).into_shape_with_order(hidden_size).ok()
                            } else {
                                None
                            }
                        },
                        _ => None,
                    };

                    if let Some(w) = weight_1d {
                        // Reshape to 2D: (batch_size, hidden_size)
                        let original_shape = arr.shape().to_vec();
                        let batch_size = arr.len() / hidden_size;

                        if let Ok(input_2d) = arr
                            .as_standard_layout()
                            .view()
                            .into_shape_with_order((batch_size, hidden_size))
                        {
                            // Compute RMS using SIMD for each row
                            let mut output_2d = Array2::<f32>::zeros((batch_size, hidden_size));

                            for i in 0..batch_size {
                                let row = input_2d.row(i);

                                // Sum of squares, vectorised through scirs2-core
                                // (`simd_dot` of the row with itself) so the claim
                                // this comment makes is the code that runs: the
                                // predecessor here was a plain scalar accumulation
                                // labelled "using SIMD".
                                let squares_sum = f32::simd_dot(&row, &row);
                                let mean_sq = squares_sum / (hidden_size as f32);
                                let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();

                                // Normalize and apply weight in one pass
                                for j in 0..hidden_size {
                                    output_2d[[i, j]] = row[j] * inv_rms * w[j];
                                }
                            }

                            // Reshape back to original shape
                            if let Ok(output) =
                                output_2d.into_shape_with_order(IxDyn(&original_shape))
                            {
                                return Ok(Tensor::F32(output));
                            }
                        }
                    }
                }

                // Fallback to standard implementation
                // Compute RMS: sqrt(mean(x^2))
                let squares = arr.mapv(|x| x * x);
                let mean_squares = squares
                    .mean_axis(Axis(last_dim))
                    .ok_or_else(|| {
                        crate::errors::compute_error("forward", "last_dim must be valid axis")
                    })?
                    .insert_axis(Axis(last_dim));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::Result;

    #[test]
    fn test_layernorm_creation() -> Result<()> {
        let ln = LayerNorm::new(vec![64], 1e-5)?;
        assert_eq!(ln.device(), Device::CPU);
        Ok(())
    }

    #[test]
    fn test_layernorm_with_device() -> Result<()> {
        let ln = LayerNorm::new_with_device(vec![32], 1e-5, Device::CPU)?;
        assert_eq!(ln.device(), Device::CPU);
        Ok(())
    }

    #[test]
    fn test_layernorm_simple() {
        let ln = LayerNorm::new_simple(128, 1e-5);
        assert_eq!(ln.device(), Device::CPU);
    }

    #[test]
    fn test_layernorm_parameter_count() -> Result<()> {
        let ln = LayerNorm::new(vec![64], 1e-5)?;
        // weight: 64 + bias: 64 = 128
        assert_eq!(ln.parameter_count(), 128);
        Ok(())
    }

    #[test]
    fn test_layernorm_set_weight() -> Result<()> {
        let mut ln = LayerNorm::new(vec![4], 1e-5)?;
        let new_weight = Tensor::full_with_shape(&[4], 2.0)?;
        ln.set_weight(new_weight)?;
        Ok(())
    }

    #[test]
    fn test_layernorm_set_bias() -> Result<()> {
        let mut ln = LayerNorm::new(vec![4], 1e-5)?;
        let new_bias = Tensor::full_with_shape(&[4], 0.5)?;
        ln.set_bias(new_bias)?;
        Ok(())
    }

    #[test]
    fn test_layernorm_to_device() -> Result<()> {
        let ln = LayerNorm::new(vec![16], 1e-5)?;
        let ln2 = ln.to_device(Device::CPU);
        assert_eq!(ln2.device(), Device::CPU);
        Ok(())
    }

    #[test]
    fn test_layernorm_forward_2d() -> Result<()> {
        let ln = LayerNorm::new(vec![4], 1e-5)?;
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])?;
        let output = ln.forward(input)?;
        assert_eq!(output.shape(), vec![2, 4]);
        Ok(())
    }

    #[test]
    fn test_layernorm_forward_normalizes() -> Result<()> {
        let ln = LayerNorm::new(vec![4], 1e-5)?;
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[1, 4])?;
        let output = ln.forward(input)?;
        // After normalization, mean should be close to 0
        let mean = output.mean()?;
        let data = mean.data()?;
        assert!(data[0].abs() < 0.1);
        Ok(())
    }

    #[test]
    fn test_layernorm_constant_input() -> Result<()> {
        let ln = LayerNorm::new(vec![4], 1e-5)?;
        let input = Tensor::full_with_shape(&[1, 4], 5.0)?;
        let output = ln.forward(input)?;
        assert_eq!(output.shape(), vec![1, 4]);
        Ok(())
    }

    #[test]
    fn test_layernorm_3d_input() -> Result<()> {
        let ln = LayerNorm::new(vec![8], 1e-5)?;
        let input = Tensor::ones(&[2, 4, 8])?;
        let output = ln.forward(input)?;
        assert_eq!(output.shape(), vec![2, 4, 8]);
        Ok(())
    }

    #[test]
    fn test_rmsnorm_creation() -> Result<()> {
        let rms = RMSNorm::new(64, 1e-6)?;
        assert_eq!(rms.device(), Device::CPU);
        Ok(())
    }

    #[test]
    fn test_rmsnorm_with_device() -> Result<()> {
        let rms = RMSNorm::new_with_device(32, 1e-6, Device::CPU)?;
        assert_eq!(rms.device(), Device::CPU);
        Ok(())
    }

    #[test]
    fn test_rmsnorm_parameter_count() -> Result<()> {
        let rms = RMSNorm::new(64, 1e-6)?;
        // weight only: 64
        assert_eq!(rms.parameter_count(), 64);
        Ok(())
    }

    #[test]
    fn test_rmsnorm_set_weight() -> Result<()> {
        let mut rms = RMSNorm::new(4, 1e-6)?;
        let new_weight = Tensor::full_with_shape(&[4], 2.0)?;
        rms.set_weight(new_weight)?;
        Ok(())
    }

    #[test]
    fn test_rmsnorm_to_device() -> Result<()> {
        let rms = RMSNorm::new(16, 1e-6)?;
        let rms2 = rms.to_device(Device::CPU);
        assert_eq!(rms2.device(), Device::CPU);
        Ok(())
    }

    #[test]
    fn test_rmsnorm_forward_2d() -> Result<()> {
        let rms = RMSNorm::new(4, 1e-6)?;
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])?;
        let output = rms.forward(input)?;
        assert_eq!(output.shape(), vec![2, 4]);
        Ok(())
    }

    #[test]
    fn test_rmsnorm_forward_3d() -> Result<()> {
        let rms = RMSNorm::new(8, 1e-6)?;
        let input = Tensor::ones(&[2, 4, 8])?;
        let output = rms.forward(input)?;
        assert_eq!(output.shape(), vec![2, 4, 8]);
        Ok(())
    }

    #[test]
    fn test_layernorm_forward_large_batch() -> Result<()> {
        let ln = LayerNorm::new(vec![16], 1e-5)?;
        let input = Tensor::ones(&[32, 16])?;
        let output = ln.forward(input)?;
        assert_eq!(output.shape(), vec![32, 16]);
        Ok(())
    }

    #[test]
    fn test_rmsnorm_forward_ones() -> Result<()> {
        let rms = RMSNorm::new(4, 1e-6)?;
        let input = Tensor::ones(&[1, 4])?;
        let output = rms.forward(input)?;
        assert_eq!(output.shape(), vec![1, 4]);
        // For uniform input of 1s, RMS = 1, so output should be weight * 1 = 1
        let data = output.data()?;
        for val in &data {
            assert!((val - 1.0).abs() < 0.1);
        }
        Ok(())
    }

    /// Regression: the CPU-input / GPU-weight `LayerNorm::forward` fallback used to
    /// upload the input as a `Pinned` buffer via `create_persistent_buffer` and never
    /// release it, leaking one Metal cache entry per call.
    ///
    /// `layernorm_gpu_to_gpu` allocates exactly one new (`Live`-tier) output buffer,
    /// and that output's own `MetalBufferHandle` releases its cache entry the moment
    /// the returned tensor drops. So once the upload is *also* released, a single
    /// `forward()` call must leave the cache exactly where it started: any residual
    /// growth means something leaked.
    ///
    /// Relies on nextest's process-per-test isolation for the global Metal buffer
    /// cache singleton (`get_metal_backend()` clones handles to one process-wide
    /// `Arc<Mutex<BufferCache>>`); under a threaded single-process runner other
    /// concurrent Metal tests could shift the count for unrelated reasons.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_layernorm_forward_cpu_input_gpu_weight_releases_upload() -> Result<()> {
        use crate::gpu_ops::metal::get_metal_backend;

        let backend = match get_metal_backend() {
            Ok(backend) => backend,
            Err(_) => {
                eprintln!("no Metal device; skipping");
                return Ok(());
            },
        };

        let mut ln = LayerNorm::new(vec![4], 1e-5)?;
        ln.weights_to_gpu(&Device::Metal(0))?;

        let entries_before = backend.buffer_cache_stats()?.entries;

        {
            let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])?;
            let output = ln.forward(input)?;
            assert!(matches!(output, Tensor::Metal(_)));
            // `output`'s `MetalBufferHandle` drops here, releasing the output
            // buffer's own cache entry before the assertion below runs.
        }

        let entries_after = backend.buffer_cache_stats()?.entries;
        assert_eq!(
            entries_after, entries_before,
            "LayerNorm::forward leaked its uploaded input buffer (pre-fix this was \
             entries_before + 1: the input upload was never released)"
        );

        Ok(())
    }
    /// Textbook LayerNorm over the trailing axis using the **population**
    /// (divide-by-`n`) variance, accumulated in `f64` so it is an independent
    /// answer rather than a re-run of the code under test.
    ///
    /// This is what `torch.nn.LayerNorm` computes, and therefore what every
    /// HuggingFace checkpoint was trained against.
    fn population_layer_norm_reference(
        input: &[f32],
        width: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; input.len()];
        for (row_in, row_out) in input.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
            let mean = row_in.iter().map(|&v| f64::from(v)).sum::<f64>() / width as f64;
            let variance = row_in
                .iter()
                .map(|&v| {
                    let d = f64::from(v) - mean;
                    d * d
                })
                .sum::<f64>()
                / width as f64;
            let inverse_std = 1.0 / (variance + f64::from(eps)).sqrt();
            for ((o, &v), (&g, &b)) in
                row_out.iter_mut().zip(row_in.iter()).zip(gamma.iter().zip(beta.iter()))
            {
                *o = (((f64::from(v) - mean) * inverse_std) * f64::from(g) + f64::from(b)) as f32;
            }
        }
        out
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "compared buffers must have equal length");
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    /// Deterministic, non-symmetric sample data. A pure `sin` sweep has a mean
    /// near zero, which makes an estimator bug smaller than it really is, so the
    /// series is offset and mildly skewed.
    fn sample_values(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| {
                let t = i as f32 * 0.137;
                t.sin() * 1.7 + (t * 0.31).cos() * 0.6 + 0.45
            })
            .collect()
    }

    fn ramp_gamma_beta(width: usize) -> (Vec<f32>, Vec<f32>) {
        let gamma = (0..width).map(|j| 0.8 + j as f32 * 0.003).collect();
        let beta = (0..width).map(|j| -0.2 + j as f32 * 0.001).collect();
        (gamma, beta)
    }

    fn layer_norm_with_params(width: usize, gamma: &[f32], beta: &[f32]) -> Result<LayerNorm> {
        let mut ln = LayerNorm::new(vec![width], 1e-5)?;
        ln.set_weight(Tensor::from_data(gamma.to_vec(), &[width])?)?;
        ln.set_bias(Tensor::from_data(beta.to_vec(), &[width])?)?;
        Ok(ln)
    }

    /// Every (size, rank) combination must produce the population-variance
    /// answer.
    ///
    /// Regression for a split estimator: 2-D/3-D tensors of 64 elements or more
    /// used to be routed to `scirs2_core::simd::normalization::
    /// simd_layer_norm_f32`, whose variance divides by `n - 1`, while everything
    /// else divided by `n`. The same layer therefore answered differently
    /// depending on how many elements its input happened to contain - by a factor
    /// of `1/sqrt(1 - 1/n)`, i.e. 1.6% at `n = 32` and still 0.07% at `n = 768`,
    /// on top of every checkpoint being normalised wrong.
    #[test]
    fn test_layer_norm_matches_population_reference_across_sizes_and_ranks() -> Result<()> {
        for &width in &[8usize, 32, 64, 256, 768] {
            let (gamma, beta) = ramp_gamma_beta(width);
            let ln = layer_norm_with_params(width, &gamma, &beta)?;

            // Rank 2 with a single row is the "one 1-D vector" case; rank 2 with
            // several rows and rank 3 exercise the multi-group loop. `[1, w]` and
            // `[3, w]` also straddle the vectorisation threshold for w = 8 and
            // w = 32 in opposite directions of total element count, which is what
            // the old code keyed off.
            let shapes: [Vec<usize>; 3] = [vec![1, width], vec![3, width], vec![2, 3, width]];

            for shape in shapes {
                let total: usize = shape.iter().product();
                let values = sample_values(total);
                let expected = population_layer_norm_reference(&values, width, &gamma, &beta, 1e-5);

                let output = ln.forward(Tensor::from_data(values, &shape)?)?;
                assert_eq!(output.shape(), shape, "shape must be preserved");
                let actual = output.to_vec_f32()?;

                let diff = max_abs_diff(&actual, &expected);
                assert!(
                    diff < 1e-5,
                    "width={width} shape={shape:?}: max|actual - population_reference| = \
                     {diff:e} (>= 1e-5). A divide-by-(n-1) variance shows up here as a \
                     uniform 1/sqrt(1 - 1/n) scale error."
                );
            }
        }
        Ok(())
    }

    /// The estimator itself, pinned on the **scalar** reduction branch.
    ///
    /// For `[1.0, 3.0]` the mean is 2 and the two estimators are maximally far
    /// apart: population variance 1 (output `±1`), sample variance 2 (output
    /// `±1/sqrt(2)` = `±0.7071`). Exactly a factor of `sqrt(2)`.
    #[test]
    fn test_layer_norm_variance_divisor_is_n_scalar_branch() -> Result<()> {
        let ln = LayerNorm::new(vec![2], 0.0)?;
        let output = ln.forward(Tensor::from_data(vec![1.0, 3.0], &[1, 2])?)?;
        let actual = output.to_vec_f32()?;

        assert!(
            (actual[0] + 1.0).abs() < 1e-4 && (actual[1] - 1.0).abs() < 1e-4,
            "expected the population (divide-by-n) answer [-1, 1], got {actual:?}; \
             [-0.7071, 0.7071] would mean the sample (divide-by-n-1) variance"
        );
        Ok(())
    }

    /// The same estimator pin on the **vectorised** reduction branch, which is
    /// the one that actually carried the bug.
    ///
    /// The `const` assertion below is the guard that keeps it that way: if
    /// [`MIN_NORM_LEN_FOR_SIMD`] ever rises above `WIDTH`, this test stops
    /// compiling rather than silently pinning the scalar branch twice.
    ///
    /// A row of 32 `-1`s followed by 32 `+1`s has mean 0 and population variance
    /// exactly 1, so the answer is `±1`. The sample variance is `64/63`, which
    /// would give `±sqrt(63/64)` = `±0.99216` - 7.8e-3 away, three orders of
    /// magnitude outside the tolerance below.
    #[test]
    fn test_layer_norm_variance_divisor_is_n_simd_branch() -> Result<()> {
        const WIDTH: usize = 64;
        const { assert!(WIDTH >= MIN_NORM_LEN_FOR_SIMD) };

        let values: Vec<f32> = (0..WIDTH).map(|i| if i < WIDTH / 2 { -1.0 } else { 1.0 }).collect();
        let ln = LayerNorm::new(vec![WIDTH], 0.0)?;
        let actual = ln.forward(Tensor::from_data(values, &[1, WIDTH])?)?.to_vec_f32()?;

        for (i, &v) in actual.iter().enumerate() {
            let expected = if i < WIDTH / 2 { -1.0f32 } else { 1.0f32 };
            assert!(
                (v - expected).abs() < 1e-4,
                "element {i}: expected {expected} (population variance 1.0), got {v}; \
                 ±0.99216 would mean the sample variance 64/63"
            );
        }
        Ok(())
    }

    /// One layer, one set of values: the answer must not depend on how the
    /// values are shaped, nor on which side of [`MIN_NORM_LEN_FOR_SIMD`] the
    /// reduction lands.
    ///
    /// Both were false before: `[3, 32]` (96 elements) took the SIMD branch while
    /// `[1, 32]` (32 elements) did not, and 3-D inputs stayed on the SIMD branch
    /// while 2-D inputs were diverted to the Metal kernel on `metal` builds -
    /// `probe_ln3d` measured 0.032129 between `[1, 3, 32]` and `[3, 32]` on
    /// identical data.
    #[test]
    fn test_layer_norm_answer_is_invariant_to_rank_and_reduction_length() -> Result<()> {
        for &width in &[8usize, 32, 96] {
            let (gamma, beta) = ramp_gamma_beta(width);
            let ln = layer_norm_with_params(width, &gamma, &beta)?;
            let values = sample_values(6 * width);

            let flat = ln.forward(Tensor::from_data(values.clone(), &[6, width])?)?.to_vec_f32()?;
            let cubed =
                ln.forward(Tensor::from_data(values.clone(), &[2, 3, width])?)?.to_vec_f32()?;
            let single_rows: Vec<f32> = values
                .chunks_exact(width)
                .map(|row| ln.forward(Tensor::from_data(row.to_vec(), &[1, width])?))
                .collect::<Result<Vec<_>>>()?
                .iter()
                .map(|t| t.to_vec_f32())
                .collect::<Result<Vec<_>>>()?
                .concat();

            assert!(
                max_abs_diff(&flat, &cubed) < 1e-5,
                "width={width}: 2-D and 3-D disagree by {:e}",
                max_abs_diff(&flat, &cubed)
            );
            assert!(
                max_abs_diff(&flat, &single_rows) < 1e-5,
                "width={width}: batched and row-at-a-time disagree by {:e} (the row-at-a-time \
                 tensors are 6x smaller, so this is the element-count dependence)",
                max_abs_diff(&flat, &single_rows)
            );
        }
        Ok(())
    }

    /// A `[32, 8]` tensor has 256 elements but reduces over groups of 8. The
    /// predecessor gated vectorisation on the total (256 >= 64 -> SIMD), so this
    /// shape answered differently from `[2, 8]`; the gate is now on the reduction
    /// length, which is 8 for both.
    #[test]
    fn test_layer_norm_wide_batch_narrow_groups_matches_reference() -> Result<()> {
        let width = 8usize;
        let (gamma, beta) = ramp_gamma_beta(width);
        let ln = layer_norm_with_params(width, &gamma, &beta)?;
        let values = sample_values(32 * width);
        let expected = population_layer_norm_reference(&values, width, &gamma, &beta, 1e-5);

        let wide = ln.forward(Tensor::from_data(values.clone(), &[32, width])?)?.to_vec_f32()?;
        assert!(max_abs_diff(&wide, &expected) < 1e-5);

        let narrow = ln
            .forward(Tensor::from_data(
                values[..2 * width].to_vec(),
                &[2, width],
            )?)?
            .to_vec_f32()?;
        assert!(
            max_abs_diff(&narrow, &expected[..2 * width]) < 1e-5,
            "the 16-element tensor must agree with the first two rows of the 256-element one"
        );
        Ok(())
    }

    /// The Metal `layernorm` kernel, the CPU kernel and the reference must all
    /// agree, for 2-D **and** 3-D inputs.
    ///
    /// Rank used to decide the estimator on `metal` builds: 2-D inputs were
    /// diverted to the GPU (population variance) while 3-D inputs stayed on the
    /// SIMD CPU branch (sample variance).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_metal_layer_norm_matches_cpu_and_reference_2d_and_3d() -> Result<()> {
        use crate::gpu_ops::metal::get_metal_backend;

        if get_metal_backend().is_err() {
            eprintln!("no Metal device; skipping");
            return Ok(());
        }
        let metal = Device::Metal(0);

        for &width in &[32usize, 64, 768] {
            let (gamma, beta) = ramp_gamma_beta(width);
            let cpu_ln = layer_norm_with_params(width, &gamma, &beta)?;
            let mut gpu_ln = layer_norm_with_params(width, &gamma, &beta)?;
            gpu_ln.weights_to_gpu(&metal)?;

            for shape in [vec![4usize, width], vec![2, 2, width]] {
                let total: usize = shape.iter().product();
                let values = sample_values(total);
                let expected = population_layer_norm_reference(&values, width, &gamma, &beta, 1e-5);

                let cpu =
                    cpu_ln.forward(Tensor::from_data(values.clone(), &shape)?)?.to_vec_f32()?;
                let gpu_input = Tensor::from_data(values, &shape)?.to_device_enum(&metal)?;
                let gpu_output = gpu_ln.forward(gpu_input)?;
                assert!(
                    matches!(gpu_output, Tensor::Metal(_)),
                    "width={width} shape={shape:?}: GPU-resident input with GPU-resident \
                     parameters must stay on the GPU"
                );
                let gpu = gpu_output.to_vec_f32()?;

                assert!(
                    max_abs_diff(&gpu, &expected) < 1e-4,
                    "width={width} shape={shape:?}: GPU vs population reference = {:e}",
                    max_abs_diff(&gpu, &expected)
                );
                assert!(
                    max_abs_diff(&cpu, &expected) < 1e-5,
                    "width={width} shape={shape:?}: CPU vs population reference = {:e}",
                    max_abs_diff(&cpu, &expected)
                );
                assert!(
                    max_abs_diff(&gpu, &cpu) < 1e-4,
                    "width={width} shape={shape:?}: GPU vs CPU = {:e}",
                    max_abs_diff(&gpu, &cpu)
                );
            }
        }
        Ok(())
    }

    /// `Tensor::to_device_enum(Metal -> CPU)` must observe the output of the
    /// kernel that produced the buffer.
    ///
    /// `layernorm_gpu_to_gpu` (like every `*_gpu_to_gpu` wrapper) commits its
    /// command buffer asynchronously and returns immediately, and its output is a
    /// freshly allocated `StorageModeShared` buffer, which reads as zeroes until
    /// the dispatch lands. `to_device_enum` used to read `buffer.contents()`
    /// directly with no flush, so it raced the very kernel whose result it was
    /// asked for - `probe_flush.rs` measured 0 of 150 values non-zero through this
    /// path against 150 of 150 through the flushing `download_buffer_to_vec`.
    ///
    /// The forward pass below is deliberately large (512 rows x 1024 features,
    /// one thread per row making three passes) so the dispatch is still in flight
    /// when the read happens.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_metal_to_device_enum_observes_the_kernel_that_wrote_the_buffer() -> Result<()> {
        use crate::gpu_ops::metal::get_metal_backend;

        if get_metal_backend().is_err() {
            eprintln!("no Metal device; skipping");
            return Ok(());
        }
        let metal = Device::Metal(0);

        const ROWS: usize = 512;
        const WIDTH: usize = 1024;
        let (gamma, beta) = ramp_gamma_beta(WIDTH);
        let mut ln = layer_norm_with_params(WIDTH, &gamma, &beta)?;
        ln.weights_to_gpu(&metal)?;

        let values = sample_values(ROWS * WIDTH);
        let expected = population_layer_norm_reference(&values, WIDTH, &gamma, &beta, 1e-5);

        // Several trials: a racing read is a race, and one lucky trial proves
        // nothing.
        for trial in 0..4 {
            let input =
                Tensor::from_data(values.clone(), &[ROWS, WIDTH])?.to_device_enum(&metal)?;
            let output = ln.forward(input)?;
            assert!(
                matches!(output, Tensor::Metal(_)),
                "trial {trial}: expected a GPU result"
            );

            // The read under test. Nothing flushes between the async dispatch
            // above and this call.
            let host = output.to_device_enum(&Device::CPU)?;
            let host = match host {
                Tensor::F32(arr) => arr,
                other => panic!(
                    "trial {trial}: expected Tensor::F32, got {:?}",
                    other.dtype()
                ),
            };
            assert_eq!(host.shape(), [ROWS, WIDTH]);
            let host = host.iter().copied().collect::<Vec<f32>>();

            let zeros = host.iter().filter(|v| **v == 0.0).count();
            assert!(
                zeros * 100 < host.len(),
                "trial {trial}: {zeros}/{} values read back as exactly zero - that is the \
                 unflushed read seeing a freshly zeroed buffer, not a LayerNorm result",
                host.len()
            );
            let diff = max_abs_diff(&host, &expected);
            assert!(
                diff < 1e-4,
                "trial {trial}: max|host - population_reference| = {diff:e}"
            );
        }
        Ok(())
    }

    /// A shape that over-states its buffer must be refused, not read out of
    /// bounds.
    ///
    /// `to_device_enum` used to size an unchecked `slice::from_raw_parts` from
    /// `shape.iter().product()` without ever consulting `buffer.length()`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_metal_to_device_enum_rejects_a_shape_larger_than_its_buffer() -> Result<()> {
        use crate::gpu_ops::metal::get_metal_backend;
        use crate::tensor::MetalTensorData;

        let backend = match get_metal_backend() {
            Ok(backend) => backend,
            Err(_) => {
                eprintln!("no Metal device; skipping");
                return Ok(());
            },
        };

        let buffer_id = backend.create_persistent_buffer(&[1.0f32, 2.0, 3.0, 4.0])?;
        let overstated = Tensor::Metal(MetalTensorData::new(
            &backend,
            buffer_id,
            vec![4096],
            crate::tensor::DType::F32,
        )?);

        let result = overstated.to_device_enum(&Device::CPU);
        assert!(
            result.is_err(),
            "a 4-element buffer described as 4096 elements must be refused"
        );
        backend.release_buffers(&[buffer_id])?;
        Ok(())
    }
}
