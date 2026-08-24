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
#[cfg(not(target_os = "macos"))]
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Direct BLAS GEMM using OxiBLAS for maximum performance
#[cfg(target_os = "macos")]
#[inline]
fn blas_sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    use oxiblas_blas::level3::gemm;
    use oxiblas_matrix::{MatMut, MatRef};

    // Bridge row-major → col-major via Cᵀ = Bᵀ·Aᵀ identity:
    // Row-major A(m×k) reinterpreted as col-major is Aᵀ(k×m), lda=k.
    // Row-major B(k×n) reinterpreted as col-major is Bᵀ(n×k), lda=n.
    // Row-major C(m×n) reinterpreted as col-major is Cᵀ(n×m), lda=n.
    // gemm(Bᵀ, Aᵀ) → Cᵀ = Bᵀ·Aᵀ = (A·B)ᵀ, so C buffer holds A·B. ✓
    let a_t = MatRef::from_column_major(a, k, m).expect("A slice must hold m*k elements");
    let b_t = MatRef::from_column_major(b, n, k).expect("B slice must hold k*n elements");
    let c_t = MatMut::from_column_major(c, n, m).expect("C slice must hold m*n elements");

    // GEMM: Cᵀ = 1.0 * Bᵀ * Aᵀ + 0.0 * Cᵀ
    gemm(1.0, b_t, a_t, 0.0, c_t);
}

/// Direct BLAS GEMM using OxiBLAS for f64
#[cfg(target_os = "macos")]
#[inline]
fn blas_dgemm(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    use oxiblas_blas::level3::gemm;
    use oxiblas_matrix::{MatMut, MatRef};

    // Bridge row-major → col-major via Cᵀ = Bᵀ·Aᵀ identity:
    // Row-major A(m×k) reinterpreted as col-major is Aᵀ(k×m), lda=k.
    // Row-major B(k×n) reinterpreted as col-major is Bᵀ(n×k), lda=n.
    // Row-major C(m×n) reinterpreted as col-major is Cᵀ(n×m), lda=n.
    // gemm(Bᵀ, Aᵀ) → Cᵀ = Bᵀ·Aᵀ = (A·B)ᵀ, so C buffer holds A·B. ✓
    let a_t = MatRef::from_column_major(a, k, m).expect("A slice must hold m*k elements");
    let b_t = MatRef::from_column_major(b, n, k).expect("B slice must hold k*n elements");
    let c_t = MatMut::from_column_major(c, n, m).expect("C slice must hold m*n elements");

    // GEMM: Cᵀ = 1.0 * Bᵀ * Aᵀ + 0.0 * Cᵀ
    gemm(1.0, b_t, a_t, 0.0, c_t);
}

/// Fallback for non-macOS: use scirs2-core SIMD GEMM for f32
#[cfg(not(target_os = "macos"))]
#[inline]
fn blas_sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let a_arr = Array2::from_shape_vec((m, k), a.to_vec())
        .expect("matrix dimensions must match slice length");
    let b_arr = Array2::from_shape_vec((k, n), b.to_vec())
        .expect("matrix dimensions must match slice length");
    let mut c_arr = Array2::from_shape_vec((m, n), c.to_vec())
        .expect("matrix dimensions must match slice length");
    f32::simd_gemm(1.0, &a_arr.view(), &b_arr.view(), 0.0, &mut c_arr);
    c.copy_from_slice(c_arr.as_slice().expect("Array2 must have contiguous slice"));
}

/// Fallback for non-macOS: use scirs2-core SIMD GEMM for f64
#[cfg(not(target_os = "macos"))]
#[inline]
fn blas_dgemm(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    let a_arr = Array2::from_shape_vec((m, k), a.to_vec())
        .expect("matrix dimensions must match slice length");
    let b_arr = Array2::from_shape_vec((k, n), b.to_vec())
        .expect("matrix dimensions must match slice length");
    let mut c_arr = Array2::from_shape_vec((m, n), c.to_vec())
        .expect("matrix dimensions must match slice length");
    f64::simd_gemm(1.0, &a_arr.view(), &b_arr.view(), 0.0, &mut c_arr);
    c.copy_from_slice(c_arr.as_slice().expect("Array2 must have contiguous slice"));
}

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
/// let linear = Linear::new(768, 3072, true);
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
    /// Cached transpose of `weight` (`[in_features, out_features]`).
    ///
    /// `Linear::forward` multiplies by `W^T`, and the weight is immutable for
    /// the whole of inference, so recomputing the transpose per forward call
    /// (a 128 MiB memcpy for a 4096x4096 f32 weight -- ~896 MiB per transformer
    /// block per token in decode) is pure waste. The transpose is computed once
    /// and rebuilt only when the weight itself changes.
    ///
    /// # Why a `OnceLock`
    ///
    /// Invalidation happens through `&mut self` ([`Linear::set_weight`],
    /// [`Linear::parameters_mut`]) but the *rebuild* has to happen inside
    /// [`Layer::forward_ref`], which only has `&self`. A plain `Option` field
    /// therefore had a one-way failure mode: anything that cleared it — notably
    /// `Model::named_tensors_mut`, which hands out `&mut Tensor` for every
    /// parameter — left the cache empty forever, and every subsequent forward
    /// pass silently paid a full transpose. `OnceLock` closes that hole:
    /// [`OnceLock::take`] clears it through `&mut self`, and `get_or_init`
    /// refills it on the next forward through `&self`.
    ///
    /// The inner `Arc` means cloning a `Linear` does not deep-copy the cached
    /// matrix; the inner `Option` records "this weight cannot be transposed"
    /// (a non-2D weight) so the fallback path is taken without re-trying.
    weight_transposed: std::sync::OnceLock<Option<std::sync::Arc<Tensor>>>,
    bias: Option<Tensor>,
    device: Device,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    weight_buffer_id: std::sync::Arc<std::sync::RwLock<Option<crate::gpu_ops::BufferId>>>,
    /// Reference-counted handle to the CUDA-resident (transposed) weight buffer.
    ///
    /// Holding a `BufferHandle` (rather than a raw `BufferId`) means invalidating the
    /// cache (`to_device`, `set_weight`) or dropping the last `Linear` clone releases
    /// the device allocation automatically instead of leaking it.
    #[cfg(feature = "cuda")]
    weight_buffer_id_cuda:
        std::sync::Arc<std::sync::RwLock<Option<crate::gpu_ops::cuda::BufferHandle>>>,
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
        // reason: tensor construction here only fails for invalid shapes, and the
        // shapes are derived from the (valid) feature counts; this public `-> Self`
        // constructor has no fallible alternative without an API break.
        #[allow(clippy::expect_used)]
        let weight =
            Tensor::randn(&[out_features, in_features]).expect("Failed to create random tensor");
        #[allow(clippy::expect_used)]
        let bias = if bias {
            Some(Tensor::zeros(&[out_features]).expect("Failed to create zero tensor"))
        } else {
            None
        };

        Self {
            weight,
            // Left empty: the first forward pass fills it. Building it eagerly
            // would transpose weights that a checkpoint load is about to replace.
            weight_transposed: std::sync::OnceLock::new(),
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
            // Dropping the handle releases the stale device allocation (refcounted).
            if let Ok(mut buffer_handle) = self.weight_buffer_id_cuda.write() {
                *buffer_handle = None;
            }
        }
        self
    }

    /// Build the cached `W^T` for a weight matrix, or `None` when the weight is
    /// not a 2-D matrix (in which case callers transpose on demand).
    fn build_transposed_weight(weight: &Tensor) -> Option<std::sync::Arc<Tensor>> {
        weight.transpose(0, 1).ok().map(std::sync::Arc::new)
    }

    /// Return `W^T`, filling the cache on first use.
    ///
    /// Takes `&self` because every forward path does. The cache is refilled here
    /// rather than at invalidation time, so clearing it through `&mut self` (a
    /// checkpoint load, `named_tensors_mut`) costs nothing and the next forward
    /// pass restores it — the transpose is never paid more than once per weight.
    ///
    /// The returned `Arc` is a refcount bump, not a copy. A weight that cannot be
    /// transposed (not 2-D) caches `None` and is transposed on demand, which
    /// simply reproduces the error each call.
    fn transposed_weight(&self) -> Result<std::sync::Arc<Tensor>> {
        match self
            .weight_transposed
            .get_or_init(|| Self::build_transposed_weight(&self.weight))
        {
            Some(cached) => Ok(std::sync::Arc::clone(cached)),
            None => Ok(std::sync::Arc::new(self.weight.transpose(0, 1)?)),
        }
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
        // Drop the stale transpose and device buffers; the next forward pass
        // rebuilds the transpose from the weight just installed.
        self.invalidate_weight_caches();
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

    /// Returns a mutable reference to the weight matrix.
    ///
    /// This is the write side of [`Linear::weight`], used by
    /// [`Model::named_tensors_mut`](crate::traits::Model::named_tensors_mut) so a
    /// checkpoint loader can copy into the live parameter in place.
    ///
    /// # Cache invalidation
    ///
    /// `Linear` caches `W^T` (and, on Metal/CUDA builds, the device-resident
    /// transposed buffer). Handing out `&mut Tensor` means the caller can mutate
    /// the weight without going through [`Linear::set_weight`], so both caches are
    /// dropped *pessimistically* before the borrow is returned;
    /// [`Linear::transposed_weight`] then recomputes the transpose on the next
    /// forward pass. Keeping the stale transpose instead would make every
    /// subsequent forward silently wrong.
    pub fn weight_mut(&mut self) -> &mut Tensor {
        self.invalidate_weight_caches();
        &mut self.weight
    }

    /// Returns a mutable reference to the bias vector if present.
    ///
    /// Returns `None` for a layer created without bias — a caller must not be able
    /// to conjure a bias into existence through a parameter iterator.
    pub fn bias_mut(&mut self) -> Option<&mut Tensor> {
        self.bias.as_mut()
    }

    /// Append this layer's parameters to `into` under `<prefix>.weight` /
    /// `<prefix>.bias`.
    ///
    /// `weight` and `bias` are the names PyTorch's `nn.Linear` uses, so a model
    /// composing these calls produces HuggingFace-shaped keys for
    /// [`Model::named_tensors`](crate::traits::Model::named_tensors) with no
    /// per-model string plumbing. A layer without bias contributes exactly one
    /// entry — the absence is never papered over with a zero vector.
    ///
    /// The references are to the *live* parameters, as the trait requires: no
    /// tensor is copied, reshaped or transposed on the way out.
    pub fn collect_named_parameters<'a>(
        &'a self,
        prefix: &str,
        into: &mut Vec<(String, &'a Tensor)>,
    ) {
        into.push((format!("{prefix}.weight"), &self.weight));
        if let Some(bias) = &self.bias {
            into.push((format!("{prefix}.bias"), bias));
        }
    }

    /// Mutable counterpart of [`Linear::collect_named_parameters`].
    ///
    /// Goes through [`Linear::parameters_mut`], so the cached transpose and any
    /// GPU-resident weight buffer are invalidated before the caller can write.
    pub fn collect_named_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        let (weight, bias) = self.parameters_mut();
        into.push((format!("{prefix}.weight"), weight));
        if let Some(bias) = bias {
            into.push((format!("{prefix}.bias"), bias));
        }
    }

    /// Borrow the weight **and** the bias mutably at the same time.
    ///
    /// `weight_mut()` followed by `bias_mut()` cannot compile: each borrows all of
    /// `*self`. Building a `Vec<(String, &mut Tensor)>` for
    /// [`Model::named_tensors_mut`](crate::traits::Model::named_tensors_mut) needs
    /// both live at once, which only disjoint *field* borrows inside this impl can
    /// give. The cache invalidation also happens once here rather than per borrow.
    ///
    /// The bias is `None` exactly when the layer was built without one.
    pub fn parameters_mut(&mut self) -> (&mut Tensor, Option<&mut Tensor>) {
        self.invalidate_weight_caches();
        (&mut self.weight, self.bias.as_mut())
    }

    /// Whether the `W^T` cache is currently unpopulated.
    ///
    /// Test-only introspection. The transpose cache is a pure performance
    /// optimisation, so its state cannot be observed through outputs — a layer
    /// whose cache never refills computes exactly the same numbers, just slower.
    /// The regression test for that failure mode therefore has to look at the
    /// cache itself.
    #[cfg(test)]
    pub(crate) fn transposed_weight_cache_is_empty(&self) -> bool {
        self.weight_transposed.get().is_none()
    }

    /// Drop every cached derivative of `self.weight`.
    ///
    /// Cheap by design: the transpose is *not* rebuilt here, only dropped.
    /// [`Linear::transposed_weight`] refills it on the next forward pass, so a
    /// caller that clears the cache and never runs a forward pass pays nothing,
    /// and one that does pays the transpose exactly once.
    fn invalidate_weight_caches(&mut self) {
        // `take` needs `&mut self`, which is precisely why every invalidation
        // path funnels through here.
        let _ = self.weight_transposed.take();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Ok(mut buffer_id) = self.weight_buffer_id.write() {
                *buffer_id = None;
            }
        }
        #[cfg(feature = "cuda")]
        {
            // Dropping the handle releases the stale device allocation (refcounted).
            if let Ok(mut buffer_handle) = self.weight_buffer_id_cuda.write() {
                *buffer_handle = None;
            }
        }
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
            let weight_t = self.transposed_weight()?;
            match weight_t.as_ref() {
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
            let weight_t = self.transposed_weight()?;
            match weight_t.as_ref() {
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
                    // Wrap in a refcounted handle so the allocation is released when the
                    // cache entry is invalidated or the last `Linear` clone drops.
                    *buffer_id_guard = Some(crate::gpu_ops::cuda::BufferHandle::new(
                        new_buffer_id,
                        device_id,
                    ));
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

    /// Owning entry point.
    ///
    /// The whole computation lives in [`Layer::forward_ref`] because none of it
    /// needs to own the input; this wrapper simply lends the value it was handed.
    /// It must *never* be implemented in terms of the trait's default
    /// `forward_ref`, which clones and calls back into `forward`.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_ref(&input)
    }

    /// Borrowing forward pass — the real implementation.
    ///
    /// Attention layers project the *same* hidden-state tensor into Q, K and V,
    /// so an owning-only API forced three deep clones of `[batch, seq, hidden]`
    /// per attention block. Nothing below needs ownership, so the borrow is free.
    fn forward_ref(&self, input: &Self::Input) -> Result<Self::Output> {
        // =====================================================================
        // GPU-TO-GPU PATH: Tensor::Metal (ZERO CPU TRANSFERS!)
        // =====================================================================
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Tensor::Metal(input_metal) = input {
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
                    return self.forward_ref(&cpu_input);
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
                .matmul_gpu_to_gpu_mps(&input_metal.buffer_id(), &weight_buffer_id, m, k, n)
                .or_else(|_e| {
                    // eprintln!(
                    //     "⚠️  MPS matmul failed: {:?}, falling back to naive Metal kernel",
                    //     e
                    // );
                    // Fallback to naive Metal kernel if MPS fails
                    backend.matmul_gpu_to_gpu(&input_metal.buffer_id(), &weight_buffer_id, m, k, n)
                })?;

            // Calculate output shape (preserve batch dimensions, change last dim)
            let mut output_shape = shape[..shape.len() - 1].to_vec();
            output_shape.push(n);

            // Create output Metal tensor
            let mut output = Tensor::Metal(MetalTensorData::new(
                &backend,
                output_buffer_id,
                output_shape.clone(),
                input_metal.dtype,
            )?);

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
                                &output_data.buffer_id(),
                                &bias_data.buffer_id(),
                                batch_dims,
                                n,
                            )?;
                            // eprintln!(
                            //     "🔍 Linear: add_bias_gpu_to_gpu succeeded, returning Metal tensor"
                            // );

                            return Ok(Tensor::Metal(MetalTensorData::new(
                                &backend,
                                output_buffer_id,
                                output_shape.clone(),
                                output_data.dtype,
                            )?));
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
        if let Tensor::CUDA(input_cuda) = input {
            use crate::gpu_ops::cuda::get_cuda_backend;
            use crate::tensor::CudaTensorData;

            // Ensure weight buffer is cached on GPU
            self.ensure_weight_buffer_cached_cuda()?;

            // Get cached weight buffer handle. Cloning the handle bumps the refcount,
            // keeping the weight buffer alive for the whole forward pass even if another
            // thread invalidates the cache concurrently.
            let weight_handle = {
                let buffer_id_guard = self.weight_buffer_id_cuda.read().map_err(|_| {
                    TrustformersError::hardware_error(
                        "Failed to acquire read lock on CUDA buffer cache",
                        "Linear::forward",
                    )
                })?;

                if let Some(handle) = buffer_id_guard.as_ref() {
                    handle.clone()
                } else {
                    // Weight not cached - fallback to CPU
                    let cpu_input = input.to_device_enum(&crate::device::Device::CPU)?;
                    return self.forward_ref(&cpu_input);
                }
            };

            // The resident input and the cached weight must live on the same CUDA
            // device for a GPU-to-GPU kernel; otherwise fall back to the host path.
            let device_id = input_cuda.device_id();
            if weight_handle.device_id() != device_id {
                let cpu_input = input.to_device_enum(&crate::device::Device::CPU)?;
                return self.forward_ref(&cpu_input);
            }
            let weight_buffer_id = weight_handle.id();

            // Get CUDA backend for the device the operands live on
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
                backend.matmul_gpu_to_gpu(&input_cuda.buffer_id(), &weight_buffer_id, m, k, n)?;

            // Calculate output shape (preserve batch dimensions, change last dim)
            let mut output_shape = shape[..shape.len() - 1].to_vec();
            output_shape.push(n);

            // Create output CUDA tensor
            let mut output = Tensor::CUDA(CudaTensorData::new(
                output_buffer_id,
                device_id,
                output_shape.clone(),
                input_cuda.dtype,
            ));

            // Handle bias if present
            if let Some(ref bias) = self.bias {
                // Try GPU-to-GPU bias addition if bias is resident on the same device
                match bias {
                    #[cfg(feature = "cuda")]
                    Tensor::CUDA(bias_data) if bias_data.device_id() == device_id => {
                        // Both output and bias are CUDA tensors on one device - GPU kernel!
                        if let Tensor::CUDA(output_data) = &output {
                            let output_buffer_id = backend.add_bias_gpu_to_gpu(
                                &output_data.buffer_id(),
                                &bias_data.buffer_id(),
                                batch_dims,
                                n,
                            )?;

                            return Ok(Tensor::CUDA(CudaTensorData::new(
                                output_buffer_id,
                                device_id,
                                output_shape.clone(),
                                output_data.dtype,
                            )));
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
        let weight_t = self.transposed_weight()?;

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

                        if let (Tensor::F32(inp), Tensor::F32(w_t)) = (input, weight_t.as_ref()) {
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
                    dispatch_matmul(input, weight_t.as_ref(), &self.device)?
                } else {
                    input.matmul(weight_t.as_ref())?
                }
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                input.matmul(weight_t.as_ref())?
            }
        } else if input_shape.len() == 3 {
            // Batched 3D input: [batch, seq_len, hidden_size] x [hidden_size, out_features]
            // Handle manually since tensor.matmul doesn't support 3D x 2D
            match (input, weight_t.as_ref()) {
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

                    // Route through the pure-Rust GEMM (OxiBLAS on macOS, scirs2-core
                    // SIMD elsewhere) once the operands are big enough to pay for it.
                    let m = inp_2d.nrows();
                    let n = w_2d.ncols();
                    let k = inp_2d.ncols();
                    const MIN_SIZE_FOR_BLAS: usize = 32;
                    let out_2d = if m < MIN_SIZE_FOR_BLAS
                        || n < MIN_SIZE_FOR_BLAS
                        || k < MIN_SIZE_FOR_BLAS
                    {
                        inp_2d.dot(&w_2d)
                    } else {
                        // Direct GEMM on the contiguous slices.
                        let inp_slice = inp_2d.as_slice().unwrap_or(&[]);
                        let w_slice = w_2d.as_slice().unwrap_or(&[]);
                        if !inp_slice.is_empty() && !w_slice.is_empty() {
                            let mut result_vec = vec![0.0f32; m * n];
                            blas_sgemm(inp_slice, w_slice, &mut result_vec, m, k, n);
                            Array2::from_shape_vec((m, n), result_vec).map_err(|e| {
                                crate::errors::compute_error(
                                    "forward",
                                    format!("{}: {e}", "BLAS result shape must match m x n"),
                                )
                            })?
                        } else {
                            // Fallback to ndarray dot if slices aren't contiguous
                            inp_2d.dot(&w_2d)
                        }
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

                    // Route through the pure-Rust GEMM (OxiBLAS on macOS, scirs2-core
                    // SIMD elsewhere) once the operands are big enough to pay for it.
                    let m = inp_2d.nrows();
                    let n = w_2d.ncols();
                    let k = inp_2d.ncols();
                    const MIN_SIZE_FOR_BLAS: usize = 32;
                    let out_2d = if m < MIN_SIZE_FOR_BLAS
                        || n < MIN_SIZE_FOR_BLAS
                        || k < MIN_SIZE_FOR_BLAS
                    {
                        inp_2d.dot(&w_2d)
                    } else {
                        // Direct GEMM on the contiguous slices.
                        let inp_slice = inp_2d.as_slice().unwrap_or(&[]);
                        let w_slice = w_2d.as_slice().unwrap_or(&[]);
                        if !inp_slice.is_empty() && !w_slice.is_empty() {
                            let mut result_vec = vec![0.0f64; m * n];
                            blas_dgemm(inp_slice, w_slice, &mut result_vec, m, k, n);
                            Array2::from_shape_vec((m, n), result_vec).map_err(|e| {
                                crate::errors::compute_error(
                                    "forward",
                                    format!("{}: {e}", "BLAS result shape must match m x n"),
                                )
                            })?
                        } else {
                            // Fallback to ndarray dot if slices aren't contiguous
                            inp_2d.dot(&w_2d)
                        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::Result;

    #[test]
    fn test_linear_creation_with_bias() {
        let linear = Linear::new(4, 3, true);
        assert_eq!(linear.weight().shape(), vec![3, 4]);
        assert!(linear.bias().is_some());
    }

    #[test]
    fn test_linear_creation_without_bias() {
        let linear = Linear::new(4, 3, false);
        assert_eq!(linear.weight().shape(), vec![3, 4]);
        assert!(linear.bias().is_none());
    }

    #[test]
    fn test_linear_device_default_cpu() {
        let linear = Linear::new(2, 2, true);
        assert_eq!(linear.device(), Device::CPU);
    }

    #[test]
    fn test_linear_to_device() {
        let linear = Linear::new(2, 2, true);
        let moved = linear.to_device(Device::CPU);
        assert_eq!(moved.device(), Device::CPU);
    }

    #[test]
    fn test_linear_parameter_count_with_bias() {
        let linear = Linear::new(4, 3, true);
        // 4*3 weights + 3 bias = 15
        assert_eq!(linear.parameter_count(), 15);
    }

    #[test]
    fn test_linear_parameter_count_without_bias() {
        let linear = Linear::new(4, 3, false);
        // 4*3 weights = 12
        assert_eq!(linear.parameter_count(), 12);
    }

    #[test]
    fn test_linear_set_weight() -> Result<()> {
        let mut linear = Linear::new(3, 2, true);
        let new_weight = Tensor::ones(&[2, 3])?;
        linear.set_weight(new_weight)?;
        assert_eq!(linear.weight().shape(), vec![2, 3]);
        Ok(())
    }

    #[test]
    fn test_linear_set_bias() -> Result<()> {
        let mut linear = Linear::new(3, 2, false);
        assert!(linear.bias().is_none());
        let new_bias = Tensor::zeros(&[2])?;
        linear.set_bias(new_bias)?;
        assert!(linear.bias().is_some());
        Ok(())
    }

    #[test]
    fn test_linear_forward_2d() -> Result<()> {
        let mut linear = Linear::new(3, 2, false);
        // Set weight to ones for easy verification
        let weight = Tensor::ones(&[2, 3])?;
        linear.set_weight(weight)?;
        let input = Tensor::ones(&[4, 3])?;
        let output = linear.forward(input)?;
        assert_eq!(output.shape(), vec![4, 2]);
        // Each output should be sum of 3 ones = 3.0
        let data = output.data()?;
        for val in &data {
            assert!((val - 3.0).abs() < 1e-3);
        }
        Ok(())
    }

    #[test]
    fn test_linear_forward_with_bias() -> Result<()> {
        let mut linear = Linear::new(3, 2, true);
        let weight = Tensor::ones(&[2, 3])?;
        let bias = Tensor::full_with_shape(&[2], 1.0)?;
        linear.set_weight(weight)?;
        linear.set_bias(bias)?;
        let input = Tensor::ones(&[1, 3])?;
        let output = linear.forward(input)?;
        assert_eq!(output.shape(), vec![1, 2]);
        // Each output = 3.0 (matmul) + 1.0 (bias) = 4.0
        let data = output.data()?;
        for val in &data {
            assert!((val - 4.0).abs() < 1e-3);
        }
        Ok(())
    }

    #[test]
    fn test_linear_forward_3d() -> Result<()> {
        let mut linear = Linear::new(4, 2, false);
        let weight = Tensor::ones(&[2, 4])?;
        linear.set_weight(weight)?;
        let input = Tensor::ones(&[2, 3, 4])?;
        let output = linear.forward(input)?;
        assert_eq!(output.shape(), vec![2, 3, 2]);
        Ok(())
    }

    #[test]
    fn test_linear_weight_ref() {
        let linear = Linear::new(5, 3, true);
        let weight = linear.weight();
        assert_eq!(weight.shape(), vec![3, 5]);
    }

    #[test]
    fn test_linear_bias_ref() {
        let linear = Linear::new(5, 3, true);
        if let Some(bias) = linear.bias() {
            assert_eq!(bias.shape(), vec![3]);
        }
    }

    #[test]
    fn test_linear_new_with_device() {
        let linear = Linear::new_with_device(4, 2, true, Device::CPU);
        assert_eq!(linear.device(), Device::CPU);
        assert_eq!(linear.weight().shape(), vec![2, 4]);
    }

    #[test]
    fn test_linear_forward_zero_weight() -> Result<()> {
        let mut linear = Linear::new(3, 2, false);
        let weight = Tensor::zeros(&[2, 3])?;
        linear.set_weight(weight)?;
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0], &[1, 3])?;
        let output = linear.forward(input)?;
        let data = output.data()?;
        for val in &data {
            assert!(val.abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_linear_forward_identity_like() -> Result<()> {
        let mut linear = Linear::new(2, 2, false);
        let weight = Tensor::eye_f32(2)?;
        linear.set_weight(weight)?;
        let input = Tensor::from_data(vec![3.0, 7.0], &[1, 2])?;
        let output = linear.forward(input)?;
        let data = output.data()?;
        assert!((data[0] - 3.0).abs() < 1e-3);
        assert!((data[1] - 7.0).abs() < 1e-3);
        Ok(())
    }

    #[test]
    fn test_linear_large_layer() {
        let linear = Linear::new(768, 3072, true);
        assert_eq!(linear.weight().shape(), vec![3072, 768]);
        // 768 * 3072 + 3072 = 2362368
        assert_eq!(linear.parameter_count(), 768 * 3072 + 3072);
    }
}
