//! Linear algebra operations for tensors.
//!
//! This module provides core linear algebra operations including matrix multiplication,
//! norm calculations, and gradient clipping. All operations include numerical stability
//! enhancements and comprehensive error handling.
//!
//! # Features
//!
//! - **Matrix Multiplication**: BLAS-accelerated matmul via scirs2-core SIMD operations
//! - **Norm Calculations**: SIMD-optimized L2 norm and squared norm computations
//! - **Gradient Clipping**: Norm-based gradient clipping for training stability
//! - **Numerical Stability**: Built-in overflow/underflow protection and NaN detection
//! - **Multi-type Support**: Full support for F32, F64, I64, C32, and C64 tensor types
//!
//! # Performance
//!
//! This module leverages scirs2-core's `SimdUnifiedOps` trait which provides:
//! - BLAS-accelerated GEMM (Accelerate on macOS, MKL on Intel, OpenBLAS fallback)
//! - SIMD-optimized element-wise operations (AVX2/AVX-512 on x86, NEON on ARM)
//! - Automatic platform detection and optimal backend selection
//!
//! # Examples
//!
//! ```no_run
//! use trustformers_core::tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Matrix multiplication (BLAS-accelerated)
//! let a = Tensor::randn(&[128, 64])?;
//! let b = Tensor::randn(&[64, 256])?;
//! let result = a.matmul(&b)?;
//!
//! // Norm calculation (SIMD-optimized)
//! let tensor = Tensor::randn(&[100])?;
//! let norm = tensor.norm()?;
//!
//! // Gradient clipping
//! let gradients = Tensor::randn(&[1000])?;
//! let clipped = gradients.clip_grad_norm(1.0)?;
//! # Ok(())
//! # }
//! ```

use super::super::{DType, Tensor};
use crate::errors::{Result, TrustformersError};
use scirs2_core::ndarray::{Array2, ArrayD, Axis, Ix2, IxDyn};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::borrow::Cow;

/// Direct BLAS GEMM using OxiBLAS for maximum performance
/// OxiBLAS provides pure Rust BLAS with SIMD optimizations
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
#[allow(dead_code)] // Reserved for f64 tensor matmul
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

/// Fallback for non-macOS: use scirs2-core SIMD GEMM
#[cfg(not(target_os = "macos"))]
#[inline]
fn blas_sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    use scirs2_core::ndarray::{Array2, ArrayView2};
    // Borrow the input slices as views — `simd_gemm` only reads `a`/`b`, so no copy is needed.
    let a_arr =
        ArrayView2::from_shape((m, k), a).expect("matrix dimensions must match slice length");
    let b_arr =
        ArrayView2::from_shape((k, n), b).expect("matrix dimensions must match slice length");
    // `simd_gemm` requires an owned output; with beta = 0.0 the prior contents of `c`
    // are discarded, so allocate zeros instead of copying `c` in.
    let mut c_arr = Array2::zeros((m, n));
    f32::simd_gemm(1.0, &a_arr, &b_arr, 0.0, &mut c_arr);
    c.copy_from_slice(c_arr.as_slice().expect("array must have contiguous layout"));
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)] // Reserved for f64 tensor matmul
#[inline]
fn blas_dgemm(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    use scirs2_core::ndarray::{Array2, ArrayView2};
    // Borrow the input slices as views — `simd_gemm` only reads `a`/`b`, so no copy is needed.
    let a_arr =
        ArrayView2::from_shape((m, k), a).expect("matrix dimensions must match slice length");
    let b_arr =
        ArrayView2::from_shape((k, n), b).expect("matrix dimensions must match slice length");
    // beta = 0.0 discards `c`'s prior contents, so allocate zeros instead of copying `c` in.
    let mut c_arr = Array2::zeros((m, n));
    f64::simd_gemm(1.0, &a_arr, &b_arr, 0.0, &mut c_arr);
    c.copy_from_slice(c_arr.as_slice().expect("array must have contiguous layout"));
}

/// Batched GEMM over the trailing two axes of two `f32` arrays.
///
/// `a` has shape `[..leading, m, k]`, `b` has shape `[..leading, k, n]`, and the
/// result has shape `[..leading, m, n]`. The leading (batch) axes must match
/// exactly; broadcasting is not performed.
///
/// The implementation flattens the leading axes into a single batch dimension and
/// runs one GEMM per batch **directly into** the destination buffer, in parallel
/// across batches. No per-batch temporaries are allocated: the operands are read
/// through their contiguous backing slices and the result chunk is written in
/// place.
fn batched_gemm_f32(a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
    use scirs2_core::ndarray::{ArrayView2, ArrayViewMut2};
    // The glob brings rayon's `IndexedParallelIterator` into scope when the
    // `parallel` feature is on; without it `par_chunks_mut` degrades to
    // `std::slice::ChunksMut` and the same `.enumerate().for_each(..)` applies.
    #[allow(unused_imports)]
    use scirs2_core::parallel_ops::{par_chunks_mut, *};

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != b_shape.len() || a_shape.len() < 3 {
        return Err(TrustformersError::shape_error(format!(
            "Batched matmul requires both operands to have the same rank (>= 3), got {:?} and {:?}",
            a_shape, b_shape
        )));
    }

    let nd = a_shape.len();
    let leading = &a_shape[..nd - 2];
    if leading != &b_shape[..nd - 2] {
        return Err(TrustformersError::shape_error(format!(
            "Batched matmul requires matching batch dimensions, got {:?} and {:?}",
            &a_shape[..nd - 2],
            &b_shape[..nd - 2]
        )));
    }

    let m = a_shape[nd - 2];
    let k = a_shape[nd - 1];
    let n = b_shape[nd - 1];
    if b_shape[nd - 2] != k {
        return Err(TrustformersError::shape_error(format!(
            "Matrix dimensions mismatch: {} vs {}",
            k,
            b_shape[nd - 2]
        )));
    }

    let mut out_shape = leading.to_vec();
    out_shape.push(m);
    out_shape.push(n);
    let mut result = ArrayD::<f32>::zeros(IxDyn(&out_shape));

    // Zero-sized products need no work (the zero-filled result is already correct).
    if m == 0 || n == 0 || leading.iter().product::<usize>() == 0 {
        return Ok(result);
    }

    // Borrow contiguous backing storage; copy only if a layout fix is required.
    let a_cow = a.as_standard_layout();
    let b_cow = b.as_standard_layout();
    let a_data = a_cow.as_slice().ok_or_else(|| {
        TrustformersError::tensor_op_error("Operand A is not contiguous", "matmul")
    })?;
    let b_data = b_cow.as_slice().ok_or_else(|| {
        TrustformersError::tensor_op_error("Operand B is not contiguous", "matmul")
    })?;
    let out_data = result.as_slice_mut().ok_or_else(|| {
        TrustformersError::tensor_op_error("Result buffer is not contiguous", "matmul")
    })?;

    // Direct BLAS pays off only for reasonably sized blocks; below the threshold
    // ndarray's `general_mat_mul` (matrixmultiply kernels) writes in place too.
    const MIN_SIZE_FOR_BLAS: usize = 32;
    let use_blas = m >= MIN_SIZE_FOR_BLAS && n >= MIN_SIZE_FOR_BLAS && k >= MIN_SIZE_FOR_BLAS;

    par_chunks_mut(out_data, m * n)
        .enumerate()
        .for_each(|(batch_index, out_chunk)| {
            let a_offset = batch_index * m * k;
            let b_offset = batch_index * k * n;
            let a_block = &a_data[a_offset..a_offset + m * k];
            let b_block = &b_data[b_offset..b_offset + k * n];

            if use_blas {
                blas_sgemm(a_block, b_block, out_chunk, m, k, n);
            } else {
                // INVARIANT: the slices above are exactly m*k, k*n and m*n elements
                // long by construction, so these shape conversions cannot fail.
                let a_view = ArrayView2::from_shape((m, k), a_block)
                    .expect("a_block holds exactly m*k elements");
                let b_view = ArrayView2::from_shape((k, n), b_block)
                    .expect("b_block holds exactly k*n elements");
                let mut out_view = ArrayViewMut2::from_shape((m, n), out_chunk)
                    .expect("out_chunk holds exactly m*n elements");
                scirs2_core::ndarray::linalg::general_mat_mul(
                    1.0,
                    &a_view,
                    &b_view,
                    0.0,
                    &mut out_view,
                );
            }
        });

    Ok(result)
}

impl Tensor {
    /// Matrix multiplication with numerical stability enhancements.
    ///
    /// Performs matrix multiplication between two tensors with support for:
    /// - 2D matrix multiplication
    /// - Batched 3D matrix multiplication
    /// - Multi-headed 4D matrix multiplication (for attention mechanisms)
    ///
    /// # Numerical behaviour
    ///
    /// The product is always computed with the BLAS-accelerated GEMM path
    /// (`oxiblas` on macOS, `scirs2-core` SIMD elsewhere); non-finite inputs
    /// propagate through the result per IEEE-754 rather than being silently
    /// rewritten. Operands are copied only when their memory layout is not
    /// already row-major contiguous.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to multiply with (right operand)
    ///
    /// # Returns
    ///
    /// A new tensor containing the matrix multiplication result.
    ///
    /// # Errors
    ///
    /// - `ShapeError`: If tensors have incompatible dimensions for matrix multiplication
    /// - `TensorOpError`: If the operation is not supported for the tensor types
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // 2D matrix multiplication
    /// let a = Tensor::randn(&[128, 64])?;
    /// let b = Tensor::randn(&[64, 256])?;
    /// let result = a.matmul(&b)?; // Shape: [128, 256]
    ///
    /// // Batched matrix multiplication
    /// let a = Tensor::randn(&[32, 128, 64])?;  // 32 batches
    /// let b = Tensor::randn(&[32, 64, 256])?;
    /// let result = a.matmul(&b)?; // Shape: [32, 128, 256]
    ///
    /// // Multi-headed attention matrices
    /// let q = Tensor::randn(&[8, 12, 512, 64])?;  // 8 batches, 12 heads
    /// let k = Tensor::randn(&[8, 12, 64, 512])?;
    /// let attention = q.matmul(&k)?; // Shape: [8, 12, 512, 512]
    /// # Ok(())
    /// # }
    /// ```
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        // GPU dispatch guards: route CUDA/Metal-capable operand combinations to the
        // matching GPU backend in `crate::gpu_ops`. These arms are `#[cfg]`-gated behind
        // the respective GPU features, so the default (CPU-only) build is unaffected.
        //
        // The `cuda` feature is the Pure-Rust oxicuda backend, reached through two arms:
        //
        // (1) GPU-resident operands. If either side is `Tensor::CUDA`, the resident
        //     dispatcher runs the multiply on the device: CUDA x CUDA on one device is a
        //     fully resident batched GEMM producing a `Tensor::CUDA` (its buffer owned by
        //     the refcounted handle lifecycle, freed when the last clone drops); a mixed
        //     host-F32 operand is uploaded to the resident operand's device (temporary
        //     buffer freed on drop); cross-device and non-F32-host mixes bounce through
        //     the host. A `Tensor::CUDA` can only exist if a runtime-checked upload
        //     already succeeded, so no availability probe is needed in this arm.
        //
        // (2) Host F32 operands. Rank >= 2 F32 matmuls (2D plus batched/broadcast N-D —
        //     see `BatchedMatmulPlan` for the broadcasting rules) are uploaded, multiplied
        //     via oxicuda-blas GEMM, and returned as a host `Tensor::F32`.
        //
        //     Unlike arm (1) (which only activates for an already GPU-resident tensor),
        //     arm (2) operates on plain host `Tensor::F32` operands that exist on *every*
        //     build, CUDA-capable hardware or not. So `#[cfg(feature = "cuda")]` alone (a
        //     compile-time check) is not a sufficient guard here — it must also be
        //     confirmed with a genuine runtime probe via `oxicuda_cuda_available()`
        //     (cached after the first call, so this stays cheap on this hot path).
        //     Without that check, any F32 matmul on a `cuda`-enabled build with no actual
        //     GPU present would unconditionally hard-error instead of falling through to
        //     the CPU path below. Empty operands (zero-sized dims) stay on the CPU path,
        //     which produces the correct empty result without touching the driver.
        //
        // Half-precision (F16/BF16) operands never enter either arm: they keep using the
        // CPU upcast path below (device-side f16 GEMM is planned 0.3.x work).
        #[cfg(feature = "cuda")]
        {
            if matches!(self, Tensor::CUDA(_)) || matches!(other, Tensor::CUDA(_)) {
                return crate::gpu_ops::cuda::dispatch_oxicuda_matmul_resident(self, other);
            }
            if let (Tensor::F32(a_arr), Tensor::F32(b_arr)) = (self, other) {
                if a_arr.ndim() >= 2
                    && b_arr.ndim() >= 2
                    && !a_arr.is_empty()
                    && !b_arr.is_empty()
                    && crate::gpu_ops::cuda::oxicuda_cuda_available()
                {
                    // Host operands carry no device tag, so use the process-wide default
                    // ordinal (0 unless overridden via `TRUSTFORMERS_CUDA_DEVICE`); the
                    // ordinal is validated against the device count at backend creation.
                    return crate::gpu_ops::cuda::dispatch_oxicuda_matmul(
                        self,
                        other,
                        crate::gpu_ops::cuda::default_cuda_device_id(),
                    );
                }
            }
        }

        // Metal-resident operands are downloaded to host F32, then dispatched to the
        // Metal matmul backend (via `Device::Metal`) which returns a host `Tensor::F32`.
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if matches!(self, Tensor::Metal(_)) || matches!(other, Tensor::Metal(_)) {
                const METAL_DEVICE_ID: usize = 0;
                let device = crate::device::Device::Metal(METAL_DEVICE_ID);
                // Download GPU-resident operands to host F32; leave host tensors as-is.
                let a_host = match self {
                    Tensor::Metal(_) => self.to_device_enum(&crate::device::Device::CPU)?,
                    _ => self.clone(),
                };
                let b_host = match other {
                    Tensor::Metal(_) => other.to_device_enum(&crate::device::Device::CPU)?,
                    _ => other.clone(),
                };
                return crate::gpu_ops::metal::dispatch_matmul(&a_host, &b_host, &device);
            }
        }

        // Ensure both inputs have contiguous memory layouts before any operations.
        //
        // The overwhelmingly common case is that both operands are already in
        // standard (row-major) layout -- callers such as `Linear::forward` force
        // contiguity themselves -- so materialise a copy *only* when the layout
        // actually needs fixing. The previous unconditional
        // `as_standard_layout().into_owned()` deep-copied both operands (96 MiB
        // for a [2048,4096] x [4096,4096] product) before every GEMM.
        let self_contiguous: Cow<'_, Tensor> = match self {
            Tensor::F32(a) if !a.is_standard_layout() => {
                Cow::Owned(Tensor::F32(a.as_standard_layout().into_owned()))
            },
            Tensor::F64(a) if !a.is_standard_layout() => {
                Cow::Owned(Tensor::F64(a.as_standard_layout().into_owned()))
            },
            _ => Cow::Borrowed(self),
        };

        let other_contiguous: Cow<'_, Tensor> = match other {
            Tensor::F32(a) if !a.is_standard_layout() => {
                Cow::Owned(Tensor::F32(a.as_standard_layout().into_owned()))
            },
            Tensor::F64(a) if !a.is_standard_layout() => {
                Cow::Owned(Tensor::F64(a.as_standard_layout().into_owned()))
            },
            _ => Cow::Borrowed(other),
        };

        match (self_contiguous.as_ref(), other_contiguous.as_ref()) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                let a_shape = a.shape();
                let b_shape = b.shape();

                if a_shape.len() < 2 || b_shape.len() < 2 {
                    return Err(TrustformersError::shape_error(
                        "Matrix multiplication requires at least 2D tensors".into(),
                    ));
                }

                let a_last = a_shape[a_shape.len() - 1];
                let b_second_last = b_shape[b_shape.len() - 2];

                if a_last != b_second_last {
                    return Err(TrustformersError::shape_error(format!(
                        "Matrix dimensions mismatch: {} vs {}",
                        a_last, b_second_last
                    )));
                }

                // Handle different dimensionalities
                if a_shape.len() == 2 && b_shape.len() == 2 {
                    // Simple 2D matrix multiplication using BLAS-accelerated GEMM
                    let a_2d = a
                        .view()
                        .into_dimensionality::<Ix2>()
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let b_2d = b
                        .view()
                        .into_dimensionality::<Ix2>()
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;

                    // NOTE: there is deliberately no pre-scan of the operands here.
                    // The former `iter().any(|x| !is_stable_f32(x))` guard read
                    // every element of both matrices (100 MiB of extra traffic for
                    // a [2048,4096] x [4096,4096] product) just to pick a branch,
                    // and it routed ordinary trained weights onto a scalar Kahan
                    // loop that also rewrote sub-1e-7 values. NaN/Inf propagate
                    // through GEMM per IEEE-754, which is the expected behaviour.
                    {
                        // Use BLAS-accelerated GEMM via scirs2-core SimdUnifiedOps
                        // C = alpha * A * B + beta * C
                        // For simple matmul: alpha = 1.0, beta = 0.0
                        let rows = a_2d.nrows();
                        let cols = b_2d.ncols();
                        let inner = a_2d.ncols();

                        // For small matrices, use ndarray's optimized dot() method
                        // Direct BLAS has overhead for very small matrices
                        const MIN_SIZE_FOR_BLAS: usize = 32;
                        if rows < MIN_SIZE_FOR_BLAS
                            || cols < MIN_SIZE_FOR_BLAS
                            || inner < MIN_SIZE_FOR_BLAS
                        {
                            let result = a_2d.dot(&b_2d);
                            Ok(Tensor::F32(result.into_dyn()))
                        } else {
                            // Use direct BLAS (cblas_sgemm on macOS, scirs2-core SIMD elsewhere)
                            // This provides 10-50x speedup via Accelerate framework on macOS
                            let a_slice = a_2d.as_slice().ok_or_else(|| {
                                TrustformersError::tensor_op_error(
                                    "Matrix A is not contiguous",
                                    "matmul",
                                )
                            })?;
                            let b_slice = b_2d.as_slice().ok_or_else(|| {
                                TrustformersError::tensor_op_error(
                                    "Matrix B is not contiguous",
                                    "matmul",
                                )
                            })?;
                            let mut result_vec = vec![0.0f32; rows * cols];
                            blas_sgemm(a_slice, b_slice, &mut result_vec, rows, inner, cols);
                            let result = Array2::from_shape_vec((rows, cols), result_vec)
                                .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                            Ok(Tensor::F32(result.into_dyn()))
                        }
                    }
                } else {
                    // Batched matrix multiplication (3-D and 4-D).
                    //
                    // Both operands are already row-major contiguous at this point,
                    // so the per-(batch, head) `as_standard_layout().into_owned()`
                    // copies the previous implementation performed were pure
                    // overhead, as was the per-iteration `result_vec` + `assign`
                    // round-trip. `batched_gemm_f32` writes each GEMM straight into
                    // its slot of the destination and runs the batches in parallel.
                    let result = batched_gemm_f32(a, b)?;
                    Ok(Tensor::F32(result))
                }
            },
            // Half-precision (F16 / BF16) matmul via f32 upcast.
            //
            // `half::f16` / `half::bf16` have no native BLAS GEMM, so we upcast both
            // operands to f32, perform the multiplication in full single precision
            // (reusing the BLAS-accelerated F32 path above), and then downcast the
            // result back to the original half-precision dtype. This mirrors the
            // standard mixed-precision convention (accumulate in f32, store in f16).
            //
            // Note: `Tensor::to_f32()` / `to_dtype()` do not implement the F16/BF16
            // arms, so the up/down casts are performed inline here using `half`'s
            // lossless `to_f32` and rounding `from_f32` conversions.
            (Tensor::F16(_), _)
            | (_, Tensor::F16(_))
            | (Tensor::BF16(_), _)
            | (_, Tensor::BF16(_)) => {
                let original_dtype = self.dtype();

                // Upcast `self` to an f32 tensor.
                let self_f32 = match self_contiguous.as_ref() {
                    Tensor::F16(a) => Tensor::F32(a.mapv(|x| x.to_f32())),
                    Tensor::BF16(a) => Tensor::F32(a.mapv(|x| x.to_f32())),
                    Tensor::F32(a) => Tensor::F32(a.clone()),
                    other_dtype => {
                        return Err(TrustformersError::tensor_op_error(
                            &format!(
                                "Matmul half-precision upcast not supported for left operand \
                                 dtype {:?}",
                                other_dtype.dtype()
                            ),
                            "matmul",
                        ));
                    },
                };

                // Upcast `other` to an f32 tensor.
                let other_f32 = match other_contiguous.as_ref() {
                    Tensor::F16(b) => Tensor::F32(b.mapv(|x| x.to_f32())),
                    Tensor::BF16(b) => Tensor::F32(b.mapv(|x| x.to_f32())),
                    Tensor::F32(b) => Tensor::F32(b.clone()),
                    other_dtype => {
                        return Err(TrustformersError::tensor_op_error(
                            &format!(
                                "Matmul half-precision upcast not supported for right operand \
                                 dtype {:?}",
                                other_dtype.dtype()
                            ),
                            "matmul",
                        ));
                    },
                };

                // Perform the multiplication in f32 (reuses the F32 BLAS path).
                let result_f32 = self_f32.matmul(&other_f32)?;

                // Downcast the f32 result back to the original half-precision dtype.
                match (original_dtype, result_f32) {
                    (DType::F16, Tensor::F32(r)) => Ok(Tensor::F16(r.mapv(half::f16::from_f32))),
                    (DType::BF16, Tensor::F32(r)) => Ok(Tensor::BF16(r.mapv(half::bf16::from_f32))),
                    // `self` was F32 but `other` was half-precision: keep the f32 result.
                    (_, result) => Ok(result),
                }
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Matmul not supported for these tensor types",
                "matmul",
            )),
        }
    }

    /// Calculate the L2 norm (Euclidean norm) of the tensor.
    ///
    /// Computes the square root of the sum of squares of all elements in the tensor.
    /// This is equivalent to the Euclidean distance from the origin in the tensor's
    /// vector space.
    ///
    /// # Mathematical Definition
    ///
    /// For a tensor `x`, the L2 norm is: `||x||_2 = sqrt(Σ x_i²)`
    ///
    /// # Performance
    ///
    /// Uses SIMD-accelerated computation via scirs2-core when the tensor
    /// can be viewed as a contiguous 1D array.
    ///
    /// # Returns
    ///
    /// The L2 norm as a scalar `f32` value.
    ///
    /// # Errors
    ///
    /// - `TensorOpError`: If the operation is not supported for the tensor type
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = Tensor::from_vec(vec![3.0, 4.0], &[2])?;
    /// let norm = tensor.norm()?; // Should be 5.0 (sqrt(3² + 4²))
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm(&self) -> Result<f32> {
        match self {
            Tensor::F32(a) => {
                // Flatten to 1D and use SIMD-accelerated norm
                let flat = a.as_standard_layout();
                let flat_view = flat
                    .view()
                    .into_shape_with_order(flat.len())
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(f32::simd_norm(&flat_view))
            },
            Tensor::F64(a) => {
                // Flatten to 1D and use SIMD-accelerated norm
                let flat = a.as_standard_layout();
                let flat_view = flat
                    .view()
                    .into_shape_with_order(flat.len())
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(f64::simd_norm(&flat_view) as f32)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Norm not supported for this tensor type",
                "norm",
            )),
        }
    }

    /// Calculate the squared L2 norm of the tensor.
    ///
    /// Computes the sum of squares of all elements in the tensor without taking
    /// the square root. This is computationally more efficient than `norm()` when
    /// only the squared norm is needed.
    ///
    /// # Mathematical Definition
    ///
    /// For a tensor `x`, the squared L2 norm is: `||x||_2² = Σ x_i²`
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the squared norm value.
    ///
    /// # Errors
    ///
    /// - `TensorOpError`: If the operation is not supported for the tensor type
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = Tensor::from_vec(vec![3.0, 4.0], &[2])?;
    /// let norm_squared = tensor.norm_squared()?; // Should be 25.0 (3² + 4²)
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm_squared(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let sum_squares = a.mapv(|x| x * x).sum();
                Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), sum_squares)))
            },
            Tensor::F64(a) => {
                let sum_squares = a.mapv(|x| x * x).sum();
                Ok(Tensor::F64(ArrayD::from_elem(IxDyn(&[]), sum_squares)))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Norm squared not supported for this tensor type",
                "norm_squared",
            )),
        }
    }

    /// Clip gradients based on their norm to prevent gradient explosion.
    ///
    /// This function implements gradient clipping by scaling the tensor values
    /// to ensure the L2 norm does not exceed the specified maximum value.
    /// This is a common technique used in training deep neural networks to
    /// prevent gradient explosion.
    ///
    /// # Algorithm
    ///
    /// 1. Calculate the current L2 norm of the tensor
    /// 2. If norm ≤ max_norm, return the tensor unchanged
    /// 3. If norm > max_norm, scale the tensor by (max_norm / norm)
    ///
    /// # Arguments
    ///
    /// * `max_norm` - The maximum allowed norm value
    ///
    /// # Returns
    ///
    /// A new tensor with clipped gradient values.
    ///
    /// # Errors
    ///
    /// - `TensorOpError`: If norm calculation or scalar multiplication fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create a tensor with large gradient values
    /// let gradients = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3])?;
    ///
    /// // Clip to maximum norm of 1.0
    /// let clipped = gradients.clip_grad_norm(1.0)?;
    ///
    /// // The resulting tensor will have norm ≤ 1.0
    /// assert!(clipped.norm()? <= 1.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Use in Training
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let gradients = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3])?;
    /// // Typical usage in gradient clipping during training
    /// let max_gradient_norm = 1.0;
    /// let clipped_gradients = gradients.clip_grad_norm(max_gradient_norm)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn clip_grad_norm(&self, max_norm: f32) -> Result<Tensor> {
        let norm = self.norm()?;
        if norm > max_norm {
            self.scalar_mul(max_norm / norm)
        } else {
            Ok(self.clone())
        }
    }

    /// Calculate L2 norm along specified dimension(s).
    ///
    /// This function computes the L2 norm along one or more dimensions, which is
    /// useful for normalization operations (e.g., in contrastive learning, CLIP models).
    ///
    /// # Arguments
    ///
    /// * `p` - The order of the norm (typically 2 for L2 norm)
    /// * `dims` - Optional dimensions along which to compute the norm. If None, computes
    ///   the norm across all dimensions (equivalent to `norm()`).
    /// * `keepdim` - If true, keeps the reduced dimensions with size 1
    ///
    /// # Returns
    ///
    /// A tensor containing the L2 norm values along the specified dimensions.
    ///
    /// # Errors
    ///
    /// - `TensorOpError`: If the operation is not supported for the tensor type
    /// - `ShapeError`: If the specified dimensions are out of bounds
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create a 2D tensor
    /// let tensor = Tensor::from_vec(vec![3.0, 4.0, 1.0, 2.0], &[2, 2])?;
    ///
    /// // Compute L2 norm along last dimension
    /// let norm = tensor.norm_dim(2, Some(vec![-1]), true)?;
    /// // Result: [[5.0], [sqrt(5)]]
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm_dim(&self, p: i32, dims: Option<Vec<i32>>, keepdim: bool) -> Result<Tensor> {
        if p != 2 {
            return Err(TrustformersError::tensor_op_error(
                &format!("Only L2 norm (p=2) is currently supported, got p={}", p),
                "norm_dim",
            ));
        }

        // If no dims specified, compute global norm
        if dims.is_none() {
            let global_norm = self.norm()?;
            return Tensor::from_vec(vec![global_norm], &[1]);
        }

        let dims = dims.ok_or_else(|| {
            crate::errors::compute_error("norm_dim", "dims checked as Some above")
        })?;

        match self {
            Tensor::F32(arr) => {
                let mut result = arr.clone();

                // Convert negative dimensions to positive
                let ndim = arr.ndim() as i32;
                let mut positive_dims: Vec<usize> = dims
                    .iter()
                    .map(|&d| {
                        let pos_d = if d < 0 { ndim + d } else { d };
                        if pos_d < 0 || pos_d >= ndim {
                            return Err(TrustformersError::shape_error(format!(
                                "Dimension {} is out of bounds for tensor with {} dimensions",
                                d, ndim
                            )));
                        }
                        Ok(pos_d as usize)
                    })
                    .collect::<Result<Vec<_>>>()?;

                // Sort in descending order to remove dimensions from back to front
                positive_dims.sort_unstable_by(|a, b| b.cmp(a));

                // Square all values
                result.mapv_inplace(|x| x * x);

                // Sum along specified dimensions
                for &dim in &positive_dims {
                    result = result.sum_axis(Axis(dim));
                }

                // Take square root
                result.mapv_inplace(|x| x.sqrt());

                // Add back dimensions if keepdim is true
                if keepdim {
                    let mut new_shape = arr.shape().to_vec();
                    for &dim in &positive_dims {
                        new_shape[dim] = 1;
                    }
                    result = result.to_shape(new_shape)?.to_owned();
                }

                Ok(Tensor::F32(result))
            },
            Tensor::F64(arr) => {
                let mut result = arr.clone();

                // Convert negative dimensions to positive
                let ndim = arr.ndim() as i32;
                let mut positive_dims: Vec<usize> = dims
                    .iter()
                    .map(|&d| {
                        let pos_d = if d < 0 { ndim + d } else { d };
                        if pos_d < 0 || pos_d >= ndim {
                            return Err(TrustformersError::shape_error(format!(
                                "Dimension {} is out of bounds for tensor with {} dimensions",
                                d, ndim
                            )));
                        }
                        Ok(pos_d as usize)
                    })
                    .collect::<Result<Vec<_>>>()?;

                // Sort in descending order
                positive_dims.sort_unstable_by(|a, b| b.cmp(a));

                // Square all values
                result.mapv_inplace(|x| x * x);

                // Sum along specified dimensions
                for &dim in &positive_dims {
                    result = result.sum_axis(Axis(dim));
                }

                // Take square root
                result.mapv_inplace(|x| x.sqrt());

                // Add back dimensions if keepdim is true
                if keepdim {
                    let mut new_shape = arr.shape().to_vec();
                    for &dim in &positive_dims {
                        new_shape[dim] = 1;
                    }
                    result = result.to_shape(new_shape)?.to_owned();
                }

                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "norm_dim not supported for this tensor type",
                "norm_dim",
            )),
        }
    }
}

#[cfg(test)]
#[allow(unused_variables)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Ix0;

    #[test]
    fn test_matmul_2d() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
        let result = a.matmul(&b)?;

        if let Tensor::F32(arr) = result {
            assert_eq!(arr.shape(), &[2, 2]);
            // Expected: [[19, 22], [43, 50]]
            assert!((arr[[0, 0]] - 19.0).abs() < 1e-6);
            assert!((arr[[0, 1]] - 22.0).abs() < 1e-6);
            assert!((arr[[1, 0]] - 43.0).abs() < 1e-6);
            assert!((arr[[1, 1]] - 50.0).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_norm() -> Result<()> {
        let tensor = Tensor::from_vec(vec![3.0, 4.0], &[2])?;
        let norm = tensor.norm()?;
        assert!((norm - 5.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_norm_squared() -> Result<()> {
        let tensor = Tensor::from_vec(vec![3.0, 4.0], &[2])?;
        let norm_squared = tensor.norm_squared()?;

        if let Tensor::F32(arr) = norm_squared {
            assert!(
                (arr.into_dimensionality::<Ix0>()
                    .expect("norm_squared should produce a scalar")
                    .into_scalar()
                    - 25.0)
                    .abs()
                    < 1e-6
            );
        }
        Ok(())
    }

    #[test]
    fn test_clip_grad_norm() -> Result<()> {
        let tensor = Tensor::from_vec(vec![10.0, 20.0], &[2])?;
        let clipped = tensor.clip_grad_norm(1.0)?;
        let norm = clipped.norm()?;
        assert!((norm - 1.0).abs() < 1e-6);
        Ok(())
    }

    /// Build a deterministic F16 tensor from f32 values for testing.
    fn f16_tensor(data: &[f32], shape: &[usize]) -> Result<Tensor> {
        use scirs2_core::ndarray::{ArrayD, IxDyn};
        let half_data: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
        Ok(Tensor::F16(
            ArrayD::from_shape_vec(IxDyn(shape), half_data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Build a deterministic BF16 tensor from f32 values for testing.
    fn bf16_tensor(data: &[f32], shape: &[usize]) -> Result<Tensor> {
        use scirs2_core::ndarray::{ArrayD, IxDyn};
        let half_data: Vec<half::bf16> = data.iter().map(|&x| half::bf16::from_f32(x)).collect();
        Ok(Tensor::BF16(
            ArrayD::from_shape_vec(IxDyn(shape), half_data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    #[test]
    fn test_matmul_f16_upcast() -> Result<()> {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] -> [[19, 22], [43, 50]]
        let a = f16_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = f16_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2])?;

        let result = a.matmul(&b)?;

        // Dtype must be preserved as F16 (upcast path downcasts the result back).
        assert_eq!(result.dtype(), DType::F16);

        match result {
            Tensor::F16(arr) => {
                // Shape must be correct.
                assert_eq!(arr.shape(), &[2, 2]);
                // All entries must be finite.
                assert!(arr.iter().all(|x| x.to_f32().is_finite()));
                // Values must match the expected f32 result within f16 rounding error.
                let expected = [19.0_f32, 22.0, 43.0, 50.0];
                for (idx, &exp) in expected.iter().enumerate() {
                    let i = idx / 2;
                    let j = idx % 2;
                    let got = arr[[i, j]].to_f32();
                    assert!(
                        (got - exp).abs() < 0.5,
                        "F16 matmul mismatch at [{}, {}]: got {}, expected {}",
                        i,
                        j,
                        got,
                        exp
                    );
                }
            },
            other => panic!("expected Tensor::F16 result, got {:?}", other),
        }
        Ok(())
    }

    #[test]
    fn test_matmul_bf16_upcast() -> Result<()> {
        // Non-square shapes to exercise the shape-propagation path: [2x3] x [3x2] -> [2x2].
        let a = bf16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let b = bf16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;

        let result = a.matmul(&b)?;

        // Dtype must be preserved as BF16.
        assert_eq!(result.dtype(), DType::BF16);

        match result {
            Tensor::BF16(arr) => {
                // Shape must be correct.
                assert_eq!(arr.shape(), &[2, 2]);
                // All entries must be finite.
                assert!(arr.iter().all(|x| x.to_f32().is_finite()));
                // Expected: A @ B = [[22, 28], [49, 64]].
                let expected = [22.0_f32, 28.0, 49.0, 64.0];
                for (idx, &exp) in expected.iter().enumerate() {
                    let i = idx / 2;
                    let j = idx % 2;
                    let got = arr[[i, j]].to_f32();
                    // BF16 has ~8 bits of mantissa: allow a generous relative tolerance.
                    assert!(
                        (got - exp).abs() < 1.0,
                        "BF16 matmul mismatch at [{}, {}]: got {}, expected {}",
                        i,
                        j,
                        got,
                        exp
                    );
                }
            },
            other => panic!("expected Tensor::BF16 result, got {:?}", other),
        }
        Ok(())
    }
}
