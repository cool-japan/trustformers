//! Batched / broadcast matrix multiplication for the oxicuda CUDA backend.
//!
//! This module carries the `Tensor::matmul` routing surface of the oxicuda backend:
//!
//! - [`BatchedMatmulPlan`] — pure host-side shape logic that turns two operand shapes
//!   into GEMM dimensions plus per-batch element offsets, implementing standard
//!   broadcasting-batch semantics (leading batch dims broadcast right-aligned, missing
//!   dims act as 1 — the NumPy `matmul` convention, a superset of the CPU path's
//!   equal-batch 3D/4D support). Being pure, it is unit-testable on any machine.
//! - Backend execution methods (`matmul_batched_f32`, `matmul_batched_gpu_to_gpu`)
//!   that run one GEMM per batch entry on the backend's stream.
//! - The two `Tensor`-level dispatch functions used by `Tensor::matmul`:
//!   [`dispatch_oxicuda_matmul`] (host-in / host-out) and
//!   [`dispatch_oxicuda_matmul_resident`] (GPU-resident operands / results).
//!
//! # Why a per-batch GEMM loop instead of `oxicuda_blas::batched::gemm_strided_batched`
//!
//! oxicuda-blas 0.4.0's batched entry points launched their 8-parameter GEMM kernel
//! with mis-ordered 17/13-element argument tuples (dimensions marshalled into pointer
//! slots) and could not produce correct results. That launch-tuple corruption is
//! fixed in oxicuda-blas 0.4.1 (verified against the published sources, 2026-07-07):
//! `gemm_strided_batched` now validates its arguments, honestly rejects unsupported
//! transpose/leading-dimension combinations, and launches a correctly matched tuple.
//! The local loop is nevertheless retained, for two reasons that are not bugs:
//!
//! 1. **Generality.** [`BatchedMatmulPlan`] produces arbitrary per-batch element
//!    offsets to express NumPy-style multi-axis broadcasting (e.g. `A` varying along
//!    the inner batch axis while `B` varies along the outer one). A single uniform
//!    element stride — all `gemm_strided_batched` can express — cannot encode those
//!    offset patterns.
//! 2. **No launch savings.** The 0.4.1 strided implementation is itself a sequential
//!    per-batch launch loop over the same single-GEMM kernel, so routing through it
//!    would not reduce kernel launches.
//!
//! Every iteration here is one asynchronous kernel launch on the backend's single
//! stream and all operands stay device-resident, so there is no host round-trip —
//! only the per-batch launch overhead any strided path would currently incur too.

use oxicuda_blas::level3::gemm_api::gemm;
use oxicuda_blas::{Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_memory::DeviceBuffer;
use scirs2_core::ndarray::{ArrayD, IxDyn};

use super::{oxicuda_backend, OxiCudaBufferHandle, OxiCudaBufferId, OxicudaCudaBackend};
use crate::errors::TrustformersError;

// ---------------------------------------------------------------------------
// Shape planning (pure host-side logic, hardware-free)
// ---------------------------------------------------------------------------

/// Execution plan for a (possibly batched, possibly broadcast) matrix multiplication.
///
/// Produced by [`BatchedMatmulPlan::new`] from the two operand shapes. The last two
/// axes of each operand are the matrix dimensions (`[.., m, k] @ [.., k, n]`); all
/// leading axes are batch dimensions and broadcast against each other right-aligned,
/// with missing leading dims treated as 1 (standard NumPy `matmul` semantics — for
/// the matched-rank equal-batch 3D/4D shapes the CPU path supports, this reproduces
/// the CPU path's batch layout exactly).
///
/// `a_offsets`/`b_offsets` hold, for every flattened output batch index (row-major
/// over `batch_shape`), the *element* offset of the corresponding input matrix within
/// its operand's flat row-major buffer. A broadcast operand contributes offset stride
/// 0 along the broadcast axis, so the same matrix is reused across those batches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchedMatmulPlan {
    /// Rows of each A matrix (and of each output matrix).
    pub m: usize,
    /// Shared inner dimension (columns of A, rows of B).
    pub k: usize,
    /// Columns of each B matrix (and of each output matrix).
    pub n: usize,
    /// Broadcast batch shape (empty for a plain 2D x 2D multiply).
    pub batch_shape: Vec<usize>,
    /// Per-output-batch element offset into A's flat buffer.
    pub a_offsets: Vec<usize>,
    /// Per-output-batch element offset into B's flat buffer.
    pub b_offsets: Vec<usize>,
}

/// Right-aligned broadcast strides (in units of one matrix) for one operand's batch
/// dims against the broadcast `batch_shape`. Axes where the operand has dim 1 (or no
/// dim at all) get stride 0, so every output batch index along that axis maps to the
/// operand's single matrix.
fn broadcast_strides(operand_batch: &[usize], batch_shape: &[usize]) -> Vec<usize> {
    let rank = batch_shape.len();
    let mut strides = vec![0usize; rank];
    let lead = rank - operand_batch.len();
    let mut acc = 1usize;
    for (i, &dim) in operand_batch.iter().enumerate().rev() {
        strides[lead + i] = if dim == 1 { 0 } else { acc };
        acc = acc.saturating_mul(dim);
    }
    strides
}

impl BatchedMatmulPlan {
    /// Build a plan from the two operand shapes.
    ///
    /// # Errors
    ///
    /// - `ShapeError` if either operand has fewer than 2 dimensions (mirroring the
    ///   CPU path's requirement).
    /// - `ShapeError` if the inner matrix dimensions disagree (`a[-1] != b[-2]`).
    /// - `ShapeError` if the leading batch dimensions cannot be broadcast (unequal
    ///   and neither is 1).
    pub fn new(a_shape: &[usize], b_shape: &[usize]) -> crate::errors::Result<Self> {
        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(TrustformersError::shape_error(
                "Matrix multiplication requires at least 2D tensors".into(),
            ));
        }

        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let k_b = b_shape[b_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];
        if k != k_b {
            return Err(TrustformersError::shape_error(format!(
                "Matrix dimensions mismatch: {} vs {}",
                k, k_b
            )));
        }

        let a_batch = &a_shape[..a_shape.len() - 2];
        let b_batch = &b_shape[..b_shape.len() - 2];
        let rank = a_batch.len().max(b_batch.len());

        // Broadcast the batch dims right-aligned (missing leading dims act as 1).
        let mut batch_shape = vec![0usize; rank];
        for axis_from_right in 0..rank {
            let da = if axis_from_right < a_batch.len() {
                a_batch[a_batch.len() - 1 - axis_from_right]
            } else {
                1
            };
            let db = if axis_from_right < b_batch.len() {
                b_batch[b_batch.len() - 1 - axis_from_right]
            } else {
                1
            };
            let out = if da == db {
                da
            } else if da == 1 {
                db
            } else if db == 1 {
                da
            } else {
                return Err(TrustformersError::shape_error(format!(
                    "Cannot broadcast matmul batch dimensions {:?} vs {:?}",
                    a_batch, b_batch
                )));
            };
            batch_shape[rank - 1 - axis_from_right] = out;
        }

        // Total number of output matrices; zero-sized batch dims yield an empty plan.
        let batch_count = batch_shape.iter().try_fold(1usize, |acc, &d| {
            acc.checked_mul(d).ok_or_else(|| {
                TrustformersError::shape_error(format!(
                    "Matmul batch shape {:?} overflows usize",
                    batch_shape
                ))
            })
        })?;

        let a_mat = m.checked_mul(k).ok_or_else(|| {
            TrustformersError::shape_error(format!("Matrix size {}x{} overflows usize", m, k))
        })?;
        let b_mat = k.checked_mul(n).ok_or_else(|| {
            TrustformersError::shape_error(format!("Matrix size {}x{} overflows usize", k, n))
        })?;

        let a_strides = broadcast_strides(a_batch, &batch_shape);
        let b_strides = broadcast_strides(b_batch, &batch_shape);

        // Unravel every flat output batch index (row-major over `batch_shape`) into a
        // multi-index and project it through each operand's broadcast strides.
        let mut a_offsets = Vec::with_capacity(batch_count);
        let mut b_offsets = Vec::with_capacity(batch_count);
        for flat in 0..batch_count {
            let mut rem = flat;
            let mut a_off = 0usize;
            let mut b_off = 0usize;
            for i in (0..rank).rev() {
                let idx = rem % batch_shape[i];
                rem /= batch_shape[i];
                a_off += idx * a_strides[i];
                b_off += idx * b_strides[i];
            }
            a_offsets.push(a_off * a_mat);
            b_offsets.push(b_off * b_mat);
        }

        Ok(Self {
            m,
            k,
            n,
            batch_shape,
            a_offsets,
            b_offsets,
        })
    }

    /// Number of output matrices (0 when a batch dimension is zero-sized).
    #[inline]
    pub fn batch_count(&self) -> usize {
        self.a_offsets.len()
    }

    /// Shape of the multiplication result: `batch_shape ++ [m, n]`.
    pub fn output_shape(&self) -> Vec<usize> {
        let mut shape = self.batch_shape.clone();
        shape.push(self.m);
        shape.push(self.n);
        shape
    }

    /// `true` when there is nothing to run on the GPU (zero-sized batch or a
    /// zero-sized matrix dimension). Degenerate problems are delegated to the CPU
    /// path, which produces the correct empty result without touching the driver.
    #[inline]
    pub fn is_degenerate(&self) -> bool {
        self.batch_count() == 0 || self.m == 0 || self.k == 0 || self.n == 0
    }
}

// ---------------------------------------------------------------------------
// Backend execution
// ---------------------------------------------------------------------------

/// Shared argument validation for the two public batched entry points.
///
/// Returns the batch count. Zero-sized problems are rejected here (the GEMM kernel
/// requires positive dimensions and oxicuda rejects zero-length device allocations);
/// callers are expected to route degenerate plans to the CPU path instead.
fn validate_batched_matmul_args(
    m: usize,
    k: usize,
    n: usize,
    a_batch_offsets: &[usize],
    b_batch_offsets: &[usize],
    op: &str,
) -> crate::errors::Result<usize> {
    if m == 0 || k == 0 || n == 0 {
        return Err(TrustformersError::shape_error(format!(
            "{}: GEMM dimensions must be non-zero (m={}, k={}, n={})",
            op, m, k, n
        )));
    }
    if a_batch_offsets.len() != b_batch_offsets.len() {
        return Err(TrustformersError::shape_error(format!(
            "{}: offset counts differ ({} vs {})",
            op,
            a_batch_offsets.len(),
            b_batch_offsets.len()
        )));
    }
    if a_batch_offsets.is_empty() {
        return Err(TrustformersError::shape_error(format!(
            "{}: batch count must be non-zero",
            op
        )));
    }
    Ok(a_batch_offsets.len())
}

impl OxicudaCudaBackend {
    /// Core batched GEMM loop: for every batch entry, `C[i] = A[a_off[i]..] @ B[b_off[i]..]`,
    /// with all matrices row-major (`A` is `[m, k]`, `B` is `[k, n]`, `C` is `[m, n]`)
    /// and the outputs packed contiguously into `c_buf` in batch order.
    ///
    /// Per-batch pointer views are built with `MatrixDesc::from_raw` (element offsets
    /// scaled to byte offsets), which performs no size validation — so every offset is
    /// bounds-checked against its buffer here first. Each `gemm` call is one
    /// asynchronous launch on the backend's stream; see the module docs for why this
    /// loop is used instead of `gemm_strided_batched` (arbitrary broadcast offsets;
    /// no launch savings upstream).
    #[allow(clippy::too_many_arguments)]
    fn batched_gemm_into(
        &self,
        a_buf: &DeviceBuffer<f32>,
        b_buf: &DeviceBuffer<f32>,
        c_buf: &mut DeviceBuffer<f32>,
        m: usize,
        k: usize,
        n: usize,
        a_batch_offsets: &[usize],
        b_batch_offsets: &[usize],
    ) -> crate::errors::Result<()> {
        let m_u32 = u32::try_from(m).map_err(|_| {
            TrustformersError::shape_error(format!("Matrix dimension m={} exceeds u32", m))
        })?;
        let k_u32 = u32::try_from(k).map_err(|_| {
            TrustformersError::shape_error(format!("Matrix dimension k={} exceeds u32", k))
        })?;
        let n_u32 = u32::try_from(n).map_err(|_| {
            TrustformersError::shape_error(format!("Matrix dimension n={} exceeds u32", n))
        })?;

        let a_mat = m * k;
        let b_mat = k * n;
        let c_mat = m * n;
        let elem = std::mem::size_of::<f32>() as u64;
        let a_base = a_buf.as_device_ptr();
        let b_base = b_buf.as_device_ptr();
        let c_base = c_buf.as_device_ptr();

        for (batch_idx, (&a_off, &b_off)) in
            a_batch_offsets.iter().zip(b_batch_offsets.iter()).enumerate()
        {
            // Bounds checks stand in for the size validation `from_raw` skips.
            let a_end = a_off.checked_add(a_mat).filter(|&end| end <= a_buf.len());
            if a_end.is_none() {
                return Err(TrustformersError::shape_error(format!(
                    "Batch {}: A offset {} + {}x{} matrix exceeds buffer length {}",
                    batch_idx,
                    a_off,
                    m,
                    k,
                    a_buf.len()
                )));
            }
            let b_end = b_off.checked_add(b_mat).filter(|&end| end <= b_buf.len());
            if b_end.is_none() {
                return Err(TrustformersError::shape_error(format!(
                    "Batch {}: B offset {} + {}x{} matrix exceeds buffer length {}",
                    batch_idx,
                    b_off,
                    k,
                    n,
                    b_buf.len()
                )));
            }
            let c_off = batch_idx * c_mat;
            if c_off + c_mat > c_buf.len() {
                return Err(TrustformersError::shape_error(format!(
                    "Batch {}: C offset {} + {}x{} matrix exceeds buffer length {}",
                    batch_idx,
                    c_off,
                    m,
                    n,
                    c_buf.len()
                )));
            }

            let a_desc = MatrixDesc::<f32>::from_raw(
                a_base + (a_off as u64) * elem,
                m_u32,
                k_u32,
                k_u32,
                Layout::RowMajor,
            );
            let b_desc = MatrixDesc::<f32>::from_raw(
                b_base + (b_off as u64) * elem,
                k_u32,
                n_u32,
                n_u32,
                Layout::RowMajor,
            );
            let mut c_desc = MatrixDescMut::<f32>::from_raw(
                c_base + (c_off as u64) * elem,
                m_u32,
                n_u32,
                n_u32,
                Layout::RowMajor,
            );

            gemm::<f32>(
                &self.handle,
                Transpose::NoTrans,
                Transpose::NoTrans,
                1.0f32,
                &a_desc,
                &b_desc,
                0.0f32,
                &mut c_desc,
            )
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Batched GEMM failed at batch {}: {}", batch_idx, e),
                    "batched_gemm_into",
                )
            })?;
        }

        Ok(())
    }

    /// Batched matrix multiplication, host-in / host-out.
    ///
    /// `a` and `b` are flat row-major buffers holding one or more `[m, k]` / `[k, n]`
    /// matrices; `a_batch_offsets[i]` / `b_batch_offsets[i]` locate the operand pair
    /// of output batch `i` (element offsets, typically produced by
    /// [`BatchedMatmulPlan`], with repeated offsets expressing broadcast). The result
    /// is `batch_count` row-major `[m, n]` matrices packed contiguously.
    ///
    /// Both operands are uploaded once and every per-batch GEMM runs against the same
    /// device buffers, so broadcast operands are not duplicated on the device.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_batched_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        a_batch_offsets: &[usize],
        b_batch_offsets: &[usize],
    ) -> crate::errors::Result<Vec<f32>> {
        let batch_count = validate_batched_matmul_args(
            m,
            k,
            n,
            a_batch_offsets,
            b_batch_offsets,
            "matmul_batched_f32",
        )?;

        let a_buf = DeviceBuffer::<f32>::from_host(a).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to upload matrix batch A to device: {}", e),
                "matmul_batched_f32",
            )
        })?;
        let b_buf = DeviceBuffer::<f32>::from_host(b).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to upload matrix batch B to device: {}", e),
                "matmul_batched_f32",
            )
        })?;

        let c_len = batch_count.checked_mul(m * n).ok_or_else(|| {
            TrustformersError::shape_error(format!(
                "Result size {} x {}x{} overflows usize",
                batch_count, m, n
            ))
        })?;
        // Zero-initialised: the GEMM epilogue reads C_old even with beta = 0, and
        // 0.0 * NaN garbage from an uninitialised allocation would poison the result.
        let mut c_buf = DeviceBuffer::<f32>::zeroed(c_len).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate result buffer on device: {}", e),
                "matmul_batched_f32",
            )
        })?;

        self.batched_gemm_into(
            &a_buf,
            &b_buf,
            &mut c_buf,
            m,
            k,
            n,
            a_batch_offsets,
            b_batch_offsets,
        )?;

        let mut result = vec![0.0f32; c_len];
        // The per-batch GEMMs launched asynchronously on the backend's
        // non-blocking stream must finish before the synchronous device→host
        // copy (on the default stream) reads the result — otherwise it returns
        // the zero-initialised buffer. See `OxicudaCudaBackend::synchronize_stream`.
        self.synchronize_stream("matmul_batched_f32")?;
        c_buf.copy_to_host(&mut result).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result back to host: {}", e),
                "matmul_batched_f32",
            )
        })?;
        Ok(result)
    }

    /// Batched matrix multiplication over two GPU-resident persistent buffers,
    /// leaving the packed result on the device.
    ///
    /// The batched analogue of [`matmul_gpu_to_gpu`](Self::matmul_gpu_to_gpu):
    /// operands are located inside their resident buffers via the per-batch element
    /// offsets (see [`matmul_batched_f32`](Self::matmul_batched_f32) for the offset
    /// contract), the freshly allocated `batch_count * m * n` result is inserted into
    /// the cache, and its new id is returned. No host round-trip occurs.
    ///
    /// Borrow handling matches the other `*_gpu_to_gpu` ops: the output is a *local*
    /// [`DeviceBuffer`] (not yet in the map), so it can be borrowed mutably for the
    /// GEMM loop while A and B are borrowed immutably from the cache — no aliasing
    /// with the map, one lock acquisition, no `unwrap`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_batched_gpu_to_gpu(
        &self,
        a_buffer_id: &OxiCudaBufferId,
        b_buffer_id: &OxiCudaBufferId,
        m: usize,
        k: usize,
        n: usize,
        a_batch_offsets: &[usize],
        b_batch_offsets: &[usize],
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let batch_count = validate_batched_matmul_args(
            m,
            k,
            n,
            a_batch_offsets,
            b_batch_offsets,
            "matmul_batched_gpu_to_gpu",
        )?;

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "matmul_batched_gpu_to_gpu",
            )
        })?;

        let a_buf = cache.get(a_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Input buffer {:?} not found in cache", a_buffer_id),
                "matmul_batched_gpu_to_gpu",
            )
        })?;
        let b_buf = cache.get(b_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Input buffer {:?} not found in cache", b_buffer_id),
                "matmul_batched_gpu_to_gpu",
            )
        })?;

        let c_len = batch_count.checked_mul(m * n).ok_or_else(|| {
            TrustformersError::shape_error(format!(
                "Result size {} x {}x{} overflows usize",
                batch_count, m, n
            ))
        })?;
        // Local allocation (not in the map); zero-initialised for the same reason as
        // in `matmul_batched_f32`.
        let mut c_buf = DeviceBuffer::<f32>::zeroed(c_len).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate result buffer on device: {}", e),
                "matmul_batched_gpu_to_gpu",
            )
        })?;

        self.batched_gemm_into(
            a_buf,
            b_buf,
            &mut c_buf,
            m,
            k,
            n,
            a_batch_offsets,
            b_batch_offsets,
        )?;

        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, c_buf);
        Ok(output_id)
    }
}

// ---------------------------------------------------------------------------
// Tensor-level dispatch
// ---------------------------------------------------------------------------

/// Dispatch matrix multiplication of two **host** tensors to the oxicuda CUDA backend.
///
/// For `F32` operands of any rank >= 2 this uploads both operands to the GPU, runs one
/// GEMM per (broadcast) batch entry, and returns a host-resident `Tensor::F32` shaped
/// `batch_shape ++ [m, n]` (see [`BatchedMatmulPlan`] for the broadcasting rules; a
/// plain 2D x 2D multiply is the `batch_count == 1` special case). Degenerate
/// (zero-sized) problems and all other dtypes fall back to the CPU `Tensor::matmul`
/// path.
///
/// The backend is obtained through the per-device [`oxicuda_backend()`] singleton
/// rather than constructed fresh per call, so the CUDA context / cuBLAS handle and any
/// GPU-resident buffers are shared across invocations on the same device ordinal.
pub fn dispatch_oxicuda_matmul(
    a: &crate::tensor::Tensor,
    b: &crate::tensor::Tensor,
    device_id: usize,
) -> crate::errors::Result<crate::tensor::Tensor> {
    use crate::tensor::Tensor;

    match (a, b) {
        (Tensor::F32(a_arr), Tensor::F32(b_arr)) => {
            let plan = BatchedMatmulPlan::new(a_arr.shape(), b_arr.shape())?;
            if plan.is_degenerate() {
                // Nothing to launch; the CPU path produces the correct empty result.
                // (`Tensor::matmul` never routes empty operands here, so no recursion.)
                return a.matmul(b);
            }

            // `iter()` walks in logical row-major order regardless of the memory
            // layout, so this doubles as the standard-layout normalisation.
            let a_data: Vec<f32> = a_arr.iter().copied().collect();
            let b_data: Vec<f32> = b_arr.iter().copied().collect();

            let backend = oxicuda_backend(device_id)?;
            let result = backend.matmul_batched_f32(
                &a_data,
                &b_data,
                plan.m,
                plan.k,
                plan.n,
                &plan.a_offsets,
                &plan.b_offsets,
            )?;

            let out = ArrayD::from_shape_vec(IxDyn(&plan.output_shape()), result).map_err(|e| {
                TrustformersError::shape_error(format!("Failed to reshape result: {}", e))
            })?;
            Ok(Tensor::F32(out))
        },
        _ => a.matmul(b),
    }
}

/// Upload one host `F32` operand next to a resident operand and run the batched
/// multiply on the resident operand's device.
///
/// The temporary upload is held through an [`OxiCudaBufferHandle`], so its device
/// memory is released when this function returns — on success *and* on every error
/// path — via the refcounted buffer lifecycle.
fn resident_matmul_with_host_upload(
    resident: &crate::tensor::CudaTensorData,
    host: &ArrayD<f32>,
    resident_is_lhs: bool,
) -> crate::errors::Result<crate::tensor::Tensor> {
    use crate::tensor::{CudaTensorData, DType, Tensor};

    let device_id = resident.device_id();
    let plan = if resident_is_lhs {
        BatchedMatmulPlan::new(&resident.shape, host.shape())?
    } else {
        BatchedMatmulPlan::new(host.shape(), &resident.shape)?
    };
    if plan.is_degenerate() {
        // Zero-sized host operand: nothing can be uploaded (oxicuda rejects empty
        // allocations); download the resident side and let the CPU path build the
        // correct empty result.
        let resident_host =
            Tensor::CUDA(resident.clone()).to_device_enum(&crate::device::Device::CPU)?;
        let host_tensor = Tensor::F32(host.clone());
        return if resident_is_lhs {
            resident_host.matmul(&host_tensor)
        } else {
            host_tensor.matmul(&resident_host)
        };
    }

    let host_data: Vec<f32> = host.iter().copied().collect();
    let backend = oxicuda_backend(device_id)?;
    let uploaded_id = backend.create_persistent_buffer(&host_data)?;
    // RAII guard: the temporary upload is freed when `uploaded` drops.
    let uploaded = OxiCudaBufferHandle::new(uploaded_id, device_id);

    let output_id = if resident_is_lhs {
        backend.matmul_batched_gpu_to_gpu(
            &resident.buffer_id(),
            &uploaded.id(),
            plan.m,
            plan.k,
            plan.n,
            &plan.a_offsets,
            &plan.b_offsets,
        )?
    } else {
        backend.matmul_batched_gpu_to_gpu(
            &uploaded.id(),
            &resident.buffer_id(),
            plan.m,
            plan.k,
            plan.n,
            &plan.a_offsets,
            &plan.b_offsets,
        )?
    };

    Ok(Tensor::CUDA(CudaTensorData::new(
        output_id,
        device_id,
        plan.output_shape(),
        DType::F32,
    )))
}

/// Dispatch matrix multiplication involving at least one GPU-resident `Tensor::CUDA`
/// operand.
///
/// Routing policy (each case documented at its arm):
///
/// - **CUDA x CUDA, same device**: fully resident batched GEMM — no host round-trip;
///   the result is a new `Tensor::CUDA` whose buffer is owned by the refcounted
///   lifecycle (freed when the last tensor clone drops).
/// - **CUDA x CUDA, different devices**: no peer-to-peer path exists yet, so both
///   operands bounce through the host and the multiply re-enters `Tensor::matmul`
///   as a host op (mirroring `to_device_enum`'s cross-device transfer policy).
/// - **CUDA x host-F32 (either order)**: the resident operand pins the computation
///   to its device; the host operand is uploaded there for the duration of the op
///   (temporary buffer freed on drop) and the result stays resident.
/// - **CUDA x host non-F32**: the resident side is downloaded and the host dtype
///   rules (e.g. the F16/BF16 upcast path) apply unchanged.
///
/// No `oxicuda_cuda_available()` probe is needed here: a `Tensor::CUDA` can only
/// exist if a runtime-checked upload already succeeded on this host.
pub fn dispatch_oxicuda_matmul_resident(
    a: &crate::tensor::Tensor,
    b: &crate::tensor::Tensor,
) -> crate::errors::Result<crate::tensor::Tensor> {
    use crate::device::Device;
    use crate::tensor::{CudaTensorData, DType, Tensor};

    // Resident buffers are F32-only today (all upload paths convert to f32); other
    // dtypes have no download path either, so reject them explicitly. Device-side
    // half precision is deliberately out of scope for 0.2.x (recorded as 0.3.x work).
    fn require_f32(data: &CudaTensorData) -> crate::errors::Result<()> {
        if data.dtype != DType::F32 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "CUDA-resident matmul supports F32 buffers only (got {:?})",
                    data.dtype
                ),
                "matmul",
            ));
        }
        Ok(())
    }

    match (a, b) {
        (Tensor::CUDA(a_data), Tensor::CUDA(b_data)) => {
            require_f32(a_data)?;
            require_f32(b_data)?;

            if a_data.device_id() != b_data.device_id() {
                // Cross-device operands: bounce both through the host (consistent
                // with `to_device_enum`'s cross-device policy; the host arm below
                // may still re-dispatch the multiply host-in/host-out on the GPU).
                let a_host = a.to_device_enum(&Device::CPU)?;
                let b_host = b.to_device_enum(&Device::CPU)?;
                return a_host.matmul(&b_host);
            }

            let device_id = a_data.device_id();
            let plan = BatchedMatmulPlan::new(&a_data.shape, &b_data.shape)?;
            if plan.is_degenerate() {
                // Unreachable in practice (resident buffers cannot be zero-sized),
                // but delegate defensively instead of erroring in the backend.
                let a_host = a.to_device_enum(&Device::CPU)?;
                let b_host = b.to_device_enum(&Device::CPU)?;
                return a_host.matmul(&b_host);
            }

            let backend = oxicuda_backend(device_id)?;
            let output_id = backend.matmul_batched_gpu_to_gpu(
                &a_data.buffer_id(),
                &b_data.buffer_id(),
                plan.m,
                plan.k,
                plan.n,
                &plan.a_offsets,
                &plan.b_offsets,
            )?;

            Ok(Tensor::CUDA(CudaTensorData::new(
                output_id,
                device_id,
                plan.output_shape(),
                DType::F32,
            )))
        },
        (Tensor::CUDA(a_data), Tensor::F32(b_arr)) => {
            require_f32(a_data)?;
            resident_matmul_with_host_upload(a_data, b_arr, true)
        },
        (Tensor::F32(a_arr), Tensor::CUDA(b_data)) => {
            require_f32(b_data)?;
            resident_matmul_with_host_upload(b_data, a_arr, false)
        },
        (Tensor::CUDA(_), _) | (_, Tensor::CUDA(_)) => {
            // Resident operand mixed with a non-F32 host operand: download the
            // resident side and let the host dtype rules decide (F16/BF16 upcast,
            // unsupported-type errors, Metal routing, ...).
            let a_host = match a {
                Tensor::CUDA(_) => a.to_device_enum(&Device::CPU)?,
                _ => a.clone(),
            };
            let b_host = match b {
                Tensor::CUDA(_) => b.to_device_enum(&Device::CPU)?,
                _ => b.clone(),
            };
            a_host.matmul(&b_host)
        },
        // No resident operand: nothing for this dispatcher to do (only reachable
        // when called directly, never from the `Tensor::matmul` guard).
        _ => a.matmul(b),
    }
}

// ---------------------------------------------------------------------------
// Hardware-free plan tests (run everywhere, including macOS)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod plan_tests {
    use super::*;

    #[test]
    fn plan_matmul_2d_is_single_batch() -> crate::errors::Result<()> {
        let plan = BatchedMatmulPlan::new(&[2, 3], &[3, 4])?;
        assert_eq!((plan.m, plan.k, plan.n), (2, 3, 4));
        assert!(plan.batch_shape.is_empty());
        assert_eq!(plan.batch_count(), 1);
        assert_eq!(plan.a_offsets, vec![0]);
        assert_eq!(plan.b_offsets, vec![0]);
        assert_eq!(plan.output_shape(), vec![2, 4]);
        assert!(!plan.is_degenerate());
        Ok(())
    }

    #[test]
    fn plan_matmul_3d_equal_batch_strides() -> crate::errors::Result<()> {
        // a: [2, 3, 4], b: [2, 4, 5] -> two independent [3,4] @ [4,5] multiplies.
        let plan = BatchedMatmulPlan::new(&[2, 3, 4], &[2, 4, 5])?;
        assert_eq!((plan.m, plan.k, plan.n), (3, 4, 5));
        assert_eq!(plan.batch_shape, vec![2]);
        assert_eq!(plan.a_offsets, vec![0, 12]); // 3*4 elements per A matrix
        assert_eq!(plan.b_offsets, vec![0, 20]); // 4*5 elements per B matrix
        assert_eq!(plan.output_shape(), vec![2, 3, 5]);
        Ok(())
    }

    #[test]
    fn plan_matmul_4d_matches_cpu_batch_layout() -> crate::errors::Result<()> {
        // Multi-head attention shape: [batch=2, heads=3, m=4, k=5] @ [2, 3, 5, 6].
        let plan = BatchedMatmulPlan::new(&[2, 3, 4, 5], &[2, 3, 5, 6])?;
        assert_eq!((plan.m, plan.k, plan.n), (4, 5, 6));
        assert_eq!(plan.batch_shape, vec![2, 3]);
        assert_eq!(plan.batch_count(), 6);
        // Row-major over [2, 3]: consecutive matrices, no broadcast.
        assert_eq!(plan.a_offsets, vec![0, 20, 40, 60, 80, 100]);
        assert_eq!(plan.b_offsets, vec![0, 30, 60, 90, 120, 150]);
        assert_eq!(plan.output_shape(), vec![2, 3, 4, 6]);
        Ok(())
    }

    #[test]
    fn plan_matmul_broadcasts_2d_rhs_across_batches() -> crate::errors::Result<()> {
        // a: [3, 2, 4] @ b: [4, 5] — the 2D rhs is shared by every batch entry.
        let plan = BatchedMatmulPlan::new(&[3, 2, 4], &[4, 5])?;
        assert_eq!(plan.batch_shape, vec![3]);
        assert_eq!(plan.a_offsets, vec![0, 8, 16]);
        assert_eq!(plan.b_offsets, vec![0, 0, 0]);
        assert_eq!(plan.output_shape(), vec![3, 2, 5]);
        Ok(())
    }

    #[test]
    fn plan_matmul_broadcasts_size_one_dims_both_sides() -> crate::errors::Result<()> {
        // a: [1, 2, 2, 3] @ b: [3, 1, 3, 4] -> batch [3, 2] with each side
        // broadcast along one axis.
        let plan = BatchedMatmulPlan::new(&[1, 2, 2, 3], &[3, 1, 3, 4])?;
        assert_eq!(plan.batch_shape, vec![3, 2]);
        assert_eq!(plan.batch_count(), 6);
        // A (matrix = 6 elems) varies along the inner axis only.
        assert_eq!(plan.a_offsets, vec![0, 6, 0, 6, 0, 6]);
        // B (matrix = 12 elems) varies along the outer axis only.
        assert_eq!(plan.b_offsets, vec![0, 0, 12, 12, 24, 24]);
        assert_eq!(plan.output_shape(), vec![3, 2, 2, 4]);
        Ok(())
    }

    #[test]
    fn plan_matmul_rejects_inner_dim_mismatch() {
        let err = BatchedMatmulPlan::new(&[2, 3], &[4, 5]);
        assert!(err.is_err(), "k mismatch (3 vs 4) must be a shape error");
    }

    #[test]
    fn plan_matmul_rejects_sub_2d_operands() {
        assert!(BatchedMatmulPlan::new(&[3], &[3, 4]).is_err());
        assert!(BatchedMatmulPlan::new(&[2, 3], &[3]).is_err());
    }

    #[test]
    fn plan_matmul_rejects_incompatible_batch_dims() {
        // Batch dims 2 vs 3 with neither equal to 1 cannot broadcast.
        let err = BatchedMatmulPlan::new(&[2, 3, 4], &[3, 4, 5]);
        assert!(err.is_err(), "batch 2 vs 3 must be a shape error");
    }

    #[test]
    fn plan_matmul_zero_batch_dim_is_degenerate() -> crate::errors::Result<()> {
        let plan = BatchedMatmulPlan::new(&[0, 2, 3], &[0, 3, 4])?;
        assert_eq!(plan.batch_count(), 0);
        assert!(plan.a_offsets.is_empty());
        assert!(plan.is_degenerate());
        assert_eq!(plan.output_shape(), vec![0, 2, 4]);
        Ok(())
    }

    #[test]
    fn plan_matmul_zero_matrix_dim_is_degenerate() -> crate::errors::Result<()> {
        let plan = BatchedMatmulPlan::new(&[2, 0], &[0, 4])?;
        assert!(plan.is_degenerate());
        assert_eq!(plan.output_shape(), vec![2, 4]);
        Ok(())
    }

    #[test]
    fn batched_matmul_arg_validation_rejects_bad_inputs() {
        // Zero dimension.
        assert!(validate_batched_matmul_args(0, 2, 3, &[0], &[0], "test").is_err());
        // Offset count mismatch.
        assert!(validate_batched_matmul_args(2, 2, 3, &[0, 1], &[0], "test").is_err());
        // Empty batch.
        assert!(validate_batched_matmul_args(2, 2, 3, &[], &[], "test").is_err());
        // Well-formed.
        match validate_batched_matmul_args(2, 2, 3, &[0, 4], &[0, 6], "test") {
            Ok(count) => assert_eq!(count, 2),
            Err(e) => panic!("valid args must pass validation: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Availability-gated GPU tests (skip cleanly without a CUDA device)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod gpu_tests {
    use super::super::oxicuda_cuda_available;
    use super::*;
    use crate::tensor::Tensor;

    /// Naive row-major batched reference: `c[i] = a[a_off[i]..] @ b[b_off[i]..]`.
    fn cpu_batched_reference(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        a_offsets: &[usize],
        b_offsets: &[usize],
    ) -> Vec<f32> {
        let mut c = vec![0.0f32; a_offsets.len() * m * n];
        for (batch, (&a_off, &b_off)) in a_offsets.iter().zip(b_offsets.iter()).enumerate() {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc += a[a_off + i * k + p] * b[b_off + p * n + j];
                    }
                    c[batch * m * n + i * n + j] = acc;
                }
            }
        }
        c
    }

    fn assert_close(result: &[f32], expected: &[f32]) {
        assert_eq!(result.len(), expected.len());
        for (idx, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "mismatch at {}: got {} expected {}",
                idx,
                got,
                want
            );
        }
    }

    #[test]
    fn oxicuda_batched_matmul_f32_parity_3d() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda batched-matmul test: no CUDA device available");
            return Ok(());
        }

        // Two batches of [2,3] @ [3,2].
        let a: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 2.0).collect();
        let b: Vec<f32> = (0..12).map(|i| (i as f32 * 0.37).sin()).collect();
        let (m, k, n) = (2usize, 3usize, 2usize);
        let a_offsets = [0usize, 6];
        let b_offsets = [0usize, 6];

        let expected = cpu_batched_reference(&a, &b, m, k, n, &a_offsets, &b_offsets);

        let backend = oxicuda_backend(0)?;
        let result = backend.matmul_batched_f32(&a, &b, m, k, n, &a_offsets, &b_offsets)?;
        assert_close(&result, &expected);
        Ok(())
    }

    #[test]
    fn oxicuda_tensor_matmul_3d_host_roundtrip() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda 3D tensor-matmul test: no CUDA device available");
            return Ok(());
        }

        // Host 3D x 3D goes through the batched host-in/host-out dispatch.
        let a_data: Vec<f32> = (0..12).map(|i| i as f32 + 1.0).collect();
        let b_data: Vec<f32> = (0..12).map(|i| 12.0 - i as f32).collect();
        let a = Tensor::from_vec(a_data.clone(), &[2, 2, 3])?;
        let b = Tensor::from_vec(b_data.clone(), &[2, 3, 2])?;

        let expected = cpu_batched_reference(&a_data, &b_data, 2, 3, 2, &[0, 6], &[0, 6]);

        let result = a.matmul(&b)?;
        match result {
            Tensor::F32(arr) => {
                assert_eq!(arr.shape(), &[2, 2, 2]);
                let flat: Vec<f32> = arr.iter().copied().collect();
                assert_close(&flat, &expected);
            },
            other => panic!("expected host F32 result, got {:?}", other),
        }
        Ok(())
    }

    #[test]
    fn oxicuda_resident_matmul_stays_on_device() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda resident-matmul test: no CUDA device available");
            return Ok(());
        }

        let device = crate::device::Device::CUDA(0);
        let a_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.25).collect();
        let b_data: Vec<f32> = (0..12).map(|i| (i as f32) - 6.0).collect();
        let a_host = Tensor::from_vec(a_data.clone(), &[2, 2, 3])?;
        let b_host = Tensor::from_vec(b_data.clone(), &[2, 3, 2])?;

        let a_dev = a_host.to_device_enum(&device)?;
        let b_dev = b_host.to_device_enum(&device)?;

        let result = a_dev.matmul(&b_dev)?;
        // The result must be GPU-resident, not a host tensor.
        let downloaded = match &result {
            Tensor::CUDA(data) => {
                assert_eq!(data.shape, vec![2, 2, 2]);
                result.to_device_enum(&crate::device::Device::CPU)?
            },
            other => panic!("expected Tensor::CUDA result, got {:?}", other),
        };

        let expected = cpu_batched_reference(&a_data, &b_data, 2, 3, 2, &[0, 6], &[0, 6]);
        match downloaded {
            Tensor::F32(arr) => {
                let flat: Vec<f32> = arr.iter().copied().collect();
                assert_close(&flat, &expected);
            },
            other => panic!("expected downloaded F32 tensor, got {:?}", other),
        }
        Ok(())
    }

    #[test]
    fn oxicuda_resident_matmul_mixed_host_operand() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda mixed-matmul test: no CUDA device available");
            return Ok(());
        }

        // Resident 3D lhs x host 2D rhs: rhs is uploaded and broadcast per batch.
        let device = crate::device::Device::CUDA(0);
        let a_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let b_data: Vec<f32> = (0..6).map(|i| 1.0 - i as f32 * 0.2).collect();
        let a_dev = Tensor::from_vec(a_data.clone(), &[2, 2, 3])?.to_device_enum(&device)?;
        let b_host = Tensor::from_vec(b_data.clone(), &[3, 2])?;

        let result = a_dev.matmul(&b_host)?;
        let expected = cpu_batched_reference(&a_data, &b_data, 2, 3, 2, &[0, 6], &[0, 0]);

        match result.to_device_enum(&crate::device::Device::CPU)? {
            Tensor::F32(arr) => {
                assert_eq!(arr.shape(), &[2, 2, 2]);
                let flat: Vec<f32> = arr.iter().copied().collect();
                assert_close(&flat, &expected);
            },
            other => panic!("expected downloadable F32 result, got {:?}", other),
        }
        Ok(())
    }

    #[test]
    fn oxicuda_resident_matmul_output_freed_on_drop() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda matmul-lifecycle test: no CUDA device available");
            return Ok(());
        }

        let device = crate::device::Device::CUDA(0);
        let a_dev = Tensor::from_vec(vec![1.0f32; 6], &[2, 3])?.to_device_enum(&device)?;
        let b_dev = Tensor::from_vec(vec![1.0f32; 6], &[3, 2])?.to_device_enum(&device)?;

        let backend = oxicuda_backend(0)?;
        let before = backend.buffer_cache_size()?;

        let result = a_dev.matmul(&b_dev)?;
        assert!(matches!(result, Tensor::CUDA(_)));
        assert_eq!(
            backend.buffer_cache_size()?,
            before + 1,
            "matmul must add exactly one resident output buffer"
        );

        drop(result);
        assert_eq!(
            backend.buffer_cache_size()?,
            before,
            "dropping the last result clone must free the output buffer"
        );
        Ok(())
    }

    #[test]
    fn oxicuda_resident_matmul_shape_mismatch_errors() -> crate::errors::Result<()> {
        if !oxicuda_cuda_available() {
            eprintln!("Skipping oxicuda matmul shape-error test: no CUDA device available");
            return Ok(());
        }

        let device = crate::device::Device::CUDA(0);
        let a_dev = Tensor::from_vec(vec![1.0f32; 6], &[2, 3])?.to_device_enum(&device)?;
        let b_dev = Tensor::from_vec(vec![1.0f32; 8], &[4, 2])?.to_device_enum(&device)?;

        assert!(
            a_dev.matmul(&b_dev).is_err(),
            "inner-dimension mismatch (3 vs 4) must error"
        );
        Ok(())
    }
}
