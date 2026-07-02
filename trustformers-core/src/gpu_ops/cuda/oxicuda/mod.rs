//! Pure-Rust oxicuda CUDA backend for tensor operations.
//!
//! This module is the Campaign C1 successor to the Campaign C0 `oxicuda_spike`
//! probe: it graduates the validated oxicuda API surface into a real, feature-gated
//! CUDA backend. For this sub-slice it provides a single host-in / host-out f32
//! matrix-multiply path implemented on top of `oxicuda-blas` GEMM.
//!
//! oxicuda is a Pure-Rust CUDA stack that loads `libcuda` at runtime (rather than
//! linking a CUDA toolkit at build time). Consequently this module *compiles* on any
//! platform — including macOS, which has no NVIDIA driver — but its kernels only
//! *execute* on a host with a real NVIDIA GPU. The accompanying parity test is gated
//! to Linux/Windows for exactly that reason.
//!
//! The whole module is compiled only under `feature = "cuda"` (see the parent
//! `#[cfg(feature = "cuda")] mod oxicuda;` in `gpu_ops/cuda.rs`), so the imports
//! and items below carry no per-item `cfg`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use oxicuda_blas::level3::gemm_api::gemm;
use oxicuda_blas::{BlasHandle, Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_dnn::norm::layer_norm;
use oxicuda_dnn::types::{TensorDesc, TensorDescMut};
use oxicuda_dnn::DnnHandle;
use oxicuda_memory::DeviceBuffer;

use crate::errors::TrustformersError;

/// Identifier for a GPU-resident persistent buffer held by [`OxicudaCudaBackend`].
///
/// This is the cudarc-free analogue of the cudarc backend's `BufferId`: a process-wide
/// monotonic `u64` minted by [`OxiCudaBufferId::new`]. It is deliberately a distinct type
/// from the cudarc `BufferId` so the two resident subsystems never alias each other's ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OxiCudaBufferId(u64);

impl OxiCudaBufferId {
    /// Mint a fresh, unique buffer id.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        OxiCudaBufferId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Returns the raw `u64` value backing this id.
    #[inline]
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl Default for OxiCudaBufferId {
    fn default() -> Self {
        Self::new()
    }
}

/// Pure-Rust oxicuda CUDA backend.
///
/// Owns the CUDA [`Context`](oxicuda_driver::Context) (kept alive via an [`Arc`] so it
/// outlives the BLAS handle) and a single reusable [`BlasHandle`] bound to that context.
pub struct OxicudaCudaBackend {
    ctx: Arc<oxicuda_driver::Context>,
    handle: BlasHandle,
    /// Cache of GPU-resident persistent buffers keyed by [`OxiCudaBufferId`].
    ///
    /// The [`DeviceBuffer<f32>`] values own their device allocations and are freed when
    /// removed/cleared (or when the backend is dropped). `DeviceBuffer<f32>` is `Send + Sync`
    /// (the allocation is a `u64` device handle, not bound to a host thread), so it lives
    /// safely behind a `Mutex<HashMap<..>>` shared across threads.
    buffer_cache: Mutex<HashMap<OxiCudaBufferId, DeviceBuffer<f32>>>,
}

impl OxicudaCudaBackend {
    /// Create a new oxicuda CUDA backend bound to the given device ordinal.
    ///
    /// Initializes the CUDA driver, selects the device, creates a context and a BLAS
    /// handle on it. Fails on any host without a usable NVIDIA GPU / `libcuda`.
    pub fn new(device_id: usize) -> crate::errors::Result<Self> {
        oxicuda_driver::init().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to initialize CUDA driver: {}", e),
                "OxicudaCudaBackend::new",
            )
        })?;

        let device = oxicuda_driver::Device::get(device_id as i32).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to get CUDA device: {}", e),
                "OxicudaCudaBackend::new",
            )
        })?;

        let ctx = Arc::new(oxicuda_driver::Context::new(&device).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create CUDA context: {}", e),
                "OxicudaCudaBackend::new",
            )
        })?);

        let handle = BlasHandle::new(&ctx).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create cuBLAS handle: {}", e),
                "OxicudaCudaBackend::new",
            )
        })?;

        Ok(Self {
            ctx,
            handle,
            buffer_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Returns the CUDA context backing this backend (used by later sub-slices
    /// that build TensorDesc/DnnHandle on the same context).
    pub fn context(&self) -> &Arc<oxicuda_driver::Context> {
        &self.ctx
    }

    // ---------------------------------------------------------------------
    // GPU-resident persistent buffer subsystem (oxicuda-native).
    //
    // Mirrors the cudarc backend's resident API (`create_persistent_buffer`,
    // `get_persistent_buffer`, `remove_persistent_buffer`, `clear_buffer_cache`,
    // `buffer_cache_size`, `download_buffer`/`buffer_to_cpu`, `matmul_gpu_to_gpu`)
    // signature-for-signature, but over owned [`DeviceBuffer<f32>`] values stored
    // directly in a `Mutex<HashMap<..>>` — no cudarc, so it builds on macOS.
    // ---------------------------------------------------------------------

    /// Upload host `data` into a new GPU-resident persistent buffer and return its id.
    ///
    /// The buffer stays on the device until [`remove_persistent_buffer`](Self::remove_persistent_buffer)
    /// or [`clear_buffer_cache`](Self::clear_buffer_cache) is called. Mirrors the cudarc
    /// backend's `create_persistent_buffer`.
    pub fn create_persistent_buffer(&self, data: &[f32]) -> crate::errors::Result<OxiCudaBufferId> {
        let buffer = DeviceBuffer::<f32>::from_host(data).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy data to device: {}", e),
                "create_persistent_buffer",
            )
        })?;

        let buffer_id = OxiCudaBufferId::new();

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "create_persistent_buffer",
            )
        })?;
        cache.insert(buffer_id, buffer);
        Ok(buffer_id)
    }

    /// Allocate an uninitialised GPU-resident persistent buffer of `len` `f32`s and return its id.
    ///
    /// The oxicuda analogue of the cudarc backend's size-based allocation helpers: it reserves
    /// device memory without a host round-trip (contents are zero-filled). `len` must be
    /// non-zero (oxicuda rejects zero-length device allocations).
    pub fn create_persistent_buffer_zeroed(
        &self,
        len: usize,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let buffer = DeviceBuffer::<f32>::zeroed(len).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate device buffer of {} f32s: {}", len, e),
                "create_persistent_buffer_zeroed",
            )
        })?;

        let buffer_id = OxiCudaBufferId::new();

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "create_persistent_buffer_zeroed",
            )
        })?;
        cache.insert(buffer_id, buffer);
        Ok(buffer_id)
    }

    /// Returns the number of `f32` elements held by the persistent buffer `id`.
    ///
    /// The oxicuda cache owns its [`DeviceBuffer`]s outright, so (unlike the cudarc
    /// `Arc<CudaSlice>` clone) the buffer itself cannot be handed out without releasing the
    /// lock; this length accessor is the safe shared-reference analogue of `get_persistent_buffer`.
    pub fn get_persistent_buffer(&self, id: &OxiCudaBufferId) -> crate::errors::Result<usize> {
        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "get_persistent_buffer",
            )
        })?;

        cache.get(id).map(|buf| buf.len()).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Buffer {:?} not found in cache", id),
                "get_persistent_buffer",
            )
        })
    }

    /// Remove a persistent buffer from the cache, freeing its device allocation.
    ///
    /// Idempotent: removing an absent id is a no-op. Mirrors the cudarc backend's
    /// `remove_persistent_buffer`.
    pub fn remove_persistent_buffer(&self, id: &OxiCudaBufferId) -> crate::errors::Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "remove_persistent_buffer",
            )
        })?;

        cache.remove(id);
        Ok(())
    }

    /// Clear every persistent buffer, freeing all cached device allocations.
    ///
    /// Mirrors the cudarc backend's `clear_buffer_cache`.
    pub fn clear_buffer_cache(&self) -> crate::errors::Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "clear_buffer_cache")
        })?;

        cache.clear();
        Ok(())
    }

    /// Returns the number of buffers currently held in the persistent cache.
    ///
    /// Mirrors the cudarc backend's `buffer_cache_size`.
    pub fn buffer_cache_size(&self) -> crate::errors::Result<usize> {
        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "buffer_cache_size")
        })?;

        Ok(cache.len())
    }

    /// Copy a GPU-resident persistent buffer back to a freshly allocated host `Vec<f32>`.
    ///
    /// Mirrors the cudarc backend's `download_buffer`.
    pub fn download_buffer(&self, buffer_id: &OxiCudaBufferId) -> crate::errors::Result<Vec<f32>> {
        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "download_buffer")
        })?;

        let buffer = cache.get(buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Buffer {:?} not found in cache", buffer_id),
                "download_buffer",
            )
        })?;

        let mut result = vec![0.0f32; buffer.len()];
        buffer.copy_to_host(&mut result).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy data from device: {}", e),
                "download_buffer",
            )
        })?;
        Ok(result)
    }

    /// Copy a GPU-resident persistent buffer back to host memory.
    ///
    /// Alias of [`download_buffer`](Self::download_buffer) provided for parity with the cudarc
    /// backend's `buffer_to_cpu`; the trailing size argument is advisory only (the resident
    /// buffer already knows its own length) and is accepted solely to match that signature.
    pub fn buffer_to_cpu(
        &self,
        buffer_id: &OxiCudaBufferId,
        _size: usize,
    ) -> crate::errors::Result<Vec<f32>> {
        self.download_buffer(buffer_id)
    }

    /// Matrix-multiply two GPU-resident persistent buffers, leaving the result on the device.
    ///
    /// `C = A @ B` where A is `[m, k]`, B is `[k, n]`, C is `[m, n]` (all row-major). Both
    /// operands must already live in the cache; the freshly allocated result C is inserted and
    /// its new id returned. No host round-trip occurs. Mirrors the cudarc backend's
    /// `matmul_gpu_to_gpu`.
    ///
    /// Borrow handling: C is allocated as a *local* [`DeviceBuffer`] (not yet in the map), so A
    /// and B can be borrowed immutably from the cache at the same time as C is borrowed mutably
    /// for GEMM — there is no aliasing with the map. Once GEMM completes those borrows are
    /// dropped and C is inserted, all under a single lock acquisition and with no `unwrap`.
    pub fn matmul_gpu_to_gpu(
        &self,
        input_buffer_id: &OxiCudaBufferId,
        weight_buffer_id: &OxiCudaBufferId,
        m: usize,
        k: usize,
        n: usize,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "matmul_gpu_to_gpu")
        })?;

        // Look up both operands (immutable borrows live only inside this block).
        let a_buf = cache.get(input_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Input buffer {:?} not found in cache", input_buffer_id),
                "matmul_gpu_to_gpu",
            )
        })?;
        let b_buf = cache.get(weight_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Weight buffer {:?} not found in cache", weight_buffer_id),
                "matmul_gpu_to_gpu",
            )
        })?;

        if a_buf.len() != m * k {
            return Err(TrustformersError::shape_error(format!(
                "Input buffer length {} doesn't match m {} * k {}",
                a_buf.len(),
                m,
                k
            )));
        }
        if b_buf.len() != k * n {
            return Err(TrustformersError::shape_error(format!(
                "Weight buffer length {} doesn't match k {} * n {}",
                b_buf.len(),
                k,
                n
            )));
        }

        // C is a standalone local allocation — not in the map — so borrowing it mutably for
        // GEMM cannot conflict with the immutable A/B borrows above.
        let mut c_buf = DeviceBuffer::<f32>::alloc(m * n).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate result buffer on device: {}", e),
                "matmul_gpu_to_gpu",
            )
        })?;

        let a_desc =
            MatrixDesc::from_buffer(a_buf, m as u32, k as u32, Layout::RowMajor).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to describe input matrix: {}", e),
                    "matmul_gpu_to_gpu",
                )
            })?;
        let b_desc =
            MatrixDesc::from_buffer(b_buf, k as u32, n as u32, Layout::RowMajor).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to describe weight matrix: {}", e),
                    "matmul_gpu_to_gpu",
                )
            })?;
        let mut c_desc =
            MatrixDescMut::from_buffer(&mut c_buf, m as u32, n as u32, Layout::RowMajor).map_err(
                |e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to describe result matrix: {}", e),
                        "matmul_gpu_to_gpu",
                    )
                },
            )?;

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
                &format!("GEMM execution failed: {}", e),
                "matmul_gpu_to_gpu",
            )
        })?;

        // GEMM done: the A/B/C descriptors (and their borrows) are no longer needed; insert C.
        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, c_buf);
        Ok(output_id)
    }

    /// GELU activation on a GPU-resident buffer, leaving the result on the device.
    ///
    /// Flat element-wise op: the output buffer has the same length (`size`) as the input.
    /// The input must already live in the cache; a freshly allocated output is inserted and
    /// its new id returned. No host round-trip occurs. Mirrors the cudarc backend's
    /// `gelu_gpu_to_gpu(input_buffer_id, size) -> BufferId`.
    ///
    /// Borrow handling mirrors [`matmul_gpu_to_gpu`](Self::matmul_gpu_to_gpu): the output is a
    /// *local* [`DeviceBuffer`] (not yet in the map), so it can be borrowed mutably for the
    /// kernel at the same time the input is borrowed immutably from the cache — there is no
    /// aliasing with the map. Once the op completes those borrows drop and the output is
    /// inserted, all under a single lock acquisition and with no `unwrap`.
    pub fn gelu_gpu_to_gpu(
        &self,
        input_buffer_id: &OxiCudaBufferId,
        size: usize,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "gelu_gpu_to_gpu")
        })?;

        let in_buf = cache.get(input_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Input buffer {:?} not found in cache", input_buffer_id),
                "gelu_gpu_to_gpu",
            )
        })?;

        if in_buf.len() != size {
            return Err(TrustformersError::shape_error(format!(
                "Input buffer length {} doesn't match size {}",
                in_buf.len(),
                size
            )));
        }

        // Output is a standalone local allocation — not in the map — so borrowing it mutably
        // cannot conflict with the immutable input borrow above.
        let mut out_buf = DeviceBuffer::<f32>::alloc(size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                "gelu_gpu_to_gpu",
            )
        })?;

        oxicuda_blas::elementwise::gelu(&self.handle, size as u32, in_buf, &mut out_buf).map_err(
            |e| {
                TrustformersError::hardware_error(
                    &format!("GELU execution failed: {}", e),
                    "gelu_gpu_to_gpu",
                )
            },
        )?;

        // Op done: the input borrow is no longer needed; insert the result.
        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, out_buf);
        Ok(output_id)
    }

    /// Broadcast bias-add over a GPU-resident `[m, n]` matrix, leaving the result on the device.
    ///
    /// Computes `output[i, j] = input[i, j] + bias[j]` for a row-major `[m, n]` input and a
    /// length-`n` bias (the standard post-GEMM linear-layer bias broadcast). Input and bias must
    /// already live in the cache; a freshly allocated `[m, n]` output is inserted and its new id
    /// returned. No host round-trip occurs. Mirrors the cudarc backend's
    /// `add_bias_gpu_to_gpu(input, bias, m, n) -> BufferId` signature and semantics, but runs the
    /// broadcast through `oxicuda_blas::elementwise::bias_add` rather than a hand-written CUDA
    /// kernel.
    ///
    /// Borrow handling mirrors [`matmul_gpu_to_gpu`](Self::matmul_gpu_to_gpu): the output is a
    /// *local* [`DeviceBuffer`] (not yet in the map), so it can be borrowed mutably for the kernel
    /// at the same time the input and bias are borrowed immutably from the cache — there is no
    /// aliasing with the map. Once the op completes those borrows drop and the output is inserted,
    /// all under a single lock acquisition and with no `unwrap`.
    pub fn add_bias_gpu_to_gpu(
        &self,
        input_buffer_id: &OxiCudaBufferId,
        bias_buffer_id: &OxiCudaBufferId,
        m: usize,
        n: usize,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let total_size = m * n;

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "add_bias_gpu_to_gpu")
        })?;

        let in_buf = cache.get(input_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Input buffer {:?} not found in cache", input_buffer_id),
                "add_bias_gpu_to_gpu",
            )
        })?;
        let bias_buf = cache.get(bias_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Bias buffer {:?} not found in cache", bias_buffer_id),
                "add_bias_gpu_to_gpu",
            )
        })?;

        if in_buf.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input buffer length {} doesn't match m {} * n {}",
                in_buf.len(),
                m,
                n
            )));
        }
        if bias_buf.len() != n {
            return Err(TrustformersError::shape_error(format!(
                "Bias buffer length {} doesn't match n {}",
                bias_buf.len(),
                n
            )));
        }

        // Output is a standalone local allocation — not in the map — so borrowing it mutably
        // cannot conflict with the immutable input/bias borrows above.
        let mut out_buf = DeviceBuffer::<f32>::alloc(total_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                "add_bias_gpu_to_gpu",
            )
        })?;

        oxicuda_blas::elementwise::bias_add(
            &self.handle,
            m as u32,
            n as u32,
            in_buf,
            bias_buf,
            &mut out_buf,
        )
        .map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Bias-add execution failed: {}", e),
                "add_bias_gpu_to_gpu",
            )
        })?;

        // Op done: the operand borrows are no longer needed; insert the result.
        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, out_buf);
        Ok(output_id)
    }

    /// Layer normalization over a GPU-resident `[seq_len, hidden_size]` tensor, result resident.
    ///
    /// Normalizes each row using population variance:
    /// `(x - mean) * rsqrt(var + eps) * weight + bias`. Input, weight, and bias must already
    /// live in the cache; a freshly allocated output is inserted and its new id returned. No
    /// host round-trip occurs. Mirrors the cudarc backend's
    /// `layernorm_gpu_to_gpu(input, weight, bias, seq_len, hidden_size, eps) -> BufferId`, and
    /// reuses the same oxicuda-dnn `layer_norm` path as the host [`layernorm_f32`](Self::layernorm_f32).
    ///
    /// Borrow handling: the three operands are borrowed immutably from the cache while the
    /// local (not-yet-inserted) output is borrowed mutably for the kernel; no aliasing with the
    /// map occurs. All borrows drop before the output is inserted, under one lock and no `unwrap`.
    pub fn layernorm_gpu_to_gpu(
        &self,
        input_buffer_id: &OxiCudaBufferId,
        weight_buffer_id: &OxiCudaBufferId,
        bias_buffer_id: &OxiCudaBufferId,
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) -> crate::errors::Result<OxiCudaBufferId> {
        let total_size = seq_len * hidden_size;

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "layernorm_gpu_to_gpu")
        })?;

        let in_buf = cache.get(input_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Input buffer {:?} not found in cache", input_buffer_id),
                "layernorm_gpu_to_gpu",
            )
        })?;
        let weight_buf = cache.get(weight_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Weight buffer {:?} not found in cache", weight_buffer_id),
                "layernorm_gpu_to_gpu",
            )
        })?;
        let bias_buf = cache.get(bias_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Bias buffer {:?} not found in cache", bias_buffer_id),
                "layernorm_gpu_to_gpu",
            )
        })?;

        if in_buf.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input buffer length {} doesn't match seq_len {} * hidden_size {}",
                in_buf.len(),
                seq_len,
                hidden_size
            )));
        }
        if weight_buf.len() != hidden_size || bias_buf.len() != hidden_size {
            return Err(TrustformersError::shape_error(format!(
                "Weight/bias buffer lengths ({}, {}) must match hidden_size {}",
                weight_buf.len(),
                bias_buf.len(),
                hidden_size
            )));
        }

        let dnn = DnnHandle::new(self.context()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create cuDNN handle: {}", e),
                "layernorm_gpu_to_gpu",
            )
        })?;

        // Output is a standalone local allocation — not in the map.
        let mut out_buf = DeviceBuffer::<f32>::alloc(total_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                "layernorm_gpu_to_gpu",
            )
        })?;

        let in_desc = TensorDesc::<f32>::matrix(in_buf, seq_len as u32, hidden_size as u32)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to describe input tensor: {}", e),
                    "layernorm_gpu_to_gpu",
                )
            })?;

        {
            let mut out_desc =
                TensorDescMut::<f32>::matrix(&mut out_buf, seq_len as u32, hidden_size as u32)
                    .map_err(|e| {
                        TrustformersError::hardware_error(
                            &format!("Failed to describe output tensor: {}", e),
                            "layernorm_gpu_to_gpu",
                        )
                    })?;

            layer_norm(&dnn, &in_desc, weight_buf, bias_buf, &mut out_desc, eps).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("LayerNorm execution failed: {}", e),
                    "layernorm_gpu_to_gpu",
                )
            })?;
        }

        // Op done: the operand borrows are no longer needed; insert the result.
        let output_id = OxiCudaBufferId::new();
        cache.insert(output_id, out_buf);
        Ok(output_id)
    }

    /// Matrix-multiply host activations against a *cached* GPU-resident weight buffer.
    ///
    /// `C = A @ B` where A is the host `[m, k]` activation matrix (uploaded fresh each call),
    /// B is the `[k, n]` weight already resident in the cache under `weight_buffer_id`, and the
    /// `[m, n]` result C is copied back to a host `Vec<f32>`. This is the hot-path forward shape:
    /// activations change every step while the weight stays parked on the device. Mirrors the
    /// cudarc backend's `matmul_with_cached_weight(a, weight_buffer_id, m, k, n) -> Vec<f32>`
    /// (host-in activations, host-out result), but runs the multiply through oxicuda-blas GEMM
    /// rather than a hand-written CUDA kernel.
    pub fn matmul_with_cached_weight(
        &self,
        a: &[f32],
        weight_buffer_id: &OxiCudaBufferId,
        m: usize,
        k: usize,
        n: usize,
    ) -> crate::errors::Result<Vec<f32>> {
        if a.len() != m * k {
            return Err(TrustformersError::shape_error(format!(
                "Activation length {} doesn't match m {} * k {}",
                a.len(),
                m,
                k
            )));
        }

        // Upload the activations to a fresh temp device buffer (they change every call).
        let a_buf = DeviceBuffer::<f32>::from_host(a).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to upload activations to device: {}", e),
                "matmul_with_cached_weight",
            )
        })?;

        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "matmul_with_cached_weight",
            )
        })?;

        let b_buf = cache.get(weight_buffer_id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Weight buffer {:?} not found in cache", weight_buffer_id),
                "matmul_with_cached_weight",
            )
        })?;

        if b_buf.len() != k * n {
            return Err(TrustformersError::shape_error(format!(
                "Weight buffer length {} doesn't match k {} * n {}",
                b_buf.len(),
                k,
                n
            )));
        }

        // C is a standalone local allocation, not in the map.
        let mut c_buf = DeviceBuffer::<f32>::alloc(m * n).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate result buffer on device: {}", e),
                "matmul_with_cached_weight",
            )
        })?;

        let a_desc = MatrixDesc::from_buffer(&a_buf, m as u32, k as u32, Layout::RowMajor)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to describe activation matrix: {}", e),
                    "matmul_with_cached_weight",
                )
            })?;
        let b_desc =
            MatrixDesc::from_buffer(b_buf, k as u32, n as u32, Layout::RowMajor).map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to describe weight matrix: {}", e),
                    "matmul_with_cached_weight",
                )
            })?;
        let mut c_desc =
            MatrixDescMut::from_buffer(&mut c_buf, m as u32, n as u32, Layout::RowMajor).map_err(
                |e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to describe result matrix: {}", e),
                        "matmul_with_cached_weight",
                    )
                },
            )?;

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
                &format!("GEMM execution failed: {}", e),
                "matmul_with_cached_weight",
            )
        })?;

        let mut result = vec![0.0f32; m * n];
        c_buf.copy_to_host(&mut result).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result back to host: {}", e),
                "matmul_with_cached_weight",
            )
        })?;

        Ok(result)
    }

    /// Human-readable description of the CUDA device this backend is bound to.
    ///
    /// Mirrors the cudarc backend's `device_info() -> String`. The ordinal is read from the
    /// oxicuda [`Context`](oxicuda_driver::Context)'s [`Device`](oxicuda_driver::Device) rather
    /// than a separately stored field, so it always reflects the device the context actually
    /// selected.
    pub fn device_info(&self) -> String {
        format!(
            "CUDA Device (ordinal: {})",
            self.context().device().ordinal()
        )
    }

    /// Perform matrix multiplication on CUDA GPU via oxicuda-blas GEMM.
    /// C = A @ B where A is [m, k], B is [k, n], C is [m, n] (row-major).
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> crate::errors::Result<Vec<f32>> {
        if a.len() != m * k {
            return Err(TrustformersError::shape_error(format!(
                "Matrix A length {} doesn't match m {} * k {}",
                a.len(),
                m,
                k
            )));
        }
        if b.len() != k * n {
            return Err(TrustformersError::shape_error(format!(
                "Matrix B length {} doesn't match k {} * n {}",
                b.len(),
                k,
                n
            )));
        }

        let a_buf = DeviceBuffer::<f32>::from_host(a).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to upload matrix A to device: {}", e),
                "matmul_f32",
            )
        })?;
        let b_buf = DeviceBuffer::<f32>::from_host(b).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to upload matrix B to device: {}", e),
                "matmul_f32",
            )
        })?;
        let mut c_buf = DeviceBuffer::<f32>::alloc(m * n).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate result buffer on device: {}", e),
                "matmul_f32",
            )
        })?;

        let a_desc = MatrixDesc::from_buffer(&a_buf, m as u32, k as u32, Layout::RowMajor)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to describe matrix A: {}", e),
                    "matmul_f32",
                )
            })?;
        let b_desc = MatrixDesc::from_buffer(&b_buf, k as u32, n as u32, Layout::RowMajor)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to describe matrix B: {}", e),
                    "matmul_f32",
                )
            })?;
        let mut c_desc =
            MatrixDescMut::from_buffer(&mut c_buf, m as u32, n as u32, Layout::RowMajor).map_err(
                |e| {
                    TrustformersError::hardware_error(
                        &format!("Failed to describe result matrix: {}", e),
                        "matmul_f32",
                    )
                },
            )?;

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
                &format!("GEMM execution failed: {}", e),
                "matmul_f32",
            )
        })?;

        let mut result = vec![0.0f32; m * n];
        c_buf.copy_to_host(&mut result).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result back to host: {}", e),
                "matmul_f32",
            )
        })?;

        Ok(result)
    }

    /// Execute GELU activation on the GPU via oxicuda-blas elementwise GELU.
    ///
    /// Flat element-wise op: the output has the same length as `input`. Mirrors the
    /// cudarc backend's `gelu_f32` host-in / host-out signature and semantics.
    pub fn gelu_f32(&self, input: &[f32]) -> crate::errors::Result<Vec<f32>> {
        let size = input.len();
        if size == 0 {
            return Ok(Vec::new());
        }

        let in_buf = DeviceBuffer::<f32>::from_host(input).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to upload input to device: {}", e),
                "gelu_f32",
            )
        })?;
        let mut out_buf = DeviceBuffer::<f32>::alloc(size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                "gelu_f32",
            )
        })?;

        oxicuda_blas::elementwise::gelu(&self.handle, size as u32, &in_buf, &mut out_buf).map_err(
            |e| {
                TrustformersError::hardware_error(
                    &format!("GELU execution failed: {}", e),
                    "gelu_f32",
                )
            },
        )?;

        let mut result = vec![0.0f32; size];
        out_buf.copy_to_host(&mut result).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result back to host: {}", e),
                "gelu_f32",
            )
        })?;

        Ok(result)
    }

    /// Execute layer normalization on the GPU via oxicuda-dnn `layer_norm`.
    ///
    /// Normalizes each row of a `[seq_len, hidden_size]` row-major tensor using
    /// population variance: `(x - mean) * rsqrt(var + eps) * weight + bias`. Mirrors
    /// the cudarc backend's `layernorm_f32` host-in / host-out signature and semantics.
    pub fn layernorm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) -> crate::errors::Result<Vec<f32>> {
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

        if total_size == 0 {
            return Ok(Vec::new());
        }

        let dnn = DnnHandle::new(self.context()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create cuDNN handle: {}", e),
                "layernorm_f32",
            )
        })?;

        let in_buf = DeviceBuffer::<f32>::from_host(input).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to upload input to device: {}", e),
                "layernorm_f32",
            )
        })?;
        let weight_buf = DeviceBuffer::<f32>::from_host(weight).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to upload weight to device: {}", e),
                "layernorm_f32",
            )
        })?;
        let bias_buf = DeviceBuffer::<f32>::from_host(bias).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to upload bias to device: {}", e),
                "layernorm_f32",
            )
        })?;
        let mut out_buf = DeviceBuffer::<f32>::alloc(total_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                "layernorm_f32",
            )
        })?;

        let in_desc = TensorDesc::<f32>::matrix(&in_buf, seq_len as u32, hidden_size as u32)
            .map_err(|e| {
                TrustformersError::hardware_error(
                    &format!("Failed to describe input tensor: {}", e),
                    "layernorm_f32",
                )
            })?;

        {
            let mut out_desc =
                TensorDescMut::<f32>::matrix(&mut out_buf, seq_len as u32, hidden_size as u32)
                    .map_err(|e| {
                        TrustformersError::hardware_error(
                            &format!("Failed to describe output tensor: {}", e),
                            "layernorm_f32",
                        )
                    })?;

            layer_norm(&dnn, &in_desc, &weight_buf, &bias_buf, &mut out_desc, eps).map_err(
                |e| {
                    TrustformersError::hardware_error(
                        &format!("LayerNorm execution failed: {}", e),
                        "layernorm_f32",
                    )
                },
            )?;
        }

        let mut result = vec![0.0f32; total_size];
        out_buf.copy_to_host(&mut result).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result back to host: {}", e),
                "layernorm_f32",
            )
        })?;

        Ok(result)
    }

    /// Causal (lower-triangular masked) softmax over a [seq_len, seq_len] score matrix.
    ///
    /// Row-major: rows = query positions, cols = key positions; position i attends to j <= i.
    /// No score scaling is applied (scale the input beforehand if required) — matching the
    /// cudarc CUDA backend and the oxicuda `causal_softmax` kernel.
    pub fn softmax_causal_f32(
        &self,
        input: &[f32],
        seq_len: usize,
    ) -> crate::errors::Result<Vec<f32>> {
        let total_size = seq_len * seq_len;

        if input.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input size {} doesn't match seq_len^2 {}",
                input.len(),
                total_size
            )));
        }

        if total_size == 0 {
            return Ok(Vec::new());
        }

        let in_buf = DeviceBuffer::<f32>::from_host(input).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to upload input to device: {}", e),
                "softmax_causal_f32",
            )
        })?;
        let mut out_buf = DeviceBuffer::<f32>::alloc(total_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                "softmax_causal_f32",
            )
        })?;

        oxicuda_blas::reduction::causal_softmax::<f32>(
            &self.handle,
            seq_len as u32,
            seq_len as u32,
            &in_buf,
            &mut out_buf,
        )
        .map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Causal softmax execution failed: {}", e),
                "softmax_causal_f32",
            )
        })?;

        let mut result = vec![0.0f32; total_size];
        out_buf.copy_to_host(&mut result).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result back to host: {}", e),
                "softmax_causal_f32",
            )
        })?;

        Ok(result)
    }

    /// Applies GPT-NeoX half-split partial rotary position embedding (RoPE).
    ///
    /// Input/output layout: flat row-major `[seq_len, num_heads, head_dim]`. The first
    /// `rotary_ndims` channels of each head are rotated (pairing lane `i` with `i + rotary_ndims/2`),
    /// remaining channels are copied through. Matches the cudarc CUDA backend exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_f32(
        &self,
        input: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_ndims: usize,
        base: f32,
    ) -> crate::errors::Result<Vec<f32>> {
        let total_size = seq_len * num_heads * head_dim;

        if input.len() != total_size {
            return Err(TrustformersError::shape_error(format!(
                "Input size {} doesn't match seq_len {} * num_heads {} * head_dim {}",
                input.len(),
                seq_len,
                num_heads,
                head_dim
            )));
        }

        if total_size == 0 {
            return Ok(Vec::new());
        }

        let dnn = DnnHandle::new(self.context()).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create cuDNN handle: {}", e),
                "rope_f32",
            )
        })?;

        let in_buf = DeviceBuffer::<f32>::from_host(input).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to upload input to device: {}", e),
                "rope_f32",
            )
        })?;
        let mut out_buf = DeviceBuffer::<f32>::alloc(total_size).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to allocate output buffer on device: {}", e),
                "rope_f32",
            )
        })?;

        oxicuda_dnn::attn::rope_neox_half_split_f32(
            &dnn,
            &in_buf,
            &mut out_buf,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
            rotary_ndims as u32,
            base,
        )
        .map_err(|e| {
            TrustformersError::hardware_error(&format!("RoPE execution failed: {}", e), "rope_f32")
        })?;

        let mut result = vec![0.0f32; total_size];
        out_buf.copy_to_host(&mut result).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy result back to host: {}", e),
                "rope_f32",
            )
        })?;

        Ok(result)
    }
}

/// Returns `true` if a real, currently-usable NVIDIA CUDA GPU is present on this host.
///
/// This is a genuine **runtime** hardware probe: it dynamically loads `libcuda`
/// (via [`oxicuda_driver::init`], which itself calls the driver loader and then
/// `cuInit`) and confirms at least one device is enumerable via
/// [`oxicuda_driver::Device::count`]. It does *not* construct a context, a BLAS
/// handle, or any other heavyweight [`OxicudaCudaBackend`] state — just the
/// minimum needed to know whether the later, real construction in
/// [`oxicuda_backend()`] would succeed.
///
/// Contrast this with `scirs2_core::simd_ops::PlatformCapabilities::detect().cuda_available`,
/// which is a **compile-time** flag (`cfg!(all(feature = "gpu", feature = "cuda"))`
/// evaluated *inside scirs2-core's own build*, gated on scirs2-core's own same-named
/// `cuda` Cargo feature) — it never probes hardware and would stay `true` on a build
/// forever, GPU or not, if scirs2-core's `cuda` feature were ever wired up. Callers
/// that need to know "is there actually a GPU right now" (e.g. an unconditional
/// fast-path taken purely because the `cuda` feature was compiled in, with no
/// `Device::CUDA`/`Tensor::CUDA` tag to fall back on) must use this function instead,
/// never `PlatformCapabilities`.
///
/// `libcuda` loading and `cuInit` are relatively expensive (dynamic linking, driver
/// handshake) and this is called from hot paths (e.g. every [`Tensor::matmul`]), so
/// the result is probed once per process and cached in a [`std::sync::OnceLock`].
/// This mirrors `oxicuda_driver`'s own `try_driver()`, which caches the loaded
/// `DriverApi` the same way — repeated calls after the first are a single atomic load.
///
/// [`Tensor::matmul`]: crate::tensor::Tensor::matmul
pub fn oxicuda_cuda_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        oxicuda_driver::init().is_ok()
            && matches!(oxicuda_driver::Device::count(), Ok(count) if count > 0)
    })
}

/// Per-device cache of [`OxicudaCudaBackend`] instances (one [`Arc`] per device ordinal).
///
/// This is the oxicuda analogue of the cudarc backend's `CUDA_BACKENDS`
/// (`cuda_split/cuda_backend.rs`): a process-wide `Lazy<Mutex<HashMap<usize, Arc<..>>>>`
/// built with the same `once_cell::sync::Lazy` primitive and the same lock discipline.
/// Keeping the backend resident across calls is what lets GPU-resident
/// [`OxiCudaBufferId`]s minted by one call survive into later calls on the same device.
///
/// The two resident subsystems are intentionally distinct statics keyed by the same
/// `usize` device ordinal but never aliasing each other's state (`CUDA_BACKENDS` holds
/// cudarc `CudaBackend`s; this holds Pure-Rust `OxicudaCudaBackend`s). Unlike the cudarc
/// static this one is *not* OS-gated: oxicuda compiles on macOS, so the singleton is
/// available wherever the `cuda-oxicuda` feature is enabled.
static OXICUDA_BACKENDS: once_cell::sync::Lazy<Mutex<HashMap<usize, Arc<OxicudaCudaBackend>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

/// Get-or-create the shared [`OxicudaCudaBackend`] for `device_id`.
///
/// Constructs the backend exactly once per device ordinal and reuses the same [`Arc`]
/// thereafter, so resident [`OxiCudaBufferId`]s persist across calls. Mirrors the cudarc
/// backend's `get_cuda_backend`: lock the cache, insert on a vacant entry, hand back a
/// clone of the `Arc`. Lock-poisoning and the (theoretically impossible) post-insert miss
/// are surfaced as [`hardware_error`](TrustformersError::hardware_error) — never `unwrap`.
///
/// `OxicudaCudaBackend` is usable behind `Arc` because all of its mutating operations take
/// `&self` and synchronise internally via the `Mutex<HashMap<..>>` buffer cache.
pub fn oxicuda_backend(device_id: usize) -> crate::errors::Result<Arc<OxicudaCudaBackend>> {
    let mut cache = OXICUDA_BACKENDS.lock().map_err(|_| {
        TrustformersError::hardware_error("Failed to lock oxicuda backend cache", "oxicuda_backend")
    })?;

    if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(device_id) {
        let backend = OxicudaCudaBackend::new(device_id)?;
        e.insert(Arc::new(backend));
    }

    cache.get(&device_id).cloned().ok_or_else(|| {
        TrustformersError::hardware_error("oxicuda backend not found", "oxicuda_backend")
    })
}

/// Dispatch matrix multiplication to the oxicuda CUDA backend.
///
/// For `F32` tensors this uploads both operands to the GPU, runs GEMM via
/// [`OxicudaCudaBackend::matmul_f32`], and returns a host-resident `Tensor::F32`.
/// All other dtypes fall back to the CPU `Tensor::matmul` path.
///
/// The backend is obtained through the per-device [`oxicuda_backend()`] singleton rather than
/// constructed fresh per call, so the CUDA context / cuBLAS handle and any GPU-resident
/// buffers are shared across invocations on the same device ordinal.
pub fn dispatch_oxicuda_matmul(
    a: &crate::tensor::Tensor,
    b: &crate::tensor::Tensor,
    device_id: usize,
) -> crate::errors::Result<crate::tensor::Tensor> {
    match (a, b) {
        (crate::tensor::Tensor::F32(a_arr), crate::tensor::Tensor::F32(b_arr)) => {
            if a_arr.ndim() != 2 || b_arr.ndim() != 2 {
                return Err(TrustformersError::shape_error(
                    "oxicuda CUDA dispatch currently only supports 2D tensors".to_string(),
                ));
            }

            let a_2d =
                a_arr.clone().into_dimensionality::<scirs2_core::ndarray::Ix2>().map_err(|e| {
                    TrustformersError::shape_error(format!("Failed to convert to 2D: {}", e))
                })?;
            let b_2d =
                b_arr.clone().into_dimensionality::<scirs2_core::ndarray::Ix2>().map_err(|e| {
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

            let a_data: Vec<f32> = a_2d.iter().copied().collect();
            let b_data: Vec<f32> = b_2d.iter().copied().collect();

            // Reuse the per-device backend singleton so the CUDA context / handle and any
            // resident buffers persist across calls (instead of `OxicudaCudaBackend::new`).
            let backend = oxicuda_backend(device_id)?;
            let result_data = backend.matmul_f32(&a_data, &b_data, m, k, n)?;

            let result_2d = scirs2_core::ndarray::Array2::from_shape_vec((m, n), result_data)
                .map_err(|e| {
                    TrustformersError::shape_error(format!("Failed to reshape result: {}", e))
                })?;

            Ok(crate::tensor::Tensor::F32(result_2d.into_dyn()))
        },
        _ => a.matmul(b),
    }
}

#[cfg(all(
    test,
    feature = "cuda",
    any(target_os = "linux", target_os = "windows")
))]
mod tests {
    use super::*;

    #[test]
    fn oxicuda_cuda_matmul_parity() -> crate::errors::Result<()> {
        // A is [m=2, k=3], B is [k=3, n=2], row-major.
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let m = 2usize;
        let k = 3usize;
        let n = 2usize;

        // Naive CPU reference, row-major.
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    expected[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }

        let backend = match OxicudaCudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping oxicuda CUDA test: no CUDA device available");
                return Ok(());
            },
        };

        let result = backend.matmul_f32(&a, &b, m, k, n)?;

        for idx in 0..(m * n) {
            assert!(
                (result[idx] - expected[idx]).abs() < 1e-3,
                "mismatch at {}: got {} expected {}",
                idx,
                result[idx],
                expected[idx]
            );
        }

        Ok(())
    }

    #[test]
    fn oxicuda_cuda_gelu_parity() -> crate::errors::Result<()> {
        // Moderate values only (oxicuda GELU lacks the cudarc ±10 clamps).
        let input = vec![-1.0f32, -0.5, 0.0, 0.5, 1.0, 2.0];

        // CPU reference: tanh-approximation GELU.
        let mut expected = vec![0.0f32; input.len()];
        for (idx, &x) in input.iter().enumerate() {
            let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
            expected[idx] = 0.5f32 * x * (1.0f32 + inner.tanh());
        }

        let backend = match OxicudaCudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping oxicuda CUDA GELU test: no CUDA device available");
                return Ok(());
            },
        };

        let result = backend.gelu_f32(&input)?;

        for idx in 0..input.len() {
            assert!(
                (result[idx] - expected[idx]).abs() < 1e-3,
                "mismatch at {}: got {} expected {}",
                idx,
                result[idx],
                expected[idx]
            );
        }

        Ok(())
    }

    #[test]
    fn oxicuda_cuda_layernorm_parity() -> crate::errors::Result<()> {
        let seq_len = 2usize;
        let hidden_size = 4usize;
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let weight = vec![1.0f32; hidden_size];
        let bias = vec![0.0f32; hidden_size];
        let eps = 1e-5f32;

        // CPU reference: per-row population-variance layer norm.
        let mut expected = vec![0.0f32; seq_len * hidden_size];
        for row in 0..seq_len {
            let offset = row * hidden_size;
            let mut sum = 0.0f32;
            for i in 0..hidden_size {
                sum += input[offset + i];
            }
            let mean = sum / hidden_size as f32;
            let mut var_sum = 0.0f32;
            for i in 0..hidden_size {
                let diff = input[offset + i] - mean;
                var_sum += diff * diff;
            }
            let variance = var_sum / hidden_size as f32;
            let std_dev = (variance + eps).sqrt();
            for i in 0..hidden_size {
                let normalized = (input[offset + i] - mean) / std_dev;
                expected[offset + i] = normalized * weight[i] + bias[i];
            }
        }

        let backend = match OxicudaCudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping oxicuda CUDA LayerNorm test: no CUDA device available");
                return Ok(());
            },
        };

        let result = backend.layernorm_f32(&input, &weight, &bias, seq_len, hidden_size, eps)?;

        for idx in 0..(seq_len * hidden_size) {
            assert!(
                (result[idx] - expected[idx]).abs() < 1e-3,
                "mismatch at {}: got {} expected {}",
                idx,
                result[idx],
                expected[idx]
            );
        }

        Ok(())
    }

    #[test]
    fn oxicuda_cuda_softmax_causal_parity() -> crate::errors::Result<()> {
        let seq_len = 4usize;
        let total = seq_len * seq_len;

        // Deterministic varied input.
        let mut input = vec![0.0f32; total];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = (i as f32 * 0.37 - 2.0).sin();
        }

        // CPU reference: causal (lower-triangular) softmax, no scaling.
        let mut expected = vec![0.0f32; total];
        for r in 0..seq_len {
            let offset = r * seq_len;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..=r {
                let v = input[offset + j];
                if v > max_val {
                    max_val = v;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..=r {
                sum += (input[offset + j] - max_val).exp();
            }
            for j in 0..seq_len {
                if j <= r {
                    expected[offset + j] = (input[offset + j] - max_val).exp() / sum;
                } else {
                    expected[offset + j] = 0.0f32;
                }
            }
        }

        let backend = match OxicudaCudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping oxicuda CUDA causal-softmax test: no CUDA device available");
                return Ok(());
            },
        };

        let result = backend.softmax_causal_f32(&input, seq_len)?;

        for idx in 0..total {
            assert!(
                (result[idx] - expected[idx]).abs() < 1e-3,
                "mismatch at {}: got {} expected {}",
                idx,
                result[idx],
                expected[idx]
            );
        }

        Ok(())
    }

    #[test]
    fn oxicuda_cuda_rope_parity() -> crate::errors::Result<()> {
        // Partial-rotary half-split case: head_dim=6, rotary_ndims=4 (half=2),
        // so dims 0<->2 and 1<->3 rotate, dims 4,5 pass through.
        let seq_len = 2usize;
        let num_heads = 1usize;
        let head_dim = 6usize;
        let rotary_ndims = 4usize;
        let base = 10000.0f32;
        let total = seq_len * num_heads * head_dim;

        // Deterministic varied input.
        let mut input = vec![0.0f32; total];
        for (idx, slot) in input.iter_mut().enumerate() {
            *slot = (idx as f32) * 0.5 + 0.1;
        }

        // CPU reference: GPT-NeoX half-split partial RoPE.
        let mut expected = vec![0.0f32; total];
        let half = rotary_ndims / 2;
        for pos in 0..seq_len {
            for h in 0..num_heads {
                let base_off = (pos * num_heads + h) * head_dim;
                for i in 0..half {
                    let freq = base.powf(-2.0 * (i as f32) / (rotary_ndims as f32));
                    let angle = (pos as f32) * freq;
                    let c = angle.cos();
                    let s = angle.sin();
                    let x_i = input[base_off + i];
                    let x_j = input[base_off + i + half];
                    expected[base_off + i] = x_i * c - x_j * s;
                    expected[base_off + i + half] = x_i * s + x_j * c;
                }
                expected[(base_off + rotary_ndims)..(base_off + head_dim)]
                    .copy_from_slice(&input[(base_off + rotary_ndims)..(base_off + head_dim)]);
            }
        }

        let backend = match OxicudaCudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping oxicuda CUDA RoPE test: no CUDA device available");
                return Ok(());
            },
        };

        let result = backend.rope_f32(&input, seq_len, num_heads, head_dim, rotary_ndims, base)?;

        for idx in 0..total {
            assert!(
                (result[idx] - expected[idx]).abs() < 1e-3,
                "mismatch at {}: got {} expected {}",
                idx,
                result[idx],
                expected[idx]
            );
        }

        Ok(())
    }

    #[test]
    fn oxicuda_cuda_resident_matmul_parity() -> crate::errors::Result<()> {
        // A is [m=2, k=3], B is [k=3, n=2], row-major.
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let m = 2usize;
        let k = 3usize;
        let n = 2usize;

        // Naive CPU reference (triple loop), row-major.
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    expected[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }

        let backend = match OxicudaCudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping oxicuda CUDA resident-matmul test: no CUDA device available");
                return Ok(());
            },
        };

        // Empty cache to start.
        assert_eq!(backend.buffer_cache_size()?, 0);

        // Upload both operands as resident persistent buffers.
        let a_id = backend.create_persistent_buffer(&a)?;
        let b_id = backend.create_persistent_buffer(&b)?;
        assert_eq!(backend.buffer_cache_size()?, 2);
        assert_eq!(backend.get_persistent_buffer(&a_id)?, m * k);
        assert_eq!(backend.get_persistent_buffer(&b_id)?, k * n);

        // GPU-to-GPU matmul: result stays resident.
        let c_id = backend.matmul_gpu_to_gpu(&a_id, &b_id, m, k, n)?;
        assert_eq!(backend.buffer_cache_size()?, 3);

        // Download and compare to the CPU reference.
        let result = backend.download_buffer(&c_id)?;
        assert_eq!(result.len(), m * n);
        for idx in 0..(m * n) {
            assert!(
                (result[idx] - expected[idx]).abs() < 1e-3,
                "mismatch at {}: got {} expected {}",
                idx,
                result[idx],
                expected[idx]
            );
        }

        // `buffer_to_cpu` is an alias of `download_buffer`.
        let result_alias = backend.buffer_to_cpu(&c_id, m * n)?;
        assert_eq!(result_alias, result);

        // Exercise remove + clear + size bookkeeping.
        backend.remove_persistent_buffer(&a_id)?;
        assert_eq!(backend.buffer_cache_size()?, 2);
        // Removing an absent id is a no-op.
        backend.remove_persistent_buffer(&a_id)?;
        assert_eq!(backend.buffer_cache_size()?, 2);

        backend.clear_buffer_cache()?;
        assert_eq!(backend.buffer_cache_size()?, 0);

        // A zeroed resident allocation reads back as all zeros.
        let z_id = backend.create_persistent_buffer_zeroed(4)?;
        let zeros = backend.download_buffer(&z_id)?;
        assert_eq!(zeros, vec![0.0f32; 4]);

        Ok(())
    }

    #[test]
    fn oxicuda_cuda_resident_gelu_parity() -> crate::errors::Result<()> {
        // Moderate values only (oxicuda GELU lacks the cudarc ±10 clamps).
        let input = vec![-1.0f32, -0.5, 0.0, 0.5, 1.0, 2.0];
        let size = input.len();

        // CPU reference: tanh-approximation GELU.
        let mut expected = vec![0.0f32; size];
        for (idx, &x) in input.iter().enumerate() {
            let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
            expected[idx] = 0.5f32 * x * (1.0f32 + inner.tanh());
        }

        let backend = match OxicudaCudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping oxicuda CUDA resident-GELU test: no CUDA device available");
                return Ok(());
            },
        };

        let in_id = backend.create_persistent_buffer(&input)?;
        let out_id = backend.gelu_gpu_to_gpu(&in_id, size)?;
        let result = backend.download_buffer(&out_id)?;
        assert_eq!(result.len(), size);

        for idx in 0..size {
            assert!(
                (result[idx] - expected[idx]).abs() < 1e-3,
                "mismatch at {}: got {} expected {}",
                idx,
                result[idx],
                expected[idx]
            );
        }

        Ok(())
    }

    #[test]
    fn oxicuda_cuda_resident_add_bias_parity() -> crate::errors::Result<()> {
        // Input is [m=3, n=4] row-major; bias is length n=4, broadcast down each row.
        let m = 3usize;
        let n = 4usize;
        let input: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let bias = vec![10.0f32, 20.0, 30.0, 40.0];

        // CPU reference: out[i, j] = input[i, j] + bias[j].
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                expected[i * n + j] = input[i * n + j] + bias[j];
            }
        }

        let backend = match OxicudaCudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping oxicuda CUDA resident-add-bias test: no CUDA device available");
                return Ok(());
            },
        };

        let in_id = backend.create_persistent_buffer(&input)?;
        let bias_id = backend.create_persistent_buffer(&bias)?;
        let out_id = backend.add_bias_gpu_to_gpu(&in_id, &bias_id, m, n)?;
        let result = backend.download_buffer(&out_id)?;
        assert_eq!(result.len(), m * n);

        for idx in 0..(m * n) {
            assert!(
                (result[idx] - expected[idx]).abs() < 1e-3,
                "mismatch at {}: got {} expected {}",
                idx,
                result[idx],
                expected[idx]
            );
        }

        Ok(())
    }

    #[test]
    fn oxicuda_cuda_resident_layernorm_parity() -> crate::errors::Result<()> {
        let seq_len = 2usize;
        let hidden_size = 4usize;
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let weight = vec![1.0f32; hidden_size];
        let bias = vec![0.0f32; hidden_size];
        let eps = 1e-5f32;

        // CPU reference: per-row population-variance layer norm.
        let mut expected = vec![0.0f32; seq_len * hidden_size];
        for row in 0..seq_len {
            let offset = row * hidden_size;
            let mut sum = 0.0f32;
            for i in 0..hidden_size {
                sum += input[offset + i];
            }
            let mean = sum / hidden_size as f32;
            let mut var_sum = 0.0f32;
            for i in 0..hidden_size {
                let diff = input[offset + i] - mean;
                var_sum += diff * diff;
            }
            let variance = var_sum / hidden_size as f32;
            let std_dev = (variance + eps).sqrt();
            for i in 0..hidden_size {
                let normalized = (input[offset + i] - mean) / std_dev;
                expected[offset + i] = normalized * weight[i] + bias[i];
            }
        }

        let backend = match OxicudaCudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!(
                    "Skipping oxicuda CUDA resident-LayerNorm test: no CUDA device available"
                );
                return Ok(());
            },
        };

        let in_id = backend.create_persistent_buffer(&input)?;
        let weight_id = backend.create_persistent_buffer(&weight)?;
        let bias_id = backend.create_persistent_buffer(&bias)?;

        let out_id = backend.layernorm_gpu_to_gpu(
            &in_id,
            &weight_id,
            &bias_id,
            seq_len,
            hidden_size,
            eps,
        )?;
        let result = backend.download_buffer(&out_id)?;
        assert_eq!(result.len(), seq_len * hidden_size);

        for idx in 0..(seq_len * hidden_size) {
            assert!(
                (result[idx] - expected[idx]).abs() < 1e-3,
                "mismatch at {}: got {} expected {}",
                idx,
                result[idx],
                expected[idx]
            );
        }

        Ok(())
    }

    #[test]
    fn oxicuda_cuda_matmul_with_cached_weight_parity() -> crate::errors::Result<()> {
        // A is [m=2, k=3] (host activations), B is [k=3, n=2] (cached weight), row-major.
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let m = 2usize;
        let k = 3usize;
        let n = 2usize;

        // Naive CPU reference (triple loop), row-major.
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    expected[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }

        let backend = match OxicudaCudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!(
                    "Skipping oxicuda CUDA cached-weight matmul test: no CUDA device available"
                );
                return Ok(());
            },
        };

        // Park the weight on the device, then multiply host activations against it twice
        // (the cached weight must survive repeated forward passes).
        let weight_id = backend.create_persistent_buffer(&b)?;
        for _ in 0..2 {
            let result = backend.matmul_with_cached_weight(&a, &weight_id, m, k, n)?;
            assert_eq!(result.len(), m * n);
            for idx in 0..(m * n) {
                assert!(
                    (result[idx] - expected[idx]).abs() < 1e-3,
                    "mismatch at {}: got {} expected {}",
                    idx,
                    result[idx],
                    expected[idx]
                );
            }
        }

        Ok(())
    }

    #[test]
    fn oxicuda_cuda_device_info_reports_ordinal() -> crate::errors::Result<()> {
        let backend = match OxicudaCudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping oxicuda CUDA device-info test: no CUDA device available");
                return Ok(());
            },
        };

        let info = backend.device_info();
        assert!(
            info.contains("ordinal: 0"),
            "device_info should report device ordinal 0, got {:?}",
            info
        );

        Ok(())
    }

    #[test]
    fn oxicuda_backend_singleton_is_shared_and_resident() -> crate::errors::Result<()> {
        // Probe whether a CUDA device exists; skip gracefully if not (CI without a GPU).
        if OxicudaCudaBackend::new(0).is_err() {
            eprintln!("Skipping oxicuda backend singleton test: no CUDA device available");
            return Ok(());
        }

        // Two get-or-create calls for the same device must return the *same* backend Arc.
        let b1 = oxicuda_backend(0)?;
        let b2 = oxicuda_backend(0)?;
        assert!(
            Arc::ptr_eq(&b1, &b2),
            "oxicuda_backend(0) must return the same Arc on repeated calls"
        );

        // A resident buffer created through one handle is visible through a later handle
        // obtained from the singleton — proving the backend (and its cache) persists.
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let id = b1.create_persistent_buffer(&data)?;
        let b3 = oxicuda_backend(0)?;
        assert_eq!(
            b3.get_persistent_buffer(&id)?,
            data.len(),
            "resident buffer minted via the singleton must be visible to a later handle"
        );
        let round_trip = b3.download_buffer(&id)?;
        assert_eq!(round_trip, data);

        // Clean up so this test leaves no resident state behind for sibling tests.
        b3.remove_persistent_buffer(&id)?;

        Ok(())
    }
}
