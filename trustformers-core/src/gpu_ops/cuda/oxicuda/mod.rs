//! Pure-Rust oxicuda CUDA backend for tensor operations.
//!
//! This module is the Campaign C1 successor to the Campaign C0 `oxicuda_spike`
//! probe: it graduates the validated oxicuda API surface into a real, feature-gated
//! CUDA backend. It provides host-in / host-out f32 kernels (GEMM, GELU, LayerNorm,
//! causal softmax, RoPE), a GPU-resident persistent-buffer subsystem with a
//! refcounted handle lifecycle, and — in the [`batched`] submodule — the batched /
//! broadcast matrix-multiply routing used by `Tensor::matmul` for both host and
//! GPU-resident (`Tensor::CUDA`) operands.
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

mod attention;
mod batched;

pub use batched::{dispatch_oxicuda_matmul, dispatch_oxicuda_matmul_resident, BatchedMatmulPlan};

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

/// Callback fired exactly once when the last clone of an [`OxiCudaBufferHandle`] drops.
///
/// Receives the device ordinal and the buffer id so the callee can locate the owning
/// backend registry entry. Boxed so tests can substitute a host-side mock and verify the
/// refcount bookkeeping without CUDA hardware.
type ReleaseFn = Box<dyn Fn(usize, OxiCudaBufferId) + Send + Sync + 'static>;

/// Shared interior of an [`OxiCudaBufferHandle`]: identity plus the release callback.
///
/// Lives behind an [`Arc`]; its [`Drop`] runs exactly once — when the *last* handle clone
/// drops — which is precisely when the device allocation must be returned to the backend.
struct BufferHandleInner {
    buffer_id: OxiCudaBufferId,
    device_id: usize,
    release: ReleaseFn,
}

impl Drop for BufferHandleInner {
    fn drop(&mut self) {
        // Runs on the last handle drop only (Arc guarantees single execution). The
        // callback itself is infallible/best-effort: Drop must never panic.
        (self.release)(self.device_id, self.buffer_id);
    }
}

/// Reference-counted RAII handle to a GPU-resident persistent buffer.
///
/// This is the lifecycle layer the raw [`OxiCudaBufferId`] lacks: the backend's
/// `buffer_cache` owns the [`DeviceBuffer<f32>`] allocations, and without a handle every
/// resident op output would stay parked on the device until
/// [`OxicudaCudaBackend::clear_buffer_cache`] — a leak in any long-running forward loop.
/// Cloning a handle is a pure refcount increment ([`Arc::clone`]); when the last clone
/// drops, the release callback removes the buffer from the owning backend's cache,
/// freeing the device memory.
///
/// Interaction with the escape hatch: `clear_buffer_cache()` (and explicit
/// `remove_persistent_buffer`) stay valid. Buffer ids are minted from a process-wide
/// monotonic counter and never reused, and removal of an absent id is an idempotent
/// no-op — so a handle dropping *after* the cache was force-cleared cannot double-free,
/// it simply removes nothing.
///
/// Thread safety: no `unsafe` is involved. `OxiCudaBufferId` is a `Copy` `u64`,
/// `device_id` is a `usize`, and the callback is `Send + Sync` by construction, so
/// `Send`/`Sync` for the handle are auto-derived soundly by the compiler.
pub struct OxiCudaBufferHandle {
    inner: Arc<BufferHandleInner>,
}

impl OxiCudaBufferHandle {
    /// Wrap a freshly minted resident buffer id in a lifecycle-managed handle.
    ///
    /// The id must identify a buffer owned by the [`oxicuda_backend()`] registry entry for
    /// `device_id` (i.e. it came from `create_persistent_buffer` / a `*_gpu_to_gpu` op on
    /// that backend). Each raw id must be wrapped **at most once**; the wrap point is the
    /// single owner and all sharing goes through clones of the returned handle.
    pub fn new(buffer_id: OxiCudaBufferId, device_id: usize) -> Self {
        Self::with_release(buffer_id, device_id, Box::new(release_persistent_buffer))
    }

    /// Construct a handle with a custom release callback.
    ///
    /// Test seam: lets the host-side refcount tests observe the free call without any
    /// CUDA hardware. Production code should use [`new`](Self::new).
    pub(crate) fn with_release(
        buffer_id: OxiCudaBufferId,
        device_id: usize,
        release: ReleaseFn,
    ) -> Self {
        Self {
            inner: Arc::new(BufferHandleInner {
                buffer_id,
                device_id,
                release,
            }),
        }
    }

    /// The resident buffer id this handle keeps alive.
    #[inline]
    pub fn id(&self) -> OxiCudaBufferId {
        self.inner.buffer_id
    }

    /// The CUDA device ordinal the buffer lives on.
    #[inline]
    pub fn device_id(&self) -> usize {
        self.inner.device_id
    }

    /// Number of live handle clones sharing this buffer (test/diagnostic aid).
    #[inline]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl Clone for OxiCudaBufferHandle {
    fn clone(&self) -> Self {
        // Pure refcount increment; the device buffer itself is never copied.
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl std::fmt::Debug for OxiCudaBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OxiCudaBufferHandle")
            .field("buffer_id", &self.inner.buffer_id)
            .field("device_id", &self.inner.device_id)
            .field("ref_count", &Arc::strong_count(&self.inner))
            .finish()
    }
}

/// Default release path for [`OxiCudaBufferHandle`]: free the buffer via the backend registry.
///
/// Best-effort by design — this runs from `Drop` and must never panic or block on driver
/// initialization:
/// - If no backend exists for `device_id` (e.g. the cache entry was never created, or the
///   process is tearing down), the release is a silent no-op; a backend is deliberately
///   **not** constructed here (backend construction touches the CUDA driver).
/// - The registry lock guard is released *before* the backend's own cache lock is taken
///   (the `match` scrutinee's guard drops at the end of the `let` statement), so the two
///   mutexes are never held simultaneously and no lock-order inversion can occur.
/// - `remove_persistent_buffer` is idempotent, so racing `clear_buffer_cache()` or an
///   explicit removal is harmless (ids are never reused).
fn release_persistent_buffer(device_id: usize, buffer_id: OxiCudaBufferId) {
    let backend = match OXICUDA_BACKENDS.lock() {
        Ok(cache) => cache.get(&device_id).cloned(),
        // A poisoned registry means another thread panicked mid-insert; freeing is
        // no longer safe to attempt and leaking one buffer at panic time is acceptable.
        Err(_) => None,
    };
    if let Some(backend) = backend {
        let _ = backend.remove_persistent_buffer(&buffer_id);
    }
}

/// Default CUDA device ordinal for operations with no resident tensor to carry one
/// (e.g. the host-in/host-out matmul dispatch in `Tensor::matmul`).
///
/// Defaults to `0`. Multi-GPU machines can redirect host-dispatched work with the
/// `TRUSTFORMERS_CUDA_DEVICE` environment variable (a non-negative integer, parsed once
/// per process; malformed values fall back to `0`). The ordinal is still validated
/// against the real device count when the backend is constructed.
pub fn default_cuda_device_id() -> usize {
    static DEFAULT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEFAULT.get_or_init(|| {
        std::env::var("TRUSTFORMERS_CUDA_DEVICE")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
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
    /// Initializes the CUDA driver, validates `device_id` against the enumerated device
    /// count, selects the device, creates a context and a BLAS handle on it. Fails on any
    /// host without a usable NVIDIA GPU / `libcuda`, and on an out-of-range ordinal.
    pub fn new(device_id: usize) -> crate::errors::Result<Self> {
        oxicuda_driver::init().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to initialize CUDA driver: {}", e),
                "OxicudaCudaBackend::new",
            )
        })?;

        // Validate the requested ordinal against the real device count so multi-GPU
        // callers get a precise range error instead of an opaque driver failure.
        let device_count = oxicuda_driver::Device::count().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to enumerate CUDA devices: {}", e),
                "OxicudaCudaBackend::new",
            )
        })?;
        let device_count = usize::try_from(device_count).unwrap_or(0);
        if device_id >= device_count {
            return Err(TrustformersError::hardware_error(
                &format!(
                    "CUDA device index {} out of range: {} device(s) available",
                    device_id, device_count
                ),
                "OxicudaCudaBackend::new",
            ));
        }

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

    /// Block the host until every kernel queued on this backend's BLAS stream
    /// has completed.
    ///
    /// The oxicuda-blas GEMM / DNN kernels launch **asynchronously** on the
    /// handle's stream, which [`oxicuda_driver::Stream::new`] creates with the
    /// `CU_STREAM_NON_BLOCKING` flag. A synchronous device→host `copy_to_host`
    /// (`cuMemcpyDtoH_v2`) runs on the legacy *default* stream, and a
    /// non-blocking stream does **not** implicitly synchronise with the default
    /// stream. Without this barrier the host therefore reads a result buffer
    /// *before* the kernel that fills it has run — returning the
    /// zero-initialised allocation (`DeviceBuffer::zeroed`) or uninitialised
    /// garbage (`DeviceBuffer::alloc`). Every device→host read-back must call
    /// this first. Kept private: it is an internal invariant of the read-back
    /// paths, not part of the public backend surface.
    fn synchronize_stream(&self, op: &'static str) -> crate::errors::Result<()> {
        self.handle.stream().synchronize().map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to synchronize CUDA stream: {}", e),
                op,
            )
        })
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
        self.synchronize_stream("download_buffer")?;
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
        self.synchronize_stream("matmul_with_cached_weight")?;
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
        self.synchronize_stream("matmul_f32")?;
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
        self.synchronize_stream("gelu_f32")?;
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
        self.synchronize_stream("layernorm_f32")?;
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

        // One square [seq_len, seq_len] matrix: rows == cols == seq_len (the
        // kernel's seq_len parameter batches flattened matrices; unused here).
        oxicuda_blas::reduction::causal_softmax::<f32>(
            &self.handle,
            seq_len as u32,
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
        self.synchronize_stream("softmax_causal_f32")?;
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
        self.synchronize_stream("rope_f32")?;
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

#[cfg(all(
    test,
    feature = "cuda",
    any(target_os = "linux", target_os = "windows")
))]
mod tests;

/// Host-side refcount tests for the resident-buffer lifecycle.
///
/// These exercise only the [`OxiCudaBufferHandle`] / registry bookkeeping — the release
/// callback is mocked — so they never touch `libcuda` and run on any platform (including
/// the macOS development hosts where the `cuda` feature merely *compiles*). This follows
/// the module's existing pattern of keeping hardware-dependent parity tests OS-gated
/// while pure bookkeeping stays universally testable.
#[cfg(test)]
mod resident_handle_tests;
