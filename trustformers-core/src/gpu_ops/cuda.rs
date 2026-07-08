//! CUDA GPU backend for tensor operations
//!
//! The `cuda` feature is implemented by the Pure-Rust **oxicuda** backend
//! (`oxicuda/mod.rs`). The legacy `cudarc` backend has been removed.

#[cfg(feature = "cuda")]
mod oxicuda;
#[cfg(feature = "cuda")]
pub use oxicuda::*;

// `CudaTensorData` (and the `feature = "cuda"`-gated call sites in
// `layers/linear.rs`, `layers/layernorm.rs`, `tensor/utils.rs`, `ops/activations.rs`)
// refer to `crate::gpu_ops::cuda::BufferId` and `crate::gpu_ops::cuda::get_cuda_backend`.
// Map those legacy names onto the oxicuda implementation so they keep resolving now that
// oxicuda *is* the `cuda` backend. `OxicudaCudaBackend` mirrors the former cudarc
// backend's method surface signature-for-signature, so the call sites are unchanged.
#[cfg(feature = "cuda")]
pub use oxicuda::oxicuda_backend as get_cuda_backend;
#[cfg(feature = "cuda")]
pub use oxicuda::OxiCudaBufferId as BufferId;
// Reference-counted RAII handle over a resident `BufferId`: `CudaTensorData` (and the
// `Linear` weight cache) hold this instead of a raw id so the device allocation is freed
// when the last clone drops instead of leaking until `clear_buffer_cache()`.
#[cfg(feature = "cuda")]
pub use oxicuda::OxiCudaBufferHandle as BufferHandle;
