//! CUDA GPU backend for tensor operations
//!
//! This module provides CUDA GPU acceleration for tensor operations on NVIDIA GPUs.
//! Uses cudarc for direct CUDA API access.
//!
//! Features:
//! - Pre-compiled CUDA kernels for common operations
//! - Persistent buffer caching to minimize CPU-GPU transfers
//! - GPU-to-GPU operations (zero CPU transfers)
//! - Cached weight buffers for inference optimization

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::driver::CudaSlice;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use std::collections::HashMap;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use std::sync::Arc;

/// Buffer ID for persistent GPU buffers
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

#[cfg(feature = "cuda")]
impl BufferId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        BufferId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

#[cfg(feature = "cuda")]
impl Default for BufferId {
    fn default() -> Self {
        Self::new()
    }
}

/// Persistent buffer cache for CUDA GPU
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub(crate) struct BufferCache {
    buffers: HashMap<BufferId, Arc<CudaSlice<f32>>>,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl BufferCache {
    pub(crate) fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    pub(crate) fn insert(&mut self, id: BufferId, buffer: Arc<CudaSlice<f32>>) {
        self.buffers.insert(id, buffer);
    }

    pub(crate) fn get(&self, id: &BufferId) -> Option<Arc<CudaSlice<f32>>> {
        self.buffers.get(id).cloned()
    }

    pub(crate) fn remove(&mut self, id: &BufferId) -> Option<Arc<CudaSlice<f32>>> {
        self.buffers.remove(id)
    }

    pub(crate) fn clear(&mut self) {
        self.buffers.clear();
    }

    pub(crate) fn len(&self) -> usize {
        self.buffers.len()
    }
}
