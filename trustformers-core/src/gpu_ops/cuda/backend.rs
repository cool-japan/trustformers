//! CUDA backend structure and initialization

use super::types::{BufferCache, BufferId};
use crate::errors::{Result, TrustformersError};

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::driver::{CudaContext, CudaStream};
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use std::sync::Arc;

/// CUDA GPU backend for matrix multiplication and element-wise operations
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub struct CudaBackend {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) buffer_cache: Arc<std::sync::Mutex<BufferCache>>,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaBackend {
    /// Create a new CUDA backend
    pub fn new(device_id: usize) -> Result<Self> {
        let context = CudaContext::new(device_id).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to create CUDA context: {}", e),
                "CudaBackend::new",
            )
        })?;

        let stream = context.default_stream();

        println!("✓ CUDA backend initialized on device {}", device_id);

        Ok(Self {
            context,
            stream,
            buffer_cache: Arc::new(std::sync::Mutex::new(BufferCache::new())),
        })
    }

    /// Get device information
    pub fn device_info(&self) -> String {
        format!("CUDA Device {}", self.context.ordinal())
    }
}

// Re-export for public API
pub use BufferId;
