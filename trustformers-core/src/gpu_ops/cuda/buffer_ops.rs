//! CUDA buffer management operations

use super::backend::CudaBackend;
use super::types::BufferId;
use crate::errors::{Result, TrustformersError};

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::driver::CudaSlice;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use std::sync::Arc;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaBackend {
    /// Create a persistent GPU buffer and return its ID
    pub fn create_persistent_buffer(&self, data: &[f32]) -> Result<BufferId> {
        let buffer = Arc::new(self.stream.memcpy_stod(data).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy data to device: {}", e),
                "create_persistent_buffer",
            )
        })?);

        let buffer_id = BufferId::new();

        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "create_persistent_buffer",
            )
        })?;

        cache.insert(buffer_id, buffer);
        Ok(buffer_id)
    }

    /// Get a persistent buffer by ID
    pub fn get_persistent_buffer(&self, id: &BufferId) -> Result<Arc<CudaSlice<f32>>> {
        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "get_persistent_buffer",
            )
        })?;

        cache.get(id).ok_or_else(|| {
            TrustformersError::hardware_error(
                &format!("Buffer {:?} not found in cache", id),
                "get_persistent_buffer",
            )
        })
    }

    /// Remove a persistent buffer from cache
    pub fn remove_persistent_buffer(&self, id: &BufferId) -> Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error(
                "Failed to lock buffer cache",
                "remove_persistent_buffer",
            )
        })?;

        cache.remove(id);
        Ok(())
    }

    /// Clear all persistent buffers
    pub fn clear_buffer_cache(&self) -> Result<()> {
        let mut cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "clear_buffer_cache")
        })?;

        cache.clear();
        Ok(())
    }

    /// Get the number of cached buffers
    pub fn buffer_cache_size(&self) -> Result<usize> {
        let cache = self.buffer_cache.lock().map_err(|_| {
            TrustformersError::hardware_error("Failed to lock buffer cache", "buffer_cache_size")
        })?;

        Ok(cache.len())
    }

    /// Download data from GPU buffer to CPU
    pub fn download_buffer(&self, buffer_id: &BufferId) -> Result<Vec<f32>> {
        let buffer = self.get_persistent_buffer(buffer_id)?;
        let data_vec = self.stream.memcpy_dtov(&*buffer).map_err(|e| {
            TrustformersError::hardware_error(
                &format!("Failed to copy data from device: {}", e),
                "download_buffer",
            )
        })?;
        Ok(data_vec)
    }

    /// Download data from GPU buffer to CPU (alias for consistency with Metal backend)
    pub fn buffer_to_cpu(&self, buffer_id: &BufferId) -> Result<Vec<f32>> {
        self.download_buffer(buffer_id)
    }
}
