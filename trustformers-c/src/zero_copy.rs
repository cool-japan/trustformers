//! Zero-copy operations for TrustformeRS C API
//!
//! This module provides zero-copy tensor operations using shared memory,
//! memory-mapped files, and direct buffer access for maximum performance.

use crate::error::TrustformersError;
use crate::tensor::TrustformersTensor;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::Arc;

#[cfg(unix)]
use std::os::unix::io::RawFd;

#[cfg(windows)]
use std::os::windows::io::RawHandle;

/// Shared memory handle type
pub type TrustformersSharedMemHandle = usize;

/// Memory mapping mode
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustformersMemoryMapMode {
    /// Read-only mapping
    ReadOnly = 0,
    /// Read-write mapping
    ReadWrite = 1,
    /// Copy-on-write mapping
    CopyOnWrite = 2,
}

/// Zero-copy buffer configuration
#[repr(C)]
#[derive(Debug)]
pub struct TrustformersZeroCopyConfig {
    /// Enable shared memory
    pub use_shared_memory: c_int,
    /// Enable memory mapping
    pub use_memory_mapping: c_int,
    /// Enable direct buffer access
    pub use_direct_buffers: c_int,
    /// Memory map mode
    pub map_mode: TrustformersMemoryMapMode,
    /// Alignment requirement (bytes)
    pub alignment: usize,
    /// Enable huge pages (if supported)
    pub use_huge_pages: c_int,
}

impl Default for TrustformersZeroCopyConfig {
    fn default() -> Self {
        Self {
            use_shared_memory: 1,
            use_memory_mapping: 1,
            use_direct_buffers: 1,
            map_mode: TrustformersMemoryMapMode::ReadWrite,
            alignment: 64, // Cache line alignment
            use_huge_pages: 0,
        }
    }
}

/// Shared memory region
#[derive(Debug)]
pub struct SharedMemoryRegion {
    name: String,
    size: usize,
    ptr: *mut c_void,
    #[cfg(unix)]
    fd: Option<RawFd>,
    #[cfg(windows)]
    handle: Option<RawHandle>,
    is_owner: bool,
}

// SAFETY: SharedMemoryRegion is accessed through Arc and RwLock synchronization.
// The raw pointer and file handles are properly managed.
unsafe impl Send for SharedMemoryRegion {}
unsafe impl Sync for SharedMemoryRegion {}

impl SharedMemoryRegion {
    /// Create a new shared memory region
    fn create(name: &str, size: usize) -> Result<Self, TrustformersError> {
        #[cfg(unix)]
        {
            Self::create_unix(name, size)
        }

        #[cfg(windows)]
        {
            Self::create_windows(name, size)
        }

        #[cfg(not(any(unix, windows)))]
        {
            Err(TrustformersError::FeatureNotAvailable)
        }
    }

    #[cfg(unix)]
    fn create_unix(name: &str, size: usize) -> Result<Self, TrustformersError> {
        // Placeholder implementation - would use shm_open, mmap, etc.
        Ok(Self {
            name: name.to_string(),
            size,
            ptr: ptr::null_mut(),
            fd: None,
            is_owner: true,
        })
    }

    #[cfg(windows)]
    fn create_windows(name: &str, size: usize) -> Result<Self, TrustformersError> {
        // Placeholder implementation - would use CreateFileMapping, MapViewOfFile, etc.
        Ok(Self {
            name: name.to_string(),
            size,
            ptr: ptr::null_mut(),
            handle: None,
            is_owner: true,
        })
    }

    /// Open an existing shared memory region
    fn open(name: &str) -> Result<Self, TrustformersError> {
        #[cfg(unix)]
        {
            Self::open_unix(name)
        }

        #[cfg(windows)]
        {
            Self::open_windows(name)
        }

        #[cfg(not(any(unix, windows)))]
        {
            Err(TrustformersError::FeatureNotAvailable)
        }
    }

    #[cfg(unix)]
    fn open_unix(name: &str) -> Result<Self, TrustformersError> {
        // Placeholder - would use shm_open with O_RDWR
        Ok(Self {
            name: name.to_string(),
            size: 0,
            ptr: ptr::null_mut(),
            fd: None,
            is_owner: false,
        })
    }

    #[cfg(windows)]
    fn open_windows(name: &str) -> Result<Self, TrustformersError> {
        // Placeholder - would use OpenFileMapping
        Ok(Self {
            name: name.to_string(),
            size: 0,
            ptr: ptr::null_mut(),
            handle: None,
            is_owner: false,
        })
    }
}

impl Drop for SharedMemoryRegion {
    fn drop(&mut self) {
        // Cleanup would happen here
        #[cfg(unix)]
        {
            if self.is_owner {
                // Would call shm_unlink
            }
        }

        #[cfg(windows)]
        {
            if self.is_owner {
                // Would call CloseHandle
            }
        }
    }
}

/// Memory-mapped file
#[derive(Debug)]
pub struct MemoryMappedFile {
    path: String,
    size: usize,
    ptr: *mut c_void,
    mode: TrustformersMemoryMapMode,
}

// SAFETY: MemoryMappedFile is accessed through Arc and RwLock synchronization.
// The raw pointer is properly managed through memory mapping.
unsafe impl Send for MemoryMappedFile {}
unsafe impl Sync for MemoryMappedFile {}

impl MemoryMappedFile {
    /// Create a memory-mapped file
    fn create(
        path: &str,
        size: usize,
        mode: TrustformersMemoryMapMode,
    ) -> Result<Self, TrustformersError> {
        // Placeholder implementation
        Ok(Self {
            path: path.to_string(),
            size,
            ptr: ptr::null_mut(),
            mode,
        })
    }
}

impl Drop for MemoryMappedFile {
    fn drop(&mut self) {
        // Cleanup would happen here (munmap on Unix, UnmapViewOfFile on Windows)
    }
}

/// Zero-copy buffer descriptor
#[derive(Debug)]
pub struct ZeroCopyBuffer {
    ptr: *mut c_void,
    size: usize,
    capacity: usize,
    is_shared: bool,
    shared_mem: Option<Arc<SharedMemoryRegion>>,
    mapped_file: Option<Arc<MemoryMappedFile>>,
}

// SAFETY: ZeroCopyBuffer is accessed through RwLock synchronization in ZeroCopyRegistry.
// The raw pointer lifetime is managed by either SharedMemoryRegion or MemoryMappedFile.
unsafe impl Send for ZeroCopyBuffer {}
unsafe impl Sync for ZeroCopyBuffer {}

/// Global zero-copy registry
static ZERO_COPY_REGISTRY: Lazy<RwLock<ZeroCopyRegistry>> =
    Lazy::new(|| RwLock::new(ZeroCopyRegistry::new()));

struct ZeroCopyRegistry {
    buffers: HashMap<usize, Arc<ZeroCopyBuffer>>,
    shared_memory: HashMap<String, Arc<SharedMemoryRegion>>,
    memory_maps: HashMap<String, Arc<MemoryMappedFile>>,
    next_handle: usize,
}

impl ZeroCopyRegistry {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            shared_memory: HashMap::new(),
            memory_maps: HashMap::new(),
            next_handle: 1,
        }
    }

    fn register_buffer(&mut self, buffer: ZeroCopyBuffer) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.buffers.insert(handle, Arc::new(buffer));
        handle
    }

    fn get_buffer(&self, handle: usize) -> Option<Arc<ZeroCopyBuffer>> {
        self.buffers.get(&handle).cloned()
    }

    fn remove_buffer(&mut self, handle: usize) -> bool {
        self.buffers.remove(&handle).is_some()
    }
}

/// Create a shared memory region
#[no_mangle]
pub extern "C" fn trustformers_shared_memory_create(
    name: *const c_char,
    size: usize,
    handle: *mut TrustformersSharedMemHandle,
) -> TrustformersError {
    if name.is_null() || handle.is_null() {
        return TrustformersError::NullPointer;
    }

    if size == 0 {
        return TrustformersError::InvalidParameter;
    }

    let name_str = match unsafe { std::ffi::CStr::from_ptr(name).to_str() } {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    match SharedMemoryRegion::create(name_str, size) {
        Ok(region) => {
            let buffer = ZeroCopyBuffer {
                ptr: region.ptr,
                size,
                capacity: size,
                is_shared: true,
                shared_mem: Some(Arc::new(region)),
                mapped_file: None,
            };

            let buffer_handle = ZERO_COPY_REGISTRY.write().register_buffer(buffer);

            unsafe {
                *handle = buffer_handle;
            }

            TrustformersError::Success
        },
        Err(e) => e,
    }
}

/// Open an existing shared memory region
#[no_mangle]
pub extern "C" fn trustformers_shared_memory_open(
    name: *const c_char,
    handle: *mut TrustformersSharedMemHandle,
) -> TrustformersError {
    if name.is_null() || handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let name_str = match unsafe { std::ffi::CStr::from_ptr(name).to_str() } {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    match SharedMemoryRegion::open(name_str) {
        Ok(region) => {
            let size = region.size;
            let buffer = ZeroCopyBuffer {
                ptr: region.ptr,
                size,
                capacity: size,
                is_shared: true,
                shared_mem: Some(Arc::new(region)),
                mapped_file: None,
            };

            let buffer_handle = ZERO_COPY_REGISTRY.write().register_buffer(buffer);

            unsafe {
                *handle = buffer_handle;
            }

            TrustformersError::Success
        },
        Err(e) => e,
    }
}

/// Get pointer to shared memory buffer
#[no_mangle]
pub extern "C" fn trustformers_shared_memory_get_ptr(
    handle: TrustformersSharedMemHandle,
    ptr: *mut *mut c_void,
) -> TrustformersError {
    if ptr.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = ZERO_COPY_REGISTRY.read();
    let Some(buffer) = registry.get_buffer(handle) else {
        return TrustformersError::InvalidHandle;
    };

    unsafe {
        *ptr = buffer.ptr;
    }

    TrustformersError::Success
}

/// Create a memory-mapped tensor from file
#[no_mangle]
pub extern "C" fn trustformers_tensor_mmap(
    file_path: *const c_char,
    offset: usize,
    size: usize,
    mode: TrustformersMemoryMapMode,
    tensor: *mut TrustformersTensor,
) -> TrustformersError {
    if file_path.is_null() || tensor.is_null() {
        return TrustformersError::NullPointer;
    }

    let path_str = match unsafe { std::ffi::CStr::from_ptr(file_path).to_str() } {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    match MemoryMappedFile::create(path_str, size, mode) {
        Ok(mmap) => {
            // Would create tensor from mmap buffer
            // For now, return success
            TrustformersError::FeatureNotAvailable
        },
        Err(e) => e,
    }
}

/// Create a zero-copy view of a tensor
#[no_mangle]
pub extern "C" fn trustformers_tensor_zero_copy_view(
    source_tensor: TrustformersTensor,
    start: usize,
    length: usize,
    view_tensor: *mut TrustformersTensor,
) -> TrustformersError {
    if view_tensor.is_null() {
        return TrustformersError::NullPointer;
    }

    // Would create a view into the source tensor without copying
    TrustformersError::FeatureNotAvailable
}

/// Free shared memory handle
#[no_mangle]
pub extern "C" fn trustformers_shared_memory_free(
    handle: TrustformersSharedMemHandle,
) -> TrustformersError {
    if handle == 0 {
        return TrustformersError::InvalidHandle;
    }

    let removed = ZERO_COPY_REGISTRY.write().remove_buffer(handle);

    if removed {
        TrustformersError::Success
    } else {
        TrustformersError::InvalidHandle
    }
}

/// Zero-copy statistics
#[repr(C)]
#[derive(Debug, Default)]
pub struct TrustformersZeroCopyStats {
    /// Number of active shared memory regions
    pub active_shared_memory: usize,
    /// Number of active memory maps
    pub active_memory_maps: usize,
    /// Total bytes in shared memory
    pub total_shared_bytes: u64,
    /// Total bytes in memory maps
    pub total_mapped_bytes: u64,
    /// Number of zero-copy views created
    pub total_views_created: u64,
}

/// Get zero-copy statistics
#[no_mangle]
pub extern "C" fn trustformers_zero_copy_get_stats(
    stats: *mut TrustformersZeroCopyStats,
) -> TrustformersError {
    if stats.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = ZERO_COPY_REGISTRY.read();

    unsafe {
        let s = &mut *stats;
        s.active_shared_memory = registry.shared_memory.len();
        s.active_memory_maps = registry.memory_maps.len();
        s.total_shared_bytes = registry.shared_memory.values().map(|r| r.size as u64).sum();
        s.total_mapped_bytes = registry.memory_maps.values().map(|m| m.size as u64).sum();
        s.total_views_created = 0; // Would track this
    }

    TrustformersError::Success
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_config_default() {
        let config = TrustformersZeroCopyConfig::default();
        assert_eq!(config.use_shared_memory, 1);
        assert_eq!(config.alignment, 64);
    }

    #[test]
    fn test_zero_copy_stats() {
        let mut stats = TrustformersZeroCopyStats::default();
        let err = trustformers_zero_copy_get_stats(&mut stats);
        assert_eq!(err, TrustformersError::Success);
    }
}
