//! Memory pooling system for TrustformeRS C API
//!
//! This module provides efficient memory pooling for tensor allocations,
//! reducing allocation overhead and memory fragmentation.

use crate::error::TrustformersError;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Memory pool handle type
pub type TrustformersMemoryPoolHandle = usize;

/// Memory pool configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TrustformersMemoryPoolConfig {
    /// Initial pool size (bytes)
    pub initial_size: usize,
    /// Maximum pool size (bytes), 0 = unlimited
    pub max_size: usize,
    /// Block size for allocations (bytes)
    pub block_size: usize,
    /// Number of size classes
    pub num_size_classes: usize,
    /// Enable automatic growth
    pub auto_grow: c_int,
    /// Growth factor (e.g., 2.0 = double on growth)
    pub growth_factor: f32,
    /// Enable garbage collection
    pub enable_gc: c_int,
    /// GC threshold (% of pool used)
    pub gc_threshold: f32,
    /// Enable statistics tracking
    pub enable_stats: c_int,
    /// Thread-local caching
    pub enable_thread_cache: c_int,
}

impl Default for TrustformersMemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 64 * 1024 * 1024, // 64 MB
            max_size: 0,                    // Unlimited
            block_size: 4096,               // 4 KB
            num_size_classes: 16,
            auto_grow: 1,
            growth_factor: 2.0,
            enable_gc: 1,
            gc_threshold: 0.8, // 80%
            enable_stats: 1,
            enable_thread_cache: 1,
        }
    }
}

/// Memory block within pool
#[derive(Debug)]
struct MemoryBlock {
    ptr: *mut u8,
    size: usize,
    is_free: bool,
    allocated_at: Option<Instant>,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

/// Size class for different allocation sizes
#[derive(Debug)]
struct SizeClass {
    size: usize,
    free_blocks: VecDeque<MemoryBlock>,
    allocated_blocks: HashMap<*mut u8, MemoryBlock>,
}

// SAFETY: SizeClass contains MemoryBlocks which are already marked Send/Sync.
// Access is synchronized through MemoryPool's RwLock.
unsafe impl Send for SizeClass {}
unsafe impl Sync for SizeClass {}

impl SizeClass {
    fn new(size: usize) -> Self {
        Self {
            size,
            free_blocks: VecDeque::new(),
            allocated_blocks: HashMap::new(),
        }
    }

    fn allocate(&mut self) -> Option<*mut u8> {
        if let Some(mut block) = self.free_blocks.pop_front() {
            block.is_free = false;
            block.allocated_at = Some(Instant::now());
            let ptr = block.ptr;
            self.allocated_blocks.insert(ptr, block);
            Some(ptr)
        } else {
            None
        }
    }

    fn deallocate(&mut self, ptr: *mut u8) -> bool {
        if let Some(mut block) = self.allocated_blocks.remove(&ptr) {
            block.is_free = true;
            block.allocated_at = None;
            self.free_blocks.push_back(block);
            true
        } else {
            false
        }
    }
}

/// Memory pool
#[derive(Debug)]
pub struct MemoryPool {
    config: TrustformersMemoryPoolConfig,
    size_classes: Vec<SizeClass>,
    total_allocated: usize,
    total_freed: usize,
    peak_usage: usize,
    allocation_count: u64,
    deallocation_count: u64,
    created_at: Instant,
}

// SAFETY: MemoryPool is accessed through RwLock synchronization in MemoryPoolRegistry.
// Raw pointer lifetimes in MemoryBlocks are managed by the pool.
unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl MemoryPool {
    fn new(config: TrustformersMemoryPoolConfig) -> Self {
        let mut size_classes = Vec::new();

        // Create size classes: 64B, 128B, 256B, ..., up to block_size
        let mut size = 64;
        for _ in 0..config.num_size_classes {
            size_classes.push(SizeClass::new(size));
            size *= 2;
            if size > config.block_size {
                break;
            }
        }

        Self {
            config,
            size_classes,
            total_allocated: 0,
            total_freed: 0,
            peak_usage: 0,
            allocation_count: 0,
            deallocation_count: 0,
            created_at: Instant::now(),
        }
    }

    fn allocate(&mut self, size: usize) -> Option<*mut u8> {
        // Find appropriate size class
        for size_class in &mut self.size_classes {
            if size <= size_class.size {
                if let Some(ptr) = size_class.allocate() {
                    self.total_allocated += size;
                    self.allocation_count += 1;

                    let current_usage = self.total_allocated - self.total_freed;
                    if current_usage > self.peak_usage {
                        self.peak_usage = current_usage;
                    }

                    return Some(ptr);
                }
            }
        }

        // No suitable block found, would need to grow pool or allocate from system
        None
    }

    fn deallocate(&mut self, ptr: *mut u8, size: usize) -> bool {
        for size_class in &mut self.size_classes {
            if size <= size_class.size {
                if size_class.deallocate(ptr) {
                    self.total_freed += size;
                    self.deallocation_count += 1;
                    return true;
                }
            }
        }
        false
    }

    fn current_usage(&self) -> usize {
        self.total_allocated.saturating_sub(self.total_freed)
    }

    fn fragmentation(&self) -> f32 {
        if self.total_allocated == 0 {
            return 0.0;
        }

        let free_memory: usize =
            self.size_classes.iter().map(|sc| sc.free_blocks.len() * sc.size).sum();

        let total_memory = self.total_allocated;

        (free_memory as f32) / (total_memory as f32)
    }

    fn needs_gc(&self) -> bool {
        if self.config.enable_gc == 0 {
            return false;
        }

        let usage_ratio = self.current_usage() as f32 / self.config.initial_size as f32;
        usage_ratio >= self.config.gc_threshold
    }

    fn run_gc(&mut self) {
        // Simple GC: return unused blocks to system
        for size_class in &mut self.size_classes {
            // Keep only a few free blocks per size class
            while size_class.free_blocks.len() > 10 {
                if let Some(block) = size_class.free_blocks.pop_back() {
                    // Would call libc free or similar here
                    drop(block);
                }
            }
        }
    }
}

/// Global memory pool registry
static MEMORY_POOL_REGISTRY: Lazy<RwLock<MemoryPoolRegistry>> =
    Lazy::new(|| RwLock::new(MemoryPoolRegistry::new()));

struct MemoryPoolRegistry {
    pools: HashMap<usize, Arc<RwLock<MemoryPool>>>,
    default_pool: Option<Arc<RwLock<MemoryPool>>>,
    next_handle: usize,
}

impl MemoryPoolRegistry {
    fn new() -> Self {
        Self {
            pools: HashMap::new(),
            default_pool: None,
            next_handle: 1,
        }
    }

    fn register(&mut self, pool: MemoryPool) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;
        let pool_arc = Arc::new(RwLock::new(pool));
        self.pools.insert(handle, pool_arc.clone());

        // First pool becomes default
        if self.default_pool.is_none() {
            self.default_pool = Some(pool_arc);
        }

        handle
    }

    fn get(&self, handle: usize) -> Option<Arc<RwLock<MemoryPool>>> {
        self.pools.get(&handle).cloned()
    }

    fn remove(&mut self, handle: usize) -> bool {
        self.pools.remove(&handle).is_some()
    }

    fn get_default(&self) -> Option<Arc<RwLock<MemoryPool>>> {
        self.default_pool.clone()
    }
}

/// Create a memory pool
#[no_mangle]
pub extern "C" fn trustformers_memory_pool_create(
    config: *const TrustformersMemoryPoolConfig,
    handle: *mut TrustformersMemoryPoolHandle,
) -> TrustformersError {
    if handle.is_null() {
        return TrustformersError::NullPointer;
    }

    let config = if config.is_null() {
        TrustformersMemoryPoolConfig::default()
    } else {
        unsafe { (*config).clone() }
    };

    let pool = MemoryPool::new(config);
    let pool_handle = MEMORY_POOL_REGISTRY.write().register(pool);

    unsafe {
        *handle = pool_handle;
    }

    TrustformersError::Success
}

/// Allocate memory from pool
#[no_mangle]
pub extern "C" fn trustformers_memory_pool_alloc(
    pool: TrustformersMemoryPoolHandle,
    size: usize,
    ptr: *mut *mut u8,
) -> TrustformersError {
    if ptr.is_null() {
        return TrustformersError::NullPointer;
    }

    if size == 0 {
        return TrustformersError::InvalidParameter;
    }

    let registry = MEMORY_POOL_REGISTRY.read();
    let pool_arc = if pool == 0 { registry.get_default() } else { registry.get(pool) };

    let Some(pool_arc) = pool_arc else {
        return TrustformersError::InvalidHandle;
    };

    drop(registry);

    let mut pool_lock = pool_arc.write();

    match pool_lock.allocate(size) {
        Some(allocated_ptr) => {
            unsafe {
                *ptr = allocated_ptr;
            }

            // Check if GC is needed
            if pool_lock.needs_gc() {
                pool_lock.run_gc();
            }

            TrustformersError::Success
        },
        None => TrustformersError::OutOfMemory,
    }
}

/// Deallocate memory back to pool
#[no_mangle]
pub extern "C" fn trustformers_memory_pool_free(
    pool: TrustformersMemoryPoolHandle,
    ptr: *mut u8,
    size: usize,
) -> TrustformersError {
    if ptr.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = MEMORY_POOL_REGISTRY.read();
    let pool_arc = if pool == 0 { registry.get_default() } else { registry.get(pool) };

    let Some(pool_arc) = pool_arc else {
        return TrustformersError::InvalidHandle;
    };

    drop(registry);

    let mut pool_lock = pool_arc.write();

    if pool_lock.deallocate(ptr, size) {
        TrustformersError::Success
    } else {
        TrustformersError::InvalidParameter
    }
}

/// Memory pool statistics
#[repr(C)]
#[derive(Debug, Default)]
pub struct TrustformersMemoryPoolStats {
    /// Total bytes allocated
    pub total_allocated: u64,
    /// Total bytes freed
    pub total_freed: u64,
    /// Current usage (bytes)
    pub current_usage: u64,
    /// Peak usage (bytes)
    pub peak_usage: u64,
    /// Number of allocations
    pub allocation_count: u64,
    /// Number of deallocations
    pub deallocation_count: u64,
    /// Fragmentation ratio (0.0 to 1.0)
    pub fragmentation: f32,
    /// Pool uptime (seconds)
    pub uptime_seconds: u64,
    /// Number of size classes
    pub num_size_classes: usize,
}

/// Get memory pool statistics
#[no_mangle]
pub extern "C" fn trustformers_memory_pool_get_stats(
    pool: TrustformersMemoryPoolHandle,
    stats: *mut TrustformersMemoryPoolStats,
) -> TrustformersError {
    if stats.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = MEMORY_POOL_REGISTRY.read();
    let pool_arc = if pool == 0 { registry.get_default() } else { registry.get(pool) };

    let Some(pool_arc) = pool_arc else {
        return TrustformersError::InvalidHandle;
    };

    drop(registry);

    let pool_lock = pool_arc.read();

    unsafe {
        let s = &mut *stats;
        s.total_allocated = pool_lock.total_allocated as u64;
        s.total_freed = pool_lock.total_freed as u64;
        s.current_usage = pool_lock.current_usage() as u64;
        s.peak_usage = pool_lock.peak_usage as u64;
        s.allocation_count = pool_lock.allocation_count;
        s.deallocation_count = pool_lock.deallocation_count;
        s.fragmentation = pool_lock.fragmentation();
        s.uptime_seconds = pool_lock.created_at.elapsed().as_secs();
        s.num_size_classes = pool_lock.size_classes.len();
    }

    TrustformersError::Success
}

/// Trigger garbage collection
#[no_mangle]
pub extern "C" fn trustformers_memory_pool_gc(
    pool: TrustformersMemoryPoolHandle,
) -> TrustformersError {
    let registry = MEMORY_POOL_REGISTRY.read();
    let pool_arc = if pool == 0 { registry.get_default() } else { registry.get(pool) };

    let Some(pool_arc) = pool_arc else {
        return TrustformersError::InvalidHandle;
    };

    drop(registry);

    let mut pool_lock = pool_arc.write();
    pool_lock.run_gc();

    TrustformersError::Success
}

/// Reset pool statistics
#[no_mangle]
pub extern "C" fn trustformers_memory_pool_reset_stats(
    pool: TrustformersMemoryPoolHandle,
) -> TrustformersError {
    let registry = MEMORY_POOL_REGISTRY.read();
    let pool_arc = if pool == 0 { registry.get_default() } else { registry.get(pool) };

    let Some(pool_arc) = pool_arc else {
        return TrustformersError::InvalidHandle;
    };

    drop(registry);

    let mut pool_lock = pool_arc.write();
    pool_lock.allocation_count = 0;
    pool_lock.deallocation_count = 0;

    TrustformersError::Success
}

/// Destroy memory pool
#[no_mangle]
pub extern "C" fn trustformers_memory_pool_destroy(
    pool: TrustformersMemoryPoolHandle,
) -> TrustformersError {
    if pool == 0 {
        return TrustformersError::InvalidHandle;
    }

    let removed = MEMORY_POOL_REGISTRY.write().remove(pool);

    if removed {
        TrustformersError::Success
    } else {
        TrustformersError::InvalidHandle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_create() {
        let mut handle: TrustformersMemoryPoolHandle = 0;
        let err = trustformers_memory_pool_create(ptr::null(), &mut handle);

        assert_eq!(err, TrustformersError::Success);
        assert_ne!(handle, 0);

        let err = trustformers_memory_pool_destroy(handle);
        assert_eq!(err, TrustformersError::Success);
    }

    #[test]
    fn test_memory_pool_stats() {
        let mut handle: TrustformersMemoryPoolHandle = 0;
        trustformers_memory_pool_create(ptr::null(), &mut handle);

        let mut stats = TrustformersMemoryPoolStats::default();
        let err = trustformers_memory_pool_get_stats(handle, &mut stats);

        assert_eq!(err, TrustformersError::Success);
        assert_eq!(stats.allocation_count, 0);

        trustformers_memory_pool_destroy(handle);
    }

    #[test]
    fn test_pool_config_default() {
        let config = TrustformersMemoryPoolConfig::default();
        assert_eq!(config.block_size, 4096);
        assert_eq!(config.auto_grow, 1);
    }
}
