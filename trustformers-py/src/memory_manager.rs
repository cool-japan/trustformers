use crate::errors::{TrustformersPyError, TrustformersPyResult};
use parking_lot::RwLock;
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Memory statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub current_usage: usize,
    pub peak_usage: usize,
    pub total_allocated: usize,
    pub total_deallocated: usize,
    pub active_allocations: usize,
}

/// Custom memory allocator with tracking
pub struct TrackingAllocator {
    stats: Arc<RwLock<MemoryStats>>,
    current_usage: AtomicUsize,
    peak_usage: AtomicUsize,
    total_allocated: AtomicUsize,
    total_deallocated: AtomicUsize,
    active_allocations: AtomicUsize,
}

impl TrackingAllocator {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(MemoryStats::default())),
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            total_allocated: AtomicUsize::new(0),
            total_deallocated: AtomicUsize::new(0),
            active_allocations: AtomicUsize::new(0),
        }
    }

    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage: self.current_usage.load(Ordering::Relaxed),
            peak_usage: self.peak_usage.load(Ordering::Relaxed),
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_deallocated: self.total_deallocated.load(Ordering::Relaxed),
            active_allocations: self.active_allocations.load(Ordering::Relaxed),
        }
    }

    pub fn reset_stats(&self) {
        self.current_usage.store(0, Ordering::Relaxed);
        self.peak_usage.store(0, Ordering::Relaxed);
        self.total_allocated.store(0, Ordering::Relaxed);
        self.total_deallocated.store(0, Ordering::Relaxed);
        self.active_allocations.store(0, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let size = layout.size();
            self.total_allocated.fetch_add(size, Ordering::Relaxed);
            let current = self.current_usage.fetch_add(size, Ordering::Relaxed) + size;
            self.active_allocations.fetch_add(1, Ordering::Relaxed);

            // Update peak usage
            let mut peak = self.peak_usage.load(Ordering::Relaxed);
            while current > peak {
                match self.peak_usage.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => peak = actual,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        let size = layout.size();
        self.total_deallocated.fetch_add(size, Ordering::Relaxed);
        self.current_usage.fetch_sub(size, Ordering::Relaxed);
        self.active_allocations.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Memory pool for efficient tensor allocation
pub struct MemoryPool<T> {
    pools: RwLock<HashMap<usize, Vec<Box<[T]>>>>,
    max_pool_size: usize,
    total_pooled: AtomicUsize,
}

impl<T: Default + Clone> MemoryPool<T> {
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: RwLock::new(HashMap::new()),
            max_pool_size,
            total_pooled: AtomicUsize::new(0),
        }
    }

    pub fn get_buffer(&self, size: usize) -> Box<[T]> {
        {
            let mut pools = self.pools.write();
            if let Some(pool) = pools.get_mut(&size) {
                if let Some(buffer) = pool.pop() {
                    self.total_pooled.fetch_sub(1, Ordering::Relaxed);
                    return buffer;
                }
            }
        }

        // Create new buffer
        vec![T::default(); size].into_boxed_slice()
    }

    pub fn return_buffer(&self, mut buffer: Box<[T]>) {
        let size = buffer.len();

        // Clear the buffer
        for item in buffer.iter_mut() {
            *item = T::default();
        }

        let mut pools = self.pools.write();
        let pool = pools.entry(size).or_insert_with(Vec::new);

        if pool.len() < self.max_pool_size {
            pool.push(buffer);
            self.total_pooled.fetch_add(1, Ordering::Relaxed);
        }
        // Otherwise, let the buffer be dropped
    }

    pub fn clear(&self) {
        let mut pools = self.pools.write();
        pools.clear();
        self.total_pooled.store(0, Ordering::Relaxed);
    }

    pub fn get_stats(&self) -> (usize, usize) {
        let pools = self.pools.read();
        let total_buffers = pools.values().map(|v| v.len()).sum::<usize>();
        let unique_sizes = pools.len();
        (total_buffers, unique_sizes)
    }
}

/// Global memory pools for different data types
pub struct GlobalMemoryManager {
    f32_pool: MemoryPool<f32>,
    i32_pool: MemoryPool<i32>,
    f64_pool: MemoryPool<f64>,
    tracking_allocator: TrackingAllocator,
}

impl GlobalMemoryManager {
    pub fn new() -> Self {
        Self {
            f32_pool: MemoryPool::new(100), // Keep up to 100 buffers per size
            i32_pool: MemoryPool::new(50),
            f64_pool: MemoryPool::new(50),
            tracking_allocator: TrackingAllocator::new(),
        }
    }

    pub fn get_f32_buffer(&self, size: usize) -> Box<[f32]> {
        self.f32_pool.get_buffer(size)
    }

    pub fn return_f32_buffer(&self, buffer: Box<[f32]>) {
        self.f32_pool.return_buffer(buffer);
    }

    pub fn get_i32_buffer(&self, size: usize) -> Box<[i32]> {
        self.i32_pool.get_buffer(size)
    }

    pub fn return_i32_buffer(&self, buffer: Box<[i32]>) {
        self.i32_pool.return_buffer(buffer);
    }

    pub fn get_f64_buffer(&self, size: usize) -> Box<[f64]> {
        self.f64_pool.get_buffer(size)
    }

    pub fn return_f64_buffer(&self, buffer: Box<[f64]>) {
        self.f64_pool.return_buffer(buffer);
    }

    pub fn get_memory_stats(&self) -> MemoryStats {
        self.tracking_allocator.get_stats()
    }

    pub fn clear_all_pools(&self) {
        self.f32_pool.clear();
        self.i32_pool.clear();
        self.f64_pool.clear();
    }

    pub fn get_pool_stats(&self) -> (usize, usize, usize) {
        let (f32_buffers, f32_sizes) = self.f32_pool.get_stats();
        let (i32_buffers, i32_sizes) = self.i32_pool.get_stats();
        let (f64_buffers, f64_sizes) = self.f64_pool.get_stats();

        (
            f32_buffers + i32_buffers + f64_buffers,
            f32_sizes + i32_sizes + f64_sizes,
            0, // Reserved for future use
        )
    }
}

// Global instance
lazy_static::lazy_static! {
    pub static ref GLOBAL_MEMORY_MANAGER: GlobalMemoryManager = GlobalMemoryManager::new();
}

/// Memory-efficient string operations
pub struct StringOptimizer;

impl StringOptimizer {
    /// Intern strings to reduce memory usage
    pub fn intern_string(s: &str) -> &'static str {
        use std::collections::HashSet;
        use std::sync::Mutex;

        lazy_static::lazy_static! {
            static ref INTERNED_STRINGS: Mutex<HashSet<&'static str>> = Mutex::new(HashSet::new());
        }

        let mut set = INTERNED_STRINGS.lock().unwrap();
        if let Some(&interned) = set.get(s) {
            interned
        } else {
            let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
            set.insert(leaked);
            leaked
        }
    }

    /// Efficient string concatenation using a buffer
    pub fn concat_strings(strings: &[&str]) -> String {
        let total_len: usize = strings.iter().map(|s| s.len()).sum();
        let mut result = String::with_capacity(total_len);
        for s in strings {
            result.push_str(s);
        }
        result
    }

    /// Format strings without heap allocation for small strings
    pub fn format_small(template: &str, replacements: &[(&str, &str)]) -> String {
        let mut result = template.to_string();
        for (pattern, replacement) in replacements {
            result = result.replace(pattern, replacement);
        }
        result
    }
}

/// RAII wrapper for automatic buffer return
pub struct BufferGuard<T, F>
where
    F: FnOnce(Box<[T]>),
{
    buffer: Option<Box<[T]>>,
    return_fn: Option<F>,
}

impl<T, F> BufferGuard<T, F>
where
    F: FnOnce(Box<[T]>),
{
    pub fn new(buffer: Box<[T]>, return_fn: F) -> Self {
        Self {
            buffer: Some(buffer),
            return_fn: Some(return_fn),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.buffer.as_ref().unwrap()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buffer.as_mut().unwrap()
    }
}

impl<T, F> Drop for BufferGuard<T, F>
where
    F: FnOnce(Box<[T]>),
{
    fn drop(&mut self) {
        if let (Some(buffer), Some(return_fn)) = (self.buffer.take(), self.return_fn.take()) {
            return_fn(buffer);
        }
    }
}

/// Memory monitoring utilities
pub struct MemoryMonitor;

impl MemoryMonitor {
    /// Get current process memory usage
    pub fn get_process_memory_usage() -> TrustformersPyResult<usize> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status")
                .map_err(|e| TrustformersPyError::IoError(e))?;

            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: usize = parts[1].parse().map_err(|_| {
                            TrustformersPyError::InvalidInputError {
                                message: "Failed to parse memory usage".to_string(),
                            }
                        })?;
                        return Ok(kb * 1024); // Convert to bytes
                    }
                }
            }
            Err(TrustformersPyError::InvalidInputError {
                message: "Memory usage not found".to_string(),
            })
        }
        #[cfg(target_os = "macos")]
        {
            // Simplified approach for macOS - use a basic estimate
            // Real implementation would require additional system dependencies
            let stats = GLOBAL_MEMORY_MANAGER.get_memory_stats();
            Ok(stats.current_usage)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Err(TrustformersPyError::InvalidInputError {
                message: "Memory monitoring not supported on this platform".to_string(),
            })
        }
    }

    /// Get memory pressure level (0-100)
    pub fn get_memory_pressure() -> f32 {
        let stats = GLOBAL_MEMORY_MANAGER.get_memory_stats();
        if stats.peak_usage == 0 {
            return 0.0;
        }
        (stats.current_usage as f32 / stats.peak_usage as f32) * 100.0
    }

    /// Trigger garbage collection if memory pressure is high
    pub fn maybe_gc() {
        if Self::get_memory_pressure() > 80.0 {
            GLOBAL_MEMORY_MANAGER.clear_all_pools();
            // Trigger Python GC if available
            #[cfg(feature = "python-gc")]
            {
                use pyo3::Python;
                if let Ok(py) = Python::acquire_gil() {
                    let py = py.python();
                    if let Ok(gc) = py.import("gc") {
                        let _ = gc.call_method0("collect");
                    }
                }
            }
        }
    }
}

/// Advanced memory pressure detector and auto-cleanup system
pub struct MemoryPressureManager {
    pressure_threshold: f64, // Percentage of system memory
    cleanup_callbacks: Vec<Box<dyn Fn() + Send + Sync>>,
    last_cleanup: std::time::Instant,
    cleanup_cooldown: std::time::Duration,
}

impl MemoryPressureManager {
    pub fn new(pressure_threshold: f64) -> Self {
        Self {
            pressure_threshold,
            cleanup_callbacks: Vec::new(),
            last_cleanup: std::time::Instant::now(),
            cleanup_cooldown: std::time::Duration::from_secs(30), // 30 second cooldown
        }
    }

    /// Add a cleanup callback that will be called during memory pressure
    pub fn add_cleanup_callback<F>(&mut self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.cleanup_callbacks.push(Box::new(callback));
    }

    /// Check if we're under memory pressure and trigger cleanup if needed
    pub fn check_and_cleanup(&mut self) -> TrustformersPyResult<bool> {
        if self.last_cleanup.elapsed() < self.cleanup_cooldown {
            return Ok(false);
        }

        let pressure = self.get_memory_pressure()?;
        if pressure > self.pressure_threshold {
            self.trigger_cleanup();
            self.last_cleanup = std::time::Instant::now();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get current memory pressure as a percentage (0.0 to 1.0)
    fn get_memory_pressure(&self) -> TrustformersPyResult<f64> {
        let _used_memory = MemoryMonitor::get_process_memory_usage()?;

        // Get available system memory (simplified approach)
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let meminfo =
                fs::read_to_string("/proc/meminfo").map_err(|e| TrustformersPyError::IoError(e))?;

            let mut total_kb = 0;
            let mut available_kb = 0;

            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        total_kb = parts[1].parse().unwrap_or(0);
                    }
                } else if line.starts_with("MemAvailable:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        available_kb = parts[1].parse().unwrap_or(0);
                    }
                }
            }

            if total_kb > 0 {
                let used_kb = total_kb - available_kb;
                let pressure = used_kb as f64 / total_kb as f64;
                Ok(pressure)
            } else {
                Ok(0.5) // Fallback to moderate pressure
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            // Simplified estimation for other platforms
            let stats = GLOBAL_MEMORY_MANAGER.get_memory_stats();
            if stats.peak_usage > 0 {
                let pressure = stats.current_usage as f64 / stats.peak_usage as f64;
                Ok(pressure.min(1.0))
            } else {
                Ok(0.0)
            }
        }
    }

    /// Trigger all registered cleanup callbacks
    fn trigger_cleanup(&self) {
        for callback in &self.cleanup_callbacks {
            callback();
        }
    }
}

/// Smart cache with automatic eviction based on memory pressure
pub struct SmartCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    cache: RwLock<HashMap<K, (V, std::time::Instant)>>,
    max_size: usize,
    ttl: std::time::Duration,
    eviction_strategy: EvictionStrategy,
}

#[derive(Clone)]
pub enum EvictionStrategy {
    LRU,  // Least Recently Used
    LFU,  // Least Frequently Used
    TTL,  // Time To Live based
    Size, // Largest items first
}

impl<K, V> SmartCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(max_size: usize, ttl: std::time::Duration, strategy: EvictionStrategy) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_size,
            ttl,
            eviction_strategy: strategy,
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write();
        if let Some((value, timestamp)) = cache.get_mut(key) {
            if timestamp.elapsed() < self.ttl {
                *timestamp = std::time::Instant::now(); // Update access time for LRU
                Some(value.clone())
            } else {
                cache.remove(key);
                None
            }
        } else {
            None
        }
    }

    pub fn insert(&self, key: K, value: V) {
        let mut cache = self.cache.write();

        // Check if we need to evict items
        if cache.len() >= self.max_size {
            self.evict_items(&mut cache);
        }

        cache.insert(key, (value, std::time::Instant::now()));
    }

    fn evict_items(&self, cache: &mut HashMap<K, (V, std::time::Instant)>) {
        let items_to_remove = (cache.len() / 4).max(1); // Remove 25% or at least 1 item

        match self.eviction_strategy {
            EvictionStrategy::LRU => {
                // Remove least recently used items
                let mut items: Vec<_> =
                    cache.iter().map(|(k, (_, timestamp))| (k.clone(), *timestamp)).collect();
                items.sort_by_key(|(_, timestamp)| *timestamp);

                for i in 0..items_to_remove.min(items.len()) {
                    cache.remove(&items[i].0);
                }
            },
            EvictionStrategy::TTL => {
                // Remove expired items first
                let now = std::time::Instant::now();
                cache.retain(|_, (_, timestamp)| now.duration_since(*timestamp) < self.ttl);

                // If still too many items, remove oldest
                if cache.len() >= self.max_size {
                    let mut items: Vec<_> =
                        cache.iter().map(|(k, (_, timestamp))| (k.clone(), *timestamp)).collect();
                    items.sort_by_key(|(_, timestamp)| *timestamp);

                    for i in 0..items_to_remove.min(items.len()) {
                        cache.remove(&items[i].0);
                    }
                }
            },
            _ => {
                // Simple random eviction for other strategies (can be enhanced)
                let keys: Vec<K> = cache.keys().take(items_to_remove).cloned().collect();
                for key in keys {
                    cache.remove(&key);
                }
            },
        }
    }

    pub fn clear(&self) {
        self.cache.write().clear();
    }

    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }
}

/// Global memory pressure manager instance
static mut GLOBAL_PRESSURE_MANAGER: Option<MemoryPressureManager> = None;
static PRESSURE_MANAGER_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize global memory pressure management
pub fn init_memory_pressure_management(threshold: f64) {
    PRESSURE_MANAGER_INIT.call_once(|| unsafe {
        GLOBAL_PRESSURE_MANAGER = Some(MemoryPressureManager::new(threshold));
    });
}

/// Get global memory pressure manager
pub fn get_pressure_manager() -> Option<&'static mut MemoryPressureManager> {
    unsafe { GLOBAL_PRESSURE_MANAGER.as_mut() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::<f32>::new(10);

        // Get and return buffers
        let buffer1 = pool.get_buffer(100);
        assert_eq!(buffer1.len(), 100);

        let buffer2 = pool.get_buffer(100);
        assert_eq!(buffer2.len(), 100);

        pool.return_buffer(buffer1);
        pool.return_buffer(buffer2);

        let (total_buffers, unique_sizes) = pool.get_stats();
        assert_eq!(total_buffers, 2);
        assert_eq!(unique_sizes, 1);
    }

    #[test]
    fn test_string_optimizer() {
        let interned1 = StringOptimizer::intern_string("test");
        let interned2 = StringOptimizer::intern_string("test");
        assert_eq!(interned1.as_ptr(), interned2.as_ptr());

        let concatenated = StringOptimizer::concat_strings(&["hello", " ", "world"]);
        assert_eq!(concatenated, "hello world");
    }

    #[test]
    fn test_buffer_guard() {
        let pool = MemoryPool::<i32>::new(5);
        let buffer = pool.get_buffer(50);

        {
            let guard = BufferGuard::new(buffer, |b| pool.return_buffer(b));
            assert_eq!(guard.as_slice().len(), 50);
        } // Buffer automatically returned here

        let (total_buffers, _) = pool.get_stats();
        assert_eq!(total_buffers, 1);
    }
}
