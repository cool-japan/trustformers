use serde::{Deserialize, Serialize};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::{Arc, Mutex, RwLock};
use trustformers_core::errors::{Result, TrustformersError};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Configuration for zero-copy operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroCopyConfig {
    pub enable_memory_mapping: bool,
    pub enable_view_sharing: bool,
    pub enable_in_place_operations: bool,
    pub enable_lazy_evaluation: bool,
    pub enable_memory_pool: bool,
    pub pool_size_mb: usize,
    pub alignment_bytes: usize,
    pub use_hugepages: bool,
    pub prefault_pages: bool,
    pub track_allocations: bool,
    pub enable_simd_ops: bool,
    pub enable_gpu_zero_copy: bool,
    pub enable_numa_awareness: bool,
    pub enable_cache_prefetch: bool,
    pub enable_async_operations: bool,
    pub cache_line_size: usize,
    pub numa_node_preference: Option<usize>,
    pub gpu_memory_pool_size: Option<usize>,
    pub memory_map_threshold: usize,
    pub async_operation_queue_size: usize,
}

impl Default for ZeroCopyConfig {
    fn default() -> Self {
        Self {
            enable_memory_mapping: true,
            enable_view_sharing: true,
            enable_in_place_operations: true,
            enable_lazy_evaluation: true,
            enable_memory_pool: true,
            pool_size_mb: 1024,   // 1GB default pool
            alignment_bytes: 64,  // Cache line alignment
            use_hugepages: false, // Requires elevated permissions
            prefault_pages: true,
            track_allocations: true,
            enable_simd_ops: true,
            enable_gpu_zero_copy: false, // Requires GPU support
            enable_numa_awareness: true,
            enable_cache_prefetch: true,
            enable_async_operations: true,
            cache_line_size: 64,
            numa_node_preference: None,
            gpu_memory_pool_size: Some(1024 * 1024 * 1024), // 1GB
            memory_map_threshold: 64 * 1024 * 1024,         // 64MB
            async_operation_queue_size: 1000,
        }
    }
}

/// Memory alignment utilities
pub fn align_to(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

pub fn is_aligned(ptr: *const u8, alignment: usize) -> bool {
    (ptr as usize) % alignment == 0
}

/// Zero-copy tensor view that doesn't own the underlying data
#[derive(Debug, Clone)]
pub struct ZeroCopyView<T>
where
    T: Clone,
{
    data: NonNull<T>,
    len: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    _marker: PhantomData<T>,
}

impl<T: Clone> ZeroCopyView<T> {
    /// Create a new view from raw pointer
    ///
    /// # Safety
    /// Caller must ensure the pointer is valid for the given length and lifetime
    pub unsafe fn from_raw_parts(
        data: *mut T,
        len: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
    ) -> Result<Self> {
        if data.is_null() {
            return Err(TrustformersError::invalid_input(                "Null pointer provided for zero-copy view (expected valid_pointer, got null_pointer)".to_string()
            ));
        }

        Ok(Self {
            data: NonNull::new_unchecked(data),
            len,
            shape,
            strides,
            offset,
            _marker: PhantomData,
        })
    }

    /// Create a view from a slice
    pub fn from_slice(slice: &[T], shape: Vec<usize>) -> Result<Self> {
        let strides = Self::compute_strides(&shape);
        let total_elements: usize = shape.iter().product();

        if slice.len() != total_elements {
            return Err(TrustformersError::invalid_input_simple(format!(
                "Slice length {} doesn't match shape {:?} (expected length matching shape, got {})",
                slice.len(),
                shape,
                slice.len()
            )));
        }

        unsafe { Self::from_raw_parts(slice.as_ptr() as *mut T, slice.len(), shape, strides, 0) }
    }

    /// Create a mutable view from a mutable slice
    pub fn from_slice_mut(slice: &mut [T], shape: Vec<usize>) -> Result<Self> {
        let strides = Self::compute_strides(&shape);
        let total_elements: usize = shape.iter().product();

        if slice.len() != total_elements {
            return Err(TrustformersError::invalid_input_simple(format!(
                "Slice length {} doesn't match shape {:?} (expected length matching shape, got {})",
                slice.len(),
                shape,
                slice.len()
            )));
        }

        unsafe { Self::from_raw_parts(slice.as_mut_ptr(), slice.len(), shape, strides, 0) }
    }

    /// Create a subview (slice) of this view
    pub fn subview(&self, ranges: &[std::ops::Range<usize>]) -> Result<Self> {
        if ranges.len() != self.shape.len() {
            return Err(TrustformersError::invalid_input(format!(
                "Number of ranges must match tensor dimensions (expected {}, got {})",
                self.shape.len(),
                ranges.len()
            )));
        }

        let mut new_shape = Vec::new();
        let mut new_offset = self.offset;

        for (i, range) in ranges.iter().enumerate() {
            if range.end > self.shape[i] {
                return Err(TrustformersError::invalid_input_simple(format!(
                    "Range end {} exceeds dimension size {} (expected within bounds, got {})",
                    range.end, self.shape[i], range.end
                )));
            }

            new_shape.push(range.end - range.start);
            new_offset += range.start * self.strides[i];
        }

        unsafe {
            Self::from_raw_parts(
                self.data.as_ptr(),
                self.len, // Keep original length for safety
                new_shape,
                self.strides.clone(),
                new_offset,
            )
        }
    }

    /// Create a transposed view
    pub fn transpose(&self, axes: &[usize]) -> Result<Self> {
        if axes.len() != self.shape.len() {
            return Err(TrustformersError::invalid_input(format!(
                "Axes length must match number of dimensions (expected {}, got {})",
                self.shape.len(),
                axes.len()
            )));
        }

        let mut new_shape = vec![0; self.shape.len()];
        let mut new_strides = vec![0; self.strides.len()];

        for (i, &axis) in axes.iter().enumerate() {
            if axis >= self.shape.len() {
                return Err(TrustformersError::invalid_input_simple(format!(
                    "Axis {} out of bounds (expected valid axis index, got {})",
                    axis, axis
                )));
            }
            new_shape[i] = self.shape[axis];
            new_strides[i] = self.strides[axis];
        }

        unsafe {
            Self::from_raw_parts(
                self.data.as_ptr(),
                self.len,
                new_shape,
                new_strides,
                self.offset,
            )
        }
    }

    /// Reshape the view (zero-copy if possible)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let total_elements: usize = new_shape.iter().product();
        let current_elements: usize = self.shape.iter().product();

        if total_elements != current_elements {
            return Err(TrustformersError::invalid_input(format!(
                "New shape must have same number of elements (expected {}, got {})",
                self.len,
                new_shape.iter().product::<usize>()
            )));
        }

        // Check if reshape is possible without copying (contiguous memory)
        if !self.is_contiguous() {
            return Err(TrustformersError::invalid_input(                "Cannot reshape non-contiguous view without copying (expected contiguous status: true, got false)".to_string()
            ));
        }

        let new_strides = Self::compute_strides(&new_shape);

        unsafe {
            Self::from_raw_parts(
                self.data.as_ptr(),
                self.len,
                new_shape,
                new_strides,
                self.offset,
            )
        }
    }

    /// Check if the view represents contiguous memory
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = Self::compute_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Get a raw slice of the data (if contiguous)
    pub fn as_slice(&self) -> Result<&[T]> {
        if !self.is_contiguous() {
            return Err(TrustformersError::invalid_input(                "Cannot get slice from non-contiguous view (expected contiguous status: true, got false)".to_string()
            ));
        }

        unsafe {
            let ptr = self.data.as_ptr().add(self.offset);
            let elements: usize = self.shape.iter().product();
            Ok(slice::from_raw_parts(ptr, elements))
        }
    }

    /// Get a mutable raw slice of the data (if contiguous)
    pub fn as_slice_mut(&mut self) -> Result<&mut [T]> {
        if !self.is_contiguous() {
            return Err(TrustformersError::invalid_input(                "Cannot get mutable slice from non-contiguous view (expected contiguous status: true, got false)".to_string()
            ));
        }

        unsafe {
            let ptr = self.data.as_ptr().add(self.offset);
            let elements: usize = self.shape.iter().product();
            Ok(slice::from_raw_parts_mut(ptr, elements))
        }
    }

    /// Get the shape of the view
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of the view
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the view is empty
    pub fn is_empty(&self) -> bool {
        self.shape.contains(&0)
    }

    /// Compute strides for C-contiguous layout
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Get element at multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> Result<&T> {
        if indices.len() != self.shape.len() {
            return Err(TrustformersError::invalid_input(format!(
                "Number of indices must match dimensions (expected {}, got {})",
                self.shape.len(),
                indices.len()
            )));
        }

        let mut offset = self.offset;
        for (i, (&index, &stride)) in indices.iter().zip(self.strides.iter()).enumerate() {
            if index >= self.shape[i] {
                return Err(TrustformersError::invalid_input_simple(format!(
                    "Index {} out of bounds for dimension {} (expected valid index, got {})",
                    index, i, index
                )));
            }
            offset += index * stride;
        }

        unsafe { Ok(&*self.data.as_ptr().add(offset)) }
    }

    /// Get mutable element at multi-dimensional index
    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T> {
        if indices.len() != self.shape.len() {
            return Err(TrustformersError::invalid_input(format!(
                "Number of indices must match dimensions (expected {}, got {})",
                self.shape.len(),
                indices.len()
            )));
        }

        let mut offset = self.offset;
        for (i, (&index, &stride)) in indices.iter().zip(self.strides.iter()).enumerate() {
            if index >= self.shape[i] {
                return Err(TrustformersError::invalid_input_simple(format!(
                    "Index {} out of bounds for dimension {} (expected valid index, got {})",
                    index, i, index
                )));
            }
            offset += index * stride;
        }

        unsafe { Ok(&mut *self.data.as_ptr().add(offset)) }
    }
}

unsafe impl<T: Send + Clone> Send for ZeroCopyView<T> {}
unsafe impl<T: Sync + Clone> Sync for ZeroCopyView<T> {}

/// Memory pool for zero-copy allocations
pub struct MemoryPool {
    config: ZeroCopyConfig,
    free_blocks: RwLock<HashMap<usize, Vec<NonNull<u8>>>>, // size -> list of blocks
    allocated_blocks: Mutex<HashMap<*mut u8, usize>>,      // ptr -> size
    total_allocated: Mutex<usize>,
    allocation_stats: Mutex<AllocationStats>,
}

// SAFETY: MemoryPool uses internal synchronization (RwLock and Mutex) to protect
// all access to raw pointers, making it safe to share across threads.
unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

#[derive(Debug, Default, Clone)]
struct AllocationStats {
    total_allocations: u64,
    total_deallocations: u64,
    peak_memory_usage: usize,
    current_memory_usage: usize,
    allocation_failures: u64,
}

impl MemoryPool {
    pub fn new(config: ZeroCopyConfig) -> Self {
        Self {
            config,
            free_blocks: RwLock::new(HashMap::new()),
            allocated_blocks: Mutex::new(HashMap::new()),
            total_allocated: Mutex::new(0),
            allocation_stats: Mutex::new(AllocationStats::default()),
        }
    }

    /// Allocate aligned memory from the pool
    pub fn allocate(&self, size: usize) -> Result<NonNull<u8>> {
        let aligned_size = align_to(size, self.config.alignment_bytes);

        // Try to reuse existing block
        if let Some(ptr) = self.try_reuse_block(aligned_size) {
            self.record_allocation(ptr.as_ptr(), aligned_size);
            return Ok(ptr);
        }

        // Allocate new block
        let layout = Layout::from_size_align(aligned_size, self.config.alignment_bytes)
            .map_err(|_| TrustformersError::runtime_error("Invalid memory layout".to_string()))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            self.record_allocation_failure();
            return Err(TrustformersError::runtime_error(
                "Memory allocation failed".to_string(),
            ));
        }

        let non_null_ptr = unsafe { NonNull::new_unchecked(ptr) };
        self.record_allocation(ptr, aligned_size);

        // Prefault pages if requested
        if self.config.prefault_pages {
            self.prefault_memory(ptr, aligned_size);
        }

        Ok(non_null_ptr)
    }

    /// Deallocate memory back to the pool
    pub fn deallocate(&self, ptr: NonNull<u8>) {
        let ptr_raw = ptr.as_ptr();

        let size = {
            let mut allocated = self.allocated_blocks.lock().expect("lock should not be poisoned");
            if let Some(size) = allocated.remove(&ptr_raw) {
                size
            } else {
                // Pointer not tracked, can't safely deallocate
                return;
            }
        };

        // Return to free blocks for reuse
        {
            let mut free_blocks = self.free_blocks.write().expect("lock should not be poisoned");
            free_blocks.entry(size).or_default().push(ptr);
        }

        self.record_deallocation(size);
    }

    /// Allocate a zero-copy view for a specific type
    pub fn allocate_view<T: Clone>(&self, shape: Vec<usize>) -> Result<ZeroCopyView<T>> {
        let elements: usize = shape.iter().product();
        let size = elements * std::mem::size_of::<T>();
        let ptr = self.allocate(size)?;

        unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            let strides = ZeroCopyView::<T>::compute_strides(&shape);
            ZeroCopyView::from_raw_parts(typed_ptr, elements, shape, strides, 0)
        }
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> AllocationStats {
        self.allocation_stats.lock().expect("lock should not be poisoned").clone()
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.allocation_stats
            .lock()
            .expect("lock should not be poisoned")
            .current_memory_usage
    }

    /// Clear unused blocks to free memory
    pub fn clear_unused(&self) {
        let mut free_blocks = self.free_blocks.write().expect("lock should not be poisoned");

        for (size, blocks) in free_blocks.drain() {
            for block in blocks {
                unsafe {
                    let layout =
                        Layout::from_size_align_unchecked(size, self.config.alignment_bytes);
                    dealloc(block.as_ptr(), layout);
                }
            }
        }
    }

    fn try_reuse_block(&self, size: usize) -> Option<NonNull<u8>> {
        let mut free_blocks = self.free_blocks.write().expect("lock should not be poisoned");

        // Try exact size first
        if let Some(blocks) = free_blocks.get_mut(&size) {
            if !blocks.is_empty() {
                return blocks.pop();
            }
        }

        // Try larger blocks (simple first-fit)
        for (&block_size, blocks) in free_blocks.iter_mut() {
            if block_size >= size && !blocks.is_empty() {
                return blocks.pop();
            }
        }

        None
    }

    fn record_allocation(&self, ptr: *mut u8, size: usize) {
        {
            let mut allocated = self.allocated_blocks.lock().expect("lock should not be poisoned");
            allocated.insert(ptr, size);
        }

        {
            let mut stats = self.allocation_stats.lock().expect("lock should not be poisoned");
            stats.total_allocations += 1;
            stats.current_memory_usage += size;
            if stats.current_memory_usage > stats.peak_memory_usage {
                stats.peak_memory_usage = stats.current_memory_usage;
            }
        }

        {
            let mut total = self.total_allocated.lock().expect("lock should not be poisoned");
            *total += size;
        }
    }

    fn record_deallocation(&self, size: usize) {
        {
            let mut stats = self.allocation_stats.lock().expect("lock should not be poisoned");
            stats.total_deallocations += 1;
            stats.current_memory_usage = stats.current_memory_usage.saturating_sub(size);
        }

        {
            let mut total = self.total_allocated.lock().expect("lock should not be poisoned");
            *total = total.saturating_sub(size);
        }
    }

    fn record_allocation_failure(&self) {
        let mut stats = self.allocation_stats.lock().expect("lock should not be poisoned");
        stats.allocation_failures += 1;
    }

    fn prefault_memory(&self, ptr: *mut u8, size: usize) {
        unsafe {
            // Touch each page to prefault
            let page_size = 4096; // Typical page size
            let mut offset = 0;
            while offset < size {
                ptr::write_volatile(ptr.add(offset), 0);
                offset += page_size;
            }
        }
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        self.clear_unused();
    }
}

/// Zero-copy tensor operations
pub struct ZeroCopyOps;

impl ZeroCopyOps {
    /// Element-wise addition without copying (in-place)
    pub fn add_inplace<T>(a: &mut ZeroCopyView<T>, b: &ZeroCopyView<T>) -> Result<()>
    where
        T: std::ops::AddAssign + Copy,
    {
        if a.shape() != b.shape() {
            return Err(TrustformersError::invalid_input(format!(
                "Shapes must match for element-wise addition (expected {:?}, got {:?})",
                a.shape(),
                b.shape()
            )));
        }

        if a.is_contiguous() && b.is_contiguous() {
            // Fast path for contiguous arrays
            let a_slice = a.as_slice_mut()?;
            let b_slice = b.as_slice()?;

            for (a_elem, &b_elem) in a_slice.iter_mut().zip(b_slice.iter()) {
                *a_elem += b_elem;
            }
        } else {
            // Slow path for non-contiguous arrays
            Self::add_inplace_strided(a, b)?;
        }

        Ok(())
    }

    fn add_inplace_strided<T>(a: &mut ZeroCopyView<T>, b: &ZeroCopyView<T>) -> Result<()>
    where
        T: std::ops::AddAssign + Copy,
    {
        // For simplicity, implement for 1D and 2D cases
        match a.ndim() {
            1 => {
                for i in 0..a.shape()[0] {
                    let a_val = a.get_mut(&[i])?;
                    let b_val = b.get(&[i])?;
                    *a_val += *b_val;
                }
            },
            2 => {
                for i in 0..a.shape()[0] {
                    for j in 0..a.shape()[1] {
                        let a_val = a.get_mut(&[i, j])?;
                        let b_val = b.get(&[i, j])?;
                        *a_val += *b_val;
                    }
                }
            },
            _ => {
                return Err(TrustformersError::invalid_input(                    format!("Strided operations only implemented for 1D and 2D (expected 1D or 2D, got {}D)",
                            a.ndim())
                ));
            },
        }

        Ok(())
    }

    /// Matrix multiplication using zero-copy views
    pub fn matmul<T>(
        a: &ZeroCopyView<T>,
        b: &ZeroCopyView<T>,
        c: &mut ZeroCopyView<T>,
    ) -> Result<()>
    where
        T: std::ops::Mul<Output = T> + std::ops::AddAssign + Copy + Default,
    {
        if a.ndim() != 2 || b.ndim() != 2 || c.ndim() != 2 {
            return Err(TrustformersError::invalid_input(format!(
                "Matrix multiplication requires 2D tensors (expected 2D tensors, got {}D tensors)",
                std::cmp::max(a.ndim(), std::cmp::max(b.ndim(), c.ndim()))
            )));
        }

        let (m, k) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);

        if k != k2 {
            return Err(TrustformersError::invalid_input(format!(
                "Inner dimensions must match for matrix multiplication (expected {}, got {})",
                k, k2
            )));
        }

        if c.shape()[0] != m || c.shape()[1] != n {
            return Err(TrustformersError::invalid_input(format!(
                "Output matrix has incorrect dimensions (expected {}x{}, got {}x{})",
                m,
                n,
                c.shape()[0],
                c.shape()[1]
            )));
        }

        // Simple implementation - in practice would use optimized BLAS
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for k_idx in 0..k {
                    let a_val = *a.get(&[i, k_idx])?;
                    let b_val = *b.get(&[k_idx, j])?;
                    sum += a_val * b_val;
                }
                *c.get_mut(&[i, j])? = sum;
            }
        }

        Ok(())
    }

    /// Transpose operation (creates new view, no data copy)
    pub fn transpose<T: Clone>(view: &ZeroCopyView<T>) -> Result<ZeroCopyView<T>> {
        if view.ndim() != 2 {
            return Err(TrustformersError::invalid_input(format!(
                "Transpose only implemented for 2D tensors (expected 2D tensor, got {}D tensor)",
                view.ndim()
            )));
        }

        view.transpose(&[1, 0])
    }

    /// Slice operation (creates new view, no data copy)
    pub fn slice<T: Clone>(
        view: &ZeroCopyView<T>,
        ranges: &[std::ops::Range<usize>],
    ) -> Result<ZeroCopyView<T>> {
        view.subview(ranges)
    }

    /// Copy data between views (only when necessary)
    pub fn copy<T>(src: &ZeroCopyView<T>, dst: &mut ZeroCopyView<T>) -> Result<()>
    where
        T: Copy,
    {
        if src.shape() != dst.shape() {
            return Err(TrustformersError::invalid_input(format!(
                "Source and destination must have same shape (expected {:?}, got {:?})",
                src.shape(),
                dst.shape()
            )));
        }

        if src.is_contiguous() && dst.is_contiguous() {
            // Fast memcpy for contiguous data
            let src_slice = src.as_slice()?;
            let dst_slice = dst.as_slice_mut()?;
            dst_slice.copy_from_slice(src_slice);
        } else {
            // Element-wise copy for non-contiguous data
            Self::copy_strided(src, dst)?;
        }

        Ok(())
    }

    fn copy_strided<T>(src: &ZeroCopyView<T>, dst: &mut ZeroCopyView<T>) -> Result<()>
    where
        T: Copy,
    {
        match src.ndim() {
            1 => {
                for i in 0..src.shape()[0] {
                    let src_val = *src.get(&[i])?;
                    *dst.get_mut(&[i])? = src_val;
                }
            },
            2 => {
                for i in 0..src.shape()[0] {
                    for j in 0..src.shape()[1] {
                        let src_val = *src.get(&[i, j])?;
                        *dst.get_mut(&[i, j])? = src_val;
                    }
                }
            },
            _ => {
                return Err(TrustformersError::invalid_input(
                    "Strided copy only implemented for 1D and 2D".to_string(),
                ));
            },
        }

        Ok(())
    }
}

/// Global memory pool instance
static GLOBAL_POOL: std::sync::OnceLock<MemoryPool> = std::sync::OnceLock::new();

/// Initialize the global memory pool
pub fn init_global_pool(config: ZeroCopyConfig) {
    GLOBAL_POOL.set(MemoryPool::new(config)).unwrap_or(());
}

/// Get the global memory pool
pub fn global_pool() -> &'static MemoryPool {
    GLOBAL_POOL.get_or_init(|| MemoryPool::new(ZeroCopyConfig::default()))
}

/// Convenience function to allocate from global pool
pub fn allocate_global<T: Clone>(shape: Vec<usize>) -> Result<ZeroCopyView<T>> {
    global_pool().allocate_view(shape)
}

/// Convenience function to deallocate to global pool
pub fn deallocate_global<T: Clone>(view: ZeroCopyView<T>) {
    // Extract the pointer and deallocate safely
    let ptr = view.data.as_ptr() as *mut u8;
    if let Some(non_null_ptr) = NonNull::new(ptr) {
        global_pool().deallocate(non_null_ptr);
    }
    // If ptr is null, we can't deallocate it, but we don't need to panic
}

/// Zero-copy buffer for efficient data transfer
pub struct ZeroCopyBuffer<T: Clone> {
    view: ZeroCopyView<T>,
    pool: Option<Arc<MemoryPool>>,
}

impl<T: Clone> ZeroCopyBuffer<T> {
    pub fn new(shape: Vec<usize>) -> Result<Self> {
        let view = allocate_global(shape)?;
        Ok(Self { view, pool: None })
    }

    pub fn with_pool(shape: Vec<usize>, pool: Arc<MemoryPool>) -> Result<Self> {
        let view = pool.allocate_view(shape)?;
        Ok(Self {
            view,
            pool: Some(pool),
        })
    }

    pub fn view(&self) -> &ZeroCopyView<T> {
        &self.view
    }

    pub fn view_mut(&mut self) -> &mut ZeroCopyView<T> {
        &mut self.view
    }

    pub fn into_view(self) -> ZeroCopyView<T> {
        self.view.clone()
    }
}

impl<T: Clone> Drop for ZeroCopyBuffer<T> {
    fn drop(&mut self) {
        let ptr = self.view.data.as_ptr() as *mut u8;
        if let Some(non_null_ptr) = NonNull::new(ptr) {
            if let Some(pool) = &self.pool {
                pool.deallocate(non_null_ptr);
            } else {
                global_pool().deallocate(non_null_ptr);
            }
        }
        // If ptr is null, we can't deallocate it, but we don't need to panic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_view_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = ZeroCopyView::from_slice(&data, vec![2, 3]).unwrap();

        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.len(), 6);
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_zero_copy_subview() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = ZeroCopyView::from_slice(&data, vec![2, 3]).unwrap();

        let subview = view.subview(&[0..1, 1..3]).unwrap();
        assert_eq!(subview.shape(), &[1, 2]);

        let first_element = subview.get(&[0, 0]).unwrap();
        assert_eq!(*first_element, 2.0); // data[0 * 3 + 1]
    }

    #[test]
    fn test_zero_copy_transpose() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = ZeroCopyView::from_slice(&data, vec![2, 3]).unwrap();

        let transposed = view.transpose(&[1, 0]).unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);

        // Original: [[1, 2, 3], [4, 5, 6]]
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert_eq!(*transposed.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*transposed.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(*transposed.get(&[1, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_memory_pool() {
        let config = ZeroCopyConfig::default();
        let pool = MemoryPool::new(config);

        let ptr1 = pool.allocate(1024).unwrap();
        let ptr2 = pool.allocate(512).unwrap();

        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());

        let stats = pool.get_stats();
        assert_eq!(stats.total_allocations, 2);
        assert!(stats.current_memory_usage >= 1536);

        pool.deallocate(ptr1);
        pool.deallocate(ptr2);

        let stats = pool.get_stats();
        assert_eq!(stats.total_deallocations, 2);
    }

    #[test]
    fn test_zero_copy_ops() {
        let mut data_a = vec![1.0f32, 2.0, 3.0, 4.0];
        let data_b = vec![1.0f32, 1.0, 1.0, 1.0];

        let mut view_a = ZeroCopyView::from_slice_mut(&mut data_a, vec![4]).unwrap();
        let view_b = ZeroCopyView::from_slice(&data_b, vec![4]).unwrap();

        ZeroCopyOps::add_inplace(&mut view_a, &view_b).unwrap();

        assert_eq!(data_a, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_alignment() {
        assert_eq!(align_to(100, 64), 128);
        assert_eq!(align_to(64, 64), 64);
        assert_eq!(align_to(65, 64), 128);

        let ptr = 0x1000 as *const u8;
        assert!(is_aligned(ptr, 64));

        let ptr = 0x1001 as *const u8;
        assert!(!is_aligned(ptr, 64));
    }
}
