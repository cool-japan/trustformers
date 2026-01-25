//! Mobile Memory Pool Module
//!
//! Provides efficient memory pooling and management for mobile inference,
//! reducing allocation overhead and memory fragmentation.

use std::alloc::{alloc, dealloc, Layout};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

/// Memory allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// First-fit allocation
    FirstFit,
    /// Best-fit allocation
    BestFit,
    /// Buddy system allocation
    BuddySystem,
    /// Size-class based allocation
    SizeClass,
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Maximum memory in bytes
    pub max_memory_bytes: usize,
    /// Allocation strategy
    pub allocation_strategy: AllocationStrategy,
    /// Enable defragmentation
    pub enable_defragmentation: bool,
}

/// Memory allocation handle
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    /// Unique allocation ID
    pub id: usize,
    /// Pointer to allocated memory
    pub ptr: *mut u8,
    /// Size of allocation
    pub size: usize,
    /// Alignment requirement
    pub alignment: usize,
    /// Allocation timestamp
    pub timestamp: std::time::Instant,
}

unsafe impl Send for MemoryAllocation {}
unsafe impl Sync for MemoryAllocation {}

/// Mobile memory pool implementation
#[derive(Debug)]
pub struct MobileMemoryPool {
    config: MemoryPoolConfig,
    allocations: Arc<Mutex<HashMap<usize, MemoryAllocation>>>,
    free_blocks: Arc<Mutex<BTreeMap<usize, Vec<FreeBlock>>>>,
    total_allocated: Arc<Mutex<usize>>,
    peak_allocated: Arc<Mutex<usize>>,
    allocation_counter: Arc<Mutex<usize>>,
    stats: Arc<Mutex<PoolStats>>,
}

/// Free memory block
#[derive(Debug, Clone)]
struct FreeBlock {
    ptr: *mut u8,
    size: usize,
    alignment: usize,
}

unsafe impl Send for FreeBlock {}
unsafe impl Sync for FreeBlock {}

/// Memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub current_allocations: usize,
    pub current_memory_bytes: usize,
    pub peak_memory_bytes: usize,
    pub fragmentation_ratio: f32,
    pub allocation_failures: usize,
    pub defragmentation_count: usize,
}

impl MobileMemoryPool {
    /// Create new memory pool
    pub fn new(config: MemoryPoolConfig) -> Result<Self> {
        if config.max_memory_bytes == 0 {
            return Err(
                TrustformersError::config_error("Memory pool size must be > 0", "new").into(),
            );
        }

        let pool = Self {
            config,
            allocations: Arc::new(Mutex::new(HashMap::new())),
            free_blocks: Arc::new(Mutex::new(BTreeMap::new())),
            total_allocated: Arc::new(Mutex::new(0)),
            peak_allocated: Arc::new(Mutex::new(0)),
            allocation_counter: Arc::new(Mutex::new(0)),
            stats: Arc::new(Mutex::new(PoolStats::default())),
        };

        // Pre-allocate some common sizes
        pool.preallocate_common_sizes()?;

        Ok(pool)
    }

    /// Allocate memory from pool
    pub fn allocate(&self, size: usize, alignment: usize) -> Result<MemoryAllocation> {
        let size = self.align_size(size, alignment);

        // Check if allocation would exceed limit
        {
            let total = self.total_allocated.lock().expect("Lock poisoned");
            if *total + size > self.config.max_memory_bytes {
                let mut stats = self.stats.lock().expect("Lock poisoned");
                stats.allocation_failures += 1;
                return Err(TrustformersError::hardware_error(
                    &format!(
                        "Memory pool exhausted: requested {}, available {}",
                        size,
                        self.config.max_memory_bytes - *total
                    ),
                    "allocate",
                )
                .into());
            }
        }

        // Try to find a suitable free block
        if let Some(allocation) = self.try_allocate_from_free(size, alignment)? {
            return Ok(allocation);
        }

        // Allocate new memory
        self.allocate_new(size, alignment)
    }

    /// Deallocate memory back to pool
    pub fn deallocate(&self, allocation: MemoryAllocation) -> Result<()> {
        // Remove from active allocations
        {
            let mut allocations = self.allocations.lock().expect("Lock poisoned");
            allocations.remove(&allocation.id);
        }

        // Add to free blocks
        {
            let mut free_blocks = self.free_blocks.lock().expect("Lock poisoned");
            let blocks = free_blocks.entry(allocation.size).or_default();
            blocks.push(FreeBlock {
                ptr: allocation.ptr,
                size: allocation.size,
                alignment: allocation.alignment,
            });
        }

        // Update statistics
        {
            let mut total = self.total_allocated.lock().expect("Lock poisoned");
            *total -= allocation.size;

            let mut stats = self.stats.lock().expect("Lock poisoned");
            stats.total_deallocations += 1;
            stats.current_allocations -= 1;
            stats.current_memory_bytes = *total;
        }

        // Trigger defragmentation if needed
        if self.config.enable_defragmentation {
            self.maybe_defragment()?;
        }

        Ok(())
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        self.stats.lock().expect("Lock poisoned").clone()
    }

    /// Get memory usage ratio (0.0 to 1.0)
    pub fn get_usage_ratio(&self) -> f32 {
        let total_allocated = *self.total_allocated.lock().expect("Lock poisoned");
        total_allocated as f32 / self.config.max_memory_bytes as f32
    }

    /// Get available memory in bytes
    pub fn get_available_memory(&self) -> usize {
        let total_allocated = *self.total_allocated.lock().expect("Lock poisoned");
        self.config.max_memory_bytes - total_allocated
    }

    /// Get peak memory usage in bytes
    pub fn get_peak_usage(&self) -> usize {
        *self.peak_allocated.lock().expect("Lock poisoned")
    }

    /// Clear all allocations (dangerous!)
    pub fn clear(&self) -> Result<()> {
        // Deallocate all active allocations
        let allocations: Vec<_> = {
            let allocs = self.allocations.lock().expect("Lock poisoned");
            allocs.values().cloned().collect()
        };

        for allocation in allocations {
            unsafe {
                let layout = Layout::from_size_align(allocation.size, allocation.alignment)
                    .map_err(|e| TrustformersError::runtime_error(e.to_string()))?;
                dealloc(allocation.ptr, layout);
            }
        }

        // Clear all tracking
        self.allocations.lock().expect("Lock poisoned").clear();
        self.free_blocks.lock().expect("Lock poisoned").clear();
        *self.total_allocated.lock().expect("Lock poisoned") = 0;

        // Reset stats
        let mut stats = self.stats.lock().expect("Lock poisoned");
        stats.current_allocations = 0;
        stats.current_memory_bytes = 0;

        Ok(())
    }

    // Private helper methods

    fn preallocate_common_sizes(&self) -> Result<()> {
        // Pre-allocate some common tensor sizes for mobile
        let common_sizes = [
            1024,    // 1KB
            4096,    // 4KB
            16384,   // 16KB
            65536,   // 64KB
            262144,  // 256KB
            1048576, // 1MB
        ];

        for &size in &common_sizes {
            // Pre-allocate 2 blocks of each size
            for _ in 0..2 {
                if let Ok(alloc) = self.allocate_new_for_preallocation(size, 64) {
                    // Immediately add to free list
                    let mut free_blocks = self.free_blocks.lock().expect("Lock poisoned");
                    let blocks = free_blocks.entry(size).or_default();
                    blocks.push(FreeBlock {
                        ptr: alloc.ptr,
                        size: alloc.size,
                        alignment: alloc.alignment,
                    });
                }
            }
        }

        Ok(())
    }

    fn try_allocate_from_free(
        &self,
        size: usize,
        alignment: usize,
    ) -> Result<Option<MemoryAllocation>> {
        let mut free_blocks = self.free_blocks.lock().expect("Lock poisoned");

        match self.config.allocation_strategy {
            AllocationStrategy::FirstFit => {
                // Find first block that fits
                for (&block_size, blocks) in free_blocks.iter_mut() {
                    if block_size >= size && !blocks.is_empty() {
                        if let Some(block) = blocks.iter().position(|b| b.alignment >= alignment) {
                            let free_block = blocks.remove(block);
                            return Ok(Some(self.create_allocation_from_block(free_block)?));
                        }
                    }
                }
            },
            AllocationStrategy::BestFit => {
                // Find smallest block that fits
                let mut best_size = usize::MAX;
                let mut best_key = None;

                for (&block_size, blocks) in free_blocks.iter() {
                    if block_size >= size
                        && block_size < best_size
                        && !blocks.is_empty()
                        && blocks.iter().any(|b| b.alignment >= alignment)
                    {
                        best_size = block_size;
                        best_key = Some(block_size);
                    }
                }

                if let Some(key) = best_key {
                    if let Some(blocks) = free_blocks.get_mut(&key) {
                        if let Some(idx) = blocks.iter().position(|b| b.alignment >= alignment) {
                            let block = blocks.remove(idx);
                            return Ok(Some(self.create_allocation_from_block(block)?));
                        }
                    }
                }
            },
            AllocationStrategy::BuddySystem => {
                // Find power-of-2 sized block
                let buddy_size = size.next_power_of_two();
                if let Some(blocks) = free_blocks.get_mut(&buddy_size) {
                    if let Some(idx) = blocks.iter().position(|b| b.alignment >= alignment) {
                        let block = blocks.remove(idx);
                        return Ok(Some(self.create_allocation_from_block(block)?));
                    }
                }
            },
            AllocationStrategy::SizeClass => {
                // Use size classes (small, medium, large)
                let size_class = self.get_size_class(size);
                if let Some(blocks) = free_blocks.get_mut(&size_class) {
                    if let Some(idx) = blocks.iter().position(|b| b.alignment >= alignment) {
                        let block = blocks.remove(idx);
                        return Ok(Some(self.create_allocation_from_block(block)?));
                    }
                }
            },
        }

        Ok(None)
    }

    fn allocate_new(&self, size: usize, alignment: usize) -> Result<MemoryAllocation> {
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| TrustformersError::runtime_error(e.to_string()))?;

        let ptr = unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                return Err(
                    TrustformersError::runtime_error("Failed to allocate memory".into()).into(),
                );
            }
            ptr
        };

        let id = {
            let mut counter = self.allocation_counter.lock().expect("Lock poisoned");
            *counter += 1;
            *counter
        };

        let allocation = MemoryAllocation {
            id,
            ptr,
            size,
            alignment,
            timestamp: std::time::Instant::now(),
        };

        // Track allocation
        {
            let mut allocations = self.allocations.lock().expect("Lock poisoned");
            allocations.insert(id, allocation.clone());
        }

        // Update statistics
        {
            let mut total = self.total_allocated.lock().expect("Lock poisoned");
            *total += size;

            let mut peak = self.peak_allocated.lock().expect("Lock poisoned");
            if *total > *peak {
                *peak = *total;
            }

            let mut stats = self.stats.lock().expect("Lock poisoned");
            stats.total_allocations += 1;
            stats.current_allocations += 1;
            stats.current_memory_bytes = *total;
            stats.peak_memory_bytes = *peak;
        }

        Ok(allocation)
    }

    /// Allocate memory for preallocation (doesn't count toward active allocations)
    fn allocate_new_for_preallocation(
        &self,
        size: usize,
        alignment: usize,
    ) -> Result<MemoryAllocation> {
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| TrustformersError::runtime_error(e.to_string()))?;

        let ptr = unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                return Err(
                    TrustformersError::runtime_error("Failed to allocate memory".into()).into(),
                );
            }
            ptr
        };

        let id = {
            let mut counter = self.allocation_counter.lock().expect("Lock poisoned");
            *counter += 1;
            *counter
        };

        let allocation = MemoryAllocation {
            id,
            ptr,
            size,
            alignment,
            timestamp: std::time::Instant::now(),
        };

        // Don't track in active allocations since this goes directly to free list
        // No stats update for preallocation

        Ok(allocation)
    }

    fn create_allocation_from_block(&self, block: FreeBlock) -> Result<MemoryAllocation> {
        let id = {
            let mut counter = self.allocation_counter.lock().expect("Lock poisoned");
            *counter += 1;
            *counter
        };

        let allocation = MemoryAllocation {
            id,
            ptr: block.ptr,
            size: block.size,
            alignment: block.alignment,
            timestamp: std::time::Instant::now(),
        };

        // Track allocation
        {
            let mut allocations = self.allocations.lock().expect("Lock poisoned");
            allocations.insert(id, allocation.clone());
        }

        // Update statistics
        {
            let mut total = self.total_allocated.lock().expect("Lock poisoned");
            *total += block.size;

            let mut peak = self.peak_allocated.lock().expect("Lock poisoned");
            if *total > *peak {
                *peak = *total;
            }

            let mut stats = self.stats.lock().expect("Lock poisoned");
            stats.total_allocations += 1;
            stats.current_allocations += 1;
            stats.current_memory_bytes = *total;
            stats.peak_memory_bytes = *peak;
        }

        Ok(allocation)
    }

    fn maybe_defragment(&self) -> Result<()> {
        let fragmentation = self.calculate_fragmentation();

        if fragmentation > 0.3 {
            // High fragmentation, trigger defragmentation
            self.defragment()?;
        }

        Ok(())
    }

    fn defragment(&self) -> Result<()> {
        // Simple defragmentation: merge adjacent free blocks
        let mut free_blocks = self.free_blocks.lock().expect("Lock poisoned");

        for (_, blocks) in free_blocks.iter_mut() {
            if blocks.len() > 1 {
                // Sort by pointer address
                blocks.sort_by_key(|b| b.ptr as usize);

                // Merge adjacent blocks
                let mut merged = Vec::new();
                let mut current = blocks[0].clone();

                for block in &blocks[1..] {
                    let current_end = current.ptr as usize + current.size;
                    let block_start = block.ptr as usize;

                    if current_end == block_start && current.alignment == block.alignment {
                        // Adjacent blocks, merge them
                        current.size += block.size;
                    } else {
                        // Non-adjacent, save current and start new
                        merged.push(current);
                        current = block.clone();
                    }
                }

                merged.push(current);
                *blocks = merged;
            }
        }

        // Update stats
        {
            let mut stats = self.stats.lock().expect("Lock poisoned");
            stats.defragmentation_count += 1;
            stats.fragmentation_ratio = self.calculate_fragmentation();
        }

        Ok(())
    }

    fn calculate_fragmentation(&self) -> f32 {
        let free_blocks = self.free_blocks.lock().expect("Lock poisoned");

        let total_free_blocks: usize = free_blocks.values().map(|blocks| blocks.len()).sum();

        let total_free_memory: usize = free_blocks
            .iter()
            .flat_map(|(_, blocks)| blocks.iter())
            .map(|block| block.size)
            .sum();

        if total_free_memory == 0 {
            return 0.0;
        }

        // Fragmentation = 1 - (largest_free_block / total_free_memory)
        let largest_free_block = free_blocks
            .iter()
            .flat_map(|(_, blocks)| blocks.iter())
            .map(|block| block.size)
            .max()
            .unwrap_or(0);

        1.0 - (largest_free_block as f32 / total_free_memory as f32)
    }

    fn align_size(&self, size: usize, alignment: usize) -> usize {
        (size + alignment - 1) & !(alignment - 1)
    }

    fn get_size_class(&self, size: usize) -> usize {
        // Define size classes for mobile
        match size {
            0..=1024 => 1024,         // 1KB
            1025..=16384 => 16384,    // 16KB
            16385..=262144 => 262144, // 256KB
            _ => 1048576,             // 1MB
        }
    }
}

impl Clone for MobileMemoryPool {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            allocations: Arc::clone(&self.allocations),
            free_blocks: Arc::clone(&self.free_blocks),
            total_allocated: Arc::clone(&self.total_allocated),
            peak_allocated: Arc::clone(&self.peak_allocated),
            allocation_counter: Arc::clone(&self.allocation_counter),
            stats: Arc::clone(&self.stats),
        }
    }
}

/// Scoped allocation for automatic cleanup
pub struct ScopedAllocation<'a> {
    pool: &'a MobileMemoryPool,
    allocation: Option<MemoryAllocation>,
}

impl<'a> ScopedAllocation<'a> {
    /// Create new scoped allocation
    pub fn new(pool: &'a MobileMemoryPool, size: usize, alignment: usize) -> Result<Self> {
        let allocation = pool.allocate(size, alignment)?;
        Ok(Self {
            pool,
            allocation: Some(allocation),
        })
    }

    /// Get pointer to allocated memory
    pub fn ptr(&self) -> *mut u8 {
        self.allocation.as_ref().expect("No allocation").ptr
    }

    /// Get size of allocation
    pub fn size(&self) -> usize {
        self.allocation.as_ref().expect("No allocation").size
    }
}

impl<'a> Drop for ScopedAllocation<'a> {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            let _ = self.pool.deallocate(allocation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryPoolConfig {
            max_memory_bytes: 10 * 1024 * 1024, // 10MB
            allocation_strategy: AllocationStrategy::FirstFit,
            enable_defragmentation: true,
        };

        let pool = MobileMemoryPool::new(config).expect("Failed to create pool");
        let stats = pool.get_stats();

        assert_eq!(stats.current_allocations, 0);
        assert_eq!(stats.current_memory_bytes, 0);
    }

    #[test]
    fn test_allocation_deallocation() {
        let config = MemoryPoolConfig {
            max_memory_bytes: 10 * 1024 * 1024,
            allocation_strategy: AllocationStrategy::BestFit,
            enable_defragmentation: false,
        };

        let pool = MobileMemoryPool::new(config).expect("Failed to create pool");

        // Allocate memory
        let alloc1 = pool.allocate(1024, 64).expect("Allocation failed");
        assert_eq!(alloc1.size, 1024);
        assert!(!alloc1.ptr.is_null());

        let stats = pool.get_stats();
        assert_eq!(stats.current_allocations, 1);

        // Deallocate
        pool.deallocate(alloc1).expect("Deallocation failed");

        let stats = pool.get_stats();
        assert_eq!(stats.current_allocations, 0);
    }

    #[test]
    fn test_scoped_allocation() {
        let config = MemoryPoolConfig {
            max_memory_bytes: 10 * 1024 * 1024,
            allocation_strategy: AllocationStrategy::FirstFit,
            enable_defragmentation: false,
        };

        let pool = MobileMemoryPool::new(config).expect("Failed to create pool");

        {
            let scoped = ScopedAllocation::new(&pool, 2048, 128).expect("Scoped allocation failed");
            assert!(!scoped.ptr().is_null());
            assert_eq!(scoped.size(), 2048);

            let stats = pool.get_stats();
            assert_eq!(stats.current_allocations, 1);
        }

        // Should be deallocated after scope
        let stats = pool.get_stats();
        assert_eq!(stats.current_allocations, 0);
    }

    #[test]
    fn test_memory_limit() {
        let config = MemoryPoolConfig {
            max_memory_bytes: 1024, // Only 1KB
            allocation_strategy: AllocationStrategy::FirstFit,
            enable_defragmentation: false,
        };

        let pool = MobileMemoryPool::new(config).expect("Failed to create pool");

        // This should succeed
        let alloc1 = pool.allocate(512, 64).expect("Allocation failed");

        // This should fail (would exceed limit)
        let result = pool.allocate(1024, 64);
        assert!(result.is_err());

        // Deallocate first allocation
        pool.deallocate(alloc1).expect("Deallocation failed");

        // Now the large allocation should succeed
        let alloc2 = pool.allocate(1024, 64).expect("Allocation failed");
        assert_eq!(alloc2.size, 1024);
    }
}
