//! Advanced memory pool for efficient tensor allocation in WebAssembly
//!
//! This module provides a memory pool system optimized for WebAssembly environments
//! to reduce allocation overhead and fragmentation.

#![allow(dead_code)]
#![allow(clippy::missing_enforced_import_renames)]

use std::collections::HashMap;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Memory pool configuration
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    max_pool_size_mb: f64,
    block_sizes: Vec<usize>,
    max_blocks_per_size: usize,
    enable_stats: bool,
}

#[wasm_bindgen]
impl MemoryPoolConfig {
    /// Create a new memory pool configuration
    #[wasm_bindgen(constructor)]
    pub fn new(max_pool_size_mb: f64) -> Self {
        Self {
            max_pool_size_mb,
            block_sizes: vec![
                1024,     // 1KB
                4096,     // 4KB
                16384,    // 16KB
                65536,    // 64KB
                262144,   // 256KB
                1048576,  // 1MB
                4194304,  // 4MB
                16777216, // 16MB
            ],
            max_blocks_per_size: 32,
            enable_stats: true,
        }
    }

    /// Create a configuration optimized for mobile devices
    pub fn mobile_optimized() -> Self {
        Self {
            max_pool_size_mb: 64.0, // Smaller pool for mobile
            block_sizes: vec![
                1024,    // 1KB
                4096,    // 4KB
                16384,   // 16KB
                65536,   // 64KB
                262144,  // 256KB
                1048576, // 1MB
            ],
            max_blocks_per_size: 16,
            enable_stats: true,
        }
    }

    /// Create a configuration optimized for desktop browsers
    pub fn desktop_optimized() -> Self {
        Self {
            max_pool_size_mb: 256.0, // Larger pool for desktop
            block_sizes: vec![
                1024,     // 1KB
                4096,     // 4KB
                16384,    // 16KB
                65536,    // 64KB
                262144,   // 256KB
                1048576,  // 1MB
                4194304,  // 4MB
                16777216, // 16MB
                67108864, // 64MB
            ],
            max_blocks_per_size: 64,
            enable_stats: true,
        }
    }

    /// Set custom block sizes
    pub fn set_block_sizes(&mut self, sizes: Vec<usize>) {
        self.block_sizes = sizes;
    }

    /// Set maximum blocks per size
    pub fn set_max_blocks_per_size(&mut self, max_blocks: usize) {
        self.max_blocks_per_size = max_blocks;
    }

    /// Enable or disable statistics collection
    pub fn set_enable_stats(&mut self, enable: bool) {
        self.enable_stats = enable;
    }
}

/// Memory pool statistics
#[wasm_bindgen]
#[derive(Debug, Clone, Default)]
pub struct MemoryPoolStats {
    total_allocations: u64,
    total_deallocations: u64,
    current_allocated_bytes: usize,
    peak_allocated_bytes: usize,
    pool_hits: u64,
    pool_misses: u64,
    fragmentation_ratio: f64,
}

#[wasm_bindgen]
impl MemoryPoolStats {
    #[wasm_bindgen(getter)]
    pub fn total_allocations(&self) -> u64 {
        self.total_allocations
    }

    #[wasm_bindgen(getter)]
    pub fn total_deallocations(&self) -> u64 {
        self.total_deallocations
    }

    #[wasm_bindgen(getter)]
    pub fn current_allocated_mb(&self) -> f64 {
        self.current_allocated_bytes as f64 / (1024.0 * 1024.0)
    }

    #[wasm_bindgen(getter)]
    pub fn peak_allocated_mb(&self) -> f64 {
        self.peak_allocated_bytes as f64 / (1024.0 * 1024.0)
    }

    #[wasm_bindgen(getter)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.pool_hits + self.pool_misses;
        if total == 0 {
            0.0
        } else {
            self.pool_hits as f64 / total as f64
        }
    }

    #[wasm_bindgen(getter)]
    pub fn fragmentation_ratio(&self) -> f64 {
        self.fragmentation_ratio
    }
}

/// Memory block for pooled allocation
struct MemoryBlock {
    data: Vec<u8>,
    size: usize,
    in_use: bool,
}

impl MemoryBlock {
    fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            size,
            in_use: false,
        }
    }

    fn acquire(&mut self) -> Option<&mut [u8]> {
        if !self.in_use {
            self.in_use = true;
            Some(&mut self.data)
        } else {
            None
        }
    }

    fn release(&mut self) {
        self.in_use = false;
        // Optionally zero out the data for security
        self.data.fill(0);
    }
}

/// Advanced memory pool for efficient allocation
#[wasm_bindgen]
pub struct MemoryPool {
    config: MemoryPoolConfig,
    pools: HashMap<usize, Vec<MemoryBlock>>,
    stats: MemoryPoolStats,
}

#[wasm_bindgen]
impl MemoryPool {
    /// Create a new memory pool with the given configuration
    #[wasm_bindgen(constructor)]
    pub fn new(config: MemoryPoolConfig) -> Result<MemoryPool, JsValue> {
        let mut pools = HashMap::new();

        // Pre-allocate pools for each block size
        for &size in &config.block_sizes {
            pools.insert(size, Vec::new());
        }

        Ok(Self {
            config,
            pools,
            stats: MemoryPoolStats::default(),
        })
    }
}

impl MemoryPool {
    /// Allocate memory from the pool
    fn allocate(&mut self, size: usize) -> Option<Vec<u8>> {
        self.stats.total_allocations += 1;

        // Find the best fitting block size
        let block_size = self.find_best_block_size(size);

        if let Some(block_size) = block_size {
            // Try to get from pool first
            if let Some(blocks) = self.pools.get_mut(&block_size) {
                for block in blocks.iter_mut() {
                    if !block.in_use {
                        block.in_use = true;
                        self.stats.pool_hits += 1;
                        self.update_allocation_stats(block_size);
                        // Return a copy of the requested size
                        return Some(vec![0u8; size]);
                    }
                }
            }

            // Pool miss - create new block if within limits
            if self.can_allocate_new_block(block_size) {
                self.stats.pool_misses += 1;
                self.update_allocation_stats(block_size);

                // Add block to pool
                if let Some(blocks) = self.pools.get_mut(&block_size) {
                    blocks.push(MemoryBlock::new(block_size));
                    return Some(vec![0u8; size]);
                }
            }
        }

        // Fallback - return None to indicate allocation failure
        None
    }

    /// Deallocate memory back to the pool (simplified for WASM)
    fn deallocate(&mut self, block_size: usize) {
        self.stats.total_deallocations += 1;
        self.stats.current_allocated_bytes =
            self.stats.current_allocated_bytes.saturating_sub(block_size);

        // Find and release a matching block
        if let Some(blocks) = self.pools.get_mut(&block_size) {
            for block in blocks.iter_mut() {
                if block.in_use {
                    block.release();
                    self.update_fragmentation_stats();
                    break;
                }
            }
        }
    }

    /// Clear all pools
    fn clear(&mut self) {
        for (_, blocks) in self.pools.iter_mut() {
            blocks.clear();
        }
        self.stats.current_allocated_bytes = 0;
        self.update_fragmentation_stats();
    }

    /// Reset statistics
    fn reset_stats(&mut self) {
        self.stats = MemoryPoolStats::default();
    }

    /// Find the best fitting block size for the requested size
    fn find_best_block_size(&self, size: usize) -> Option<usize> {
        self.config.block_sizes.iter().find(|&&block_size| block_size >= size).copied()
    }

    /// Check if we can allocate a new block
    fn can_allocate_new_block(&self, block_size: usize) -> bool {
        let max_pool_bytes = (self.config.max_pool_size_mb * 1024.0 * 1024.0) as usize;
        let current_blocks = self.pools.get(&block_size).map(|b| b.len()).unwrap_or(0);

        current_blocks < self.config.max_blocks_per_size
            && self.stats.current_allocated_bytes + block_size <= max_pool_bytes
    }

    /// Update allocation statistics
    fn update_allocation_stats(&mut self, size: usize) {
        self.stats.current_allocated_bytes += size;
        if self.stats.current_allocated_bytes > self.stats.peak_allocated_bytes {
            self.stats.peak_allocated_bytes = self.stats.current_allocated_bytes;
        }
        self.update_fragmentation_stats();
    }

    /// Update fragmentation statistics
    fn update_fragmentation_stats(&mut self) {
        let mut total_blocks = 0;
        let mut free_blocks = 0;

        for blocks in self.pools.values() {
            total_blocks += blocks.len();
            free_blocks += blocks.iter().filter(|b| !b.in_use).count();
        }

        if total_blocks > 0 {
            self.stats.fragmentation_ratio = free_blocks as f64 / total_blocks as f64;
        }
    }
}

/// Allocate memory using a memory pool approach
#[wasm_bindgen]
pub fn pool_allocate(size: usize) -> Result<js_sys::Uint8Array, JsValue> {
    if size == 0 {
        return Err(JsValue::from_str("Cannot allocate zero bytes"));
    }

    // Simple allocation for WASM environment
    let data = vec![0u8; size];
    Ok(js_sys::Uint8Array::from(&data[..]))
}

/// Placeholder for memory deallocation
#[wasm_bindgen]
pub fn pool_deallocate(_size: usize) -> Result<bool, JsValue> {
    // In WASM, memory is managed by the JS garbage collector
    Ok(true)
}

/// Get memory usage recommendations based on current environment
#[wasm_bindgen]
pub fn get_memory_recommendations() -> Result<String, JsValue> {
    let memory_stats = crate::get_memory_stats();
    let current_mb = memory_stats.used_mb();

    let recommendations = if current_mb < 32.0 {
        "Low memory usage detected. Consider enabling larger memory pools for better performance."
    } else if current_mb < 128.0 {
        "Moderate memory usage. Current settings should be optimal."
    } else if current_mb < 256.0 {
        "High memory usage detected. Consider enabling memory pool optimizations."
    } else {
        "Very high memory usage. Consider reducing model size or enabling aggressive memory optimization."
    };

    Ok(recommendations.to_string())
}
