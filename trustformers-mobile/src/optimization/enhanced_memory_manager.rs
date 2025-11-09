//! Enhanced Memory Manager for Mobile AI Optimization
//!
//! Provides advanced memory management capabilities specifically designed for
//! mobile AI inference, including smart allocation strategies, memory pressure
//! handling, and automatic optimization.

use crate::{MobileBackend, MobilePlatform};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::TrustformersError;

/// Enhanced memory manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedMemoryConfig {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Memory pressure threshold (0.0-1.0)
    pub pressure_threshold: f32,
    /// Enable automatic garbage collection
    pub enable_auto_gc: bool,
    /// Enable memory compression
    pub enable_compression: bool,
    /// Enable memory encryption for sensitive data
    pub enable_encryption: bool,
    /// Memory allocation tracking
    pub enable_tracking: bool,
    /// Platform-specific optimizations
    pub platform_optimizations: bool,
    /// Memory prefetching strategy
    pub prefetch_strategy: PrefetchStrategy,
    /// Cache line size optimization
    pub cache_line_optimization: bool,
    /// NUMA-aware allocation (for multi-core devices)
    pub numa_aware: bool,
}

impl Default for EnhancedMemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 512 * 1024 * 1024, // 512MB default
            pressure_threshold: 0.8,             // 80% usage threshold
            enable_auto_gc: true,
            enable_compression: false,
            enable_encryption: false,
            enable_tracking: true,
            platform_optimizations: true,
            prefetch_strategy: PrefetchStrategy::Adaptive,
            cache_line_optimization: true,
            numa_aware: false,
        }
    }
}

/// Memory prefetching strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    /// No prefetching
    None,
    /// Sequential prefetching
    Sequential,
    /// Stride-based prefetching
    Stride,
    /// Pattern-based prefetching
    Pattern,
    /// Adaptive prefetching
    Adaptive,
}

/// Advanced allocation strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AdvancedAllocationStrategy {
    /// Linear allocation with compaction
    LinearCompact,
    /// Segregated free lists
    SegregatedFree,
    /// Slab allocation for fixed sizes
    Slab,
    /// Thread-local allocation
    ThreadLocal,
    /// Generational allocation
    Generational,
    /// Machine learning guided allocation
    MLGuided,
}

/// Memory allocation metadata
#[derive(Debug)]
pub struct AllocationMetadata {
    /// Allocation ID
    pub id: usize,
    /// Size in bytes
    pub size: usize,
    /// Alignment requirement
    pub alignment: usize,
    /// Allocation timestamp
    pub timestamp: Instant,
    /// Last access timestamp
    pub last_access: Instant,
    /// Access frequency
    pub access_count: AtomicUsize,
    /// Memory type (model weights, activations, etc.)
    pub memory_type: MemoryType,
    /// Priority level
    pub priority: AllocationPriority,
    /// Lifetime hint
    pub lifetime_hint: LifetimeHint,
    /// Compression status
    pub compressed: bool,
    /// Encryption status
    pub encrypted: bool,
}

/// Memory type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// Model weights (persistent)
    ModelWeights,
    /// Intermediate activations (temporary)
    Activations,
    /// Input/output buffers
    IOBuffers,
    /// Gradient storage
    Gradients,
    /// Cache data
    Cache,
    /// Scratch space
    Scratch,
    /// System metadata
    Metadata,
}

/// Allocation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AllocationPriority {
    /// Critical system allocations
    Critical,
    /// High priority (model weights)
    High,
    /// Normal priority (activations)
    Normal,
    /// Low priority (cache)
    Low,
    /// Background (temporary data)
    Background,
}

/// Lifetime hint for allocations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LifetimeHint {
    /// Very short-lived (< 1ms)
    VeryShort,
    /// Short-lived (< 100ms)
    Short,
    /// Medium-lived (< 1s)
    Medium,
    /// Long-lived (< 60s)
    Long,
    /// Persistent (entire session)
    Persistent,
}

/// Memory pressure level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryPressure {
    /// Normal operation
    Normal,
    /// Light pressure
    Light,
    /// Moderate pressure
    Moderate,
    /// High pressure
    High,
    /// Critical pressure
    Critical,
}

/// Enhanced memory manager
pub struct EnhancedMemoryManager {
    config: EnhancedMemoryConfig,
    platform: MobilePlatform,
    backend: MobileBackend,

    // Memory tracking
    allocations: Arc<Mutex<HashMap<usize, AllocationMetadata>>>,
    memory_usage: AtomicUsize,
    peak_memory: AtomicUsize,
    next_allocation_id: AtomicUsize,

    // Free memory tracking
    free_blocks: Arc<Mutex<BTreeMap<usize, Vec<usize>>>>, // size -> offsets
    memory_pool: Arc<Mutex<Vec<u8>>>,

    // Memory pressure management
    pressure_level: Arc<Mutex<MemoryPressure>>,
    pressure_callbacks: Arc<Mutex<Vec<Box<dyn Fn(MemoryPressure) + Send + Sync>>>>,

    // Performance analytics
    allocation_stats: Arc<Mutex<AllocationStats>>,
    access_patterns: Arc<Mutex<HashMap<usize, AccessPattern>>>,

    // Background management
    gc_enabled: AtomicBool,
    compression_enabled: AtomicBool,

    // Platform-specific optimizations
    cache_line_size: usize,
    page_size: usize,
    numa_nodes: usize,
}

/// Memory allocation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AllocationStats {
    /// Total allocations made
    pub total_allocations: usize,
    /// Total deallocations made
    pub total_deallocations: usize,
    /// Current active allocations
    pub active_allocations: usize,
    /// Total bytes allocated
    pub total_allocated_bytes: usize,
    /// Total bytes deallocated
    pub total_deallocated_bytes: usize,
    /// Current allocated bytes
    pub current_allocated_bytes: usize,
    /// Peak memory usage
    pub peak_memory_bytes: usize,
    /// Average allocation size
    pub avg_allocation_size: f64,
    /// Allocation failures
    pub allocation_failures: usize,
    /// Garbage collection runs
    pub gc_runs: usize,
    /// Compression operations
    pub compression_ops: usize,
    /// Fragmentation ratio
    pub fragmentation_ratio: f32,
    /// Cache hit rate
    pub cache_hit_rate: f32,
}

/// Memory access pattern
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Access frequency
    pub frequency: usize,
    /// Average access interval
    pub avg_interval_ms: f64,
    /// Sequential access ratio
    pub sequential_ratio: f32,
    /// Random access ratio
    pub random_ratio: f32,
    /// Read/write ratio
    pub read_write_ratio: f32,
    /// Last access time
    pub last_access: Instant,
    /// Access history
    pub access_history: VecDeque<Instant>,
}

impl EnhancedMemoryManager {
    /// Create a new enhanced memory manager
    pub fn new(
        config: EnhancedMemoryConfig,
        platform: MobilePlatform,
        backend: MobileBackend,
    ) -> Result<Self> {
        // Initialize memory pool
        let memory_pool = vec![0u8; config.max_memory_bytes];

        // Initialize free blocks with the entire memory pool as available
        let mut initial_free_blocks = BTreeMap::new();
        initial_free_blocks.insert(config.max_memory_bytes, vec![0]); // Entire pool available at offset 0

        // Detect platform-specific parameters
        let (cache_line_size, page_size, numa_nodes) = Self::detect_platform_parameters(platform);

        // Extract config fields before moving
        let enable_auto_gc = config.enable_auto_gc;
        let enable_compression = config.enable_compression;

        Ok(Self {
            config,
            platform,
            backend,
            allocations: Arc::new(Mutex::new(HashMap::new())),
            memory_usage: AtomicUsize::new(0),
            peak_memory: AtomicUsize::new(0),
            next_allocation_id: AtomicUsize::new(1),
            free_blocks: Arc::new(Mutex::new(initial_free_blocks)),
            memory_pool: Arc::new(Mutex::new(memory_pool)),
            pressure_level: Arc::new(Mutex::new(MemoryPressure::Normal)),
            pressure_callbacks: Arc::new(Mutex::new(Vec::new())),
            allocation_stats: Arc::new(Mutex::new(AllocationStats::default())),
            access_patterns: Arc::new(Mutex::new(HashMap::new())),
            gc_enabled: AtomicBool::new(enable_auto_gc),
            compression_enabled: AtomicBool::new(enable_compression),
            cache_line_size,
            page_size,
            numa_nodes,
        })
    }

    /// Allocate memory with enhanced tracking and optimization
    pub fn allocate(
        &self,
        size: usize,
        alignment: usize,
        memory_type: MemoryType,
        priority: AllocationPriority,
        lifetime_hint: LifetimeHint,
    ) -> Result<usize> {
        // Check memory pressure first
        self.check_memory_pressure()?;

        // Align size to cache line if optimization is enabled
        let aligned_size = if self.config.cache_line_optimization {
            self.align_to_cache_line(size)
        } else {
            size
        };

        // Try to allocate
        let allocation_id = self.allocate_internal(
            aligned_size,
            alignment,
            memory_type,
            priority,
            lifetime_hint,
        )?;

        // Update statistics
        self.update_allocation_stats(aligned_size, true);

        // Track access patterns if enabled
        if self.config.enable_tracking {
            self.init_access_pattern(allocation_id);
        }

        // Trigger background optimization if needed
        self.maybe_trigger_background_optimization();

        Ok(allocation_id)
    }

    /// Deallocate memory
    pub fn deallocate(&self, allocation_id: usize) -> Result<()> {
        let size = {
            let mut allocations = self.allocations.lock().unwrap();
            let metadata = allocations.remove(&allocation_id).ok_or_else(|| {
                TrustformersError::invalid_input("Invalid allocation ID".to_string())
            })?;
            metadata.size
        };

        // Update memory usage
        self.memory_usage.fetch_sub(size, Ordering::Relaxed);

        // Update statistics
        self.update_allocation_stats(size, false);

        // Add to free blocks
        self.add_free_block(size);

        Ok(())
    }

    /// Record memory access for pattern analysis
    pub fn record_access(&self, allocation_id: usize, access_type: AccessType) {
        if !self.config.enable_tracking {
            return;
        }

        // Update allocation metadata
        if let Ok(mut allocations) = self.allocations.lock() {
            if let Some(metadata) = allocations.get_mut(&allocation_id) {
                metadata.last_access = Instant::now();
                metadata.access_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Update access patterns
        if let Ok(mut patterns) = self.access_patterns.lock() {
            patterns
                .entry(allocation_id)
                .or_insert_with(|| AccessPattern {
                    frequency: 0,
                    avg_interval_ms: 0.0,
                    sequential_ratio: 0.0,
                    random_ratio: 0.0,
                    read_write_ratio: 0.0,
                    last_access: Instant::now(),
                    access_history: VecDeque::with_capacity(100),
                })
                .record_access(access_type);
        }
    }

    /// Trigger garbage collection
    pub fn garbage_collect(&self) -> Result<usize> {
        if !self.gc_enabled.load(Ordering::Relaxed) {
            return Ok(0);
        }

        let mut freed_bytes = 0;
        let now = Instant::now();

        // Collect expired short-lived allocations
        let mut expired_allocations = Vec::new();

        if let Ok(allocations) = self.allocations.lock() {
            for (&id, metadata) in allocations.iter() {
                let age = now.duration_since(metadata.timestamp);
                let should_collect = match metadata.lifetime_hint {
                    LifetimeHint::VeryShort => age > Duration::from_millis(1),
                    LifetimeHint::Short => age > Duration::from_millis(100),
                    LifetimeHint::Medium => {
                        age > Duration::from_secs(1)
                            && metadata.access_count.load(Ordering::Relaxed) == 0
                    },
                    _ => false,
                };

                if should_collect && metadata.priority <= AllocationPriority::Low {
                    expired_allocations.push(id);
                }
            }
        }

        // Free expired allocations
        for allocation_id in expired_allocations {
            if self.deallocate(allocation_id).is_ok() {
                freed_bytes += self
                    .allocations
                    .lock()
                    .unwrap()
                    .get(&allocation_id)
                    .map(|m| m.size)
                    .unwrap_or(0);
            }
        }

        // Update statistics
        if let Ok(mut stats) = self.allocation_stats.lock() {
            stats.gc_runs += 1;
        }

        Ok(freed_bytes)
    }

    /// Get current memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let current_usage = self.memory_usage.load(Ordering::Relaxed);
        let peak_usage = self.peak_memory.load(Ordering::Relaxed);
        let max_memory = self.config.max_memory_bytes;

        let allocation_stats = self.allocation_stats.lock().unwrap().clone();
        let pressure_level = *self.pressure_level.lock().unwrap();

        MemoryStats {
            current_usage_bytes: current_usage,
            peak_usage_bytes: peak_usage,
            max_memory_bytes: max_memory,
            usage_percentage: (current_usage as f32 / max_memory as f32) * 100.0,
            pressure_level,
            allocation_stats,
            fragmentation_score: self.calculate_fragmentation_score(),
            cache_efficiency: self.calculate_cache_efficiency(),
        }
    }

    /// Register callback for memory pressure events
    pub fn register_pressure_callback<F>(&self, callback: F)
    where
        F: Fn(MemoryPressure) + Send + Sync + 'static,
    {
        if let Ok(mut callbacks) = self.pressure_callbacks.lock() {
            callbacks.push(Box::new(callback));
        }
    }

    /// Optimize memory layout for better cache performance
    pub fn optimize_layout(&self) -> Result<usize> {
        let mut optimized_bytes = 0;

        if !self.config.cache_line_optimization {
            return Ok(0);
        }

        // Analyze access patterns and reorganize frequently accessed data
        if let Ok(patterns) = self.access_patterns.lock() {
            let mut hot_allocations = Vec::new();

            for (&id, pattern) in patterns.iter() {
                if pattern.frequency > 100 && pattern.sequential_ratio > 0.8 {
                    hot_allocations.push(id);
                }
            }

            // Reorganize hot allocations for better cache locality
            // This would involve actual memory movement in a real implementation
            optimized_bytes = hot_allocations.len() * self.cache_line_size;
        }

        Ok(optimized_bytes)
    }

    /// Internal allocation implementation
    fn allocate_internal(
        &self,
        size: usize,
        alignment: usize,
        memory_type: MemoryType,
        priority: AllocationPriority,
        lifetime_hint: LifetimeHint,
    ) -> Result<usize> {
        // Find suitable free block
        let offset = self.find_free_block(size, alignment)?;

        // Create allocation metadata
        let allocation_id = self.next_allocation_id.fetch_add(1, Ordering::Relaxed);
        let metadata = AllocationMetadata {
            id: allocation_id,
            size,
            alignment,
            timestamp: Instant::now(),
            last_access: Instant::now(),
            access_count: AtomicUsize::new(0),
            memory_type,
            priority,
            lifetime_hint,
            compressed: false,
            encrypted: false,
        };

        // Store allocation metadata
        self.allocations.lock().unwrap().insert(allocation_id, metadata);

        // Update memory usage
        let new_usage = self.memory_usage.fetch_add(size, Ordering::Relaxed) + size;

        // Update peak usage
        let mut peak = self.peak_memory.load(Ordering::Relaxed);
        while new_usage > peak {
            match self.peak_memory.compare_exchange_weak(
                peak,
                new_usage,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => peak = current,
            }
        }

        Ok(allocation_id)
    }

    /// Find a suitable free block
    fn find_free_block(&self, size: usize, alignment: usize) -> Result<usize> {
        let free_blocks = self.free_blocks.lock().unwrap();

        // Find the smallest block that fits
        for (&block_size, offsets) in free_blocks.range(size..) {
            if !offsets.is_empty() {
                return Ok(offsets[0]);
            }
        }

        Err(TrustformersError::hardware_error(
            "No suitable free block found",
            "try_allocate_from_free",
        )
        .into())
    }

    /// Add a block to the free list
    fn add_free_block(&self, size: usize) {
        // In a real implementation, this would track actual memory offsets
        let mut free_blocks = self.free_blocks.lock().unwrap();
        free_blocks.entry(size).or_default().push(0);
    }

    /// Check and update memory pressure
    fn check_memory_pressure(&self) -> Result<()> {
        let current_usage = self.memory_usage.load(Ordering::Relaxed);
        let max_memory = self.config.max_memory_bytes;
        let usage_ratio = current_usage as f32 / max_memory as f32;

        let new_pressure = match usage_ratio {
            r if r < 0.5 => MemoryPressure::Normal,
            r if r < 0.65 => MemoryPressure::Light,
            r if r < 0.8 => MemoryPressure::Moderate,
            r if r < 0.95 => MemoryPressure::High,
            _ => MemoryPressure::Critical,
        };

        let old_pressure = {
            let mut pressure_level = self.pressure_level.lock().unwrap();
            let old = *pressure_level;
            *pressure_level = new_pressure;
            old
        };

        // Trigger callbacks if pressure level changed
        if new_pressure != old_pressure {
            if let Ok(callbacks) = self.pressure_callbacks.lock() {
                for callback in callbacks.iter() {
                    callback(new_pressure);
                }
            }
        }

        // Take action based on pressure level
        match new_pressure {
            MemoryPressure::High => {
                self.garbage_collect().ok();
            },
            MemoryPressure::Critical => {
                return Err(TrustformersError::hardware_error(
                    "Critical memory pressure",
                    "allocate",
                )
                .into());
            },
            _ => {},
        }

        Ok(())
    }

    /// Detect platform-specific parameters
    fn detect_platform_parameters(platform: MobilePlatform) -> (usize, usize, usize) {
        match platform {
            MobilePlatform::Ios => (64, 16384, 1), // ARM64 cache line, 16KB page, 1 NUMA node
            MobilePlatform::Android => (64, 4096, 1), // ARM64 cache line, 4KB page, 1 NUMA node
            MobilePlatform::Generic => (64, 4096, 1), // Conservative defaults
        }
    }

    /// Align size to cache line boundary
    fn align_to_cache_line(&self, size: usize) -> usize {
        (size + self.cache_line_size - 1) & !(self.cache_line_size - 1)
    }

    /// Initialize access pattern tracking
    fn init_access_pattern(&self, allocation_id: usize) {
        if let Ok(mut patterns) = self.access_patterns.lock() {
            patterns.insert(
                allocation_id,
                AccessPattern {
                    frequency: 0,
                    avg_interval_ms: 0.0,
                    sequential_ratio: 0.0,
                    random_ratio: 0.0,
                    read_write_ratio: 0.0,
                    last_access: Instant::now(),
                    access_history: VecDeque::with_capacity(100),
                },
            );
        }
    }

    /// Update allocation statistics
    fn update_allocation_stats(&self, size: usize, is_allocation: bool) {
        if let Ok(mut stats) = self.allocation_stats.lock() {
            if is_allocation {
                stats.total_allocations += 1;
                stats.active_allocations += 1;
                stats.total_allocated_bytes += size;
                stats.current_allocated_bytes += size;

                if stats.current_allocated_bytes > stats.peak_memory_bytes {
                    stats.peak_memory_bytes = stats.current_allocated_bytes;
                }

                stats.avg_allocation_size =
                    stats.total_allocated_bytes as f64 / stats.total_allocations as f64;
            } else {
                stats.total_deallocations += 1;
                stats.active_allocations = stats.active_allocations.saturating_sub(1);
                stats.total_deallocated_bytes += size;
                stats.current_allocated_bytes = stats.current_allocated_bytes.saturating_sub(size);
            }
        }
    }

    /// Maybe trigger background optimization
    fn maybe_trigger_background_optimization(&self) {
        let usage_ratio =
            self.memory_usage.load(Ordering::Relaxed) as f32 / self.config.max_memory_bytes as f32;

        if usage_ratio > self.config.pressure_threshold {
            // Trigger background garbage collection
            if self.gc_enabled.load(Ordering::Relaxed) {
                let _ = std::thread::spawn({
                    let manager = self.clone_weak();
                    move || {
                        if let Some(manager) = manager.upgrade() {
                            let _ = manager.garbage_collect();
                        }
                    }
                });
            }
        }
    }

    /// Calculate memory fragmentation score
    fn calculate_fragmentation_score(&self) -> f32 {
        // Simplified fragmentation calculation
        let free_blocks = self.free_blocks.lock().unwrap();
        if free_blocks.is_empty() {
            return 0.0;
        }

        let total_free_blocks: usize = free_blocks.values().map(|v| v.len()).sum();
        let unique_sizes = free_blocks.len();

        if unique_sizes == 0 {
            return 0.0;
        }

        // Higher score means more fragmentation
        (total_free_blocks as f32 / unique_sizes as f32 - 1.0).max(0.0).min(1.0)
    }

    /// Calculate cache efficiency
    fn calculate_cache_efficiency(&self) -> f32 {
        if !self.config.enable_tracking {
            return 0.0;
        }

        let patterns = self.access_patterns.lock().unwrap();
        if patterns.is_empty() {
            return 0.0;
        }

        let avg_sequential_ratio: f32 =
            patterns.values().map(|p| p.sequential_ratio).sum::<f32>() / patterns.len() as f32;

        avg_sequential_ratio
    }

    /// Create weak reference for background operations
    fn clone_weak(&self) -> std::sync::Weak<Self> {
        // This would require Arc<Self> in a real implementation
        // For now, just return a dummy weak reference
        std::sync::Weak::new()
    }
}

// Note: std::sync::Weak already has an upgrade() method, no need to implement it

/// Memory access type
#[derive(Debug, Clone, Copy)]
pub enum AccessType {
    Read,
    Write,
    ReadWrite,
}

/// Complete memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_usage_bytes: usize,
    /// Peak memory usage in bytes
    pub peak_usage_bytes: usize,
    /// Maximum available memory in bytes
    pub max_memory_bytes: usize,
    /// Usage percentage
    pub usage_percentage: f32,
    /// Current memory pressure level
    pub pressure_level: MemoryPressure,
    /// Detailed allocation statistics
    pub allocation_stats: AllocationStats,
    /// Memory fragmentation score (0.0-1.0)
    pub fragmentation_score: f32,
    /// Cache efficiency score (0.0-1.0)
    pub cache_efficiency: f32,
}

impl AccessPattern {
    /// Record a memory access
    fn record_access(&mut self, access_type: AccessType) {
        let now = Instant::now();

        // Update frequency
        self.frequency += 1;

        // Update access history
        self.access_history.push_back(now);
        if self.access_history.len() > 100 {
            self.access_history.pop_front();
        }

        // Calculate average interval
        if self.access_history.len() > 1 {
            let total_duration: Duration = self
                .access_history
                .iter()
                .zip(self.access_history.iter().skip(1))
                .map(|(a, b)| b.duration_since(*a))
                .sum();

            self.avg_interval_ms =
                total_duration.as_millis() as f64 / (self.access_history.len() - 1) as f64;
        }

        // Update read/write ratio
        match access_type {
            AccessType::Read => {
                self.read_write_ratio = (self.read_write_ratio * (self.frequency - 1) as f32 + 1.0)
                    / self.frequency as f32
            },
            AccessType::Write => {
                self.read_write_ratio =
                    (self.read_write_ratio * (self.frequency - 1) as f32) / self.frequency as f32
            },
            AccessType::ReadWrite => {}, // No change to ratio
        }

        self.last_access = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_memory_manager_creation() {
        let config = EnhancedMemoryConfig::default();
        let manager =
            EnhancedMemoryManager::new(config, MobilePlatform::Generic, MobileBackend::CPU);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_memory_allocation_and_deallocation() {
        let config = EnhancedMemoryConfig::default();
        let manager =
            EnhancedMemoryManager::new(config, MobilePlatform::Generic, MobileBackend::CPU)
                .unwrap();

        let allocation_id = manager.allocate(
            1024,
            8,
            MemoryType::Activations,
            AllocationPriority::Normal,
            LifetimeHint::Short,
        );
        assert!(allocation_id.is_ok());

        let allocation_id = allocation_id.unwrap();
        let result = manager.deallocate(allocation_id);
        assert!(result.is_ok());
    }

    #[test]
    fn test_memory_pressure_detection() {
        let mut config = EnhancedMemoryConfig::default();
        config.max_memory_bytes = 1024; // Very small for testing

        let manager =
            EnhancedMemoryManager::new(config, MobilePlatform::Generic, MobileBackend::CPU)
                .unwrap();

        // Allocate enough to trigger pressure
        for _ in 0..10 {
            let _ = manager.allocate(
                100,
                8,
                MemoryType::Activations,
                AllocationPriority::Normal,
                LifetimeHint::Short,
            );
        }

        let stats = manager.get_memory_stats();
        assert!(stats.pressure_level >= MemoryPressure::Light);
    }

    #[test]
    fn test_garbage_collection() {
        let config = EnhancedMemoryConfig::default();
        let manager =
            EnhancedMemoryManager::new(config, MobilePlatform::Generic, MobileBackend::CPU)
                .unwrap();

        // Create some short-lived allocations
        for _ in 0..5 {
            let _ = manager.allocate(
                1024,
                8,
                MemoryType::Scratch,
                AllocationPriority::Low,
                LifetimeHint::VeryShort,
            );
        }

        // Wait a bit for allocations to expire
        std::thread::sleep(Duration::from_millis(10));

        let freed_bytes = manager.garbage_collect();
        assert!(freed_bytes.is_ok());
    }
}
