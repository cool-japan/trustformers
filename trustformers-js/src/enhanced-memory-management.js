/**
 * Enhanced Memory Management System for TrustformeRS
 *
 * Advanced memory pooling, zero-copy operations, and smart allocation
 * strategies optimized for high-performance tensor operations.
 */

import { SciRS2Core, SciRS2Tensor } from './scirs2-integration.js';

/**
 * Enhanced Memory Manager with Advanced Pooling Strategies
 */
export class EnhancedMemoryManager {
  constructor(options = {}) {
    this.options = {
      maxTotalMemory: options.maxTotalMemory || 2 * 1024 * 1024 * 1024, // 2GB default
      maxPoolSize: options.maxPoolSize || 200,
      enableSmartPooling: options.enableSmartPooling !== false,
      enableZeroCopy: options.enableZeroCopy !== false,
      enableMemoryMapping: options.enableMemoryMapping !== false,
      gcThreshold: options.gcThreshold || 0.8,
      enablePredictivePreallocation: options.enablePredictivePreallocation !== false,
      enableMemoryDefragmentation: options.enableMemoryDefragmentation !== false,
      ...options
    };

    // Multi-tier memory pools
    this.pools = {
      small: new SmartMemoryPool({ maxSize: 10 * 1024 * 1024 }), // 10MB
      medium: new SmartMemoryPool({ maxSize: 100 * 1024 * 1024 }), // 100MB
      large: new SmartMemoryPool({ maxSize: 500 * 1024 * 1024 }), // 500MB
      huge: new SmartMemoryPool({ maxSize: this.options.maxTotalMemory })
    };

    // Zero-copy buffer manager
    this.zeroCopyManager = new EnhancedZeroCopyManager(this.options);

    // Memory mapping for large tensors
    this.memoryMapper = new MemoryMapper();

    // Predictive allocation system
    this.predictor = new AllocationPredictor();

    // Memory defragmentation system
    this.defragmenter = new MemoryDefragmenter();

    // Statistics and monitoring
    this.stats = {
      totalAllocations: 0,
      totalDeallocations: 0,
      poolHits: 0,
      poolMisses: 0,
      zeroCopyOperations: 0,
      memoryMappedOperations: 0,
      predictiveAllocations: 0,
      defragmentationEvents: 0,
      currentMemoryUsage: 0,
      peakMemoryUsage: 0,
      memoryEfficiency: 0
    };

    // Performance monitoring
    this.performanceMonitor = new MemoryPerformanceMonitor(this);

    // Auto-cleanup timer
    this.cleanupInterval = setInterval(() => this.performMaintenance(), 10000); // 10 seconds

    // WASM module reference
    this.wasmModule = null;
  }

  /**
   * Initialize with WASM module
   */
  initialize(wasmModule) {
    this.wasmModule = wasmModule;
    this.zeroCopyManager.initialize(wasmModule);
    this.memoryMapper.initialize(wasmModule);
    this.performanceMonitor.start();
  }

  /**
   * Enhanced tensor allocation with smart pooling
   * @param {Array<number>} shape - Tensor shape
   * @param {string} dtype - Data type
   * @param {Object} options - Allocation options
   * @returns {Object} Allocated tensor
   */
  allocateTensor(shape, dtype = 'f32', options = {}) {
    const {
      zeroCopy = this.options.enableZeroCopy,
      useMemoryMapping = false,
      initializeToZero = true,
      enablePredictive = this.options.enablePredictivePreallocation
    } = options;

    // Calculate memory requirements
    const memorySize = this._calculateMemorySize(shape, dtype);
    const poolTier = this._selectPoolTier(memorySize);

    // Update statistics
    this.stats.totalAllocations++;

    // Try predictive allocation first
    if (enablePredictive) {
      const predictedTensor = this.predictor.getPredictedTensor(shape, dtype);
      if (predictedTensor) {
        this.stats.predictiveAllocations++;
        return predictedTensor;
      }
    }

    // Try zero-copy allocation
    if (zeroCopy && this.zeroCopyManager.canAllocate(memorySize)) {
      const zeroCopyTensor = this.zeroCopyManager.allocate(shape, dtype, options);
      if (zeroCopyTensor) {
        this.stats.zeroCopyOperations++;
        this._updateMemoryUsage(memorySize);
        return zeroCopyTensor;
      }
    }

    // Try memory mapping for very large tensors
    if (useMemoryMapping && memorySize > 100 * 1024 * 1024) { // 100MB threshold
      const mappedTensor = this.memoryMapper.allocate(shape, dtype, options);
      if (mappedTensor) {
        this.stats.memoryMappedOperations++;
        this._updateMemoryUsage(memorySize);
        return mappedTensor;
      }
    }

    // Try pool allocation
    const poolTensor = this.pools[poolTier].acquire(shape, dtype, options);
    if (poolTensor) {
      this.stats.poolHits++;
      this._updateMemoryUsage(memorySize);
      return poolTensor;
    }

    // Direct allocation as fallback
    this.stats.poolMisses++;
    const tensor = this._createTensorDirect(shape, dtype, options);
    this._updateMemoryUsage(memorySize);

    // Learn from allocation pattern
    this.predictor.recordAllocation(shape, dtype, tensor);

    return tensor;
  }

  /**
   * Enhanced tensor deallocation
   * @param {Object} tensor - Tensor to deallocate
   * @param {Object} options - Deallocation options
   */
  deallocateTensor(tensor, options = {}) {
    const {
      forceRelease = false,
      returnToPool = true
    } = options;

    if (!tensor) return;

    this.stats.totalDeallocations++;

    try {
      // Determine tensor type and size
      const shape = this._getTensorShape(tensor);
      const dtype = this._getTensorDtype(tensor);
      const memorySize = this._calculateMemorySize(shape, dtype);

      // Update memory usage
      this._updateMemoryUsage(-memorySize);

      // Handle different tensor types
      if (tensor._zeroCopyManaged) {
        this.zeroCopyManager.deallocate(tensor, options);
      } else if (tensor._memoryMapped) {
        this.memoryMapper.deallocate(tensor, options);
      } else if (returnToPool && !forceRelease) {
        const poolTier = this._selectPoolTier(memorySize);
        this.pools[poolTier].release(tensor, options);
      } else {
        this._freeTensorDirect(tensor);
      }

      // Learn from deallocation pattern
      this.predictor.recordDeallocation(shape, dtype);

    } catch (error) {
      console.warn('Enhanced memory deallocation error:', error);
      // Fallback to direct free
      this._freeTensorDirect(tensor);
    }
  }

  /**
   * Batch tensor operations with memory optimization
   * @param {Array<Object>} tensors - Array of tensors
   * @param {Function} operation - Operation to perform
   * @param {Object} options - Batch options
   * @returns {Array} Results
   */
  async batchOperation(tensors, operation, options = {}) {
    const {
      enableGroupAllocation = true,
      enableTemporaryPooling = true,
      memoryBudget = this.options.maxTotalMemory * 0.5
    } = options;

    const startMemory = this.stats.currentMemoryUsage;
    const results = [];

    try {
      // Group tensors by size for efficient allocation
      const tensorGroups = enableGroupAllocation ?
        this._groupTensorsBySimilarity(tensors) :
        [tensors];

      for (const group of tensorGroups) {
        // Check memory budget
        if (this.stats.currentMemoryUsage > memoryBudget) {
          await this.performMaintenance();
        }

        // Process group
        const groupResults = await Promise.all(
          group.map(tensor => operation(tensor))
        );
        results.push(...groupResults);
      }

      return results;

    } finally {
      // Cleanup temporary allocations
      if (enableTemporaryPooling) {
        const memoryDelta = this.stats.currentMemoryUsage - startMemory;
        if (memoryDelta > 0) {
          this._cleanupTemporaryAllocations();
        }
      }
    }
  }

  /**
   * Smart memory compaction and defragmentation
   */
  async compactMemory() {
    if (!this.options.enableMemoryDefragmentation) return;

    const startTime = performance.now();

    try {
      // Analyze fragmentation
      const fragmentation = this.defragmenter.analyzeFragmentation(this.pools);

      if (fragmentation.score > 0.3) { // 30% fragmentation threshold
        console.warn('Memory fragmentation detected, performing compaction...');

        // Perform defragmentation
        await this.defragmenter.defragment(this.pools);

        this.stats.defragmentationEvents++;
        console.warn(`Memory compaction completed in ${performance.now() - startTime}ms`);
      }

    } catch (error) {
      console.warn('Memory compaction failed:', error);
    }
  }

  /**
   * Perform routine maintenance
   */
  async performMaintenance() {
    try {
      // Cleanup expired pools
      Object.values(this.pools).forEach(pool => pool.cleanup());

      // Zero-copy manager maintenance
      this.zeroCopyManager.performMaintenance();

      // Memory mapping cleanup
      this.memoryMapper.cleanup();

      // Update efficiency metrics
      this._calculateMemoryEfficiency();

      // Trigger GC if needed
      if (this._shouldTriggerGC()) {
        this._triggerGarbageCollection();
      }

      // Compact memory if needed
      await this.compactMemory();

    } catch (error) {
      console.warn('Memory maintenance error:', error);
    }
  }

  /**
   * Get comprehensive memory statistics
   */
  getStatistics() {
    return {
      ...this.stats,
      poolStats: Object.fromEntries(
        Object.entries(this.pools).map(([tier, pool]) => [
          tier,
          pool.getStatistics()
        ])
      ),
      zeroCopyStats: this.zeroCopyManager.getStatistics(),
      memoryMapStats: this.memoryMapper.getStatistics(),
      performanceMetrics: this.performanceMonitor.getMetrics(),
      predictiveStats: this.predictor.getStatistics()
    };
  }

  /**
   * Calculate memory size for shape and dtype
   * @private
   */
  _calculateMemorySize(shape, dtype) {
    const elements = shape.reduce((acc, dim) => acc * dim, 1);
    const bytesPerElement = this._getBytesPerElement(dtype);
    return elements * bytesPerElement;
  }

  /**
   * Get bytes per element for data type
   * @private
   */
  _getBytesPerElement(dtype) {
    const typeMap = {
      'f32': 4, 'f64': 8,
      'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
      'u8': 1, 'u16': 2, 'u32': 4, 'u64': 8
    };
    return typeMap[dtype] || 4; // Default to f32
  }

  /**
   * Select appropriate pool tier based on memory size
   * @private
   */
  _selectPoolTier(memorySize) {
    if (memorySize <= 10 * 1024 * 1024) return 'small';      // <= 10MB
    if (memorySize <= 100 * 1024 * 1024) return 'medium';    // <= 100MB
    if (memorySize <= 500 * 1024 * 1024) return 'large';     // <= 500MB
    return 'huge';                                            // > 500MB
  }

  /**
   * Update memory usage statistics
   * @private
   */
  _updateMemoryUsage(delta) {
    this.stats.currentMemoryUsage += delta;
    this.stats.peakMemoryUsage = Math.max(
      this.stats.peakMemoryUsage,
      this.stats.currentMemoryUsage
    );
  }

  /**
   * Group tensors by similarity for batch processing
   * @private
   */
  _groupTensorsBySimilarity(tensors) {
    const groups = new Map();

    tensors.forEach(tensor => {
      const shape = this._getTensorShape(tensor);
      const dtype = this._getTensorDtype(tensor);
      const key = `${shape.join('x')}_${dtype}`;

      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key).push(tensor);
    });

    return Array.from(groups.values());
  }

  /**
   * Get tensor shape
   * @private
   */
  _getTensorShape(tensor) {
    if (tensor.shape) {
      if (typeof tensor.shape === 'function') {
        return Array.from(tensor.shape());
      } else if (Array.isArray(tensor.shape)) {
        return tensor.shape;
      } 
        return Array.from(tensor.shape);
      
    }
    return [tensor.data?.length || 0];
  }

  /**
   * Get tensor data type
   * @private
   */
  _getTensorDtype(tensor) {
    if (tensor.dtype) {
      if (typeof tensor.dtype === 'function') {
        return tensor.dtype();
      }
      return tensor.dtype;
    }
    return 'f32'; // Default
  }

  /**
   * Create tensor directly (fallback)
   * @private
   */
  _createTensorDirect(shape, dtype, options) {
    if (this.wasmModule) {
      const totalElements = shape.reduce((acc, dim) => acc * dim, 1);
      const data = this._createTypedArray(dtype, totalElements);

      if (options.initializeToZero) {
        data.fill(0);
      }

      return this.wasmModule.WasmTensor.from_array(data, new Uint32Array(shape));
    }

    // Fallback to SciRS2Tensor
    return new SciRS2Tensor(
      this._createTypedArray(dtype, shape.reduce((acc, dim) => acc * dim, 1)),
      shape,
      { dtype }
    );
  }

  /**
   * Create typed array for data type
   * @private
   */
  _createTypedArray(dtype, size) {
    const typeMap = {
      'f32': Float32Array,
      'f64': Float64Array,
      'i8': Int8Array,
      'i16': Int16Array,
      'i32': Int32Array,
      'u8': Uint8Array,
      'u16': Uint16Array,
      'u32': Uint32Array
    };

    const TypedArrayClass = typeMap[dtype] || Float32Array;
    return new TypedArrayClass(size);
  }

  /**
   * Free tensor directly
   * @private
   */
  _freeTensorDirect(tensor) {
    if (tensor && typeof tensor.free === 'function') {
      tensor.free();
    } else if (tensor && typeof tensor.dispose === 'function') {
      tensor.dispose();
    }
  }

  /**
   * Calculate memory efficiency
   * @private
   */
  _calculateMemoryEfficiency() {
    const totalPossibleMemory = this.options.maxTotalMemory;
    const currentUsage = this.stats.currentMemoryUsage;
    const poolHitRate = this.stats.poolHits / (this.stats.poolHits + this.stats.poolMisses || 1);

    this.stats.memoryEfficiency = (poolHitRate * 0.7) + ((1 - currentUsage / totalPossibleMemory) * 0.3);
  }

  /**
   * Determine if garbage collection should be triggered
   * @private
   */
  _shouldTriggerGC() {
    const memoryUtilization = this.stats.currentMemoryUsage / this.options.maxTotalMemory;
    return memoryUtilization > this.options.gcThreshold;
  }

  /**
   * Trigger garbage collection
   * @private
   */
  _triggerGarbageCollection() {
    if (typeof window !== 'undefined' && window.gc) {
      window.gc();
    } else if (typeof global !== 'undefined' && global.gc) {
      global.gc();
    }
  }

  /**
   * Cleanup temporary allocations
   * @private
   */
  _cleanupTemporaryAllocations() {
    // Implementation would depend on tracking temporary allocations
    // This is a placeholder for the cleanup logic
  }

  /**
   * Cleanup memory manager
   */
  dispose() {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }

    this.performanceMonitor.stop();
    this.zeroCopyManager.dispose();
    this.memoryMapper.dispose();

    Object.values(this.pools).forEach(pool => pool.dispose());
  }
}

/**
 * Smart Memory Pool with advanced algorithms
 */
class SmartMemoryPool {
  constructor(options = {}) {
    this.maxSize = options.maxSize || 100 * 1024 * 1024; // 100MB default
    this.pools = new Map(); // shape+dtype -> tensors
    this.accessTimes = new Map(); // tensor -> last access time
    this.creationTimes = new Map(); // tensor -> creation time
    this.maxAge = options.maxAge || 300000; // 5 minutes
    this.currentSize = 0;

    this.stats = {
      hits: 0,
      misses: 0,
      evictions: 0,
      totalAllocated: 0
    };
  }

  /**
   * Acquire tensor from pool
   */
  acquire(shape, dtype, options = {}) {
    const key = this._getKey(shape, dtype);
    const pool = this.pools.get(key);

    if (pool && pool.length > 0) {
      const tensor = pool.pop();
      this.accessTimes.set(tensor, Date.now());
      this.stats.hits++;
      return tensor;
    }

    this.stats.misses++;
    return null;
  }

  /**
   * Release tensor to pool
   */
  release(tensor, options = {}) {
    if (!tensor) return;

    try {
      const shape = this._getTensorShape(tensor);
      const dtype = this._getTensorDtype(tensor);
      const key = this._getKey(shape, dtype);
      const tensorSize = this._calculateTensorSize(tensor);

      // Check if we have space
      if (this.currentSize + tensorSize > this.maxSize) {
        this._evictOldTensors(tensorSize);
      }

      if (this.currentSize + tensorSize <= this.maxSize) {
        const pool = this.pools.get(key) || [];
        pool.push(tensor);
        this.pools.set(key, pool);

        this.accessTimes.set(tensor, Date.now());
        this.creationTimes.set(tensor, Date.now());
        this.currentSize += tensorSize;
      }

    } catch (error) {
      console.warn('Smart pool release error:', error);
    }
  }

  /**
   * Cleanup expired tensors
   */
  cleanup() {
    const now = Date.now();
    const expired = [];

    for (const [tensor, creationTime] of this.creationTimes.entries()) {
      if (now - creationTime > this.maxAge) {
        expired.push(tensor);
      }
    }

    expired.forEach(tensor => this._removeTensor(tensor));
  }

  /**
   * Get pool statistics
   */
  getStatistics() {
    return {
      ...this.stats,
      currentSize: this.currentSize,
      maxSize: this.maxSize,
      poolCount: this.pools.size,
      tensorCount: this.creationTimes.size
    };
  }

  /**
   * Evict old tensors to make space
   * @private
   */
  _evictOldTensors(neededSpace) {
    const tensors = Array.from(this.accessTimes.entries())
      .sort(([, a], [, b]) => a - b); // Sort by access time (oldest first)

    let freedSpace = 0;
    for (const [tensor] of tensors) {
      if (freedSpace >= neededSpace) break;

      const tensorSize = this._calculateTensorSize(tensor);
      this._removeTensor(tensor);
      freedSpace += tensorSize;
      this.stats.evictions++;
    }
  }

  /**
   * Remove tensor from pool
   * @private
   */
  _removeTensor(tensor) {
    // Remove from all data structures
    this.accessTimes.delete(tensor);
    this.creationTimes.delete(tensor);

    // Remove from pools
    for (const [key, pool] of this.pools.entries()) {
      const index = pool.indexOf(tensor);
      if (index !== -1) {
        pool.splice(index, 1);
        if (pool.length === 0) {
          this.pools.delete(key);
        }
        break;
      }
    }

    // Free tensor
    if (tensor.free) {
      tensor.free();
    }
  }

  /**
   * Generate key for tensor
   * @private
   */
  _getKey(shape, dtype) {
    return `${shape.join('x')}_${dtype}`;
  }

  /**
   * Get tensor shape (helper)
   * @private
   */
  _getTensorShape(tensor) {
    if (tensor.shape) {
      return typeof tensor.shape === 'function' ? Array.from(tensor.shape()) : tensor.shape;
    }
    return [tensor.data?.length || 0];
  }

  /**
   * Get tensor dtype (helper)
   * @private
   */
  _getTensorDtype(tensor) {
    return tensor.dtype || 'f32';
  }

  /**
   * Calculate tensor memory size
   * @private
   */
  _calculateTensorSize(tensor) {
    const shape = this._getTensorShape(tensor);
    const dtype = this._getTensorDtype(tensor);
    const elements = shape.reduce((acc, dim) => acc * dim, 1);
    const bytesPerElement = dtype === 'f64' ? 8 : 4;
    return elements * bytesPerElement;
  }

  /**
   * Dispose pool
   */
  dispose() {
    for (const pool of this.pools.values()) {
      pool.forEach(tensor => {
        if (tensor.free) tensor.free();
      });
    }
    this.pools.clear();
    this.accessTimes.clear();
    this.creationTimes.clear();
  }
}

/**
 * Enhanced Zero-Copy Manager
 */
class EnhancedZeroCopyManager {
  constructor(options = {}) {
    this.options = options;
    this.wasmModule = null;
    this.sharedBuffers = new Map();
    this.stats = {
      allocations: 0,
      deallocations: 0,
      bytesAllocated: 0,
      bytesDeallocated: 0
    };
  }

  initialize(wasmModule) {
    this.wasmModule = wasmModule;
  }

  canAllocate(size) {
    // Check if zero-copy allocation is beneficial
    return size > 1024 && this.wasmModule !== null;
  }

  allocate(shape, dtype, options = {}) {
    if (!this.wasmModule) return null;

    try {
      const size = shape.reduce((acc, dim) => acc * dim, 1) * this._getBytesPerElement(dtype);

      // Use SharedArrayBuffer if available
      let buffer;
      if (typeof SharedArrayBuffer !== 'undefined') {
        buffer = new SharedArrayBuffer(size);
      } else {
        buffer = new ArrayBuffer(size);
      }

      const tensor = this._wrapBuffer(buffer, shape, dtype);
      tensor._zeroCopyManaged = true;

      this.stats.allocations++;
      this.stats.bytesAllocated += size;

      return tensor;

    } catch (error) {
      console.warn('Zero-copy allocation failed:', error);
      return null;
    }
  }

  deallocate(tensor, options = {}) {
    if (tensor._zeroCopyManaged) {
      const size = this._calculateSize(tensor);
      this.stats.deallocations++;
      this.stats.bytesDeallocated += size;
    }
  }

  getStatistics() {
    return { ...this.stats };
  }

  performMaintenance() {
    // Cleanup shared buffers
    this.sharedBuffers.clear();
  }

  _getBytesPerElement(dtype) {
    const typeMap = { 'f32': 4, 'f64': 8, 'i32': 4, 'u32': 4 };
    return typeMap[dtype] || 4;
  }

  _wrapBuffer(buffer, shape, dtype) {
    // Create view of buffer based on dtype
    const TypedArrayClass = dtype === 'f64' ? Float64Array : Float32Array;
    const data = new TypedArrayClass(buffer);

    return new SciRS2Tensor(data, shape, { dtype, _zeroCopyManaged: true });
  }

  _calculateSize(tensor) {
    const shape = tensor.shape || [tensor.data?.length || 0];
    const elements = shape.reduce((acc, dim) => acc * dim, 1);
    return elements * 4; // Simplified
  }

  dispose() {
    this.sharedBuffers.clear();
  }
}

/**
 * Memory Mapper for very large tensors
 */
class MemoryMapper {
  constructor() {
    this.mappedTensors = new Map();
    this.stats = { mappings: 0, unmappings: 0 };
  }

  initialize(wasmModule) {
    this.wasmModule = wasmModule;
  }

  allocate(shape, dtype, options = {}) {
    // Simplified memory mapping - would use actual file-backed mapping in production
    const tensor = new SciRS2Tensor(
      new Float32Array(shape.reduce((acc, dim) => acc * dim, 1)),
      shape,
      { dtype, _memoryMapped: true }
    );

    this.mappedTensors.set(tensor, { shape, dtype });
    this.stats.mappings++;

    return tensor;
  }

  deallocate(tensor, options = {}) {
    if (this.mappedTensors.has(tensor)) {
      this.mappedTensors.delete(tensor);
      this.stats.unmappings++;
    }
  }

  getStatistics() {
    return { ...this.stats };
  }

  cleanup() {
    // Cleanup expired mappings
  }

  dispose() {
    this.mappedTensors.clear();
  }
}

/**
 * Allocation Predictor using machine learning
 */
class AllocationPredictor {
  constructor() {
    this.patterns = new Map();
    this.predictions = new Map();
    this.stats = { predictions: 0, hits: 0 };
  }

  recordAllocation(shape, dtype, tensor) {
    const key = this._getPatternKey(shape, dtype);
    const pattern = this.patterns.get(key) || { count: 0, frequency: 0 };
    pattern.count++;
    pattern.frequency = Date.now();
    this.patterns.set(key, pattern);
  }

  recordDeallocation(shape, dtype) {
    // Update deallocation patterns
  }

  getPredictedTensor(shape, dtype) {
    const key = this._getPatternKey(shape, dtype);
    const pattern = this.patterns.get(key);

    if (pattern && pattern.count > 5) { // Threshold for prediction
      this.stats.predictions++;
      // Would implement actual prediction logic
      return null; // Placeholder
    }

    return null;
  }

  getStatistics() {
    return { ...this.stats };
  }

  _getPatternKey(shape, dtype) {
    return `${shape.join('x')}_${dtype}`;
  }
}

/**
 * Memory Defragmenter
 */
class MemoryDefragmenter {
  analyzeFragmentation(pools) {
    // Simplified fragmentation analysis
    let totalPools = 0;
    let emptyPools = 0;

    Object.values(pools).forEach(pool => {
      totalPools++;
      if (pool.currentSize === 0) {
        emptyPools++;
      }
    });

    const score = emptyPools / totalPools;
    return { score, totalPools, emptyPools };
  }

  async defragment(pools) {
    // Simplified defragmentation
    Object.values(pools).forEach(pool => {
      if (pool.currentSize === 0) {
        pool.cleanup();
      }
    });
  }
}

/**
 * Memory Performance Monitor
 */
class MemoryPerformanceMonitor {
  constructor(memoryManager) {
    this.memoryManager = memoryManager;
    this.metrics = {};
    this.isRunning = false;
    this.interval = null;
  }

  start() {
    if (this.isRunning) return;

    this.isRunning = true;
    this.interval = setInterval(() => {
      this._collectMetrics();
    }, 1000);
  }

  stop() {
    this.isRunning = false;
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  getMetrics() {
    return { ...this.metrics };
  }

  _collectMetrics() {
    this.metrics = {
      timestamp: Date.now(),
      memoryUsage: this.memoryManager.stats.currentMemoryUsage,
      allocationRate: this._calculateAllocationRate(),
      efficiency: this.memoryManager.stats.memoryEfficiency,
      poolPerformance: this._analyzePoolPerformance()
    };
  }

  _calculateAllocationRate() {
    // Calculate allocations per second
    return this.memoryManager.stats.totalAllocations /
           ((Date.now() - this.startTime || Date.now()) / 1000);
  }

  _analyzePoolPerformance() {
    const poolStats = Object.values(this.memoryManager.pools).map(pool => pool.getStatistics());
    return {
      averageHitRate: poolStats.reduce((sum, stat) => sum + (stat.hits / (stat.hits + stat.misses || 1)), 0) / poolStats.length,
      totalEvictions: poolStats.reduce((sum, stat) => sum + stat.evictions, 0)
    };
  }
}

// Export factory functions
export function createEnhancedMemoryManager(options = {}) {
  return new EnhancedMemoryManager(options);
}

export function withEnhancedMemoryManagement(operation, tensors = [], options = {}) {
  const memoryManager = createEnhancedMemoryManager();

  try {
    return operation();
  } finally {
    // Cleanup allocated tensors
    tensors.forEach(tensor => {
      if (tensor) {
        memoryManager.deallocateTensor(tensor, { forceRelease: options.autoRelease });
      }
    });
  }
}

export default {
  EnhancedMemoryManager,
  SmartMemoryPool,
  EnhancedZeroCopyManager,
  MemoryMapper,
  AllocationPredictor,
  MemoryDefragmenter,
  MemoryPerformanceMonitor,
  createEnhancedMemoryManager,
  withEnhancedMemoryManagement
};