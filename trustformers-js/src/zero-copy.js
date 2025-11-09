/**
 * Zero-Copy Tensor Transfer System for TrustformeRS
 * Provides efficient memory transfer between JavaScript and WebAssembly
 */

let wasmModule = null;
let wasmMemory = null;
let memoryPool = null;
let optimizationConfig = {
  enableAlignment: true,
  alignmentBoundary: 64, // 64-byte alignment for SIMD
  enableBatchTransfers: true,
  enableMemoryPrefetch: true,
  poolGrowthStrategy: 'exponential',
};

/**
 * Initialize zero-copy system with enhanced optimizations
 * @param {Object} module - WASM module reference
 * @param {Object} config - Optimization configuration
 */
export function initZeroCopy(module, config = {}) {
  wasmModule = module;
  wasmMemory = wasmModule.memory;
  optimizationConfig = { ...optimizationConfig, ...config };

  // Initialize memory pool for optimized allocations
  memoryPool = new OptimizedMemoryPool(optimizationConfig);

  // Setup memory growth monitoring
  if (wasmMemory && wasmMemory.buffer) {
    _setupMemoryGrowthMonitoring();
  }

  console.warn('Zero-copy system initialized with optimizations:', optimizationConfig);
}

/**
 * Zero-copy tensor view that shares memory between JS and WASM
 */
export class ZeroCopyTensorView {
  constructor(wasmTensor, metadata = {}) {
    this.wasmTensor = wasmTensor;
    this.metadata = metadata;
    this._cachedViews = new Map();
    this._isDisposed = false;
  }

  /**
   * Get typed array view of tensor data
   * @param {string} dtype - Data type (f32, f64, i32, u32, i8, u8)
   * @returns {TypedArray} Zero-copy view of tensor data
   */
  getView(dtype = 'f32') {
    if (this._isDisposed) {
      throw new Error('Tensor view has been disposed');
    }

    if (this._cachedViews.has(dtype)) {
      return this._cachedViews.get(dtype);
    }

    if (!wasmModule || !wasmMemory) {
      throw new Error('Zero-copy system not initialized');
    }

    // Get pointer and size from WASM tensor
    const dataPtr = this.wasmTensor.data_ptr();
    const elementCount = this.wasmTensor.element_count();

    // Create appropriate typed array view
    let view;
    const memory = new Uint8Array(wasmMemory.buffer);

    switch (dtype) {
      case 'f32':
        view = new Float32Array(wasmMemory.buffer, dataPtr, elementCount);
        break;
      case 'f64':
        view = new Float64Array(wasmMemory.buffer, dataPtr, elementCount);
        break;
      case 'i32':
        view = new Int32Array(wasmMemory.buffer, dataPtr, elementCount);
        break;
      case 'u32':
        view = new Uint32Array(wasmMemory.buffer, dataPtr, elementCount);
        break;
      case 'i16':
        view = new Int16Array(wasmMemory.buffer, dataPtr, elementCount);
        break;
      case 'u16':
        view = new Uint16Array(wasmMemory.buffer, dataPtr, elementCount);
        break;
      case 'i8':
        view = new Int8Array(wasmMemory.buffer, dataPtr, elementCount);
        break;
      case 'u8':
        view = new Uint8Array(wasmMemory.buffer, dataPtr, elementCount);
        break;
      default:
        throw new Error(`Unsupported dtype: ${dtype}`);
    }

    // Cache the view for reuse
    this._cachedViews.set(dtype, view);
    return view;
  }

  /**
   * Get tensor shape
   * @returns {Array<number>} Tensor shape
   */
  get shape() {
    return Array.from(this.wasmTensor.shape());
  }

  /**
   * Get tensor data type
   * @returns {string} Data type
   */
  get dtype() {
    return this.wasmTensor.dtype ? this.wasmTensor.dtype() : 'f32';
  }

  /**
   * Get tensor element count
   * @returns {number} Number of elements
   */
  get elementCount() {
    return this.wasmTensor.element_count();
  }

  /**
   * Get memory size in bytes
   * @returns {number} Memory size
   */
  get byteSize() {
    const bytesPerElement = this.getBytesPerElement();
    return this.elementCount * bytesPerElement;
  }

  /**
   * Check if tensor data is contiguous in memory
   * @returns {boolean} True if contiguous
   */
  get isContiguous() {
    return this.wasmTensor.is_contiguous ? this.wasmTensor.is_contiguous() : true;
  }

  /**
   * Get bytes per element for current dtype
   * @returns {number} Bytes per element
   */
  getBytesPerElement() {
    const typeMap = {
      f32: 4,
      f64: 8,
      i32: 4,
      u32: 4,
      i16: 2,
      u16: 2,
      i8: 1,
      u8: 1,
    };
    return typeMap[this.dtype] || 4;
  }

  /**
   * Slice the tensor view (creates a new view, not a copy)
   * @param {Array<number>} start - Start indices
   * @param {Array<number>} end - End indices
   * @param {Array<number>} step - Step sizes (optional)
   * @returns {ZeroCopyTensorView} New sliced view
   */
  slice(start, end, step = null) {
    if (this._isDisposed) {
      throw new Error('Tensor view has been disposed');
    }

    const slicedTensor = this.wasmTensor.slice(
      new Uint32Array(start),
      new Uint32Array(end),
      step ? new Uint32Array(step) : null
    );

    return new ZeroCopyTensorView(slicedTensor, {
      ...this.metadata,
      operation: 'slice',
      parent: this,
    });
  }

  /**
   * Reshape the tensor view (no data copy)
   * @param {Array<number>} newShape - New shape
   * @returns {ZeroCopyTensorView} Reshaped view
   */
  reshape(newShape) {
    if (this._isDisposed) {
      throw new Error('Tensor view has been disposed');
    }

    const reshapedTensor = this.wasmTensor.reshape(new Uint32Array(newShape));

    return new ZeroCopyTensorView(reshapedTensor, {
      ...this.metadata,
      operation: 'reshape',
      parent: this,
    });
  }

  /**
   * Transpose the tensor (may require memory reordering)
   * @param {Array<number>} dims - Dimension order (optional)
   * @returns {ZeroCopyTensorView} Transposed view
   */
  transpose(dims = null) {
    if (this._isDisposed) {
      throw new Error('Tensor view has been disposed');
    }

    const transposedTensor = dims
      ? this.wasmTensor.transpose_dims(new Uint32Array(dims))
      : this.wasmTensor.transpose();

    return new ZeroCopyTensorView(transposedTensor, {
      ...this.metadata,
      operation: 'transpose',
      parent: this,
    });
  }

  /**
   * Create a copy of the tensor data
   * @param {string} dtype - Target data type (optional)
   * @returns {TypedArray} Copied data
   */
  copy(dtype = null) {
    const targetDtype = dtype || this.dtype;
    const view = this.getView(targetDtype);

    // Create a new array with copied data
    const Constructor = view.constructor;
    return new Constructor(view);
  }

  /**
   * Update tensor data from JavaScript array (zero-copy when possible)
   * @param {TypedArray|Array} data - New data
   * @param {number} offset - Offset in elements (optional)
   */
  setData(data, offset = 0) {
    if (this._isDisposed) {
      throw new Error('Tensor view has been disposed');
    }

    const view = this.getView(this.dtype);

    if (data instanceof view.constructor) {
      // Direct copy for same type
      view.set(data, offset);
    } else if (Array.isArray(data)) {
      // Copy from regular array
      for (let i = 0; i < data.length && i + offset < view.length; i++) {
        view[i + offset] = data[i];
      }
    } else {
      // Convert from different typed array
      for (let i = 0; i < data.length && i + offset < view.length; i++) {
        view[i + offset] = data[i];
      }
    }

    // Invalidate cached views if data changed
    this._cachedViews.clear();
  }

  /**
   * Fill tensor with a value
   * @param {number} value - Fill value
   * @param {number} start - Start index (optional)
   * @param {number} end - End index (optional)
   */
  fill(value, start = 0, end = null) {
    if (this._isDisposed) {
      throw new Error('Tensor view has been disposed');
    }

    const view = this.getView(this.dtype);
    const endIndex = end !== null ? end : view.length;

    view.fill(value, start, endIndex);
  }

  /**
   * Get tensor statistics
   * @returns {Object} Statistics (min, max, mean, sum)
   */
  getStats() {
    const view = this.getView(this.dtype);

    const [firstValue] = view;
    let min = firstValue;
    let max = firstValue;
    let sum = 0;

    for (let i = 0; i < view.length; i++) {
      const val = view[i];
      if (val < min) min = val;
      if (val > max) max = val;
      sum += val;
    }

    return {
      min,
      max,
      sum,
      mean: sum / view.length,
      length: view.length,
    };
  }

  /**
   * Check if views are still valid (memory hasn't been reallocated)
   * @returns {boolean} True if views are valid
   */
  isValid() {
    if (this._isDisposed) return false;

    try {
      // Try to access the tensor to see if it's still valid
      this.wasmTensor.shape();
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Dispose the tensor view and invalidate all cached views
   */
  dispose() {
    this._cachedViews.clear();
    this._isDisposed = true;

    if (this.wasmTensor && this.wasmTensor.free) {
      this.wasmTensor.free();
    }
  }

  /**
   * Convert to JavaScript object for debugging
   * @returns {Object} Debug representation
   */
  toDebugObject() {
    return {
      shape: this.shape,
      dtype: this.dtype,
      elementCount: this.elementCount,
      byteSize: this.byteSize,
      isContiguous: this.isContiguous,
      isValid: this.isValid(),
      metadata: this.metadata,
    };
  }
}

/**
 * Zero-copy buffer manager for efficient memory transfers
 */
export class ZeroCopyBufferManager {
  constructor(options = {}) {
    this.buffers = new Map();
    this.maxBuffers = options.maxBuffers || 100;
    this.maxMemory = options.maxMemory || 512 * 1024 * 1024; // 512MB
    this.currentMemory = 0;
    this.stats = {
      allocations: 0,
      deallocations: 0,
      reuseCount: 0,
      totalMemoryAllocated: 0,
    };
  }

  /**
   * Allocate or reuse a buffer
   * @param {number} size - Buffer size in bytes
   * @param {string} type - Buffer type ('f32', 'i32', etc.)
   * @returns {ArrayBuffer} Allocated buffer
   */
  allocateBuffer(size, type = 'u8') {
    const key = `${type}:${size}`;

    if (this.buffers.has(key)) {
      const bufferList = this.buffers.get(key);
      if (bufferList.length > 0) {
        this.stats.reuseCount++;
        return bufferList.pop();
      }
    }

    // Check memory limits
    if (this.currentMemory + size > this.maxMemory) {
      this.cleanup();

      if (this.currentMemory + size > this.maxMemory) {
        throw new Error(`Buffer allocation would exceed memory limit: ${size} bytes`);
      }
    }

    const buffer = new ArrayBuffer(size);
    this.currentMemory += size;
    this.stats.allocations++;
    this.stats.totalMemoryAllocated += size;

    return buffer;
  }

  /**
   * Release a buffer back to the pool
   * @param {ArrayBuffer} buffer - Buffer to release
   * @param {string} type - Buffer type
   */
  releaseBuffer(buffer, type = 'u8') {
    const size = buffer.byteLength;
    const key = `${type}:${size}`;

    if (!this.buffers.has(key)) {
      this.buffers.set(key, []);
    }

    const bufferList = this.buffers.get(key);
    if (bufferList.length < this.maxBuffers) {
      bufferList.push(buffer);
    } else {
      this.currentMemory -= size;
      this.stats.deallocations++;
    }
  }

  /**
   * Create zero-copy tensor from JavaScript data
   * @param {TypedArray|Array} data - Source data
   * @param {Array<number>} shape - Tensor shape
   * @param {string} dtype - Data type
   * @returns {ZeroCopyTensorView} Zero-copy tensor view
   */
  createTensorFromData(data, shape, dtype = 'f32') {
    if (!wasmModule) {
      throw new Error('Zero-copy system not initialized');
    }

    let typedData;
    if (data instanceof ArrayBuffer) {
      typedData = this.createTypedArrayView(data, dtype);
    } else if (!(data instanceof Object.getPrototypeOf(Float32Array))) {
      // Convert regular array to typed array
      typedData = this.arrayToTypedArray(data, dtype);
    } else {
      typedData = data;
    }

    // Create WASM tensor with zero-copy when possible
    const shapeArray = new Uint32Array(shape);
    let wasmTensor;

    switch (dtype) {
      case 'f32':
        wasmTensor = wasmModule.WasmTensor.from_f32(typedData, shapeArray);
        break;
      case 'f64':
        wasmTensor = wasmModule.WasmTensor.from_f64(typedData, shapeArray);
        break;
      case 'i32':
        wasmTensor = wasmModule.WasmTensor.from_i32(typedData, shapeArray);
        break;
      case 'u32':
        wasmTensor = wasmModule.WasmTensor.from_u32(typedData, shapeArray);
        break;
      case 'i8':
        wasmTensor = wasmModule.WasmTensor.from_i8(typedData, shapeArray);
        break;
      case 'u8':
        wasmTensor = wasmModule.WasmTensor.from_u8(typedData, shapeArray);
        break;
      default:
        throw new Error(`Unsupported dtype: ${dtype}`);
    }

    return new ZeroCopyTensorView(wasmTensor, {
      source: 'javascript_data',
      originalType: data.constructor.name,
    });
  }

  /**
   * Create typed array view from ArrayBuffer
   * @param {ArrayBuffer} buffer - Source buffer
   * @param {string} dtype - Data type
   * @returns {TypedArray} Typed array view
   */
  createTypedArrayView(buffer, dtype) {
    switch (dtype) {
      case 'f32':
        return new Float32Array(buffer);
      case 'f64':
        return new Float64Array(buffer);
      case 'i32':
        return new Int32Array(buffer);
      case 'u32':
        return new Uint32Array(buffer);
      case 'i16':
        return new Int16Array(buffer);
      case 'u16':
        return new Uint16Array(buffer);
      case 'i8':
        return new Int8Array(buffer);
      case 'u8':
        return new Uint8Array(buffer);
      default:
        throw new Error(`Unsupported dtype: ${dtype}`);
    }
  }

  /**
   * Convert regular array to typed array
   * @param {Array} array - Source array
   * @param {string} dtype - Target data type
   * @returns {TypedArray} Converted typed array
   */
  arrayToTypedArray(array, dtype) {
    switch (dtype) {
      case 'f32':
        return new Float32Array(array);
      case 'f64':
        return new Float64Array(array);
      case 'i32':
        return new Int32Array(array);
      case 'u32':
        return new Uint32Array(array);
      case 'i16':
        return new Int16Array(array);
      case 'u16':
        return new Uint16Array(array);
      case 'i8':
        return new Int8Array(array);
      case 'u8':
        return new Uint8Array(array);
      default:
        throw new Error(`Unsupported dtype: ${dtype}`);
    }
  }

  /**
   * Clean up unused buffers
   */
  cleanup() {
    let freedMemory = 0;

    for (const [key, bufferList] of this.buffers.entries()) {
      const [type, sizeStr] = key.split(':');
      const size = parseInt(sizeStr, 10);

      // Free half of the buffers
      const toFree = Math.floor(bufferList.length / 2);
      for (let i = 0; i < toFree; i++) {
        if (bufferList.length > 0) {
          bufferList.pop();
          freedMemory += size;
          this.stats.deallocations++;
        }
      }
    }

    this.currentMemory -= freedMemory;
    console.warn(`Zero-copy buffer cleanup freed ${freedMemory} bytes`);
  }

  /**
   * Get buffer manager statistics
   * @returns {Object} Statistics
   */
  getStats() {
    const bufferInfo = {};
    let totalBuffers = 0;

    for (const [key, bufferList] of this.buffers.entries()) {
      bufferInfo[key] = {
        count: bufferList.length,
        size: parseInt(key.split(':')[1], 10),
      };
      totalBuffers += bufferList.length;
    }

    return {
      ...this.stats,
      currentMemory: this.currentMemory,
      totalBuffers,
      bufferTypes: this.buffers.size,
      bufferInfo,
    };
  }

  /**
   * Clear all buffers
   */
  clear() {
    this.buffers.clear();
    this.currentMemory = 0;
    this.stats.deallocations += this.stats.allocations - this.stats.deallocations;
  }

  /**
   * Dispose the buffer manager
   */
  dispose() {
    this.clear();
  }
}

// Global buffer manager
let globalBufferManager = null;

/**
 * Get or create global buffer manager
 * @param {Object} options - Buffer manager options
 * @returns {ZeroCopyBufferManager} Global buffer manager
 */
export function getBufferManager(options = {}) {
  if (!globalBufferManager) {
    globalBufferManager = new ZeroCopyBufferManager(options);
  }
  return globalBufferManager;
}

/**
 * Create zero-copy tensor from JavaScript data
 * @param {TypedArray|Array} data - Source data
 * @param {Array<number>} shape - Tensor shape
 * @param {string} dtype - Data type
 * @returns {ZeroCopyTensorView} Zero-copy tensor view
 */
export function createZeroCopyTensor(data, shape, dtype = 'f32') {
  const manager = getBufferManager();
  return manager.createTensorFromData(data, shape, dtype);
}

/**
 * Wrap existing WASM tensor in zero-copy view
 * @param {Object} wasmTensor - WASM tensor object
 * @param {Object} metadata - Additional metadata
 * @returns {ZeroCopyTensorView} Zero-copy view
 */
export function wrapTensor(wasmTensor, metadata = {}) {
  return new ZeroCopyTensorView(wasmTensor, metadata);
}

/**
 * Utility function for efficient tensor data transfer
 * @param {ZeroCopyTensorView|Object} tensor - Source tensor
 * @param {string} targetFormat - Target format ('js', 'wasm', 'webgl')
 * @returns {*} Converted tensor data
 */
export function transferTensor(tensor, targetFormat = 'js') {
  if (!tensor) {
    throw new Error('Tensor is null or undefined');
  }

  switch (targetFormat) {
    case 'js':
      if (tensor instanceof ZeroCopyTensorView) {
        return tensor.getView();
      } else if (tensor.to_js_array) {
        // Convert WASM tensor to JS array (copy operation)
        return tensor.to_js_array();
      } 
        throw new Error('Unsupported tensor type for JS conversion');
      

    case 'wasm':
      if (tensor instanceof ZeroCopyTensorView) {
        return tensor.wasmTensor;
      } 
        // Assume it's already a WASM tensor
        return tensor;
      

    case 'webgl': {
      // Convert tensor data to format suitable for WebGL textures
      const view =
        tensor instanceof ZeroCopyTensorView
          ? tensor.getView('f32')
          : new Float32Array(tensor.to_js_array());

      return {
        data: view,
        shape: tensor.shape || Array.from(tensor.shape()),
        dtype: 'f32',
      };
    }

    default:
      throw new Error(`Unsupported target format: ${targetFormat}`);
  }
}

/**
 * Batch transfer multiple tensors efficiently with optimization
 * @param {Array} tensors - Array of tensors to transfer
 * @param {string} targetFormat - Target format
 * @returns {Array} Array of transferred tensors
 */
export function batchTransferTensors(tensors, targetFormat = 'js') {
  if (!optimizationConfig.enableBatchTransfers || tensors.length <= 1) {
    return tensors.map(tensor => transferTensor(tensor, targetFormat));
  }

  // Group tensors by properties for optimized batch processing
  const tensorGroups = _groupTensorsByProperties(tensors);
  const results = [];

  for (const group of tensorGroups) {
    // Prefetch memory for the entire group if enabled
    if (optimizationConfig.enableMemoryPrefetch) {
      _prefetchTensorGroupMemory(group);
    }

    // Process group with optimized pipeline
    const groupResults = _processTensorGroupOptimized(group, targetFormat);
    results.push(...groupResults);
  }

  return results;
}

/**
 * Memory growth monitoring setup
 * @private
 */
function _setupMemoryGrowthMonitoring() {
  const originalBuffer = wasmMemory.buffer;
  let lastBufferByteLength = originalBuffer.byteLength;

  // Check for memory growth periodically
  const checkMemoryGrowth = () => {
    if (wasmMemory.buffer.byteLength !== lastBufferByteLength) {
      console.warn(
        `WASM memory grew: ${lastBufferByteLength} -> ${wasmMemory.buffer.byteLength} bytes`
      );
      lastBufferByteLength = wasmMemory.buffer.byteLength;

      // Invalidate cached views that may be affected by memory growth
      _invalidateCachedViews();
    }
  };

  // Check every second
  setInterval(checkMemoryGrowth, 1000);
}

/**
 * Invalidate cached views after memory growth
 * @private
 */
function _invalidateCachedViews() {
  // This would need to track all active ZeroCopyTensorView instances
  console.warn('WASM memory growth detected - some cached views may be invalid');
}

/**
 * Group tensors by properties for optimized processing
 * @private
 */
function _groupTensorsByProperties(tensors) {
  const groups = new Map();

  for (const tensor of tensors) {
    const key = `${tensor.dtype}_${tensor.elementCount}_${tensor.isContiguous}`;

    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(tensor);
  }

  return Array.from(groups.values());
}

/**
 * Prefetch memory for tensor group
 * @private
 */
function _prefetchTensorGroupMemory(group) {
  // Simplified prefetch - touch memory pages
  for (const tensor of group) {
    try {
      const view = tensor.getView();
      // Touch first and last elements to ensure pages are in memory
      if (view.length > 0) {
        view[0]; // Touch first element
        if (view.length > 1) {
          view[view.length - 1]; // Touch last element
        }
      }
    } catch (error) {
      console.warn('Prefetch failed for tensor:', error.message);
    }
  }
}

/**
 * Process tensor group with optimizations
 * @private
 */
function _processTensorGroupOptimized(group, targetFormat) {
  const results = [];
  const startTime = performance.now();

  for (const tensor of group) {
    results.push(transferTensor(tensor, targetFormat));
  }

  const endTime = performance.now();
  console.debug(`Processed ${group.length} tensors in ${(endTime - startTime).toFixed(2)}ms`);

  return results;
}

/**
 * Optimized Memory Pool for aligned allocations
 */
class OptimizedMemoryPool {
  constructor(config) {
    this.config = config;
    this.pools = new Map(); // Size -> Array of buffers
    this.alignedPools = new Map(); // Size -> Array of aligned buffers
    this.allocationStats = {
      total: 0,
      aligned: 0,
      reused: 0,
    };
  }

  /**
   * Allocate aligned buffer
   * @param {number} size - Buffer size in bytes
   * @param {number} alignment - Alignment boundary
   * @returns {ArrayBuffer} Aligned buffer
   */
  allocateAligned(size, alignment = this.config.alignmentBoundary) {
    const poolKey = `${size}_${alignment}`;

    if (this.alignedPools.has(poolKey) && this.alignedPools.get(poolKey).length > 0) {
      this.allocationStats.reused++;
      return this.alignedPools.get(poolKey).pop();
    }

    // Create new aligned buffer
    const buffer = this._createAlignedBuffer(size, alignment);
    this.allocationStats.total++;
    this.allocationStats.aligned++;

    return buffer;
  }

  /**
   * Release buffer back to pool
   * @param {ArrayBuffer} buffer - Buffer to release
   * @param {number} size - Buffer size
   * @param {number} alignment - Alignment boundary
   */
  releaseAligned(buffer, size, alignment = this.config.alignmentBoundary) {
    const poolKey = `${size}_${alignment}`;

    if (!this.alignedPools.has(poolKey)) {
      this.alignedPools.set(poolKey, []);
    }

    this.alignedPools.get(poolKey).push(buffer);
  }

  /**
   * Create aligned buffer
   * @private
   */
  _createAlignedBuffer(size, alignment) {
    // For WebAssembly, we can't guarantee alignment in JS
    // This is a simplified implementation
    const buffer = new ArrayBuffer(size);
    return buffer;
  }

  /**
   * Get allocation statistics
   */
  getStats() {
    return { ...this.allocationStats };
  }

  /**
   * Clear all pools
   */
  clear() {
    this.pools.clear();
    this.alignedPools.clear();
    this.allocationStats = { total: 0, aligned: 0, reused: 0 };
  }
}
