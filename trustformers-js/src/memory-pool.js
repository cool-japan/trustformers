/**
 * Memory Pool Manager for TrustformeRS
 * Provides efficient memory management and pooling strategies
 */

let wasmModule = null;

/**
 * Initialize memory pool with WASM module
 * @param {Object} module - WASM module reference
 */
export function initMemoryPool(module) {
  wasmModule = module;
}

/**
 * Memory pool for tensor operations
 */
export class TensorMemoryPool {
  constructor(options = {}) {
    this.pools = new Map(); // shape -> [tensor1, tensor2, ...]
    this.maxPoolSize = options.maxPoolSize || 100;
    this.maxTotalMemory = options.maxTotalMemory || 1024 * 1024 * 1024; // 1GB default
    this.currentMemory = 0;
    this.allocatedTensors = new WeakSet();
    this.stats = {
      allocations: 0,
      deallocations: 0,
      poolHits: 0,
      poolMisses: 0,
      totalMemoryAllocated: 0,
      currentMemoryUsage: 0
    };

    // Cleanup timer
    this.cleanupInterval = setInterval(() => this.cleanup(), 30000); // 30 seconds
  }

  /**
   * Get a tensor from the pool or create a new one
   * @param {number[]} shape - Tensor shape
   * @param {string} dtype - Data type (f32, f64, i32, etc.)
   * @param {boolean} zero - Whether to zero the tensor
   * @returns {Object} Tensor from pool or new tensor
   */
  acquire(shape, dtype = 'f32', zero = true) {
    if (!wasmModule) {
      throw new Error('Memory pool not initialized');
    }

    const key = this.getShapeKey(shape, dtype);
    const pool = this.pools.get(key) || [];

    let tensor;
    if (pool.length > 0) {
      tensor = pool.pop();
      this.stats.poolHits++;
      
      if (zero) {
        this.zeroTensor(tensor);
      }
    } else {
      tensor = this.createTensor(shape, dtype);
      this.stats.poolMisses++;
    }

    this.allocatedTensors.add(tensor);
    this.stats.allocations++;
    
    return tensor;
  }

  /**
   * Return a tensor to the pool
   * @param {Object} tensor - Tensor to return
   */
  release(tensor) {
    if (!tensor || !this.allocatedTensors.has(tensor)) {
      return;
    }

    try {
      const shape = tensor.shape();
      const dtype = tensor.dtype ? tensor.dtype() : 'f32';
      const key = this.getShapeKey(Array.from(shape), dtype);
      
      const pool = this.pools.get(key) || [];
      
      if (pool.length < this.maxPoolSize) {
        pool.push(tensor);
        this.pools.set(key, pool);
      } else {
        // Pool is full, free the tensor
        this.freeTensor(tensor);
      }

      this.allocatedTensors.delete(tensor);
      this.stats.deallocations++;
    } catch (error) {
      console.warn('Error releasing tensor to pool:', error);
      this.freeTensor(tensor);
    }
  }

  /**
   * Create a new tensor
   * @param {number[]} shape - Tensor shape
   * @param {string} dtype - Data type
   * @returns {Object} New tensor
   */
  createTensor(shape, dtype) {
    const tensorSize = shape.reduce((a, b) => a * b, 1);
    const bytesPerElement = this.getBytesPerElement(dtype);
    const memoryRequired = tensorSize * bytesPerElement;

    // Check memory limits
    if (this.currentMemory + memoryRequired > this.maxTotalMemory) {
      this.performGarbageCollection();
      
      if (this.currentMemory + memoryRequired > this.maxTotalMemory) {
        throw new Error(`Memory limit exceeded: ${this.currentMemory + memoryRequired} > ${this.maxTotalMemory}`);
      }
    }

    let tensor;
    const shapeArray = new Uint32Array(shape);

    switch (dtype) {
      case 'f32':
        tensor = wasmModule.WasmTensor.zeros(shapeArray);
        break;
      case 'f64':
        tensor = wasmModule.WasmTensor.zeros_f64(shapeArray);
        break;
      case 'i32':
        tensor = wasmModule.WasmTensor.zeros_i32(shapeArray);
        break;
      case 'u32':
        tensor = wasmModule.WasmTensor.zeros_u32(shapeArray);
        break;
      case 'i8':
        tensor = wasmModule.WasmTensor.zeros_i8(shapeArray);
        break;
      case 'u8':
        tensor = wasmModule.WasmTensor.zeros_u8(shapeArray);
        break;
      default:
        throw new Error(`Unsupported dtype: ${dtype}`);
    }

    this.currentMemory += memoryRequired;
    this.stats.totalMemoryAllocated += memoryRequired;
    this.stats.currentMemoryUsage = this.currentMemory;

    return tensor;
  }

  /**
   * Zero out a tensor
   * @param {Object} tensor - Tensor to zero
   */
  zeroTensor(tensor) {
    if (tensor.zero_) {
      tensor.zero_();
    } else if (tensor.fill_) {
      tensor.fill_(0);
    }
  }

  /**
   * Free a tensor
   * @param {Object} tensor - Tensor to free
   */
  freeTensor(tensor) {
    try {
      if (tensor.shape) {
        const shape = Array.from(tensor.shape());
        const dtype = tensor.dtype ? tensor.dtype() : 'f32';
        const tensorSize = shape.reduce((a, b) => a * b, 1);
        const bytesPerElement = this.getBytesPerElement(dtype);
        const memoryFreed = tensorSize * bytesPerElement;
        this.currentMemory = Math.max(0, this.currentMemory - memoryFreed);
        this.stats.currentMemoryUsage = this.currentMemory;
      }

      if (tensor.free) {
        tensor.free();
      }
    } catch (error) {
      console.warn('Error freeing tensor:', error);
    }
  }

  /**
   * Get memory key for shape and dtype
   * @param {number[]} shape - Tensor shape
   * @param {string} dtype - Data type
   * @returns {string} Shape key
   */
  getShapeKey(shape, dtype) {
    return `${dtype}:${shape.join('x')}`;
  }

  /**
   * Get bytes per element for data type
   * @param {string} dtype - Data type
   * @returns {number} Bytes per element
   */
  getBytesPerElement(dtype) {
    const typeMap = {
      'f32': 4,
      'f64': 8,
      'i32': 4,
      'u32': 4,
      'i16': 2,
      'u16': 2,
      'i8': 1,
      'u8': 1
    };
    return typeMap[dtype] || 4;
  }

  /**
   * Perform garbage collection
   */
  performGarbageCollection() {
    console.warn('Performing memory pool garbage collection...');
    
    const beforeMemory = this.currentMemory;
    let freedMemory = 0;

    // Free half of each pool
    for (const [key, pool] of this.pools.entries()) {
      const toFree = Math.floor(pool.length / 2);
      for (let i = 0; i < toFree; i++) {
        const tensor = pool.pop();
        if (tensor) {
          this.freeTensor(tensor);
          freedMemory += this.getShapeMemory(key);
        }
      }
    }

    console.warn(`GC freed ${freedMemory} bytes (${beforeMemory} -> ${this.currentMemory})`);
  }

  /**
   * Get memory usage for a shape key
   * @param {string} key - Shape key
   * @returns {number} Memory usage in bytes
   */
  getShapeMemory(key) {
    const [dtype, shapeStr] = key.split(':');
    const shape = shapeStr.split('x').map(Number);
    const tensorSize = shape.reduce((a, b) => a * b, 1);
    return tensorSize * this.getBytesPerElement(dtype);
  }

  /**
   * Clean up empty pools and old tensors
   */
  cleanup() {
    // Remove empty pools
    for (const [key, pool] of this.pools.entries()) {
      if (pool.length === 0) {
        this.pools.delete(key);
      }
    }

    // Optionally trigger GC if memory usage is high
    if (this.currentMemory > this.maxTotalMemory * 0.8) {
      this.performGarbageCollection();
    }
  }

  /**
   * Get memory pool statistics
   * @returns {Object} Pool statistics
   */
  getStats() {
    const poolInfo = {};
    let totalPooledTensors = 0;

    for (const [key, pool] of this.pools.entries()) {
      poolInfo[key] = {
        count: pool.length,
        memoryPerTensor: this.getShapeMemory(key),
        totalMemory: pool.length * this.getShapeMemory(key)
      };
      totalPooledTensors += pool.length;
    }

    return {
      ...this.stats,
      totalPooledTensors,
      poolInfo,
      memoryEfficiency: this.stats.allocations > 0 ? this.stats.poolHits / this.stats.allocations : 0
    };
  }

  /**
   * Clear all pools and free memory
   */
  clear() {
    for (const [key, pool] of this.pools.entries()) {
      for (const tensor of pool) {
        this.freeTensor(tensor);
      }
    }
    this.pools.clear();
    this.currentMemory = 0;
    this.stats.currentMemoryUsage = 0;
  }

  /**
   * Dispose the memory pool
   */
  dispose() {
    this.clear();
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
  }
}

/**
 * WebGL Memory Pool for WebGL textures and buffers
 */
export class WebGLMemoryPool {
  constructor(gl, options = {}) {
    this.gl = gl;
    this.texturePool = new Map(); // "width:height:format" -> [texture1, texture2, ...]
    this.bufferPool = new Map(); // "size:usage" -> [buffer1, buffer2, ...]
    this.framebufferPool = [];
    this.maxPoolSize = options.maxPoolSize || 50;
    this.stats = {
      textureAllocations: 0,
      bufferAllocations: 0,
      texturePoolHits: 0,
      bufferPoolHits: 0
    };
  }

  /**
   * Acquire a texture from the pool
   * @param {number} width - Texture width
   * @param {number} height - Texture height
   * @param {number} format - Texture format
   * @param {number} type - Texture type
   * @returns {WebGLTexture} Texture from pool or new texture
   */
  acquireTexture(width, height, format = null, type = null) {
    format = format || this.gl.RGBA;
    type = type || this.gl.FLOAT;
    
    const key = `${width}:${height}:${format}:${type}`;
    const pool = this.texturePool.get(key) || [];

    let texture;
    if (pool.length > 0) {
      texture = pool.pop();
      this.stats.texturePoolHits++;
    } else {
      texture = this.createTexture(width, height, format, type);
      this.stats.textureAllocations++;
    }

    return texture;
  }

  /**
   * Release a texture back to the pool
   * @param {WebGLTexture} texture - Texture to release
   * @param {number} width - Texture width
   * @param {number} height - Texture height
   * @param {number} format - Texture format
   * @param {number} type - Texture type
   */
  releaseTexture(texture, width, height, format = null, type = null) {
    format = format || this.gl.RGBA;
    type = type || this.gl.FLOAT;
    
    const key = `${width}:${height}:${format}:${type}`;
    const pool = this.texturePool.get(key) || [];

    if (pool.length < this.maxPoolSize) {
      pool.push(texture);
      this.texturePool.set(key, pool);
    } else {
      this.gl.deleteTexture(texture);
    }
  }

  /**
   * Create a new texture
   * @param {number} width - Texture width
   * @param {number} height - Texture height
   * @param {number} format - Texture format
   * @param {number} type - Texture type
   * @returns {WebGLTexture} New texture
   */
  createTexture(width, height, format, type) {
    const texture = this.gl.createTexture();
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
    
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, format, width, height, 0, format, type, null);
    
    return texture;
  }

  /**
   * Acquire a buffer from the pool
   * @param {number} size - Buffer size
   * @param {number} usage - Buffer usage
   * @returns {WebGLBuffer} Buffer from pool or new buffer
   */
  acquireBuffer(size, usage = null) {
    usage = usage || this.gl.STATIC_DRAW;
    
    const key = `${size}:${usage}`;
    const pool = this.bufferPool.get(key) || [];

    let buffer;
    if (pool.length > 0) {
      buffer = pool.pop();
      this.stats.bufferPoolHits++;
    } else {
      buffer = this.createBuffer(size, usage);
      this.stats.bufferAllocations++;
    }

    return buffer;
  }

  /**
   * Release a buffer back to the pool
   * @param {WebGLBuffer} buffer - Buffer to release
   * @param {number} size - Buffer size
   * @param {number} usage - Buffer usage
   */
  releaseBuffer(buffer, size, usage = null) {
    usage = usage || this.gl.STATIC_DRAW;
    
    const key = `${size}:${usage}`;
    const pool = this.bufferPool.get(key) || [];

    if (pool.length < this.maxPoolSize) {
      pool.push(buffer);
      this.bufferPool.set(key, pool);
    } else {
      this.gl.deleteBuffer(buffer);
    }
  }

  /**
   * Create a new buffer
   * @param {number} size - Buffer size
   * @param {number} usage - Buffer usage
   * @returns {WebGLBuffer} New buffer
   */
  createBuffer(size, usage) {
    const buffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, size, usage);
    return buffer;
  }

  /**
   * Get pool statistics
   * @returns {Object} Pool statistics
   */
  getStats() {
    const textureCount = Array.from(this.texturePool.values()).reduce((sum, pool) => sum + pool.length, 0);
    const bufferCount = Array.from(this.bufferPool.values()).reduce((sum, pool) => sum + pool.length, 0);

    return {
      ...this.stats,
      pooledTextures: textureCount,
      pooledBuffers: bufferCount,
      textureTypes: this.texturePool.size,
      bufferTypes: this.bufferPool.size
    };
  }

  /**
   * Clear all pools
   */
  clear() {
    // Clear texture pools
    for (const pool of this.texturePool.values()) {
      for (const texture of pool) {
        this.gl.deleteTexture(texture);
      }
    }
    this.texturePool.clear();

    // Clear buffer pools
    for (const pool of this.bufferPool.values()) {
      for (const buffer of pool) {
        this.gl.deleteBuffer(buffer);
      }
    }
    this.bufferPool.clear();

    // Clear framebuffer pool
    for (const framebuffer of this.framebufferPool) {
      this.gl.deleteFramebuffer(framebuffer);
    }
    this.framebufferPool = [];
  }
}

/**
 * Global memory manager
 */
export class MemoryManager {
  constructor(options = {}) {
    this.tensorPool = new TensorMemoryPool(options.tensor);
    this.webglPool = null;
    this.autoCleanup = options.autoCleanup !== false;
    this.cleanupThreshold = options.cleanupThreshold || 0.8;
    
    if (this.autoCleanup) {
      this.startAutoCleanup();
    }
  }

  /**
   * Initialize WebGL memory pool
   * @param {WebGLRenderingContext} gl - WebGL context
   * @param {Object} options - WebGL pool options
   */
  initWebGL(gl, options = {}) {
    this.webglPool = new WebGLMemoryPool(gl, options);
  }

  /**
   * Start automatic cleanup
   */
  startAutoCleanup() {
    this.cleanupInterval = setInterval(() => {
      const stats = this.tensorPool.getStats();
      const memoryUsage = stats.currentMemoryUsage / (1024 * 1024 * 1024); // GB
      
      if (memoryUsage > this.cleanupThreshold) {
        console.warn(`Auto-cleanup triggered at ${memoryUsage.toFixed(2)}GB usage`);
        this.cleanup();
      }
    }, 10000); // Check every 10 seconds
  }

  /**
   * Perform memory cleanup
   */
  cleanup() {
    this.tensorPool.cleanup();
    if (this.webglPool) {
      // WebGL pools don't need explicit cleanup as they auto-manage
    }
  }

  /**
   * Get comprehensive memory statistics
   * @returns {Object} Memory statistics
   */
  getStats() {
    const stats = {
      tensor: this.tensorPool.getStats(),
      webgl: this.webglPool ? this.webglPool.getStats() : null
    };

    // Add WASM memory stats if available
    if (wasmModule && wasmModule.get_memory_stats) {
      stats.wasm = wasmModule.get_memory_stats();
    }

    return stats;
  }

  /**
   * Dispose all memory pools
   */
  dispose() {
    this.tensorPool.dispose();
    if (this.webglPool) {
      this.webglPool.clear();
    }
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
  }
}

// Global memory manager instance
let globalMemoryManager = null;

/**
 * Get or create global memory manager
 * @param {Object} options - Memory manager options
 * @returns {MemoryManager} Global memory manager
 */
export function getMemoryManager(options = {}) {
  if (!globalMemoryManager) {
    globalMemoryManager = new MemoryManager(options);
  }
  return globalMemoryManager;
}

/**
 * Utility function to wrap tensor operations with memory management
 * @param {Function} operation - Tensor operation function
 * @param {Array} inputs - Input tensors
 * @param {Object} options - Options including auto-release
 * @returns {*} Operation result
 */
export async function withMemoryManagement(operation, inputs = [], options = {}) {
  const manager = getMemoryManager();
  const { autoRelease = true } = options;

  try {
    const result = await operation();
    return result;
  } finally {
    if (autoRelease) {
      for (const tensor of inputs) {
        if (tensor && manager.tensorPool.allocatedTensors.has(tensor)) {
          manager.tensorPool.release(tensor);
        }
      }
    }
  }
}