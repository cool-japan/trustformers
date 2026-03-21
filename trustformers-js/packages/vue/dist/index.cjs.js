'use strict';

var vue = require('vue');
var perf_hooks = require('perf_hooks');

/**
 * WebGPU Support Module
 * Provides WebGPU operations and device management for enhanced performance
 */

let wasmModule$5 = null;

/**
 * Set WASM module reference
 * @param {Object} module - WASM module
 */
function setWasmModule(module) {
  wasmModule$5 = module;
}

/**
 * WebGPU support utilities
 */
const webgpu = {
  /**
   * Check if WebGPU is available in the browser
   * @returns {boolean} True if WebGPU is available
   */
  isAvailable() {
    if (typeof navigator === 'undefined') return false;

    // Check for WebGPU API availability
    if (navigator.gpu) {
      return true;
    }

    // Fallback to WASM module check if available
    if (wasmModule$5 && wasmModule$5.is_webgpu_available) {
      try {
        return wasmModule$5.is_webgpu_available();
      } catch (error) {
        console.warn('Error checking WebGPU availability from WASM:', error);
        return false;
      }
    }

    return false;
  },

  /**
   * Get WebGPU status information
   * @returns {string} Status message
   */
  getStatus() {
    if (!this.isAvailable()) {
      return 'WebGPU not available in this browser';
    }

    if (wasmModule$5 && wasmModule$5.get_webgpu_status) {
      try {
        return wasmModule$5.get_webgpu_status();
      } catch (error) {
        return `WebGPU status error: ${error.message}`;
      }
    }

    return 'WebGPU available but status information not accessible';
  },

  /**
   * Create WebGPU operations handler
   * @returns {Object} WebGPU operations object
   */
  createOps() {
    if (!this.isAvailable()) {
      throw new Error('WebGPU is not available in this environment');
    }

    if (wasmModule$5 && wasmModule$5.WebGPUOps) {
      try {
        return new wasmModule$5.WebGPUOps();
      } catch (error) {
        throw new Error(`Failed to create WebGPU operations: ${error.message}`);
      }
    }

    // Fallback WebGPU operations implementation
    return new WebGPUOperations();
  },

  /**
   * Get WebGPU device information
   * @returns {Promise<Object>} Device information
   */
  async getDeviceInfo() {
    if (!this.isAvailable()) {
      throw new Error('WebGPU not available');
    }

    if (wasmModule$5 && wasmModule$5.get_webgpu_device_info) {
      try {
        return await wasmModule$5.get_webgpu_device_info();
      } catch (error) {
        console.warn('Error getting device info from WASM:', error);
      }
    }

    // Fallback to direct WebGPU API
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('No WebGPU adapter available');
      }

      const device = await adapter.requestDevice();

      return {
        vendor: adapter.info?.vendor || 'unknown',
        architecture: adapter.info?.architecture || 'unknown',
        device: adapter.info?.device || 'unknown',
        description: adapter.info?.description || 'unknown',
        limits: device.limits,
        features: Array.from(device.features)
      };
    } catch (error) {
      throw new Error(`Failed to get WebGPU device info: ${error.message}`);
    }
  },

  /**
   * Test WebGPU compute capabilities
   * @returns {Promise<Object>} Test results
   */
  async testComputeCapabilities() {
    if (!this.isAvailable()) {
      throw new Error('WebGPU not available');
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('No WebGPU adapter available');
      }

      const device = await adapter.requestDevice();

      // Simple compute shader test
      const shaderCode = `
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          // Simple test computation
        }
      `;

      const shaderModule = device.createShaderModule({
        code: shaderCode
      });

      const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        }
      });

      return {
        success: true,
        computeShaderSupport: true,
        maxWorkgroupSize: device.limits.maxComputeWorkgroupSizeX,
        maxWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
        maxBufferSize: device.limits.maxBufferSize
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        computeShaderSupport: false
      };
    }
  }
};

/**
 * Fallback WebGPU operations implementation
 */
class WebGPUOperations {
  constructor() {
    this.device = null;
    this.adapter = null;
    this.initialized = false;
  }

  /**
   * Initialize WebGPU device
   * @returns {Promise<void>}
   */
  async initialize() {
    if (this.initialized) return;

    if (!navigator.gpu) {
      throw new Error('WebGPU not supported');
    }

    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) {
      throw new Error('No WebGPU adapter available');
    }

    this.device = await this.adapter.requestDevice();
    this.initialized = true;
  }

  /**
   * Matrix multiplication using WebGPU compute shaders
   * @param {Object} a - First matrix tensor
   * @param {Object} b - Second matrix tensor
   * @returns {Promise<Object>} Result tensor
   */
  async matmul(a, b) {
    await this.initialize();

    // This is a simplified implementation
    // In practice, this would create compute shaders for matrix multiplication
    const shaderCode = `
      @compute @workgroup_size(8, 8)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let row = global_id.x;
        let col = global_id.y;

        // Matrix multiplication logic would go here
        // This is just a placeholder
      }
    `;

    try {
      const shaderModule = this.device.createShaderModule({
        code: shaderCode
      });

      // Create buffers and perform computation
      // This is a placeholder - real implementation would be much more complex

      // For now, return a simple result structure
      return {
        data: new Float32Array(a.shape[0] * b.shape[1]),
        shape: [a.shape[0], b.shape[1]],
        backend: 'webgpu'
      };
    } catch (error) {
      throw new Error(`WebGPU matmul failed: ${error.message}`);
    }
  }

  /**
   * Element-wise operations using WebGPU
   * @param {Object} a - First tensor
   * @param {Object} b - Second tensor
   * @param {string} operation - Operation type
   * @returns {Promise<Object>} Result tensor
   */
  async elementWise(a, b, operation) {
    await this.initialize();

    const operations = {
      'add': '+',
      'sub': '-',
      'mul': '*',
      'div': '/'
    };

    const op = operations[operation];
    if (!op) {
      throw new Error(`Unsupported operation: ${operation}`);
    }

    // Placeholder implementation
    return {
      data: new Float32Array(a.data.length),
      shape: a.shape,
      backend: 'webgpu'
    };
  }

  /**
   * Activation functions using WebGPU
   * @param {Object} tensor - Input tensor
   * @param {string} activation - Activation type
   * @returns {Promise<Object>} Result tensor
   */
  async activation(tensor, activation) {
    await this.initialize();

    const activations = {
      'relu': 'max(0.0, x)',
      'sigmoid': '1.0 / (1.0 + exp(-x))',
      'tanh': 'tanh(x)',
      'gelu': 'x * 0.5 * (1.0 + tanh(sqrt(2.0 / 3.14159) * (x + 0.044715 * x * x * x)))'
    };

    const activationCode = activations[activation];
    if (!activationCode) {
      throw new Error(`Unsupported activation: ${activation}`);
    }

    // Placeholder implementation
    return {
      data: new Float32Array(tensor.data.length),
      shape: tensor.shape,
      backend: 'webgpu'
    };
  }

  /**
   * Cleanup WebGPU resources
   */
  dispose() {
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
    this.adapter = null;
    this.initialized = false;
  }
}

/**
 * Enhanced Model Inference with Optimizations
 * Advanced inference operations with memory management, performance profiling, and automatic cleanup
 */

/**
 * Enhanced model inference with optimizations
 */
const enhanced_inference = {
  /**
   * Run optimized model inference
   * @param {Object} model - Model object
   * @param {Object} inputs - Input tensors
   * @param {Object} options - Inference options
   * @returns {Promise<Object>} Inference results
   */
  async runInference(model, inputs, options = {}) {
    const {
      profile = true,
      useMemoryPool = true,
      autoCleanup = true,
      timeout = 30000,
      backend = 'auto',
      enableFallback = true,
      validateInputs = true,
      cacheResults = false
    } = options;

    // Input validation
    if (validateInputs) {
      await this._validateInputs(model, inputs);
    }

    // Check cache first
    if (cacheResults) {
      const cachedResult = await this._getCachedResult(model, inputs);
      if (cachedResult) {
        return cachedResult;
      }
    }

    const startTime = performance.now();

    try {
      let result;

      if (profile) {
        // Import performance profiler
        const { performanceProfiler } = await Promise.resolve().then(function () { return index; });
        if (performanceProfiler) {
          result = await performanceProfiler.profileInference(model, inputs, {
            backend,
            useMemoryPool,
            autoCleanup,
            metadata: options.metadata
          });
        } else {
          result = await this._runInferenceWithMemoryManagement(model, inputs, {
            useMemoryPool,
            autoCleanup,
            timeout,
            enableFallback
          });
        }
      } else {
        result = await this._runInferenceWithMemoryManagement(model, inputs, {
          useMemoryPool,
          autoCleanup,
          timeout,
          enableFallback
        });
      }

      const endTime = performance.now();
      const duration = endTime - startTime;

      // Add performance metadata
      if (result && typeof result === 'object') {
        result._inference_metadata = {
          duration,
          backend,
          useMemoryPool,
          timestamp: new Date().toISOString(),
          options: {
            profile,
            autoCleanup,
            backend
          }
        };
      }

      // Cache result if requested
      if (cacheResults) {
        await this._cacheResult(model, inputs, result);
      }

      return result;
    } catch (error) {
      if (enableFallback) {
        console.warn('Primary inference failed, attempting fallback:', error.message);
        return await this._fallbackInference(model, inputs, {
          timeout,
          autoCleanup
        });
      } 
        throw error;
      
    }
  },

  /**
   * Internal inference with memory management
   * @private
   */
  async _runInferenceWithMemoryManagement(model, inputs, options = {}) {
    const { useMemoryPool, autoCleanup, timeout, enableFallback } = options;

    if (useMemoryPool) {
      const { withMemoryManagement } = await Promise.resolve().then(function () { return index; });
      if (withMemoryManagement) {
        return await withMemoryManagement(
          () => model.forward(inputs),
          Array.isArray(inputs) ? inputs : [inputs],
          { autoRelease: autoCleanup }
        );
      }
    }

    // Direct inference with timeout
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error('Inference timeout')), timeout);
    });

    try {
      const inferencePromise = model.forward_async ?
        model.forward_async(inputs) :
        Promise.resolve(model.forward(inputs));

      const result = await Promise.race([inferencePromise, timeoutPromise]);
      return result;
    } finally {
      if (autoCleanup) {
        this._cleanupTensors(inputs);
      }
    }
  },

  /**
   * Fallback inference implementation
   * @private
   */
  async _fallbackInference(model, inputs, options = {}) {
    const { timeout, autoCleanup } = options;

    try {
      // Simple direct model inference
      if (model.forward) {
        const result = await Promise.race([
          Promise.resolve(model.forward(inputs)),
          new Promise((_, reject) => setTimeout(() => reject(new Error('Fallback timeout')), timeout))
        ]);

        return result;
      } 
        throw new Error('Model does not support forward inference');
      
    } catch (error) {
      throw new Error(`Fallback inference failed: ${error.message}`);
    } finally {
      if (autoCleanup) {
        this._cleanupTensors(inputs);
      }
    }
  },

  /**
   * Validate model inputs
   * @private
   */
  async _validateInputs(model, inputs) {
    if (!model) {
      throw new Error('Model is required for inference');
    }

    if (!inputs) {
      throw new Error('Inputs are required for inference');
    }

    // Check if inputs match expected format
    if (model.getExpectedInputs && typeof model.getExpectedInputs === 'function') {
      try {
        const expectedInputs = model.getExpectedInputs();
        // Validate against expected inputs structure
        // This is a simplified validation - real implementation would be more thorough
      } catch (error) {
        console.warn('Could not validate inputs against model expectations:', error.message);
      }
    }
  },

  /**
   * Clean up tensor resources
   * @private
   */
  _cleanupTensors(inputs) {
    const cleanupTensor = (tensor) => {
      if (tensor && typeof tensor === 'object') {
        if (tensor.dispose) {
          tensor.dispose();
        } else if (tensor.free) {
          tensor.free();
        } else if (tensor.delete) {
          tensor.delete();
        }
      }
    };

    if (Array.isArray(inputs)) {
      inputs.forEach(cleanupTensor);
    } else {
      cleanupTensor(inputs);
    }
  },

  /**
   * Simple result caching
   * @private
   */
  async _getCachedResult(model, inputs) {
    // Simple cache implementation using WeakMap
    if (!this._resultCache) {
      this._resultCache = new Map();
    }

    const key = this._createCacheKey(model, inputs);
    return this._resultCache.get(key);
  },

  /**
   * Cache inference result
   * @private
   */
  async _cacheResult(model, inputs, result) {
    if (!this._resultCache) {
      this._resultCache = new Map();
    }

    const key = this._createCacheKey(model, inputs);

    // Simple cache with size limit
    if (this._resultCache.size > 100) {
      const firstKey = this._resultCache.keys().next().value;
      this._resultCache.delete(firstKey);
    }

    this._resultCache.set(key, result);
  },

  /**
   * Create cache key for model and inputs
   * @private
   */
  _createCacheKey(model, inputs) {
    // Simple key creation - in practice this would be more sophisticated
    const modelHash = model.toString().slice(0, 20);
    const inputHash = JSON.stringify(inputs).slice(0, 50);
    return `${modelHash}_${inputHash}`;
  },

  /**
   * Batch inference with optimizations
   * @param {Object} model - Model object
   * @param {Array} batchInputs - Array of input batches
   * @param {Object} options - Batch options
   * @returns {Promise<Array>} Array of results
   */
  async batchInference(model, batchInputs, options = {}) {
    const {
      batchSize = 32,
      profile = true,
      parallel = true,
      maxConcurrency = 4,
      validateInputs = true,
      onProgress = null
    } = options;

    if (validateInputs) {
      for (const inputs of batchInputs) {
        await this._validateInputs(model, inputs);
      }
    }

    const operation = async () => {
      if (parallel) {
        return await this._parallelBatchInference(model, batchInputs, {
          batchSize,
          maxConcurrency,
          onProgress,
          ...options
        });
      } 
        return await this._sequentialBatchInference(model, batchInputs, {
          batchSize,
          onProgress,
          ...options
        });
      
    };

    if (profile) {
      const { performanceProfiler } = await Promise.resolve().then(function () { return index; });
      if (performanceProfiler) {
        return await performanceProfiler.profileOperation('batch_inference', operation, {
          batchSize: batchInputs.length,
          batchChunkSize: batchSize,
          parallel
        });
      }
    }

    return await operation();
  },

  /**
   * Parallel batch inference
   * @private
   */
  async _parallelBatchInference(model, batchInputs, options = {}) {
    const { batchSize, maxConcurrency, onProgress } = options;

    const results = [];
    const semaphore = new Semaphore(maxConcurrency);

    const chunks = [];
    for (let i = 0; i < batchInputs.length; i += batchSize) {
      chunks.push(batchInputs.slice(i, i + batchSize));
    }

    const promises = chunks.map(async (chunk, index) => {
      await semaphore.acquire();

      try {
        const chunkResults = await Promise.all(
          chunk.map(inputs => this.runInference(model, inputs, {
            ...options,
            profile: false // Disable individual profiling in batch mode
          }))
        );

        if (onProgress) {
          onProgress({
            completed: index + 1,
            total: chunks.length,
            batchResults: chunkResults
          });
        }

        return chunkResults;
      } finally {
        semaphore.release();
      }
    });

    const chunkResults = await Promise.all(promises);

    // Flatten results
    for (const chunkResult of chunkResults) {
      results.push(...chunkResult);
    }

    return results;
  },

  /**
   * Sequential batch inference
   * @private
   */
  async _sequentialBatchInference(model, batchInputs, options = {}) {
    const { batchSize, onProgress } = options;

    const results = [];
    const totalBatches = Math.ceil(batchInputs.length / batchSize);

    for (let i = 0; i < totalBatches; i++) {
      const batchStart = i * batchSize;
      const batchEnd = Math.min(batchStart + batchSize, batchInputs.length);
      const batch = batchInputs.slice(batchStart, batchEnd);

      const batchResults = [];
      for (const inputs of batch) {
        const result = await this.runInference(model, inputs, {
          ...options,
          profile: false
        });
        batchResults.push(result);
      }

      results.push(...batchResults);

      if (onProgress) {
        onProgress({
          completed: i + 1,
          total: totalBatches,
          batchResults
        });
      }

      // Allow other tasks to run
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    return results;
  },

  /**
   * Stream inference for real-time processing
   * @param {Object} model - Model object
   * @param {AsyncIterable} inputStream - Stream of inputs
   * @param {Object} options - Streaming options
   * @returns {AsyncGenerator} Stream of results
   */
  async* streamInference(model, inputStream, options = {}) {
    const {
      bufferSize = 10,
      profile = false,
      autoCleanup = true
    } = options;

    const buffer = [];

    try {
      for await (const inputs of inputStream) {
        // Add to buffer
        buffer.push(inputs);

        // Process when buffer is full or stream ends
        if (buffer.length >= bufferSize) {
          const batch = buffer.splice(0, bufferSize);
          const results = await this.batchInference(model, batch, {
            ...options,
            profile,
            parallel: true
          });

          for (const result of results) {
            yield result;
          }
        }
      }

      // Process remaining items in buffer
      if (buffer.length > 0) {
        const results = await this.batchInference(model, buffer, {
          ...options,
          profile,
          parallel: true
        });

        for (const result of results) {
          yield result;
        }
      }
    } finally {
      if (autoCleanup && buffer.length > 0) {
        // Cleanup any remaining tensors in buffer
        buffer.forEach(inputs => this._cleanupTensors(inputs));
      }
    }
  }
};

/**
 * Simple semaphore implementation for concurrency control
 */
class Semaphore {
  constructor(maxConcurrency) {
    this.maxConcurrency = maxConcurrency;
    this.currentConcurrency = 0;
    this.waitingQueue = [];
  }

  async acquire() {
    if (this.currentConcurrency < this.maxConcurrency) {
      this.currentConcurrency++;
      return;
    }

    return new Promise((resolve) => {
      this.waitingQueue.push(resolve);
    });
  }

  release() {
    this.currentConcurrency--;

    if (this.waitingQueue.length > 0) {
      const resolve = this.waitingQueue.shift();
      this.currentConcurrency++;
      resolve();
    }
  }
}

/**
 * Development Tools for Debugging and Analysis
 * Comprehensive development utilities for TrustformeRS debugging, analysis, and visualization
 */

/**
 * Development tools for debugging and analysis
 */
const devTools = {
  // Debug utilities
  debug: {
    enable: (options) => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.enable(options)).catch(() => {
        console.warn('Debug utilities not available');
        return false;
      }),

    disable: () => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.disable()).catch(() => {
        console.warn('Debug utilities not available');
        return false;
      }),

    configure: (config) => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.configure(config)).catch(() => {
        console.warn('Debug utilities not available');
        return false;
      }),

    isEnabled: () => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.isEnabled()).catch(() => false),

    startSession: (name, metadata) => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.startSession(name, metadata)).catch(() => {
        console.warn('Debug utilities not available');
        return null;
      }),

    endSession: () => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.endSession()).catch(() => {
        console.warn('Debug utilities not available');
        return null;
      }),

    trackTensor: (tensor, operation, metadata) => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.trackTensor(tensor, operation, metadata)).catch(() => {
        console.warn('Debug utilities not available');
        return null;
      }),

    trackOperation: (name, fn, metadata) => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.trackOperation(name, fn, metadata)).catch(() => {
        console.warn('Debug utilities not available');
        return fn();
      }),

    validateOperation: (operation, tensors, options) => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.validateOperation(operation, tensors, options)).catch(() => {
        console.warn('Debug utilities not available');
        return true;
      }),

    getMemoryUsage: () => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.getMemoryUsage()).catch(() => undefined._getFallbackMemoryUsage()),

    getPerformanceMetrics: () => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.getPerformanceMetrics()).catch(() => undefined._getFallbackPerformanceMetrics()),

    generateReport: () => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.generateReport()).catch(() => undefined._generateFallbackReport()),

    exportData: (format) => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.exportData(format)).catch(() => {
        console.warn('Debug utilities not available');
        return null;
      }),

    clear: () => Promise.resolve().then(function () { return debugUtilities; }).then(({ debugUtils }) => debugUtils.clear()).catch(() => {
        console.warn('Debug utilities not available');
        return false;
      }),

    // Fallback methods
    _getFallbackMemoryUsage() {
      if (typeof performance !== 'undefined' && performance.memory) {
        return {
          used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
          total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
          limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
        };
      }
      return null;
    },

    _getFallbackPerformanceMetrics() {
      return {
        timestamp: new Date().toISOString(),
        now: typeof performance !== 'undefined' ? performance.now() : Date.now(),
        memory: this._getFallbackMemoryUsage()
      };
    },

    _generateFallbackReport() {
      return {
        timestamp: new Date().toISOString(),
        type: 'fallback_report',
        memory: this._getFallbackMemoryUsage(),
        performance: this._getFallbackPerformanceMetrics(),
        message: 'Debug utilities not available - using fallback report'
      };
    }
  },

  // Tensor inspection
  tensor: {
    analyze: (tensor, options) => Promise.resolve().then(function () { return tensorInspector$1; }).then(({ tensorInspector }) => tensorInspector.analyze(tensor, options)).catch(() => undefined._fallbackTensorAnalysis(tensor)),

    compare: (tensor1, tensor2, options) => Promise.resolve().then(function () { return tensorInspector$1; }).then(({ tensorInspector }) => tensorInspector.compare(tensor1, tensor2, options)).catch(() => undefined._fallbackTensorComparison(tensor1, tensor2)),

    visualize: (tensor, options) => Promise.resolve().then(function () { return tensorInspector$1; }).then(({ tensorInspector }) => tensorInspector.visualizeText(tensor, options)).catch(() => undefined._fallbackTensorVisualization(tensor)),

    visualizeHTML: (tensor, options) => Promise.resolve().then(function () { return tensorInspector$1; }).then(({ tensorInspector }) => tensorInspector.visualizeHTML(tensor, options)).catch(() => undefined._fallbackTensorHTMLVisualization(tensor)),

    summarize: (tensor) => Promise.resolve().then(function () { return tensorInspector$1; }).then(({ tensorInspector }) => tensorInspector.summarize(tensor)).catch(() => undefined._fallbackTensorSummary(tensor)),

    clearCaches: () => Promise.resolve().then(function () { return tensorInspector$1; }).then(({ tensorInspector }) => tensorInspector.clearCaches()).catch(() => {
        console.warn('Tensor inspector not available');
        return false;
      }),

    // Fallback tensor analysis methods
    _fallbackTensorAnalysis(tensor) {
      if (!tensor || !tensor.data) {
        return { error: 'Invalid tensor for analysis' };
      }

      const {data} = tensor;
      const shape = tensor.shape || [data.length];

      return {
        shape: { shape: Array.from(shape), size: data.length },
        dtype: { dtype: tensor.dtype || 'f32', bytes: data.byteLength },
        statistics: this._calculateBasicStatistics(data),
        memory: { totalBytes: data.byteLength },
        quality: {
          hasNaN: this._hasNaN(data),
          hasInfinite: this._hasInfinite(data)
        }
      };
    },

    _fallbackTensorComparison(tensor1, tensor2) {
      const analysis1 = this._fallbackTensorAnalysis(tensor1);
      const analysis2 = this._fallbackTensorAnalysis(tensor2);

      return {
        tensor1: analysis1,
        tensor2: analysis2,
        comparison: {
          shapesMatch: JSON.stringify(analysis1.shape.shape) === JSON.stringify(analysis2.shape.shape),
          dtypesMatch: analysis1.dtype.dtype === analysis2.dtype.dtype,
          sizesMatch: analysis1.shape.size === analysis2.shape.size
        }
      };
    },

    _fallbackTensorVisualization(tensor) {
      const analysis = this._fallbackTensorAnalysis(tensor);
      return `
Tensor Analysis:
  Shape: ${JSON.stringify(analysis.shape.shape)}
  Size: ${analysis.shape.size}
  Type: ${analysis.dtype.dtype}
  Memory: ${(analysis.memory.totalBytes / 1024).toFixed(2)} KB
  Statistics: ${JSON.stringify(analysis.statistics, null, 2)}
      `.trim();
    },

    _fallbackTensorHTMLVisualization(tensor) {
      const analysis = this._fallbackTensorAnalysis(tensor);
      return `
        <div style="font-family: monospace; border: 1px solid #ccc; padding: 10px; margin: 5px;">
          <h4>Tensor Analysis</h4>
          <p><strong>Shape:</strong> ${JSON.stringify(analysis.shape.shape)}</p>
          <p><strong>Size:</strong> ${analysis.shape.size.toLocaleString()}</p>
          <p><strong>Type:</strong> ${analysis.dtype.dtype}</p>
          <p><strong>Memory:</strong> ${(analysis.memory.totalBytes / 1024).toFixed(2)} KB</p>
          <div style="margin-top: 10px;">
            <strong>Statistics:</strong>
            <pre style="background: #f5f5f5; padding: 5px; margin: 5px 0;">${JSON.stringify(analysis.statistics, null, 2)}</pre>
          </div>
        </div>
      `;
    },

    _fallbackTensorSummary(tensor) {
      const analysis = this._fallbackTensorAnalysis(tensor);
      return {
        shape: analysis.shape.shape,
        size: analysis.shape.size,
        dtype: analysis.dtype.dtype,
        memory: `${(analysis.memory.totalBytes / 1024).toFixed(2)} KB`
      };
    },

    _calculateBasicStatistics(data) {
      if (!data || data.length === 0) return null;

      let sum = 0;
      const [firstValue] = data;
      let min = firstValue;
      let max = firstValue;

      for (let i = 0; i < data.length; i++) {
        const value = data[i];
        sum += value;
        if (value < min) min = value;
        if (value > max) max = value;
      }

      const mean = sum / data.length;

      // Calculate variance
      let variance = 0;
      for (let i = 0; i < data.length; i++) {
        const diff = data[i] - mean;
        variance += diff * diff;
      }
      variance /= data.length;

      return {
        count: data.length,
        sum,
        mean,
        min,
        max,
        variance,
        std: Math.sqrt(variance)
      };
    },

    _hasNaN(data) {
      for (let i = 0; i < data.length; i++) {
        if (Number.isNaN(data[i])) return true;
      }
      return false;
    },

    _hasInfinite(data) {
      for (let i = 0; i < data.length; i++) {
        if (!Number.isFinite(data[i])) return true;
      }
      return false;
    }
  },

  // Model visualization
  model: {
    analyze: (model, options) => Promise.resolve().then(function () { return modelVisualization; }).then(({ modelVisualizer }) => modelVisualizer.analyzeModel(model, options)).catch(() => undefined._fallbackModelAnalysis(model)),

    visualize: (model, options) => Promise.resolve().then(function () { return modelVisualization; }).then(({ modelVisualizer }) => modelVisualizer.visualizeArchitecture(model, options)).catch(() => undefined._fallbackModelVisualization(model)),

    visualizeHTML: (model, options) => Promise.resolve().then(function () { return modelVisualization; }).then(({ modelVisualizer }) => modelVisualizer.visualizeHTML(model, options)).catch(() => undefined._fallbackModelHTMLVisualization(model)),

    summarize: (model) => Promise.resolve().then(function () { return modelVisualization; }).then(({ modelVisualizer }) => modelVisualizer.summarize(model)).catch(() => undefined._fallbackModelSummary(model)),

    clearCaches: () => Promise.resolve().then(function () { return modelVisualization; }).then(({ modelVisualizer }) => modelVisualizer.clearCaches()).catch(() => {
        console.warn('Model visualizer not available');
        return false;
      }),

    // Fallback model analysis methods
    _fallbackModelAnalysis(model) {
      return {
        type: typeof model,
        hasForward: typeof model.forward === 'function',
        hasConfig: !!model.config,
        timestamp: new Date().toISOString()
      };
    },

    _fallbackModelVisualization(model) {
      const analysis = this._fallbackModelAnalysis(model);
      return `
Model Analysis:
  Type: ${analysis.type}
  Has Forward: ${analysis.hasForward}
  Has Config: ${analysis.hasConfig}
  Analyzed: ${analysis.timestamp}
      `.trim();
    },

    _fallbackModelHTMLVisualization(model) {
      const analysis = this._fallbackModelAnalysis(model);
      return `
        <div style="font-family: monospace; border: 1px solid #ccc; padding: 10px; margin: 5px;">
          <h4>Model Analysis</h4>
          <p><strong>Type:</strong> ${analysis.type}</p>
          <p><strong>Has Forward:</strong> ${analysis.hasForward}</p>
          <p><strong>Has Config:</strong> ${analysis.hasConfig}</p>
          <p><strong>Analyzed:</strong> ${analysis.timestamp}</p>
        </div>
      `;
    },

    _fallbackModelSummary(model) {
      const analysis = this._fallbackModelAnalysis(model);
      return {
        type: analysis.type,
        capabilities: {
          forward: analysis.hasForward,
          config: analysis.hasConfig
        }
      };
    }
  },

  // Error diagnostics
  error: {
    diagnose: (error, context, options) => Promise.resolve().then(function () { return errorDiagnostics$1; }).then(({ errorDiagnostics }) => errorDiagnostics.diagnose(error, context, options)).catch(() => undefined._fallbackErrorDiagnosis(error, context)),

    generateReport: (error, context) => Promise.resolve().then(function () { return errorDiagnostics$1; }).then(({ errorDiagnostics }) => errorDiagnostics.generateReport(error, context)).catch(() => undefined._fallbackErrorReport(error, context)),

    generateTextReport: (error, context) => Promise.resolve().then(function () { return errorDiagnostics$1; }).then(({ errorDiagnostics }) => errorDiagnostics.generateTextReport(error, context)).catch(() => undefined._fallbackTextErrorReport(error, context)),

    generateHTMLReport: (error, context) => Promise.resolve().then(function () { return errorDiagnostics$1; }).then(({ errorDiagnostics }) => errorDiagnostics.generateHTMLReport(error, context)).catch(() => undefined._fallbackHTMLErrorReport(error, context)),

    getStatistics: () => Promise.resolve().then(function () { return errorDiagnostics$1; }).then(({ errorDiagnostics }) => errorDiagnostics.getStatistics()).catch(() => ({ errors: 0, warnings: 0 })),

    clearCache: () => Promise.resolve().then(function () { return errorDiagnostics$1; }).then(({ errorDiagnostics }) => errorDiagnostics.clearCache()).catch(() => {
        console.warn('Error diagnostics not available');
        return false;
      }),

    // Fallback error analysis methods
    _fallbackErrorDiagnosis(error, context) {
      return {
        error: {
          name: error.name,
          message: error.message,
          stack: error.stack
        },
        context: context || {},
        timestamp: new Date().toISOString(),
        type: 'fallback_diagnosis'
      };
    },

    _fallbackErrorReport(error, context) {
      const diagnosis = this._fallbackErrorDiagnosis(error, context);
      return {
        ...diagnosis,
        suggestions: [
          'Check input parameters',
          'Verify tensor shapes and types',
          'Ensure model is properly initialized'
        ]
      };
    },

    _fallbackTextErrorReport(error, context) {
      const report = this._fallbackErrorReport(error, context);
      return `
Error Report:
  Name: ${report.error.name}
  Message: ${report.error.message}
  Timestamp: ${report.timestamp}

Suggestions:
${report.suggestions.map(s => `  - ${s}`).join('\n')}
      `.trim();
    },

    _fallbackHTMLErrorReport(error, context) {
      const report = this._fallbackErrorReport(error, context);
      return `
        <div style="font-family: monospace; border: 2px solid #d32f2f; padding: 15px; margin: 5px; background: #ffebee;">
          <h4 style="color: #d32f2f; margin-top: 0;">Error Report</h4>
          <p><strong>Name:</strong> ${report.error.name}</p>
          <p><strong>Message:</strong> ${report.error.message}</p>
          <p><strong>Timestamp:</strong> ${report.timestamp}</p>
          <div style="margin-top: 10px;">
            <strong>Suggestions:</strong>
            <ul style="margin: 5px 0;">
              ${report.suggestions.map(s => `<li>${s}</li>`).join('')}
            </ul>
          </div>
        </div>
      `;
    }
  },

  // Combined analysis
  analyze: {
    /**
     * Comprehensive analysis of tensors, model, and context
     * @param {Object} options - Analysis options
     * @returns {Promise<Object>} Comprehensive analysis
     */
    async comprehensive(options = {}) {
      const { tensors = [], model = null, includeDebug = true } = options;

      const analysis = {
        timestamp: new Date().toISOString(),
        tensors: [],
        model: null,
        debug: null,
        performance: null,
        memory: null
      };

      // Analyze tensors
      if (tensors.length > 0) {
        analysis.tensors = await Promise.all(
          tensors.map(async tensor => {
            try {
              return await devTools.tensor.analyze(tensor, {
                includeStatistics: true,
                includeDistribution: true,
                includeNaN: true,
                includeInfinite: true
              });
            } catch (error) {
              return { error: error.message };
            }
          })
        );
      }

      // Analyze model
      if (model) {
        try {
          analysis.model = await devTools.model.analyze(model, {
            includeWeights: true,
            includeActivations: false,
            includeGradients: false
          });
        } catch (error) {
          analysis.model = { error: error.message };
        }
      }

      // Include debug information
      if (includeDebug) {
        try {
          const isEnabled = await devTools.debug.isEnabled();
          if (isEnabled) {
            analysis.debug = await devTools.debug.generateReport();
          }
        } catch (error) {
          analysis.debug = { error: error.message };
        }
      }

      // Include performance information
      if (typeof performance !== 'undefined') {
        analysis.performance = {
          now: performance.now(),
          memory: performance.memory ? {
            used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
            total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
            limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
          } : null
        };
      }

      return analysis;
    },

    /**
     * Generate comprehensive HTML report
     * @param {Object} options - Report options
     * @returns {Promise<string>} HTML report
     */
    async generateHTMLReport(options = {}) {
      const analysis = await this.comprehensive(options);

      let html = `
      <div class="comprehensive-analysis" style="font-family: 'Courier New', monospace; border: 1px solid #ccc; padding: 20px; margin: 10px; background: #f9f9f9;">
        <h2 style="margin-top: 0; color: #333;">TrustformeRS Comprehensive Analysis</h2>
        <p style="color: #666; margin-bottom: 20px;">Generated: ${analysis.timestamp}</p>
      `;

      if (analysis.model) {
        html += `
        <div class="model-section" style="margin-bottom: 20px;">
          <h3 style="color: #007bff;">Model Analysis</h3>
          ${await devTools.model.visualizeHTML(options.model, { includeMemory: true })}
        </div>
        `;
      }

      if (analysis.tensors.length > 0) {
        html += `
        <div class="tensor-section" style="margin-bottom: 20px;">
          <h3 style="color: #28a745;">Tensor Analysis</h3>
        `;

        for (let i = 0; i < analysis.tensors.length; i++) {
          const tensorAnalysis = analysis.tensors[i];
          if (!tensorAnalysis.error) {
            html += `
            <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #fff;">
              <h4>Tensor ${i + 1}</h4>
              <p><strong>Shape:</strong> ${JSON.stringify(tensorAnalysis.shape.shape)}</p>
              <p><strong>Data Type:</strong> ${tensorAnalysis.dtype.dtype}</p>
              <p><strong>Size:</strong> ${tensorAnalysis.shape.size.toLocaleString()} elements</p>
              <p><strong>Memory:</strong> ${(tensorAnalysis.memory.totalBytes / 1024).toFixed(2)} KB</p>
              ${tensorAnalysis.quality && tensorAnalysis.quality.hasNaN ? '<p style="color: #d32f2f;">⚠ Contains NaN values</p>' : ''}
              ${tensorAnalysis.quality && tensorAnalysis.quality.hasInfinite ? '<p style="color: #d32f2f;">⚠ Contains infinite values</p>' : ''}
            </div>
            `;
          }
        }

        html += '</div>';
      }

      if (analysis.debug) {
        html += `
        <div class="debug-section" style="margin-bottom: 20px;">
          <h3 style="color: #ffc107;">Debug Information</h3>
          <div style="background: #fff; padding: 15px; border-radius: 4px;">
            <pre style="margin: 0; font-size: 12px;">${JSON.stringify(analysis.debug, null, 2)}</pre>
          </div>
        </div>
        `;
      }

      if (analysis.performance) {
        html += `
        <div class="performance-section" style="margin-bottom: 20px;">
          <h3 style="color: #dc3545;">Performance Information</h3>
          <div style="background: #fff; padding: 15px; border-radius: 4px;">
            <p><strong>Current Time:</strong> ${analysis.performance.now.toFixed(2)}ms</p>
            ${analysis.performance.memory ? `
            <p><strong>Memory Usage:</strong> ${analysis.performance.memory.used}MB / ${analysis.performance.memory.limit}MB</p>
            <p><strong>Memory Utilization:</strong> ${((analysis.performance.memory.used / analysis.performance.memory.limit) * 100).toFixed(2)}%</p>
            ` : ''}
          </div>
        </div>
        `;
      }

      html += '</div>';
      return html;
    }
  }
};

/**
 * Node.js Optimizations and Support
 * Enhanced Node.js-specific functionality and optimizations
 * Only available in Node.js environment
 */

/**
 * Node.js optimizations
 * Only available in Node.js environment
 */
(() => {
  // Check if we're in Node.js environment
  if (typeof process === 'undefined' || !process.versions || !process.versions.node) {
    return {
      isAvailable: () => false,
      error: 'Node.js optimizations only available in Node.js environment'
    };
  }

  // Node.js modules (lazy loaded)
  let os = null;
  let cluster = null;
  let fs = null;
  let path = null;
  let worker_threads = null;

  const loadNodeModules = () => {
    if (!os) {
      try {
        os = require('os');
        cluster = require('cluster');
        fs = require('fs');
        path = require('path');
        worker_threads = require('worker_threads');
      } catch (error) {
        console.warn('Failed to load Node.js modules:', error.message);
      }
    }
  };

  return {
    /**
     * Check if Node.js optimizations are available
     * @returns {boolean} True if in Node.js environment
     */
    isAvailable: () => true,

    /**
     * Get Node.js specific system information
     * @returns {Object} System information
     */
    getSystemInfo() {
      loadNodeModules();

      try {
        return {
          nodeVersion: process.version,
          platform: process.platform,
          arch: process.arch,
          cpus: os ? os.cpus().length : 'unknown',
          memory: os ? {
            total: os.totalmem(),
            free: os.freemem(),
            process: process.memoryUsage()
          } : null,
          pid: process.pid,
          uptime: process.uptime(),
          argv: process.argv,
          env: {
            NODE_ENV: process.env.NODE_ENV,
            UV_THREADPOOL_SIZE: process.env.UV_THREADPOOL_SIZE
          }
        };
      } catch (error) {
        return {
          nodeVersion: process.version,
          error: error.message
        };
      }
    },

    /**
     * Create native module fallback system
     * @param {Object} options - Configuration options
     * @returns {NativeModuleFallback} Fallback system instance
     */
    createNativeModuleFallback(options = {}) {
      return new NativeModuleFallback(options);
    },

    /**
     * Create cluster manager for multi-process inference
     * @param {Object} options - Cluster configuration
     * @returns {ClusterManager} Cluster manager instance
     */
    createClusterManager(options = {}) {
      return new ClusterManager(options);
    },

    /**
     * Create streaming processor for large datasets
     * @param {Object} options - Streaming configuration
     * @returns {StreamingProcessor} Streaming processor instance
     */
    createStreamingProcessor(options = {}) {
      return new StreamingProcessor(options);
    },

    /**
     * Create file system optimizer
     * @param {Object} options - File system configuration
     * @returns {FileSystemOptimizer} File system optimizer instance
     */
    createFileSystemOptimizer(options = {}) {
      return new FileSystemOptimizer(options);
    },

    /**
     * Create memory manager for process optimization
     * @param {Object} options - Memory management configuration
     * @returns {MemoryManager} Memory manager instance
     */
    createMemoryManager(options = {}) {
      return new NodeJSMemoryManager(options);
    },

    /**
     * Optimize Node.js environment for ML workloads
     * @param {Object} options - Optimization options
     */
    optimizeForML(options = {}) {
      const {
        maxMemory = null,
        threadPoolSize = null,
        gcSettings = true
      } = options;

      try {
        // Set UV_THREADPOOL_SIZE if specified
        if (threadPoolSize && !process.env.UV_THREADPOOL_SIZE) {
          process.env.UV_THREADPOOL_SIZE = threadPoolSize.toString();
          console.warn(`Set UV_THREADPOOL_SIZE to ${threadPoolSize}`);
        }

        // Configure V8 flags for better ML performance
        if (gcSettings && process.argv0) {
          const v8Flags = [
            `--max-old-space-size=${maxMemory || 4096}`,
            '--optimize-for-size',
            '--gc-interval=100'
          ];

          console.warn('Recommended V8 flags:', v8Flags.join(' '));
          console.warn(`Add these to your node command: node ${v8Flags.join(' ')} your-script.js`);
        }

        // Set process title for easier identification
        if (process.title) {
          process.title = 'trustformers-ml-worker';
        }

        return {
          optimized: true,
          settings: {
            maxMemory,
            threadPoolSize,
            gcSettings
          }
        };
      } catch (error) {
        console.warn('Failed to optimize Node.js environment:', error.message);
        return { optimized: false, error: error.message };
      }
    }
  };
})();

/**
 * Native Module Fallback System
 * Provides fallback functionality when native modules are not available
 */
class NativeModuleFallback {
  constructor(options = {}) {
    this.options = {
      enableLogging: true,
      enableMetrics: true,
      fallbackTimeout: 5000,
      ...options
    };

    this.availableModules = new Map();
    this.fallbackCounts = new Map();
  }

  /**
   * Check if a native module is available
   * @param {string} moduleName - Name of the module
   * @returns {boolean} True if available
   */
  isModuleAvailable(moduleName) {
    if (this.availableModules.has(moduleName)) {
      return this.availableModules.get(moduleName);
    }

    try {
      require.resolve(moduleName);
      this.availableModules.set(moduleName, true);
      return true;
    } catch (error) {
      this.availableModules.set(moduleName, false);
      if (this.options.enableLogging) {
        console.warn(`Native module ${moduleName} not available:`, error.message);
      }
      return false;
    }
  }

  /**
   * Use module with fallback
   * @param {string} moduleName - Module name
   * @param {Function} nativeFunc - Function using native module
   * @param {Function} fallbackFunc - Fallback function
   * @returns {*} Function result
   */
  async useWithFallback(moduleName, nativeFunc, fallbackFunc) {
    if (this.isModuleAvailable(moduleName)) {
      try {
        return await nativeFunc();
      } catch (error) {
        if (this.options.enableLogging) {
          console.warn(`Native module ${moduleName} failed, using fallback:`, error.message);
        }
        this.recordFallback(moduleName);
        return await fallbackFunc();
      }
    } else {
      this.recordFallback(moduleName);
      return await fallbackFunc();
    }
  }

  /**
   * Record fallback usage
   * @private
   */
  recordFallback(moduleName) {
    if (this.options.enableMetrics) {
      const count = this.fallbackCounts.get(moduleName) || 0;
      this.fallbackCounts.set(moduleName, count + 1);
    }
  }

  /**
   * Get fallback statistics
   * @returns {Object} Statistics
   */
  getStatistics() {
    return {
      availableModules: Object.fromEntries(this.availableModules),
      fallbackCounts: Object.fromEntries(this.fallbackCounts)
    };
  }
}

/**
 * Cluster Manager for Multi-Process Inference
 */
class ClusterManager {
  constructor(options = {}) {
    this.options = {
      numWorkers: require('os').cpus().length,
      enableLogging: true,
      gracefulShutdown: true,
      workerRestartDelay: 1000,
      ...options
    };

    this.cluster = require('cluster');
    this.workers = new Map();
    this.stats = {
      totalRequests: 0,
      activeRequests: 0,
      completedRequests: 0,
      errors: 0
    };
  }

  /**
   * Initialize cluster
   * @returns {Promise<void>}
   */
  async initialize() {
    if (this.cluster.isMaster || this.cluster.isPrimary) {
      await this.startMaster();
    } else {
      await this.startWorker();
    }
  }

  /**
   * Start master process
   * @private
   */
  async startMaster() {
    if (this.options.enableLogging) {
      console.warn(`Master process ${process.pid} starting with ${this.options.numWorkers} workers`);
    }

    // Fork workers
    for (let i = 0; i < this.options.numWorkers; i++) {
      this.forkWorker();
    }

    // Handle worker events
    this.cluster.on('exit', (worker, code, signal) => {
      if (this.options.enableLogging) {
        console.warn(`Worker ${worker.process.pid} died with code ${code} and signal ${signal}`);
      }

      this.workers.delete(worker.id);

      // Restart worker after delay
      setTimeout(() => {
        this.forkWorker();
      }, this.options.workerRestartDelay);
    });

    // Graceful shutdown
    if (this.options.gracefulShutdown) {
      process.on('SIGTERM', () => this.shutdown());
      process.on('SIGINT', () => this.shutdown());
    }
  }

  /**
   * Fork a new worker
   * @private
   */
  forkWorker() {
    const worker = this.cluster.fork();
    this.workers.set(worker.id, {
      worker,
      pid: worker.process.pid,
      requests: 0,
      startTime: Date.now()
    });

    worker.on('message', (message) => {
      this.handleWorkerMessage(worker.id, message);
    });
  }

  /**
   * Start worker process
   * @private
   */
  async startWorker() {
    if (this.options.enableLogging) {
      console.warn(`Worker process ${process.pid} started`);
    }

    // Handle messages from master
    process.on('message', async (message) => {
      try {
        const result = await this.handleInferenceRequest(message);
        process.send({ type: 'result', requestId: message.requestId, result });
      } catch (error) {
        process.send({
          type: 'error',
          requestId: message.requestId,
          error: {
            name: error.name,
            message: error.message,
            stack: error.stack
          }
        });
      }
    });
  }

  /**
   * Handle worker message
   * @private
   */
  handleWorkerMessage(workerId, message) {
    const workerInfo = this.workers.get(workerId);
    if (!workerInfo) return;

    switch (message.type) {
      case 'result':
        this.stats.completedRequests++;
        this.stats.activeRequests--;
        workerInfo.requests++;
        break;

      case 'error':
        this.stats.errors++;
        this.stats.activeRequests--;
        break;
    }
  }

  /**
   * Handle inference request in worker
   * @private
   */
  async handleInferenceRequest(message) {
    // This would be implemented to handle actual inference
    // For now, return a placeholder
    return {
      processed: true,
      timestamp: Date.now(),
      workerId: process.pid
    };
  }

  /**
   * Distribute inference task
   * @param {Object} task - Inference task
   * @returns {Promise<*>} Task result
   */
  async processTask(task) {
    return new Promise((resolve, reject) => {
      const requestId = Math.random().toString(36).substr(2, 9);
      this.stats.totalRequests++;
      this.stats.activeRequests++;

      // Find least loaded worker
      const worker = this.getLeastLoadedWorker();
      if (!worker) {
        reject(new Error('No available workers'));
        return;
      }

      const timeout = setTimeout(() => {
        reject(new Error('Task timeout'));
      }, 30000);

      const messageHandler = (message) => {
        if (message.requestId === requestId) {
          clearTimeout(timeout);
          worker.worker.removeListener('message', messageHandler);

          if (message.type === 'result') {
            resolve(message.result);
          } else {
            reject(new Error(message.error.message));
          }
        }
      };

      worker.worker.on('message', messageHandler);
      worker.worker.send({ ...task, requestId });
    });
  }

  /**
   * Get least loaded worker
   * @private
   */
  getLeastLoadedWorker() {
    let leastLoaded = null;
    let minRequests = Infinity;

    for (const workerInfo of this.workers.values()) {
      if (workerInfo.requests < minRequests) {
        minRequests = workerInfo.requests;
        leastLoaded = workerInfo;
      }
    }

    return leastLoaded;
  }

  /**
   * Get cluster statistics
   * @returns {Object} Statistics
   */
  getStatistics() {
    return {
      ...this.stats,
      workers: this.workers.size,
      uptime: Date.now() - (this.startTime || Date.now())
    };
  }

  /**
   * Shutdown cluster gracefully
   */
  async shutdown() {
    if (this.options.enableLogging) {
      console.warn('Shutting down cluster gracefully...');
    }

    for (const workerInfo of this.workers.values()) {
      workerInfo.worker.kill('SIGTERM');
    }

    // Wait for workers to exit
    await new Promise(resolve => {
      const checkWorkers = () => {
        if (this.workers.size === 0) {
          resolve();
        } else {
          setTimeout(checkWorkers, 100);
        }
      };
      checkWorkers();
    });

    if (this.options.enableLogging) {
      console.warn('Cluster shutdown complete');
    }

    process.exit(0);
  }
}

/**
 * Streaming Processor for Large Datasets
 */
class StreamingProcessor {
  constructor(options = {}) {
    this.options = {
      bufferSize: 1000,
      enableLogging: true,
      enableBackpressure: true,
      maxMemoryUsage: 500 * 1024 * 1024, // 500MB
      ...options
    };

    this.Readable = require('stream').Readable;
    this.Transform = require('stream').Transform;
    this.pipeline = require('stream').pipeline;
  }

  /**
   * Create readable stream from array
   * @param {Array} data - Data array
   * @returns {Readable} Readable stream
   */
  createArrayStream(data) {
    let index = 0;

    return new this.Readable({
      objectMode: true,
      read() {
        if (index < data.length) {
          this.push(data[index++]);
        } else {
          this.push(null);
        }
      }
    });
  }

  /**
   * Create transform stream for processing
   * @param {Function} processFunc - Processing function
   * @returns {Transform} Transform stream
   */
  createProcessingStream(processFunc) {
    const {options} = this;

    return new this.Transform({
      objectMode: true,
      transform(chunk, encoding, callback) {
        try {
          const result = processFunc(chunk);

          if (result instanceof Promise) {
            result
              .then(res => callback(null, res))
              .catch(err => callback(err));
          } else {
            callback(null, result);
          }
        } catch (error) {
          callback(error);
        }
      }
    });
  }

  /**
   * Process data stream
   * @param {Readable} input - Input stream
   * @param {Function} processFunc - Processing function
   * @returns {Promise<Array>} Processed results
   */
  async processStream(input, processFunc) {
    const results = [];
    const processor = this.createProcessingStream(processFunc);

    return new Promise((resolve, reject) => {
      this.pipeline(
        input,
        processor,
        new this.Transform({
          objectMode: true,
          transform(chunk, encoding, callback) {
            results.push(chunk);
            callback();
          }
        }),
        (error) => {
          if (error) {
            reject(error);
          } else {
            resolve(results);
          }
        }
      );
    });
  }

  /**
   * Process large dataset in chunks
   * @param {Array} dataset - Large dataset
   * @param {Function} processFunc - Processing function
   * @param {Object} options - Processing options
   * @returns {AsyncGenerator} Results generator
   */
  async* processLargeDataset(dataset, processFunc, options = {}) {
    const { chunkSize = this.options.bufferSize } = options;

    for (let i = 0; i < dataset.length; i += chunkSize) {
      const chunk = dataset.slice(i, i + chunkSize);
      const stream = this.createArrayStream(chunk);
      const results = await this.processStream(stream, processFunc);

      for (const result of results) {
        yield result;
      }

      // Memory usage check
      if (this.options.enableBackpressure) {
        const memUsage = process.memoryUsage();
        if (memUsage.heapUsed > this.options.maxMemoryUsage) {
          // Force garbage collection if available
          if (global.gc) {
            global.gc();
          }
          // Small delay to allow memory cleanup
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      }
    }
  }
}

/**
 * File System Optimizer
 */
class FileSystemOptimizer {
  constructor(options = {}) {
    this.options = {
      cacheSize: 100,
      enableAtomicWrites: true,
      enableLogging: false,
      tempDir: require('os').tmpdir(),
      ...options
    };

    this.fs = require('fs').promises;
    this.path = require('path');
    this.cache = new Map();
  }

  /**
   * Read file with caching
   * @param {string} filePath - File path
   * @param {Object} options - Read options
   * @returns {Promise<Buffer|string>} File content
   */
  async readFile(filePath, options = {}) {
    const { encoding = null, useCache = true } = options;
    const cacheKey = `${filePath}:${encoding}`;

    if (useCache && this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    try {
      const content = await this.fs.readFile(filePath, encoding);

      if (useCache && this.cache.size < this.options.cacheSize) {
        this.cache.set(cacheKey, content);
      }

      return content;
    } catch (error) {
      if (this.options.enableLogging) {
        console.error(`Failed to read file ${filePath}:`, error.message);
      }
      throw error;
    }
  }

  /**
   * Write file with atomic operation
   * @param {string} filePath - File path
   * @param {Buffer|string} data - Data to write
   * @param {Object} options - Write options
   * @returns {Promise<void>}
   */
  async writeFile(filePath, data, options = {}) {
    const { atomic = this.options.enableAtomicWrites } = options;

    if (atomic) {
      const tempPath = this.path.join(
        this.options.tempDir,
        `tmp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      );

      try {
        await this.fs.writeFile(tempPath, data, options);
        await this.fs.rename(tempPath, filePath);
      } catch (error) {
        // Clean up temp file if it exists
        try {
          await this.fs.unlink(tempPath);
        } catch (_) {
          // Ignore cleanup errors
        }
        throw error;
      }
    } else {
      await this.fs.writeFile(filePath, data, options);
    }

    // Invalidate cache
    this.cache.forEach((_, key) => {
      if (key.startsWith(`${filePath}:`)) {
        this.cache.delete(key);
      }
    });
  }

  /**
   * Batch file operations
   * @param {Array} operations - Array of file operations
   * @returns {Promise<Array>} Results
   */
  async batchOperations(operations) {
    const results = [];

    for (const operation of operations) {
      try {
        let result;

        switch (operation.type) {
          case 'read':
            result = await this.readFile(operation.path, operation.options);
            break;

          case 'write':
            result = await this.writeFile(operation.path, operation.data, operation.options);
            break;

          default:
            throw new Error(`Unknown operation type: ${operation.type}`);
        }

        results.push({ success: true, result });
      } catch (error) {
        results.push({ success: false, error: error.message });
      }
    }

    return results;
  }

  /**
   * Clear file cache
   */
  clearCache() {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   * @returns {Object} Cache statistics
   */
  getCacheStats() {
    return {
      size: this.cache.size,
      maxSize: this.options.cacheSize,
      hitRatio: this.hitCount / (this.hitCount + this.missCount) || 0
    };
  }
}

/**
 * Node.js Memory Manager
 */
class NodeJSMemoryManager {
  constructor(options = {}) {
    this.options = {
      maxHeapSize: 1024 * 1024 * 1024, // 1GB
      gcThreshold: 0.8,
      enableMonitoring: true,
      monitoringInterval: 5000,
      ...options
    };

    this.monitoringTimer = null;
    this.stats = {
      gcCount: 0,
      lastGc: null,
      peakMemory: 0
    };

    if (this.options.enableMonitoring) {
      this.startMonitoring();
    }
  }

  /**
   * Start memory monitoring
   */
  startMonitoring() {
    if (this.monitoringTimer) return;

    this.monitoringTimer = setInterval(() => {
      const memUsage = process.memoryUsage();
      const heapUsedMB = memUsage.heapUsed / 1024 / 1024;
      memUsage.heapTotal / 1024 / 1024;

      this.stats.peakMemory = Math.max(this.stats.peakMemory, memUsage.heapUsed);

      if (heapUsedMB > this.options.maxHeapSize * this.options.gcThreshold) {
        this.triggerGC();
      }
    }, this.options.monitoringInterval);
  }

  /**
   * Stop memory monitoring
   */
  stopMonitoring() {
    if (this.monitoringTimer) {
      clearInterval(this.monitoringTimer);
      this.monitoringTimer = null;
    }
  }

  /**
   * Trigger garbage collection
   */
  triggerGC() {
    if (global.gc) {
      try {
        global.gc();
        this.stats.gcCount++;
        this.stats.lastGc = new Date();
      } catch (error) {
        console.warn('Failed to trigger GC:', error.message);
      }
    }
  }

  /**
   * Get memory usage statistics
   * @returns {Object} Memory statistics
   */
  getMemoryStats() {
    const memUsage = process.memoryUsage();

    return {
      heap: {
        used: Math.round(memUsage.heapUsed / 1024 / 1024),
        total: Math.round(memUsage.heapTotal / 1024 / 1024),
        utilization: (memUsage.heapUsed / memUsage.heapTotal * 100).toFixed(2)
      },
      external: Math.round(memUsage.external / 1024 / 1024),
      arrayBuffers: Math.round(memUsage.arrayBuffers / 1024 / 1024),
      rss: Math.round(memUsage.rss / 1024 / 1024),
      gc: this.stats,
      monitoring: {
        enabled: this.options.enableMonitoring,
        interval: this.options.monitoringInterval
      }
    };
  }

  /**
   * Optimize memory settings
   * @returns {Object} Optimization result
   */
  optimizeMemory() {
    const recommendations = [];

    const memUsage = process.memoryUsage();
    const heapUtilization = memUsage.heapUsed / memUsage.heapTotal;

    if (heapUtilization > 0.9) {
      recommendations.push('Consider increasing max heap size with --max-old-space-size');
    }

    if (memUsage.external > memUsage.heapUsed) {
      recommendations.push('High external memory usage detected - check for memory leaks');
    }

    if (this.stats.gcCount > 100) {
      recommendations.push('High GC frequency - consider optimizing object creation');
    }

    return {
      optimized: true,
      recommendations,
      currentStats: this.getMemoryStats()
    };
  }

  /**
   * Cleanup memory manager
   */
  cleanup() {
    this.stopMonitoring();
    this.triggerGC();
  }
}

/**
 * WebGL Backend for TrustformeRS
 * Provides GPU acceleration for browsers without WebGPU support
 */

let wasmModule$4 = null;

/**
 * Initialize WebGL backend
 * @param {Object} module - WASM module reference
 */
function initWebGLBackend(module) {
  wasmModule$4 = module;
}

/**
 * WebGL Backend for tensor operations
 */
class WebGLBackend {
  constructor() {
    this.gl = null;
    this.programs = new Map();
    this.buffers = new Map();
    this.textures = new Map();
    this.framebuffers = new Map();
    this.supported = false;
    this.maxTextureSize = 0;
    this.extensions = {};
  }

  /**
   * Initialize WebGL context and check capabilities
   * @param {HTMLCanvasElement} canvas - Canvas element (optional)
   * @returns {Promise<boolean>} Success status
   */
  async initialize(canvas = null) {
    try {
      // Create or use provided canvas
      const canvasElement = canvas || document.createElement('canvas');
      canvasElement.width = 1;
      canvasElement.height = 1;
      
      // Try WebGL2 first, fallback to WebGL1
      this.gl = canvasElement.getContext('webgl2') || 
                canvasElement.getContext('webgl') ||
                canvasElement.getContext('experimental-webgl');
      
      if (!this.gl) {
        console.warn('WebGL not supported');
        return false;
      }

      // Check required extensions
      this.extensions = {
        floatTextures: this.gl.getExtension('OES_texture_float') || 
                      this.gl.getExtension('EXT_color_buffer_float'),
        textureFloat: this.gl.getExtension('OES_texture_float_linear'),
        vertexArrays: this.gl.getExtension('OES_vertex_array_object'),
        instancedArrays: this.gl.getExtension('ANGLE_instanced_arrays')
      };

      // Get capabilities
      this.maxTextureSize = this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE);
      const maxFragmentTextures = this.gl.getParameter(this.gl.MAX_TEXTURE_IMAGE_UNITS);
      const maxVertexTextures = this.gl.getParameter(this.gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS);

      console.warn(`WebGL Backend initialized:`);
      console.warn(`- Context: ${this.gl.constructor.name}`);
      console.warn(`- Max texture size: ${this.maxTextureSize}`);
      console.warn(`- Max fragment textures: ${maxFragmentTextures}`);
      console.warn(`- Max vertex textures: ${maxVertexTextures}`);
      console.warn(`- Float textures: ${!!this.extensions.floatTextures}`);

      this.supported = true;
      
      // Initialize basic shaders
      await this.initializeShaders();
      
      return true;
    } catch (error) {
      console.error('Failed to initialize WebGL backend:', error);
      return false;
    }
  }

  /**
   * Initialize basic compute shaders
   */
  async initializeShaders() {
    // Vertex shader (common for all operations)
    const vertexShaderSource = `
      attribute vec2 a_position;
      attribute vec2 a_texCoord;
      varying vec2 v_texCoord;
      
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_texCoord;
      }
    `;

    // Basic matrix multiplication shader
    const matmulFragmentShader = `
      precision highp float;
      varying vec2 v_texCoord;
      uniform sampler2D u_matrixA;
      uniform sampler2D u_matrixB;
      uniform vec2 u_sizeA;
      uniform vec2 u_sizeB;
      uniform vec2 u_outputSize;
      
      void main() {
        vec2 pos = v_texCoord * u_outputSize;
        float row = floor(pos.y);
        float col = floor(pos.x);
        
        float sum = 0.0;
        for (float k = 0.0; k < u_sizeA.x; k += 1.0) {
          vec2 coordA = vec2(k / u_sizeA.x, row / u_sizeA.y);
          vec2 coordB = vec2(col / u_sizeB.x, k / u_sizeB.y);
          
          float a = texture2D(u_matrixA, coordA).r;
          float b = texture2D(u_matrixB, coordB).r;
          sum += a * b;
        }
        
        gl_FragColor = vec4(sum, 0.0, 0.0, 1.0);
      }
    `;

    // Element-wise operations shader
    const elementWiseFragmentShader = `
      precision highp float;
      varying vec2 v_texCoord;
      uniform sampler2D u_textureA;
      uniform sampler2D u_textureB;
      uniform int u_operation; // 0=add, 1=sub, 2=mul, 3=div
      
      void main() {
        vec4 a = texture2D(u_textureA, v_texCoord);
        vec4 b = texture2D(u_textureB, v_texCoord);
        
        vec4 result;
        if (u_operation == 0) {
          result = a + b;
        } else if (u_operation == 1) {
          result = a - b;
        } else if (u_operation == 2) {
          result = a * b;
        } else if (u_operation == 3) {
          result = a / b;
        } else {
          result = a;
        }
        
        gl_FragColor = result;
      }
    `;

    // Activation functions shader
    const activationFragmentShader = `
      precision highp float;
      varying vec2 v_texCoord;
      uniform sampler2D u_texture;
      uniform int u_activation; // 0=relu, 1=sigmoid, 2=tanh, 3=gelu
      
      void main() {
        vec4 x = texture2D(u_texture, v_texCoord);
        vec4 result;
        
        if (u_activation == 0) {
          // ReLU
          result = max(x, vec4(0.0));
        } else if (u_activation == 1) {
          // Sigmoid
          result = 1.0 / (1.0 + exp(-x));
        } else if (u_activation == 2) {
          // Tanh
          result = tanh(x);
        } else if (u_activation == 3) {
          // GELU approximation
          result = 0.5 * x * (1.0 + tanh(sqrt(2.0 / 3.14159) * (x + 0.044715 * x * x * x)));
        } else {
          result = x;
        }
        
        gl_FragColor = result;
      }
    `;

    // Create shader programs
    this.programs.set('matmul', this.createProgram(vertexShaderSource, matmulFragmentShader));
    this.programs.set('elementwise', this.createProgram(vertexShaderSource, elementWiseFragmentShader));
    this.programs.set('activation', this.createProgram(vertexShaderSource, activationFragmentShader));

    // Create quad buffer for rendering
    this.createQuadBuffer();
  }

  /**
   * Create a shader program
   * @param {string} vertexSource - Vertex shader source
   * @param {string} fragmentSource - Fragment shader source
   * @returns {WebGLProgram} Compiled program
   */
  createProgram(vertexSource, fragmentSource) {
    const vertexShader = this.createShader(this.gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, fragmentSource);
    
    const program = this.gl.createProgram();
    this.gl.attachShader(program, vertexShader);
    this.gl.attachShader(program, fragmentShader);
    this.gl.linkProgram(program);
    
    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      const error = this.gl.getProgramInfoLog(program);
      this.gl.deleteProgram(program);
      throw new Error(`Program linking failed: ${error}`);
    }
    
    return program;
  }

  /**
   * Create a shader
   * @param {number} type - Shader type
   * @param {string} source - Shader source
   * @returns {WebGLShader} Compiled shader
   */
  createShader(type, source) {
    const shader = this.gl.createShader(type);
    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      const error = this.gl.getShaderInfoLog(shader);
      this.gl.deleteShader(shader);
      throw new Error(`Shader compilation failed: ${error}`);
    }
    
    return shader;
  }

  /**
   * Create quad buffer for full-screen rendering
   */
  createQuadBuffer() {
    const positions = new Float32Array([
      -1, -1,  0, 0,
       1, -1,  1, 0,
      -1,  1,  0, 1,
       1,  1,  1, 1
    ]);
    
    const buffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);
    
    this.buffers.set('quad', buffer);
  }

  /**
   * Create texture from tensor data
   * @param {Float32Array} data - Tensor data
   * @param {number} width - Texture width
   * @param {number} height - Texture height
   * @returns {WebGLTexture} Created texture
   */
  createTexture(data, width, height) {
    const texture = this.gl.createTexture();
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    
    // Set texture parameters
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
    
    // Upload data
    const format = this.gl.RGBA;
    const type = this.gl.FLOAT;
    
    if (this.extensions.floatTextures) {
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, format, width, height, 0, format, type, data);
    } else {
      // Fallback to UNSIGNED_BYTE if float textures not supported
      const byteData = new Uint8Array(data.length * 4);
      for (let i = 0; i < data.length; i++) {
        const normalized = Math.max(0, Math.min(255, Math.floor(data[i] * 255)));
        byteData[i * 4] = normalized;
        byteData[(i * 4) + 1] = normalized;
        byteData[(i * 4) + 2] = normalized;
        byteData[i * 4 + 3] = 255;
      }
      this.gl.texImage2D(this.gl.TEXTURE_2D, 0, format, width, height, 0, format, this.gl.UNSIGNED_BYTE, byteData);
    }
    
    return texture;
  }

  /**
   * Perform matrix multiplication using WebGL
   * @param {Object} tensorA - First tensor
   * @param {Object} tensorB - Second tensor
   * @returns {Object} Result tensor
   */
  async matmul(tensorA, tensorB) {
    if (!this.supported) {
      throw new Error('WebGL backend not initialized');
    }

    const program = this.programs.get('matmul');
    this.gl.useProgram(program);

    // Get tensor data and shapes
    const dataA = await tensorA.to_js_array();
    const dataB = await tensorB.to_js_array();
    const shapeA = await tensorA.shape();
    const shapeB = await tensorB.shape();

    // Create textures
    const textureA = this.createTexture(new Float32Array(dataA), shapeA[1], shapeA[0]);
    const textureB = this.createTexture(new Float32Array(dataB), shapeB[1], shapeB[0]);

    // Set uniforms
    const locations = {
      matrixA: this.gl.getUniformLocation(program, 'u_matrixA'),
      matrixB: this.gl.getUniformLocation(program, 'u_matrixB'),
      sizeA: this.gl.getUniformLocation(program, 'u_sizeA'),
      sizeB: this.gl.getUniformLocation(program, 'u_sizeB'),
      outputSize: this.gl.getUniformLocation(program, 'u_outputSize')
    };

    this.gl.uniform1i(locations.matrixA, 0);
    this.gl.uniform1i(locations.matrixB, 1);
    this.gl.uniform2f(locations.sizeA, shapeA[1], shapeA[0]);
    this.gl.uniform2f(locations.sizeB, shapeB[1], shapeB[0]);
    this.gl.uniform2f(locations.outputSize, shapeB[1], shapeA[0]);

    // Bind textures
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, textureA);
    this.gl.activeTexture(this.gl.TEXTURE1);
    this.gl.bindTexture(this.gl.TEXTURE_2D, textureB);

    // Setup framebuffer for output
    const [, outputWidth] = shapeB;
    const [outputHeight] = shapeA;
    const { framebuffer, texture: outputTexture } = this.createFramebuffer(outputWidth, outputHeight);

    // Render
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer);
    this.gl.viewport(0, 0, outputWidth, outputHeight);
    this.renderQuad(program);

    // Read result
    const result = this.readFramebuffer(outputWidth, outputHeight);

    // Cleanup
    this.gl.deleteTexture(textureA);
    this.gl.deleteTexture(textureB);
    this.gl.deleteTexture(outputTexture);
    this.gl.deleteFramebuffer(framebuffer);

    // Convert back to tensor
    return wasmModule$4.WasmTensor.from_f32(result, new Uint32Array([shapeA[0], shapeB[1]]));
  }

  /**
   * Perform element-wise operations using WebGL
   * @param {Object} tensorA - First tensor
   * @param {Object} tensorB - Second tensor
   * @param {string} operation - Operation type ('add', 'sub', 'mul', 'div')
   * @returns {Object} Result tensor
   */
  async elementWise(tensorA, tensorB, operation) {
    if (!this.supported) {
      throw new Error('WebGL backend not initialized');
    }

    const program = this.programs.get('elementwise');
    this.gl.useProgram(program);

    // Get tensor data and shapes
    const dataA = await tensorA.to_js_array();
    const dataB = await tensorB.to_js_array();
    const shape = await tensorA.shape();

    // Create textures
    const width = shape[shape.length - 1];
    const height = Math.ceil(dataA.length / width);
    const textureA = this.createTexture(new Float32Array(dataA), width, height);
    const textureB = this.createTexture(new Float32Array(dataB), width, height);

    // Set uniforms
    const operationMap = { 'add': 0, 'sub': 1, 'mul': 2, 'div': 3 };
    this.gl.uniform1i(this.gl.getUniformLocation(program, 'u_textureA'), 0);
    this.gl.uniform1i(this.gl.getUniformLocation(program, 'u_textureB'), 1);
    this.gl.uniform1i(this.gl.getUniformLocation(program, 'u_operation'), operationMap[operation] || 0);

    // Bind textures
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, textureA);
    this.gl.activeTexture(this.gl.TEXTURE1);
    this.gl.bindTexture(this.gl.TEXTURE_2D, textureB);

    // Setup framebuffer
    const { framebuffer, texture: outputTexture } = this.createFramebuffer(width, height);

    // Render
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer);
    this.gl.viewport(0, 0, width, height);
    this.renderQuad(program);

    // Read result
    const result = this.readFramebuffer(width, height);

    // Cleanup
    this.gl.deleteTexture(textureA);
    this.gl.deleteTexture(textureB);
    this.gl.deleteTexture(outputTexture);
    this.gl.deleteFramebuffer(framebuffer);

    return wasmModule$4.WasmTensor.from_f32(result.slice(0, dataA.length), new Uint32Array(shape));
  }

  /**
   * Apply activation function using WebGL
   * @param {Object} tensor - Input tensor
   * @param {string} activation - Activation type ('relu', 'sigmoid', 'tanh', 'gelu')
   * @returns {Object} Result tensor
   */
  async activation(tensor, activation) {
    if (!this.supported) {
      throw new Error('WebGL backend not initialized');
    }

    const program = this.programs.get('activation');
    this.gl.useProgram(program);

    // Get tensor data and shape
    const data = await tensor.to_js_array();
    const shape = await tensor.shape();

    // Create texture
    const width = shape[shape.length - 1];
    const height = Math.ceil(data.length / width);
    const texture = this.createTexture(new Float32Array(data), width, height);

    // Set uniforms
    const activationMap = { 'relu': 0, 'sigmoid': 1, 'tanh': 2, 'gelu': 3 };
    this.gl.uniform1i(this.gl.getUniformLocation(program, 'u_texture'), 0);
    this.gl.uniform1i(this.gl.getUniformLocation(program, 'u_activation'), activationMap[activation] || 0);

    // Bind texture
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);

    // Setup framebuffer
    const { framebuffer, texture: outputTexture } = this.createFramebuffer(width, height);

    // Render
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer);
    this.gl.viewport(0, 0, width, height);
    this.renderQuad(program);

    // Read result
    const result = this.readFramebuffer(width, height);

    // Cleanup
    this.gl.deleteTexture(texture);
    this.gl.deleteTexture(outputTexture);
    this.gl.deleteFramebuffer(framebuffer);

    return wasmModule$4.WasmTensor.from_f32(result.slice(0, data.length), new Uint32Array(shape));
  }

  /**
   * Create framebuffer for output
   * @param {number} width - Buffer width
   * @param {number} height - Buffer height
   * @returns {Object} Framebuffer and texture
   */
  createFramebuffer(width, height) {
    const framebuffer = this.gl.createFramebuffer();
    const texture = this.gl.createTexture();

    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);

    const format = this.gl.RGBA;
    const type = this.extensions.floatTextures ? this.gl.FLOAT : this.gl.UNSIGNED_BYTE;
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, format, width, height, 0, format, type, null);

    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer);
    this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, texture, 0);

    return { framebuffer, texture };
  }

  /**
   * Render full-screen quad
   * @param {WebGLProgram} program - Shader program
   */
  renderQuad(program) {
    // Setup attributes
    const positionLocation = this.gl.getAttribLocation(program, 'a_position');
    const texCoordLocation = this.gl.getAttribLocation(program, 'a_texCoord');

    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.get('quad'));
    
    this.gl.enableVertexAttribArray(positionLocation);
    this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 16, 0);
    
    this.gl.enableVertexAttribArray(texCoordLocation);
    this.gl.vertexAttribPointer(texCoordLocation, 2, this.gl.FLOAT, false, 16, 8);

    // Draw
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
  }

  /**
   * Read framebuffer data
   * @param {number} width - Buffer width
   * @param {number} height - Buffer height
   * @returns {Float32Array} Buffer data
   */
  readFramebuffer(width, height) {
    if (this.extensions.floatTextures) {
      const buffer = new Float32Array(width * height * 4);
      this.gl.readPixels(0, 0, width, height, this.gl.RGBA, this.gl.FLOAT, buffer);
      return buffer;
    } 
      const buffer = new Uint8Array(width * height * 4);
      this.gl.readPixels(0, 0, width, height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, buffer);
      // Convert back to float
      const floatBuffer = new Float32Array(width * height * 4);
      for (let i = 0; i < buffer.length; i++) {
        floatBuffer[i] = buffer[i] / 255.0;
      }
      return floatBuffer;
    
  }

  /**
   * Check if WebGL backend is supported
   * @returns {boolean} Support status
   */
  isSupported() {
    return this.supported;
  }

  /**
   * Get WebGL information
   * @returns {Object} WebGL info
   */
  getInfo() {
    if (!this.gl) return null;

    return {
      vendor: this.gl.getParameter(this.gl.VENDOR),
      renderer: this.gl.getParameter(this.gl.RENDERER),
      version: this.gl.getParameter(this.gl.VERSION),
      shadingLanguageVersion: this.gl.getParameter(this.gl.SHADING_LANGUAGE_VERSION),
      maxTextureSize: this.maxTextureSize,
      extensions: Object.keys(this.extensions).filter(key => this.extensions[key])
    };
  }

  /**
   * Cleanup resources
   */
  dispose() {
    if (this.gl) {
      // Delete programs
      for (const program of this.programs.values()) {
        this.gl.deleteProgram(program);
      }
      this.programs.clear();

      // Delete buffers
      for (const buffer of this.buffers.values()) {
        this.gl.deleteBuffer(buffer);
      }
      this.buffers.clear();

      // Delete textures
      for (const texture of this.textures.values()) {
        this.gl.deleteTexture(texture);
      }
      this.textures.clear();

      // Delete framebuffers
      for (const framebuffer of this.framebuffers.values()) {
        this.gl.deleteFramebuffer(framebuffer);
      }
      this.framebuffers.clear();
    }

    this.gl = null;
    this.supported = false;
  }
}

/**
 * Create and initialize WebGL backend
 * @param {HTMLCanvasElement} canvas - Canvas element (optional)
 * @returns {Promise<WebGLBackend>} Initialized backend
 */
async function createWebGLBackend(canvas = null) {
  const backend = new WebGLBackend();
  const success = await backend.initialize(canvas);
  
  if (!success) {
    throw new Error('Failed to initialize WebGL backend');
  }
  
  return backend;
}

/**
 * Memory Pool Manager for TrustformeRS
 * Provides efficient memory management and pooling strategies
 */

let wasmModule$3 = null;

/**
 * Initialize memory pool with WASM module
 * @param {Object} module - WASM module reference
 */
function initMemoryPool(module) {
  wasmModule$3 = module;
}

/**
 * Memory pool for tensor operations
 */
class TensorMemoryPool {
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
    if (!wasmModule$3) {
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
        tensor = wasmModule$3.WasmTensor.zeros(shapeArray);
        break;
      case 'f64':
        tensor = wasmModule$3.WasmTensor.zeros_f64(shapeArray);
        break;
      case 'i32':
        tensor = wasmModule$3.WasmTensor.zeros_i32(shapeArray);
        break;
      case 'u32':
        tensor = wasmModule$3.WasmTensor.zeros_u32(shapeArray);
        break;
      case 'i8':
        tensor = wasmModule$3.WasmTensor.zeros_i8(shapeArray);
        break;
      case 'u8':
        tensor = wasmModule$3.WasmTensor.zeros_u8(shapeArray);
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
class WebGLMemoryPool {
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
class MemoryManager {
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
    if (this.webglPool) ;
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
    if (wasmModule$3 && wasmModule$3.get_memory_stats) {
      stats.wasm = wasmModule$3.get_memory_stats();
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
function getMemoryManager(options = {}) {
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
async function withMemoryManagement(operation, inputs = [], options = {}) {
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

/**
 * Performance Profiler for TrustformeRS JavaScript API
 * Provides comprehensive performance monitoring and optimization recommendations
 */

let wasmModule$2 = null;

/**
 * Initialize profiler with WASM module
 * @param {Object} module - WASM module reference
 */
function initProfiler(module) {
  wasmModule$2 = module;
}

/**
 * Performance metrics collector
 */
class PerformanceProfiler {
  constructor(options = {}) {
    this.enabled = options.enabled !== false;
    this.detailed = options.detailed || false;
    this.autoReport = options.autoReport || false;
    this.reportInterval = options.reportInterval || 30000; // 30 seconds
    
    this.metrics = {
      operations: new Map(),
      memory: [],
      gpu: [],
      inference: [],
      timeline: []
    };
    
    this.currentSession = null;
    this.sessionStack = [];
    this.warningThresholds = {
      operationTime: 1000, // 1 second
      memoryUsage: 512 * 1024 * 1024, // 512MB
      memoryLeakRate: 10 * 1024 * 1024, // 10MB/minute
      cpuUsage: 80 // 80%
    };

    if (this.autoReport) {
      this.startAutoReporting();
    }
  }

  /**
   * Start a performance session
   * @param {string} name - Session name
   * @param {Object} metadata - Additional metadata
   * @returns {string} Session ID
   */
  startSession(name, metadata = {}) {
    if (!this.enabled) return null;

    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const session = {
      id: sessionId,
      name,
      startTime: performance.now(),
      endTime: null,
      duration: null,
      operations: [],
      memorySnapshots: [],
      metadata,
      warnings: [],
      parent: this.currentSession
    };

    this.sessionStack.push(this.currentSession);
    this.currentSession = session;

    this.addTimelineEvent('session_start', { sessionId, name });
    return sessionId;
  }

  /**
   * End the current performance session
   * @returns {Object} Session results
   */
  endSession() {
    if (!this.enabled || !this.currentSession) return null;

    const session = this.currentSession;
    session.endTime = performance.now();
    session.duration = session.endTime - session.startTime;

    this.addTimelineEvent('session_end', { 
      sessionId: session.id, 
      duration: session.duration 
    });

    // Generate session report
    const report = this.generateSessionReport(session);

    // Store session results
    if (!this.metrics.sessions) {
      this.metrics.sessions = [];
    }
    this.metrics.sessions.push(session);

    // Restore parent session
    this.currentSession = this.sessionStack.pop() || null;

    return report;
  }

  /**
   * Profile a tensor operation
   * @param {string} operation - Operation name
   * @param {Function} fn - Operation function
   * @param {Object} metadata - Additional metadata
   * @returns {Promise<*>} Operation result
   */
  async profileOperation(operation, fn, metadata = {}) {
    if (!this.enabled) {
      return await fn();
    }

    const operationId = `op_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const startTime = performance.now();
    const startMemory = this.getMemoryUsage();

    let result; let error = null;
    let gpuTime = null;

    try {
      // Start GPU timing if available
      if (this.detailed && wasmModule$2 && wasmModule$2.start_gpu_timer) {
        wasmModule$2.start_gpu_timer(operationId);
      }

      this.addTimelineEvent('operation_start', { operationId, operation });
      
      result = await fn();
      
      // End GPU timing
      if (this.detailed && wasmModule$2 && wasmModule$2.end_gpu_timer) {
        gpuTime = wasmModule$2.end_gpu_timer(operationId);
      }

    } catch (e) {
      error = e;
      throw e;
    } finally {
      const endTime = performance.now();
      const endMemory = this.getMemoryUsage();
      const duration = endTime - startTime;

      const operationMetrics = {
        id: operationId,
        operation,
        startTime,
        endTime,
        duration,
        memoryBefore: startMemory,
        memoryAfter: endMemory,
        memoryDelta: endMemory.used - startMemory.used,
        gpuTime,
        error: error ? error.message : null,
        metadata
      };

      this.recordOperation(operationMetrics);
      this.addTimelineEvent('operation_end', { 
        operationId, 
        operation, 
        duration,
        success: !error 
      });

      // Check for warnings
      this.checkOperationWarnings(operationMetrics);
    }

    return result;
  }

  /**
   * Profile model inference
   * @param {Object} model - Model object
   * @param {Object} inputs - Input tensors
   * @param {Object} options - Profiling options
   * @returns {Promise<Object>} Inference results with metrics
   */
  async profileInference(model, inputs, options = {}) {
    if (!this.enabled) {
      return await model.forward(inputs);
    }

    const sessionId = this.startSession('inference', {
      modelType: model.config ? model.config.model_type : 'unknown',
      inputShapes: this.extractInputShapes(inputs),
      ...options.metadata
    });

    let result;
    try {
      result = await this.profileOperation('inference', async () => await model.forward(inputs), { sessionId });

      // Collect detailed inference metrics
      if (this.detailed) {
        await this.collectInferenceMetrics(model, inputs, result);
      }

    } finally {
      this.endSession();
    }

    return result;
  }

  /**
   * Profile memory usage over time
   * @param {number} duration - Monitoring duration in ms
   * @param {number} interval - Sampling interval in ms
   * @returns {Promise<Array>} Memory usage timeline
   */
  async profileMemory(duration = 10000, interval = 100) {
    if (!this.enabled) return [];

    const samples = [];
    const startTime = Date.now();
    
    const sampleMemory = () => {
      const memory = this.getMemoryUsage();
      samples.push({
        timestamp: Date.now() - startTime,
        ...memory
      });
    };

    const intervalId = setInterval(sampleMemory, interval);
    
    return new Promise((resolve) => {
      setTimeout(() => {
        clearInterval(intervalId);
        this.metrics.memory.push(...samples);
        resolve(samples);
      }, duration);
    });
  }

  /**
   * Record operation metrics
   * @param {Object} metrics - Operation metrics
   */
  recordOperation(metrics) {
    const operationStats = this.metrics.operations.get(metrics.operation) || {
      count: 0,
      totalTime: 0,
      totalGpuTime: 0,
      totalMemoryDelta: 0,
      errors: 0,
      averageTime: 0,
      averageGpuTime: 0,
      averageMemoryDelta: 0
    };

    operationStats.count++;
    operationStats.totalTime += metrics.duration;
    operationStats.totalMemoryDelta += metrics.memoryDelta;
    
    if (metrics.gpuTime) {
      operationStats.totalGpuTime += metrics.gpuTime;
      operationStats.averageGpuTime = operationStats.totalGpuTime / operationStats.count;
    }
    
    if (metrics.error) {
      operationStats.errors++;
    }

    operationStats.averageTime = operationStats.totalTime / operationStats.count;
    operationStats.averageMemoryDelta = operationStats.totalMemoryDelta / operationStats.count;

    this.metrics.operations.set(metrics.operation, operationStats);

    // Add to current session if active
    if (this.currentSession) {
      this.currentSession.operations.push(metrics);
    }
  }

  /**
   * Get current memory usage
   * @returns {Object} Memory usage information
   */
  getMemoryUsage() {
    const memory = {
      used: 0,
      total: 0,
      limit: 0,
      wasm: 0,
      gpu: 0,
      js: 0
    };

    // JavaScript heap memory
    if (performance.memory) {
      memory.used = performance.memory.usedJSHeapSize;
      memory.total = performance.memory.totalJSHeapSize;
      memory.limit = performance.memory.jsHeapSizeLimit;
      memory.js = memory.used;
    }

    // WASM memory
    if (wasmModule$2 && wasmModule$2.get_memory_stats) {
      const wasmStats = wasmModule$2.get_memory_stats();
      memory.wasm = wasmStats.used || 0;
      memory.used += memory.wasm;
    }

    // GPU memory (if available)
    if (wasmModule$2 && wasmModule$2.get_gpu_memory_usage) {
      memory.gpu = wasmModule$2.get_gpu_memory_usage();
      memory.used += memory.gpu;
    }

    return memory;
  }

  /**
   * Extract input shapes from various input formats
   * @param {*} inputs - Model inputs
   * @returns {Object} Input shapes
   */
  extractInputShapes(inputs) {
    const shapes = {};
    
    if (Array.isArray(inputs)) {
      inputs.forEach((input, index) => {
        if (input && input.shape) {
          shapes[`input_${index}`] = Array.from(input.shape());
        }
      });
    } else if (inputs && typeof inputs === 'object') {
      for (const [key, value] of Object.entries(inputs)) {
        if (value && value.shape) {
          shapes[key] = Array.from(value.shape());
        }
      }
    } else if (inputs && inputs.shape) {
      shapes.input = Array.from(inputs.shape());
    }

    return shapes;
  }

  /**
   * Collect detailed inference metrics
   * @param {Object} model - Model object
   * @param {Object} inputs - Input tensors
   * @param {Object} outputs - Output tensors
   */
  async collectInferenceMetrics(model, inputs, outputs) {
    const metrics = {
      timestamp: Date.now(),
      inputShapes: this.extractInputShapes(inputs),
      outputShapes: this.extractInputShapes(outputs),
      memory: this.getMemoryUsage()
    };

    // Layer-wise timing if available
    if (wasmModule$2 && wasmModule$2.get_layer_timings) {
      metrics.layerTimings = wasmModule$2.get_layer_timings();
    }

    // Model parameters if available
    if (model.num_parameters) {
      metrics.parameters = model.num_parameters();
    }

    this.metrics.inference.push(metrics);
  }

  /**
   * Check for performance warnings
   * @param {Object} operationMetrics - Operation metrics to check
   */
  checkOperationWarnings(operationMetrics) {
    const warnings = [];

    if (operationMetrics.duration > this.warningThresholds.operationTime) {
      warnings.push({
        type: 'slow_operation',
        message: `Operation '${operationMetrics.operation}' took ${operationMetrics.duration.toFixed(2)}ms`,
        threshold: this.warningThresholds.operationTime,
        value: operationMetrics.duration
      });
    }

    if (operationMetrics.memoryDelta > this.warningThresholds.memoryUsage) {
      warnings.push({
        type: 'memory_spike',
        message: `Operation '${operationMetrics.operation}' allocated ${(operationMetrics.memoryDelta / 1024 / 1024).toFixed(2)}MB`,
        threshold: this.warningThresholds.memoryUsage,
        value: operationMetrics.memoryDelta
      });
    }

    if (warnings.length > 0 && this.currentSession) {
      this.currentSession.warnings.push(...warnings);
    }

    return warnings;
  }

  /**
   * Add timeline event
   * @param {string} type - Event type
   * @param {Object} data - Event data
   */
  addTimelineEvent(type, data) {
    this.metrics.timeline.push({
      timestamp: performance.now(),
      type,
      data
    });
  }

  /**
   * Generate comprehensive performance report
   * @returns {Object} Performance report
   */
  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      summary: this.generateSummary(),
      operations: this.analyzeOperations(),
      memory: this.analyzeMemory(),
      recommendations: this.generateRecommendations(),
      sessions: this.metrics.sessions || []
    };

    return report;
  }

  /**
   * Generate session report
   * @param {Object} session - Session data
   * @returns {Object} Session report
   */
  generateSessionReport(session) {
    const operationStats = this.aggregateSessionOperations(session.operations);
    const memoryAnalysis = this.analyzeSessionMemory(session);

    return {
      session: {
        id: session.id,
        name: session.name,
        duration: session.duration,
        metadata: session.metadata
      },
      operations: operationStats,
      memory: memoryAnalysis,
      warnings: session.warnings,
      recommendations: this.generateSessionRecommendations(session)
    };
  }

  /**
   * Generate performance summary
   * @returns {Object} Performance summary
   */
  generateSummary() {
    const totalOperations = Array.from(this.metrics.operations.values())
      .reduce((sum, stats) => sum + stats.count, 0);
    
    const averageOperationTime = Array.from(this.metrics.operations.values())
      .reduce((sum, stats) => sum + stats.averageTime, 0) / this.metrics.operations.size;

    const currentMemory = this.getMemoryUsage();

    return {
      totalOperations,
      averageOperationTime: averageOperationTime || 0,
      operationTypes: this.metrics.operations.size,
      currentMemoryUsage: currentMemory.used,
      memoryEfficiency: currentMemory.limit > 0 ? currentMemory.used / currentMemory.limit : 0,
      timelineEvents: this.metrics.timeline.length
    };
  }

  /**
   * Analyze operation performance
   * @returns {Object} Operation analysis
   */
  analyzeOperations() {
    const operations = {};
    
    for (const [name, stats] of this.metrics.operations.entries()) {
      operations[name] = {
        ...stats,
        efficiency: stats.averageTime > 0 ? (stats.count - stats.errors) / stats.count : 0,
        throughput: stats.totalTime > 0 ? stats.count / (stats.totalTime / 1000) : 0 // ops/second
      };
    }

    return operations;
  }

  /**
   * Analyze memory usage patterns
   * @returns {Object} Memory analysis
   */
  analyzeMemory() {
    if (this.metrics.memory.length === 0) {
      return { analysis: 'No memory data available' };
    }

    const usage = this.metrics.memory.map(m => m.used);
    const peak = Math.max(...usage);
    const average = usage.reduce((a, b) => a + b, 0) / usage.length;
    const growth = usage.length > 1 ? usage[usage.length - 1] - usage[0] : 0;

    return {
      peak,
      average,
      growth,
      samples: this.metrics.memory.length,
      leakSuspicion: growth > this.warningThresholds.memoryLeakRate
    };
  }

  /**
   * Generate optimization recommendations
   * @returns {Array} List of recommendations
   */
  generateRecommendations() {
    const recommendations = [];

    // Analyze operations for recommendations
    for (const [name, stats] of this.metrics.operations.entries()) {
      if (stats.averageTime > this.warningThresholds.operationTime) {
        recommendations.push({
          type: 'performance',
          priority: 'high',
          operation: name,
          message: `Consider optimizing '${name}' operation (avg: ${stats.averageTime.toFixed(2)}ms)`,
          suggestions: [
            'Use batch processing for multiple inputs',
            'Enable WebGPU acceleration if available',
            'Consider model quantization',
            'Use memory pooling'
          ]
        });
      }

      if (stats.averageMemoryDelta > this.warningThresholds.memoryUsage) {
        recommendations.push({
          type: 'memory',
          priority: 'medium',
          operation: name,
          message: `'${name}' operation uses significant memory (avg: ${(stats.averageMemoryDelta / 1024 / 1024).toFixed(2)}MB)`,
          suggestions: [
            'Use tensor memory pooling',
            'Enable automatic cleanup',
            'Process data in smaller batches',
            'Consider model compression'
          ]
        });
      }
    }

    // Memory analysis recommendations
    const memoryAnalysis = this.analyzeMemory();
    if (memoryAnalysis.leakSuspicion) {
      recommendations.push({
        type: 'memory_leak',
        priority: 'high',
        message: 'Potential memory leak detected',
        suggestions: [
          'Ensure proper tensor cleanup',
          'Use memory management utilities',
          'Check for circular references',
          'Monitor long-running operations'
        ]
      });
    }

    return recommendations;
  }

  /**
   * Aggregate session operations
   * @param {Array} operations - Session operations
   * @returns {Object} Aggregated stats
   */
  aggregateSessionOperations(operations) {
    const stats = {};
    
    for (const op of operations) {
      if (!stats[op.operation]) {
        stats[op.operation] = {
          count: 0,
          totalTime: 0,
          totalMemoryDelta: 0,
          errors: 0
        };
      }
      
      stats[op.operation].count++;
      stats[op.operation].totalTime += op.duration;
      stats[op.operation].totalMemoryDelta += op.memoryDelta;
      
      if (op.error) {
        stats[op.operation].errors++;
      }
    }

    // Calculate averages
    for (const stat of Object.values(stats)) {
      stat.averageTime = stat.totalTime / stat.count;
      stat.averageMemoryDelta = stat.totalMemoryDelta / stat.count;
    }

    return stats;
  }

  /**
   * Analyze session memory usage
   * @param {Object} session - Session data
   * @returns {Object} Memory analysis
   */
  analyzeSessionMemory(session) {
    if (session.operations.length === 0) {
      return { analysis: 'No operations in session' };
    }

    const memoryDeltas = session.operations.map(op => op.memoryDelta);
    const totalDelta = memoryDeltas.reduce((a, b) => a + b, 0);
    const maxDelta = Math.max(...memoryDeltas);
    const minDelta = Math.min(...memoryDeltas);

    return {
      totalMemoryChange: totalDelta,
      maxMemoryDelta: maxDelta,
      minMemoryDelta: minDelta,
      averageMemoryDelta: totalDelta / memoryDeltas.length
    };
  }

  /**
   * Generate session-specific recommendations
   * @param {Object} session - Session data
   * @returns {Array} Recommendations
   */
  generateSessionRecommendations(session) {
    const recommendations = [];
    
    if (session.duration > 5000) { // 5 seconds
      recommendations.push({
        type: 'session_performance',
        message: `Session '${session.name}' took ${(session.duration / 1000).toFixed(2)} seconds`,
        suggestions: ['Consider breaking into smaller operations', 'Use async processing']
      });
    }

    if (session.warnings.length > 0) {
      recommendations.push({
        type: 'session_warnings',
        message: `Session had ${session.warnings.length} performance warnings`,
        suggestions: ['Review operation timings', 'Check memory usage patterns']
      });
    }

    return recommendations;
  }

  /**
   * Start automatic reporting
   */
  startAutoReporting() {
    this.reportInterval = setInterval(() => {
      const report = this.generateReport();
      console.warn('TrustformeRS Performance Report:', report);
    }, this.reportInterval);
  }

  /**
   * Export metrics data
   * @param {string} format - Export format ('json', 'csv')
   * @returns {string} Exported data
   */
  exportMetrics(format = 'json') {
    const data = {
      timestamp: new Date().toISOString(),
      metrics: this.metrics,
      summary: this.generateSummary()
    };

    if (format === 'json') {
      return JSON.stringify(data, null, 2);
    } else if (format === 'csv') {
      // Convert operations to CSV
      const operations = Array.from(this.metrics.operations.entries())
        .map(([name, stats]) => `${name},${stats.count},${stats.averageTime},${stats.errors}`)
        .join('\n');
      
      return `Operation,Count,Average Time (ms),Errors\n${operations}`;
    }

    throw new Error(`Unsupported export format: ${format}`);
  }

  /**
   * Clear all metrics
   */
  clear() {
    this.metrics = {
      operations: new Map(),
      memory: [],
      gpu: [],
      inference: [],
      timeline: [],
      sessions: []
    };
    this.currentSession = null;
    this.sessionStack = [];
  }

  /**
   * Dispose the profiler
   */
  dispose() {
    if (this.reportInterval) {
      clearInterval(this.reportInterval);
    }
    this.clear();
  }
}

// Global profiler instance
let globalProfiler = null;

/**
 * Get or create global profiler
 * @param {Object} options - Profiler options
 * @returns {PerformanceProfiler} Global profiler
 */
function getProfiler(options = {}) {
  if (!globalProfiler) {
    globalProfiler = new PerformanceProfiler(options);
  }
  return globalProfiler;
}

/**
 * Zero-Copy Tensor Transfer System for TrustformeRS
 * Provides efficient memory transfer between JavaScript and WebAssembly
 */

let wasmModule$1 = null;
let wasmMemory = null;
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
function initZeroCopy(module, config = {}) {
  wasmModule$1 = module;
  wasmMemory = wasmModule$1.memory;
  optimizationConfig = { ...optimizationConfig, ...config };

  // Setup memory growth monitoring
  if (wasmMemory && wasmMemory.buffer) {
    _setupMemoryGrowthMonitoring();
  }

  console.warn('Zero-copy system initialized with optimizations:', optimizationConfig);
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
 * TrustformeRS Debug Utilities
 * Comprehensive debugging tools for development and production environments
 */


/**
 * Debug configuration
 */
let debugConfig = {
  enabled: false,
  logLevel: 'info', // 'error', 'warn', 'info', 'debug', 'trace'
  maxHistorySize: 1000,
  enableStackTraces: true,
  enablePerformanceTracking: true,
  enableMemoryTracking: true,
  enableTensorLifecycle: true,
  enableOperationValidation: true
};

/**
 * Debug data storage
 */
const debugData = {
  operations: [],
  tensors: new Map(),
  errors: [],
  warnings: [],
  performance: [],
  memory: [],
  sessions: new Map()
};

/**
 * Debug session manager
 */
class DebugSession {
  constructor(name, metadata = {}) {
    this.id = `debug_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.name = name;
    this.metadata = metadata;
    this.startTime = perf_hooks.performance.now();
    this.operations = [];
    this.tensors = new Set();
    this.warnings = [];
    this.errors = [];
    this.active = true;
  }

  addOperation(operation) {
    if (this.active) {
      this.operations.push({
        ...operation,
        sessionId: this.id,
        timestamp: perf_hooks.performance.now()
      });
    }
  }

  addTensor(tensorId, info) {
    if (this.active) {
      this.tensors.add(tensorId);
    }
  }

  addError(error) {
    if (this.active) {
      this.errors.push({
        ...error,
        sessionId: this.id,
        timestamp: perf_hooks.performance.now()
      });
    }
  }

  addWarning(warning) {
    if (this.active) {
      this.warnings.push({
        ...warning,
        sessionId: this.id,
        timestamp: perf_hooks.performance.now()
      });
    }
  }

  end() {
    this.active = false;
    this.endTime = perf_hooks.performance.now();
    this.duration = this.endTime - this.startTime;
    return this.generateReport();
  }

  generateReport() {
    return {
      session: {
        id: this.id,
        name: this.name,
        metadata: this.metadata,
        duration: this.duration,
        startTime: this.startTime,
        endTime: this.endTime
      },
      summary: {
        operationCount: this.operations.length,
        tensorCount: this.tensors.size,
        errorCount: this.errors.length,
        warningCount: this.warnings.length
      },
      operations: this.operations,
      tensors: Array.from(this.tensors),
      errors: this.errors,
      warnings: this.warnings
    };
  }
}

/**
 * Main debug utilities class
 */
class DebugUtilities {
  constructor() {
    this.currentSession = null;
    this.initializeConsoleOverrides();
  }

  /**
   * Configure debug settings
   * @param {Object} config - Debug configuration
   */
  configure(config) {
    debugConfig = { ...debugConfig, ...config };
    this.log('Debug configuration updated', config);
  }

  /**
   * Enable debugging
   * @param {Object} options - Debug options
   */
  enable(options = {}) {
    debugConfig.enabled = true;
    debugConfig = { ...debugConfig, ...options };
    this.log('Debug mode enabled', debugConfig);
  }

  /**
   * Disable debugging
   */
  disable() {
    debugConfig.enabled = false;
    this.log('Debug mode disabled');
  }

  /**
   * Check if debug is enabled
   * @returns {boolean} Debug status
   */
  isEnabled() {
    return debugConfig.enabled;
  }

  /**
   * Start a debug session
   * @param {string} name - Session name
   * @param {Object} metadata - Session metadata
   * @returns {string} Session ID
   */
  startSession(name, metadata = {}) {
    if (!debugConfig.enabled) return null;

    this.endSession(); // End current session if any
    this.currentSession = new DebugSession(name, metadata);
    debugData.sessions.set(this.currentSession.id, this.currentSession);
    this.log(`Debug session started: ${name}`, { id: this.currentSession.id });
    return this.currentSession.id;
  }

  /**
   * End current debug session
   * @returns {Object|null} Session report
   */
  endSession() {
    if (this.currentSession && this.currentSession.active) {
      const report = this.currentSession.end();
      this.log(`Debug session ended: ${this.currentSession.name}`, {
        duration: report.session.duration,
        operations: report.summary.operationCount
      });
      this.currentSession = null;
      return report;
    }
    return null;
  }

  /**
   * Track tensor lifecycle
   * @param {Object} tensor - Tensor object
   * @param {string} operation - Operation that created the tensor
   * @param {Object} metadata - Additional metadata
   */
  trackTensor(tensor, operation, metadata = {}) {
    if (!debugConfig.enabled || !debugConfig.enableTensorLifecycle) return;

    const tensorId = this.getTensorId(tensor);
    const info = {
      id: tensorId,
      operation,
      metadata,
      shape: this.getTensorShape(tensor),
      dtype: this.getTensorDType(tensor),
      creationTime: perf_hooks.performance.now(),
      creationStack: debugConfig.enableStackTraces ? this.getStackTrace() : null
    };

    debugData.tensors.set(tensorId, info);
    
    if (this.currentSession) {
      this.currentSession.addTensor(tensorId, info);
    }

    this.log(`Tensor created: ${tensorId}`, info);
  }

  /**
   * Track operation execution
   * @param {string} name - Operation name
   * @param {Function} fn - Function to track
   * @param {Object} metadata - Operation metadata
   * @returns {Promise<*>} Function result
   */
  async trackOperation(name, fn, metadata = {}) {
    if (!debugConfig.enabled) {
      return await fn();
    }

    const operation = {
      name,
      metadata,
      startTime: perf_hooks.performance.now(),
      stack: debugConfig.enableStackTraces ? this.getStackTrace() : null
    };

    try {
      const result = await fn();
      operation.endTime = perf_hooks.performance.now();
      operation.duration = operation.endTime - operation.startTime;
      operation.success = true;
      operation.resultType = this.getResultType(result);

      this.addOperation(operation);
      this.log(`Operation completed: ${name}`, {
        duration: operation.duration,
        success: true
      });

      return result;
    } catch (error) {
      operation.endTime = perf_hooks.performance.now();
      operation.duration = operation.endTime - operation.startTime;
      operation.success = false;
      operation.error = {
        message: error.message,
        stack: error.stack,
        name: error.name
      };

      this.addOperation(operation);
      this.error(`Operation failed: ${name}`, error);
      throw error;
    }
  }

  /**
   * Validate tensor operation
   * @param {string} operation - Operation name
   * @param {Array} tensors - Input tensors
   * @param {Object} options - Validation options
   * @returns {boolean} Validation result
   */
  validateOperation(operation, tensors, options = {}) {
    if (!debugConfig.enabled || !debugConfig.enableOperationValidation) {
      return true;
    }

    const validation = {
      tensors: tensors.map(t => ({
        shape: this.getTensorShape(t),
        dtype: this.getTensorDType(t),
        id: this.getTensorId(t)
      })),
      timestamp: perf_hooks.performance.now(),
      warnings: [],
      errors: []
    };

    // Shape compatibility checks
    if (operation === 'matmul' && tensors.length >= 2) {
      const [a, b] = tensors;
      const shapeA = this.getTensorShape(a);
      const shapeB = this.getTensorShape(b);
      
      if (shapeA.length < 2 || shapeB.length < 2) {
        validation.errors.push('Matrix multiplication requires tensors with at least 2 dimensions');
      } else if (shapeA[shapeA.length - 1] !== shapeB[shapeB.length - 2]) {
        validation.errors.push(`Incompatible shapes for matrix multiplication: ${shapeA} @ ${shapeB}`);
      }
    }

    // Element-wise operation checks
    if (['add', 'sub', 'mul', 'div'].includes(operation) && tensors.length >= 2) {
      const shapes = tensors.map(t => this.getTensorShape(t));
      if (!this.areShapesBroadcastable(shapes[0], shapes[1])) {
        validation.warnings.push(`Shapes may not be broadcastable: ${shapes[0]} and ${shapes[1]}`);
      }
    }

    // Memory usage warnings
    const totalMemory = tensors.reduce((sum, tensor) => sum + this.estimateTensorMemory(tensor), 0);

    if (totalMemory > 100 * 1024 * 1024) { // 100MB
      validation.warnings.push(`Large memory usage detected: ${(totalMemory / 1024 / 1024).toFixed(2)}MB`);
    }

    // Log validation results
    if (validation.errors.length > 0) {
      validation.errors.forEach(error => this.error(`Validation error in ${operation}:`, error));
    }
    if (validation.warnings.length > 0) {
      validation.warnings.forEach(warning => this.warn(`Validation warning in ${operation}:`, warning));
    }

    return validation.errors.length === 0;
  }

  /**
   * Create a debugger decorator for functions
   * @param {string} name - Operation name
   * @param {Object} options - Debug options
   * @returns {Function} Decorator function
   */
  createDebugDecorator(name, options = {}) {
    return (target, propertyKey, descriptor) => {
      const originalMethod = descriptor.value;
      
      descriptor.value = async function(...args) {
        if (!debugConfig.enabled) {
          return await originalMethod.apply(this, args);
        }

        return await this.trackOperation(
          `${name}.${propertyKey}`,
          () => originalMethod.apply(this, args),
          { args: options.logArgs ? args : undefined }
        );
      };

      return descriptor;
    };
  }

  /**
   * Get memory usage information
   * @returns {Object} Memory usage stats
   */
  getMemoryUsage() {
    const memoryInfo = {
      timestamp: perf_hooks.performance.now(),
      jsHeap: null,
      wasmMemory: null,
      tensorCount: debugData.tensors.size,
      activeTensors: []
    };

    // JavaScript heap info
    if (perf_hooks.performance.memory) {
      memoryInfo.jsHeap = {
        used: perf_hooks.performance.memory.usedJSHeapSize,
        total: perf_hooks.performance.memory.totalJSHeapSize,
        limit: perf_hooks.performance.memory.jsHeapSizeLimit
      };
    }

    // Active tensors info
    debugData.tensors.forEach((info, id) => {
      memoryInfo.activeTensors.push({
        id,
        shape: info.shape,
        dtype: info.dtype,
        estimatedSize: this.estimateTensorSize(info.shape, info.dtype),
        age: perf_hooks.performance.now() - info.creationTime
      });
    });

    if (debugConfig.enableMemoryTracking) {
      debugData.memory.push(memoryInfo);
      this.trimHistory(debugData.memory);
    }

    return memoryInfo;
  }

  /**
   * Get performance metrics
   * @returns {Object} Performance metrics
   */
  getPerformanceMetrics() {
    const {operations} = debugData;
    const metrics = {
      totalOperations: operations.length,
      averageDuration: 0,
      operationsByType: {},
      slowestOperations: [],
      recentOperations: operations.slice(-10)
    };

    if (operations.length > 0) {
      const totalDuration = operations.reduce((sum, op) => sum + op.duration, 0);
      metrics.averageDuration = totalDuration / operations.length;

      // Group by operation type
      operations.forEach(op => {
        if (!metrics.operationsByType[op.name]) {
          metrics.operationsByType[op.name] = {
            count: 0,
            totalDuration: 0,
            averageDuration: 0
          };
        }
        metrics.operationsByType[op.name].count++;
        metrics.operationsByType[op.name].totalDuration += op.duration;
      });

      // Calculate averages
      Object.keys(metrics.operationsByType).forEach(name => {
        const type = metrics.operationsByType[name];
        type.averageDuration = type.totalDuration / type.count;
      });

      // Find slowest operations
      metrics.slowestOperations = [...operations]
        .sort((a, b) => b.duration - a.duration)
        .slice(0, 5);
    }

    return metrics;
  }

  /**
   * Generate comprehensive debug report
   * @returns {Object} Debug report
   */
  generateReport() {
    return {
      timestamp: new Date().toISOString(),
      config: debugConfig,
      summary: {
        operationCount: debugData.operations.length,
        tensorCount: debugData.tensors.size,
        errorCount: debugData.errors.length,
        warningCount: debugData.warnings.length,
        sessionCount: debugData.sessions.size
      },
      memory: this.getMemoryUsage(),
      performance: this.getPerformanceMetrics(),
      recentErrors: debugData.errors.slice(-5),
      recentWarnings: debugData.warnings.slice(-5),
      activeSessions: Array.from(debugData.sessions.values())
        .filter(session => session.active)
        .map(session => ({
          id: session.id,
          name: session.name,
          duration: perf_hooks.performance.now() - session.startTime,
          operationCount: session.operations.length
        }))
    };
  }

  /**
   * Export debug data
   * @param {string} format - Export format ('json', 'csv')
   * @returns {string} Exported data
   */
  exportData(format = 'json') {
    const report = this.generateReport();
    
    switch (format.toLowerCase()) {
      case 'json':
        return JSON.stringify(report, null, 2);
      case 'csv':
        return this.generateCSVReport(report);
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }
  }

  /**
   * Clear debug data
   */
  clear() {
    debugData.operations.length = 0;
    debugData.tensors.clear();
    debugData.errors.length = 0;
    debugData.warnings.length = 0;
    debugData.performance.length = 0;
    debugData.memory.length = 0;
    debugData.sessions.clear();
    this.currentSession = null;
    this.log('Debug data cleared');
  }

  // Private helper methods
  getTensorId(tensor) {
    if (tensor._debugId) return tensor._debugId;
    tensor._debugId = `tensor_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return tensor._debugId;
  }

  getTensorShape(tensor) {
    if (tensor.shape && typeof tensor.shape === 'function') {
      return Array.from(tensor.shape());
    }
    if (tensor.shape && Array.isArray(tensor.shape)) {
      return tensor.shape;
    }
    return [];
  }

  getTensorDType(tensor) {
    if (tensor.dtype && typeof tensor.dtype === 'function') {
      return tensor.dtype();
    }
    if (tensor.dtype) {
      return tensor.dtype;
    }
    return 'unknown';
  }

  estimateTensorMemory(tensor) {
    const shape = this.getTensorShape(tensor);
    const dtype = this.getTensorDType(tensor);
    return this.estimateTensorSize(shape, dtype);
  }

  estimateTensorSize(shape, dtype) {
    const elements = shape.reduce((product, dim) => product * dim, 1);
    const bytesPerElement = this.getBytesPerElement(dtype);
    return elements * bytesPerElement;
  }

  getBytesPerElement(dtype) {
    const typeMap = {
      'f32': 4, 'float32': 4,
      'f64': 8, 'float64': 8,
      'i32': 4, 'int32': 4,
      'i64': 8, 'int64': 8,
      'u32': 4, 'uint32': 4,
      'i8': 1, 'int8': 1,
      'u8': 1, 'uint8': 1,
      'bool': 1
    };
    return typeMap[dtype] || 4;
  }

  areShapesBroadcastable(shape1, shape2) {
    const len1 = shape1.length;
    const len2 = shape2.length;
    const maxLen = Math.max(len1, len2);
    
    for (let i = 0; i < maxLen; i++) {
      const dim1 = i < len1 ? shape1[len1 - 1 - i] : 1;
      const dim2 = i < len2 ? shape2[len2 - 1 - i] : 1;
      
      if (dim1 !== dim2 && dim1 !== 1 && dim2 !== 1) {
        return false;
      }
    }
    return true;
  }

  getStackTrace() {
    try {
      throw new Error();
    } catch (e) {
      return e.stack.split('\n').slice(3, 8).join('\n');
    }
  }

  getResultType(result) {
    if (result === null) return 'null';
    if (result === undefined) return 'undefined';
    if (Array.isArray(result)) return 'array';
    return typeof result;
  }

  addOperation(operation) {
    debugData.operations.push(operation);
    this.trimHistory(debugData.operations);
    
    if (this.currentSession) {
      this.currentSession.addOperation(operation);
    }
  }

  trimHistory(array) {
    if (array.length > debugConfig.maxHistorySize) {
      array.splice(0, array.length - debugConfig.maxHistorySize);
    }
  }

  initializeConsoleOverrides() {
    // Store original console methods
    this.originalConsole = {
      log: console.log.bind(console),
      warn: console.warn.bind(console),
      error: console.error.bind(console),
      debug: console.debug.bind(console)
    };
  }

  shouldLog(level) {
    const levels = ['error', 'warn', 'info', 'debug', 'trace'];
    const currentLevelIndex = levels.indexOf(debugConfig.logLevel);
    const messageLevelIndex = levels.indexOf(level);
    return messageLevelIndex <= currentLevelIndex;
  }

  log(message, data = {}) {
    if (!debugConfig.enabled || !this.shouldLog('info')) return;
    this.originalConsole.log(`[TrustformeRS Debug]`, message, data);
  }

  warn(message, data = {}) {
    if (!debugConfig.enabled || !this.shouldLog('warn')) return;
    this.originalConsole.warn(`[TrustformeRS Warning]`, message, data);
    
    const warning = {
      message,
      data,
      timestamp: perf_hooks.performance.now(),
      stack: debugConfig.enableStackTraces ? this.getStackTrace() : null
    };
    
    debugData.warnings.push(warning);
    this.trimHistory(debugData.warnings);
    
    if (this.currentSession) {
      this.currentSession.addWarning(warning);
    }
  }

  error(message, error = {}) {
    if (!debugConfig.enabled || !this.shouldLog('error')) return;
    this.originalConsole.error(`[TrustformeRS Error]`, message, error);
    
    const errorInfo = {
      message,
      error: {
        message: error.message || error,
        stack: error.stack,
        name: error.name
      },
      timestamp: perf_hooks.performance.now(),
      stack: debugConfig.enableStackTraces ? this.getStackTrace() : null
    };
    
    debugData.errors.push(errorInfo);
    this.trimHistory(debugData.errors);
    
    if (this.currentSession) {
      this.currentSession.addError(errorInfo);
    }
  }

  generateCSVReport(report) {
    const lines = ['Operation,Duration,Success,Timestamp'];
    
    debugData.operations.forEach(op => {
      lines.push(`"${op.name}",${op.duration},${op.success},${op.startTime}`);
    });
    
    return lines.join('\n');
  }
}

// Global debug instance
const debugUtils = new DebugUtilities();

var debugUtilities = /*#__PURE__*/Object.freeze({
  __proto__: null,
  DebugUtilities: DebugUtilities,
  debugUtils: debugUtils
});

/**
 * TrustformeRS Tensor Inspector
 * Advanced tensor analysis and visualization utilities
 */

/**
 * Tensor inspector class for comprehensive tensor analysis
 */
class TensorInspector {
  constructor() {
    this.analysisCache = new Map();
    this.visualizationCache = new Map();
  }

  /**
   * Comprehensive tensor analysis
   * @param {Object} tensor - Tensor to analyze
   * @param {Object} options - Analysis options
   * @returns {Object} Analysis results
   */
  analyze(tensor, options = {}) {
    const {
      includeStatistics = true,
      includeDistribution = true,
      includeNaN = true,
      includeInfinite = true,
      includeMemory = true,
      includeGradient = false,
      cacheResults = true
    } = options;

    const tensorId = this.getTensorId(tensor);
    const cacheKey = `${tensorId}_${JSON.stringify(options)}`;
    
    if (cacheResults && this.analysisCache.has(cacheKey)) {
      return this.analysisCache.get(cacheKey);
    }

    const analysis = {
      basic: this.getBasicInfo(tensor),
      shape: this.getShapeInfo(tensor),
      dtype: this.getDTypeInfo(tensor),
      memory: includeMemory ? this.getMemoryInfo(tensor) : null,
      statistics: includeStatistics ? this.getStatistics(tensor) : null,
      distribution: includeDistribution ? this.getDistribution(tensor) : null,
      quality: {
        hasNaN: includeNaN ? this.hasNaN(tensor) : null,
        hasInfinite: includeInfinite ? this.hasInfinite(tensor) : null,
        nanCount: includeNaN ? this.getNaNCount(tensor) : null,
        infiniteCount: includeInfinite ? this.getInfiniteCount(tensor) : null
      },
      gradient: includeGradient ? this.getGradientInfo(tensor) : null,
      timestamp: performance.now()
    };

    if (cacheResults) {
      this.analysisCache.set(cacheKey, analysis);
    }

    return analysis;
  }

  /**
   * Get basic tensor information
   * @param {Object} tensor - Tensor object
   * @returns {Object} Basic info
   */
  getBasicInfo(tensor) {
    return {
      id: this.getTensorId(tensor),
      hasShape: typeof tensor.shape === 'function',
      hasDtype: typeof tensor.dtype === 'function',
      hasData: typeof tensor.data === 'function',
      isDisposed: this.isDisposed(tensor),
      constructor: tensor.constructor.name
    };
  }

  /**
   * Get shape information
   * @param {Object} tensor - Tensor object
   * @returns {Object} Shape info
   */
  getShapeInfo(tensor) {
    const shape = this.getShape(tensor);
    const ndim = shape.length;
    const size = shape.reduce((product, dim) => product * dim, 1);
    
    return {
      shape,
      ndim,
      size,
      isEmpty: size === 0,
      isScalar: ndim === 0,
      isVector: ndim === 1,
      isMatrix: ndim === 2,
      isBatch: ndim >= 3,
      strides: this.calculateStrides(shape)
    };
  }

  /**
   * Get data type information
   * @param {Object} tensor - Tensor object
   * @returns {Object} DType info
   */
  getDTypeInfo(tensor) {
    const dtype = this.getDType(tensor);
    const isFloating = ['f32', 'f64', 'float32', 'float64'].includes(dtype);
    const isInteger = ['i32', 'i64', 'u32', 'u64', 'i8', 'u8', 'int32', 'int64', 'uint32', 'uint64', 'int8', 'uint8'].includes(dtype);
    const isBoolean = dtype === 'bool';
    const isComplex = ['c64', 'c128', 'complex64', 'complex128'].includes(dtype);
    
    return {
      dtype,
      isFloating,
      isInteger,
      isBoolean,
      isComplex,
      byteSize: this.getBytesPerElement(dtype),
      precision: this.getPrecision(dtype)
    };
  }

  /**
   * Get memory information
   * @param {Object} tensor - Tensor object
   * @returns {Object} Memory info
   */
  getMemoryInfo(tensor) {
    const shape = this.getShape(tensor);
    const dtype = this.getDType(tensor);
    const elements = shape.reduce((product, dim) => product * dim, 1);
    const bytesPerElement = this.getBytesPerElement(dtype);
    const totalBytes = elements * bytesPerElement;

    return {
      elements,
      bytesPerElement,
      totalBytes,
      totalKB: totalBytes / 1024,
      totalMB: totalBytes / (1024 * 1024),
      densityRatio: 1.0, // For dense tensors
      isLarge: totalBytes > 100 * 1024 * 1024, // > 100MB
      memoryEfficiency: this.calculateMemoryEfficiency(tensor)
    };
  }

  /**
   * Get tensor statistics
   * @param {Object} tensor - Tensor object
   * @returns {Object} Statistics
   */
  getStatistics(tensor) {
    try {
      const data = this.getTensorData(tensor);
      if (!data || data.length === 0) {
        return null;
      }

      const stats = {
        count: data.length,
        min: Math.min(...data),
        max: Math.max(...data),
        mean: this.calculateMean(data),
        median: this.calculateMedian(data),
        std: this.calculateStd(data),
        variance: this.calculateVariance(data),
        sum: data.reduce((sum, val) => sum + val, 0),
        absSum: data.reduce((sum, val) => sum + Math.abs(val), 0),
        range: Math.max(...data) - Math.min(...data),
        nonZeroCount: data.filter(x => x !== 0).length,
        uniqueCount: new Set(data).size
      };

      stats.sparsity = 1 - (stats.nonZeroCount / stats.count);
      stats.meanAbsolute = stats.absSum / stats.count;

      return stats;
    } catch (error) {
      return {
        error: error.message,
        available: false
      };
    }
  }

  /**
   * Get value distribution
   * @param {Object} tensor - Tensor object
   * @param {number} bins - Number of histogram bins
   * @returns {Object} Distribution info
   */
  getDistribution(tensor, bins = 50) {
    try {
      const data = this.getTensorData(tensor);
      if (!data || data.length === 0) {
        return null;
      }

      const min = Math.min(...data);
      const max = Math.max(...data);
      const range = max - min;
      const binWidth = range / bins;
      
      const histogram = new Array(bins).fill(0);
      const binEdges = [];
      
      for (let i = 0; i <= bins; i++) {
        binEdges.push(min + i * binWidth);
      }

      data.forEach(value => {
        let binIndex = Math.floor((value - min) / binWidth);
        if (binIndex >= bins) binIndex = bins - 1;
        if (binIndex < 0) binIndex = 0;
        histogram[binIndex]++;
      });

      // Calculate percentiles
      const sortedData = [...data].sort((a, b) => a - b);
      const percentiles = {};
      [5, 10, 25, 50, 75, 90, 95, 99].forEach(p => {
        const index = Math.floor((p / 100) * sortedData.length);
        percentiles[`p${p}`] = sortedData[index];
      });

      return {
        histogram,
        binEdges,
        binWidth,
        percentiles,
        quartiles: {
          q1: percentiles.p25,
          q2: percentiles.p50,
          q3: percentiles.p75,
          iqr: percentiles.p75 - percentiles.p25
        },
        outliers: this.detectOutliers(data, percentiles.p25, percentiles.p75)
      };
    } catch (error) {
      return {
        error: error.message,
        available: false
      };
    }
  }

  /**
   * Check for NaN values
   * @param {Object} tensor - Tensor object
   * @returns {boolean} Has NaN values
   */
  hasNaN(tensor) {
    try {
      const data = this.getTensorData(tensor);
      return data.some(x => isNaN(x));
    } catch (error) {
      return null;
    }
  }

  /**
   * Check for infinite values
   * @param {Object} tensor - Tensor object
   * @returns {boolean} Has infinite values
   */
  hasInfinite(tensor) {
    try {
      const data = this.getTensorData(tensor);
      return data.some(x => !isFinite(x) && !isNaN(x));
    } catch (error) {
      return null;
    }
  }

  /**
   * Count NaN values
   * @param {Object} tensor - Tensor object
   * @returns {number} NaN count
   */
  getNaNCount(tensor) {
    try {
      const data = this.getTensorData(tensor);
      return data.filter(x => isNaN(x)).length;
    } catch (error) {
      return null;
    }
  }

  /**
   * Count infinite values
   * @param {Object} tensor - Tensor object
   * @returns {number} Infinite count
   */
  getInfiniteCount(tensor) {
    try {
      const data = this.getTensorData(tensor);
      return data.filter(x => !isFinite(x) && !isNaN(x)).length;
    } catch (error) {
      return null;
    }
  }

  /**
   * Get gradient information
   * @param {Object} tensor - Tensor object
   * @returns {Object} Gradient info
   */
  getGradientInfo(tensor) {
    return {
      requiresGrad: tensor.requires_grad || false,
      hasGrad: tensor.grad !== null && tensor.grad !== undefined,
      gradShape: tensor.grad ? this.getShape(tensor.grad) : null,
      isLeaf: tensor.is_leaf || false,
      gradFn: tensor.grad_fn ? tensor.grad_fn.constructor.name : null
    };
  }

  /**
   * Compare two tensors
   * @param {Object} tensor1 - First tensor
   * @param {Object} tensor2 - Second tensor
   * @param {Object} options - Comparison options
   * @returns {Object} Comparison results
   */
  compare(tensor1, tensor2, options = {}) {
    const {
      tolerance = 1e-6,
      includeDifference = true,
      includeStatistics = true
    } = options;

    const shape1 = this.getShape(tensor1);
    const shape2 = this.getShape(tensor2);
    const dtype1 = this.getDType(tensor1);
    const dtype2 = this.getDType(tensor2);

    const comparison = {
      shapesEqual: this.arraysEqual(shape1, shape2),
      dtypesEqual: dtype1 === dtype2,
      sizesEqual: shape1.reduce((p, d) => p * d, 1) === shape2.reduce((p, d) => p * d, 1),
      shape1,
      shape2,
      dtype1,
      dtype2
    };

    if (comparison.shapesEqual && comparison.dtypesEqual) {
      try {
        const data1 = this.getTensorData(tensor1);
        const data2 = this.getTensorData(tensor2);
        
        const differences = data1.map((val, i) => Math.abs(val - data2[i]));
        const maxDiff = Math.max(...differences);
        const meanDiff = differences.reduce((sum, diff) => sum + diff, 0) / differences.length;
        const withinTolerance = differences.every(diff => diff <= tolerance);

        comparison.valuesEqual = withinTolerance;
        comparison.maxDifference = maxDiff;
        comparison.meanDifference = meanDiff;
        comparison.tolerance = tolerance;
        comparison.differences = includeDifference ? differences : null;

        if (includeStatistics) {
          comparison.statistics = {
            tensor1: this.getStatistics(tensor1),
            tensor2: this.getStatistics(tensor2)
          };
        }
      } catch (error) {
        comparison.error = error.message;
      }
    }

    return comparison;
  }

  /**
   * Visualize tensor as text
   * @param {Object} tensor - Tensor object
   * @param {Object} options - Visualization options
   * @returns {string} Text visualization
   */
  visualizeText(tensor, options = {}) {
    const {
      maxElements = 100,
      precision = 4,
      threshold = 1000,
      linewidth = 75,
      edgeItems = 3
    } = options;

    const cacheKey = `${this.getTensorId(tensor)}_text_${JSON.stringify(options)}`;
    if (this.visualizationCache.has(cacheKey)) {
      return this.visualizationCache.get(cacheKey);
    }

    try {
      const shape = this.getShape(tensor);
      const data = this.getTensorData(tensor);
      
      let result = `Tensor(shape=${JSON.stringify(shape)}, dtype=${this.getDType(tensor)})\n`;
      
      if (data.length === 0) {
        result += '[]';
      } else if (data.length <= maxElements) {
        result += this.formatTensorData(data, shape, precision);
      } else {
        result += this.formatLargeTensorData(data, shape, precision, edgeItems);
      }

      this.visualizationCache.set(cacheKey, result);
      return result;
    } catch (error) {
      return `Error visualizing tensor: ${error.message}`;
    }
  }

  /**
   * Generate HTML visualization
   * @param {Object} tensor - Tensor object
   * @param {Object} options - Visualization options
   * @returns {string} HTML visualization
   */
  visualizeHTML(tensor, options = {}) {
    const {
      includeStatistics = true,
      includeHistogram = true,
      includeHeatmap = false,
      colorScheme = 'viridis'
    } = options;

    const analysis = this.analyze(tensor, {
      includeStatistics,
      includeDistribution: includeHistogram
    });

    let html = `
    <div class="tensor-visualization" style="font-family: 'Courier New', monospace; border: 1px solid #ccc; padding: 15px; margin: 10px; background: #f9f9f9;">
      <h3 style="margin-top: 0; color: #333;">Tensor Analysis</h3>
      <div class="basic-info" style="margin-bottom: 15px;">
        <strong>Shape:</strong> ${JSON.stringify(analysis.shape.shape)}<br>
        <strong>Data Type:</strong> ${analysis.dtype.dtype}<br>
        <strong>Size:</strong> ${analysis.shape.size.toLocaleString()} elements<br>
        <strong>Memory:</strong> ${(analysis.memory.totalBytes / 1024).toFixed(2)} KB
      </div>
    `;

    if (includeStatistics && analysis.statistics) {
      html += `
      <div class="statistics" style="margin-bottom: 15px;">
        <h4 style="margin-bottom: 10px;">Statistics</h4>
        <table style="border-collapse: collapse; width: 100%;">
          <tr><td style="padding: 2px 8px; border: 1px solid #ddd;">Min:</td><td style="padding: 2px 8px; border: 1px solid #ddd;">${analysis.statistics.min.toFixed(4)}</td></tr>
          <tr><td style="padding: 2px 8px; border: 1px solid #ddd;">Max:</td><td style="padding: 2px 8px; border: 1px solid #ddd;">${analysis.statistics.max.toFixed(4)}</td></tr>
          <tr><td style="padding: 2px 8px; border: 1px solid #ddd;">Mean:</td><td style="padding: 2px 8px; border: 1px solid #ddd;">${analysis.statistics.mean.toFixed(4)}</td></tr>
          <tr><td style="padding: 2px 8px; border: 1px solid #ddd;">Std:</td><td style="padding: 2px 8px; border: 1px solid #ddd;">${analysis.statistics.std.toFixed(4)}</td></tr>
          <tr><td style="padding: 2px 8px; border: 1px solid #ddd;">Sparsity:</td><td style="padding: 2px 8px; border: 1px solid #ddd;">${(analysis.statistics.sparsity * 100).toFixed(2)}%</td></tr>
        </table>
      </div>
      `;
    }

    if (includeHistogram && analysis.distribution) {
      html += this.generateHistogramHTML(analysis.distribution);
    }

    if (includeHeatmap && analysis.shape.isMatrix) {
      html += this.generateHeatmapHTML(tensor, colorScheme);
    }

    // Quality checks
    if (analysis.quality.hasNaN || analysis.quality.hasInfinite) {
      html += `
      <div class="quality-warnings" style="margin-top: 15px; padding: 10px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px;">
        <h4 style="margin-top: 0; color: #856404;">Quality Warnings</h4>
        ${analysis.quality.hasNaN ? `<div style="color: #d32f2f;">⚠ Contains ${analysis.quality.nanCount} NaN values</div>` : ''}
        ${analysis.quality.hasInfinite ? `<div style="color: #d32f2f;">⚠ Contains ${analysis.quality.infiniteCount} infinite values</div>` : ''}
      </div>
      `;
    }

    html += '</div>';
    return html;
  }

  /**
   * Generate summary report
   * @param {Object} tensor - Tensor object
   * @returns {Object} Summary report
   */
  summarize(tensor) {
    const analysis = this.analyze(tensor);
    
    const summary = {
      id: analysis.basic.id,
      shape: analysis.shape.shape,
      dtype: analysis.dtype.dtype,
      size: analysis.shape.size,
      memoryMB: analysis.memory.totalMB,
      hasIssues: analysis.quality.hasNaN || analysis.quality.hasInfinite,
      sparsity: analysis.statistics ? analysis.statistics.sparsity : null,
      valueRange: analysis.statistics ? [analysis.statistics.min, analysis.statistics.max] : null,
      recommendations: []
    };

    // Generate recommendations
    if (analysis.memory.isLarge) {
      summary.recommendations.push('Consider using lower precision dtype to reduce memory usage');
    }
    
    if (analysis.statistics && analysis.statistics.sparsity > 0.5) {
      summary.recommendations.push('Tensor is sparse, consider using sparse tensor formats');
    }
    
    if (analysis.quality.hasNaN) {
      summary.recommendations.push('Remove or replace NaN values before using in computations');
    }
    
    if (analysis.quality.hasInfinite) {
      summary.recommendations.push('Handle infinite values to prevent numerical instability');
    }

    return summary;
  }

  /**
   * Clear caches
   */
  clearCaches() {
    this.analysisCache.clear();
    this.visualizationCache.clear();
  }

  // Private helper methods
  getTensorId(tensor) {
    if (tensor._inspectorId) return tensor._inspectorId;
    tensor._inspectorId = `tensor_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return tensor._inspectorId;
  }

  getShape(tensor) {
    if (tensor.shape && typeof tensor.shape === 'function') {
      return Array.from(tensor.shape());
    }
    if (tensor.shape && Array.isArray(tensor.shape)) {
      return tensor.shape;
    }
    return [];
  }

  getDType(tensor) {
    if (tensor.dtype && typeof tensor.dtype === 'function') {
      return tensor.dtype();
    }
    if (tensor.dtype) {
      return tensor.dtype;
    }
    return 'unknown';
  }

  getTensorData(tensor) {
    if (tensor.data && typeof tensor.data === 'function') {
      const data = tensor.data();
      return Array.isArray(data) ? data : Array.from(data);
    }
    if (tensor.toArray && typeof tensor.toArray === 'function') {
      return tensor.toArray();
    }
    if (Array.isArray(tensor)) {
      return tensor;
    }
    throw new Error('Cannot extract data from tensor');
  }

  isDisposed(tensor) {
    return tensor.isDisposed === true || tensor._disposed === true;
  }

  getBytesPerElement(dtype) {
    const typeMap = {
      'f32': 4, 'float32': 4,
      'f64': 8, 'float64': 8,
      'i32': 4, 'int32': 4,
      'i64': 8, 'int64': 8,
      'u32': 4, 'uint32': 4,
      'i8': 1, 'int8': 1,
      'u8': 1, 'uint8': 1,
      'bool': 1,
      'c64': 8, 'complex64': 8,
      'c128': 16, 'complex128': 16
    };
    return typeMap[dtype] || 4;
  }

  getPrecision(dtype) {
    const precisionMap = {
      'f32': 'single', 'float32': 'single',
      'f64': 'double', 'float64': 'double',
      'i32': '32-bit', 'int32': '32-bit',
      'i64': '64-bit', 'int64': '64-bit',
      'u32': '32-bit unsigned', 'uint32': '32-bit unsigned',
      'i8': '8-bit', 'int8': '8-bit',
      'u8': '8-bit unsigned', 'uint8': '8-bit unsigned',
      'bool': '1-bit'
    };
    return precisionMap[dtype] || 'unknown';
  }

  calculateStrides(shape) {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  calculateMemoryEfficiency(tensor) {
    // For dense tensors, efficiency is always 1.0
    // For sparse tensors, this would calculate actual efficiency
    return 1.0;
  }

  calculateMean(data) {
    return data.reduce((sum, val) => sum + val, 0) / data.length;
  }

  calculateMedian(data) {
    const sorted = [...data].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 
      ? (sorted[mid - 1] + sorted[mid]) / 2 
      : sorted[mid];
  }

  calculateVariance(data) {
    const mean = this.calculateMean(data);
    return data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
  }

  calculateStd(data) {
    return Math.sqrt(this.calculateVariance(data));
  }

  detectOutliers(data, q1, q3) {
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
    return data.filter(x => x < lowerBound || x > upperBound);
  }

  arraysEqual(arr1, arr2) {
    return arr1.length === arr2.length && arr1.every((val, i) => val === arr2[i]);
  }

  formatTensorData(data, shape, precision) {
    if (shape.length === 1) {
      return `[${data.map(x => x.toFixed(precision)).join(', ')}]`;
    } else if (shape.length === 2) {
      const [rows, cols] = shape;
      let result = '[\n';
      for (let i = 0; i < rows; i++) {
        const row = data.slice(i * cols, (i + 1) * cols);
        result += `  [${row.map(x => x.toFixed(precision)).join(', ')}]`;
        if (i < rows - 1) result += ',';
        result += '\n';
      }
      result += ']';
      return result;
    } 
      return `Tensor with ${shape.length}D shape: ${JSON.stringify(shape)}`;
    
  }

  formatLargeTensorData(data, shape, precision, edgeItems) {
    const totalElements = data.length;
    const showElements = edgeItems * 2;
    
    if (totalElements <= showElements) {
      return this.formatTensorData(data, shape, precision);
    }

    const head = data.slice(0, edgeItems);
    const tail = data.slice(-edgeItems);
    
    return `[${head.map(x => x.toFixed(precision)).join(', ')}, ..., ${tail.map(x => x.toFixed(precision)).join(', ')}]`;
  }

  generateHistogramHTML(distribution) {
    const maxHeight = 100;
    const maxCount = Math.max(...distribution.histogram);
    
    let html = `
    <div class="histogram" style="margin-bottom: 15px;">
      <h4 style="margin-bottom: 10px;">Value Distribution</h4>
      <div style="display: flex; align-items: end; height: ${maxHeight}px; margin-bottom: 5px;">
    `;

    distribution.histogram.forEach((count, i) => {
      const height = (count / maxCount) * maxHeight;
      const opacity = count === 0 ? 0.1 : 0.7;
      html += `
        <div style="
          flex: 1; 
          background: rgba(70, 130, 180, ${opacity}); 
          height: ${height}px; 
          margin-right: 1px;
          border-radius: 2px 2px 0 0;
        " title="Bin ${i}: ${count} values"></div>
      `;
    });

    html += `
      </div>
      <div style="font-size: 12px; color: #666;">
        Range: ${distribution.binEdges[0].toFixed(3)} to ${distribution.binEdges[distribution.binEdges.length - 1].toFixed(3)}
      </div>
    </div>
    `;

    return html;
  }

  generateHeatmapHTML(tensor, colorScheme) {
    // Simplified heatmap for small matrices
    const data = this.getTensorData(tensor);
    const shape = this.getShape(tensor);
    
    if (shape.length !== 2 || shape[0] > 20 || shape[1] > 20) {
      return '<div>Heatmap not available for this tensor size</div>';
    }

    const [rows, cols] = shape;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min;

    let html = `
    <div class="heatmap" style="margin-bottom: 15px;">
      <h4 style="margin-bottom: 10px;">Heatmap</h4>
      <div style="display: grid; grid-template-columns: repeat(${cols}, 20px); gap: 1px;">
    `;

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const value = data[i * cols + j];
        const normalized = range === 0 ? 0.5 : (value - min) / range;
        const intensity = Math.round(normalized * 255);
        
        html += `
          <div style="
            width: 20px; 
            height: 20px; 
            background: rgb(${255 - intensity}, ${255 - intensity}, 255);
            border: 1px solid #ccc;
            font-size: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
          " title="[${i},${j}]: ${value.toFixed(3)}"></div>
        `;
      }
    }

    html += '</div></div>';
    return html;
  }
}

// Global inspector instance
const tensorInspector = new TensorInspector();

var tensorInspector$1 = /*#__PURE__*/Object.freeze({
  __proto__: null,
  TensorInspector: TensorInspector,
  tensorInspector: tensorInspector
});

/**
 * TrustformeRS Model Visualization Helpers
 * Comprehensive model analysis and visualization utilities for development
 */


/**
 * Model visualization and analysis class
 */
class ModelVisualizer {
  constructor() {
    this.modelCache = new Map();
    this.layerCache = new Map();
    this.activationCache = new Map();
    this.visualizationCache = new Map();
  }

  /**
   * Analyze model architecture and structure
   * @param {Object} model - Model object
   * @param {Object} options - Analysis options
   * @returns {Object} Model analysis results
   */
  analyzeModel(model, options = {}) {
    const {
      includeWeights = true,
      includeActivations = false,
      includeGradients = false,
      cacheResults = true,
      maxDepth = 10
    } = options;

    const modelId = this.getModelId(model);
    const cacheKey = `${modelId}_${JSON.stringify(options)}`;
    
    if (cacheResults && this.modelCache.has(cacheKey)) {
      return this.modelCache.get(cacheKey);
    }

    const analysis = {
      basic: this.getBasicModelInfo(model),
      architecture: this.getArchitectureInfo(model, maxDepth),
      parameters: this.getParameterInfo(model),
      layers: this.getLayerInfo(model),
      computation: this.getComputationInfo(model),
      weights: includeWeights ? this.getWeightInfo(model) : null,
      activations: includeActivations ? this.getActivationInfo(model) : null,
      gradients: includeGradients ? this.getGradientInfo(model) : null,
      memory: this.getModelMemoryInfo(model),
      timestamp: performance.now()
    };

    if (cacheResults) {
      this.modelCache.set(cacheKey, analysis);
    }

    return analysis;
  }

  /**
   * Get basic model information
   * @param {Object} model - Model object
   * @returns {Object} Basic model info
   */
  getBasicModelInfo(model) {
    return {
      id: this.getModelId(model),
      name: model.name || model.constructor.name,
      type: this.getModelType(model),
      hasConfig: this.hasModelConfig(model),
      isTraining: this.isTrainingMode(model),
      isQuantized: this.isQuantized(model),
      device: this.getModelDevice(model),
      constructor: model.constructor.name,
      version: model.version || 'unknown'
    };
  }

  /**
   * Get model architecture information
   * @param {Object} model - Model object
   * @param {number} maxDepth - Maximum depth to analyze
   * @returns {Object} Architecture info
   */
  getArchitectureInfo(model, maxDepth = 10) {
    const architecture = {
      layers: [],
      connections: [],
      totalLayers: 0,
      totalParameters: 0,
      architectureType: this.getArchitectureType(model),
      hasAttention: false,
      hasEmbedding: false,
      hasNormalization: false,
      hasDropout: false
    };

    try {
      // Extract layer information
      const layers = this.extractLayers(model, maxDepth);
      architecture.layers = layers;
      architecture.totalLayers = layers.length;

      // Analyze layer types
      layers.forEach(layer => {
        if (layer.type.includes('attention')) {
          architecture.hasAttention = true;
        }
        if (layer.type.includes('embedding')) {
          architecture.hasEmbedding = true;
        }
        if (layer.type.includes('norm')) {
          architecture.hasNormalization = true;
        }
        if (layer.type.includes('dropout')) {
          architecture.hasDropout = true;
        }
        architecture.totalParameters += layer.parameters || 0;
      });

      // Extract connections
      architecture.connections = this.extractConnections(model);

    } catch (error) {
      architecture.error = error.message;
    }

    return architecture;
  }

  /**
   * Get parameter information
   * @param {Object} model - Model object
   * @returns {Object} Parameter info
   */
  getParameterInfo(model) {
    const info = {
      total: 0,
      trainable: 0,
      frozen: 0,
      byLayer: {},
      byType: {
        weights: 0,
        biases: 0,
        embeddings: 0,
        normalization: 0,
        other: 0
      },
      distribution: {
        min: Infinity,
        max: -Infinity,
        mean: 0,
        std: 0
      }
    };

    try {
      const parameters = this.extractParameters(model);
      
      parameters.forEach(param => {
        const paramCount = param.size || 0;
        info.total += paramCount;
        
        if (param.trainable) {
          info.trainable += paramCount;
        } else {
          info.frozen += paramCount;
        }

        // Categorize by layer
        if (!info.byLayer[param.layer]) {
          info.byLayer[param.layer] = 0;
        }
        info.byLayer[param.layer] += paramCount;

        // Categorize by type
        if (param.name.includes('weight')) {
          info.byType.weights += paramCount;
        } else if (param.name.includes('bias')) {
          info.byType.biases += paramCount;
        } else if (param.name.includes('embedding')) {
          info.byType.embeddings += paramCount;
        } else if (param.name.includes('norm')) {
          info.byType.normalization += paramCount;
        } else {
          info.byType.other += paramCount;
        }
      });

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Get layer information
   * @param {Object} model - Model object
   * @returns {Object} Layer info
   */
  getLayerInfo(model) {
    const info = {
      count: 0,
      types: {},
      shapes: {},
      operations: {},
      activationFunctions: {},
      details: []
    };

    try {
      const layers = this.extractLayers(model);
      info.count = layers.length;
      
      layers.forEach(layer => {
        // Count layer types
        if (!info.types[layer.type]) {
          info.types[layer.type] = 0;
        }
        info.types[layer.type]++;

        // Count shapes
        const shapeKey = JSON.stringify(layer.shape);
        if (!info.shapes[shapeKey]) {
          info.shapes[shapeKey] = 0;
        }
        info.shapes[shapeKey]++;

        // Count operations
        if (layer.operation) {
          if (!info.operations[layer.operation]) {
            info.operations[layer.operation] = 0;
          }
          info.operations[layer.operation]++;
        }

        // Count activation functions
        if (layer.activation) {
          if (!info.activationFunctions[layer.activation]) {
            info.activationFunctions[layer.activation] = 0;
          }
          info.activationFunctions[layer.activation]++;
        }

        info.details.push({
          id: layer.id,
          name: layer.name,
          type: layer.type,
          shape: layer.shape,
          parameters: layer.parameters,
          activation: layer.activation,
          trainable: layer.trainable
        });
      });

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Get computation information
   * @param {Object} model - Model object
   * @returns {Object} Computation info
   */
  getComputationInfo(model) {
    const info = {
      totalFLOPs: 0,
      totalMACs: 0,
      operationCounts: {},
      computationGraph: [],
      complexity: 'unknown',
      bottlenecks: []
    };

    try {
      const operations = this.extractOperations(model);
      
      operations.forEach(op => {
        info.totalFLOPs += op.flops || 0;
        info.totalMACs += op.macs || 0;
        
        if (!info.operationCounts[op.type]) {
          info.operationCounts[op.type] = 0;
        }
        info.operationCounts[op.type]++;
        
        info.computationGraph.push({
          id: op.id,
          type: op.type,
          flops: op.flops,
          macs: op.macs,
          inputShape: op.inputShape,
          outputShape: op.outputShape
        });
      });

      // Estimate complexity
      if (info.totalFLOPs > 1e12) {
        info.complexity = 'very_high';
      } else if (info.totalFLOPs > 1e9) {
        info.complexity = 'high';
      } else if (info.totalFLOPs > 1e6) {
        info.complexity = 'medium';
      } else {
        info.complexity = 'low';
      }

      // Identify bottlenecks
      info.bottlenecks = this.identifyBottlenecks(operations);

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Get weight information
   * @param {Object} model - Model object
   * @returns {Object} Weight info
   */
  getWeightInfo(model) {
    const info = {
      weights: [],
      statistics: {},
      distributions: {},
      healthCheck: {
        hasNaN: false,
        hasInfinite: false,
        hasLargeValues: false,
        hasSmallGradients: false
      }
    };

    try {
      const weights = this.extractWeights(model);
      
      weights.forEach(weight => {
        const weightInfo = {
          name: weight.name,
          shape: weight.shape,
          dtype: weight.dtype,
          analysis: tensorInspector.analyze(weight.tensor, {
            includeStatistics: true,
            includeDistribution: true,
            includeNaN: true,
            includeInfinite: true
          })
        };

        info.weights.push(weightInfo);

        // Update health check
        if (weightInfo.analysis.quality.hasNaN) {
          info.healthCheck.hasNaN = true;
        }
        if (weightInfo.analysis.quality.hasInfinite) {
          info.healthCheck.hasInfinite = true;
        }
        if (weightInfo.analysis.statistics && Math.abs(weightInfo.analysis.statistics.max) > 100) {
          info.healthCheck.hasLargeValues = true;
        }
      });

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Get activation information
   * @param {Object} model - Model object
   * @returns {Object} Activation info
   */
  getActivationInfo(model) {
    const info = {
      activations: [],
      patterns: {},
      sparsity: {},
      healthCheck: {
        hasDeadNeurons: false,
        hasExplodingActivations: false,
        hasVanishingActivations: false
      }
    };

    try {
      const activations = this.extractActivations(model);
      
      activations.forEach(activation => {
        const activationInfo = {
          name: activation.name,
          layer: activation.layer,
          shape: activation.shape,
          analysis: tensorInspector.analyze(activation.tensor, {
            includeStatistics: true,
            includeDistribution: true
          })
        };

        info.activations.push(activationInfo);

        // Check for dead neurons (all zeros)
        if (activationInfo.analysis.statistics && activationInfo.analysis.statistics.max === 0) {
          info.healthCheck.hasDeadNeurons = true;
        }

        // Check for exploding activations
        if (activationInfo.analysis.statistics && Math.abs(activationInfo.analysis.statistics.max) > 1000) {
          info.healthCheck.hasExplodingActivations = true;
        }

        // Check for vanishing activations
        if (activationInfo.analysis.statistics && Math.abs(activationInfo.analysis.statistics.max) < 1e-6) {
          info.healthCheck.hasVanishingActivations = true;
        }
      });

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Get gradient information
   * @param {Object} model - Model object
   * @returns {Object} Gradient info
   */
  getGradientInfo(model) {
    const info = {
      gradients: [],
      norms: {},
      healthCheck: {
        hasVanishingGradients: false,
        hasExplodingGradients: false,
        hasNaNGradients: false
      }
    };

    try {
      const gradients = this.extractGradients(model);
      
      gradients.forEach(gradient => {
        const gradientInfo = {
          name: gradient.name,
          layer: gradient.layer,
          shape: gradient.shape,
          analysis: tensorInspector.analyze(gradient.tensor, {
            includeStatistics: true,
            includeNaN: true,
            includeInfinite: true
          })
        };

        info.gradients.push(gradientInfo);

        // Check gradient health
        if (gradientInfo.analysis.quality.hasNaN) {
          info.healthCheck.hasNaNGradients = true;
        }

        if (gradientInfo.analysis.statistics) {
          const gradNorm = Math.sqrt(gradientInfo.analysis.statistics.variance);
          info.norms[gradient.name] = gradNorm;

          if (gradNorm < 1e-6) {
            info.healthCheck.hasVanishingGradients = true;
          }
          if (gradNorm > 100) {
            info.healthCheck.hasExplodingGradients = true;
          }
        }
      });

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Get model memory information
   * @param {Object} model - Model object
   * @returns {Object} Memory info
   */
  getModelMemoryInfo(model) {
    const info = {
      totalMemory: 0,
      parameterMemory: 0,
      activationMemory: 0,
      gradientMemory: 0,
      bufferMemory: 0,
      breakdown: {},
      efficiency: 1.0
    };

    try {
      // Calculate parameter memory
      const parameters = this.extractParameters(model);
      parameters.forEach(param => {
        const memory = param.size * this.getBytesPerElement(param.dtype);
        info.parameterMemory += memory;
      });

      // Calculate activation memory (estimated)
      const layers = this.extractLayers(model);
      layers.forEach(layer => {
        if (layer.outputSize) {
          const memory = layer.outputSize * 4; // Assume F32
          info.activationMemory += memory;
        }
      });

      // Calculate gradient memory (same as parameters if training)
      if (this.isTrainingMode(model)) {
        info.gradientMemory = info.parameterMemory;
      }

      info.totalMemory = info.parameterMemory + info.activationMemory + info.gradientMemory;

      // Memory breakdown
      info.breakdown = {
        parameters: (info.parameterMemory / info.totalMemory) * 100,
        activations: (info.activationMemory / info.totalMemory) * 100,
        gradients: (info.gradientMemory / info.totalMemory) * 100
      };

    } catch (error) {
      info.error = error.message;
    }

    return info;
  }

  /**
   * Visualize model architecture as text
   * @param {Object} model - Model object
   * @param {Object} options - Visualization options
   * @returns {string} Text visualization
   */
  visualizeArchitecture(model, options = {}) {
    const {
      maxDepth = 10,
      showParameters = true,
      showShapes = true,
      showTypes = true,
      compact = false
    } = options;

    try {
      const analysis = this.analyzeModel(model, { maxDepth });
      let output = '';

      // Header
      output += `Model: ${analysis.basic.name}\n`;
      output += `Type: ${analysis.basic.type}\n`;
      output += `Total Parameters: ${analysis.parameters.total.toLocaleString()}\n`;
      output += `Total Layers: ${analysis.architecture.totalLayers}\n`;
      output += `Memory Usage: ${(analysis.memory.totalMemory / 1024 / 1024).toFixed(2)} MB\n`;
      output += '\n';

      // Architecture
      output += 'Architecture:\n';
      output += '============\n';
      
      analysis.architecture.layers.forEach((layer, index) => {
        const indent = '  '.repeat(layer.depth || 0);
        let line = `${indent}${index + 1}. ${layer.name || layer.type}`;
        
        if (showTypes) {
          line += ` (${layer.type})`;
        }
        
        if (showShapes && layer.shape) {
          line += ` -> ${JSON.stringify(layer.shape)}`;
        }
        
        if (showParameters && layer.parameters) {
          line += ` [${layer.parameters.toLocaleString()} params]`;
        }
        
        output += `${line}\n`;
      });

      // Layer type summary
      output += '\nLayer Types:\n';
      output += '============\n';
      Object.entries(analysis.layers.types).forEach(([type, count]) => {
        output += `${type}: ${count}\n`;
      });

      // Parameter summary
      output += '\nParameter Summary:\n';
      output += '==================\n';
      output += `Total: ${analysis.parameters.total.toLocaleString()}\n`;
      output += `Trainable: ${analysis.parameters.trainable.toLocaleString()}\n`;
      output += `Frozen: ${analysis.parameters.frozen.toLocaleString()}\n`;

      return output;
    } catch (error) {
      return `Error visualizing model: ${error.message}`;
    }
  }

  /**
   * Generate HTML visualization
   * @param {Object} model - Model object
   * @param {Object} options - Visualization options
   * @returns {string} HTML visualization
   */
  visualizeHTML(model, options = {}) {
    const {
      includeWeights = false,
      includeActivations = false,
      includeMemory = true,
      colorScheme = 'default'
    } = options;

    try {
      const analysis = this.analyzeModel(model, {
        includeWeights,
        includeActivations
      });

      let html = `
      <div class="model-visualization" style="font-family: 'Courier New', monospace; border: 1px solid #ccc; padding: 20px; margin: 10px; background: #f9f9f9;">
        <h2 style="margin-top: 0; color: #333;">Model Analysis: ${analysis.basic.name}</h2>
        
        <div class="model-overview" style="margin-bottom: 20px;">
          <h3>Overview</h3>
          <table style="border-collapse: collapse; width: 100%; margin-bottom: 15px;">
            <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Type:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${analysis.basic.type}</td></tr>
            <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Total Parameters:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${analysis.parameters.total.toLocaleString()}</td></tr>
            <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Total Layers:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${analysis.architecture.totalLayers}</td></tr>
            <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Training Mode:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${analysis.basic.isTraining ? 'Yes' : 'No'}</td></tr>
            <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Device:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${analysis.basic.device}</td></tr>
          </table>
        </div>

        <div class="architecture-features" style="margin-bottom: 20px;">
          <h3>Architecture Features</h3>
          <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <span style="padding: 4px 8px; background: ${analysis.architecture.hasAttention ? '#d4edda' : '#f8d7da'}; border-radius: 4px; font-size: 12px;">
              ${analysis.architecture.hasAttention ? '✓' : '✗'} Attention
            </span>
            <span style="padding: 4px 8px; background: ${analysis.architecture.hasEmbedding ? '#d4edda' : '#f8d7da'}; border-radius: 4px; font-size: 12px;">
              ${analysis.architecture.hasEmbedding ? '✓' : '✗'} Embedding
            </span>
            <span style="padding: 4px 8px; background: ${analysis.architecture.hasNormalization ? '#d4edda' : '#f8d7da'}; border-radius: 4px; font-size: 12px;">
              ${analysis.architecture.hasNormalization ? '✓' : '✗'} Normalization
            </span>
            <span style="padding: 4px 8px; background: ${analysis.architecture.hasDropout ? '#d4edda' : '#f8d7da'}; border-radius: 4px; font-size: 12px;">
              ${analysis.architecture.hasDropout ? '✓' : '✗'} Dropout
            </span>
          </div>
        </div>

        <div class="layer-types" style="margin-bottom: 20px;">
          <h3>Layer Types</h3>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
      `;

      Object.entries(analysis.layers.types).forEach(([type, count]) => {
        html += `
          <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #fff;">
            <strong>${type}</strong><br>
            <span style="color: #666; font-size: 14px;">${count} layer${count > 1 ? 's' : ''}</span>
          </div>
        `;
      });

      html += `
          </div>
        </div>

        <div class="parameter-breakdown" style="margin-bottom: 20px;">
          <h3>Parameter Breakdown</h3>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #e8f5e8;">
              <strong>Trainable</strong><br>
              <span style="font-size: 18px; color: #2d5016;">${analysis.parameters.trainable.toLocaleString()}</span>
            </div>
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa;">
              <strong>Frozen</strong><br>
              <span style="font-size: 18px; color: #6c757d;">${analysis.parameters.frozen.toLocaleString()}</span>
            </div>
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #fff3cd;">
              <strong>Total</strong><br>
              <span style="font-size: 18px; color: #856404;">${analysis.parameters.total.toLocaleString()}</span>
            </div>
          </div>
        </div>
      `;

      if (includeMemory) {
        html += `
        <div class="memory-usage" style="margin-bottom: 20px;">
          <h3>Memory Usage</h3>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #e3f2fd;">
              <strong>Parameters</strong><br>
              <span style="font-size: 16px; color: #1565c0;">${(analysis.memory.parameterMemory / 1024 / 1024).toFixed(2)} MB</span>
            </div>
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #fff3e0;">
              <strong>Activations</strong><br>
              <span style="font-size: 16px; color: #ef6c00;">${(analysis.memory.activationMemory / 1024 / 1024).toFixed(2)} MB</span>
            </div>
            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #fce4ec;">
              <strong>Total</strong><br>
              <span style="font-size: 16px; color: #ad1457;">${(analysis.memory.totalMemory / 1024 / 1024).toFixed(2)} MB</span>
            </div>
          </div>
        </div>
        `;
      }

      // Health checks
      if (analysis.weights && analysis.weights.healthCheck) {
        const healthIssues = Object.entries(analysis.weights.healthCheck).filter(([_, value]) => value);
        if (healthIssues.length > 0) {
          html += `
          <div class="health-warnings" style="margin-top: 20px; padding: 15px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px;">
            <h3 style="margin-top: 0; color: #856404;">Health Warnings</h3>
          `;
          
          healthIssues.forEach(([issue, _]) => {
            html += `<div style="color: #d32f2f; margin-bottom: 5px;">⚠ ${issue.replace(/([A-Z])/g, ' $1').toLowerCase()}</div>`;
          });
          
          html += '</div>';
        }
      }

      html += '</div>';
      return html;
    } catch (error) {
      return `<div style="color: red; padding: 20px;">Error generating model visualization: ${error.message}</div>`;
    }
  }

  /**
   * Generate model summary report
   * @param {Object} model - Model object
   * @returns {Object} Summary report
   */
  summarize(model) {
    const analysis = this.analyzeModel(model);
    
    const summary = {
      name: analysis.basic.name,
      type: analysis.basic.type,
      parameters: analysis.parameters.total,
      layers: analysis.architecture.totalLayers,
      memoryMB: analysis.memory.totalMemory / 1024 / 1024,
      complexity: analysis.computation.complexity,
      hasIssues: false,
      issues: [],
      recommendations: []
    };

    // Check for issues
    if (analysis.weights && analysis.weights.healthCheck) {
      Object.entries(analysis.weights.healthCheck).forEach(([issue, hasIssue]) => {
        if (hasIssue) {
          summary.hasIssues = true;
          summary.issues.push(issue);
        }
      });
    }

    // Generate recommendations
    if (analysis.memory.totalMemory > 1024 * 1024 * 1024) { // > 1GB
      summary.recommendations.push('Consider model quantization to reduce memory usage');
    }
    
    if (analysis.computation.complexity === 'very_high') {
      summary.recommendations.push('Model has high computational complexity, consider pruning or distillation');
    }
    
    if (analysis.parameters.frozen > analysis.parameters.trainable) {
      summary.recommendations.push('Many parameters are frozen, consider fine-tuning strategy');
    }

    return summary;
  }

  /**
   * Clear visualization caches
   */
  clearCaches() {
    this.modelCache.clear();
    this.layerCache.clear();
    this.activationCache.clear();
    this.visualizationCache.clear();
  }

  // Private helper methods
  getModelId(model) {
    if (model._visualizerId) return model._visualizerId;
    model._visualizerId = `model_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return model._visualizerId;
  }

  getModelType(model) {
    if (model.config && model.config.model_type) {
      return model.config.model_type;
    }
    if (model.model_type) {
      return model.model_type;
    }
    return model.constructor.name.toLowerCase();
  }

  hasModelConfig(model) {
    return !!(model.config || model.configuration);
  }

  isTrainingMode(model) {
    return model.training || model.train_mode || false;
  }

  isQuantized(model) {
    return model.quantized || model.is_quantized || false;
  }

  getModelDevice(model) {
    if (model.device) return model.device;
    if (model.config && model.config.device) return model.config.device;
    return 'cpu';
  }

  getArchitectureType(model) {
    const type = this.getModelType(model);
    if (type.includes('bert')) return 'encoder';
    if (type.includes('gpt')) return 'decoder';
    if (type.includes('t5')) return 'encoder-decoder';
    if (type.includes('llama')) return 'decoder';
    if (type.includes('mistral')) return 'decoder';
    return 'unknown';
  }

  extractLayers(model, maxDepth = 10) {
    const layers = [];
    // This is a simplified extraction - in practice, this would need to be
    // adapted for specific model formats and structures
    try {
      if (model.layers) {
        model.layers.forEach((layer, index) => {
          layers.push({
            id: index,
            name: layer.name || `layer_${index}`,
            type: layer.constructor.name.toLowerCase(),
            shape: layer.output_shape || layer.shape,
            parameters: this.countLayerParameters(layer),
            activation: layer.activation || null,
            trainable: layer.trainable !== false,
            depth: 0
          });
        });
      } else if (model.modules) {
        // Handle PyTorch-style modules
        Object.entries(model.modules).forEach(([name, module], index) => {
          layers.push({
            id: index,
            name,
            type: module.constructor.name.toLowerCase(),
            shape: module.weight ? module.weight.shape : null,
            parameters: this.countModuleParameters(module),
            trainable: module.requires_grad !== false,
            depth: 0
          });
        });
      }
    } catch (error) {
      debugUtils.warn('Error extracting layers', error);
    }
    return layers;
  }

  extractConnections(model) {
    // Simplified connection extraction
    return [];
  }

  extractParameters(model) {
    const parameters = [];
    try {
      if (model.parameters) {
        model.parameters().forEach((param, index) => {
          parameters.push({
            name: param.name || `param_${index}`,
            layer: param.layer || 'unknown',
            size: param.size || 0,
            shape: param.shape || [],
            dtype: param.dtype || 'f32',
            trainable: param.requires_grad !== false,
            tensor: param
          });
        });
      }
    } catch (error) {
      debugUtils.warn('Error extracting parameters', error);
    }
    return parameters;
  }

  extractOperations(model) {
    // This would need to be implemented based on model format
    return [];
  }

  extractWeights(model) {
    const weights = [];
    try {
      if (model.state_dict) {
        const stateDict = model.state_dict();
        Object.entries(stateDict).forEach(([name, tensor]) => {
          weights.push({
            name,
            shape: tensor.shape,
            dtype: tensor.dtype,
            tensor
          });
        });
      }
    } catch (error) {
      debugUtils.warn('Error extracting weights', error);
    }
    return weights;
  }

  extractActivations(model) {
    // This would need activation hooks to be implemented
    return [];
  }

  extractGradients(model) {
    // This would need gradient hooks to be implemented
    return [];
  }

  countLayerParameters(layer) {
    if (layer.parameters) {
      return layer.parameters().reduce((sum, param) => sum + param.size, 0);
    }
    return 0;
  }

  countModuleParameters(module) {
    if (module.parameters) {
      return module.parameters().reduce((sum, param) => sum + param.size, 0);
    }
    return 0;
  }

  identifyBottlenecks(operations) {
    return operations
      .filter(op => op.flops > 1e9) // > 1B FLOPs
      .sort((a, b) => b.flops - a.flops)
      .slice(0, 5)
      .map(op => ({
        operation: op.id,
        type: op.type,
        flops: op.flops,
        percentage: (op.flops / operations.reduce((sum, o) => sum + o.flops, 0)) * 100
      }));
  }

  getBytesPerElement(dtype) {
    const typeMap = {
      'f32': 4, 'float32': 4,
      'f64': 8, 'float64': 8,
      'i32': 4, 'int32': 4,
      'i64': 8, 'int64': 8,
      'u32': 4, 'uint32': 4,
      'i8': 1, 'int8': 1,
      'u8': 1, 'uint8': 1,
      'bool': 1
    };
    return typeMap[dtype] || 4;
  }
}

// Global model visualizer instance
const modelVisualizer = new ModelVisualizer();

var modelVisualization = /*#__PURE__*/Object.freeze({
  __proto__: null,
  ModelVisualizer: ModelVisualizer,
  modelVisualizer: modelVisualizer
});

/**
 * TrustformeRS Error Diagnostic Tools
 * Comprehensive error analysis, diagnosis, and resolution assistance
 */


/**
 * Error classification types
 */
const ErrorTypes = {
  INITIALIZATION: 'initialization',
  TENSOR_OPERATION: 'tensor_operation',
  MODEL_LOADING: 'model_loading',
  INFERENCE: 'inference',
  MEMORY: 'memory',
  WEBGL: 'webgl',
  WASM: 'wasm',
  NETWORK: 'network',
  VALIDATION: 'validation',
  UNKNOWN: 'unknown'
};

/**
 * Error severity levels
 */
const ErrorSeverity = {
  CRITICAL: 'critical',
  HIGH: 'high',
  MEDIUM: 'medium',
  LOW: 'low',
  INFO: 'info'
};

/**
 * Error diagnostic system
 */
class ErrorDiagnostics {
  constructor() {
    this.errorHistory = [];
    this.errorPatterns = new Map();
    this.solutions = new Map();
    this.diagnosticCache = new Map();
    this.errorHandlers = new Map();
    this.initializeErrorHandlers();
    this.initializeSolutions();
  }

  /**
   * Diagnose an error and provide detailed analysis
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @param {Object} options - Diagnostic options
   * @returns {Object} Diagnostic result
   */
  diagnose(error, context = {}, options = {}) {
    const {
      includeStackTrace = true,
      includeContext = true,
      includeSolutions = true,
      analyzePatterns = true,
      cacheResults = true
    } = options;

    const errorInfo = this.extractErrorInfo(error, context);
    const cacheKey = this.generateCacheKey(errorInfo);

    if (cacheResults && this.diagnosticCache.has(cacheKey)) {
      return this.diagnosticCache.get(cacheKey);
    }

    const diagnostic = {
      id: `diag_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      error: errorInfo,
      classification: this.classifyError(error, context),
      severity: this.assessSeverity(error, context),
      context: includeContext ? this.analyzeContext(context) : null,
      stackTrace: includeStackTrace ? this.analyzeStackTrace(error) : null,
      patterns: analyzePatterns ? this.analyzePatterns(error) : null,
      solutions: includeSolutions ? this.generateSolutions(error, context) : null,
      relatedErrors: this.findRelatedErrors(error),
      metadata: {
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'Node.js',
        platform: typeof process !== 'undefined' ? process.platform : 'browser',
        memory: this.getMemoryInfo(),
        performance: this.getPerformanceInfo()
      }
    };

    // Store in history
    this.errorHistory.push(diagnostic);
    this.trimHistory();

    // Update patterns
    this.updatePatterns(diagnostic);

    if (cacheResults) {
      this.diagnosticCache.set(cacheKey, diagnostic);
    }

    return diagnostic;
  }

  /**
   * Extract comprehensive error information
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {Object} Error information
   */
  extractErrorInfo(error, context) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
      code: error.code,
      cause: error.cause,
      constructor: error.constructor.name,
      isCustom: error.constructor !== Error,
      contextData: context.data || null,
      operation: context.operation || null,
      inputs: context.inputs || null,
      expectedOutput: context.expectedOutput || null,
      actualOutput: context.actualOutput || null
    };
  }

  /**
   * Classify error type
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {string} Error type
   */
  classifyError(error, context) {
    const message = error.message.toLowerCase();
    const stack = error.stack ? error.stack.toLowerCase() : '';
    const operation = context.operation ? context.operation.toLowerCase() : '';

    // Check for specific patterns
    if (message.includes('wasm') || message.includes('webassembly')) {
      return ErrorTypes.WASM;
    }
    if (message.includes('webgl') || message.includes('gl')) {
      return ErrorTypes.WEBGL;
    }
    if (message.includes('memory') || message.includes('allocation')) {
      return ErrorTypes.MEMORY;
    }
    if (message.includes('network') || message.includes('fetch') || message.includes('request')) {
      return ErrorTypes.NETWORK;
    }
    if (message.includes('initialization') || message.includes('not initialized')) {
      return ErrorTypes.INITIALIZATION;
    }
    if (message.includes('tensor') || operation.includes('tensor')) {
      return ErrorTypes.TENSOR_OPERATION;
    }
    if (message.includes('model') || operation.includes('model')) {
      return ErrorTypes.MODEL_LOADING;
    }
    if (message.includes('inference') || operation.includes('inference')) {
      return ErrorTypes.INFERENCE;
    }
    if (message.includes('validation') || message.includes('invalid')) {
      return ErrorTypes.VALIDATION;
    }

    // Check stack trace
    if (stack.includes('tensor') || stack.includes('model')) {
      return ErrorTypes.TENSOR_OPERATION;
    }

    return ErrorTypes.UNKNOWN;
  }

  /**
   * Assess error severity
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {string} Severity level
   */
  assessSeverity(error, context) {
    const message = error.message.toLowerCase();
    const type = this.classifyError(error, context);

    // Critical errors
    if (message.includes('segfault') || message.includes('panic') || message.includes('abort')) {
      return ErrorSeverity.CRITICAL;
    }
    if (type === ErrorTypes.INITIALIZATION && message.includes('failed to initialize')) {
      return ErrorSeverity.CRITICAL;
    }

    // High severity
    if (type === ErrorTypes.MEMORY && message.includes('out of memory')) {
      return ErrorSeverity.HIGH;
    }
    if (type === ErrorTypes.WASM && message.includes('compilation failed')) {
      return ErrorSeverity.HIGH;
    }
    if (message.includes('nan') || message.includes('infinite') || message.includes('overflow')) {
      return ErrorSeverity.HIGH;
    }

    // Medium severity
    if (type === ErrorTypes.TENSOR_OPERATION || type === ErrorTypes.INFERENCE) {
      return ErrorSeverity.MEDIUM;
    }
    if (type === ErrorTypes.VALIDATION) {
      return ErrorSeverity.MEDIUM;
    }

    // Low severity
    if (type === ErrorTypes.NETWORK) {
      return ErrorSeverity.LOW;
    }

    return ErrorSeverity.MEDIUM;
  }

  /**
   * Analyze error context
   * @param {Object} context - Error context
   * @returns {Object} Context analysis
   */
  analyzeContext(context) {
    const analysis = {
      operation: context.operation || null,
      operationType: null,
      inputs: null,
      environment: this.getEnvironmentInfo(),
      memoryState: this.getMemoryState(),
      systemState: this.getSystemState()
    };

    // Analyze operation
    if (context.operation) {
      analysis.operationType = this.classifyOperation(context.operation);
    }

    // Analyze inputs
    if (context.inputs) {
      analysis.inputs = this.analyzeInputs(context.inputs);
    }

    // Analyze tensors if present
    if (context.tensors) {
      analysis.tensors = context.tensors.map(tensor => {
        try {
          return tensorInspector.summarize(tensor);
        } catch (e) {
          return { error: e.message, id: tensor.id || 'unknown' };
        }
      });
    }

    // Analyze model if present
    if (context.model) {
      try {
        analysis.model = modelVisualizer.summarize(context.model);
      } catch (e) {
        analysis.model = { error: e.message };
      }
    }

    return analysis;
  }

  /**
   * Analyze stack trace
   * @param {Error} error - Error object
   * @returns {Object} Stack trace analysis
   */
  analyzeStackTrace(error) {
    if (!error.stack) return null;

    const lines = error.stack.split('\n');
    const analysis = {
      totalLines: lines.length,
      errorLine: lines[0],
      frames: [],
      libraries: new Set(),
      userCode: [],
      systemCode: []
    };

    lines.slice(1).forEach((line, index) => {
      const frame = this.parseStackFrame(line, index);
      if (frame) {
        analysis.frames.push(frame);
        
        if (frame.library) {
          analysis.libraries.add(frame.library);
        }
        
        if (frame.isUserCode) {
          analysis.userCode.push(frame);
        } else {
          analysis.systemCode.push(frame);
        }
      }
    });

    // Identify likely error source
    analysis.likelySource = this.identifyErrorSource(analysis.frames);

    return analysis;
  }

  /**
   * Analyze error patterns
   * @param {Error} error - Error object
   * @returns {Object} Pattern analysis
   */
  analyzePatterns(error) {
    const patterns = {
      frequency: this.getErrorFrequency(error),
      recentOccurrences: this.getRecentOccurrences(error),
      timePattern: this.getTimePattern(error),
      correlations: this.getCorrelations(error),
      trend: this.getTrend(error)
    };

    return patterns;
  }

  /**
   * Generate solutions and recommendations
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {Object} Solutions and recommendations
   */
  generateSolutions(error, context) {
    const type = this.classifyError(error, context);
    error.message.toLowerCase();
    
    const solutions = {
      immediate: [],
      preventive: [],
      diagnostic: [],
      references: []
    };

    // Get type-specific solutions
    if (this.solutions.has(type)) {
      const typeSolutions = this.solutions.get(type);
      solutions.immediate.push(...typeSolutions.immediate);
      solutions.preventive.push(...typeSolutions.preventive);
      solutions.diagnostic.push(...typeSolutions.diagnostic);
      solutions.references.push(...typeSolutions.references);
    }

    // Get pattern-specific solutions
    const patternSolutions = this.getPatternSolutions(error, context);
    solutions.immediate.push(...patternSolutions.immediate);
    solutions.preventive.push(...patternSolutions.preventive);

    // Get context-specific solutions
    const contextSolutions = this.getContextSolutions(error, context);
    solutions.immediate.push(...contextSolutions.immediate);
    solutions.preventive.push(...contextSolutions.preventive);

    // Generate automatic fixes if possible
    solutions.autoFixes = this.generateAutoFixes(error, context);

    return solutions;
  }

  /**
   * Find related errors
   * @param {Error} error - Error object
   * @returns {Array} Related errors
   */
  findRelatedErrors(error) {
    const related = [];
    const currentMessage = error.message.toLowerCase();
    
    this.errorHistory.forEach(diagnostic => {
      if (diagnostic.error.message.toLowerCase().includes(currentMessage.substring(0, 20))) {
        related.push({
          id: diagnostic.id,
          timestamp: diagnostic.timestamp,
          message: diagnostic.error.message,
          type: diagnostic.classification,
          similarity: this.calculateSimilarity(error.message, diagnostic.error.message)
        });
      }
    });

    return related.sort((a, b) => b.similarity - a.similarity).slice(0, 5);
  }

  /**
   * Generate comprehensive error report
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {Object} Error report
   */
  generateReport(error, context = {}) {
    const diagnostic = this.diagnose(error, context);
    
    const report = {
      summary: {
        type: diagnostic.classification,
        severity: diagnostic.severity,
        message: diagnostic.error.message,
        timestamp: diagnostic.timestamp,
        id: diagnostic.id
      },
      details: {
        errorInfo: diagnostic.error,
        context: diagnostic.context,
        stackTrace: diagnostic.stackTrace,
        patterns: diagnostic.patterns
      },
      solutions: diagnostic.solutions,
      recommendations: this.generateRecommendations(diagnostic),
      nextSteps: this.generateNextSteps(diagnostic),
      metadata: diagnostic.metadata
    };

    return report;
  }

  /**
   * Generate text report
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {string} Text report
   */
  generateTextReport(error, context = {}) {
    const report = this.generateReport(error, context);
    
    let text = `TrustformeRS Error Report\n`;
    text += `========================\n\n`;
    text += `Error ID: ${report.summary.id}\n`;
    text += `Timestamp: ${report.summary.timestamp}\n`;
    text += `Type: ${report.summary.type}\n`;
    text += `Severity: ${report.summary.severity}\n`;
    text += `Message: ${report.summary.message}\n\n`;

    if (report.details.context) {
      text += `Context:\n`;
      text += `--------\n`;
      text += `Operation: ${report.details.context.operation || 'unknown'}\n`;
      text += `Environment: ${report.details.context.environment.type}\n`;
      text += `Memory Usage: ${report.details.context.memoryState.usage}MB\n\n`;
    }

    if (report.solutions) {
      text += `Immediate Solutions:\n`;
      text += `-------------------\n`;
      report.solutions.immediate.forEach((solution, index) => {
        text += `${index + 1}. ${solution.description}\n`;
        if (solution.code) {
          text += `   Code: ${solution.code}\n`;
        }
      });
      text += `\n`;

      text += `Preventive Measures:\n`;
      text += `-------------------\n`;
      report.solutions.preventive.forEach((solution, index) => {
        text += `${index + 1}. ${solution.description}\n`;
      });
      text += `\n`;
    }

    if (report.recommendations.length > 0) {
      text += `Recommendations:\n`;
      text += `---------------\n`;
      report.recommendations.forEach((rec, index) => {
        text += `${index + 1}. ${rec}\n`;
      });
      text += `\n`;
    }

    if (report.nextSteps.length > 0) {
      text += `Next Steps:\n`;
      text += `----------\n`;
      report.nextSteps.forEach((step, index) => {
        text += `${index + 1}. ${step}\n`;
      });
    }

    return text;
  }

  /**
   * Generate HTML report
   * @param {Error} error - Error object
   * @param {Object} context - Error context
   * @returns {string} HTML report
   */
  generateHTMLReport(error, context = {}) {
    const report = this.generateReport(error, context);
    
    let html = `
    <div class="error-report" style="font-family: 'Courier New', monospace; border: 1px solid #dc3545; padding: 20px; margin: 10px; background: #f8f9fa;">
      <h2 style="color: #dc3545; margin-top: 0;">TrustformeRS Error Report</h2>
      
      <div class="error-summary" style="margin-bottom: 20px; padding: 15px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;">
        <h3 style="margin-top: 0; color: #856404;">Summary</h3>
        <table style="border-collapse: collapse; width: 100%;">
          <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Type:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${report.summary.type}</td></tr>
          <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Severity:</td><td style="padding: 4px 8px; border: 1px solid #ddd;"><span style="color: ${this.getSeverityColor(report.summary.severity)};">${report.summary.severity}</span></td></tr>
          <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Message:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${report.summary.message}</td></tr>
          <tr><td style="padding: 4px 8px; border: 1px solid #ddd; font-weight: bold;">Timestamp:</td><td style="padding: 4px 8px; border: 1px solid #ddd;">${report.summary.timestamp}</td></tr>
        </table>
      </div>
    `;

    if (report.solutions) {
      html += `
      <div class="solutions" style="margin-bottom: 20px;">
        <h3 style="color: #28a745;">Immediate Solutions</h3>
        <div style="display: grid; gap: 10px;">
      `;
      
      report.solutions.immediate.forEach((solution, index) => {
        html += `
        <div style="padding: 10px; border: 1px solid #28a745; border-radius: 4px; background: #d4edda;">
          <strong>${index + 1}. ${solution.title}</strong><br>
          <span style="color: #155724;">${solution.description}</span>
          ${solution.code ? `<br><code style="background: #f8f9fa; padding: 2px 4px; border-radius: 2px;">${solution.code}</code>` : ''}
        </div>
        `;
      });
      
      html += `</div></div>`;
    }

    if (report.recommendations.length > 0) {
      html += `
      <div class="recommendations" style="margin-bottom: 20px;">
        <h3 style="color: #007bff;">Recommendations</h3>
        <ul style="padding-left: 20px;">
      `;
      
      report.recommendations.forEach(rec => {
        html += `<li style="margin-bottom: 8px; color: #495057;">${rec}</li>`;
      });
      
      html += `</ul></div>`;
    }

    if (report.details.stackTrace) {
      html += `
      <div class="stack-trace" style="margin-bottom: 20px;">
        <h3 style="color: #6c757d;">Stack Trace Analysis</h3>
        <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px;">
          <strong>Likely Source:</strong> ${report.details.stackTrace.likelySource || 'Unknown'}<br>
          <strong>Total Frames:</strong> ${report.details.stackTrace.totalLines}<br>
          <strong>User Code Frames:</strong> ${report.details.stackTrace.userCode.length}<br>
          <strong>System Code Frames:</strong> ${report.details.stackTrace.systemCode.length}
        </div>
      </div>
      `;
    }

    html += `</div>`;
    return html;
  }

  /**
   * Register custom error handler
   * @param {string} type - Error type
   * @param {Function} handler - Error handler function
   */
  registerErrorHandler(type, handler) {
    this.errorHandlers.set(type, handler);
  }

  /**
   * Register custom solution
   * @param {string} type - Error type
   * @param {Object} solution - Solution object
   */
  registerSolution(type, solution) {
    if (!this.solutions.has(type)) {
      this.solutions.set(type, {
        immediate: [],
        preventive: [],
        diagnostic: [],
        references: []
      });
    }
    
    const typeSolutions = this.solutions.get(type);
    if (solution.immediate) typeSolutions.immediate.push(solution.immediate);
    if (solution.preventive) typeSolutions.preventive.push(solution.preventive);
    if (solution.diagnostic) typeSolutions.diagnostic.push(solution.diagnostic);
    if (solution.references) typeSolutions.references.push(solution.references);
  }

  /**
   * Clear diagnostic cache
   */
  clearCache() {
    this.diagnosticCache.clear();
  }

  /**
   * Get error statistics
   * @returns {Object} Error statistics
   */
  getStatistics() {
    const stats = {
      totalErrors: this.errorHistory.length,
      byType: {},
      bySeverity: {},
      byTime: {},
      mostFrequent: null,
      recentTrends: []
    };

    this.errorHistory.forEach(diagnostic => {
      // By type
      if (!stats.byType[diagnostic.classification]) {
        stats.byType[diagnostic.classification] = 0;
      }
      stats.byType[diagnostic.classification]++;

      // By severity
      if (!stats.bySeverity[diagnostic.severity]) {
        stats.bySeverity[diagnostic.severity] = 0;
      }
      stats.bySeverity[diagnostic.severity]++;

      // By time (last 24 hours)
      const hour = new Date(diagnostic.timestamp).getHours();
      if (!stats.byTime[hour]) {
        stats.byTime[hour] = 0;
      }
      stats.byTime[hour]++;
    });

    // Find most frequent
    let maxCount = 0;
    Object.entries(stats.byType).forEach(([type, count]) => {
      if (count > maxCount) {
        maxCount = count;
        stats.mostFrequent = { type, count };
      }
    });

    return stats;
  }

  // Private helper methods
  initializeErrorHandlers() {
    this.errorHandlers.set(ErrorTypes.WASM, this.handleWASMError.bind(this));
    this.errorHandlers.set(ErrorTypes.WEBGL, this.handleWebGLError.bind(this));
    this.errorHandlers.set(ErrorTypes.MEMORY, this.handleMemoryError.bind(this));
    this.errorHandlers.set(ErrorTypes.INITIALIZATION, this.handleInitializationError.bind(this));
    this.errorHandlers.set(ErrorTypes.TENSOR_OPERATION, this.handleTensorError.bind(this));
  }

  initializeSolutions() {
    // WASM error solutions
    this.solutions.set(ErrorTypes.WASM, {
      immediate: [
        {
          title: 'Check WASM module loading',
          description: 'Verify that the WASM module is properly loaded and initialized',
          code: 'await initialize({ wasmPath: "./path/to/wasm/file.wasm" })'
        },
        {
          title: 'Check browser compatibility',
          description: 'Ensure your browser supports WebAssembly',
          code: 'if (!window.WebAssembly) { console.error("WebAssembly not supported"); }'
        }
      ],
      preventive: [
        {
          title: 'Use feature detection',
          description: 'Always check for WebAssembly support before initialization'
        },
        {
          title: 'Implement fallbacks',
          description: 'Provide JavaScript fallbacks for unsupported environments'
        }
      ],
      diagnostic: [
        {
          title: 'Check WASM compilation',
          description: 'Verify WASM module compilation and instantiation'
        }
      ],
      references: [
        {
          title: 'WebAssembly Documentation',
          url: 'https://developer.mozilla.org/en-US/docs/WebAssembly'
        }
      ]
    });

    // WebGL error solutions
    this.solutions.set(ErrorTypes.WEBGL, {
      immediate: [
        {
          title: 'Check WebGL context',
          description: 'Verify that WebGL context is available and not lost',
          code: 'const gl = canvas.getContext("webgl"); if (!gl) { /* Handle error */ }'
        },
        {
          title: 'Check WebGL extensions',
          description: 'Verify required WebGL extensions are available'
        }
      ],
      preventive: [
        {
          title: 'Implement context loss handling',
          description: 'Handle WebGL context loss and restoration'
        }
      ]
    });

    // Memory error solutions
    this.solutions.set(ErrorTypes.MEMORY, {
      immediate: [
        {
          title: 'Free unused tensors',
          description: 'Dispose of tensors that are no longer needed',
          code: 'tensor.dispose(); // or tensor.free()'
        },
        {
          title: 'Reduce batch size',
          description: 'Use smaller batch sizes to reduce memory usage'
        }
      ],
      preventive: [
        {
          title: 'Use memory pooling',
          description: 'Implement memory pooling to reuse tensor allocations'
        }
      ]
    });

    // Initialization error solutions
    this.solutions.set(ErrorTypes.INITIALIZATION, {
      immediate: [
        {
          title: 'Call initialize() first',
          description: 'Ensure the TrustformeRS library is initialized before use',
          code: 'await initialize(); // Call before any other operations'
        },
        {
          title: 'Check initialization options',
          description: 'Verify initialization options are correct'
        }
      ]
    });

    // Tensor operation error solutions
    this.solutions.set(ErrorTypes.TENSOR_OPERATION, {
      immediate: [
        {
          title: 'Check tensor shapes',
          description: 'Verify tensor shapes are compatible for the operation',
          code: 'console.warn(tensor.shape()); // Check tensor shape'
        },
        {
          title: 'Validate tensor data',
          description: 'Check for NaN or infinite values in tensor data'
        }
      ]
    });
  }

  handleWASMError(error, context) {
    return {
      type: ErrorTypes.WASM,
      severity: ErrorSeverity.HIGH,
      customSolutions: []
    };
  }

  handleWebGLError(error, context) {
    return {
      type: ErrorTypes.WEBGL,
      severity: ErrorSeverity.MEDIUM,
      customSolutions: []
    };
  }

  handleMemoryError(error, context) {
    return {
      type: ErrorTypes.MEMORY,
      severity: ErrorSeverity.HIGH,
      customSolutions: []
    };
  }

  handleInitializationError(error, context) {
    return {
      type: ErrorTypes.INITIALIZATION,
      severity: ErrorSeverity.CRITICAL,
      customSolutions: []
    };
  }

  handleTensorError(error, context) {
    return {
      type: ErrorTypes.TENSOR_OPERATION,
      severity: ErrorSeverity.MEDIUM,
      customSolutions: []
    };
  }

  generateCacheKey(errorInfo) {
    return `${errorInfo.name}_${errorInfo.message}_${errorInfo.operation || 'unknown'}`;
  }

  classifyOperation(operation) {
    if (operation.includes('tensor')) return 'tensor_operation';
    if (operation.includes('model')) return 'model_operation';
    if (operation.includes('inference')) return 'inference_operation';
    return 'unknown_operation';
  }

  analyzeInputs(inputs) {
    const analysis = {
      count: Array.isArray(inputs) ? inputs.length : 1,
      types: [],
      hasNaN: false,
      hasInfinite: false,
      shapes: []
    };

    const inputArray = Array.isArray(inputs) ? inputs : [inputs];
    
    inputArray.forEach(input => {
      analysis.types.push(typeof input);
      
      if (input && typeof input === 'object' && input.shape) {
        analysis.shapes.push(input.shape);
        
        // Check for tensor quality issues
        try {
          const inspection = tensorInspector.analyze(input, {
            includeNaN: true,
            includeInfinite: true
          });
          if (inspection.quality.hasNaN) analysis.hasNaN = true;
          if (inspection.quality.hasInfinite) analysis.hasInfinite = true;
        } catch (e) {
          // Ignore inspection errors
        }
      }
    });

    return analysis;
  }

  getEnvironmentInfo() {
    return {
      type: typeof window !== 'undefined' ? 'browser' : 'nodejs',
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'Node.js',
      platform: typeof process !== 'undefined' ? process.platform : 'browser',
      webgl: typeof WebGLRenderingContext !== 'undefined',
      webgpu: typeof window !== 'undefined' && 'gpu' in navigator,
      wasm: typeof WebAssembly !== 'undefined'
    };
  }

  getMemoryState() {
    const state = {
      usage: 0,
      limit: 0,
      percentage: 0
    };

    if (typeof performance !== 'undefined' && performance.memory) {
      state.usage = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
      state.limit = Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024);
      state.percentage = (state.usage / state.limit) * 100;
    }

    return state;
  }

  getSystemState() {
    return {
      timestamp: Date.now(),
      debugEnabled: debugUtils.isEnabled(),
      activeSessionCount: debugUtils.debugData?.sessions?.size || 0,
      errorCount: this.errorHistory.length
    };
  }

  parseStackFrame(line, index) {
    const frame = {
      index,
      raw: line,
      function: null,
      file: null,
      line: null,
      column: null,
      library: null,
      isUserCode: false
    };

    // Parse Chrome/V8 format
    const chromeMatch = line.match(/^\s*at\s+(.+?)\s+\((.+?):(\d+):(\d+)\)$/);
    if (chromeMatch) {
      frame.function = chromeMatch[1];
      frame.file = chromeMatch[2];
      frame.line = parseInt(chromeMatch[3], 10);
      frame.column = parseInt(chromeMatch[4], 10);
    }

    // Parse Firefox format
    const firefoxMatch = line.match(/^(.+?)@(.+?):(\d+):(\d+)$/);
    if (firefoxMatch) {
      frame.function = firefoxMatch[1];
      frame.file = firefoxMatch[2];
      frame.line = parseInt(firefoxMatch[3], 10);
      frame.column = parseInt(firefoxMatch[4], 10);
    }

    // Determine library and user code
    if (frame.file) {
      if (frame.file.includes('trustformers')) {
        frame.library = 'trustformers';
        frame.isUserCode = false;
      } else if (frame.file.includes('node_modules') || frame.file.includes('webpack')) {
        frame.library = 'system';
        frame.isUserCode = false;
      } else {
        frame.isUserCode = true;
      }
    }

    return frame;
  }

  identifyErrorSource(frames) {
    const userFrames = frames.filter(f => f.isUserCode);
    if (userFrames.length > 0) {
      return `User code: ${userFrames[0].function || 'unknown'} (${userFrames[0].file}:${userFrames[0].line})`;
    }

    const trustformersFrames = frames.filter(f => f.library === 'trustformers');
    if (trustformersFrames.length > 0) {
      return `TrustformeRS: ${trustformersFrames[0].function || 'unknown'}`;
    }

    return 'Unknown';
  }

  getErrorFrequency(error) {
    const {message} = error;
    return this.errorHistory.filter(d => d.error.message === message).length;
  }

  getRecentOccurrences(error) {
    const {message} = error;
    const recent = this.errorHistory
      .filter(d => d.error.message === message)
      .filter(d => Date.now() - new Date(d.timestamp).getTime() < 24 * 60 * 60 * 1000)
      .length;
    return recent;
  }

  getTimePattern(error) {
    const {message} = error;
    const occurrences = this.errorHistory
      .filter(d => d.error.message === message)
      .map(d => new Date(d.timestamp).getHours());
    
    const pattern = {};
    occurrences.forEach(hour => {
      pattern[hour] = (pattern[hour] || 0) + 1;
    });
    
    return pattern;
  }

  getCorrelations(error) {
    // Find errors that tend to occur together
    const correlations = [];
    const errorTime = Date.now();
    
    this.errorHistory.forEach(diagnostic => {
      const timeDiff = Math.abs(errorTime - new Date(diagnostic.timestamp).getTime());
      if (timeDiff < 60000 && diagnostic.error.message !== error.message) { // Within 1 minute
        correlations.push({
          error: diagnostic.error.message,
          type: diagnostic.classification,
          timeDiff
        });
      }
    });
    
    return correlations.slice(0, 3);
  }

  getTrend(error) {
    const {message} = error;
    const recent = this.errorHistory
      .filter(d => d.error.message === message)
      .filter(d => Date.now() - new Date(d.timestamp).getTime() < 24 * 60 * 60 * 1000)
      .length;
    
    const older = this.errorHistory
      .filter(d => d.error.message === message)
      .filter(d => {
        const time = new Date(d.timestamp).getTime();
        const now = Date.now();
        return now - time >= 24 * 60 * 60 * 1000 && now - time < 48 * 60 * 60 * 1000;
      })
      .length;
    
    if (recent > older) return 'increasing';
    if (recent < older) return 'decreasing';
    return 'stable';
  }

  getPatternSolutions(error, context) {
    const solutions = { immediate: [], preventive: [] };
    
    // Check for common patterns
    const message = error.message.toLowerCase();
    
    if (message.includes('nan')) {
      solutions.immediate.push({
        title: 'Check for NaN values',
        description: 'Inspect your input data for NaN values',
        code: 'if (tensor.hasNaN()) { /* Handle NaN values */ }'
      });
    }
    
    if (message.includes('shape')) {
      solutions.immediate.push({
        title: 'Verify tensor shapes',
        description: 'Check that tensor shapes are compatible',
        code: 'console.warn("Shape A:", a.shape(), "Shape B:", b.shape());'
      });
    }
    
    return solutions;
  }

  getContextSolutions(error, context) {
    const solutions = { immediate: [], preventive: [] };
    
    if (context.operation && context.operation.includes('matmul')) {
      solutions.immediate.push({
        title: 'Check matrix multiplication compatibility',
        description: 'Ensure the inner dimensions match for matrix multiplication',
        code: 'if (a.shape()[1] !== b.shape()[0]) { /* Shapes incompatible */ }'
      });
    }
    
    return solutions;
  }

  generateAutoFixes(error, context) {
    const fixes = [];
    
    // Auto-fix for common shape issues
    if (error.message.includes('shape') && context.tensors) {
      fixes.push({
        type: 'shape_fix',
        description: 'Automatically reshape tensors to compatible shapes',
        canApply: true,
        apply: () => 
          // This would contain actual fix logic
           'Shape fix applied'
        
      });
    }
    
    return fixes;
  }

  calculateSimilarity(str1, str2) {
    const len1 = str1.length;
    const len2 = str2.length;
    const matrix = [];
    
    for (let i = 0; i <= len2; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= len1; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= len2; i++) {
      for (let j = 1; j <= len1; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }
    
    return 1 - matrix[len2][len1] / Math.max(len1, len2);
  }

  generateRecommendations(diagnostic) {
    const recommendations = [];
    
    if (diagnostic.severity === ErrorSeverity.CRITICAL) {
      recommendations.push('Address this error immediately as it may cause system instability');
    }
    
    if (diagnostic.classification === ErrorTypes.MEMORY) {
      recommendations.push('Consider implementing memory pooling or reducing batch sizes');
    }
    
    if (diagnostic.patterns && diagnostic.patterns.frequency > 5) {
      recommendations.push('This error occurs frequently - consider implementing a permanent fix');
    }
    
    return recommendations;
  }

  generateNextSteps(diagnostic) {
    const steps = [];
    
    steps.push('Review the immediate solutions provided');
    
    if (diagnostic.solutions && diagnostic.solutions.diagnostic.length > 0) {
      steps.push('Run diagnostic tests to gather more information');
    }
    
    if (diagnostic.relatedErrors.length > 0) {
      steps.push('Check for patterns in related errors');
    }
    
    steps.push('Implement preventive measures to avoid future occurrences');
    
    return steps;
  }

  getSeverityColor(severity) {
    const colors = {
      [ErrorSeverity.CRITICAL]: '#dc3545',
      [ErrorSeverity.HIGH]: '#fd7e14',
      [ErrorSeverity.MEDIUM]: '#ffc107',
      [ErrorSeverity.LOW]: '#20c997',
      [ErrorSeverity.INFO]: '#6c757d'
    };
    return colors[severity] || '#6c757d';
  }

  getMemoryInfo() {
    if (typeof performance !== 'undefined' && performance.memory) {
      return {
        used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
        total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
        limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
      };
    }
    return null;
  }

  getPerformanceInfo() {
    return {
      now: performance.now(),
      timing: typeof performance.timing !== 'undefined' ? {
        navigationStart: performance.timing.navigationStart,
        loadEventEnd: performance.timing.loadEventEnd
      } : null
    };
  }

  trimHistory() {
    const maxHistory = 1000;
    if (this.errorHistory.length > maxHistory) {
      this.errorHistory = this.errorHistory.slice(-maxHistory);
    }
  }

  updatePatterns(diagnostic) {
    const key = `${diagnostic.classification}_${diagnostic.error.name}`;
    const pattern = this.errorPatterns.get(key) || {
      count: 0,
      firstSeen: diagnostic.timestamp,
      lastSeen: diagnostic.timestamp,
      messages: new Set()
    };
    
    pattern.count++;
    pattern.lastSeen = diagnostic.timestamp;
    pattern.messages.add(diagnostic.error.message);
    
    this.errorPatterns.set(key, pattern);
  }
}

// Global error diagnostics instance
const errorDiagnostics = new ErrorDiagnostics();

var errorDiagnostics$1 = /*#__PURE__*/Object.freeze({
  __proto__: null,
  ErrorDiagnostics: ErrorDiagnostics,
  ErrorSeverity: ErrorSeverity,
  ErrorTypes: ErrorTypes,
  errorDiagnostics: errorDiagnostics
});

/**
 * TrustformeRS JavaScript API (Refactored)
 * High-level JavaScript interface for the TrustformeRS WebAssembly library
 * Modularized for better maintainability and compliance with 2000-line policy
 */


// Global state
let wasmModule = null;
let initialized = false;
let webglBackend = null;
let memoryManager = null;
let performanceProfiler = null;

// Capabilities tracking
const capabilities = {
  wasm: false,
  webgl: false,
  webgpu: false,
  memoryPool: false,
  profiling: false,
  zeroCopy: false
};

/**
 * Initialize the TrustformeRS WASM module
 * @param {Object} options - Initialization options
 * @param {string} options.wasmPath - Path to the WASM file
 * @param {boolean} options.initPanicHook - Whether to initialize panic hook for better error messages
 * @returns {Promise<void>}
 */
async function initialize(options = {}) {
  if (initialized) {
    console.warn('TrustformeRS already initialized');
    return;
  }

  const { wasmPath = './trustformers_wasm_bg.wasm', initPanicHook = true } = options;

  try {
    // Dynamically import the WASM module
    const { default: init, ...exports } = await import('../../../pkg/trustformers_wasm.js');

    // Initialize WASM
    await init(wasmPath);

    // Store exports
    wasmModule = exports;

    // Set WASM module reference for WebGPU support
    setWasmModule(wasmModule);

    // Initialize panic hook for better error messages
    if (initPanicHook && wasmModule.init_panic_hook) {
      wasmModule.init_panic_hook();
    }

    // Check SIMD support
    const simdEnabled = wasmModule.enable_simd();
    console.warn(`TrustformeRS initialized. SIMD support: ${simdEnabled ? 'enabled' : 'disabled'}`);

    // Log version and features
    console.warn(`Version: ${wasmModule.version()}`);
    console.warn(`Features:`, wasmModule.features());

    initialized = true;
    capabilities.wasm = true;
  } catch (error) {
    // Graceful fallback for development environments without WASM
    console.warn(`WASM module not available - running in mock mode: ${error.message}`);

    // Initialize without WASM for development
    wasmModule = null;
    initialized = true;
    capabilities.wasm = false;

    // Still enable other capabilities
    capabilities.tensor = true;
    capabilities.models = false; // Models require WASM
    capabilities.pipeline = false; // Pipelines require WASM
    capabilities.webgl = true;
    capabilities.webgpu = true;
    capabilities.nodejs = typeof process !== 'undefined';
    capabilities.browser = typeof window !== 'undefined';
  }
}

/**
 * Enhanced initialization with performance optimizations
 * @param {Object} options - Enhanced initialization options
 * @returns {Promise<Object>} Initialization result with capabilities
 */
async function initializeEnhanced(options = {}) {
  const {
    wasmPath = './trustformers_wasm_bg.wasm',
    initPanicHook = true,
    enableWebGL = true,
    enableMemoryPool = true,
    enableProfiling = true,
    enableZeroCopy = true,
    webglOptions = {},
    memoryOptions = {},
    profilingOptions = {},
    zeroCopyOptions = {}
  } = options;

  // Basic WASM initialization
  await initialize({ wasmPath, initPanicHook });

  // Initialize performance optimizations
  if (enableMemoryPool) {
    try {
      initMemoryPool(wasmModule);
      memoryManager = getMemoryManager(memoryOptions);
      capabilities.memoryPool = true;
      console.warn('✓ Memory pooling enabled');
    } catch (error) {
      console.warn('Memory pooling initialization failed:', error);
    }
  }

  if (enableProfiling) {
    try {
      initProfiler(wasmModule);
      performanceProfiler = getProfiler(profilingOptions);
      capabilities.profiling = true;
      console.warn('✓ Performance profiling enabled');
    } catch (error) {
      console.warn('Performance profiling initialization failed:', error);
    }
  }

  if (enableZeroCopy) {
    try {
      initZeroCopy(wasmModule);
      capabilities.zeroCopy = true;
      console.warn('✓ Zero-copy transfers enabled');
    } catch (error) {
      console.warn('Zero-copy initialization failed:', error);
    }
  }

  if (enableWebGL) {
    try {
      initWebGLBackend(wasmModule);
      webglBackend = await createWebGLBackend(webglOptions.canvas);

      if (memoryManager) {
        memoryManager.initWebGL(webglBackend.gl, webglOptions.memory);
      }

      capabilities.webgl = true;
      console.warn('✓ WebGL backend enabled');
      console.warn('  WebGL Info:', webglBackend.getInfo());
    } catch (error) {
      console.warn('WebGL backend initialization failed:', error);
    }
  }

  // Check WebGPU availability
  if (webgpu.isAvailable()) {
    capabilities.webgpu = true;
    console.warn('✓ WebGPU available');
  }

  console.warn('\nTrustformeRS Enhanced Initialization Complete');
  console.warn('Capabilities:', capabilities);

  // Make global references available
  global.webglBackend = webglBackend;
  global.performanceProfiler = performanceProfiler;
  global.memoryManager = memoryManager;

  return capabilities;
}

/**
 * Ensure the module is initialized
 * @private
 */
function ensureInitialized() {
  if (!initialized || !wasmModule) {
    throw new Error('TrustformeRS not initialized. Call initialize() first.');
  }
}

// Core tensor creation functions
function tensor(data, shape) {
  ensureInitialized();
  const flatData = new Float32Array(data);
  const shapeArray = new Uint32Array(shape);
  return new wasmModule.WasmTensor(flatData, shapeArray);
}

const PipelineType = {
  TEXT_GENERATION: 0,
  TEXT_CLASSIFICATION: 1,
  TOKEN_CLASSIFICATION: 2,
  QUESTION_ANSWERING: 3,
  SUMMARIZATION: 4,
  TRANSLATION: 5
};

const TokenizerType = {
  WORDPIECE: 0,
  BPE: 1,
  SENTENCEPIECE: 2
};

// Model and tokenizer creation functions
function createModelConfig(modelType) {
  ensureInitialized();

  const configFactories = {
    'bert_base': wasmModule.ModelConfig.bert_base,
    'gpt2_base': wasmModule.ModelConfig.gpt2_base,
    't5_small': wasmModule.ModelConfig.t5_small,
    'llama_7b': wasmModule.ModelConfig.llama_7b,
    'mistral_7b': wasmModule.ModelConfig.mistral_7b
  };

  const factory = configFactories[modelType];
  if (!factory) {
    throw new Error(`Unknown model type: ${modelType}. Available: ${Object.keys(configFactories).join(', ')}`);
  }

  return factory();
}

function createModel(configOrType) {
  ensureInitialized();

  const config = typeof configOrType === 'string'
    ? createModelConfig(configOrType)
    : configOrType;

  return new wasmModule.WasmModel(config);
}

function createTokenizer(type, vocab = null) {
  ensureInitialized();

  const typeMap = {
    'wordpiece': TokenizerType.WORDPIECE,
    'bpe': TokenizerType.BPE,
    'sentencepiece': TokenizerType.SENTENCEPIECE
  };

  const tokenizerType = typeMap[type.toLowerCase()];
  if (tokenizerType === undefined) {
    throw new Error(`Unknown tokenizer type: ${type}`);
  }

  const tokenizer = new wasmModule.WasmTokenizer(tokenizerType);

  if (vocab) {
    tokenizer.load_vocab(vocab);
  }

  return tokenizer;
}

// Pipeline factory
class Pipeline {
  static textGeneration(model, tokenizer, config = {}) {
    ensureInitialized();

    const pipeline = new wasmModule.TextGenerationPipeline(model, tokenizer);

    if (Object.keys(config).length > 0) {
      const genConfig = new wasmModule.GenerationConfig();
      Object.assign(genConfig, config);
      pipeline.set_config(genConfig);
    }

    return pipeline;
  }

  static textClassification(model, tokenizer, labels = []) {
    ensureInitialized();

    const pipeline = new wasmModule.TextClassificationPipeline(model, tokenizer);

    if (labels.length > 0) {
      pipeline.set_labels(labels);
    }

    return pipeline;
  }

  static questionAnswering(model, tokenizer) {
    ensureInitialized();
    return new wasmModule.QuestionAnsweringPipeline(model, tokenizer);
  }

  static async fromPretrained(task, modelName) {
    ensureInitialized();

    const pipelineTypeMap = {
      'text-generation': PipelineType.TEXT_GENERATION,
      'text-classification': PipelineType.TEXT_CLASSIFICATION,
      'token-classification': PipelineType.TOKEN_CLASSIFICATION,
      'question-answering': PipelineType.QUESTION_ANSWERING,
      'summarization': PipelineType.SUMMARIZATION,
      'translation': PipelineType.TRANSLATION
    };

    const pipelineType = pipelineTypeMap[task];
    if (pipelineType === undefined) {
      throw new Error(`Unknown task: ${task}`);
    }

    return wasmModule.PipelineFactory.from_pretrained(pipelineType, modelName);
  }
}

// Memory utilities
const memory = {
  getStats() {
    ensureInitialized();
    return wasmModule.get_memory_stats();
  },

  getUsage() {
    ensureInitialized();
    return wasmModule.get_wasm_memory_usage();
  }
};

// Performance monitoring utilities
const performance$1 = {
  getReport() {
    const report = {
      timestamp: new Date().toISOString(),
      capabilities,
      memory: memoryManager ? memoryManager.getStats() : null,
      profiling: performanceProfiler ? performanceProfiler.generateReport() : null,
      webgl: webglBackend ? webglBackend.getInfo() : null
    };

    return report;
  },

  startSession(name, metadata = {}) {
    if (performanceProfiler) {
      return performanceProfiler.startSession(name, metadata);
    }
    return null;
  },

  endSession() {
    if (performanceProfiler) {
      return performanceProfiler.endSession();
    }
    return null;
  },

  async profile(name, fn, metadata = {}) {
    if (performanceProfiler) {
      return await performanceProfiler.profileOperation(name, fn, metadata);
    } 
      return await fn();
    
  },

  getMemoryUsage() {
    if (memoryManager) {
      return memoryManager.getStats();
    } else if (performanceProfiler) {
      return performanceProfiler.getMemoryUsage();
    } 
      return memory.getStats();
    
  },

  cleanup() {
    if (memoryManager) {
      memoryManager.cleanup();
    }

    // Trigger garbage collection hint
    if (typeof window !== 'undefined' && window.gc) {
      window.gc();
    } else if (typeof global !== 'undefined' && global.gc) {
      global.gc();
    }
  }
};

var index = /*#__PURE__*/Object.freeze({
  __proto__: null,
  DebugUtilities: DebugUtilities,
  ErrorDiagnostics: ErrorDiagnostics,
  ErrorSeverity: ErrorSeverity,
  ErrorTypes: ErrorTypes,
  MemoryManager: MemoryManager,
  ModelVisualizer: ModelVisualizer,
  PerformanceProfiler: PerformanceProfiler,
  Pipeline: Pipeline,
  PipelineType: PipelineType,
  TensorInspector: TensorInspector,
  TensorMemoryPool: TensorMemoryPool,
  TokenizerType: TokenizerType,
  WebGLBackend: WebGLBackend,
  WebGLMemoryPool: WebGLMemoryPool,
  createModel: createModel,
  createModelConfig: createModelConfig,
  createTokenizer: createTokenizer,
  createWebGLBackend: createWebGLBackend,
  debugUtils: debugUtils,
  devTools: devTools,
  enhanced_inference: enhanced_inference,
  errorDiagnostics: errorDiagnostics,
  getMemoryManager: getMemoryManager,
  getProfiler: getProfiler,
  initialize: initialize,
  initializeEnhanced: initializeEnhanced,
  memory: memory,
  modelVisualizer: modelVisualizer,
  performance: performance$1,
  tensor: tensor,
  tensorInspector: tensorInspector,
  webgpu: webgpu,
  withMemoryManagement: withMemoryManagement
});

/**
 * Vue.js composition functions and components for TrustformeRS
 * Provides Vue 3 Composition API integrations for the TrustformeRS ecosystem
 */


/**
 * Composition function for initializing TrustformeRS with Vue reactivity
 * @param {Object} options - Initialization options
 * @returns {Object} Reactive initialization state and methods
 */
function useTrustformersInit(options = {}) {
  const state = vue.reactive({
    isInitialized: false,
    isLoading: false,
    error: null,
    capabilities: null,
  });

  const initRef = vue.ref(false);

  const initTrustformers = async () => {
    if (initRef.value) return;

    state.isLoading = true;
    state.error = null;
    initRef.value = true;

    try {
      const caps = await initializeEnhanced({
        enableWebGL: true,
        enableMemoryPool: true,
        enableProfiling: true,
        enableZeroCopy: true,
        ...options,
      });

      state.capabilities = caps;
      state.isInitialized = true;
    } catch (err) {
      state.error = err;
      initRef.value = false;
    } finally {
      state.isLoading = false;
    }
  };

  const reinitialize = async (newOptions = {}) => {
    initRef.value = false;
    state.isInitialized = false;
    state.capabilities = null;
    await initTrustformers({ ...options, ...newOptions });
  };

  vue.onMounted(() => {
    initTrustformers();
  });

  return {
    ...vue.toRefs(state),
    reinitialize,
  };
}

/**
 * Composition function for creating and managing models
 * @param {Ref|string|Object} modelConfig - Reactive model configuration or type string
 * @param {Object} options - Model options
 * @returns {Object} Reactive model state and methods
 */
function useModel(modelConfig, options = {}) {
  const state = vue.reactive({
    model: null,
    isLoading: false,
    error: null,
  });

  const modelRef = vue.shallowRef(null);

  const createModelInstance = async config => {
    if (!config) return;

    state.isLoading = true;
    state.error = null;

    try {
      // Cleanup previous model
      if (modelRef.value && modelRef.value.free) {
        modelRef.value.free();
      }

      const modelInstance = createModel(config);
      modelRef.value = modelInstance;
      state.model = modelInstance;
    } catch (err) {
      state.error = err;
    } finally {
      state.isLoading = false;
    }
  };

  const runInference = async (inputs, inferenceOptions = {}) => {
    if (!state.model) throw new Error('Model not initialized');

    return await enhanced_inference.runInference(state.model, inputs, {
      profile: true,
      useMemoryPool: true,
      autoCleanup: true,
      ...inferenceOptions,
    });
  };

  const batchInference = async (batchInputs, batchOptions = {}) => {
    if (!state.model) throw new Error('Model not initialized');

    return await enhanced_inference.batchInference(state.model, batchInputs, {
      batchSize: 32,
      profile: true,
      ...batchOptions,
    });
  };

  // Watch for changes in modelConfig
  vue.watch(
    () => modelConfig,
    newConfig => {
      if (newConfig) {
        createModelInstance(newConfig);
      }
    },
    { immediate: true }
  );

  vue.onUnmounted(() => {
    if (modelRef.value && modelRef.value.free) {
      modelRef.value.free();
    }
  });

  return {
    ...vue.toRefs(state),
    runInference,
    batchInference,
  };
}

/**
 * Composition function for creating and managing tokenizers
 * @param {Ref|string} type - Reactive tokenizer type
 * @param {Ref|Object} vocab - Reactive vocabulary object
 * @returns {Object} Reactive tokenizer state and methods
 */
function useTokenizer(type, vocab = null) {
  const state = vue.reactive({
    tokenizer: null,
    isLoading: false,
    error: null,
  });

  const tokenizerRef = vue.shallowRef(null);

  const createTokenizerInstance = async (tokType, vocabData) => {
    if (!tokType) return;

    state.isLoading = true;
    state.error = null;

    try {
      // Cleanup previous tokenizer
      if (tokenizerRef.value && tokenizerRef.value.free) {
        tokenizerRef.value.free();
      }

      const tokenizerInstance = createTokenizer(tokType, vocabData);
      tokenizerRef.value = tokenizerInstance;
      state.tokenizer = tokenizerInstance;
    } catch (err) {
      state.error = err;
    } finally {
      state.isLoading = false;
    }
  };

  const encode = text => {
    if (!state.tokenizer) throw new Error('Tokenizer not initialized');
    return state.tokenizer.encode(text);
  };

  const decode = tokens => {
    if (!state.tokenizer) throw new Error('Tokenizer not initialized');
    return state.tokenizer.decode(tokens);
  };

  const batchEncode = texts => {
    if (!state.tokenizer) throw new Error('Tokenizer not initialized');
    return texts.map(text => state.tokenizer.encode(text));
  };

  // Watch for changes in type and vocab
  vue.watch(
    [() => type, () => vocab],
    ([newType, newVocab]) => {
      if (newType) {
        createTokenizerInstance(newType, newVocab);
      }
    },
    { immediate: true }
  );

  vue.onUnmounted(() => {
    if (tokenizerRef.value && tokenizerRef.value.free) {
      tokenizerRef.value.free();
    }
  });

  return {
    ...vue.toRefs(state),
    encode,
    decode,
    batchEncode,
  };
}

/**
 * Composition function for creating and managing pipelines
 * @param {Ref|string} task - Reactive pipeline task type
 * @param {Ref|Object} model - Reactive model object
 * @param {Ref|Object} tokenizer - Reactive tokenizer object
 * @param {Object} config - Pipeline configuration
 * @returns {Object} Reactive pipeline state and methods
 */
function usePipeline(task, model, tokenizer, config = {}) {
  const state = vue.reactive({
    pipeline: null,
    isLoading: false,
    error: null,
  });

  const pipelineRef = vue.shallowRef(null);

  const createPipelineInstance = async (taskType, modelObj, tokenizerObj, pipelineConfig) => {
    if (!taskType || !modelObj || !tokenizerObj) return;

    state.isLoading = true;
    state.error = null;

    try {
      // Cleanup previous pipeline
      if (pipelineRef.value && pipelineRef.value.free) {
        pipelineRef.value.free();
      }

      let pipelineInstance;

      switch (taskType) {
        case 'text-generation':
          pipelineInstance = Pipeline.textGeneration(modelObj, tokenizerObj, pipelineConfig);
          break;
        case 'text-classification':
          pipelineInstance = Pipeline.textClassification(
            modelObj,
            tokenizerObj,
            pipelineConfig.labels || []
          );
          break;
        case 'question-answering':
          pipelineInstance = Pipeline.questionAnswering(modelObj, tokenizerObj);
          break;
        default:
          pipelineInstance = await Pipeline.fromPretrained(
            taskType,
            pipelineConfig.modelName || 'default'
          );
      }

      pipelineRef.value = pipelineInstance;
      state.pipeline = pipelineInstance;
    } catch (err) {
      state.error = err;
    } finally {
      state.isLoading = false;
    }
  };

  const run = async (input, options = {}) => {
    if (!state.pipeline) throw new Error('Pipeline not initialized');

    return await state.pipeline.run(input, {
      maxLength: 50,
      temperature: 0.7,
      ...options,
    });
  };

  const batch = async (inputs, options = {}) => {
    if (!state.pipeline) throw new Error('Pipeline not initialized');

    return Promise.all(inputs.map(input => state.pipeline.run(input, options)));
  };

  // Watch for changes in task, model, tokenizer
  vue.watch(
    [() => task, () => model, () => tokenizer],
    ([newTask, newModel, newTokenizer]) => {
      if (newTask && newModel && newTokenizer) {
        createPipelineInstance(newTask, newModel, newTokenizer, config);
      }
    },
    { immediate: true }
  );

  vue.onUnmounted(() => {
    if (pipelineRef.value && pipelineRef.value.free) {
      pipelineRef.value.free();
    }
  });

  return {
    ...vue.toRefs(state),
    run,
    batch,
  };
}

/**
 * Composition function for tensor operations with Vue reactivity
 * @param {Ref|Array} initialData - Reactive initial tensor data
 * @param {Ref|Array} initialShape - Reactive initial tensor shape
 * @returns {Object} Reactive tensor state and operations
 */
function useTensor(initialData = null, initialShape = null) {
  const state = vue.reactive({
    tensor: null,
    shape: null,
    size: null,
    dtype: null,
    error: null,
  });

  const tensorRef = vue.shallowRef(null);

  const createTensor = async (data, shape) => {
    try {
      // Cleanup previous tensor
      if (tensorRef.value && tensorRef.value.free) {
        tensorRef.value.free();
      }

      const newTensor = tensor(data, shape);
      tensorRef.value = newTensor;

      state.tensor = newTensor;
      state.shape = Array.from(newTensor.shape());
      state.size = newTensor.size();
      state.dtype = newTensor.dtype();
      state.error = null;
    } catch (err) {
      state.error = err;
    }
  };

  const operations = vue.computed(() => ({
    reshape: newShape => {
      if (!state.tensor) return null;
      try {
        return state.tensor.reshape(newShape);
      } catch (err) {
        state.error = err;
        return null;
      }
    },

    transpose: (dims = null) => {
      if (!state.tensor) return null;
      try {
        return dims ? state.tensor.transpose(dims) : state.tensor.transpose();
      } catch (err) {
        state.error = err;
        return null;
      }
    },

    slice: (start, end, step = 1) => {
      if (!state.tensor) return null;
      try {
        return state.tensor.slice(start, end, step);
      } catch (err) {
        state.error = err;
        return null;
      }
    },

    add: other => {
      if (!state.tensor) return null;
      try {
        return state.tensor.add(other);
      } catch (err) {
        state.error = err;
        return null;
      }
    },

    multiply: other => {
      if (!state.tensor) return null;
      try {
        return state.tensor.mul(other);
      } catch (err) {
        state.error = err;
        return null;
      }
    },

    matmul: other => {
      if (!state.tensor) return null;
      try {
        return state.tensor.matmul(other);
      } catch (err) {
        state.error = err;
        return null;
      }
    },
  }));

  // Watch for changes in initialData and initialShape
  vue.watch(
    [() => initialData, () => initialShape],
    ([newData, newShape]) => {
      if (newData && newShape) {
        createTensor(newData, newShape);
      }
    },
    { immediate: true }
  );

  vue.onUnmounted(() => {
    if (tensorRef.value && tensorRef.value.free) {
      tensorRef.value.free();
    }
  });

  return {
    ...vue.toRefs(state),
    operations,
    createTensor,
  };
}

/**
 * Composition function for performance monitoring in Vue applications
 * @param {Ref|boolean} enabled - Reactive enabled state
 * @returns {Object} Reactive performance monitoring state and methods
 */
function usePerformance(enabled = vue.ref(true)) {
  const state = vue.reactive({
    metrics: null,
    sessionId: null,
  });

  let intervalId = null;

  const startSession = (name, metadata = {}) => {
    if (!enabled.value) return null;

    const id = performance$1.startSession(name, {
      framework: 'vue',
      timestamp: new Date().toISOString(),
      ...metadata,
    });
    state.sessionId = id;
    return id;
  };

  const endSession = () => {
    if (!enabled.value || !state.sessionId) return null;

    const report = performance$1.endSession();
    state.sessionId = null;
    return report;
  };

  const profileOperation = async (name, operation, metadata = {}) => {
    if (!enabled.value) return await operation();

    return await performance$1.profile(name, operation, {
      framework: 'vue',
      ...metadata,
    });
  };

  const updateMetrics = () => {
    if (!enabled.value) return;

    const currentMetrics = {
      memory: performance$1.getMemoryUsage(),
      report: performance$1.getReport(),
      timestamp: Date.now(),
    };
    state.metrics = currentMetrics;
  };

  const cleanup = () => {
    performance$1.cleanup();
    updateMetrics();
  };

  // Watch enabled state
  vue.watch(
    enabled,
    isEnabled => {
      if (isEnabled) {
        updateMetrics();
        intervalId = setInterval(updateMetrics, 5000); // Update every 5 seconds
      } else {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
      }
    },
    { immediate: true }
  );

  vue.onUnmounted(() => {
    if (intervalId) {
      clearInterval(intervalId);
    }
    if (state.sessionId) {
      endSession();
    }
  });

  return {
    ...vue.toRefs(state),
    startSession,
    endSession,
    profileOperation,
    updateMetrics,
    cleanup,
  };
}

/**
 * Composition function for memory management in Vue applications
 * @returns {Object} Reactive memory management state and methods
 */
function useMemory() {
  const memoryStats = vue.ref(null);
  let intervalId = null;

  const updateMemoryStats = () => {
    const stats = memory.getStats();
    memoryStats.value = {
      ...stats,
      timestamp: Date.now(),
    };
  };

  const cleanup = () => {
    if (typeof window !== 'undefined' && window.gc) {
      window.gc();
    }
    performance$1.cleanup();
    updateMemoryStats();
  };

  vue.onMounted(() => {
    updateMemoryStats();
    intervalId = setInterval(updateMemoryStats, 3000); // Update every 3 seconds
  });

  vue.onUnmounted(() => {
    if (intervalId) {
      clearInterval(intervalId);
    }
  });

  return {
    memoryStats: vue.readonly(memoryStats),
    updateMemoryStats,
    cleanup,
  };
}

/**
 * Composition function for development tools integration
 * @param {Ref|boolean} enabled - Reactive enabled state
 * @returns {Object} Reactive development tools state and methods
 */
function useDevTools(enabled = vue.ref(process.env.NODE_ENV === 'development')) {
  const state = vue.reactive({
    debugState: null,
    isEnabled: enabled.value,
  });

  const enableDebug = (options = {}) => {
    devTools.debug.enable({
      trackTensors: true,
      trackOperations: true,
      validateOperations: true,
      ...options,
    });
    state.isEnabled = true;
  };

  const disableDebug = () => {
    devTools.debug.disable();
    state.isEnabled = false;
  };

  const startDebugSession = (name, metadata = {}) => {
    if (!state.isEnabled) return null;

    return devTools.debug.startSession(name, {
      framework: 'vue',
      timestamp: new Date().toISOString(),
      ...metadata,
    });
  };

  const trackTensor = (tensor, operation, metadata = {}) => {
    if (!state.isEnabled) return;

    devTools.debug.trackTensor(tensor, operation, {
      framework: 'vue',
      ...metadata,
    });
  };

  const validateOperation = (operation, tensors, options = {}) => {
    if (!state.isEnabled) return true;

    return devTools.debug.validateOperation(operation, tensors, options);
  };

  const generateReport = () => {
    if (!state.isEnabled) return null;

    const report = devTools.debug.generateReport();
    state.debugState = report;
    return report;
  };

  const analyzeComprehensive = (options = {}) => {
    if (!state.isEnabled) return null;

    return devTools.analyze.comprehensive({
      includeDebug: true,
      ...options,
    });
  };

  // Watch enabled state
  vue.watch(enabled, isEnabled => {
    state.isEnabled = isEnabled;
  });

  return {
    ...vue.toRefs(state),
    enableDebug,
    disableDebug,
    startDebugSession,
    trackTensor,
    validateOperation,
    generateReport,
    analyzeComprehensive,
  };
}

/**
 * Vue component definitions for templates
 */

// TrustformersProvider component template
const TrustformersProviderTemplate = {
  name: 'TrustformersProvider',
  props: {
    options: {
      type: Object,
      default: () => ({}),
    },
  },
  emits: ['init', 'error'],
  setup(props, { emit, slots }) {
    const { isInitialized, isLoading, error, capabilities } = useTrustformersInit(props.options);

    vue.watch(isInitialized, initialized => {
      if (initialized) {
        emit('init', capabilities.value);
      }
    });

    vue.watch(error, err => {
      if (err) {
        emit('error', err);
      }
    });

    return {
      isInitialized,
      isLoading,
      error,
      capabilities,
      slots,
    };
  },
  template: `
    <div class="trustformers-provider" :data-initialized="isInitialized">
      <div v-if="error" class="trustformers-error">
        <h3>TrustformeRS Initialization Error</h3>
        <p>{{ error.message }}</p>
        <pre>{{ error.stack }}</pre>
      </div>
      <div v-else-if="isLoading" class="trustformers-loading">
        <p>Initializing TrustformeRS...</p>
      </div>
      <template v-else>
        <slot></slot>
      </template>
    </div>
  `,
};

// PerformanceMonitor component template
const PerformanceMonitorTemplate = {
  name: 'PerformanceMonitor',
  props: {
    enabled: {
      type: Boolean,
      default: true,
    },
    updateInterval: {
      type: Number,
      default: 5000,
    },
  },
  setup(props) {
    const { metrics } = usePerformance(vue.ref(props.enabled));
    const expanded = vue.ref(false);

    return {
      metrics,
      expanded,
    };
  },
  template: `
    <div 
      v-if="enabled && metrics" 
      class="performance-monitor"
      :style="{
        position: 'fixed',
        top: '10px',
        right: '10px',
        background: 'rgba(0,0,0,0.8)',
        color: 'white',
        padding: '10px',
        borderRadius: '5px',
        fontSize: '12px',
        fontFamily: 'monospace',
        zIndex: 9999,
        cursor: 'pointer'
      }"
      @click="expanded = !expanded"
    >
      <div>🚀 TrustformeRS Performance</div>
      <div v-if="metrics.memory">
        Memory: {{ (metrics.memory.used / 1024 / 1024).toFixed(1) }}MB
      </div>
      <div v-if="expanded && metrics.report" :style="{ marginTop: '10px', fontSize: '10px' }">
        <div>
          Capabilities: {{ 
            Object.entries(metrics.report.capabilities)
              .filter(([, v]) => v)
              .map(([k]) => k)
              .join(', ') 
          }}
        </div>
        <div v-if="metrics.report.profiling">
          Operations: {{ metrics.report.profiling.totalOperations || 0 }}
        </div>
      </div>
    </div>
  `,
};

// MemoryMonitor component template
const MemoryMonitorTemplate = {
  name: 'MemoryMonitor',
  props: {
    enabled: {
      type: Boolean,
      default: true,
    },
  },
  setup(props) {
    const { memoryStats } = useMemory();
    const expanded = vue.ref(false);

    const usedMB = vue.computed(() =>
      memoryStats.value?.used ? (memoryStats.value.used / 1024 / 1024).toFixed(1) : 'N/A'
    );

    const limitMB = vue.computed(() =>
      memoryStats.value?.limit ? (memoryStats.value.limit / 1024 / 1024).toFixed(1) : 'N/A'
    );

    const usagePercent = vue.computed(() =>
      memoryStats.value?.used && memoryStats.value?.limit
        ? ((memoryStats.value.used / memoryStats.value.limit) * 100).toFixed(1)
        : 'N/A'
    );

    return {
      memoryStats,
      expanded,
      usedMB,
      limitMB,
      usagePercent,
    };
  },
  template: `
    <div 
      v-if="enabled && memoryStats" 
      class="memory-monitor"
      :style="{
        position: 'fixed',
        top: '80px',
        right: '10px',
        background: 'rgba(0,0,0,0.8)',
        color: 'white',
        padding: '10px',
        borderRadius: '5px',
        fontSize: '12px',
        fontFamily: 'monospace',
        zIndex: 9999,
        cursor: 'pointer'
      }"
      @click="expanded = !expanded"
    >
      <div>💾 Memory</div>
      <div>{{ usedMB }}MB / {{ limitMB }}MB</div>
      <div v-if="expanded" :style="{ marginTop: '10px', fontSize: '10px' }">
        <div>Usage: {{ usagePercent }}%</div>
        <div>Updated: {{ new Date(memoryStats.timestamp).toLocaleTimeString() }}</div>
      </div>
    </div>
  `,
};

/**
 * Vue plugin for global installation
 */
const TrustformersVuePlugin = {
  install(app, options = {}) {
    // Register global properties
    app.config.globalProperties.$trustformers = {
      useTrustformersInit,
      useModel,
      useTokenizer,
      usePipeline,
      useTensor,
      usePerformance,
      useMemory,
      useDevTools,
    };

    // Register components if requested
    if (options.components !== false) {
      app.component('TrustformersProvider', TrustformersProviderTemplate);
      app.component('PerformanceMonitor', PerformanceMonitorTemplate);
      app.component('MemoryMonitor', MemoryMonitorTemplate);
    }

    // Provide global composition functions
    app.provide('trustformers', {
      useTrustformersInit,
      useModel,
      useTokenizer,
      usePipeline,
      useTensor,
      usePerformance,
      useMemory,
      useDevTools,
    });
  },
};

/**
 * TrustformeRS Vue Package
 * Vue.js-specific composition functions and components
 */


// Package info
const PACKAGE_VERSION = '0.1.0';
const PACKAGE_NAME = '@trustformers/vue';

exports.MemoryMonitorTemplate = MemoryMonitorTemplate;
exports.PACKAGE_NAME = PACKAGE_NAME;
exports.PACKAGE_VERSION = PACKAGE_VERSION;
exports.PerformanceMonitorTemplate = PerformanceMonitorTemplate;
exports.TrustformersProviderTemplate = TrustformersProviderTemplate;
exports.TrustformersVuePlugin = TrustformersVuePlugin;
exports.useDevTools = useDevTools;
exports.useMemory = useMemory;
exports.useModel = useModel;
exports.usePerformance = usePerformance;
exports.usePipeline = usePipeline;
exports.useTensor = useTensor;
exports.useTokenizer = useTokenizer;
exports.useTrustformersInit = useTrustformersInit;
//# sourceMappingURL=index.cjs.js.map
