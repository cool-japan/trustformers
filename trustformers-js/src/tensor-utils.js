/**
 * Comprehensive Tensor Utilities
 * Advanced tensor creation, manipulation, and statistical operations
 */

/**
 * Tensor creation utilities
 */
export const tensor_utils = {
  /**
   * Create tensor from typed array
   * @param {TypedArray} data - Typed array data
   * @param {number[]} shape - Tensor shape
   * @returns {Promise<Object>} Tensor object
   */
  async fromTypedArray(data, shape) {
    const { getRawModule } = await import('./index.js');
    const wasmModule = getRawModule();

    const typeMap = {
      'Float32Array': wasmModule.WasmTensor.from_f32,
      'Float64Array': wasmModule.WasmTensor.from_f64,
      'Int32Array': wasmModule.WasmTensor.from_i32,
      'Uint32Array': wasmModule.WasmTensor.from_u32,
      'Int8Array': wasmModule.WasmTensor.from_i8,
      'Uint8Array': wasmModule.WasmTensor.from_u8
    };

    const constructor = typeMap[data.constructor.name];
    if (!constructor) {
      throw new Error(`Unsupported typed array type: ${data.constructor.name}`);
    }

    return constructor(data, new Uint32Array(shape));
  },

  /**
   * Create tensor from nested JavaScript array
   * @param {Array} array - Nested array
   * @returns {Promise<Object>} Tensor object
   */
  async fromNestedArray(array) {
    // Calculate shape
    const shape = this._calculateNestedArrayShape(array);

    // Flatten array
    const flatArray = this._flattenArray(array);

    const { tensor } = await import('./index.js');
    return tensor(flatArray, shape);
  },

  /**
   * Calculate shape of nested array
   * @private
   */
  _calculateNestedArrayShape(array) {
    const shape = [];
    let current = array;

    while (Array.isArray(current)) {
      shape.push(current.length);
      current = current[0];
    }

    return shape;
  },

  /**
   * Flatten nested array recursively
   * @private
   */
  _flattenArray(array) {
    const result = [];

    const flatten = (arr) => {
      for (const item of arr) {
        if (Array.isArray(item)) {
          flatten(item);
        } else {
          result.push(item);
        }
      }
    };

    flatten(array);
    return result;
  },

  /**
   * Create random tensor with advanced probabilistic distributions
   * @param {number[]} shape - Tensor shape
   * @param {string} distribution - Distribution type with advanced options
   * @param {Object} params - Distribution parameters
   * @param {Object} options - Advanced options
   * @returns {Promise<Object>|Object} Enhanced tensor object
   */
  async random(shape, distribution = 'normal', params = {}, options = {}) {
    const { use_advanced = true, seed = null, async_mode = false } = options;

    if (use_advanced && (async_mode || this._isAdvancedDistribution(distribution))) {
      const { create_advanced_tensor } = await import('./index.js');
      return create_advanced_tensor(shape, {
        type: distribution,
        ...params
      }, seed);
    }

    // Enhanced standard distributions with better quality
    if (use_advanced) {
      try {
        const enhancedTensor = await this._createEnhancedRandomTensor(shape, distribution, params, seed);
        return enhancedTensor;
      } catch (error) {
        console.warn('Failed to create enhanced random tensor, falling back to standard:', error.message);
        return this._fallbackRandom(shape, distribution, params);
      }
    }

    return this._fallbackRandom(shape, distribution, params);
  },

  /**
   * Check if distribution requires advanced processing
   * @private
   */
  _isAdvancedDistribution(distribution) {
    const advancedDistributions = [
      'multivariate_normal',
      'mixture_of_gaussians',
      'beta_bernoulli',
      'dirichlet_multinomial',
      'student_t',
      'gamma',
      'beta',
      'chi_squared',
      'weibull',
      'lognormal'
    ];

    return advancedDistributions.includes(distribution);
  },

  /**
   * Create enhanced random tensor with improved quality
   * @private
   */
  async _createEnhancedRandomTensor(shape, distribution, params, seed) {
    try {
      const { enhanced_random } = await import('./random-integration.js');
      const totalElements = shape.reduce((a, b) => a * b, 1);
      const data = new Float32Array(totalElements);

      // Set seed if provided
      if (seed !== null) {
        // This would set the seed in the enhanced random system
        // For now, we'll use Math.random as fallback
      }

      switch (distribution) {
        case 'normal': {
          const { mean = 0, std = 1 } = params;
          for (let i = 0; i < totalElements; i += 2) {
            // Use Box-Muller transform for better quality
            const u1 = Math.random();
            const u2 = Math.random();
            const mag = std * Math.sqrt(-2 * Math.log(u1));
            data[i] = mean + mag * Math.cos(2 * Math.PI * u2);
            if (i + 1 < totalElements) {
              data[i + 1] = mean + mag * Math.sin(2 * Math.PI * u2);
            }
          }
          break;
        }

        case 'uniform': {
          const { low = 0, high = 1 } = params;
          for (let i = 0; i < totalElements; i++) {
            data[i] = low + (high - low) * Math.random();
          }
          break;
        }

        case 'exponential': {
          const { rate = 1 } = params;
          for (let i = 0; i < totalElements; i++) {
            data[i] = -Math.log(Math.random()) / rate;
          }
          break;
        }

        case 'chi_squared': {
          const { df = 1 } = params;
          for (let i = 0; i < totalElements; i++) {
            let sum = 0;
            for (let j = 0; j < df; j++) {
              const z = this._randomNormal();
              sum += z * z;
            }
            data[i] = sum;
          }
          break;
        }

        case 'gamma': {
          const { shape: k = 1, scale = 1 } = params;
          for (let i = 0; i < totalElements; i++) {
            data[i] = this._randomGamma(k, scale);
          }
          break;
        }

        case 'beta': {
          const { alpha = 1, beta = 1 } = params;
          for (let i = 0; i < totalElements; i++) {
            data[i] = this._randomBeta(alpha, beta);
          }
          break;
        }

        case 'student_t': {
          const { df = 1 } = params;
          for (let i = 0; i < totalElements; i++) {
            const z = this._randomNormal();
            const chi2 = this._randomChiSquared(df);
            data[i] = z / Math.sqrt(chi2 / df);
          }
          break;
        }

        default:
          // Enhanced normal as fallback
          for (let i = 0; i < totalElements; i += 2) {
            const [z1, z2] = this._boxMuller();
            data[i] = z1;
            if (i + 1 < totalElements) {
              data[i + 1] = z2;
            }
          }
      }

      const { getRawModule } = await import('./index.js');
      const wasmModule = getRawModule();
      const wasm_tensor = wasmModule.WasmTensor.from_array(data, new Uint32Array(shape));

      // Add enhanced metadata
      wasm_tensor._enhanced_random = true;
      wasm_tensor._distribution = distribution;
      wasm_tensor._params = params;
      wasm_tensor._quality_level = 'high';
      wasm_tensor._generation_timestamp = Date.now();

      return wasm_tensor;
    } catch (error) {
      throw new Error(`Enhanced random tensor creation failed: ${error.message}`);
    }
  },

  /**
   * Box-Muller transform for normal distribution
   * @private
   */
  _boxMuller() {
    const u1 = Math.random();
    const u2 = Math.random();
    const mag = Math.sqrt(-2 * Math.log(u1));
    return [
      mag * Math.cos(2 * Math.PI * u2),
      mag * Math.sin(2 * Math.PI * u2)
    ];
  },

  /**
   * Generate random normal value
   * @private
   */
  _randomNormal() {
    return this._boxMuller()[0];
  },

  /**
   * Generate random gamma value using Marsaglia and Tsang's method
   * @private
   */
  _randomGamma(shape, scale) {
    if (shape < 1) {
      // Use transformation for shape < 1
      return this._randomGamma(shape + 1, scale) * Math.pow(Math.random(), 1 / shape);
    }

    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);

    while (true) {
      let x; let v;
      do {
        x = this._randomNormal();
        v = 1 + c * x;
      } while (v <= 0);

      v = v * v * v;
      const u = Math.random();

      if (u < 1 - 0.0331 * x * x * x * x) {
        return d * v * scale;
      }

      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
        return d * v * scale;
      }
    }
  },

  /**
   * Generate random beta value
   * @private
   */
  _randomBeta(alpha, beta) {
    const x = this._randomGamma(alpha, 1);
    const y = this._randomGamma(beta, 1);
    return x / (x + y);
  },

  /**
   * Generate random chi-squared value
   * @private
   */
  _randomChiSquared(df) {
    return this._randomGamma(df / 2, 2);
  },

  /**
   * Fallback random tensor generation
   * @private
   */
  async _fallbackRandom(shape, distribution, params) {
    const { getRawModule } = await import('./index.js');
    const wasmModule = getRawModule();

    const distributionMap = {
      'normal': () => wasmModule.WasmTensor.normal(new Uint32Array(shape), params.mean || 0, params.std || 1),
      'uniform': () => wasmModule.WasmTensor.uniform(new Uint32Array(shape), params.low || 0, params.high || 1),
      'binomial': () => wasmModule.WasmTensor.binomial(new Uint32Array(shape), params.n || 1, params.p || 0.5)
    };

    const generator = distributionMap[distribution];
    if (!generator) {
      console.warn(`Unknown distribution: ${distribution}, using normal distribution`);
      return wasmModule.WasmTensor.normal(new Uint32Array(shape), 0, 1);
    }

    return generator();
  },

  /**
   * Advanced Bayesian tensor creation
   * @param {number[]} shape - Tensor shape
   * @param {Object} prior_config - Prior distribution configuration
   * @param {Object} options - Creation options
   * @returns {Promise<Object>} Bayesian tensor
   */
  async create_bayesian(shape, prior_config, options = {}) {
    const { create_bayesian_tensor } = await import('./index.js');
    return create_bayesian_tensor(shape, prior_config, options.seed);
  },

  /**
   * Statistical analysis of tensor
   * @param {Object} tensor - Tensor to analyze
   * @param {Object} options - Analysis options
   * @returns {Promise<Object>} Statistical analysis results
   */
  async analyze_statistics(tensor, options = {}) {
    const { analyze_tensor } = await import('./index.js');
    return analyze_tensor(tensor, options.seed);
  },

  /**
   * Create tensor with mixture of distributions
   * @param {number[]} shape - Tensor shape
   * @param {Array} components - Array of distribution components
   * @param {Array} weights - Mixture weights
   * @param {Object} options - Options
   * @returns {Promise<Object>} Mixture tensor
   */
  async create_mixture(shape, components, weights, options = {}) {
    // Validate mixture components
    if (components.length !== weights.length) {
      throw new Error('Number of components must match number of weights');
    }

    if (Math.abs(weights.reduce((a, b) => a + b, 0) - 1.0) > 1e-6) {
      throw new Error('Weights must sum to 1.0');
    }

    const mixture_config = {
      type: 'mixture_of_gaussians',
      components,
      weights
    };

    const { create_advanced_tensor } = await import('./index.js');
    return create_advanced_tensor(shape, mixture_config, options.seed);
  },

  /**
   * Advanced tensor reshaping with validation
   * @param {Object} tensor - Input tensor
   * @param {Array<number>} newShape - New shape
   * @param {Object} options - Reshape options
   * @returns {Object} Reshaped tensor
   */
  reshape(tensor, newShape, options = {}) {
    const { validate = true, preserveType = true, allowPartial = false } = options;

    if (validate) {
      const currentSize = this._getTensorSize(tensor);
      const newSize = newShape.reduce((a, b) => a * b, 1);

      if (!allowPartial && currentSize !== newSize) {
        throw new Error(`Cannot reshape tensor: size mismatch (${currentSize} vs ${newSize})`);
      }
    }

    // Handle different tensor formats
    if (tensor.reshape && typeof tensor.reshape === 'function') {
      return tensor.reshape(newShape);
    }

    // Fallback manual reshape
    return {
      data: tensor.data,
      shape: new Uint32Array(newShape),
      dtype: preserveType ? (tensor.dtype || 'f32') : 'f32',
      _reshaped: true,
      _original_shape: tensor.shape
    };
  },

  /**
   * Get tensor size
   * @private
   */
  _getTensorSize(tensor) {
    if (tensor.shape) {
      if (Array.isArray(tensor.shape)) {
        return tensor.shape.reduce((a, b) => a * b, 1);
      } else if (tensor.shape.reduce) {
        return Array.from(tensor.shape).reduce((a, b) => a * b, 1);
      }
    }

    if (tensor.data && tensor.data.length) {
      return tensor.data.length;
    }

    return 0;
  },

  /**
   * Enhanced tensor slicing with advanced indexing
   * @param {Object} tensor - Input tensor
   * @param {Array} indices - Slice indices (can be multi-dimensional)
   * @param {Object} options - Slicing options
   * @returns {Object} Sliced tensor
   */
  slice(tensor, indices, options = {}) {
    const { step = 1, validate = true, copyData = true } = options;

    // Handle different indexing formats
    if (!Array.isArray(indices[0])) {
      // Simple 1D slicing
      return this._slice1D(tensor, indices, { step, validate, copyData });
    } 
      // Multi-dimensional slicing
      return this._sliceMultiD(tensor, indices, { step, validate, copyData });
    
  },

  /**
   * 1D tensor slicing
   * @private
   */
  _slice1D(tensor, indices, options) {
    const { step, validate, copyData } = options;
    const [start, end] = indices;

    if (validate) {
      const maxLength = this._getTensorSize(tensor);
      if (start < 0 || end > maxLength || start >= end) {
        throw new Error(`Invalid slice indices: [${start}, ${end}] for tensor of size ${maxLength}`);
      }
    }

    const sliceLength = Math.ceil((end - start) / step);
    const slicedData = copyData ? new Float32Array(sliceLength) : tensor.data.subarray(start, end);

    if (copyData && step === 1) {
      slicedData.set(tensor.data.subarray(start, end));
    } else if (copyData) {
      // Handle step > 1
      for (let i = 0; i < sliceLength; i++) {
        slicedData[i] = tensor.data[start + i * step];
      }
    }

    return {
      data: slicedData,
      shape: new Uint32Array([sliceLength]),
      dtype: tensor.dtype || 'f32',
      _sliced: true,
      _original_shape: tensor.shape,
      _slice_info: { start, end, step }
    };
  },

  /**
   * Multi-dimensional tensor slicing
   * @private
   */
  _sliceMultiD(tensor, indices, options) {
    // This is a simplified implementation
    // A full implementation would handle complex multi-dimensional indexing
    console.warn('Multi-dimensional slicing not fully implemented, falling back to 1D slice');
    return this._slice1D(tensor, indices[0], options);
  },

  /**
   * Concatenate tensors along specified dimension
   * @param {Array} tensors - Array of tensors to concatenate
   * @param {number} dim - Dimension along which to concatenate
   * @param {Object} options - Concatenation options
   * @returns {Object} Concatenated tensor
   */
  concat(tensors, dim = 0, options = {}) {
    const { validate = true, dtype = null } = options;

    if (!Array.isArray(tensors) || tensors.length === 0) {
      throw new Error('At least one tensor is required for concatenation');
    }

    if (validate) {
      this._validateConcatenation(tensors, dim);
    }

    // Calculate output shape
    const outputShape = [...tensors[0].shape];
    for (let i = 1; i < tensors.length; i++) {
      outputShape[dim] += tensors[i].shape[dim];
    }

    // Calculate total elements
    const totalElements = outputShape.reduce((a, b) => a * b, 1);
    const outputData = new Float32Array(totalElements);

    // Concatenate data
    this._concatenateData(tensors, outputData, outputShape, dim);

    return {
      data: outputData,
      shape: new Uint32Array(outputShape),
      dtype: dtype || tensors[0].dtype || 'f32',
      _concatenated: true,
      _source_tensors: tensors.length
    };
  },

  /**
   * Validate tensors for concatenation
   * @private
   */
  _validateConcatenation(tensors, dim) {
    const firstShape = tensors[0].shape;

    for (let i = 1; i < tensors.length; i++) {
      const currentShape = tensors[i].shape;

      if (firstShape.length !== currentShape.length) {
        throw new Error(`Shape mismatch: tensor 0 has ${firstShape.length} dimensions, tensor ${i} has ${currentShape.length}`);
      }

      for (let j = 0; j < firstShape.length; j++) {
        if (j !== dim && firstShape[j] !== currentShape[j]) {
          throw new Error(`Shape mismatch at dimension ${j}: ${firstShape[j]} vs ${currentShape[j]}`);
        }
      }
    }
  },

  /**
   * Concatenate tensor data
   * @private
   */
  _concatenateData(tensors, outputData, outputShape, dim) {
    // This is a simplified implementation for 1D and 2D cases
    // A full implementation would handle arbitrary dimensions

    if (outputShape.length === 1) {
      let offset = 0;
      for (const tensor of tensors) {
        outputData.set(tensor.data, offset);
        offset += tensor.data.length;
      }
    } else if (outputShape.length === 2 && dim === 0) {
      // Concatenate along first dimension (rows)
      let offset = 0;
      for (const tensor of tensors) {
        outputData.set(tensor.data, offset);
        offset += tensor.data.length;
      }
    } else {
      // General case - simplified
      console.warn('General multi-dimensional concatenation not fully implemented');
      let offset = 0;
      for (const tensor of tensors) {
        outputData.set(tensor.data, offset);
        offset += tensor.data.length;
      }
    }
  },

  /**
   * Split tensor into multiple tensors
   * @param {Object} tensor - Input tensor to split
   * @param {number|Array} split_size - Size(s) of splits
   * @param {number} dim - Dimension along which to split
   * @returns {Array} Array of split tensors
   */
  split(tensor, split_size, dim = 0) {
    const shape = Array.from(tensor.shape);

    if (typeof split_size === 'number') {
      // Equal splits
      const num_splits = Math.ceil(shape[dim] / split_size);
      const sizes = new Array(num_splits).fill(split_size);
      if (shape[dim] % split_size !== 0) {
        sizes[sizes.length - 1] = shape[dim] % split_size;
      }
      return this._splitBySizes(tensor, sizes, dim);
    } 
      // Splits by specified sizes
      return this._splitBySizes(tensor, split_size, dim);
    
  },

  /**
   * Split tensor by specified sizes
   * @private
   */
  _splitBySizes(tensor, sizes, dim) {
    const results = [];
    let currentOffset = 0;

    for (const size of sizes) {
      const sliceIndices = [currentOffset, currentOffset + size];
      const sliced = this.slice(tensor, sliceIndices, { validate: false });
      results.push(sliced);
      currentOffset += size;
    }

    return results;
  }
};

/**
 * Async utilities for batch processing
 */
export const async_utils = {
  /**
   * Process tensors in batches asynchronously
   * @param {Array} tensors - Array of tensors to process
   * @param {Function} processFunc - Processing function
   * @param {number} batchSize - Batch size
   * @param {Object} options - Processing options
   * @returns {Promise<Array>} Array of processed results
   */
  async processBatch(tensors, processFunc, batchSize = 32, options = {}) {
    const {
      parallel = true,
      maxConcurrency = 4,
      onProgress = null,
      onError = null
    } = options;

    const results = [];
    const totalBatches = Math.ceil(tensors.length / batchSize);

    if (parallel) {
      const semaphore = new Semaphore(maxConcurrency);
      const batchPromises = [];

      for (let i = 0; i < totalBatches; i++) {
        const batchStart = i * batchSize;
        const batchEnd = Math.min(batchStart + batchSize, tensors.length);
        const batch = tensors.slice(batchStart, batchEnd);

        const batchPromise = semaphore.acquire().then(async () => {
          try {
            const batchResults = await Promise.all(batch.map(processFunc));

            if (onProgress) {
              onProgress({
                completed: i + 1,
                total: totalBatches,
                batchIndex: i,
                batchResults
              });
            }

            return batchResults;
          } catch (error) {
            if (onError) {
              onError(error, { batchIndex: i, batch });
            } else {
              throw error;
            }
          } finally {
            semaphore.release();
          }
        });

        batchPromises.push(batchPromise);
      }

      const batchResults = await Promise.all(batchPromises);

      // Flatten results
      for (const batchResult of batchResults) {
        if (batchResult) {
          results.push(...batchResult);
        }
      }
    } else {
      // Sequential processing
      for (let i = 0; i < totalBatches; i++) {
        const batchStart = i * batchSize;
        const batchEnd = Math.min(batchStart + batchSize, tensors.length);
        const batch = tensors.slice(batchStart, batchEnd);

        try {
          const batchResults = await Promise.all(batch.map(processFunc));
          results.push(...batchResults);

          if (onProgress) {
            onProgress({
              completed: i + 1,
              total: totalBatches,
              batchIndex: i,
              batchResults
            });
          }

          // Allow other tasks to run
          await new Promise(resolve => setTimeout(resolve, 0));
        } catch (error) {
          if (onError) {
            onError(error, { batchIndex: i, batch });
          } else {
            throw error;
          }
        }
      }
    }

    return results;
  },

  /**
   * Run inference with automatic memory management
   * @param {Object} model - Model object
   * @param {Object} inputs - Input tensors
   * @param {Object} options - Options including cleanup
   * @returns {Promise<Object>} Inference results
   */
  async runInference(model, inputs, options = {}) {
    const { autoCleanup = true, timeout = 30000 } = options;

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
        // Clean up input tensors
        this._cleanupInputs(inputs);
      }
    }
  },

  /**
   * Clean up input tensors
   * @private
   */
  _cleanupInputs(inputs) {
    const cleanup = (tensor) => {
      if (tensor && typeof tensor === 'object') {
        if (tensor.free) {
          tensor.free();
        } else if (tensor.dispose) {
          tensor.dispose();
        } else if (tensor.delete) {
          tensor.delete();
        }
      }
    };

    if (Array.isArray(inputs)) {
      inputs.forEach(cleanup);
    } else {
      cleanup(inputs);
    }
  },

  /**
   * Process stream of data asynchronously
   * @param {AsyncIterable} stream - Input stream
   * @param {Function} processFunc - Processing function
   * @param {Object} options - Streaming options
   * @returns {AsyncGenerator} Processed stream
   */
  async* processStream(stream, processFunc, options = {}) {
    const {
      bufferSize = 10,
      parallel = true,
      onError = null
    } = options;

    const buffer = [];

    try {
      for await (const item of stream) {
        buffer.push(item);

        if (buffer.length >= bufferSize) {
          const batch = buffer.splice(0, bufferSize);

          try {
            if (parallel) {
              const results = await Promise.all(batch.map(processFunc));
              for (const result of results) {
                yield result;
              }
            } else {
              for (const item of batch) {
                const result = await processFunc(item);
                yield result;
              }
            }
          } catch (error) {
            if (onError) {
              onError(error, { batch });
            } else {
              throw error;
            }
          }
        }
      }

      // Process remaining items
      if (buffer.length > 0) {
        try {
          if (parallel) {
            const results = await Promise.all(buffer.map(processFunc));
            for (const result of results) {
              yield result;
            }
          } else {
            for (const item of buffer) {
              const result = await processFunc(item);
              yield result;
            }
          }
        } catch (error) {
          if (onError) {
            onError(error, { batch: buffer });
          } else {
            throw error;
          }
        }
      }
    } finally {
      // Cleanup buffer if needed
      buffer.length = 0;
    }
  }
};

/**
 * Simple semaphore for concurrency control
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

export default {
  tensor_utils,
  async_utils
};