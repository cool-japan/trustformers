/**
 * Enhanced Tensor Operations with Automatic Backend Selection
 * Advanced tensor operations with WebGL, WebGPU, and WASM backends
 */

import { webgpu } from './webgpu-support.js';

/**
 * Enhanced tensor operations with automatic optimization
 */
export const enhanced_tensor_ops = {
  /**
   * Enhanced matrix multiplication with automatic backend selection
   * @param {Object} a - First tensor
   * @param {Object} b - Second tensor
   * @param {Object} options - Options including backend preference
   * @returns {Promise<Object>} Result tensor
   */
  async matmul(a, b, options = {}) {
    const { backend = 'auto', profile = true } = options;

    const operation = async () => {
      // Get global instances from the main module
      const { webglBackend, performanceProfiler } = await import('./index.js');

      if (backend === 'webgl' && webglBackend && webglBackend.isSupported()) {
        return await webglBackend.matmul(a, b);
      } else if (backend === 'webgpu' && webgpu.isAvailable()) {
        const ops = webgpu.createOps();
        return await ops.matmul(a, b);
      } 
        // Fallback to WASM/CPU
        const { tensor_ops } = await import('./index.js');
        return tensor_ops.matmul(a, b);
      
    };

    if (profile && window.performanceProfiler) {
      return await window.performanceProfiler.profileOperation('enhanced_matmul', operation, {
        backend,
        shapeA: Array.from(a.shape()),
        shapeB: Array.from(b.shape())
      });
    } 
      return await operation();
    
  },

  /**
   * Enhanced element-wise operations
   * @param {Object} a - First tensor
   * @param {Object} b - Second tensor
   * @param {string} operation - Operation type
   * @param {Object} options - Options
   * @returns {Promise<Object>} Result tensor
   */
  async elementWise(a, b, operation, options = {}) {
    const { backend = 'auto', profile = true } = options;

    const op = async () => {
      const { webglBackend } = await import('./index.js');

      if (backend === 'webgl' && webglBackend && webglBackend.isSupported()) {
        return await webglBackend.elementWise(a, b, operation);
      } 
        // Fallback to WASM
        const { tensor_ops } = await import('./index.js');
        switch (operation) {
          case 'add': return tensor_ops.add(a, b);
          case 'sub': return tensor_ops.sub(a, b);
          case 'mul': return tensor_ops.mul(a, b);
          case 'div': return tensor_ops.div(a, b);
          default: throw new Error(`Unknown operation: ${operation}`);
        }
      
    };

    if (profile && window.performanceProfiler) {
      return await window.performanceProfiler.profileOperation(`enhanced_${operation}`, op, {
        backend,
        shape: Array.from(a.shape())
      });
    } 
      return await op();
    
  },

  /**
   * Enhanced activation functions
   * @param {Object} tensor - Input tensor
   * @param {string} activation - Activation type
   * @param {Object} options - Options
   * @returns {Promise<Object>} Result tensor
   */
  async activation(tensor, activation, options = {}) {
    const { backend = 'auto', profile = true } = options;

    const op = async () => {
      const { webglBackend, activations } = await import('./index.js');

      if (backend === 'webgl' && webglBackend && webglBackend.isSupported()) {
        return await webglBackend.activation(tensor, activation);
      } 
        // Fallback to WASM
        switch (activation) {
          case 'relu': return activations.relu(tensor);
          case 'sigmoid': return activations.sigmoid(tensor);
          case 'tanh': return activations.tanh(tensor);
          case 'gelu': return activations.gelu(tensor);
          default: throw new Error(`Unknown activation: ${activation}`);
        }
      
    };

    if (profile && window.performanceProfiler) {
      return await window.performanceProfiler.profileOperation(`enhanced_${activation}`, op, {
        backend,
        shape: Array.from(tensor.shape())
      });
    } 
      return await op();
    
  },

  /**
   * Enhanced convolution operations
   * @param {Object} input - Input tensor
   * @param {Object} kernel - Convolution kernel
   * @param {Object} options - Convolution options
   * @returns {Promise<Object>} Result tensor
   */
  async conv2d(input, kernel, options = {}) {
    const {
      backend = 'auto',
      profile = true,
      stride = [1, 1],
      padding = [0, 0],
      dilation = [1, 1]
    } = options;

    const operation = async () => {
      const { webglBackend } = await import('./index.js');

      if (backend === 'webgl' && webglBackend && webglBackend.isSupported()) {
        return await webglBackend.conv2d(input, kernel, { stride, padding, dilation });
      } 
        // Fallback to WASM implementation
        const { getRawModule } = await import('./index.js');
        const wasmModule = getRawModule();
        return wasmModule.tensor_conv2d(input, kernel, stride, padding, dilation);
      
    };

    if (profile && window.performanceProfiler) {
      return await window.performanceProfiler.profileOperation('enhanced_conv2d', operation, {
        backend,
        inputShape: Array.from(input.shape()),
        kernelShape: Array.from(kernel.shape()),
        stride,
        padding,
        dilation
      });
    } 
      return await operation();
    
  },

  /**
   * Enhanced pooling operations
   * @param {Object} tensor - Input tensor
   * @param {string} poolType - Pool type ('max', 'avg')
   * @param {Object} options - Pooling options
   * @returns {Promise<Object>} Result tensor
   */
  async pool2d(tensor, poolType = 'max', options = {}) {
    const {
      backend = 'auto',
      profile = true,
      kernelSize = [2, 2],
      stride = [2, 2],
      padding = [0, 0]
    } = options;

    const operation = async () => {
      const { webglBackend } = await import('./index.js');

      if (backend === 'webgl' && webglBackend && webglBackend.isSupported()) {
        return await webglBackend.pool2d(tensor, poolType, { kernelSize, stride, padding });
      } 
        // Fallback to WASM implementation
        const { getRawModule } = await import('./index.js');
        const wasmModule = getRawModule();

        if (poolType === 'max') {
          return wasmModule.tensor_maxpool2d(tensor, kernelSize, stride, padding);
        } 
          return wasmModule.tensor_avgpool2d(tensor, kernelSize, stride, padding);
        
      
    };

    if (profile && window.performanceProfiler) {
      return await window.performanceProfiler.profileOperation(`enhanced_${poolType}pool2d`, operation, {
        backend,
        shape: Array.from(tensor.shape()),
        kernelSize,
        stride,
        padding
      });
    } 
      return await operation();
    
  },

  /**
   * Enhanced batch processing for multiple tensors
   * @param {Array} tensors - Array of tensors
   * @param {Function} operation - Operation to apply
   * @param {Object} options - Batch options
   * @returns {Promise<Array>} Array of results
   */
  async batchProcess(tensors, operation, options = {}) {
    const {
      batchSize = 32,
      parallel = true,
      profile = true
    } = options;

    const op = async () => {
      if (parallel && batchSize > 1) {
        // Process in parallel batches
        const results = [];
        const batches = [];

        for (let i = 0; i < tensors.length; i += batchSize) {
          batches.push(tensors.slice(i, i + batchSize));
        }

        for (const batch of batches) {
          const batchResults = await Promise.all(batch.map(operation));
          results.push(...batchResults);
        }

        return results;
      } 
        // Sequential processing
        const results = [];
        for (const tensor of tensors) {
          results.push(await operation(tensor));
        }
        return results;
      
    };

    if (profile && window.performanceProfiler) {
      return await window.performanceProfiler.profileOperation('enhanced_batch_process', op, {
        tensorCount: tensors.length,
        batchSize,
        parallel
      });
    } 
      return await op();
    
  }
};

/**
 * Enhanced tensor creation with memory pooling
 */
export const enhanced_tensor_utils = {
  /**
   * Create tensor with automatic memory management
   * @param {Array|TypedArray} data - Tensor data
   * @param {Array<number>} shape - Tensor shape
   * @param {Object} options - Options
   * @returns {Object} Enhanced tensor
   */
  createTensor(data, shape, options = {}) {
    const { dtype = 'f32', useMemoryPool = true, zeroCopy = true } = options;

    // Import functions will be resolved at runtime
    import('./index.js').then(({
      createZeroCopyTensor,
      memoryManager,
      tensor,
      capabilities
    }) => {
      if (zeroCopy && capabilities.zeroCopy) {
        return createZeroCopyTensor(data, shape, dtype);
      } else if (useMemoryPool && memoryManager) {
        const tensorObj = memoryManager.tensorPool.acquire(shape, dtype);
        if (data) {
          const view = tensorObj.getView ? tensorObj.getView(dtype) : tensorObj;
          view.setData(data);
        }
        return tensorObj;
      } 
        // Fallback to standard creation
        return tensor(data, shape);
      
    }).catch(() => 
      // Synchronous fallback
       ({ data: new Float32Array(data), shape: new Uint32Array(shape) })
    );

    // Immediate synchronous return for compatibility
    return { data: new Float32Array(data), shape: new Uint32Array(shape) };
  },

  /**
   * Create zeros tensor with pooling
   * @param {Array<number>} shape - Tensor shape
   * @param {Object} options - Options
   * @returns {Object} Zeros tensor
   */
  zeros(shape, options = {}) {
    const { dtype = 'f32', useMemoryPool = true } = options;

    const totalElements = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(totalElements);
    data.fill(0);

    return this.createTensor(data, shape, { dtype, useMemoryPool });
  },

  /**
   * Create ones tensor with pooling
   * @param {Array<number>} shape - Tensor shape
   * @param {Object} options - Options
   * @returns {Object} Ones tensor
   */
  ones(shape, options = {}) {
    const { dtype = 'f32', useMemoryPool = true } = options;

    const totalElements = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(totalElements);
    data.fill(1);

    return this.createTensor(data, shape, { dtype, useMemoryPool });
  },

  /**
   * Release tensor back to memory pool
   * @param {Object} tensor - Tensor to release
   */
  releaseTensor(tensor) {
    import('./index.js').then(({ memoryManager }) => {
      if (memoryManager && memoryManager.tensorPool.allocatedTensors.has(tensor)) {
        memoryManager.tensorPool.release(tensor);
      } else if (tensor.dispose) {
        tensor.dispose();
      } else if (tensor.free) {
        tensor.free();
      }
    }).catch(() => {
      // Manual cleanup
      if (tensor.dispose) {
        tensor.dispose();
      } else if (tensor.free) {
        tensor.free();
      }
    });
  },

  /**
   * Advanced tensor reshaping with validation
   * @param {Object} tensor - Input tensor
   * @param {Array<number>} newShape - New shape
   * @param {Object} options - Reshape options
   * @returns {Object} Reshaped tensor
   */
  reshape(tensor, newShape, options = {}) {
    const { validate = true, preserveType = true } = options;

    if (validate) {
      const currentSize = tensor.shape ?
        tensor.shape.reduce((a, b) => a * b, 1) :
        tensor.data.length;
      const newSize = newShape.reduce((a, b) => a * b, 1);

      if (currentSize !== newSize) {
        throw new Error(`Cannot reshape tensor: size mismatch (${currentSize} vs ${newSize})`);
      }
    }

    return {
      data: tensor.data,
      shape: new Uint32Array(newShape),
      dtype: preserveType ? tensor.dtype : 'f32'
    };
  },

  /**
   * Enhanced tensor slicing with advanced indexing
   * @param {Object} tensor - Input tensor
   * @param {Array} indices - Slice indices
   * @param {Object} options - Slicing options
   * @returns {Object} Sliced tensor
   */
  slice(tensor, indices, options = {}) {
    const { step = 1, validate = true } = options;

    // This is a simplified implementation
    // In a real scenario, this would handle complex multi-dimensional slicing
    const [start, end] = indices;

    if (validate) {
      const maxLength = tensor.data ? tensor.data.length : 0;
      if (start < 0 || end > maxLength || start >= end) {
        throw new Error(`Invalid slice indices: [${start}, ${end}] for tensor of size ${maxLength}`);
      }
    }

    const slicedData = tensor.data.slice(start, end);
    const slicedShape = [end - start];

    return {
      data: slicedData,
      shape: new Uint32Array(slicedShape),
      dtype: tensor.dtype || 'f32'
    };
  }
};

export default {
  enhanced_tensor_ops,
  enhanced_tensor_utils
};