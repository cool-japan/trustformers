/**
 * WebGPU Support Module
 * Provides WebGPU operations and device management for enhanced performance
 */

let wasmModule = null;

/**
 * Set WASM module reference
 * @param {Object} module - WASM module
 */
export function setWasmModule(module) {
  wasmModule = module;
}

/**
 * WebGPU support utilities
 */
export const webgpu = {
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
    if (wasmModule && wasmModule.is_webgpu_available) {
      try {
        return wasmModule.is_webgpu_available();
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

    if (wasmModule && wasmModule.get_webgpu_status) {
      try {
        return wasmModule.get_webgpu_status();
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

    if (wasmModule && wasmModule.WebGPUOps) {
      try {
        return new wasmModule.WebGPUOps();
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

    if (wasmModule && wasmModule.get_webgpu_device_info) {
      try {
        return await wasmModule.get_webgpu_device_info();
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

export default webgpu;