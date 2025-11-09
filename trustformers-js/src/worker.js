/**
 * TrustformeRS Web Worker
 * Provides background processing capabilities for heavy ML operations
 */

import * as TrustformeRS from './index.js';

// Worker state
let initialized = false;
const models = new Map();
const tensors = new Map();

/**
 * Initialize the worker with TrustformeRS
 */
async function initialize(config = {}) {
  if (initialized) return { success: true };
  
  try {
    await TrustformeRS.init(config);
    initialized = true;
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Load a model in the worker
 */
async function loadModel(modelId, modelPath, config = {}) {
  try {
    if (!initialized) {
      throw new Error('Worker not initialized. Call initialize() first.');
    }
    
    const model = await TrustformeRS.loadModel(modelPath, config);
    models.set(modelId, model);
    
    return { 
      success: true, 
      modelId,
      info: {
        name: model.name || 'Unknown',
        type: model.type || 'Unknown',
        parameters: model.parameters || 0
      }
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Run inference on a model
 */
async function runInference(modelId, input, options = {}) {
  try {
    const model = models.get(modelId);
    if (!model) {
      throw new Error(`Model ${modelId} not found. Load the model first.`);
    }
    
    const result = await model.forward(input, options);
    
    // Convert tensors to transferable objects if needed
    const transferableResult = serializeForTransfer(result);
    
    return { success: true, result: transferableResult };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Create a tensor in the worker
 */
async function createTensor(tensorId, data, shape, dtype = 'f32') {
  try {
    const tensor = TrustformeRS.tensor(data, shape, dtype);
    tensors.set(tensorId, tensor);
    
    return { 
      success: true, 
      tensorId,
      info: {
        shape: tensor.shape,
        dtype: tensor.dtype,
        size: tensor.size
      }
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Perform tensor operations
 */
async function tensorOperation(operation, tensorIds, ...args) {
  try {
    const inputTensors = tensorIds.map(id => {
      const tensor = tensors.get(id);
      if (!tensor) throw new Error(`Tensor ${id} not found`);
      return tensor;
    });
    
    let result;
    switch (operation) {
      case 'add':
        result = TrustformeRS.add(inputTensors[0], inputTensors[1]);
        break;
      case 'multiply':
        result = TrustformeRS.multiply(inputTensors[0], inputTensors[1]);
        break;
      case 'matmul':
        result = TrustformeRS.matmul(inputTensors[0], inputTensors[1]);
        break;
      case 'reshape':
        result = TrustformeRS.reshape(inputTensors[0], args[0]);
        break;
      case 'slice':
        result = TrustformeRS.slice(inputTensors[0], ...args);
        break;
      default:
        throw new Error(`Unknown operation: ${operation}`);
    }
    
    const resultId = `result_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    tensors.set(resultId, result);
    
    return { 
      success: true, 
      resultId,
      info: {
        shape: result.shape,
        dtype: result.dtype,
        size: result.size
      }
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Get tensor data for transfer back to main thread
 */
async function getTensorData(tensorId) {
  try {
    const tensor = tensors.get(tensorId);
    if (!tensor) throw new Error(`Tensor ${tensorId} not found`);
    
    const data = await tensor.data();
    return { 
      success: true, 
      data: Array.from(data), // Convert to regular array for transfer
      shape: tensor.shape,
      dtype: tensor.dtype
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Clean up resources
 */
async function cleanup(resourceType, resourceId) {
  try {
    switch (resourceType) {
      case 'model':
        if (models.has(resourceId)) {
          const model = models.get(resourceId);
          if (model.dispose) await model.dispose();
          models.delete(resourceId);
        }
        break;
      case 'tensor':
        if (tensors.has(resourceId)) {
          const tensor = tensors.get(resourceId);
          if (tensor.dispose) await tensor.dispose();
          tensors.delete(resourceId);
        }
        break;
      case 'all':
        // Clean up all resources
        for (const [id, model] of models) {
          if (model.dispose) await model.dispose();
        }
        for (const [id, tensor] of tensors) {
          if (tensor.dispose) await tensor.dispose();
        }
        models.clear();
        tensors.clear();
        break;
    }
    
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Get worker status and memory usage
 */
async function getStatus() {
  return {
    success: true,
    status: {
      initialized,
      models: models.size,
      tensors: tensors.size,
      memory: performance.memory ? {
        used: performance.memory.usedJSHeapSize,
        total: performance.memory.totalJSHeapSize,
        limit: performance.memory.jsHeapSizeLimit
      } : null
    }
  };
}

/**
 * Serialize complex objects for transfer
 */
function serializeForTransfer(obj) {
  if (obj && typeof obj === 'object') {
    if (obj.constructor && obj.constructor.name === 'Tensor') {
      return {
        type: 'tensor',
        shape: obj.shape,
        dtype: obj.dtype,
        data: Array.from(obj.data ? obj.data() : [])
      };
    }
    
    if (Array.isArray(obj)) {
      return obj.map(serializeForTransfer);
    }
    
    const serialized = {};
    for (const [key, value] of Object.entries(obj)) {
      serialized[key] = serializeForTransfer(value);
    }
    return serialized;
  }
  
  return obj;
}

// Message handler
self.onmessage = async function(event) {
  const { id, command, ...params } = event.data;
  
  let result;
  try {
    switch (command) {
      case 'initialize':
        result = await initialize(params.config);
        break;
      case 'loadModel':
        result = await loadModel(params.modelId, params.modelPath, params.config);
        break;
      case 'runInference':
        result = await runInference(params.modelId, params.input, params.options);
        break;
      case 'createTensor':
        result = await createTensor(params.tensorId, params.data, params.shape, params.dtype);
        break;
      case 'tensorOperation':
        result = await tensorOperation(params.operation, params.tensorIds, ...params.args);
        break;
      case 'getTensorData':
        result = await getTensorData(params.tensorId);
        break;
      case 'cleanup':
        result = await cleanup(params.resourceType, params.resourceId);
        break;
      case 'getStatus':
        result = await getStatus();
        break;
      default:
        result = { success: false, error: `Unknown command: ${command}` };
    }
  } catch (error) {
    result = { success: false, error: error.message };
  }
  
  // Send response back to main thread
  self.postMessage({ id, ...result });
};

// Handle worker errors
self.onerror = function(error) {
  console.error('TrustformeRS Worker Error:', error);
  self.postMessage({ 
    type: 'error', 
    error: error.message || 'Unknown worker error' 
  });
};

// Handle unhandled promise rejections
self.onunhandledrejection = function(event) {
  console.error('TrustformeRS Worker Unhandled Promise Rejection:', event.reason);
  self.postMessage({ 
    type: 'error', 
    error: event.reason?.message || 'Unhandled promise rejection' 
  });
};

// Signal that worker is ready
self.postMessage({ type: 'ready' });