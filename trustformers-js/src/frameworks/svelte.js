/**
 * Svelte integration utilities for TrustformeRS
 * Provides Svelte-specific integrations using stores and reactive statements
 */

import { writable, readable, derived, get } from 'svelte/store';
import { onDestroy } from 'svelte';
import {
  initialize,
  initializeEnhanced,
  createModel,
  createTokenizer,
  Pipeline,
  tensor,
  enhanced_inference,
  performance,
  devTools,
  memory,
} from '../index.js';

/**
 * TrustformeRS Initialization Store
 * Manages the initialization state of TrustformeRS
 * @param {Object} options - Initialization options
 * @returns {Object} Svelte stores for initialization state
 */
export function createTrustformersStore(options = {}) {
  const isInitialized = writable(false);
  const isLoading = writable(false);
  const error = writable(null);
  const capabilities = writable(null);

  let initPromise = null;

  const initialize = async (newOptions = {}) => {
    if (initPromise) return initPromise;

    isLoading.set(true);
    error.set(null);

    try {
      initPromise = initializeEnhanced({
        enableWebGL: true,
        enableMemoryPool: true,
        enableProfiling: true,
        enableZeroCopy: true,
        ...options,
        ...newOptions,
      });

      const caps = await initPromise;
      capabilities.set(caps);
      isInitialized.set(true);
      return caps;
    } catch (err) {
      error.set(err);
      initPromise = null;
      throw err;
    } finally {
      isLoading.set(false);
    }
  };

  const reinitialize = async (newOptions = {}) => {
    initPromise = null;
    isInitialized.set(false);
    capabilities.set(null);
    return await initialize(newOptions);
  };

  // Auto-initialize
  initialize();

  return {
    isInitialized: { subscribe: isInitialized.subscribe },
    isLoading: { subscribe: isLoading.subscribe },
    error: { subscribe: error.subscribe },
    capabilities: { subscribe: capabilities.subscribe },
    initialize,
    reinitialize,
  };
}

/**
 * Model Store Factory
 * Creates a store for managing a specific model
 * @param {string|Object} modelConfig - Model configuration
 * @param {Object} options - Model options
 * @returns {Object} Svelte stores for model state
 */
export function createModelStore(modelConfig, options = {}) {
  const model = writable(null);
  const isLoading = writable(false);
  const error = writable(null);

  let modelInstance = null;

  const createModelInstance = async config => {
    if (!config) return;

    isLoading.set(true);
    error.set(null);

    try {
      // Cleanup previous model
      if (modelInstance && modelInstance.free) {
        modelInstance.free();
      }

      modelInstance = createModel(config);
      model.set(modelInstance);
    } catch (err) {
      error.set(err);
    } finally {
      isLoading.set(false);
    }
  };

  const runInference = async (inputs, inferenceOptions = {}) => {
    if (!modelInstance) throw new Error('Model not initialized');

    return await enhanced_inference.runInference(modelInstance, inputs, {
      profile: true,
      useMemoryPool: true,
      autoCleanup: true,
      ...inferenceOptions,
    });
  };

  const batchInference = async (batchInputs, batchOptions = {}) => {
    if (!modelInstance) throw new Error('Model not initialized');

    return await enhanced_inference.batchInference(modelInstance, batchInputs, {
      batchSize: 32,
      profile: true,
      ...batchOptions,
    });
  };

  const cleanup = () => {
    if (modelInstance && modelInstance.free) {
      modelInstance.free();
    }
    modelInstance = null;
    model.set(null);
  };

  // Initialize model
  if (modelConfig) {
    createModelInstance(modelConfig);
  }

  return {
    model: { subscribe: model.subscribe },
    isLoading: { subscribe: isLoading.subscribe },
    error: { subscribe: error.subscribe },
    runInference,
    batchInference,
    createModel: createModelInstance,
    cleanup,
  };
}

/**
 * Tokenizer Store Factory
 * Creates a store for managing a tokenizer
 * @param {string} type - Tokenizer type
 * @param {Object} vocab - Vocabulary object
 * @returns {Object} Svelte stores for tokenizer state
 */
export function createTokenizerStore(type, vocab = null) {
  const tokenizer = writable(null);
  const isLoading = writable(false);
  const error = writable(null);

  let tokenizerInstance = null;

  const createTokenizerInstance = async (tokType, vocabData) => {
    if (!tokType) return;

    isLoading.set(true);
    error.set(null);

    try {
      // Cleanup previous tokenizer
      if (tokenizerInstance && tokenizerInstance.free) {
        tokenizerInstance.free();
      }

      tokenizerInstance = createTokenizer(tokType, vocabData);
      tokenizer.set(tokenizerInstance);
    } catch (err) {
      error.set(err);
    } finally {
      isLoading.set(false);
    }
  };

  const encode = text => {
    if (!tokenizerInstance) throw new Error('Tokenizer not initialized');
    return tokenizerInstance.encode(text);
  };

  const decode = tokens => {
    if (!tokenizerInstance) throw new Error('Tokenizer not initialized');
    return tokenizerInstance.decode(tokens);
  };

  const batchEncode = texts => {
    if (!tokenizerInstance) throw new Error('Tokenizer not initialized');
    return texts.map(text => tokenizerInstance.encode(text));
  };

  const cleanup = () => {
    if (tokenizerInstance && tokenizerInstance.free) {
      tokenizerInstance.free();
    }
    tokenizerInstance = null;
    tokenizer.set(null);
  };

  // Initialize tokenizer
  if (type) {
    createTokenizerInstance(type, vocab);
  }

  return {
    tokenizer: { subscribe: tokenizer.subscribe },
    isLoading: { subscribe: isLoading.subscribe },
    error: { subscribe: error.subscribe },
    encode,
    decode,
    batchEncode,
    createTokenizer: createTokenizerInstance,
    cleanup,
  };
}

/**
 * Pipeline Store Factory
 * Creates a store for managing a pipeline
 * @param {string} task - Pipeline task type
 * @param {Object} modelStore - Model store
 * @param {Object} tokenizerStore - Tokenizer store
 * @param {Object} config - Pipeline configuration
 * @returns {Object} Svelte stores for pipeline state
 */
export function createPipelineStore(task, modelStore, tokenizerStore, config = {}) {
  const pipeline = writable(null);
  const isLoading = writable(false);
  const error = writable(null);

  let pipelineInstance = null;

  const createPipelineInstance = async (taskType, modelObj, tokenizerObj, pipelineConfig) => {
    if (!taskType || !modelObj || !tokenizerObj) return;

    isLoading.set(true);
    error.set(null);

    try {
      // Cleanup previous pipeline
      if (pipelineInstance && pipelineInstance.free) {
        pipelineInstance.free();
      }

      let newPipeline;
      switch (taskType) {
        case 'text-generation':
          newPipeline = Pipeline.textGeneration(modelObj, tokenizerObj, pipelineConfig);
          break;
        case 'text-classification':
          newPipeline = Pipeline.textClassification(
            modelObj,
            tokenizerObj,
            pipelineConfig.labels || []
          );
          break;
        case 'question-answering':
          newPipeline = Pipeline.questionAnswering(modelObj, tokenizerObj);
          break;
        default:
          newPipeline = await Pipeline.fromPretrained(
            taskType,
            pipelineConfig.modelName || 'default'
          );
      }

      pipelineInstance = newPipeline;
      pipeline.set(pipelineInstance);
    } catch (err) {
      error.set(err);
    } finally {
      isLoading.set(false);
    }
  };

  const run = async (input, options = {}) => {
    if (!pipelineInstance) throw new Error('Pipeline not initialized');

    return await pipelineInstance.run(input, {
      maxLength: 50,
      temperature: 0.7,
      ...options,
    });
  };

  const batch = async (inputs, options = {}) => {
    if (!pipelineInstance) throw new Error('Pipeline not initialized');

    return Promise.all(inputs.map(input => pipelineInstance.run(input, options)));
  };

  const cleanup = () => {
    if (pipelineInstance && pipelineInstance.free) {
      pipelineInstance.free();
    }
    pipelineInstance = null;
    pipeline.set(null);
  };

  // Create derived store that watches model and tokenizer
  const dependencies = derived(
    [modelStore.model, tokenizerStore.tokenizer],
    ([$model, $tokenizer]) => ({ model: $model, tokenizer: $tokenizer })
  );

  // Subscribe to dependencies and recreate pipeline when they change
  const unsubscribe = dependencies.subscribe(({ model: modelObj, tokenizer: tokenizerObj }) => {
    if (task && modelObj && tokenizerObj) {
      createPipelineInstance(task, modelObj, tokenizerObj, config);
    }
  });

  return {
    pipeline: { subscribe: pipeline.subscribe },
    isLoading: { subscribe: isLoading.subscribe },
    error: { subscribe: error.subscribe },
    run,
    batch,
    cleanup,
    unsubscribe,
  };
}

/**
 * Tensor Store Factory
 * Creates a store for managing a tensor
 * @param {Array} initialData - Initial tensor data
 * @param {Array} initialShape - Initial tensor shape
 * @returns {Object} Svelte stores for tensor state
 */
export function createTensorStore(initialData = null, initialShape = null) {
  const tensorState = writable({
    tensor: null,
    shape: null,
    size: null,
    dtype: null,
  });
  const error = writable(null);

  let tensorInstance = null;

  const createTensor = async (data, shape) => {
    try {
      // Cleanup previous tensor
      if (tensorInstance && tensorInstance.free) {
        tensorInstance.free();
      }

      tensorInstance = tensor(data, shape);

      tensorState.set({
        tensor: tensorInstance,
        shape: Array.from(tensorInstance.shape()),
        size: tensorInstance.size(),
        dtype: tensorInstance.dtype(),
      });
      error.set(null);
    } catch (err) {
      error.set(err);
    }
  };

  const operations = {
    reshape: newShape => {
      if (!tensorInstance) return null;
      try {
        return tensorInstance.reshape(newShape);
      } catch (err) {
        error.set(err);
        return null;
      }
    },

    transpose: (dims = null) => {
      if (!tensorInstance) return null;
      try {
        return dims ? tensorInstance.transpose(dims) : tensorInstance.transpose();
      } catch (err) {
        error.set(err);
        return null;
      }
    },

    slice: (start, end, step = 1) => {
      if (!tensorInstance) return null;
      try {
        return tensorInstance.slice(start, end, step);
      } catch (err) {
        error.set(err);
        return null;
      }
    },

    add: other => {
      if (!tensorInstance) return null;
      try {
        return tensorInstance.add(other);
      } catch (err) {
        error.set(err);
        return null;
      }
    },

    multiply: other => {
      if (!tensorInstance) return null;
      try {
        return tensorInstance.mul(other);
      } catch (err) {
        error.set(err);
        return null;
      }
    },

    matmul: other => {
      if (!tensorInstance) return null;
      try {
        return tensorInstance.matmul(other);
      } catch (err) {
        error.set(err);
        return null;
      }
    },
  };

  const cleanup = () => {
    if (tensorInstance && tensorInstance.free) {
      tensorInstance.free();
    }
    tensorInstance = null;
    tensorState.set({
      tensor: null,
      shape: null,
      size: null,
      dtype: null,
    });
  };

  // Initialize tensor
  if (initialData && initialShape) {
    createTensor(initialData, initialShape);
  }

  return {
    tensorState: { subscribe: tensorState.subscribe },
    error: { subscribe: error.subscribe },
    operations,
    createTensor,
    cleanup,
  };
}

/**
 * Performance Monitoring Store
 * Creates a store for performance metrics
 * @param {boolean} enabled - Whether monitoring is enabled
 * @returns {Object} Svelte stores for performance state
 */
export function createPerformanceStore(enabled = true) {
  const metrics = writable(null);
  const sessionId = writable(null);

  let intervalId = null;

  const startSession = (name, metadata = {}) => {
    if (!enabled) return null;

    const id = performance.startSession(name, {
      framework: 'svelte',
      timestamp: new Date().toISOString(),
      ...metadata,
    });
    sessionId.set(id);
    return id;
  };

  const endSession = () => {
    if (!enabled || !get(sessionId)) return null;

    const report = performance.endSession();
    sessionId.set(null);
    return report;
  };

  const profileOperation = async (name, operation, metadata = {}) => {
    if (!enabled) return await operation();

    return await performance.profile(name, operation, {
      framework: 'svelte',
      ...metadata,
    });
  };

  const updateMetrics = () => {
    if (!enabled) return;

    const currentMetrics = {
      memory: performance.getMemoryUsage(),
      report: performance.getReport(),
      timestamp: Date.now(),
    };
    metrics.set(currentMetrics);
  };

  const cleanup = () => {
    performance.cleanup();
    updateMetrics();
  };

  const startMonitoring = () => {
    if (enabled) {
      updateMetrics();
      intervalId = setInterval(updateMetrics, 5000); // Update every 5 seconds
    }
  };

  const stopMonitoring = () => {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  };

  // Auto-start monitoring
  if (enabled) {
    startMonitoring();
  }

  return {
    metrics: { subscribe: metrics.subscribe },
    sessionId: { subscribe: sessionId.subscribe },
    startSession,
    endSession,
    profileOperation,
    updateMetrics,
    cleanup,
    startMonitoring,
    stopMonitoring,
  };
}

/**
 * Memory Monitoring Store
 * Creates a store for memory statistics
 * @returns {Object} Svelte stores for memory state
 */
export function createMemoryStore() {
  const memoryStats = writable(null);

  let intervalId = null;

  const updateMemoryStats = () => {
    const stats = memory.getStats();
    memoryStats.set({
      ...stats,
      timestamp: Date.now(),
    });
  };

  const cleanup = () => {
    if (typeof window !== 'undefined' && window.gc) {
      window.gc();
    }
    performance.cleanup();
    updateMemoryStats();
  };

  const startMonitoring = () => {
    updateMemoryStats();
    intervalId = setInterval(updateMemoryStats, 3000); // Update every 3 seconds
  };

  const stopMonitoring = () => {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  };

  // Auto-start monitoring
  startMonitoring();

  return {
    memoryStats: { subscribe: memoryStats.subscribe },
    updateMemoryStats,
    cleanup,
    startMonitoring,
    stopMonitoring,
  };
}

/**
 * Development Tools Store
 * Creates a store for development utilities
 * @param {boolean} enabled - Whether dev tools are enabled
 * @returns {Object} Svelte stores for dev tools state
 */
export function createDevToolsStore(enabled = false) {
  const debugState = writable(null);
  const isEnabled = writable(enabled);

  const enableDebug = (options = {}) => {
    devTools.debug.enable({
      trackTensors: true,
      trackOperations: true,
      validateOperations: true,
      ...options,
    });
    isEnabled.set(true);
  };

  const disableDebug = () => {
    devTools.debug.disable();
    isEnabled.set(false);
  };

  const startDebugSession = (name, metadata = {}) => {
    if (!get(isEnabled)) return null;

    return devTools.debug.startSession(name, {
      framework: 'svelte',
      timestamp: new Date().toISOString(),
      ...metadata,
    });
  };

  const trackTensor = (tensor, operation, metadata = {}) => {
    if (!get(isEnabled)) return;

    devTools.debug.trackTensor(tensor, operation, {
      framework: 'svelte',
      ...metadata,
    });
  };

  const validateOperation = (operation, tensors, options = {}) => {
    if (!get(isEnabled)) return true;

    return devTools.debug.validateOperation(operation, tensors, options);
  };

  const generateReport = () => {
    if (!get(isEnabled)) return null;

    const report = devTools.debug.generateReport();
    debugState.set(report);
    return report;
  };

  const analyzeComprehensive = (options = {}) => {
    if (!get(isEnabled)) return null;

    return devTools.analyze.comprehensive({
      includeDebug: true,
      ...options,
    });
  };

  return {
    debugState: { subscribe: debugState.subscribe },
    isEnabled: { subscribe: isEnabled.subscribe },
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
 * Svelte Component Helper Functions
 */

/**
 * Create a cleanup function for Svelte components
 * @param {Array} stores - Array of stores to cleanup
 * @returns {Function} Cleanup function
 */
export function createCleanupFunction(stores) {
  return () => {
    stores.forEach(store => {
      if (store.cleanup) {
        store.cleanup();
      }
      if (store.unsubscribe) {
        store.unsubscribe();
      }
    });
  };
}

/**
 * Create a Svelte action for tensor cleanup
 * @param {HTMLElement} node - DOM element
 * @param {Object} tensor - Tensor to manage
 * @returns {Object} Action object
 */
export function tensorCleanupAction(node, tensor) {
  const cleanup = () => {
    if (tensor && tensor.free) {
      tensor.free();
    }
  };

  return {
    destroy: cleanup,
  };
}

/**
 * Create a Svelte action for performance monitoring
 * @param {HTMLElement} node - DOM element
 * @param {Object} options - Monitoring options
 * @returns {Object} Action object
 */
export function performanceMonitorAction(node, options = {}) {
  const { enabled = true, updateInterval = 5000 } = options;
  const performanceStore = createPerformanceStore(enabled);

  let unsubscribe = null;

  if (enabled) {
    unsubscribe = performanceStore.metrics.subscribe(metrics => {
      if (metrics) {
        // Update node with performance data
        node.setAttribute(
          'data-performance',
          JSON.stringify({
            memory: metrics.memory ? `${(metrics.memory.used / 1024 / 1024).toFixed(1)}MB` : 'N/A',
            timestamp: new Date(metrics.timestamp).toLocaleTimeString(),
          })
        );
      }
    });
  }

  return {
    destroy() {
      if (unsubscribe) unsubscribe();
      performanceStore.stopMonitoring();
    },
  };
}

/**
 * Utility function to create a complete TrustformeRS setup for a Svelte app
 * @param {Object} options - Setup options
 * @returns {Object} Complete setup with all stores
 */
export function createTrustformersSetup(options = {}) {
  const {
    initOptions = {},
    modelConfig = null,
    tokenizerType = null,
    tokenizerVocab = null,
    pipelineTask = null,
    pipelineConfig = {},
    enablePerformanceMonitoring = true,
    enableMemoryMonitoring = true,
    enableDevTools = false,
  } = options;

  // Core stores
  const trustformersStore = createTrustformersStore(initOptions);
  const modelStore = modelConfig ? createModelStore(modelConfig) : null;
  const tokenizerStore = tokenizerType ? createTokenizerStore(tokenizerType, tokenizerVocab) : null;
  const pipelineStore =
    pipelineTask && modelStore && tokenizerStore
      ? createPipelineStore(pipelineTask, modelStore, tokenizerStore, pipelineConfig)
      : null;

  // Monitoring stores
  const performanceStore = enablePerformanceMonitoring ? createPerformanceStore(true) : null;
  const memoryStore = enableMemoryMonitoring ? createMemoryStore() : null;
  const devToolsStore = enableDevTools ? createDevToolsStore(enableDevTools) : null;

  const cleanup = () => {
    if (modelStore) modelStore.cleanup();
    if (tokenizerStore) tokenizerStore.cleanup();
    if (pipelineStore) {
      pipelineStore.cleanup();
      pipelineStore.unsubscribe();
    }
    if (performanceStore) performanceStore.stopMonitoring();
    if (memoryStore) memoryStore.stopMonitoring();
  };

  return {
    trustformers: trustformersStore,
    model: modelStore,
    tokenizer: tokenizerStore,
    pipeline: pipelineStore,
    performance: performanceStore,
    memory: memoryStore,
    devTools: devToolsStore,
    cleanup,
  };
}

export default {
  // Store factories
  createTrustformersStore,
  createModelStore,
  createTokenizerStore,
  createPipelineStore,
  createTensorStore,
  createPerformanceStore,
  createMemoryStore,
  createDevToolsStore,

  // Utility functions
  createCleanupFunction,
  createTrustformersSetup,

  // Actions
  tensorCleanupAction,
  performanceMonitorAction,
};
